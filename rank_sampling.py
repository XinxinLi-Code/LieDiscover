"""Fixed sampling and provenance for validation-based nullity selection.

Only saved train/val jets and the already loaded Jacobian model are read. The
test split is never opened. All three fits share the same training scale pool.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from sampling_utils import dataset_identity


DEFAULT_RANK_REPLICATES = 3
DEFAULT_RANK_SCALE_POINTS = 1000
DEFAULT_RANK_VALIDATION_POINTS = 1000
SAMPLING_VERSION = "balanced_trajectory_rank_v1"


def balanced_indices(shape, count, seed):
    """Match the pilot's trajectory-balanced sampling without replacement."""
    shape = tuple(int(size) for size in shape)
    if len(shape) < 2 or min(shape) <= 0 or int(count) <= 0:
        raise ValueError("Sampling requires a nonempty trajectory grid and positive count.")
    per_trajectory = math.prod(shape[1:])
    rounds = math.ceil(int(count) / shape[0])
    if rounds > per_trajectory:
        raise ValueError(f"Requested {count} points from a dataset of size {math.prod(shape)}.")
    rng = np.random.default_rng(int(seed))
    rows = np.stack([
        rng.choice(per_trajectory, rounds, replace=False) + index * per_trajectory
        for index in range(shape[0])
    ], axis=1).reshape(-1)[:int(count)]
    return tuple(int(index) for index in rows)


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_shape(folder, split):
    return tuple(torch.load(folder / f"{split}-u.pt", map_location="cpu",
                            weights_only=True, mmap=True).shape)


def _trajectory_seeds(folder, split):
    path = folder / f"{split}-metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    trajectories = metadata.get("raw_trajectories", [])
    return [item["seed"] for item in trajectories] if trajectories else None


def _selected_split_fields(folder, split, keys, shape, indices):
    """Read only requested val rows; avoid materializing its entire float64 grid."""
    rows = torch.as_tensor(indices, dtype=torch.long)
    spatial_count = math.prod(shape[2:])
    time_indices = rows // spatial_count % shape[1]
    space_indices = rows % spatial_count
    selected, files = {}, []
    for key in keys:
        path = folder / f"{split}-{key}.pt"
        field = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if key == "t":
            if field.numel() != shape[1]:
                raise ValueError(f"Time coordinate layout does not match {path}.")
            values = field.reshape(-1)[time_indices]
        elif key in ("x", "y"):
            if field.numel() != spatial_count:
                raise ValueError(f"Spatial coordinate layout does not match {path}.")
            values = field.reshape(-1)[space_indices]
        else:
            if tuple(field.shape) != shape:
                raise ValueError(f"Jet layout does not match {path}.")
            values = field.reshape(-1)[rows]
        selected[key] = values.to(dtype=torch.float64)
        files.append({"name": path.name, "sha256": _file_hash(path),
                      "shape": list(field.shape), "dtype": str(field.dtype)})
    identity = {"version": 1, "task": folder.name, "split": split,
                "size": math.prod(shape), "layout": "IC,time,spatial", "files": files}
    identity["sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return selected, identity


def _sample_contexts(fields, model, *, task, input_dim, output_dim, device):
    """Compute pointwise model Jacobians once, in bounded float64 batches."""
    from reward import SampleContext, _feature_from_data
    from feature_normalization import prepare_search_model

    model = prepare_search_model(model, device)
    model.eval()
    count = len(next(iter(fields.values())))
    contexts = []
    for start in range(0, count, 256):
        batch = {key: value[start:start + 256].to(device=device, dtype=torch.float64)
                 for key, value in fields.items()}
        if not all(bool(torch.isfinite(value).all()) for value in batch.values()):
            raise ValueError("Rank-selection samples contain nonfinite saved jets.")
        features = _feature_from_data(batch, task)
        jacobians = torch.func.vmap(torch.func.jacrev(model))(features).detach()
        if tuple(jacobians.shape[1:]) != (output_dim, input_dim):
            raise ValueError("Frozen model Jacobian shape differs from the PDE interface.")
        negative_identity = -torch.eye(output_dim, device=device, dtype=torch.float64)
        full = torch.cat((jacobians, negative_identity.expand(len(features), -1, -1)), dim=2)
        if not bool(torch.isfinite(full).all()):
            raise ValueError("Frozen model produced nonfinite rank-selection Jacobians.")
        for offset in range(len(features)):
            data = {key: value[offset:offset + 1].detach() for key, value in batch.items()}
            contexts.append(SampleContext(data=data, df=full[offset]))
    return contexts


def build_rank_reference(args, model, train_loader, device):
    """Build all fixed sample roles, retaining exact train exclusions for holdout."""
    from reward import RankReferenceContext

    subset = getattr(train_loader, "dataset", None)
    dataset = getattr(subset, "dataset", subset)
    if getattr(dataset, "mode", None) != "train" or not isinstance(getattr(dataset, "data", None), dict):
        raise ValueError("Validated rank selection requires a named PDE training dataset.")
    folder = Path(getattr(args, "data_dir", "./data")) / args.task
    shape = _load_shape(folder, "train")
    validation_shape = _load_shape(folder, "val")
    if math.prod(shape) != len(dataset):
        raise ValueError("Loaded training dataset differs from the saved sampling grid.")
    base_seed = int(args.seed if getattr(args, "fix_seed", None) is None else args.fix_seed)
    if base_seed < 0 or int(args.sample) < 0:
        raise ValueError("Validated rank selection requires nonnegative sampling seeds and windows.")
    replicates = int(getattr(args, "rank_replicates", DEFAULT_RANK_REPLICATES))
    if replicates != 3:
        raise ValueError("The calibrated production rank rule requires exactly three sampling replicates.")
    count = int(args.sample_size)
    if count <= 0:
        raise ValueError("sample_size must be positive.")
    seeds = tuple(replicates * base_seed + index for index in range(replicates))
    start, stop = int(args.sample) * count, (int(args.sample) + 1) * count
    fitting = [balanced_indices(shape, stop, seed)[start:stop] for seed in seeds]
    scale_count = int(getattr(args, "rank_scale_points", DEFAULT_RANK_SCALE_POINTS))
    validation_count = int(getattr(args, "rank_validation_points", DEFAULT_RANK_VALIDATION_POINTS))
    scale_seed, validation_seed = 424242, 20261015
    scale_rows = balanced_indices(shape, scale_count, scale_seed)
    validation_rows = balanced_indices(validation_shape, validation_count, validation_seed)
    train_seeds, val_seeds = _trajectory_seeds(folder, "train"), _trajectory_seeds(folder, "val")
    verified = train_seeds is not None and val_seeds is not None
    if verified:
        if len(train_seeds) != shape[0] or len(val_seeds) != validation_shape[0]:
            raise ValueError("Trajectory identities do not match the saved jet grids.")
        if set(train_seeds).intersection(val_seeds):
            raise ValueError("Training and rank-validation trajectory identities overlap.")

    # A training row may serve multiple fitting/scale roles, but is cached once.
    train_rows = tuple(dict.fromkeys(index for group in [scale_rows, *fitting] for index in group))
    positions = {index: position for position, index in enumerate(train_rows)}
    row_tensor = torch.as_tensor(train_rows, dtype=torch.long)
    train_fields = {key: field[row_tensor].to(dtype=torch.float64) for key, field in dataset.data.items()}
    val_fields, val_identity = _selected_split_fields(
        folder, "val", tuple(dataset.data), validation_shape, validation_rows)
    fields = {key: torch.cat((train_fields[key], val_fields[key])) for key in train_fields}
    samples = _sample_contexts(fields, model, task=args.task, input_dim=args.input_dim,
                               output_dim=args.output_dim, device=device)
    search_samples = {
        "version": 1, "dataset": dataset_identity(dataset), "indices": sorted(train_rows),
        "sampler": SAMPLING_VERSION, "seed": base_seed, "sample": int(args.sample),
        "sample_size": count, "lineage_complete": True, "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "scope": "union_of_all_fitting_and_normalization_reference_training_rows",
    }
    provenance = {
        "search_samples": search_samples, "sampling_version": SAMPLING_VERSION,
        "run_seed": int(args.seed), "sampling_seed": base_seed,
        "roles": {
            "fitting": [{"split": "train", "seed": seed, "indices": list(indices)}
                        for seed, indices in zip(seeds, fitting)],
            "normalization_reference": {"split": "train", "seed": scale_seed,
                                        "indices": list(scale_rows)},
            "selection_validation": {"dataset": val_identity, "split": "val",
                                     "seed": validation_seed, "indices": list(validation_rows),
                                     "role": "used_for_rank_and_library_selection_not_test"},
        },
        "trajectory_disjointness_verified": verified,
        "train_trajectory_seeds": train_seeds, "validation_trajectory_seeds": val_seeds,
        "validation_scope": "Separate split for symmetry selection; may have selected the frozen WSINDy fit.",
        "test_split_used": False,
    }
    # Checkpoints historically save only search_samples. Keep every data role
    # there as well; val rows remain separate from the train exclusion indices.
    search_samples["fitting_groups"] = provenance["roles"]["fitting"]
    search_samples["normalization_reference"] = provenance["roles"]["normalization_reference"]
    search_samples["selection_validation"] = provenance["roles"]["selection_validation"]
    return RankReferenceContext(
        samples=samples,
        subset_indices=tuple(tuple(positions[index] for index in group) for group in fitting),
        scale_indices=tuple(positions[index] for index in scale_rows),
        validation_indices=tuple(range(len(train_rows), len(samples))),
        provenance=provenance,
    )
