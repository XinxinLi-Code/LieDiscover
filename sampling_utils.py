"""Shared deterministic sampling utilities for discovery and RS-AC.

All symmetry-discovery paths consume a window of training samples. This module
makes that window explicit:

1. Build a seed-dependent permutation of the whole dataset.
2. Iterate that permutation with ``shuffle=False``.
3. Select ``sample * sample_size : (sample + 1) * sample_size`` downstream.

This keeps the existing ``seed/sample/sample_size`` behavior while ensuring
that discovery, reward, pruning, and evaluation helpers see the same data order.
"""

from __future__ import annotations

import hashlib
import json
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset


DEFAULT_SAMPLE_SIZE = 100


def seeded_permutation_indices(dataset_size: int, seed: int) -> Tuple[int, ...]:
    """Return a deterministic full-dataset permutation."""

    dataset_size = int(dataset_size)
    if dataset_size < 0:
        raise ValueError("dataset_size must be non-negative.")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return tuple(int(idx) for idx in torch.randperm(dataset_size, generator=generator).tolist())


def sample_window_indices(
    dataset_size: int,
    *,
    seed: int,
    sample: int = 0,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> Tuple[int, ...]:
    """Return the explicit dataset indices for one seeded sample window."""

    sample = int(sample)
    sample_size = int(sample_size)
    if sample < 0:
        raise ValueError("sample must be non-negative.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    permutation = seeded_permutation_indices(dataset_size, seed)
    start = sample * sample_size
    stop = (sample + 1) * sample_size
    return permutation[start:stop]


def build_seeded_permutation_loader(
    dataset,
    *,
    seed: int,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader over a deterministic full-dataset permutation."""

    permutation = seeded_permutation_indices(len(dataset), seed)
    subset = Subset(dataset, list(permutation))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    loader.sample_seed = int(seed)
    loader.sample_permutation = permutation
    return loader


def describe_sample_window(
    dataset_size: int,
    *,
    seed: int,
    sample: int = 0,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    max_items: int = 12,
) -> str:
    """Compact human-readable description for logs."""

    indices = sample_window_indices(
        dataset_size,
        seed=seed,
        sample=sample,
        sample_size=sample_size,
    )
    shown = ",".join(str(idx) for idx in indices[: int(max_items)])
    suffix = "" if len(indices) <= int(max_items) else ",..."
    return (
        f"seed={int(seed)}, sample={int(sample)}, sample_size={int(sample_size)}, "
        f"count={len(indices)}, indices=[{shown}{suffix}]"
    )


def dataset_identity(dataset):
    """Fingerprint the loaded values and their row order, independently of paths.

    Cache on the dataset, invalidating when a tensor is replaced or mutated.
    Hashing is done once per dataset, not once per rollout or candidate library.
    """
    if isinstance(getattr(dataset, "data", None), dict):
        fields = dataset.data
    elif hasattr(dataset, "X") and hasattr(dataset, "y"):
        fields = {"x": dataset.X, "y": dataset.y}
    else:
        raise TypeError("Dataset provenance requires named tensor fields.")
    if not fields or any(not torch.is_tensor(value) for value in fields.values()):
        raise TypeError("Dataset provenance requires nonempty tensor fields.")
    task = getattr(dataset, "pde_name", "top" if hasattr(dataset, "X") else None)
    split = getattr(dataset, "mode", "train")
    versions = tuple(
        (name, id(value), tuple(value.shape), str(value.dtype), value._version)
        for name, value in sorted(fields.items())
    )
    cache_key = (task, split, len(dataset), versions)
    cached = getattr(dataset, "_sample_identity_cache", None)
    if cached is not None and cached[0] == cache_key:
        return dict(cached[1])
    description = {
        "version": 1, "task": task, "split": split, "size": len(dataset),
        "layout": "dataset_row_order_v1",
        "fields": [
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(fields.items())
        ],
    }
    digest = hashlib.sha256(json.dumps(description, sort_keys=True).encode("utf-8"))
    for _, value in sorted(fields.items()):
        raw = value.detach().cpu().contiguous().view(torch.uint8).reshape(-1).numpy()
        view = memoryview(raw)
        for start in range(0, len(view), 1024 * 1024):
            digest.update(view[start:start + 1024 * 1024])
    description["sha256"] = digest.hexdigest()
    dataset._sample_identity_cache = (cache_key, description)
    return dict(description)


def validate_indices(indices, dataset_size):
    values = tuple(indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in values):
        raise ValueError("Sample indices must be integers.")
    if len(set(values)) != len(values):
        raise ValueError("Sample indices must be unique.")
    if any(index < 0 or index >= dataset_size for index in values):
        raise ValueError("Sample index is outside the original dataset.")
    return values


def capture_search_samples(loader, *, sample, sample_size, sample_count=None):
    """Capture exact rows for the shared batch-size-one permutation loader.

    Unknown/custom loader orders are left unverified, rather than inferred from
    a global RNG seed. The independent validator will require explicit indices.
    """
    permutation = getattr(loader, "sample_permutation", None)
    subset = getattr(loader, "dataset", None)
    dataset = getattr(subset, "dataset", None)
    if permutation is None or dataset is None or getattr(loader, "batch_size", None) != 1:
        return None
    if tuple(subset.indices) != tuple(permutation) or len(permutation) != len(dataset):
        raise ValueError("The loader no longer matches its recorded sample permutation.")
    start = int(sample) * int(sample_size)
    indices = tuple(int(index) for index in permutation[start:start + int(sample_size)])
    if sample_count is not None and len(indices) != int(sample_count):
        raise ValueError("Recorded search indices do not match the processed sample count.")
    try:
        identity = dataset_identity(dataset)
    except TypeError:
        return None
    return {
        "version": 1, "dataset": identity, "indices": list(indices),
        "sampler": "seeded_permutation", "seed": int(loader.sample_seed),
        "sample": int(sample), "sample_size": int(sample_size),
        "torch_version": str(torch.__version__), "lineage_complete": True,
    }


def merge_search_samples(current, previous):
    """Exclude original search rows as well as later extraction/pruning rows."""
    if current is None:
        return None
    merged = dict(current)
    if previous is None:
        merged["lineage_complete"] = False
        merged["missing_source_provenance"] = True
        return merged
    if previous.get("version") != 1 or previous.get("dataset") != current["dataset"]:
        raise ValueError("Source and current generators do not have matching dataset provenance.")
    old = validate_indices(previous["indices"], current["dataset"]["size"])
    new = validate_indices(current["indices"], current["dataset"]["size"])
    merged["indices"] = sorted(set(old).union(new))
    merged["lineage_complete"] = bool(
        current.get("lineage_complete", False) and previous.get("lineage_complete", False)
    )
    merged["source_windows"] = [
        window
        for source in (previous, current)
        for window in (source.get("source_windows") or [{
            key: source[key] for key in (
                "sampler", "seed", "sample", "sample_size", "groups", "scale_reference",
                "fitting_groups", "normalization_reference",
            ) if key in source
        }])
    ]
    # Validation trajectories also influence rank selection, but their row
    # numbers refer to a different dataset and must never enter train indices.
    validation_history = []
    for source in (previous, current):
        records = list(source.get("selection_validation_history", []))
        if source.get("selection_validation") is not None:
            records.append(source["selection_validation"])
        for record in records:
            if record not in validation_history:
                validation_history.append(record)
    if validation_history:
        merged["selection_validation_history"] = validation_history
    return merged
