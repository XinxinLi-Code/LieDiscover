"""Generate float64 PDE jets and full-resolution fields for weak discovery.

Each split uses independently generated initial conditions.  Full fields are
saved one trajectory at a time to avoid holding a 2D split in memory; existing
training and discovery loaders continue to read the standard jet files under
data/<task>/<split>-<field>.pt.  Full fields are stored under data/raw/<task>.
Numerical time derivatives use second-order centered differences.  Existing
raw files can be reused with --rebuild-jets-from-raw without any simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


TASKS = (
    "heat", "burger", "kdv", "wave", "rd", "schrodinger",
    "damped_burger", "forced_transport",
)
SPLITS = ("train", "val", "test")
GRID_VERSION = "consistent_periodic_grid_v2"
NUMERICAL_TIME_TASKS = tuple(task for task in TASKS if task != "forced_transport")
ONE_DIMENSIONAL_KEYS = (
    "t", "x", "u", "dudx", "dudxdx", "dudt", "dudtdx",
)
COUPLED_KEYS = (
    "t", "x", "y", "u", "dudt", "dudx", "dudy", "dudxdx",
    "dudydy", "dudtdx", "dudtdy", "dudxdy", "v", "dvdt",
    "dvdx", "dvdy", "dvdxdx", "dvdydy", "dvdtdx", "dvdtdy", "dvdxdy",
)


def jet_keys(task: str) -> tuple[str, ...]:
    if task == "kdv":
        return (
            "t", "x", "u", "dudx", "dudxdx", "dudxdxdx", "dudt",
            "dudtdx", "dudtdxdx",
        )
    if task == "wave":
        return (
            "t", "x", "y", "u", "dudt", "dudx", "dudy", "dudtdt",
            "dudxdx", "dudydy", "dudtdx", "dudtdy", "dudxdy",
        )
    if task in ("rd", "schrodinger"):
        return COUPLED_KEYS
    if task in TASKS:
        return ONE_DIMENSIONAL_KEYS
    raise ValueError(f"Unsupported task: {task}")


def trajectory_seed(seed: int, task: str, split: str, ic_index: int) -> int:
    """A stable seed independent of task order and requested split sizes."""
    state = np.random.SeedSequence(
        [seed, TASKS.index(task), SPLITS.index(split), ic_index]
    ).generate_state(1)
    return int(state[0])


def _tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype)


def _check_finite(arrays: dict[str, np.ndarray], task: str, split: str, index: int) -> None:
    for key, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(
                f"{task}/{split} IC {index}: non-finite {key}; reduce dt or "
                "initial-condition amplitude before generating the dataset."
            )


def _time_metadata(task: str) -> dict[str, Any]:
    numerical = task in NUMERICAL_TIME_TASKS
    return {
        "saved_jet_time_crop": [1, 1],
        "time_derivative_order": 2 if numerical else None,
        "time_derivative_source": (
            "second-order central finite difference of simulated trajectory"
            if numerical else "analytic solution"
        ),
        "jet_dtype": "float64",
    }


def generate_split(
    task: str,
    split: str,
    output_dir: str | Path,
    *,
    n_ics: int,
    n: int = 100,
    steps: int = 1000,
    dt: float = 0.002,
    seed: int = 0,
    L: float | None = None,
    n_terms: int | None = None,
    alpha: float = 1.0,
    c: float = 1.0,
    amplitude: float = 0.5,
    overwrite: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    if task not in TASKS or split not in SPLITS:
        raise ValueError(f"Unsupported task/split: {task}/{split}")
    minimum_steps = 5 if task in NUMERICAL_TIME_TASKS else 3
    if n_ics < 1 or n < 6 or steps < minimum_steps or dt <= 0 or seed < 0:
        raise ValueError(f"Require n_ics >= 1, n >= 6, steps >= {minimum_steps}, dt > 0, seed >= 0.")
    L = (2 * np.pi if task == "forced_transport" else 20.0) if L is None else L
    n_terms = (5 if task == "kdv" else 2 if task in ("wave", "rd", "schrodinger") else 10) if n_terms is None else n_terms
    if L <= 0 or n_terms < 1:
        raise ValueError("Require L > 0 and n_terms >= 1.")
    if 2 * n_terms >= n:
        raise ValueError("The spatial grid must resolve the initial Fourier modes: n > 2 * n_terms.")
    module = importlib.import_module("data_utils." + task)
    source_path = Path(module.__file__).resolve()
    generator_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    data_dir = Path(output_dir).resolve()
    task_dir = data_dir / task
    raw_dir = data_dir / "raw" / task
    keys = jet_keys(task)
    expected_paths = [task_dir / f"{split}-{key}.pt" for key in keys]
    expected_paths.append(task_dir / f"{split}-metadata.json")
    existing_raw = list(raw_dir.glob(f"{split}-*.pt")) if raw_dir.exists() else []
    if not overwrite and (any(path.exists() for path in expected_paths) or existing_raw):
        raise FileExistsError(
            f"{task}/{split} already exists in {data_dir}; use a new --output-dir "
            "or explicitly pass --overwrite."
        )
    if overwrite:
        for path in existing_raw:
            path.unlink()
    task_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    field_parts: dict[str, list[torch.Tensor]] = {key: [] for key in keys if key not in ("t", "x", "y")}
    coordinates: dict[str, torch.Tensor] = {}
    raw_entries = []
    residual_ratios = []
    source_dx = source_dy = None
    for ic_index in range(n_ics):
        ic_seed = trajectory_seed(seed, task, split, ic_index)
        np.random.seed(ic_seed)
        torch.manual_seed(ic_seed)
        parameters = dict(n_ics=1, L=L, n=n, n_terms=n_terms, dt=dt, num_steps=steps)
        if task == "damped_burger":
            parameters.update(alpha=alpha, amplitude=amplitude)
        elif task == "forced_transport":
            parameters.update(c=c, amplitude=amplitude)
        raw = getattr(module, "get_" + task + "_data")(**parameters)
        if task in ("wave", "rd", "schrodinger"):
            raw_keys = ("t", "x", "y", "u") if task == "wave" else ("t", "x", "y", "u", "v")
        else:
            raw_keys = ONE_DIMENSIONAL_KEYS if task == "forced_transport" else ("t", "x", "u")
        raw_arrays = dict(zip(raw_keys, raw))
        _check_finite(raw_arrays, task, split, ic_index)
        if task in ("wave", "rd", "schrodinger"):
            source_dx = float(raw_arrays["x"][0, 1] - raw_arrays["x"][0, 0])
            source_dy = float(raw_arrays["y"][1, 0] - raw_arrays["y"][0, 0])
        else:
            source_dx = float(raw_arrays["x"][1] - raw_arrays["x"][0])
        raw_path = raw_dir / f"{split}-{ic_index:04d}.pt"
        # Float64 retains temporal resolution for weak second-time derivatives.
        bundle = {key: _tensor(array, torch.float64) for key, array in raw_arrays.items()}
        bundle["metadata"] = {
            "task": task, "split": split, "ic_index": ic_index, "seed": ic_seed,
            "grid_version": GRID_VERSION, "generator_sha256": generator_hash,
            "dx": source_dx, "dy": source_dy, "dt": dt,
        }
        torch.save(bundle, raw_path)
        raw_entries.append({
            "path": str(raw_path.relative_to(data_dir)), "ic_index": ic_index,
            "seed": ic_seed, "u_shape": list(raw_arrays["u"].shape),
        })
        del bundle

        derivative_parameters = {"alpha": alpha} if task == "damped_burger" else {"c": c} if task == "forced_transport" else {}
        derivatives = module.compute_derivative(*raw, **derivative_parameters)
        if task in ("damped_burger", "forced_transport"):
            residual_ratios.extend(np.asarray(derivatives[-1]).tolist())
            derivatives = derivatives[:-1]
        jet_arrays = dict(zip(keys, derivatives))
        _check_finite(jet_arrays, task, split, ic_index)
        for key, array in jet_arrays.items():
            value = _tensor(array, torch.float64)
            if not torch.isfinite(value).all():
                raise ValueError(f"{task}/{split} IC {ic_index}: {key} exceeds the float64 range.")
            if key in ("t", "x", "y"):
                if key in coordinates and not torch.equal(coordinates[key], value):
                    raise ValueError("Trajectories in one split must have identical coordinates.")
                coordinates[key] = value
            else:
                field_parts[key].append(value)
        if not quiet:
            print(f"{task}/{split}: saved full-resolution trajectory {ic_index + 1}/{n_ics}", flush=True)
        del raw, raw_arrays, derivatives, jet_arrays

    shapes = {}
    for key in keys:
        value = coordinates[key] if key in coordinates else torch.cat(field_parts[key], dim=0)
        shapes[key] = list(value.shape)
        torch.save(value, task_dir / f"{split}-{key}.pt")
        if key in field_parts:
            del field_parts[key]

    metadata: dict[str, Any] = {
        "schema_version": 1, "task": task, "split": split,
        "grid_version": GRID_VERSION, "generator_sha256": generator_hash,
        "seed": seed, "n_ics": n_ics, "n": n, "steps": steps,
        "L": L, "dt": dt, "dx": source_dx, "dy": source_dy,
        "n_terms": n_terms, "alpha": alpha if task == "damped_burger" else None,
        "c": c if task == "forced_transport" else None,
        "amplitude": amplitude if task in ("damped_burger", "forced_transport") else None,
        "initial_condition_distribution": "uniform Fourier coefficients" if task in ("damped_burger", "forced_transport") else "Gaussian Fourier coefficients",
        "periodic_endpoint_excluded": task not in ("damped_burger", "forced_transport"),
        "saved_jet_spatial_stride": 10 if task in ("wave", "rd", "schrodinger") else 1,
        "saved_jet_time_stride": 1,
        "saved_jet_space_crop": [2, 2] if task == "kdv" else [1, 1],
        "raw_dtype": "float64", "jet_shapes": shapes,
        "jet_builder_sha256": generator_hash,
        "raw_trajectories": raw_entries,
        "independent_initial_conditions_between_splits": True,
        **_time_metadata(task),
    }
    if residual_ratios:
        metadata["finite_difference_residual_ratio_mean"] = float(np.mean(residual_ratios))
        metadata["finite_difference_residual_ratio_max"] = float(np.max(residual_ratios))
    metadata_path = task_dir / f"{split}-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def rebuild_split_from_raw(
    task: str,
    split: str,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Re-export jets from matching raw files without simulating or writing raw.

    The original generator hash continues to identify the raw trajectories;
    ``jet_builder_sha256`` separately identifies the current derivative code.
    All trajectories are checked before any existing jet file is replaced.
    """
    if task not in TASKS or split not in SPLITS:
        raise ValueError(f"Unsupported task/split: {task}/{split}")
    data_dir = Path(output_dir).resolve()
    task_dir = data_dir / task
    metadata_path = task_dir / f"{split}-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("task") != task or metadata.get("split") != split:
        raise ValueError(f"Task/split mismatch in {metadata_path}")
    if metadata.get("grid_version") != GRID_VERSION or metadata.get("raw_dtype") != "float64":
        raise ValueError("Raw rebuild requires the matching grid-version float64 dataset.")
    keys = jet_keys(task)
    if not overwrite and any((task_dir / f"{split}-{key}.pt").exists() for key in keys):
        raise FileExistsError("Rebuilding existing jets requires --overwrite; raw files are never overwritten.")
    entries = metadata.get("raw_trajectories", [])
    if sorted(entry["ic_index"] for entry in entries) != list(range(metadata["n_ics"])):
        raise ValueError("Raw metadata must contain exactly one entry per initial condition.")
    module = importlib.import_module("data_utils." + task)
    jet_builder_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    raw_keys = (("t", "x", "y", "u") if task == "wave" else
                ("t", "x", "y", "u", "v") if task in ("rd", "schrodinger") else
                ONE_DIMENSIONAL_KEYS if task == "forced_transport" else ("t", "x", "u"))
    parameters = ({"alpha": metadata["alpha"]} if task == "damped_burger" else
                  {"c": metadata["c"]} if task == "forced_transport" else {})
    parts = {key: [] for key in keys if key not in ("t", "x", "y")}
    coordinates: dict[str, torch.Tensor] = {}
    residual_ratios = []
    for entry in sorted(entries, key=lambda item: item["ic_index"]):
        index = int(entry["ic_index"])
        specified = Path(entry["path"])
        raw_path = specified if specified.is_absolute() else data_dir / specified
        if not raw_path.is_file():
            raw_path = data_dir / "raw" / task / f"{split}-{index:04d}.pt"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing raw trajectory: {raw_path}; rebuild never invokes a simulator.")
        bundle = torch.load(raw_path, map_location="cpu", weights_only=True, mmap=True)
        raw_metadata = bundle.get("metadata", {})
        expected_identity = {
            "task": task, "split": split, "ic_index": index, "seed": entry["seed"],
            "grid_version": metadata["grid_version"],
            "generator_sha256": metadata["generator_sha256"],
        }
        if any(raw_metadata.get(key) != value for key, value in expected_identity.items()):
            raise ValueError(f"Raw trajectory identity disagrees with metadata: {raw_path}")
        arrays = {}
        for key in raw_keys:
            value = bundle.get(key)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
                raise ValueError(f"Expected float64 raw {key}: {raw_path}")
            arrays[key] = value.numpy()
        _check_finite(arrays, task, split, index)
        if list(arrays["u"].shape) != entry["u_shape"] or arrays["u"].shape[0] != 1:
            raise ValueError(f"Raw trajectory shape disagrees with metadata: {raw_path}")
        expected_times = np.arange(metadata["steps"], dtype=np.float64) * metadata["dt"]
        if arrays["t"].shape != expected_times.shape or not np.allclose(arrays["t"], expected_times, rtol=1e-12, atol=1e-14):
            raise ValueError(f"Raw time coordinates disagree with metadata: {raw_path}")
        axis = np.linspace(-metadata["L"] / 2, metadata["L"] / 2, metadata["n"],
                           endpoint=not metadata["periodic_endpoint_excluded"])
        expected_x, expected_y = np.meshgrid(axis, axis) if "y" in arrays else (axis, None)
        for key, expected in (("x", expected_x), ("y", expected_y)):
            if expected is not None and (arrays[key].shape != expected.shape or
                    not np.allclose(arrays[key], expected, rtol=1e-12, atol=1e-14)):
                raise ValueError(f"Raw {key} coordinates disagree with metadata: {raw_path}")
        for key in ("dt", "dx", "dy"):
            expected = metadata.get(key)
            value = raw_metadata.get(key)
            agrees = value is None if expected is None else value is not None and np.isclose(value, expected, rtol=1e-12, atol=1e-14)
            if not agrees:
                raise ValueError(f"Raw {key} spacing disagrees with metadata: {raw_path}")
        expected_field_shape = (1, metadata["steps"], *expected_x.shape)
        for key in raw_keys:
            if key not in ("t", "x", "y") and arrays[key].shape != expected_field_shape:
                raise ValueError(f"Raw {key} shape disagrees with its coordinate grids: {raw_path}")
        derivatives = module.compute_derivative(*(arrays[key] for key in raw_keys), **parameters)
        if task in ("damped_burger", "forced_transport"):
            residual_ratios.extend(np.asarray(derivatives[-1]).tolist())
            derivatives = derivatives[:-1]
        jet_arrays = dict(zip(keys, derivatives))
        _check_finite(jet_arrays, task, split, index)
        for key, array in jet_arrays.items():
            value = _tensor(array, torch.float64)
            if key in ("t", "x", "y"):
                if key in coordinates and not torch.equal(coordinates[key], value):
                    raise ValueError("Raw trajectories must share identical coordinates.")
                coordinates[key] = value
            else:
                parts[key].append(value)
        if not quiet:
            print(f"{task}/{split}: rebuilt jets from raw trajectory {index + 1}/{len(entries)}", flush=True)
        del bundle, arrays, derivatives, jet_arrays
    shapes = {}
    for key in keys:
        value = coordinates[key] if key in coordinates else torch.cat(parts[key], dim=0)
        shapes[key] = list(value.shape)
        torch.save(value, task_dir / f"{split}-{key}.pt")
        if key in parts:
            del parts[key]
    metadata.update(_time_metadata(task))
    metadata.update(jet_shapes=shapes, jet_builder_sha256=jet_builder_hash,
                    jet_rebuilt_from_raw=True)
    if residual_ratios:
        metadata["finite_difference_residual_ratio_mean"] = float(np.mean(residual_ratios))
        metadata["finite_difference_residual_ratio_max"] = float(np.max(residual_ratios))
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", choices=TASKS)
    selection.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument("--output-dir", default="./data", help="Original data root; saves <task>/<split>-<field>.pt.")
    parser.add_argument("--n-ics", type=int, default=10, help="Independent training initial conditions.")
    parser.add_argument("--n-val-ics", type=int, default=2)
    parser.add_argument("--n-test-ics", type=int, default=2)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--steps", "--num-steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--L", type=float, default=None, help="Default 20, except forced_transport uses 2*pi.")
    parser.add_argument("--n-terms", type=int, default=None, help="Default KdV 5; 2D 2; other tasks 10.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace selected splits in output-dir.")
    parser.add_argument("--rebuild-jets-from-raw", action="store_true",
                        help="Read existing metadata/raw files and rebuild jets only; never simulate or overwrite raw.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    tasks = (args.task,) if args.task else tuple(args.tasks or TASKS)
    counts = {"train": args.n_ics, "val": args.n_val_ics, "test": args.n_test_ics}
    for task in tasks:
        for split, count in counts.items():
            if args.rebuild_jets_from_raw:
                metadata = rebuild_split_from_raw(task, split, args.output_dir,
                                                  overwrite=args.overwrite, quiet=args.quiet)
                print(f"Rebuilt {task}/{split} jets from existing raw: {metadata['jet_shapes']['u']}", flush=True)
                continue
            metadata = generate_split(
                task, split, args.output_dir, n_ics=count, n=args.n,
                steps=args.steps, dt=args.dt, seed=args.seed, L=args.L,
                n_terms=args.n_terms, alpha=args.alpha, c=args.c,
                amplitude=args.amplitude, overwrite=args.overwrite, quiet=args.quiet,
            )
            print(
                f"Generated {task}/{split}: {count} ICs, dx={metadata['dx']:.9g}, "
                f"dt={args.dt:.9g}, jets={metadata['jet_shapes']['u']}", flush=True,
            )


if __name__ == "__main__":
    main()
