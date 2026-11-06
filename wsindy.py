"""Weak-window sparse PDE identification from observed solution fields.

Time derivatives and linear spatial derivatives are transferred to compact
test functions by discrete summation by parts. No known equation, Jacobian,
support, or symmetry dimension is used in fitting. Coarse saved 2D fields use
a clearly labelled hybrid mode with the stored spatial jets; retain raw fine
grids to obtain a fully weak spatial fit. Potential Burgers' nonconservative
``u_x**2`` likewise needs a measured spatial jet.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


TASK_INPUTS = {
    "heat": ["u", "u_x", "u_xx"],
    "burger": ["u", "u_x", "u_xx"],
    "damped_burger": ["u", "u_x", "u_xx"],
    "forced_transport": ["x", "u", "u_x", "u_xx"],
    "kdv": ["u", "u_x", "u_xx", "u_xxx"],
    "wave": ["u", "u_x", "u_y", "u_xx", "u_yy", "u_xy"],
    "rd": ["u", "u_x", "u_y", "u_xx", "u_yy", "u_xy", "v", "v_x", "v_y", "v_xx", "v_yy", "v_xy"],
    "schrodinger": ["u", "u_x", "u_y", "u_xx", "u_yy", "u_xy", "v", "v_x", "v_y", "v_xx", "v_yy", "v_xy"],
}
JET_FILES = {
    "u_x": "dudx", "u_y": "dudy", "u_xx": "dudxdx",
    "u_yy": "dudydy", "u_xy": "dudxdy", "u_xxx": "dudxdxdx",
    "v_x": "dvdx", "v_y": "dvdy", "v_xx": "dvdxdx",
    "v_yy": "dvdydy", "v_xy": "dvdxdy",
}

# Shared by the initial fit and downstream discovery. Keep the actual threshold
# grid in every newly saved specification so a later run cannot silently use a
# different sparse-regression search.
DEFAULT_FIT_OPTIONS = {
    "windows": 1200, "val_windows": 400,
    "time_window": None, "space_window": None, "poly_degree": 3,
    "thresholds": None, "validation_tolerance": 0.10,
    "validation_floor": 1e-5, "bootstrap": 8, "spatial_mode": "auto",
}


def _threshold_candidates(thresholds: list[float] | None) -> list[float]:
    values = [0.0, *np.geomspace(1e-5, 1.0, 61).tolist()] if thresholds is None else list(thresholds)
    if not values or any(isinstance(value, (bool, np.bool_))
                         or not isinstance(value, (int, float, np.integer, np.floating))
                         or not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("thresholds must be finite nonnegative values")
    return [float(value) for value in values]


@dataclass(frozen=True)
class WeakTerm:
    expression: str
    kind: str
    field: str = "u"
    powers: tuple[int, int] = (0, 0)
    derivative: str = ""


def candidate_library(task: str, poly_degree: int = 3) -> list[WeakTerm]:
    """Generic polynomial, derivative, transport, and forcing candidates."""
    if task not in TASK_INPUTS:
        raise ValueError(f"Unsupported task: {task}")
    if not 1 <= poly_degree <= 6:
        raise ValueError("poly_degree must be between 1 and 6")
    terms = [WeakTerm("1", "polynomial")]
    coupled = task in ("rd", "schrodinger")
    for degree in range(1, poly_degree + 1):
        for u_power in range(degree, -1, -1) if coupled else (degree,):
            v_power = degree - u_power
            factors = []
            for field, power in (("u", u_power), ("v", v_power)):
                if power:
                    factors.append(field if power == 1 else f"{field}**{power}")
            terms.append(WeakTerm("*".join(factors), "polynomial", powers=(u_power, v_power)))
    for name in TASK_INPUTS[task]:
        if "_" in name:
            field, derivative = name.split("_")
            terms.append(WeakTerm(name, "linear", field=field, derivative=derivative))
    if "y" not in "".join(TASK_INPUTS[task]):
        terms.extend([
            WeakTerm("u*u_x", "quadratic_transport"),
            WeakTerm("u**2*u_x", "cubic_transport"),
        ])
    if task == "burger":
        terms.append(WeakTerm("u_x**2", "gradient_square"))
    if task == "forced_transport":
        terms.extend([WeakTerm("sin(x)", "sin"), WeakTerm("cos(x)", "cos")])
    return terms


def resolve_fit_options(specification: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the initial library and fit configuration without refitting.

    Older artifacts omitted ``thresholds`` from ``fit_options`` but retained
    each tested threshold in the validation table. That ordered table preserves
    custom grids as well as the default grid. Other omitted options use the
    unchanged initial-fit defaults.
    """
    if not isinstance(specification, Mapping):
        raise ValueError("WSINDy specification must be a mapping")
    recorded = specification.get("fit_options", {})
    if not isinstance(recorded, Mapping):
        raise ValueError("WSINDy fit_options must be a mapping")
    options = {key: recorded.get(key, value) for key, value in DEFAULT_FIT_OPTIONS.items()}
    if options["thresholds"] is None:
        metrics = specification.get("metrics", {})
        candidates = metrics.get("validation_candidates") if isinstance(metrics, Mapping) else None
        if candidates is not None:
            if (not isinstance(candidates, list) or not candidates
                    or any(not isinstance(row, Mapping) or "threshold" not in row for row in candidates)):
                raise ValueError("Invalid saved WSINDy validation threshold candidates")
            options["thresholds"] = [row["threshold"] for row in candidates]
    options["thresholds"] = _threshold_candidates(options["thresholds"])
    for name, minimum in (("windows", 2), ("val_windows", 2), ("poly_degree", 1), ("bootstrap", 0)):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"WSINDy {name} must be an integer >= {minimum}")
    for name in ("time_window", "space_window"):
        value = options[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 7):
            raise ValueError(f"WSINDy {name} must be None or an integer >= 7")
    for name in ("validation_tolerance", "validation_floor"):
        value = options[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value < 0):
            raise ValueError(f"WSINDy {name} must be finite and nonnegative")
    if options["spatial_mode"] not in ("auto", "adjoint", "continuous", "jets"):
        raise ValueError("Invalid saved WSINDy spatial_mode")
    terms = candidate_library(specification.get("task"), options["poly_degree"])
    if specification.get("terms") != [term.expression for term in terms]:
        raise ValueError("Saved WSINDy terms do not match its task and polynomial degree")
    return options


def _load_tensor(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Expected tensor in {path}")
    return value.detach().numpy()


class _RawPacks:
    """Small mmap cache for one-trajectory raw packs; never concatenate grids."""
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.cache: dict[int, dict[str, Any]] = {}

    def load(self, index: int) -> dict[str, Any]:
        if index not in self.cache:
            value = torch.load(self.paths[index], map_location="cpu", weights_only=True, mmap=True)
            if not isinstance(value, dict):
                raise ValueError(f"Raw pack must be a dictionary: {self.paths[index]}")
            value = {key: (item.detach().numpy() if isinstance(item, torch.Tensor) else item)
                     for key, item in value.items()}
            if len(self.cache) >= 4:
                self.cache.pop(next(iter(self.cache)))
            self.cache[index] = value
        return self.cache[index]


class _PackedField:
    def __init__(self, store: _RawPacks, field: str):
        self.store = store
        self.field = field
        first = store.load(0)[field]
        if first.shape[0] != 1:
            raise ValueError("Each raw pack must contain exactly one trajectory")
        self.shape = (len(store.paths), *first.shape[1:])
        self.ndim = len(self.shape)

    def __getitem__(self, selection: tuple[Any, ...]) -> np.ndarray:
        trajectory, *slices = selection
        value = self.store.load(int(trajectory))[self.field]
        if value.shape != (1, *self.shape[1:]):
            raise ValueError("Raw trajectories must use the same grid shape")
        return value[(0, *slices)]


def _spacing(coordinate: np.ndarray, name: str) -> float:
    values = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    if values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid {name} coordinate")
    differences = np.diff(values)
    # Averaging over the full coordinate span avoids the systematic median
    # difference bias of saved float32 time coordinates.
    step = float((values[-1] - values[0]) / (values.size - 1))
    if step <= 0 or not np.allclose(differences, step, rtol=2e-4, atol=1e-7):
        raise ValueError(f"WSINDy requires a uniformly increasing {name} grid")
    return step


def load_field_data(task: str, data_dir: str | Path, split: str) -> dict[str, Any]:
    """Prefer retained raw grids, otherwise use the existing cropped files."""
    root = Path(data_dir)
    directory = root / "raw" / task
    packs = sorted(directory.glob(f"{split}-[0-9][0-9][0-9][0-9].pt"))
    if not packs and not (directory / f"{split}-u.pt").exists():
        directory = root / task
    required = ["t", "x", "u"]
    if task in ("wave", "rd", "schrodinger"):
        required.append("y")
    if task in ("rd", "schrodinger"):
        required.append("v")
    if packs:
        store = _RawPacks(packs)
        first = store.load(0)
        result = {name: (_PackedField(store, name) if name in ("u", "v") else first[name])
                  for name in required}
        result["raw_files"] = [str(path.resolve()) for path in packs]
    else:
        result = {name: _load_tensor(directory / f"{split}-{name}.pt") for name in required}
    fields = result["u"]
    if fields.ndim not in (3, 4):
        raise ValueError("Solution fields must have shape (trajectory,time,x) or (trajectory,time,y,x)")
    if fields.shape[1] != result["t"].size:
        raise ValueError("Time grid and field shape do not match")
    x = result["x"]
    x_axis = x[0, :] if x.ndim == 2 else x
    if x_axis.size != fields.shape[-1]:
        raise ValueError("x grid and field shape do not match")
    spacings = [_spacing(result["t"], "t")]
    if fields.ndim == 4:
        y = result["y"]
        y_axis = y[:, 0] if y.ndim == 2 else y
        if y_axis.size != fields.shape[-2]:
            raise ValueError("y grid and field shape do not match")
        spacings.append(_spacing(y_axis, "y"))
    spacings.append(_spacing(x_axis, "x"))
    if "v" in result and result["v"].shape != fields.shape:
        raise ValueError("Coupled fields must have the same shape")
    result.update(directory=directory, split=split, spacings=spacings, x_axis=x_axis)
    return result


def _central_stencil(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Fourth-order accurate central differences for observed field samples."""
    if order == 1:
        return np.array([-2, -1, 1, 2]), np.array([1., -8., 8., -1.]) / 12
    if order == 2:
        return np.array([-2, -1, 0, 1, 2]), np.array([-1., 16., -30., 16., -1.]) / 12
    if order == 3:
        return np.array([-3, -2, -1, 1, 2, 3]), np.array([1., -8., 13., -13., 8., -1.]) / 8
    raise ValueError(f"Unsupported pointwise derivative order: {order}")


def _derivative_axes(derivative: str, ndim: int) -> dict[int, int]:
    axes = {"t": 0, "x": ndim - 1}
    if ndim == 3:
        axes["y"] = 1
    if not derivative or set(derivative) - axes.keys():
        raise ValueError(f"Unsupported pointwise derivative: {derivative}")
    return {axis: derivative.count(name) for name, axis in axes.items() if name in derivative}


def _sample_derivative(field: Any, trajectory: int, indices: np.ndarray,
                       spacings: list[float], derivative: str) -> np.ndarray:
    orders = _derivative_axes(derivative, indices.shape[1])
    stencils = [(axis, *_central_stencil(order)) for axis, order in orders.items()]
    result = np.zeros(len(indices), dtype=np.float64)
    for positions in product(*(range(len(offsets)) for _, offsets, _ in stencils)):
        selection = [indices[:, axis] for axis in range(indices.shape[1])]
        coefficient = 1.0
        for position, (axis, offsets, weights) in zip(positions, stencils):
            selection[axis] = selection[axis] + offsets[position]
            coefficient *= weights[position] / spacings[axis]**orders[axis]
        result += coefficient * np.asarray(field[(trajectory, *selection)], dtype=np.float64)
    return result


def pointwise_design(task: str, data: dict[str, Any], terms: list[WeakTerm], *,
                     points: int, seed: int,
                     explicit_jets: Mapping[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample the same observed fields/library for ordinary pointwise SINDy.

    Saved jets are used only from the selected field directory and only when
    their complete grid shape matches. Missing jets are estimated locally from
    the observed fields with fourth-order central stencils, including temporal
    targets; no equation RHS is consulted. Raw trajectory packs remain mapped
    individually, and only sampled values are converted to float64.
    When explicit_jets is supplied, every required derivative must come from
    that mapping. No stored-jet or finite-difference fallback is permitted.
    """
    if task not in TASK_INPUTS:
        raise ValueError(f"Unsupported task: {task}")
    if isinstance(points, bool) or not isinstance(points, (int, np.integer)) or points < 2:
        raise ValueError("At least two pointwise samples are required")
    shape = data["u"].shape
    grid_shape = shape[1:]
    output_fields = ["u", "v"] if task in ("rd", "schrodinger") else ["u"]
    output_terms = [f"{field}_{'tt' if task == 'wave' else 't'}" for field in output_fields]
    derivatives = {term.expression for term in terms if term.kind == "linear"}
    if any(term.kind in ("quadratic_transport", "cubic_transport", "gradient_square") for term in terms):
        derivatives.add("u_x")
    derivatives.update(output_terms)
    jets = {}
    derivative_sources = {}
    margins = np.zeros(len(grid_shape), dtype=int)
    for name in sorted(derivatives):
        field, derivative = name.split("_", 1)
        if field not in output_fields:
            raise ValueError(f"Unavailable observed field for {name}")
        if explicit_jets is not None:
            if name not in explicit_jets or explicit_jets[name].shape != shape:
                raise ValueError(f"Explicit pointwise jets missing or shape-mismatched: {name}")
            jets[name] = explicit_jets[name]
            derivative_sources[name] = {"mode": "explicit_jet"}
            continue
        axes = _derivative_axes(derivative, len(grid_shape))
        filename = "d" + field + "".join("d" + coordinate for coordinate in derivative)
        path = Path(data["directory"]) / f"{data['split']}-{filename}.pt"
        mismatch = False
        if path.exists():
            values = _load_tensor(path)
            if values.shape == shape:
                jets[name] = values
                derivative_sources[name] = {"mode": "stored_jet", "path": str(path.resolve())}
                continue
            mismatch = True
        for axis, order in axes.items():
            offsets, _ = _central_stencil(order)
            margins[axis] = max(margins[axis], int(np.abs(offsets).max()))
        derivative_sources[name] = {
            "mode": "fourth_order_central_difference", "field": field,
            "reason": "stored_jet_shape_mismatch" if mismatch else "stored_jet_unavailable",
        }
    interior_shape = (shape[0], *(np.asarray(grid_shape) - 2 * margins).tolist())
    if min(interior_shape) <= 0:
        raise ValueError("Observed field is too small for pointwise central-difference stencils")
    population = int(np.prod(interior_shape))
    count = min(int(points), population)
    if count < 2:
        raise ValueError("Observed field has fewer than two valid pointwise samples")
    rng = np.random.default_rng(seed)
    interior_flat = rng.choice(population, size=count, replace=False)
    indices = np.column_stack(np.unravel_index(interior_flat, interior_shape))
    indices[:, 1:] += margins
    values = {name: np.empty(count, dtype=np.float64) for name in [*output_fields, *sorted(derivatives)]}
    # Grouping by trajectory prevents repeatedly reopening large raw packs.
    for trajectory in np.unique(indices[:, 0]):
        rows = np.flatnonzero(indices[:, 0] == trajectory)
        coordinates = indices[rows, 1:]
        selection = (int(trajectory), *(coordinates[:, axis] for axis in range(len(grid_shape))))
        for field in output_fields:
            values[field][rows] = np.asarray(data[field][selection], dtype=np.float64)
        for name in derivatives:
            if name in jets:
                values[name][rows] = np.asarray(jets[name][selection], dtype=np.float64)
            else:
                field, derivative = name.split("_", 1)
                values[name][rows] = _sample_derivative(
                    data[field], int(trajectory), coordinates, data["spacings"], derivative)
    design = np.empty((count, len(terms)), dtype=np.float64)
    for column, term in enumerate(terms):
        if term.kind == "polynomial":
            value = values["u"]**term.powers[0]
            if term.powers[1]:
                value = value * values["v"]**term.powers[1]
        elif term.kind == "linear":
            value = values[term.expression]
        elif term.kind == "quadratic_transport":
            value = values["u"] * values["u_x"]
        elif term.kind == "cubic_transport":
            value = values["u"]**2 * values["u_x"]
        elif term.kind == "gradient_square":
            value = values["u_x"]**2
        elif term.kind in ("sin", "cos"):
            coordinates = np.asarray(data["x_axis"], dtype=np.float64)[indices[:, -1]]
            value = np.sin(coordinates) if term.kind == "sin" else np.cos(coordinates)
        else:
            raise ValueError(f"Unknown pointwise term: {term.kind}")
        design[:, column] = value
    targets = np.column_stack([values[name] for name in output_terms])
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(targets)):
        raise ValueError("Nonfinite observations or pointwise derivatives")
    return design, targets, {
        "requested_points": int(points), "points": count, "seed": int(seed),
        "point_indices": indices.tolist(),
        "flat_indices": np.ravel_multi_index(indices.T, shape).tolist(),
        "trajectory_indices": indices[:, 0].tolist(),
        "sampling": "uniform_without_replacement_over_valid_grid_points",
        "sampling_interior_margins": margins.tolist(), "derivative_sources": derivative_sources,
        "directory": str(Path(data["directory"]).resolve()), "split": data["split"],
        "shape": list(shape), "spacings": list(data["spacings"]),
        "raw_files": data.get("raw_files", []), "dtype": "float64",
    }


def _compact_weight(length: int, rng: np.random.Generator) -> np.ndarray:
    """A smooth compact bump with room for the transferred FD stencil."""
    if length < 7:
        raise ValueError("Weak windows need at least seven grid points per axis")
    weight = np.zeros(length, dtype=np.float64)
    z = np.linspace(-1.0, 1.0, length - 4)
    weight[2:-2] = (1.0 - z * z) ** 6 * (1.0 + rng.uniform(-0.4, 0.4) * z)
    return weight


def _derivative(weight: np.ndarray, spacing: float, order: int) -> np.ndarray:
    if order == 1:
        return (np.roll(weight, -1) - np.roll(weight, 1)) / (2 * spacing)
    if order == 2:
        return (np.roll(weight, -1) - 2 * weight + np.roll(weight, 1)) / spacing**2
    if order == 3:
        return (np.roll(weight, -2) - 2 * np.roll(weight, -1)
                + 2 * np.roll(weight, 1) - np.roll(weight, 2)) / (2 * spacing**3)
    raise ValueError(f"Unsupported derivative order: {order}")


def _continuous_derivative(weight: np.ndarray, spacing: float, order: int) -> np.ndarray:
    """Analytic derivative of the known bump, used for analytic trajectories."""
    if len(weight) < 9:
        raise ValueError("Continuous test derivatives need at least nine points")
    z = np.linspace(-1.0, 1.0, len(weight) - 4)
    index = int(np.argmin(np.abs(z - 0.5)))
    slope = (weight[2:-2][index] / (1.0 - z[index]**2)**6 - 1.0) / z[index]
    polynomial = np.polynomial.Polynomial([1.0, 0.0, -1.0])**6 * np.polynomial.Polynomial([1.0, slope])
    result = np.zeros_like(weight)
    radius = spacing * (len(weight) - 5) / 2.0
    result[2:-2] = polynomial.deriv(order)(z) / radius**order
    result[2] = result[-3] = 0.0
    return result


def _outer(weights: list[np.ndarray]) -> np.ndarray:
    value = weights[0]
    for other in weights[1:]:
        value = np.multiply.outer(value, other)
    return value


def _window_lengths(shape: tuple[int, ...], time_window: int | None,
                    space_window: int | None) -> tuple[int, ...]:
    requested = [time_window or min(101, shape[0])]
    requested.extend([space_window or (41 if len(shape) == 2 else 9)] * (len(shape) - 1))
    lengths = tuple(min(int(w), n) for n, w in zip(shape, requested))
    if any(w < 7 for w in lengths):
        raise ValueError("Field is too small for compact weak windows (minimum seven points)")
    return lengths


def weak_design(task: str, data: dict[str, Any], terms: list[WeakTerm], *,
                windows: int, seed: int, time_window: int | None = None,
                space_window: int | None = None, spatial_mode: str = "auto",
                explicit_jets: Mapping[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct weak integrals using observed fields and compact windows."""
    if windows < 2:
        raise ValueError("At least two weak windows are required")
    if spatial_mode not in ("auto", "adjoint", "continuous", "jets"):
        raise ValueError("spatial_mode must be auto, adjoint, continuous, or jets")
    shape = data["u"].shape[1:]
    coarse_2d = len(shape) == 3 and min(shape[-2:]) < 17
    mode = ("jets" if coarse_2d else ("continuous" if task == "forced_transport" else "adjoint")) if spatial_mode == "auto" else spatial_mode
    lengths = _window_lengths(shape, time_window, space_window)
    if mode == "continuous" and min(lengths) < 9:
        raise ValueError("Continuous test derivatives need at least nine points per weak window axis")
    rng = np.random.default_rng(seed)
    coupled = "v" in data
    output_fields = ["u", "v"] if coupled else ["u"]
    jets = {}
    needed_jets = [term.expression for term in terms if term.kind == "linear"] if mode == "jets" else []
    if any(term.kind == "gradient_square" for term in terms):
        needed_jets.append("u_x")
    for name in needed_jets:
        if explicit_jets is not None:
            # Noise experiments must never read clean jets from data.directory.
            if name not in explicit_jets:
                raise ValueError(f"Explicit processed jets are missing {name}")
            values = explicit_jets[name]
            if values.shape != data["u"].shape:
                raise ValueError(f"Explicit processed jet shape mismatch: {name}")
            jets[name] = values
            continue
        path = data["directory"] / f"{data['split']}-{JET_FILES[name]}.pt"
        if path.exists():
            values = _load_tensor(path)
            if values.shape == data["u"].shape:
                jets[name] = values
            elif mode == "jets":
                raise ValueError(f"Stored jet shape does not match raw solution field: {path}")
            # Retained raw fields precede the standard crop/downsampling. The
            # nonconservative gradient-square term can use local raw-field
            # differences rather than incorrectly aligning the cropped jets.
        elif mode == "jets":
            raise FileNotFoundError(f"Hybrid spatial mode needs {path}; retain a fine raw grid instead")
    design = np.empty((windows, len(terms)), dtype=np.float64)
    targets = np.empty((windows, len(output_fields)), dtype=np.float64)
    rows = []
    volume_element = float(np.prod(data["spacings"]))
    for row in range(windows):
        trajectory = int(rng.integers(data["u"].shape[0]))
        starts = [int(rng.integers(n - w + 1)) for n, w in zip(shape, lengths)]
        slices = tuple(slice(s, s + w) for s, w in zip(starts, lengths))
        selection = (trajectory,) + slices
        fields = {name: np.asarray(data[name][selection], dtype=np.float64) for name in output_fields}
        axis_weights = [_compact_weight(w, rng) for w in lengths]
        phi = _outer(axis_weights)
        normalization = float(phi.sum())
        phi /= normalization
        transfer = {}
        derivative_operator = _continuous_derivative if mode == "continuous" else _derivative
        for derivative in set(term.derivative for term in terms if term.kind == "linear"):
            weights = list(axis_weights)
            if derivative == "xy":
                weights[-1] = -derivative_operator(weights[-1], data["spacings"][-1], 1)
                weights[-2] = -derivative_operator(weights[-2], data["spacings"][-2], 1)
            else:
                axis = -1 if derivative[0] == "x" else -2
                order = len(derivative)
                weights[axis] = (-1)**order * derivative_operator(weights[axis], data["spacings"][axis], order)
            transfer[derivative] = _outer(weights) / normalization
        weights = list(axis_weights)
        temporal_order = 2 if task == "wave" else 1
        weights[0] = (-1)**temporal_order * derivative_operator(weights[0], data["spacings"][0], temporal_order)
        phi_time = _outer(weights) / normalization
        for column, field in enumerate(output_fields):
            targets[row, column] = np.sum(phi_time * fields[field]) * volume_element
        u = fields["u"]
        v = fields.get("v")
        for column, term in enumerate(terms):
            if term.kind == "polynomial":
                value = u**term.powers[0]
                if term.powers[1]:
                    value = value * v**term.powers[1]
                integral = np.sum(phi * value)
            elif term.kind == "linear":
                integral = (np.sum(phi * jets[term.expression][selection]) if mode == "jets"
                            else np.sum(transfer[term.derivative] * fields[term.field]))
            elif term.kind == "quadratic_transport":
                # Exact summation by parts for the observed-grid central u D_x u.
                integral = (np.sum(transfer["x"] * u**2 / 2.0) if mode == "continuous"
                            else np.sum((phi[..., :-1] - phi[..., 1:])
                                        * u[..., :-1] * u[..., 1:]) / (2 * data["spacings"][-1]))
            elif term.kind == "cubic_transport":
                integral = np.sum(transfer["x"] * u**3 / 3.0)
            elif term.kind == "gradient_square":
                jet = jets.get("u_x")
                value = (np.asarray(jet[selection], dtype=np.float64) if jet is not None
                         else (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2 * data["spacings"][-1]))
                integral = np.sum(phi * value**2)
            elif term.kind in ("sin", "cos"):
                coordinates = data["x_axis"][slices[-1]]
                value = np.sin(coordinates) if term.kind == "sin" else np.cos(coordinates)
                integral = np.sum(phi * value)
            else:
                raise ValueError(f"Unknown weak term: {term.kind}")
            design[row, column] = integral * volume_element
        rows.append([trajectory, *starts])
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(targets)):
        raise ValueError("Nonfinite observations or weak integrals")
    warnings = []
    if coarse_2d:
        warnings.append("Coarse 2D saved fields cannot reliably resolve spatial weak derivatives; retain the fine raw grid for a fully weak fit.")
    if mode == "jets":
        warnings.append("Hybrid mode: time derivatives are weak; spatial derivative candidates use measured stored jets.")
    if task == "burger":
        warnings.append("Nonconservative u_x**2 uses a stored spatial jet or local central difference; this candidate is not derivative-free.")
    return design, targets, {
        "spatial_mode": mode, "time_mode": "continuous_test_derivative" if mode == "continuous" else "finite_difference_adjoint",
        "linear_spatial_mode": "stored_jets" if mode == "jets" else ("continuous_test_derivative" if mode == "continuous" else "finite_difference_adjoint"),
        "quadratic_transport_mode": "conservative_flux_weak" if mode == "continuous" else "exact_central_difference_summation_by_parts",
        "cubic_transport_mode": "conservative_flux_weak_approximation",
        "window_lengths": list(lengths), "window_indices": rows, "warnings": warnings,
    }


def _stlsq(matrix: np.ndarray, targets: np.ndarray, threshold: float,
           max_iterations: int = 30) -> np.ndarray:
    result = np.linalg.lstsq(matrix, targets, rcond=1e-10)[0]
    for output in range(targets.shape[1]):
        active = np.ones(matrix.shape[1], dtype=bool)
        for _ in range(max_iterations):
            retained = active & (np.abs(result[:, output]) >= threshold)
            if np.array_equal(retained, active):
                break
            active = retained
            result[:, output] = 0.0
            if not np.any(active):
                break
            result[active, output] = np.linalg.lstsq(matrix[:, active], targets[:, output], rcond=1e-10)[0]
    return result


def sparse_regression(train_matrix: np.ndarray, train_targets: np.ndarray,
                      val_matrix: np.ndarray, val_targets: np.ndarray, *,
                      thresholds: list[float] | None = None,
                      validation_tolerance: float = 0.10,
                      validation_floor: float = 1e-5) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose sparsity from independent validation weak windows, not truth."""
    if validation_tolerance < 0:
        raise ValueError("validation_tolerance must be nonnegative")
    if not np.isfinite(validation_floor) or validation_floor < 0:
        raise ValueError("validation_floor must be finite and nonnegative")
    scales = np.sqrt(np.mean(train_matrix**2, axis=0))
    scales = np.where(scales > np.finfo(float).tiny, scales, 1.0)
    target_scales = np.sqrt(np.mean(train_targets**2, axis=0))
    target_scales = np.where(target_scales > np.finfo(float).tiny, target_scales, 1.0)
    train = train_matrix / scales
    targets = train_targets / target_scales
    candidates = _threshold_candidates(thresholds)
    rows = []
    models = []
    for threshold in candidates:
        normalized_coefficients = _stlsq(train, targets, threshold)
        coefficients = normalized_coefficients * target_scales / scales[:, None]
        residual = (val_matrix @ coefficients - val_targets) / target_scales
        squared = np.mean(residual**2, axis=1)
        mse = float(np.mean(squared))
        standard_error = float(np.std(squared, ddof=1) / np.sqrt(len(squared)))
        rows.append({"threshold": float(threshold), "validation_mse": mse,
                     "validation_standard_error": standard_error,
                     "support_size": int(np.count_nonzero(coefficients))})
        models.append(coefficients)
    best = min(range(len(rows)), key=lambda i: rows[i]["validation_mse"])
    allowance = (rows[best]["validation_mse"] * (1 + validation_tolerance)
                 + rows[best]["validation_standard_error"] + validation_floor**2)
    eligible = [i for i, row in enumerate(rows) if row["validation_mse"] <= allowance]
    chosen = min(eligible, key=lambda i: (rows[i]["support_size"], rows[i]["validation_mse"]))
    coefficients = models[chosen]
    relative = lambda a, y: float(np.linalg.norm(a @ coefficients - y) / max(np.linalg.norm(y), np.finfo(float).tiny))
    return coefficients, {
        "selected_threshold": rows[chosen]["threshold"],
        "train_relative_l2": relative(train_matrix, train_targets),
        "validation_relative_l2": relative(val_matrix, val_targets),
        "validation_selection": "sparsest_within_one_standard_error_plus_relative_tolerance_and_integration_floor",
        "validation_floor": validation_floor,
        "validation_candidates": rows, "column_scales": scales.tolist(),
        "target_scales": target_scales.tolist(),
    }


def equation_strings(output_terms: list[str], terms: list[str], coefficients: list[list[float]]) -> list[str]:
    equations = []
    for output, row in zip(output_terms, coefficients):
        fragments = [f"{coefficient:+.8g}*({term})" for term, coefficient in zip(terms, row) if coefficient != 0]
        equations.append(f"{output} = " + (" ".join(fragments) if fragments else "0"))
    return equations


def fit_wsindy(task: str, data_dir: str | Path, seed: int = 0, *,
               windows: int = DEFAULT_FIT_OPTIONS["windows"],
               val_windows: int = DEFAULT_FIT_OPTIONS["val_windows"],
               time_window: int | None = DEFAULT_FIT_OPTIONS["time_window"],
               space_window: int | None = DEFAULT_FIT_OPTIONS["space_window"],
               poly_degree: int = DEFAULT_FIT_OPTIONS["poly_degree"],
               thresholds: list[float] | None = DEFAULT_FIT_OPTIONS["thresholds"],
               validation_tolerance: float = DEFAULT_FIT_OPTIONS["validation_tolerance"],
               validation_floor: float = DEFAULT_FIT_OPTIONS["validation_floor"],
               bootstrap: int = DEFAULT_FIT_OPTIONS["bootstrap"],
               spatial_mode: str = DEFAULT_FIT_OPTIONS["spatial_mode"]) -> dict[str, Any]:
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    terms = candidate_library(task, poly_degree)
    train_data = load_field_data(task, data_dir, "train")
    val_data = load_field_data(task, data_dir, "val")
    options = dict(time_window=time_window, space_window=space_window, spatial_mode=spatial_mode)
    train_matrix, train_targets, train_meta = weak_design(task, train_data, terms, windows=windows, seed=seed, **options)
    val_matrix, val_targets, val_meta = weak_design(task, val_data, terms, windows=val_windows, seed=seed + 100003, **options)
    if train_meta["spatial_mode"] != val_meta["spatial_mode"]:
        raise ValueError("Training and validation must use the same spatial weak formulation")
    coefficients, metrics = sparse_regression(train_matrix, train_targets, val_matrix, val_targets,
                                               thresholds=thresholds, validation_tolerance=validation_tolerance,
                                               validation_floor=validation_floor)
    if bootstrap:
        rng = np.random.default_rng(seed + 200003)
        scales = np.asarray(metrics["column_scales"])
        target_scales = np.asarray(metrics["target_scales"])
        fits = []
        for _ in range(bootstrap):
            indices = rng.integers(len(train_matrix), size=len(train_matrix))
            fit = _stlsq(train_matrix[indices] / scales, train_targets[indices] / target_scales,
                         metrics["selected_threshold"])
            fits.append(fit * target_scales / scales[:, None])
        fits = np.asarray(fits)
        metrics["bootstrap_support_frequency"] = np.mean(fits != 0, axis=0).T.tolist()
        metrics["bootstrap_coefficient_std"] = np.std(fits, axis=0).T.tolist()
        metrics["bootstrap_note"] = "Resampling overlapping weak windows assesses fit sensitivity; it is not an independent-trajectory confidence interval."
    output_terms = ["u_t", "v_t"] if task in ("rd", "schrodinger") else (["u_tt"] if task == "wave" else ["u_t"])
    expressions = [term.expression for term in terms]
    coefficient_rows = coefficients.T.tolist()
    data_metadata = {}
    for name, data, metadata in (("train", train_data, train_meta), ("val", val_data, val_meta)):
        data_metadata[name] = {"directory": str(data["directory"].resolve()),
                               "shape": list(data["u"].shape), "spacings": data["spacings"],
                               "raw_files": data.get("raw_files", []), **metadata}
    return {
        "format": "liediscover_wsindy_v1", "task": task, "seed": int(seed),
        "input_terms": TASK_INPUTS[task], "output_terms": output_terms,
        "terms": expressions, "coefficients": coefficient_rows,
        "equations": equation_strings(output_terms, expressions, coefficient_rows),
        "metrics": metrics,
        "fit_options": {"windows": windows, "val_windows": val_windows,
                        "time_window": time_window, "space_window": space_window,
                        "poly_degree": poly_degree, "bootstrap": bootstrap,
                        "thresholds": _threshold_candidates(thresholds),
                        "validation_tolerance": validation_tolerance,
                        "validation_floor": validation_floor, "spatial_mode": spatial_mode},
        "data": data_metadata,
    }


def save_wsindy(result: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASK_INPUTS), required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--windows", type=int, default=DEFAULT_FIT_OPTIONS["windows"])
    parser.add_argument("--val-windows", type=int, default=DEFAULT_FIT_OPTIONS["val_windows"])
    parser.add_argument("--time-window", type=int)
    parser.add_argument("--space-window", type=int)
    parser.add_argument("--poly-degree", type=int, default=DEFAULT_FIT_OPTIONS["poly_degree"])
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_FIT_OPTIONS["bootstrap"])
    parser.add_argument("--validation-tolerance", type=float, default=DEFAULT_FIT_OPTIONS["validation_tolerance"])
    parser.add_argument("--validation-floor", type=float, default=DEFAULT_FIT_OPTIONS["validation_floor"],
                        help="Absolute allowance in normalized weak validation RMS; suppresses integration-scale nuisance terms")
    parser.add_argument("--spatial-mode", choices=["auto", "adjoint", "continuous", "jets"], default=DEFAULT_FIT_OPTIONS["spatial_mode"])
    args = parser.parse_args(argv)
    result = fit_wsindy(args.task, args.data_dir, seed=args.seed, windows=args.windows,
                        val_windows=args.val_windows, time_window=args.time_window,
                        space_window=args.space_window, poly_degree=args.poly_degree,
                        bootstrap=args.bootstrap, validation_tolerance=args.validation_tolerance,
                        validation_floor=args.validation_floor,
                        spatial_mode=args.spatial_mode)
    save_wsindy(result, args.output)
    for equation in result["equations"]:
        print(equation)
    print(f"weak train relative L2: {result['metrics']['train_relative_l2']:.6g}")
    print(f"weak validation relative L2: {result['metrics']['validation_relative_l2']:.6g}")
    for warning in result["data"]["train"]["warnings"]:
        print(f"Warning: {warning}")
    print(f"Saved: {Path(args.output).resolve()}")
    return result


if __name__ == "__main__":
    main()
