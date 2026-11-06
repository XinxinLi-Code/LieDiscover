"""Load a frozen WSINDy equation for symmetry discovery without a network."""

from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Mapping
import json
from pathlib import Path

import torch
from torch import nn

from discovery import _TASK_CONFIGS


WSINDY_FORMAT = "liediscover_wsindy_v1"
_FUNCTIONS = {"sin": torch.sin, "cos": torch.cos, "exp": torch.exp,
              "tanh": torch.tanh, "log": torch.log, "sqrt": torch.sqrt}


def task_terms(task):
    """Canonical network input/evolution terms, without using a true equation."""
    if task not in _TASK_CONFIGS:
        raise ValueError(f"Unsupported PDE task {task!r}.")
    config = _TASK_CONFIGS[task]
    outputs = len(config["dep_vars"])
    return tuple(config["target_terms"][:-outputs]), tuple(config["target_terms"][-outputs:])


def jet_data_key(term):
    """Translate a canonical jet name to the saved tensor's existing key."""
    if "_" not in term:
        return term
    field, derivative = term.split("_", 1)
    if field not in {"u", "v"} or not derivative or set(derivative) - set("txy"):
        raise ValueError(f"Unsupported saved jet name {term!r}.")
    return "d" + field + "".join("d" + coordinate for coordinate in derivative)


def _check_tree(node, names):
    if isinstance(node, ast.Expression):
        _check_tree(node.body, names)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)) or not math.isfinite(node.value):
            raise ValueError("Equation constants must be finite real numbers.")
    elif isinstance(node, ast.Name):
        if node.id not in names and node.id != "pi":
            raise ValueError(f"Equation refers to unavailable input {node.id!r}.")
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _check_tree(node.operand, names)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        _check_tree(node.left, names)
        _check_tree(node.right, names)
        if isinstance(node.op, ast.Pow):
            exponent = node.right
            if isinstance(exponent, ast.UnaryOp) and isinstance(exponent.op, (ast.UAdd, ast.USub)):
                exponent = exponent.operand
            if not isinstance(exponent, ast.Constant):
                raise ValueError("Equation powers require a constant exponent.")
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
        if len(node.args) != 1 or node.keywords:
            raise ValueError("Equation functions require exactly one positional argument.")
        _check_tree(node.args[0], names)
    else:
        raise ValueError(f"Unsupported equation syntax: {type(node).__name__}.")


def _evaluate_tree(node, values, reference):
    if isinstance(node, ast.Expression):
        return _evaluate_tree(node.body, values, reference)
    if isinstance(node, ast.Constant):
        return reference.new_tensor(node.value)
    if isinstance(node, ast.Name):
        return reference.new_tensor(math.pi) if node.id == "pi" else values[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_tree(node.operand, values, reference)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left = _evaluate_tree(node.left, values, reference)
        right = _evaluate_tree(node.right, values, reference)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return left ** right
    return _FUNCTIONS[node.func.id](_evaluate_tree(node.args[0], values, reference))


def validate_wsindy_specification(specification, *, task=None):
    """Validate the JSON contract and reject executable expression syntax."""
    if not isinstance(specification, Mapping) or specification.get("format") != WSINDY_FORMAT:
        raise ValueError(f"Expected WSINDy specification format {WSINDY_FORMAT!r}.")
    task = specification.get("task") if task is None else task
    if specification.get("task") != task:
        raise ValueError("WSINDy specification does not match the requested task.")
    inputs, outputs = task_terms(task)
    if specification.get("input_terms") != list(inputs) or specification.get("output_terms") != list(outputs):
        raise ValueError("WSINDy input/output terms must match the task's canonical jet order.")
    terms = specification.get("terms")
    if not isinstance(terms, list) or not terms or any(not isinstance(term, str) or not term.strip() for term in terms):
        raise ValueError("WSINDy terms must be a nonempty list of expressions.")
    trees = []
    for term in terms:
        if len(term) > 2048:
            raise ValueError("WSINDy expression is too long.")
        try:
            tree = ast.parse(term, mode="eval")
        except (SyntaxError, RecursionError) as exc:
            raise ValueError(f"Invalid equation expression {term!r}.") from exc
        if sum(1 for _ in ast.walk(tree)) > 256:
            raise ValueError("WSINDy expression is too complex.")
        _check_tree(tree, set(inputs))
        trees.append(tree)
    try:
        coefficients = torch.as_tensor(specification.get("coefficients"), dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("WSINDy coefficients must be a rectangular numeric matrix.") from exc
    if tuple(coefficients.shape) != (len(outputs), len(terms)) or not bool(torch.isfinite(coefficients).all()):
        raise ValueError("WSINDy coefficient matrix has an invalid shape or non-finite values.")
    validated = dict(format=WSINDY_FORMAT, task=task, input_terms=list(inputs),
                     output_terms=list(outputs), terms=list(terms), coefficients=coefficients.tolist())
    return validated, trees


class ExplicitEquation(nn.Module):
    """Fixed physical RHS; its analytic derivative is obtained by autograd."""

    def __init__(self, specification, *, task=None, dtype=torch.float32):
        super().__init__()
        self.specification, self._trees = validate_wsindy_specification(specification, task=task)
        self.input_terms = tuple(self.specification["input_terms"])
        self.input_dim = len(self.input_terms)
        self.output_dim = len(self.specification["output_terms"])
        self.register_buffer("coefficients", torch.tensor(self.specification["coefficients"], dtype=dtype))

    def restore_coefficients(self):
        """Restore the saved JSON values at the equation's current precision."""
        self.coefficients.copy_(torch.tensor(
            self.specification["coefficients"],
            device=self.coefficients.device, dtype=self.coefficients.dtype,
        ))

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} physical jet inputs.")
        reference = x[..., 0]
        values = {name: x[..., index] for index, name in enumerate(self.input_terms)}
        columns = [torch.broadcast_to(_evaluate_tree(tree, values, reference), reference.shape)
                   for tree in self._trees]
        library = torch.stack(columns, dim=-1)
        return library @ self.coefficients.T


DEFAULT_WSINDY_RUN_DIR = "./result/wsindy_residual"
DEFAULT_WSINDY_SEED = 0


def resolve_wsindy_json(task, *, wsindy_json=None, wsindy_run_dir=None,
                       wsindy_seed=DEFAULT_WSINDY_SEED):
    """Keep the equation's fitting seed independent of the search seed."""
    if isinstance(wsindy_seed, bool) or not isinstance(wsindy_seed, int) or wsindy_seed < 0:
        raise ValueError("WSINDy seed must be a nonnegative integer.")
    if wsindy_json is not None:
        return Path(wsindy_json).expanduser().resolve()
    root = Path(wsindy_run_dir or DEFAULT_WSINDY_RUN_DIR).expanduser()
    return (root / f"{task}_{wsindy_seed}" / "wsindy.json").resolve()


def load_wsindy_model(path, *, task, device="cpu", expected_sha256=None,
                      input_dim=None, output_dim=None):
    """Read and hash the same JSON bytes, then construct only the explicit RHS."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"WSINDy equation JSON not found: {path}. Fit WSINDy first or provide "
            "--wsindy-json; symmetry discovery does not train or load a surrogate automatically."
        )
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"WSINDy equation JSON has changed since the saved search: {path} "
            f"(expected SHA256 {expected_sha256}, found {digest})."
        )
    try:
        specification = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid WSINDy equation JSON: {path}") from exc
    model = ExplicitEquation(specification, task=task, dtype=torch.float64).to(device=device)
    if ((input_dim is not None and model.input_dim != input_dim)
            or (output_dim is not None and model.output_dim != output_dim)):
        raise ValueError("WSINDy equation dimensions do not match discovery arguments.")
    model.eval()
    model.jacobian_source_metadata = {
        "jacobian_source": "wsindy", "task": task,
        "wsindy_json": str(path), "wsindy_sha256": digest,
        "dtype": "torch.float64", "residual_network": False,
    }
    return model
