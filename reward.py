"""Episodic reward for RL control of symbolic Theta expansion.

During an episode, each candidate term is temporarily appended to the current
Theta set and evaluated from the cached ``C^T C`` Gram matrix using the same
float64 SVD form as ``discovery.symmetry_discovery``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.autograd.functional import jvp

from discovery import (
    _TASK_CONFIGS, _parse_theta_terms, _eval_sympy_expr,
    compute_theta_n, get_theta_n_evaluator,
)
from library_evaluation_cache import LibraryEvaluation, PackedRows, pack_rows
from feature_normalization import (
    SEARCH_DTYPE, prepare_search_model, search_data,
    prolonged_column_scales, restore_coefficient_basis,
    restrict_library_scales, validate_coefficient_scales,
)
from spectral import (
    SpectralCut,
    RankSelectionConfig, RankDecision, RankUnresolvedError, select_validated_nullspace,
    DEFAULT_SPECTRAL_FLOOR, estimate_spectral_cut, select_null_basis,
    relative_singular_values,
    active_singular_condition_number as _active_singular_condition_number,
)
from theta_utils import (
    ThetaRegistry,
    ThetaTermError,
    normalize_theta_term,
    split_theta_terms,
)
from sampling_utils import capture_search_samples


@dataclass(frozen=True)
class StepHealthConfig:
    """In-episode hard-filter thresholds."""

    eps: float = 1e-12
    kappa_max: Optional[float] = 1e7
    min_cut_confidence: Optional[float] = None
    verbose: bool = False


@dataclass(frozen=True)
class StepCheck:
    """Diagnostic result for one generated action."""

    term: str
    accepted: bool
    reason: str
    tree_length: int
    reward: float = 0.0
    D_prev: int = 0
    D_raw: int = 0
    D_t: int = 0
    threshold: float = float("nan")
    cut_confidence: float = 0.0
    kappa: float = float("inf")
    sigma_max: float = float("nan")
    sigma_min_active: float = float("nan")
    rank_residual: float = float("nan")
    rank_scale_residual: float = float("nan")
    rank_stability: float = float("nan")
    function_stability_S: float = float("nan")
    rank_selection: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StepActionReward:
    """Reward assigned to one generated action during an episode."""

    term: str
    reward: float
    status: str
    tree_length: float
    D_t: int
    breakthrough_bonus: float
    delay_steps: int
    eta_discount: float
    rejected: bool


@dataclass(frozen=True)
class StepRewardResult:
    """Debug view for SVD-filtered episodic rewards."""

    rewards: Tuple[float, ...]
    action_rewards: Tuple[StepActionReward, ...]
    D0: int
    D_best: int
    delta_D: int
    threshold_initial: float
    threshold_best: float
    basis_source: str
    theta_terms: Tuple[str, ...]
    null_basis: torch.Tensor
    feature_scales: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float64))
    D_final: int = 0
    kappa: float = float("inf")
    spectral_floor: float = DEFAULT_SPECTRAL_FLOOR
    coefficient_coordinates: str = "normalized"
    normalization_metadata: dict = field(default_factory=dict)


def reward_eta_lower_bound(k_max: int) -> float:
    """Lower bound that keeps the longest breakthrough discount above 0.5."""

    k_max = int(k_max)
    if k_max <= 1:
        return 0.0
    return float(0.5 ** (1.0 / float(k_max - 1)))


def validate_reward_eta(eta: float, k_max: Optional[int]) -> float:
    """Validate eta in the delayed-breakthrough reward."""

    eta = float(eta)
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError(f"reward eta must be finite and positive, got {eta}.")
    if k_max is not None:
        lower = reward_eta_lower_bound(int(k_max))
        if eta <= lower:
            raise ValueError(
                f"reward eta must be > 0.5 ** (1 / (K_max - 1)) = {lower:.6g} "
                f"for K_max={int(k_max)}, got {eta:.6g}."
            )
    return eta


@dataclass
class SampleContext:
    """Cached sample data and neural tangent row for one data point."""

    data: Mapping[str, torch.Tensor]
    df: torch.Tensor


@dataclass(frozen=True)
class RankReferenceContext:
    """Fixed role indices into one cached list of samples and Jacobians.

    The normalization pool, three fitting subsets, and selection-validation
    points have fixed identities for the entire search and subsequent pruning.
    """

    samples: Sequence[SampleContext]
    subset_indices: Tuple[Tuple[int, ...], ...]
    scale_indices: Tuple[int, ...]
    validation_indices: Tuple[int, ...]
    provenance: dict = field(default_factory=dict)
    # Raw function values depend only on this fixed sample context, not on a
    # library's column scales. Share them across rollouts and pruning trials.
    function_value_cache: dict = field(default_factory=dict, init=False, compare=False, repr=False)

    def __post_init__(self):
        if len(self.subset_indices) != 3:
            raise ValueError("Exactly three fitting subsets are required.")
        groups = (*self.subset_indices, self.scale_indices, self.validation_indices)
        for group in groups:
            if not group or len(set(group)) != len(group):
                raise ValueError("Rank sample roles must be nonempty with unique positions.")
            if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(self.samples)
                   for i in group):
                raise ValueError("Rank sample position is invalid.")
        fitting = set().union(*map(set, self.subset_indices), set(self.scale_indices))
        if fitting.intersection(self.validation_indices):
            raise ValueError("Selection-validation points must not enter fitting or normalization.")


@dataclass
class EpisodeGramState:
    """State carried through one SVD-filtered episode."""

    model: object
    train_loader: object
    task: str
    device: torch.device
    sample_index: int
    sample_source: str
    sample_indices: Tuple[int, ...]
    input_dim: int
    output_dim: int
    variable_names: Tuple[str, ...]
    variable_dim: int
    theta_terms: List[str]
    initial_theta_terms: Tuple[str, ...]
    samples: List[SampleContext]
    c_rows: List[torch.Tensor]  # J_i (Theta_n_i / column_scales), in variable blocks
    gram: torch.Tensor
    initial_gram: torch.Tensor
    initial_D0: int
    initial_threshold: float
    current_D: int
    D_best: int
    h: int
    k: int
    threshold_best: float
    registry: ThetaRegistry
    generated_terms: List[str]
    tree_lengths: List[int]
    rejected: List[bool]
    step_checks: List[StepCheck]
    step_rewards: List[float]
    step_action_rewards: List[StepActionReward]
    term_ci_cache: dict
    verbose: bool = False
    feature_scales: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float64))
    initial_feature_scales: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float64))
    term_scale_cache: dict = field(default_factory=dict)
    scale_eps: float = 1e-12
    spectral_floor: float = DEFAULT_SPECTRAL_FLOOR
    last_kappa: float = float("inf")
    normalization_metadata: dict = field(default_factory=dict)
    rank_reference: Optional[RankReferenceContext] = None
    rank_config: Optional[RankSelectionConfig] = None
    rank_decision: Optional[RankDecision] = None
    initial_rank_decision: Optional[RankDecision] = None
    evaluation_cache: object = None


def clone_episode_initial_state(state: EpisodeGramState) -> EpisodeGramState:
    """Clone a cached initial episode state for a fresh rollout.

    Heavy read-only objects such as samples and initial C rows are shared.
    Mutable per-episode containers such as the registry and reward traces are
    rebuilt so one rollout cannot leak accepted/rejected terms into another.
    """

    registry = ThetaRegistry(
        theta_acc=tuple(state.initial_theta_terms),
        variable_names=state.variable_names,
    )
    return EpisodeGramState(
        model=state.model,
        train_loader=state.train_loader,
        task=state.task,
        device=state.device,
        sample_index=state.sample_index,
        sample_source=state.sample_source,
        sample_indices=tuple(state.sample_indices),
        input_dim=int(state.input_dim),
        output_dim=int(state.output_dim),
        variable_names=tuple(state.variable_names),
        variable_dim=int(state.variable_dim),
        theta_terms=list(state.initial_theta_terms),
        initial_theta_terms=tuple(state.initial_theta_terms),
        samples=state.samples,
        c_rows=state.c_rows.copy(),
        gram=state.initial_gram.clone(),
        initial_gram=state.initial_gram.clone(),
        initial_D0=int(state.initial_D0),
        initial_threshold=float(state.initial_threshold),
        current_D=int(state.initial_D0),
        D_best=int(state.initial_D0),
        h=0,
        k=0,
        threshold_best=float(state.initial_threshold),
        registry=registry,
        generated_terms=[],
        tree_lengths=[],
        rejected=[],
        step_checks=[],
        step_rewards=[],
        step_action_rewards=[],
        term_ci_cache=state.term_ci_cache,
        verbose=bool(state.verbose),
        feature_scales=state.initial_feature_scales.clone(),
        initial_feature_scales=state.initial_feature_scales.clone(),
        term_scale_cache=state.term_scale_cache,
        scale_eps=state.scale_eps,
        spectral_floor=state.spectral_floor,
        last_kappa=state.last_kappa,
        normalization_metadata=dict(state.normalization_metadata),
        rank_reference=state.rank_reference, rank_config=state.rank_config,
        rank_decision=state.initial_rank_decision,
        initial_rank_decision=state.initial_rank_decision,
        evaluation_cache=state.evaluation_cache,
    )


def _as_tensor(value, device) -> torch.Tensor:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=device, dtype=SEARCH_DTYPE)
    return torch.as_tensor(value, dtype=SEARCH_DTYPE, device=device)


def _task_variable_names(task: str) -> Tuple[str, ...]:
    if task == "top":
        return ("p0", "p1", "p2", "p3")
    config = _TASK_CONFIGS[task]
    return tuple(config["indep_vars"] + config["dep_vars"])


def _feature_from_data(data: Mapping[str, torch.Tensor], task: str) -> torch.Tensor:
    if task == "top":
        return data["x"].reshape(data["x"].shape[0], -1)
    if task in ["heat", "burger", "damped_burger"]:
        return torch.stack([data["u"], data["dudx"], data["dudxdx"]], dim=1)
    if task == "forced_transport":
        return torch.stack([data["x"], data["u"], data["dudx"], data["dudxdx"]], dim=1)
    if task == "kdv":
        return torch.stack(
            [data["u"], data["dudx"], data["dudxdx"], data["dudxdxdx"]],
            dim=1,
        )
    if task == "wave":
        return torch.stack(
            [
                data["u"],
                data["dudx"],
                data["dudy"],
                data["dudxdx"],
                data["dudydy"],
                data["dudxdy"],
            ],
            dim=1,
        )
    if task in ["rd", "schrodinger"]:
        return torch.stack(
            [
                data["u"],
                data["dudx"],
                data["dudy"],
                data["dudxdx"],
                data["dudydy"],
                data["dudxdy"],
                data["v"],
                data["dvdx"],
                data["dvdy"],
                data["dvdxdx"],
                data["dvdydy"],
                data["dvdxdy"],
            ],
            dim=1,
        )
    raise NotImplementedError(f"Unsupported task for reward.py: {task}")


def _df_from_model(
    model,
    feature: torch.Tensor,
    *,
    task: str,
    input_dim: int,
    output_dim: int,
    device,
) -> torch.Tensor:
    model = prepare_search_model(model, device)
    model_dtype = SEARCH_DTYPE
    feature = feature.to(device=device, dtype=model_dtype)
    if task == "top":
        df = torch.zeros((output_dim, input_dim), device=device, dtype=model_dtype)
    elif task in ["heat", "burger", "damped_burger", "forced_transport", "kdv", "wave", "rd", "schrodinger"]:
        df = torch.zeros((output_dim, input_dim + output_dim), device=device, dtype=model_dtype)
        df[:, input_dim:] = -torch.eye(output_dim, device=device, dtype=model_dtype)
    else:
        raise NotImplementedError(f"Unsupported task for reward.py: {task}")

    for i in range(input_dim):
        v = torch.zeros(input_dim, device=device, dtype=model_dtype)
        v[i] = 1
        df[:, i] = jvp(model, feature[0], v=v)[1]
    return df


def _move_loader_batch_to_device(data: Mapping[str, torch.Tensor], device) -> Mapping[str, torch.Tensor]:
    """Move one batch to float64 before evaluation without changing sampling."""
    return search_data(data, device)


def _iter_train_loader_window(
    train_loader,
    *,
    sample: int,
    sample_size: int,
    device,
):
    """Yield the same train_loader window used by discovery.symmetry_discovery."""

    sample_size = int(sample_size)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    start = int(sample) * sample_size
    stop = (int(sample) + 1) * sample_size
    for j, data in enumerate(train_loader):
        if j < start:
            continue
        if j >= stop:
            break
        yield _move_loader_batch_to_device(data, device)


def _collect_fixed_train_loader_window(
    train_loader,
    *,
    sample: int,
    sample_size: int,
    seed: int,
    device,
) -> List[Mapping[str, torch.Tensor]]:
    """Collect a deterministic DataLoader window.

    Shared loaders from ``sampling_utils`` already have a fixed order and no
    explicit generator. For legacy shuffled loaders, temporarily seed either the
    loader generator or the global torch RNG while the iterator is consumed, then
    restore the original state so policy sampling is not perturbed.
    """

    loader_generator = getattr(train_loader, "generator", None)
    if loader_generator is None:
        rng_state = torch.random.get_rng_state()
        torch.random.manual_seed(int(seed))
        try:
            return list(
                _iter_train_loader_window(
                    train_loader,
                    sample=sample,
                    sample_size=sample_size,
                    device=device,
                )
            )
        finally:
            torch.random.set_rng_state(rng_state)

    rng_state = loader_generator.get_state()
    loader_generator.manual_seed(int(seed))
    try:
        return list(
            _iter_train_loader_window(
                train_loader,
                sample=sample,
                sample_size=sample_size,
                device=device,
            )
        )
    finally:
        loader_generator.set_state(rng_state)


def _gram_from_rows(c_rows: Sequence[torch.Tensor]) -> torch.Tensor:
    if not c_rows:
        raise ValueError("c_rows must be non-empty.")
    cols = c_rows[0].shape[1]
    gram = torch.zeros((cols, cols), device=c_rows[0].device, dtype=torch.float64)
    for ci in c_rows:
        ci = ci.to(dtype=torch.float64)
        gram = gram + ci.T @ ci
    return gram


def _svd_from_gram(gram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match discovery.symmetry_discovery's SVD on a cached C^T C matrix."""

    _, S2, Vh = torch.linalg.svd(gram.to(torch.float64))
    return torch.sqrt(torch.clamp(S2, min=0.0)), Vh


def _spectral_cut_kwargs(**cut_kwargs) -> dict:
    params = dict(cut_kwargs)
    params.setdefault("threshold_min", None)
    params.setdefault("threshold_max", None)
    params.setdefault("use_threshold_bounds", False)
    params.setdefault("spectral_floor", DEFAULT_SPECTRAL_FLOOR)
    return params


def _state_cut_kwargs(state, **kwargs):
    params = {
        "threshold_min": state.normalization_metadata.get("threshold_min"),
        "threshold_max": state.normalization_metadata.get("threshold_max"),
        "use_threshold_bounds": state.normalization_metadata.get("use_threshold_bounds", False),
        "spectral_floor": state.spectral_floor,
    }
    fixed_threshold = state.normalization_metadata.get("fixed_threshold")
    if fixed_threshold is not None:
        params["fixed_threshold"] = fixed_threshold
    params.update(kwargs)
    return _spectral_cut_kwargs(**params)


def _concat_ci_blocks(
    old_ci: torch.Tensor,
    delta_ci: torch.Tensor,
    *,
    old_r: int,
    delta_r: int,
    variable_dim: int,
) -> torch.Tensor:
    """Append delta terms while preserving discovery.py's variable-block order."""

    blocks = []
    for block_idx in range(variable_dim):
        old_block = old_ci[:, block_idx * old_r : (block_idx + 1) * old_r]
        delta_block = delta_ci[:, block_idx * delta_r : (block_idx + 1) * delta_r]
        blocks.append(torch.cat([old_block, delta_block], dim=1))
    return torch.cat(blocks, dim=1)


def _compute_ci_rows_for_terms(
    samples: Sequence[SampleContext],
    *,
    task: str,
    device,
    theta_terms: Sequence[str],
    eps_s: float = 1e-12,
    scale_indices=None,
    prepare_evaluator=False,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Compute prolonged-column RMS and normalized C rows from the same B rows.

    Scaling happens before multiplication by the surrogate Jacobian. Each
    coefficient column receives one fixed scale across all reference samples.
    """
    theta_string = ",".join(theta_terms)
    evaluate = (get_theta_n_evaluator(task, device, theta_terms=theta_string)
                if prepare_evaluator else None)
    prolonged_rows = []
    for sample in samples:
        data = search_data(sample.data, device)
        theta_n = (evaluate(data) if evaluate is not None else
                   compute_theta_n(data, task, device, theta_terms=theta_string))
        prolonged_rows.append(theta_n.to(torch.float64))
    scale_rows = prolonged_rows if scale_indices is None else [prolonged_rows[i] for i in scale_indices]
    scales = prolonged_column_scales(scale_rows, eps_s=eps_s)
    rows = [
        sample.df.to(torch.float64) @ (theta_n / scales.unsqueeze(0))
        for sample, theta_n in zip(samples, prolonged_rows)
    ]
    return rows, scales


def _cached_normalized_ci_rows(
    samples, *, task, device, theta_terms, variable_dim, scale_cache, ci_cache, eps_s,
    scale_indices=None,
):
    """Cache each term's variable-block columns and scales on fixed samples."""
    missing = tuple(
        term for term in theta_terms
        if term not in scale_cache or (term,) not in ci_cache
    )
    if missing:
        missing_rows, values = _compute_ci_rows_for_terms(
            samples, task=task, device=device, theta_terms=missing, eps_s=eps_s,
            scale_indices=scale_indices,
            prepare_evaluator=scale_indices is not None,
        )
        values = validate_coefficient_scales(values, len(missing), variable_dim, device)
        blocks = values.reshape(variable_dim, len(missing))
        for j, term in enumerate(missing):
            scale_cache[term] = blocks[:, j].detach().clone()
            indices = [b * len(missing) + j for b in range(variable_dim)]
            term_rows = [row[:, indices].detach().clone() for row in missing_rows]
            ci_cache[(term,)] = pack_rows(term_rows) if scale_indices is not None else term_rows
    scales = torch.stack([scale_cache[term] for term in theta_terms], dim=1).reshape(-1)
    scales = validate_coefficient_scales(scales, len(theta_terms), variable_dim, device)
    cached = [ci_cache[(term,)] for term in theta_terms]
    if all(isinstance(rows, PackedRows) for rows in cached):
        packed = torch.stack([rows.packed for rows in cached], dim=3)
        rows = PackedRows(packed.reshape(len(samples), packed.shape[1], -1))
    else:
        rows = [
            torch.stack([ci_cache[(term,)][i] for term in theta_terms], dim=2)
            .reshape(samples[i].df.shape[0], variable_dim * len(theta_terms))
            .to(device=device, dtype=torch.float64)
            for i in range(len(samples))
        ]
    return rows, scales


def _primary_gram(rows, rank_reference=None):
    selected = rows if rank_reference is None else [rows[i] for i in rank_reference.subset_indices[0]]
    return _gram_from_rows(selected)


def _function_rows_for_terms(rank_reference, *, task, device, theta_terms, scales):
    """Map normalized coefficients to generator values on the common scale pool.

    Evaluate only the original functions, with the same expression interpreter
    as prolongation. No derivatives, Jacobians, or new sample points are needed.
    Columns keep variable-block order and use the existing prolonged scales.
    """
    terms = tuple(theta_terms)
    names = _task_variable_names(task)
    device = torch.device(device)
    scales = validate_coefficient_scales(scales, len(terms), len(names), device)
    cache = rank_reference.function_value_cache
    field_key = ("fields", task, str(device))
    count = len(rank_reference.scale_indices)
    if field_key not in cache:
        samples = [rank_reference.samples[i] for i in rank_reference.scale_indices]
        data = {
            name: torch.stack([
                torch.as_tensor(sample.data[name], dtype=SEARCH_DTYPE, device=device).reshape(())
                for sample in samples
            ]) for name in names if name in samples[0].data
        }
        # Constant-only libraries also work without coordinate fields.
        data["_function_reference"] = torch.zeros(count, dtype=SEARCH_DTYPE, device=device)
        cache[field_key] = data
    data = cache[field_key]
    keys = [("values", task, str(device), term) for term in terms]
    missing = [(term, key) for term, key in zip(terms, keys) if key not in cache]
    if missing:
        expressions = _parse_theta_terms(task, tuple(term for term, _ in missing))
        values = []
        for expression in expressions:
            value = _eval_sympy_expr(expression, data, device).to(dtype=SEARCH_DTYPE)
            values.append(value.expand(count) if value.ndim == 0 else value.reshape(count))
        values = torch.stack(values, dim=1).detach()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Generator functions must be finite on the scale-reference pool.")
        for j, (_, key) in enumerate(missing):
            cache[key] = values[:, j].clone()
    values = torch.stack([cache[key] for key in keys], dim=1)
    rows = torch.zeros((count, len(names), len(names) * len(terms)),
                       dtype=SEARCH_DTYPE, device=device)
    for component in range(len(names)):
        block = slice(component * len(terms), (component + 1) * len(terms))
        rows[:, component, block] = values / scales[block]
    return rows.reshape(count * len(names), -1) / math.sqrt(count)


def _validated_rank_from_rows(rows, rank_reference, rank_config, *, function_rows,
                              primary_gram=None):
    decompositions = [
        _svd_from_gram(primary_gram if index == 0 and primary_gram is not None else
                       _gram_from_rows([rows[i] for i in indices]))
        for index, indices in enumerate(rank_reference.subset_indices)
    ]
    return select_validated_nullspace(
        tuple(item[0] for item in decompositions),
        tuple(item[1] for item in decompositions),
        torch.cat([rows[i] for i in rank_reference.validation_indices], dim=0),
        rank_config,
        scale_rows=torch.cat([rows[i] for i in rank_reference.scale_indices], dim=0),
        function_rows=function_rows,
    )


def _library_cache_key(state, terms, construction):
    # Context/config identities and exact column order are part of the key;
    # get() additionally checks the actual float64 scale values.
    term_versions = []
    for term in terms:
        rows = state.term_ci_cache.get((term,))
        scales = state.term_scale_cache.get(term)
        packed = getattr(rows, "packed", None)
        term_versions.append((id(rows), id(packed), getattr(packed, "_version", None),
                              id(scales), getattr(scales, "_version", None)))
    # Keep construction paths separate: even equal-valued matrices can reach
    # different backend kernels through different tensor layouts.
    parent = tuple(state.theta_terms) if construction in ("append", "prune") else ()
    return (id(state.rank_reference), id(state.samples), id(state.term_ci_cache),
            state.task, str(state.device), state.scale_eps, state.rank_config,
            tuple(terms), tuple(term_versions), construction, parent)


def _cached_library_evaluation(state, terms, scales, *, construction="append"):
    if state.rank_reference is None or state.evaluation_cache is None:
        return None
    return state.evaluation_cache.get(_library_cache_key(state, terms, construction), scales)


def _remember_library_evaluation(state, terms, rows, scales, gram, decision, *, construction="append"):
    if state.rank_reference is not None and state.evaluation_cache is not None:
        state.evaluation_cache.put(_library_cache_key(state, terms, construction),
                                   LibraryEvaluation(rows, scales, gram, decision))


def _append_library_rows(old_rows, delta_rows, *, old_r, variable_dim, packed=False):
    if packed:
        old_rows, delta_rows = pack_rows(old_rows), pack_rows(delta_rows)
        if isinstance(old_rows, PackedRows) and isinstance(delta_rows, PackedRows):
            old = old_rows.packed.reshape(len(old_rows), old_rows.packed.shape[1], variable_dim, old_r)
            delta = delta_rows.packed.reshape(len(delta_rows), delta_rows.packed.shape[1], variable_dim, -1)
            joined = torch.cat((old, delta), dim=3)
            return PackedRows(joined.reshape(len(old_rows), joined.shape[1], -1))
    return [_concat_ci_blocks(old_ci, delta_ci, old_r=old_r, delta_r=1, variable_dim=variable_dim)
            for old_ci, delta_ci in zip(old_rows, delta_rows)]


def _rank_metadata(reference, decision):
    metadata = decision.metadata()
    metadata["sampling"] = {key: value for key, value in reference.provenance.items()
                            if key != "search_samples"}
    return metadata


def build_episode_state(
    *, model, train_loader, task: str, device, sample: int, input_dim: int,
    output_dim: int, initial_theta_terms: Union[str, Sequence[str]],
    initial_D0: Optional[int] = None, initial_threshold: Optional[float] = None,
    fix_flag: bool = False, fix_seed: int = 0, sample_size: int = 100,
    verbose: bool = False, sample_contexts=None, scale_eps: float = 1e-12,
    term_scale_cache=None, term_ci_cache=None, rank_reference=None, rank_config=None,
    evaluation_cache=None, **cut_kwargs,
) -> EpisodeGramState:
    """Build a normalized Gram matrix on a fixed reference sample context.

    Symbolic expressions remain unchanged; cached C rows use normalized columns.
    ``initial_D0`` is an expected value for diagnostics, never a rank override.
    ``initial_threshold`` supplies a fixed relative cutoff for every evaluation.
    """
    variable_names = _task_variable_names(task)
    initial_terms = tuple(split_theta_terms(initial_theta_terms))
    registry = ThetaRegistry(theta_acc=initial_terms, variable_names=variable_names)
    samples = rank_reference.samples if rank_reference is not None else sample_contexts
    sample_source = "fixed_reference_samples"
    if samples is None:
        model = prepare_search_model(model, device)
        model.eval()
        samples = []
        data_iter = (
            _collect_fixed_train_loader_window(
                train_loader, sample=sample, sample_size=sample_size,
                seed=fix_seed, device=device,
            ) if fix_flag else _iter_train_loader_window(
                train_loader, sample=sample, sample_size=sample_size, device=device,
            )
        )
        with torch.no_grad():
            for data in data_iter:
                df = _df_from_model(
                    model, _feature_from_data(data, task), task=task,
                    input_dim=input_dim, output_dim=output_dim, device=device,
                )
                samples.append(SampleContext(data=data, df=df))
    if not samples:
        raise ValueError("No reference samples were processed; check the sample window.")
    scales_cache = {} if term_scale_cache is None else term_scale_cache
    ci_cache = {} if term_ci_cache is None else term_ci_cache
    c_rows, scales = _cached_normalized_ci_rows(
        samples, task=task, device=device, theta_terms=initial_terms,
        variable_dim=len(variable_names), scale_cache=scales_cache,
        ci_cache=ci_cache, eps_s=scale_eps,
        scale_indices=rank_reference.scale_indices if rank_reference is not None else None,
    )
    gram = _primary_gram(c_rows, rank_reference)
    params = _spectral_cut_kwargs(**cut_kwargs)
    if initial_threshold is not None:
        params["fixed_threshold"] = initial_threshold
    rank_decision = None
    if rank_reference is not None:
        if initial_threshold is not None or params["use_threshold_bounds"]:
            raise ValueError("Validated dimension selection cannot use legacy fixed/bounded cutoffs.")
        rank_config = rank_config or RankSelectionConfig(spectral_floor=params["spectral_floor"])
        if rank_config.spectral_floor != params["spectral_floor"]:
            raise ValueError("Rank selection and search must use the same spectral floor.")
        rank_decision = _validated_rank_from_rows(
            c_rows, rank_reference, rank_config, primary_gram=gram,
            function_rows=_function_rows_for_terms(
                rank_reference, task=task, device=device, theta_terms=initial_terms, scales=scales),
        )
        if not rank_decision.resolved:
            raise RankUnresolvedError(rank_decision)
        actual_D0 = int(rank_decision.dimension)
        threshold = rank_decision.threshold
        kappa = rank_decision.kappa
    else:
        initial_S, _ = _svd_from_gram(gram)
        initial_cut = estimate_spectral_cut(initial_S, **params)
        actual_D0 = int(initial_cut.null_dim)
        threshold = float(initial_cut.threshold)
        kappa, _, _ = _active_singular_condition_number(
            initial_S, threshold=initial_cut.threshold, spectral_floor=params["spectral_floor"],
        )
    metadata = {
        "normalization_version": 2, "method": "prolonged_column_rms",
        "scale_layout": "variable_blocks",
        "scale_source": "prolonged_features_before_jacobian", "scale_eps": float(scale_eps),
        "spectral_floor": float(params["spectral_floor"]),
        "threshold_min": params["threshold_min"], "threshold_max": params["threshold_max"],
        "use_threshold_bounds": params["use_threshold_bounds"],
        "threshold_kind": "relative", "spectrum": "normalized_system",
        "computation_dtype": "float64", "jacobian_dtype": "float64",
        "fixed_threshold": initial_threshold,
        "sample_index": int(sample), "sample_size": int(sample_size),
        "sample_seed": int(fix_seed), "sample_count": len(samples),
        "expected_D0": initial_D0,
    }
    source = getattr(model, "jacobian_source_metadata", None)
    if source is not None:
        metadata["jacobian_source"] = dict(source)
    provenance = (rank_reference.provenance.get("search_samples") if rank_reference is not None else
                  capture_search_samples(train_loader, sample=sample, sample_size=sample_size,
                                         sample_count=len(samples)))
    sample_indices = ()
    if provenance is not None:
        sample_indices = tuple(provenance["indices"])
        metadata["search_samples"] = provenance
        metadata["sample_indices"] = list(sample_indices)
        metadata["sample_seed"] = provenance.get("seed", fix_seed)
    if rank_decision is not None:
        metadata["rank_selection"] = _rank_metadata(rank_reference, rank_decision)
        metadata["threshold_kind"] = "display_only"
    else:
        metadata["rank_selection"] = {
            "method": "fixed_relative_cutoff" if initial_threshold is not None else "largest_gap",
            "legacy": True, "resolved": True, "dimension": actual_D0,
            "threshold_role": "selection_cutoff", "threshold": float(threshold),
        }
    state = EpisodeGramState(
        model=model, train_loader=train_loader, task=task, device=torch.device(device),
        sample_index=int(sample), sample_source=sample_source, sample_indices=sample_indices,
        input_dim=int(input_dim), output_dim=int(output_dim), variable_names=variable_names,
        variable_dim=len(variable_names), theta_terms=list(initial_terms),
        initial_theta_terms=initial_terms, samples=samples, c_rows=c_rows, gram=gram,
        initial_gram=gram.clone(), initial_D0=actual_D0,
        initial_threshold=float(threshold), current_D=actual_D0, D_best=actual_D0,
        h=0, k=0, threshold_best=float(threshold), registry=registry,
        generated_terms=[], tree_lengths=[], rejected=[], step_checks=[], step_rewards=[],
        step_action_rewards=[], term_ci_cache=ci_cache, verbose=bool(verbose),
        feature_scales=scales, initial_feature_scales=scales.clone(),
        term_scale_cache=scales_cache, scale_eps=float(scale_eps),
        spectral_floor=float(params["spectral_floor"]), last_kappa=kappa,
        normalization_metadata=metadata,
        rank_reference=rank_reference, rank_config=rank_config,
        rank_decision=rank_decision, initial_rank_decision=rank_decision,
        evaluation_cache=evaluation_cache,
    )
    _remember_library_evaluation(state, initial_terms, c_rows, scales, gram, rank_decision,
                                 construction="initial")
    return state


def evaluate_library_subset(reference_state, theta_terms, *, scales=None, **cut_kwargs):
    """Restrict normalized reference columns and their fixed scales for pruning."""
    terms = tuple(split_theta_terms(theta_terms))
    positions = {term: i for i, term in enumerate(reference_state.theta_terms)}
    if len(set(terms)) != len(terms) or any(term not in positions for term in terms):
        raise ValueError("A pruning trial must be a unique subset of the reference library.")
    retained = [positions[term] for term in terms]
    old_r = len(reference_state.theta_terms)
    indices = [b * old_r + j for b in range(reference_state.variable_dim) for j in retained]
    fixed_scales = restrict_library_scales(
        reference_state.feature_scales, retained, num_terms=old_r,
        variable_dim=reference_state.variable_dim, device=reference_state.device,
    )
    if scales is not None:
        scales = validate_coefficient_scales(
            scales, len(terms), reference_state.variable_dim, reference_state.device,
        )
        expected_scales = validate_coefficient_scales(
            fixed_scales, len(terms), reference_state.variable_dim, reference_state.device,
        )
        if not torch.equal(scales, expected_scales):
            raise ValueError("Pruning must reuse the original scales of retained terms.")
    cached = _cached_library_evaluation(reference_state, terms, fixed_scales, construction="prune")
    if cached is not None:
        rows, gram, decision = cached.rows.copy(), cached.gram, cached.decision
    else:
        if isinstance(reference_state.c_rows, PackedRows):
            rows = PackedRows(reference_state.c_rows.packed[:, :, indices])
        else:
            rows = [ci[:, indices] for ci in reference_state.c_rows]
        gram = _primary_gram(rows, reference_state.rank_reference)
        decision = (None if reference_state.rank_reference is None else
                    _validated_rank_from_rows(rows, reference_state.rank_reference,
                                              reference_state.rank_config, primary_gram=gram,
                                              function_rows=_function_rows_for_terms(
                                                  reference_state.rank_reference,
                                                  task=reference_state.task,
                                                  device=reference_state.device,
                                                  theta_terms=terms, scales=fixed_scales)))
        _remember_library_evaluation(reference_state, terms, rows, fixed_scales, gram, decision,
                                     construction="prune")
    if decision is not None and not decision.resolved:
        raise RankUnresolvedError(decision)
    trial = replace(
        reference_state, theta_terms=list(terms), c_rows=rows, gram=gram,
        feature_scales=fixed_scales, rank_decision=decision,
    )
    return compute_step_reward_result(state=trial, **cut_kwargs)


def _append_step_reward(
    state: EpisodeGramState,
    check: StepCheck,
    *,
    status: str,
    breakthrough_bonus: float,
    delay_steps: Optional[int] = None,
    eta_discount: float = 1.0,
) -> None:
    if delay_steps is None:
        delay_steps = state.h
    state.step_checks.append(check)
    state.step_rewards.append(float(check.reward))
    state.step_action_rewards.append(
        StepActionReward(
            term=check.term,
            reward=float(check.reward),
            status=status,
            tree_length=float(check.tree_length),
            D_t=int(check.D_t),
            breakthrough_bonus=float(breakthrough_bonus),
            delay_steps=int(delay_steps),
            eta_discount=float(eta_discount),
            rejected=not bool(check.accepted),
        )
    )
    # Every proposal, including an invalid or rejected one, advances the MDP.
    state.h = 0 if check.accepted and check.D_t > check.D_prev else state.h + 1
    state.k += 1


def try_add_term_with_svd(
    state: EpisodeGramState,
    term,
    *,
    health_config: StepHealthConfig = StepHealthConfig(),
    library=None,
    alpha: float = 0.1,
    lambda_len: float = 0.02,
    eta: float = 0.86,
    k_max: Optional[int] = None,
    **cut_kwargs,
) -> StepCheck:
    """Try one generated term using candidate SVD filtering and step reward."""

    eta = validate_reward_eta(eta, k_max)

    try:
        normalized = normalize_theta_term(
            term,
            library=library,
            variable_names=state.variable_names,
        )
    except ThetaTermError as exc:
        check = StepCheck(
            term=str(term),
            accepted=False,
            reason=f"invalid:{exc}",
            tree_length=1,
            reward=0.0,
            D_prev=state.current_D,
            D_raw=state.current_D,
            D_t=state.current_D,
            cut_confidence=0.0,
        )
        state.generated_terms.append(str(term))
        state.tree_lengths.append(check.tree_length)
        state.rejected.append(True)
        _append_step_reward(state, check, status="rejected", breakthrough_bonus=0.0)
        return check

    duplicate = state.registry.duplicate_location(normalized)
    state.generated_terms.append(normalized.canonical)
    state.tree_lengths.append(int(normalized.complexity))

    if normalized.is_zero:
        reason = "zero_term"
        state.rejected.append(True)
        check = StepCheck(
            term=normalized.canonical,
            accepted=False,
            reason=reason,
            tree_length=normalized.complexity,
            reward=0.0,
            D_prev=state.current_D,
            D_raw=state.current_D,
            D_t=state.current_D,
            cut_confidence=0.0,
        )
        _append_step_reward(state, check, status="rejected", breakthrough_bonus=0.0)
        return check

    if duplicate is not None:
        reason = f"duplicate_{duplicate}"
        state.rejected.append(True)
        check = StepCheck(
            term=normalized.canonical,
            accepted=False,
            reason=reason,
            tree_length=normalized.complexity,
            reward=0.0,
            D_prev=state.current_D,
            D_raw=state.current_D,
            D_t=state.current_D,
            cut_confidence=0.0,
        )
        _append_step_reward(state, check, status="rejected", breakthrough_bonus=0.0)
        return check

    old_r = len(state.theta_terms)
    delta_terms = (normalized.canonical,)
    trial_terms = tuple(state.theta_terms) + delta_terms
    cached = None
    try:
        old_scales = validate_coefficient_scales(
            state.feature_scales, old_r, state.variable_dim, state.device,
        )
        if normalized.canonical in state.term_scale_cache:
            delta_scales = state.term_scale_cache[normalized.canonical]
            new_scales = torch.cat((old_scales.reshape(state.variable_dim, old_r),
                                    delta_scales.reshape(state.variable_dim, 1)), dim=1).reshape(-1)
            cached = _cached_library_evaluation(state, trial_terms, new_scales)
        if cached is not None:
            new_rows, new_scales, new_gram = cached.rows.copy(), cached.scales, cached.gram
        else:
            delta_rows, delta_scales = _cached_normalized_ci_rows(
                state.samples, task=state.task, device=state.device, theta_terms=delta_terms,
                variable_dim=state.variable_dim, scale_cache=state.term_scale_cache,
                ci_cache=state.term_ci_cache, eps_s=state.scale_eps,
                scale_indices=state.rank_reference.scale_indices if state.rank_reference is not None else None,
            )
            new_rows = _append_library_rows(
                state.c_rows, delta_rows, old_r=old_r, variable_dim=state.variable_dim,
                packed=state.rank_reference is not None,
            )
            new_scales = torch.cat((old_scales.reshape(state.variable_dim, old_r),
                                    delta_scales.reshape(state.variable_dim, len(delta_terms))),
                                   dim=1).reshape(-1)
            new_gram = _primary_gram(new_rows, state.rank_reference)
    except (ValueError, RuntimeError, KeyError, NotImplementedError) as exc:
        check = StepCheck(
            term=normalized.canonical, accepted=False,
            reason=f"invalid_numeric_feature:{exc}", tree_length=normalized.complexity,
            D_prev=state.current_D, D_raw=state.current_D, D_t=state.current_D,
        )
        state.registry.add_rej(normalized)
        state.rejected.append(True)
        _append_step_reward(state, check, status="rejected", breakthrough_bonus=0.0)
        return check

    finite = torch.isfinite(new_gram).all()
    accepted = True
    reason = "accepted"
    rank_decision = None
    if not bool(finite):
        accepted = False
        reason = "non_finite_gram"

    if accepted:
        try:
            if state.rank_reference is not None:
                if cached is not None:
                    rank_decision = cached.decision
                else:
                    rank_decision = _validated_rank_from_rows(
                        new_rows, state.rank_reference, state.rank_config, primary_gram=new_gram,
                        function_rows=_function_rows_for_terms(
                            state.rank_reference, task=state.task, device=state.device,
                            theta_terms=trial_terms, scales=new_scales))
                    _remember_library_evaluation(state, trial_terms, new_rows, new_scales,
                                                 new_gram, rank_decision)
                if not rank_decision.resolved:
                    raise RankUnresolvedError(rank_decision)
                D_raw = int(rank_decision.dimension)
                kappa = rank_decision.kappa
                sigma_max, sigma_min_active = rank_decision.sigma_max, rank_decision.sigma_min_active
                threshold = rank_decision.threshold
                cut_confidence = 1.0
            else:
                S_t, _ = _svd_from_gram(new_gram)
                cut_t = estimate_spectral_cut(S_t, **_state_cut_kwargs(state, **cut_kwargs))
                kappa, sigma_max, sigma_min_active = _active_singular_condition_number(
                    S_t,
                    threshold=cut_t.threshold,
                    eps=health_config.eps, spectral_floor=state.spectral_floor,
                )
                D_raw = int(cut_t.null_dim)
                cut_confidence = float(cut_t.confidence)
                threshold = float(cut_t.threshold)
            D_t = D_raw
            if (
                state.rank_reference is None
                and
                health_config.min_cut_confidence is not None
                and D_raw > int(state.current_D)
                and cut_confidence < float(health_config.min_cut_confidence)
            ):
                accepted = False
                reason = "low_confidence_svd_cut"
        except RankUnresolvedError:
            accepted = False
            reason = "dimension_unresolved"
            kappa = float("inf")
            sigma_max = sigma_min_active = threshold = float("nan")
            D_raw = D_t = int(state.current_D)
            cut_confidence = 0.0
        except Exception as exc:
            accepted = False
            reason = f"cached_gram_svd_failed:{exc}"
            kappa = float("inf")
            sigma_max = float("nan")
            sigma_min_active = float("nan")
            D_raw = int(state.current_D)
            D_t = int(state.current_D)
            cut_confidence = 0.0
            threshold = float("nan")
    else:
        kappa = float("inf")
        sigma_max = float("nan")
        sigma_min_active = float("nan")
        D_raw = int(state.current_D)
        D_t = int(state.current_D)
        cut_confidence = 0.0
        threshold = float("nan")

    if (
        accepted
        and health_config.kappa_max is not None
        and float(kappa) > float(health_config.kappa_max)
    ):
        accepted = False
        reason = "ill_conditioned_svd_kappa"

    if not accepted:
        # D_raw describes the trial library; D_t describes the actual next state.
        D_t = int(state.current_D)

    breakthrough_bonus = 0.0
    delay_steps = state.h
    eta_discount = float(eta) ** int(delay_steps)
    reward = 0.0
    status = "rejected"
    if accepted:
        gain = int(D_t) - int(state.current_D)
        survival_reward = float(alpha) * math.exp(-float(lambda_len) * float(normalized.complexity))
        breakthrough_bonus = float(gain) * float(eta_discount)
        if gain > 0:
            status = "breakthrough"
        elif gain < 0:
            status = "dimension_decrease"
        else:
            status = "survival"
        reward = float(breakthrough_bonus + survival_reward)

    check = StepCheck(
        term=normalized.canonical,
        accepted=accepted,
        reason=reason,
        tree_length=int(normalized.complexity),
        reward=float(reward),
        D_prev=int(state.current_D),
        D_raw=int(D_raw),
        D_t=int(D_t),
        threshold=float(threshold),
        cut_confidence=float(cut_confidence),
        kappa=float(kappa),
        sigma_max=float(sigma_max),
        sigma_min_active=float(sigma_min_active),
        rank_residual=rank_decision.residual if rank_decision is not None else float("nan"),
        rank_scale_residual=rank_decision.scale_residual if rank_decision is not None else float("nan"),
        rank_stability=rank_decision.stability if rank_decision is not None else float("nan"),
        function_stability_S=(rank_decision.function_stability
                              if rank_decision is not None else float("nan")),
        rank_selection=rank_decision.metadata() if rank_decision is not None else {},
    )

    if accepted:
        state.registry.add_acc(normalized)
        state.theta_terms.append(normalized.canonical)
        state.c_rows = new_rows
        state.gram = new_gram
        state.feature_scales = new_scales
        state.last_kappa = float(kappa)
        state.rejected.append(False)
        state.current_D = int(D_t)
        state.D_best = max(int(state.D_best), int(D_t))
        state.threshold_best = float(threshold)
        state.rank_decision = rank_decision
        if rank_decision is not None:
            state.normalization_metadata["rank_selection"] = _rank_metadata(state.rank_reference, rank_decision)
        else:
            state.normalization_metadata["rank_selection"] = {
                **state.normalization_metadata.get("rank_selection", {}),
                "dimension": int(D_t), "threshold": float(threshold),
            }
    else:
        state.registry.add_rej(normalized)
        state.rejected.append(True)

    _append_step_reward(
        state,
        check,
        status=status,
        breakthrough_bonus=float(breakthrough_bonus),
        delay_steps=int(delay_steps),
        eta_discount=float(eta_discount),
    )
    if bool(health_config.verbose) or bool(state.verbose):
        print(
            f"step_cached_gram_svd term={normalized.canonical}, accepted={accepted}, reason={reason}, "
            f"D_raw={D_raw}, D_t={D_t}, threshold={threshold:.6g}, "
            f"confidence={cut_confidence:.6g}, kappa={kappa:.6g}, "
            f"R={check.rank_residual:.6g}, S={check.rank_stability:.6g}, "
            f"scale_R={check.rank_scale_residual:.6g}, "
            f"function_S={check.function_stability_S:.6g}, "
            f"delay={delay_steps}, eta_discount={eta_discount:.6g}, "
            f"reward={reward:.6g}"
        )
    return check


def compute_step_reward_result(
    *,
    state: EpisodeGramState,
    **cut_kwargs,
) -> Tuple[StepRewardResult, torch.Tensor, torch.Tensor]:
    """Finalize an episode by packaging rewards already computed in step()."""

    rewards = tuple(float(value) for value in state.step_rewards)
    action_rewards = tuple(state.step_action_rewards)

    final_terms = tuple(str(term) for term in state.theta_terms)
    metadata = dict(state.normalization_metadata)
    if state.rank_reference is not None:
        decision = state.rank_decision
        if decision is None:
            raise RuntimeError("Validated state is missing its cached rank decision.")
        if not decision.resolved:
            raise RankUnresolvedError(decision)
        final_S, final_Vh = decision.spectra[0], decision.vhs[0]
        threshold_best = float(decision.threshold)
        D_final = int(decision.dimension)
        final_W = decision.basis
        kappa = decision.kappa
        metadata["rank_selection"] = _rank_metadata(state.rank_reference, decision)
        basis_source = "validated_top_gap_first_subset_gram_float64"
    else:
        final_S, final_Vh = _svd_from_gram(state.gram)
        cut_final = estimate_spectral_cut(final_S, **_state_cut_kwargs(state, **cut_kwargs))
        threshold_best = float(cut_final.threshold)
        D_final = int(cut_final.null_dim)
        final_W = select_null_basis(final_S, final_Vh, threshold_best, spectral_floor=state.spectral_floor)
        kappa = (_active_singular_condition_number(
            final_S, threshold=threshold_best, spectral_floor=state.spectral_floor,
        )[0] if final_S.numel() else state.last_kappa)
        basis_source = "cached_gram_float64_svd"
        metadata["rank_selection"] = {
            **metadata.get("rank_selection", {}),
            "dimension": D_final, "threshold": threshold_best,
        }
    D_best = max(int(state.D_best), D_final)

    expected_cols = int(state.variable_dim) * len(final_terms)
    if final_W is None or final_W.numel() == 0:
        null_basis = torch.empty(0, expected_cols, device=state.device, dtype=torch.float64)
    else:
        null_basis = final_W

    result = StepRewardResult(
        rewards=rewards,
        action_rewards=action_rewards,
        D0=int(state.initial_D0),
        D_best=int(D_best),
        delta_D=max(0, int(D_best) - int(state.initial_D0)),
        threshold_initial=float(state.initial_threshold),
        threshold_best=float(threshold_best),
        basis_source=basis_source,
        theta_terms=final_terms,
        null_basis=null_basis.detach().cpu(),
        feature_scales=state.feature_scales.detach().cpu().clone(),
        D_final=int(D_final),
        kappa=kappa,
        spectral_floor=state.spectral_floor,
        normalization_metadata=metadata,
    )
    if bool(state.verbose):
        print(
            f"D0: {result.D0}\n"
            f"D_final: {D_final}\n"
            f"D_best: {result.D_best}\n"
            f"DeltaD: {result.delta_D}\n"
            f"threshold_best: {result.threshold_best:.6g}"
        )
    return result, final_S, final_Vh


def compute_step_reward_result_fast(
    *,
    state: EpisodeGramState,
) -> Tuple[StepRewardResult, torch.Tensor, torch.Tensor]:
    """Finalize training episodes without an extra final SVD.

    Step rewards and ``D_best`` have already been computed during accepted
    transitions. This fast path is sufficient for policy optimization; use the
    full finalizer when a final null basis is needed for visualization.
    """

    if state.rank_reference is not None:
        # The accepted decision already stores its spectra and basis. Reusing it
        # is both cheap and essential: a fresh single-gap cut would change d.
        return compute_step_reward_result(state=state)
    rewards = tuple(float(value) for value in state.step_rewards)
    action_rewards = tuple(state.step_action_rewards)
    final_terms = tuple(str(term) for term in state.theta_terms)
    expected_cols = int(state.variable_dim) * len(final_terms)
    null_basis = torch.empty(0, expected_cols, device=state.device, dtype=torch.float64)
    final_S = torch.empty(0, device=state.device, dtype=torch.float64)
    final_Vh = torch.empty(0, expected_cols, device=state.device, dtype=torch.float64)
    result = StepRewardResult(
        rewards=rewards,
        action_rewards=action_rewards,
        D0=int(state.initial_D0),
        D_best=int(state.D_best),
        delta_D=max(0, int(state.D_best) - int(state.initial_D0)),
        threshold_initial=float(state.initial_threshold),
        threshold_best=float(state.threshold_best),
        basis_source="step_cached_gram_no_final_svd",
        theta_terms=final_terms,
        null_basis=null_basis.detach().cpu(),
        feature_scales=state.feature_scales.detach().cpu().clone(),
        D_final=int(state.current_D),
        kappa=state.last_kappa,
        spectral_floor=state.spectral_floor,
        normalization_metadata=dict(state.normalization_metadata),
    )
    if bool(state.verbose):
        print(
            f"D0: {result.D0}\n"
            f"D_best: {result.D_best}\n"
            f"DeltaD: {result.delta_D}\n"
            f"threshold_best: {result.threshold_best:.6g}"
        )
    return result, final_S, final_Vh


def finalize_episode_with_svd(
    state: EpisodeGramState,
    **cut_kwargs,
) -> Tuple[StepRewardResult, torch.Tensor, torch.Tensor]:
    """Return the final discovery tensors and rewards accumulated during steps."""

    result, final_S, final_Vh = compute_step_reward_result(state=state, **cut_kwargs)
    if bool(state.verbose):
        print("final_S:", final_S)
    return result, final_S, final_Vh


def _default_generator_labels(task: str, variable_dim: int) -> Tuple[str, ...]:
    label_map = {
        "heat": (r"$\partial_t$", r"$\partial_x$", r"$\partial_u$"),
        "burger": (r"$\partial_t$", r"$\partial_x$", r"$\partial_u$"),
        "damped_burger": (r"$\partial_t$", r"$\partial_x$", r"$\partial_u$"),
        "forced_transport": (r"$\partial_t$", r"$\partial_x$", r"$\partial_u$"),
        "kdv": (r"$\partial_t$", r"$\partial_x$", r"$\partial_u$"),
        "wave": (r"$\partial_t$", r"$\partial_x$", r"$\partial_y$", r"$\partial_u$"),
        "rd": (r"$\partial_t$", r"$\partial_x$", r"$\partial_y$", r"$\partial_u$", r"$\partial_v$"),
        "schrodinger": (r"$\partial_t$", r"$\partial_x$", r"$\partial_y$", r"$\partial_u$", r"$\partial_v$"),
        "top": (
            r"$\partial_{p^0}$",
            r"$\partial_{p^1}$",
            r"$\partial_{p^2}$",
            r"$\partial_{p^3}$",
        ),
    }
    labels = label_map.get(task, ())
    if len(labels) >= variable_dim:
        return tuple(labels[:variable_dim])
    return tuple(rf"$\partial_{{z_{idx}}}$" for idx in range(variable_dim))


def vis_dynamic_generators(
    *,
    S: torch.Tensor,
    basis: torch.Tensor,
    task: str,
    sample: Union[int, str],
    theta_terms: Sequence[str],
    variable_labels: Optional[Sequence[str]] = None,
    save_vis_path: str = "./vis",
    feature_scales: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    normalization_metadata: Optional[dict] = None,
) -> torch.Tensor:
    """Visualize a dynamic sparse basis without assuming a fixed theta length."""

    import matplotlib.pyplot as plt
    import json
    from pathlib import Path

    theta_terms = tuple(str(term) for term in theta_terms)
    if basis.ndim != 2:
        raise ValueError(f"basis must be rank-2, got {tuple(basis.shape)}.")
    if basis.shape[1] % len(theta_terms) != 0:
        raise ValueError(
            f"Cannot reshape basis with {basis.shape[1]} columns and "
            f"{len(theta_terms)} theta terms."
        )

    variable_dim = basis.shape[1] // len(theta_terms)
    labels = (
        tuple(variable_labels)
        if variable_labels is not None
        else _default_generator_labels(task, variable_dim)
    )

    out_dir = Path(save_vis_path) / f"{task}_{sample}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_png in out_dir.glob("generator_*.png"):
        old_png.unlink()
    for old_name in ("singular_value.png", "generator.pt", "theta_terms.txt",
                     "feature_scales.pt", "column_scales.pt", "normalization.json"):
        old_file = out_dir / old_name
        if old_file.exists():
            old_file.unlink()

    basis_cpu = basis.detach().cpu()
    W = basis_cpu.reshape(basis_cpu.shape[0], variable_dim, len(theta_terms))
    torch.save(W, out_dir / "generator.pt")
    (out_dir / "theta_terms.txt").write_text(",".join(theta_terms), encoding="utf-8")

    metadata = dict(normalization_metadata or {})
    metadata["task"] = task
    validated_rank = metadata.get("rank_selection", {}).get("method") == "validated_top_gap"
    if feature_scales is not None:
        supplied_scales = torch.as_tensor(feature_scales, dtype=torch.float64, device="cpu")
        column_scales = validate_coefficient_scales(
            supplied_scales, len(theta_terms), variable_dim, "cpu",
        )
        is_column_scales = (
            supplied_scales.numel() != len(theta_terms)
            or metadata.get("normalization_version") == 2
            or metadata.get("method") == "prolonged_column_rms"
        )
        scales = column_scales if is_column_scales else supplied_scales
        torch.save(scales, out_dir / "feature_scales.pt")
        if is_column_scales:
            torch.save(column_scales, out_dir / "column_scales.pt")
            metadata.update({
                "normalization_version": 2, "method": "prolonged_column_rms",
                "scale_layout": "variable_blocks",
                "scale_source": "prolonged_features_before_jacobian",
            })
        else:
            metadata.update({
                "normalization_version": 1, "method": "rms", "scale_layout": "terms",
                "scale_source": "feature_values",
            })
        metadata.update({
            "coefficient_coordinates": "original",
            "theta_terms": list(theta_terms),
            "threshold_kind": "display_only" if validated_rank else "relative",
            "spectrum": "normalized_system",
            "selected_threshold": (None if validated_rank else threshold),
        })
        if validated_rank:
            metadata["selected_dimension"] = metadata["rank_selection"]["dimension"]
        (out_dir / "normalization.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8",
        )
    S_cpu = relative_singular_values(
        S.detach().cpu(), floor=metadata.get("spectral_floor", DEFAULT_SPECTRAL_FLOOR),
    )
    n = basis_cpu.shape[0]
    start = max(0, len(S_cpu) - n - 1)
    x_idx = list(range(start, len(S_cpu)))
    plt.plot(x_idx, S_cpu.numpy()[start:])
    plt.xticks(x_idx)
    plt.xlabel("Index")
    if threshold is not None and math.isfinite(float(threshold)) and not validated_rank:
        plt.axhline(float(threshold), color="tab:red", linestyle="--", label="Cutoff")
        plt.legend()
    plt.ylabel("Relative singular value")
    plt.title("Normalized symmetry system")
    plt.savefig(out_dir / "singular_value.png", bbox_inches="tight")
    plt.clf()

    for i in range(n):
        plt.figure(figsize=(max(6.0, 0.55 * len(theta_terms)), 3.2))
        plt.imshow(W[i].numpy(), aspect="auto")
        plt.colorbar(shrink=0.5)
        plt.xticks(range(len(theta_terms)), theta_terms, rotation=45, ha="right")
        plt.yticks(range(variable_dim), labels)
        plt.tick_params(axis="both", which="both", length=0)
        plt.tight_layout()
        plt.savefig(out_dir / f"generator_{len(S_cpu) - n + i + 1}.png", bbox_inches="tight")
        plt.close()
    return W


def sparsify_basis_for_visualization(
    basis: torch.Tensor,
    *,
    device,
    epsilon1: float = 1e-4,
    epsilon2: float = 1e-4,
    log_interval: int = 500,
    max_iter: int = 500,
    verbose: bool = False,
) -> torch.Tensor:
    """Apply utils.basis_sparsification before plotting generators."""

    from utils import basis_sparsification

    if basis is None or basis.numel() == 0:
        return basis
    sparse = basis_sparsification(
        Q=basis.to(device=device).T,
        epsilon1=float(epsilon1),
        epsilon2=float(epsilon2),
        device=device,
        log_interval=int(log_interval),
        max_iter=max_iter,
        verbose=bool(verbose),
    )
    return sparse.T.detach().cpu()
