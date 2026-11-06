"""Resample a saved RS-AC policy, prune redundant Theta terms, and visualize.

Examples:

    python pruning_and_visualize.py burger --batch-size 100
    python pruning_and_visualize.py burger --reuse-generator
    python pruning_and_visualize.py wave --checkpoint ./checkpoints/rs_ac/wave/policy_last.pt

The pruning stage implements Post-RL support-based pruning:

1. Pick the best sampled final episode.
2. Use its final Theta as a reference library.
3. Remove exact library identities, map the coefficients, and revalidate their
   independent span. Then sparsify and compute each theta term's support score.
4. Try deleting non-mandatory terms from low support to high support.
5. Keep a deletion only when nullity, conditioning, and the generator subspace
   satisfy their tolerances against the fixed reference library.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from lie_discover import EQUATION_ALIASES, build_equation_argv, normalize_equation_name
from feature_normalization import (
    DEFAULT_SUBSPACE_TOLERANCE, projector_subspace_distance, restored_subspace_basis,
    restore_coefficient_basis, restrict_library_scales,
)
from library_identities import prepare_generator_library, retain_identity_metadata
from reward import sparsify_basis_for_visualization, vis_dynamic_generators
from spectral import RankUnresolvedError
from rollout import collect_rollouts
from sampling_utils import (
    build_seeded_permutation_loader, describe_sample_window, merge_search_samples,
)
from wsindy_source import resolve_wsindy_json
from train_rs_ac import (
    _episode_total_return,
    _resolve_task_defaults,
    select_final_episode,
    build_arg_parser as build_train_arg_parser,
    build_env,
    build_policy,
    build_train_dataset,
    device_from_args,
    load_search_model,
    set_seed,
)


DEFAULT_REUSE_GENERATORS: Dict[str, Dict[str, object]] = {}


@dataclass(frozen=True)
class ThetaEvaluation:
    theta_terms: Tuple[str, ...]
    result: object
    final_S: torch.Tensor
    final_Vh: torch.Tensor
    kappa: float
    threshold: float
    D_final: int
    feature_scales: torch.Tensor


@dataclass(frozen=True)
class PruneTrial:
    term: str
    accepted: bool
    reason: str
    theta_terms: Tuple[str, ...]
    D_best: int
    D_final: Optional[int]
    kappa: float
    threshold: float
    support_score: float
    subspace_error: float = float("nan")


@dataclass(frozen=True)
class PruneResult:
    reference: ThetaEvaluation
    pruned: ThetaEvaluation
    support_scores: Dict[str, float]
    trials: Tuple[PruneTrial, ...]
    subspace_error: float = 0.0
    subspace_tolerance: float = DEFAULT_SUBSPACE_TOLERANCE


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "equation",
        nargs="?",
        default="wave",
        choices=tuple(sorted(EQUATION_ALIASES)),
        help="Equation/task name. Default: wave.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Saved policy checkpoint. Defaults to checkpoints/rs_ac/<task>_<seed>/policy_last.pt, with legacy task-directory fallback.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of final policy episodes to sample before selecting the best one.",
    )
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--rank-selection", choices=("validated", "gap"), default=None,
                        help="Override the saved dimension rule; PDE defaults use validation and stability.")
    parser.add_argument("--rank-function-stability-tolerance", type=float, default=None,
                        help="Override the saved generator-function subspace stability tolerance.")
    parser.add_argument("--jacobian-source", choices=("wsindy",), default=None)
    parser.add_argument("--wsindy-json", default=None,
                        help="Explicitly re-evaluate the library with this WSINDy equation JSON.")
    parser.add_argument("--wsindy-run-dir", default=None)
    parser.add_argument("--wsindy-seed", type=int, default=None,
                        help="Equation fitting seed; independent of the search/sampling seed.")
    parser.add_argument("--use-threshold-bounds", action=argparse.BooleanOptionalAction, default=None,
                        help="Override the checkpoint's spectral mode; default for old checkpoints is unbounded.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--vis-sample", default=None)
    parser.add_argument("--vis-dir", default=None)
    parser.add_argument("--sparsify-max-iter", type=int, default=None)
    parser.add_argument("--sparsify-verbose", action="store_true")
    parser.add_argument("--print-best-details", action="store_true")

    parser.add_argument(
        "--reuse-generator",
        action="store_true",
        help="Reuse a saved generator's library; recompute normalized support on the search samples.",
    )
    parser.add_argument(
        "--generator-path",
        default=None,
        help=(
            "Saved generator.pt used with --reuse-generator. Directories are "
            "accepted. Defaults to vis/<task>_<seed>/generator.pt."
        ),
    )
    parser.add_argument(
        "--theta-terms",
        default=None,
        help=(
            "Comma-separated theta terms. With --generator-path, these must match "
            "the saved generator. Without --generator-path, the terms are evaluated "
            "directly as a fixed Theta library."
        ),
    )
    parser.add_argument("--no-prune", action="store_true")
    parser.add_argument(
        "--mandatory-terms",
        default=None,
        help="Comma-separated terms that pruning must keep. Defaults to theta0.",
    )
    parser.add_argument(
        "--prune-condition-tolerance",
        type=float,
        default=2.0,
        help="Allow trial kappa <= reference kappa * this tolerance.",
    )
    parser.add_argument(
        "--prune-subspace-tolerance",
        type=float,
        default=DEFAULT_SUBSPACE_TOLERANCE,
        help="Maximum original-coefficient projector distance from the fixed reference space (0 < value < 1).",
    )
    parser.add_argument(
        "--prune-max-kappa",
        type=float,
        default=None,
        help="Optional absolute kappa cap for accepted pruned libraries.",
    )
    parser.add_argument(
        "--prune-ignore-d-change-below-support",
        type=float,
        default=None,
        help=(
            "Deprecated and rejected: pruning must preserve the actual null-space dimension."
        ),
    )
    return parser


def _split_terms(text: Optional[Sequence[str] | str]) -> Optional[Tuple[str, ...]]:
    if text is None:
        return None
    if isinstance(text, str):
        stripped = text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        values = tuple(part.strip() for part in stripped.split(",") if part.strip())
    else:
        values = tuple(str(part).strip() for part in text if str(part).strip())
    return values or None


def _load_theta_terms_sidecar(generator_path: Path) -> Optional[Tuple[str, ...]]:
    """Load theta terms from theta_terms.txt next to a generator file/directory."""

    path = Path(generator_path)
    candidates = []
    if path.is_dir():
        candidates.append(path / "theta_terms.txt")
    else:
        candidates.append(path.with_name("theta_terms.txt"))
        candidates.append(path.parent / "theta_terms.txt")
    for candidate in candidates:
        if candidate.exists():
            return _split_terms(candidate.read_text(encoding="utf-8"))
    return None


def _resolve_generator_path(path_text: Optional[str], *, task: str, seed: int) -> Path:
    """Resolve a saved generator path, accepting directories as shorthand."""

    if path_text is None:
        path = Path("vis") / f"{task}_{seed}" / "generator.pt"
    else:
        path = Path(path_text)
        if path.is_dir():
            path = path / "generator.pt"
    return path


def _load_checkpoint(path: Path, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _default_checkpoint_path(task: str, seed: int) -> Path:
    seeded = Path(f"./checkpoints/rs_ac/{task}_{seed}/policy_last.pt")
    legacy = Path(f"./checkpoints/rs_ac/{task}/policy_last.pt")
    return legacy if not seeded.exists() and legacy.exists() else seeded


def _load_tensor(path: Path, device: Optional[torch.device] = None):
    try:
        tensor = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location=device)
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected {path} to contain a tensor, got {type(tensor)!r}.")
    return tensor


_SOURCE_KEYS = {"jacobian_source", "wsindy_json", "wsindy_run_dir", "wsindy_seed", "wsindy_sha256"}


def _saved_jacobian_source(record):
    """Read a policy or generator record without inferring a network default."""
    if not isinstance(record, dict):
        return None
    metadata = record.get("jacobian_source")
    if isinstance(metadata, dict):
        return dict(metadata)
    saved = record.get("args", record)
    if not isinstance(saved, dict):
        return None
    source = saved.get("jacobian_source")
    if source in ("wsindy", "surrogate"):
        keys = _SOURCE_KEYS | {"task", "surrogate_checkpoint"}
        return {key: saved[key] for key in keys if key in saved}
    return None


def _apply_jacobian_source(args, cli_args, *, saved_source=None):
    """Restore a WSINDy source; explicit equation routing starts a new evaluation."""
    cli_source = getattr(cli_args, "jacobian_source", None)
    ws_overrides = {key: getattr(cli_args, key, None)
                    for key in ("wsindy_json", "wsindy_run_dir", "wsindy_seed")}
    ws_overrides = {key: value for key, value in ws_overrides.items() if value is not None}
    if cli_source not in (None, "wsindy"):
        raise ValueError("Only the saved WSINDy equation is supported as a Jacobian source.")
    if saved_source and saved_source.get("task", args.task) != args.task:
        raise ValueError("Saved Jacobian source belongs to a different equation/task.")
    old_source = (saved_source or {}).get("jacobian_source")
    if old_source not in (None, "wsindy") and not ws_overrides:
        raise ValueError(
            f"Saved Jacobian source {old_source!r} is no longer supported. "
            "Select --wsindy-json or --wsindy-run-dir explicitly to re-evaluate the saved library."
        )
    args.jacobian_source = "wsindy"
    args.wsindy_sha256 = None
    # Obsolete checkpoint fields do not select or load a neural network.
    vars(args).pop("surrogate_checkpoint", None)
    if saved_source and old_source == "wsindy":
        for key in _SOURCE_KEYS - {"jacobian_source"}:
            if saved_source.get(key) is not None:
                setattr(args, key, saved_source[key])
    if "wsindy_run_dir" in ws_overrides or "wsindy_seed" in ws_overrides:
        args.wsindy_json = None
    for key, value in ws_overrides.items():
        setattr(args, key, value)
    if ws_overrides:
        selected_path = resolve_wsindy_json(
            args.task, wsindy_json=getattr(args, "wsindy_json", None),
            wsindy_run_dir=getattr(args, "wsindy_run_dir", None),
            wsindy_seed=getattr(args, "wsindy_seed", 0),
        )
        previous_path = (saved_source or {}).get("wsindy_json")
        if previous_path is None or selected_path != Path(previous_path).expanduser().resolve():
            args.wsindy_sha256 = None
    args._source_jacobian_metadata = saved_source


def _jacobian_source_lineage(previous, current, *, has_source_artifact):
    """Keep source changes visible even when only an old library is reused."""
    if not has_source_artifact:
        return None
    same = False
    if previous and previous.get("jacobian_source") == current.get("jacobian_source") == "wsindy":
        same = bool(previous.get("wsindy_sha256")) and previous["wsindy_sha256"] == current.get("wsindy_sha256")
    return {"previous": previous, "current": dict(current), "source_match_verified": same,
            "re_evaluated": not same,
            "reason": "same_source" if same else ("source_changed" if previous else "previous_source_unrecorded")}


def _restore_generator_jacobian_source(args, cli_args, metadata):
    policy_source = args._source_jacobian_metadata
    generator_source = _saved_jacobian_source(metadata)
    args.jacobian_source_reference = "generator" if generator_source else "policy_checkpoint"
    if policy_source and generator_source:
        old_kind, new_kind = (source.get("jacobian_source") for source in (policy_source, generator_source))
        conflict = old_kind != new_kind
        if old_kind == new_kind == "wsindy" and all(source.get("wsindy_sha256") for source in (policy_source, generator_source)):
            conflict = policy_source["wsindy_sha256"] != generator_source["wsindy_sha256"]
        elif old_kind == new_kind:
            path_key = "wsindy_json" if old_kind == "wsindy" else "surrogate_checkpoint"
            old_path, new_path = policy_source.get(path_key), generator_source.get(path_key)
            conflict = bool(old_path and new_path) and Path(old_path).expanduser().resolve() != Path(new_path).expanduser().resolve()
        if conflict:
            # Only an actual source/path change can opt into re-evaluation.
            # Repeating the same JSON path must not bypass a conflicting hash.
            selected = argparse.Namespace(**vars(args))
            _apply_jacobian_source(selected, cli_args, saved_source=generator_source)
            explicit = any(getattr(cli_args, key, None) is not None
                           for key in ("jacobian_source", "wsindy_json", "wsindy_run_dir", "wsindy_seed"))
            changed = any(selected.jacobian_source != source.get("jacobian_source")
                          for source in (policy_source, generator_source))
            if selected.jacobian_source == "wsindy":
                path = resolve_wsindy_json(selected.task, wsindy_json=selected.wsindy_json,
                                           wsindy_run_dir=selected.wsindy_run_dir, wsindy_seed=selected.wsindy_seed)
                changed = changed or any(source.get("wsindy_json") is not None
                                         and path != Path(source["wsindy_json"]).expanduser().resolve()
                                         for source in (policy_source, generator_source))
            if not (explicit and changed):
                raise ValueError("Policy checkpoint and generator record different Jacobian sources. "
                                 "Select a different --wsindy-json or --wsindy-run-dir for explicit re-evaluation.")
    _apply_jacobian_source(args, cli_args, saved_source=generator_source or policy_source)


def _train_args_from_checkpoint(
    *,
    equation: str,
    checkpoint_state,
    cli_args: argparse.Namespace,
) -> argparse.Namespace:
    """Rebuild the original training Namespace, then apply overrides."""

    train_parser = build_train_arg_parser()
    train_args = train_parser.parse_args(build_equation_argv(equation))
    saved_args = checkpoint_state.get("args", {}) if isinstance(checkpoint_state, dict) else {}
    normalization_version = (
        checkpoint_state.get("normalization_version", saved_args.get("normalization_version"))
        if isinstance(checkpoint_state, dict)
        else None
    )
    skipped_thresholds = []
    for key, value in saved_args.items():
        if normalization_version not in (1, 2) and key in {"spectral_threshold_min", "spectral_threshold_max"}:
            skipped_thresholds.append(key)
            continue
        if key == "normalization_version":
            continue
        if key in _SOURCE_KEYS:
            continue
        if hasattr(train_args, key):
            setattr(train_args, key, value)
    if skipped_thresholds:
        print(
            "Migrating legacy checkpoint: replacing absolute SVD bounds with "
            "the current normalized relative threshold defaults.",
            flush=True,
        )

    train_args.task = normalize_equation_name(equation)
    if getattr(cli_args, "rank_selection", None) is not None:
        train_args.rank_selection = cli_args.rank_selection
    if getattr(cli_args, "rank_function_stability_tolerance", None) is not None:
        train_args.rank_function_stability_tolerance = cli_args.rank_function_stability_tolerance
    _apply_jacobian_source(train_args, cli_args, saved_source=_saved_jacobian_source(checkpoint_state))
    if cli_args.use_threshold_bounds is not None:
        train_args.use_threshold_bounds = cli_args.use_threshold_bounds
    if not train_args.use_threshold_bounds:
        train_args.spectral_threshold_min = None
        train_args.spectral_threshold_max = None
    if getattr(cli_args, "data_dir", None) is not None:
        train_args.data_dir = str(cli_args.data_dir)
    if cli_args.gpu is not None:
        train_args.gpu = int(cli_args.gpu)
    if cli_args.seed is not None:
        train_args.seed = int(cli_args.seed)
    if cli_args.sample is not None:
        train_args.sample = int(cli_args.sample)
    if cli_args.sample_size is not None:
        train_args.sample_size = int(cli_args.sample_size)
    if cli_args.temperature is not None:
        train_args.final_vis_temperature = float(cli_args.temperature)

    if cli_args.vis_sample is not None:
        train_args.final_vis_sample = str(cli_args.vis_sample)
    else:
        base_sample = str(train_args.seed)
        suffix = "" if bool(cli_args.no_prune) else "_pruned"
        train_args.final_vis_sample = f"{base_sample}{suffix}"

    if cli_args.vis_dir is not None:
        train_args.final_vis_dir = str(cli_args.vis_dir)
    if cli_args.sparsify_max_iter is not None:
        train_args.sparsify_max_iter = int(cli_args.sparsify_max_iter)
    if bool(cli_args.sparsify_verbose):
        train_args.sparsify_verbose = True

    train_args.visualize_final = True
    train_args.final_vis_batch_size = int(cli_args.batch_size)
    train_args.final_vis_greedy = bool(cli_args.greedy)
    train_args.fast_finalize = False
    train_args.print_best_episode_details = bool(cli_args.print_best_details)
    return _resolve_task_defaults(train_args)


def _support_scores(
    sparse_basis: torch.Tensor,
    *,
    theta_terms: Sequence[str],
    variable_dim: int,
) -> Dict[str, float]:
    sparse_basis = sparse_basis.detach().cpu().to(torch.float64)
    if sparse_basis.ndim == 3:
        W = sparse_basis
        if W.shape[1] != int(variable_dim) or W.shape[2] != len(theta_terms):
            raise ValueError(
                f"Saved generator shape {tuple(W.shape)} is incompatible with "
                f"variable_dim={int(variable_dim)} and {len(theta_terms)} theta terms."
            )
    elif sparse_basis.ndim == 2:
        W = sparse_basis.reshape(sparse_basis.shape[0], int(variable_dim), len(theta_terms))
    else:
        raise ValueError(
            f"support basis must be rank-2 sparse rows or rank-3 generator tensor, "
            f"got shape {tuple(sparse_basis.shape)}."
        )
    scores = torch.sqrt(torch.sum(W**2, dim=(0, 1)))
    return {str(term): float(scores[idx].item()) for idx, term in enumerate(theta_terms)}


def evaluate_theta_terms(
    env,
    theta_terms: Sequence[str],
    *,
    feature_scales: Optional[torch.Tensor] = None,
    reference_state=None,
) -> ThetaEvaluation:
    final = env.evaluate_library(
        tuple(theta_terms),
        feature_scales=feature_scales,
        reference_state=reference_state,
    )
    threshold = float(final.result.threshold_best)
    return ThetaEvaluation(
        theta_terms=tuple(str(term) for term in final.result.theta_terms),
        result=final.result,
        final_S=final.final_S,
        final_Vh=final.final_Vh,
        kappa=float(final.result.kappa),
        threshold=threshold,
        D_final=int(final.result.D_final),
        feature_scales=final.result.feature_scales.detach().cpu().to(torch.float64).clone(),
    )


def _rank_summary(evaluation: ThetaEvaluation) -> str:
    rank = getattr(evaluation.result, "normalization_metadata", {}).get("rank_selection", {})
    if rank.get("method") == "validated_top_gap":
        return (f"rank_rule=validated, validation_R={rank.get('residual')}, "
                f"stability_S={rank.get('stability')}, scale_R={rank.get('scale_residual')}, "
                f"function_S={rank.get('function_stability')}, rank_resolved={rank.get('resolved')}, "
                f"threshold_role=display_only, ")
    return ""


def prepare_theta_evaluation(env, evaluation, mandatory_terms=(), *, reference_state=None,
                             subspace_tolerance=DEFAULT_SUBSPACE_TOLERANCE):
    prepared = prepare_generator_library(
        env=env, result=evaluation.result, final_S=evaluation.final_S,
        final_Vh=evaluation.final_Vh, mandatory_terms=mandatory_terms,
        reference_state=reference_state, subspace_tolerance=subspace_tolerance,
    )
    result = prepared.result
    if result is evaluation.result:
        return evaluation, prepared.reference_state, prepared.mandatory_terms
    evaluation = replace(
        evaluation, theta_terms=tuple(result.theta_terms), result=result,
        final_S=prepared.final_S, final_Vh=prepared.final_Vh, kappa=float(result.kappa),
        threshold=float(result.threshold_best), D_final=int(result.D_final),
        feature_scales=result.feature_scales.detach().cpu().to(torch.float64).clone(),
    )
    return evaluation, prepared.reference_state, prepared.mandatory_terms


def post_rl_backward_prune(
    *,
    env,
    theta_terms: Sequence[str],
    mandatory_terms: Sequence[str],
    device: torch.device,
    sparsify_epsilon1: float,
    sparsify_epsilon2: float,
    sparsify_log_interval: int,
    sparsify_max_iter: Optional[int],
    sparsify_verbose: bool,
    support_basis: Optional[torch.Tensor] = None,
    condition_tolerance: float = 2.0,
    subspace_tolerance: float = DEFAULT_SUBSPACE_TOLERANCE,
    max_kappa: Optional[float] = None,
    ignore_d_change_below_support: Optional[float] = None,
) -> PruneResult:
    """Prune against fixed reference nullity, conditioning, and generator space."""

    if ignore_d_change_below_support is not None:
        raise ValueError("Pruning must preserve the actual null-space dimension; the support escape is no longer supported.")
    if not math.isfinite(float(condition_tolerance)) or float(condition_tolerance) < 1.0:
        raise ValueError("condition_tolerance must be finite and at least 1.")
    subspace_tolerance = float(subspace_tolerance)
    if not math.isfinite(subspace_tolerance) or not 0 < subspace_tolerance < 1:
        raise ValueError("subspace_tolerance must be finite and strictly between 0 and 1.")
    if max_kappa is not None and (not math.isfinite(float(max_kappa)) or float(max_kappa) < 1.0):
        raise ValueError("max_kappa must be finite and at least 1.")

    theta_terms = tuple(str(term) for term in theta_terms)
    reference = evaluate_theta_terms(env, theta_terms)
    reference_state = env.state
    if support_basis is not None:
        # Validate the saved layout before a symbolic change of coordinates.
        # Its amplitudes and old numerical null space are not used for scoring.
        _support_scores(support_basis, theta_terms=reference.theta_terms,
                        variable_dim=reference.result.null_basis.shape[1] // len(reference.theta_terms))
    reference, reference_state, mandatory_terms = prepare_theta_evaluation(
        env, reference, mandatory_terms, reference_state=reference_state,
        subspace_tolerance=subspace_tolerance,
    )
    theta_terms = reference.theta_terms
    if reference.result.null_basis.shape[0] != reference.D_final:
        raise ValueError("Reference basis size does not match its actual nullity.")
    reference_basis = restored_subspace_basis(
        reference.result.null_basis, reference.feature_scales, theta_terms, theta_terms,
    )
    registry = getattr(reference_state, "registry", None)

    def term_key(term):
        return registry.normalize(term).key if registry is not None else str(term)

    mandatory = {term_key(term) for term in mandatory_terms}
    variable_dim = int(reference.result.null_basis.shape[1] // len(theta_terms))
    support_source = "normalized_reference_sparsification"
    support_basis = sparsify_basis_for_visualization(
        reference.result.null_basis,
        device=device,
        epsilon1=sparsify_epsilon1,
        epsilon2=sparsify_epsilon2,
        log_interval=sparsify_log_interval,
        max_iter=sparsify_max_iter,
        verbose=bool(sparsify_verbose),
    )
    support_scores = _support_scores(
        support_basis,
        theta_terms=theta_terms,
        variable_dim=variable_dim,
    )

    print("=== Post-RL Support Pruning ===", flush=True)
    print(
        f"reference: D_final={reference.D_final}, D_best={reference.result.D_best}, "
        f"DeltaD={reference.result.delta_D}, kappa={reference.kappa:.6g}, "
        f"threshold_best={reference.threshold:.6g}, "
        + _rank_summary(reference) +
        f"subspace_tolerance={subspace_tolerance:.6g}, "
        f"support_source={support_source}, theta={','.join(theta_terms)}",
        flush=True,
    )
    print("support_scores:", flush=True)
    for term in theta_terms:
        print(f"  {term}: {support_scores[term]:.6g}", flush=True)

    current_terms: List[str] = list(theta_terms)
    trials: List[PruneTrial] = []
    reference_D = reference.D_final
    reference_kappa = float(reference.kappa)

    original_pos = {term: idx for idx, term in enumerate(theta_terms)}

    def retained_scales(terms: Sequence[str]) -> torch.Tensor:
        positions = torch.tensor([original_pos[term] for term in terms], dtype=torch.long)
        return restrict_library_scales(
            reference.feature_scales, positions,
            num_terms=len(theta_terms), variable_dim=variable_dim,
        )

    prune_order = sorted(
        [term for term in theta_terms if term_key(term) not in mandatory],
        key=lambda term: (support_scores.get(term, float("inf")), -original_pos[term]),
    )
    print(
        "prune_order_by_support="
        + ",".join(f"{term}:{support_scores.get(term, float('inf')):.6g}" for term in prune_order),
        flush=True,
    )

    for term in prune_order:
        if term not in current_terms:
            continue

        trial_terms = tuple(candidate for candidate in current_terms if candidate != term)
        if not trial_terms:
            print(f"try_remove={term}, accepted=False, reason=empty_library", flush=True)
            continue
        support_score = float(support_scores.get(term, float("inf")))
        try:
            trial_eval = evaluate_theta_terms(
                env,
                trial_terms,
                feature_scales=retained_scales(trial_terms),
                reference_state=reference_state,
            )
        except RankUnresolvedError as exc:
            # No validated dimension is different from an empty null space.
            # Keep the accepted library and continue trying other deletions.
            reason = f"dimension_unresolved:{exc}"
            trials.append(PruneTrial(
                term=term, accepted=False, reason=reason,
                theta_terms=tuple(current_terms), D_best=int(reference.result.D_best),
                D_final=None, kappa=float("inf"), threshold=float("nan"),
                support_score=support_score,
            ))
            print(f"try_remove={term}, accepted=False, reason={reason}, "
                  f"theta={','.join(current_terms)}", flush=True)
            continue

        accepted = True
        reason = "accepted"
        subspace_error = float("nan")
        if trial_eval.D_final != reference_D:
            accepted = False
            reason = f"D_changed:{trial_eval.D_final}!={reference_D}"
        elif not math.isfinite(trial_eval.kappa):
            accepted = False
            reason = "non_finite_trial_kappa"
        elif not math.isfinite(reference_kappa):
            accepted = False
            reason = "non_finite_reference_kappa"
        elif max_kappa is not None and float(trial_eval.kappa) > float(max_kappa):
            accepted = False
            reason = f"kappa_above_max:{trial_eval.kappa:.6g}>{float(max_kappa):.6g}"
        elif (
            float(trial_eval.kappa) > float(reference_kappa) * float(condition_tolerance)
        ):
            accepted = False
            reason = (
                f"kappa_worse:{trial_eval.kappa:.6g}>"
                f"{float(reference_kappa) * float(condition_tolerance):.6g}"
            )

        if accepted:
            try:
                if trial_eval.result.null_basis.shape[0] != trial_eval.D_final:
                    raise ValueError("Trial basis size does not match its actual nullity.")
                trial_basis = restored_subspace_basis(
                    trial_eval.result.null_basis, trial_eval.feature_scales,
                    trial_terms, theta_terms,
                )
                subspace_error = projector_subspace_distance(reference_basis, trial_basis)
                if subspace_error > subspace_tolerance:
                    accepted = False
                    reason = f"subspace_changed:{subspace_error:.6g}>{subspace_tolerance:.6g}"
            except (ValueError, RuntimeError) as exc:
                accepted = False
                reason = f"invalid_subspace_basis:{exc}"

        if accepted:
            current_terms = list(trial_terms)

        trials.append(
            PruneTrial(
                term=term,
                accepted=bool(accepted),
                reason=reason,
                theta_terms=tuple(trial_terms if accepted else current_terms),
                D_best=int(trial_eval.result.D_best),
                D_final=trial_eval.D_final,
                kappa=float(trial_eval.kappa),
                threshold=float(trial_eval.threshold),
                support_score=support_score,
                subspace_error=subspace_error,
            )
        )
        print(
            f"try_remove={term}, accepted={accepted}, reason={reason}, "
            f"D_final={trial_eval.D_final}, D_best={trial_eval.result.D_best}, kappa={trial_eval.kappa:.6g}, "
            f"threshold_best={trial_eval.threshold:.6g}, "
            + _rank_summary(trial_eval) +
            f"subspace_error={subspace_error:.6g}, "
            f"support={support_score:.6g}, "
            f"theta={','.join(current_terms)}",
            flush=True,
        )

    pruned = evaluate_theta_terms(
        env,
        tuple(current_terms),
        feature_scales=retained_scales(current_terms),
        reference_state=reference_state,
    )
    pruned = replace(pruned, result=retain_identity_metadata(reference.result, pruned.result))
    if pruned.D_final != reference_D or pruned.result.null_basis.shape[0] != reference_D:
        raise RuntimeError("Final recomputation changed the retained nullity.")
    if not math.isfinite(reference_kappa) or not math.isfinite(pruned.kappa):
        raise RuntimeError("Final recomputation requires finite reference and final condition numbers.")
    if pruned.kappa > reference_kappa * float(condition_tolerance):
        raise RuntimeError("Final recomputation exceeded the pruning conditioning tolerance.")
    if max_kappa is not None and pruned.kappa > float(max_kappa):
        raise RuntimeError("Final recomputation exceeded the maximum condition number.")
    final_basis = restored_subspace_basis(
        pruned.result.null_basis, pruned.feature_scales, pruned.theta_terms, theta_terms,
    )
    final_error = projector_subspace_distance(reference_basis, final_basis)
    if final_error > subspace_tolerance:
        raise RuntimeError("Final recomputation exceeded the pruning subspace tolerance.")
    print(
        f"pruned: D_final={pruned.D_final}, D_best={pruned.result.D_best}, DeltaD={pruned.result.delta_D}, "
        f"kappa={pruned.kappa:.6g}, threshold_best={pruned.threshold:.6g}, "
        + _rank_summary(pruned) +
        f"subspace_error={final_error:.6g}, "
        f"theta={','.join(pruned.theta_terms)}",
        flush=True,
    )
    return PruneResult(
        reference=reference,
        pruned=pruned,
        support_scores=support_scores,
        trials=tuple(trials),
        subspace_error=final_error,
        subspace_tolerance=subspace_tolerance,
    )


def visualize_theta_evaluation(
    *,
    evaluation: ThetaEvaluation,
    args: argparse.Namespace,
    device: torch.device,
    pruning_metadata: Optional[Dict[str, object]] = None,
    source_search_samples=None,
    inherit_search_samples: bool = False,
    candidate_selection: Optional[Dict[str, object]] = None,
) -> None:
    print(
        f"evaluation: D0={evaluation.result.D0}, "
        f"D_best={evaluation.result.D_best}, "
        f"DeltaD={evaluation.result.delta_D}, "
        f"threshold_best={evaluation.threshold:.6g}, "
        f"kappa={evaluation.kappa:.6g}, "
        f"null_basis_shape={tuple(evaluation.result.null_basis.shape)}, "
        f"theta={','.join(evaluation.theta_terms)}",
        flush=True,
    )
    sparse_basis = sparsify_basis_for_visualization(
        evaluation.result.null_basis,
        device=device,
        epsilon1=args.sparsify_epsilon1,
        epsilon2=args.sparsify_epsilon2,
        log_interval=args.sparsify_log_interval,
        max_iter=args.sparsify_max_iter,
        verbose=bool(args.sparsify_verbose),
    )
    original_basis = restore_coefficient_basis(sparse_basis, evaluation.feature_scales)
    metadata = dict(evaluation.result.normalization_metadata)
    metadata["run_seed"] = int(args.seed) if hasattr(args, "seed") else None
    source_lineage = getattr(args, "jacobian_source_lineage", None)
    if isinstance(source_lineage, dict):
        metadata["jacobian_source_lineage"] = dict(source_lineage)
    if inherit_search_samples:
        merged = merge_search_samples(metadata.get("search_samples"), source_search_samples)
        metadata["search_samples"] = merged
        if merged is not None:
            metadata["sample_indices"] = merged["indices"]
    if pruning_metadata is not None:
        metadata["pruning"] = dict(pruning_metadata)
    if candidate_selection is not None:
        metadata["candidate_selection"] = dict(candidate_selection)
    vis_dynamic_generators(
        S=evaluation.final_S,
        basis=original_basis,
        task=args.task,
        sample=args.final_vis_sample,
        theta_terms=evaluation.theta_terms,
        save_vis_path=args.final_vis_dir,
        feature_scales=evaluation.feature_scales,
        threshold=evaluation.threshold,
        normalization_metadata=metadata,
    )
    print(
        f"saved sparse generators to {args.final_vis_dir}/{args.task}_{args.final_vis_sample}",
        flush=True,
    )


def _default_reuse_config(task: str, seed: int) -> Dict[str, object]:
    """Default to the seed-based RS-AC output naming."""

    path = Path("vis") / f"{task}_{seed}" / "generator.pt"
    if path.exists():
        return {"generator_path": str(path)}
    return dict(DEFAULT_REUSE_GENERATORS.get(task, {"generator_path": str(path)}))


def _reuse_generator_inputs(args: argparse.Namespace, cli_args: argparse.Namespace):
    defaults = _default_reuse_config(args.task, int(args.seed))
    generator_path = _resolve_generator_path(
        cli_args.generator_path or defaults.get("generator_path"),
        task=args.task,
        seed=int(args.seed),
    )
    if not generator_path.exists():
        raise FileNotFoundError(f"Saved generator file not found: {generator_path}")
    theta_terms = (
        _split_terms(cli_args.theta_terms)
        or _load_theta_terms_sidecar(generator_path)
        or defaults.get("theta_terms")
    )
    if theta_terms is None:
        raise ValueError(
            f"Cannot infer theta terms for {generator_path}. Put theta_terms.txt "
            f"next to generator.pt or pass --theta-terms."
        )
    return generator_path, tuple(str(term) for term in theta_terms)


def main(argv: Optional[Sequence[str]] = None) -> None:
    cli_args = build_arg_parser().parse_args(argv)
    if cli_args.prune_ignore_d_change_below_support is not None:
        raise ValueError("--prune-ignore-d-change-below-support is no longer supported; pruning must preserve the actual null-space dimension.")
    equation = normalize_equation_name(cli_args.equation)
    theta_terms_override = _split_terms(cli_args.theta_terms)
    reuse_generator = bool(cli_args.reuse_generator or cli_args.generator_path is not None)
    fixed_theta_only = bool(theta_terms_override is not None and not reuse_generator)
    checkpoint_path = None
    if cli_args.checkpoint is not None or not (reuse_generator or fixed_theta_only):
        checkpoint_path = (
            Path(cli_args.checkpoint)
            if cli_args.checkpoint is not None
            else _default_checkpoint_path(equation, cli_args.seed if cli_args.seed is not None else 0)
        )

    bootstrap_args = build_train_arg_parser().parse_args(build_equation_argv(equation))
    if cli_args.gpu is not None:
        bootstrap_args.gpu = int(cli_args.gpu)
    device = device_from_args(bootstrap_args)
    checkpoint_state = _load_checkpoint(checkpoint_path, device) if checkpoint_path is not None else {}
    args = _train_args_from_checkpoint(
        equation=equation,
        checkpoint_state=checkpoint_state,
        cli_args=cli_args,
    )
    source_search_samples = checkpoint_state.get("search_samples") if checkpoint_path is not None else None
    candidate_selection = None
    if reuse_generator:
        generator_path, theta_terms = _reuse_generator_inputs(args, cli_args)
        source_metadata_path = generator_path.parent / "normalization.json"
        source_metadata = (json.loads(source_metadata_path.read_text(encoding="utf-8"))
                           if source_metadata_path.exists() else {})
        source_search_samples = source_metadata.get("search_samples")
        if theta_terms_override is None:
            candidate_selection = source_metadata.get("candidate_selection")
        _restore_generator_jacobian_source(args, cli_args, source_metadata)
    device = device_from_args(args)
    set_seed(int(args.seed))

    train_dataset = build_train_dataset(args)
    data_seed = args.seed if args.fix_seed is None else args.fix_seed
    train_loader = build_seeded_permutation_loader(
        train_dataset,
        batch_size=1,
        seed=data_seed,
    )

    model = load_search_model(args, device)
    current_source = getattr(model, "jacobian_source_metadata", None)
    if isinstance(current_source, dict):
        args.jacobian_source_lineage = _jacobian_source_lineage(
            args._source_jacobian_metadata, current_source,
            has_source_artifact=bool(reuse_generator or checkpoint_path is not None),
        )
        if args.jacobian_source_lineage is not None:
            args.jacobian_source_lineage["reference"] = getattr(args, "jacobian_source_reference", "policy_checkpoint")
        if args.jacobian_source_lineage and args.jacobian_source_lineage["re_evaluated"]:
            print("Re-evaluating the saved policy/library with the selected Jacobian source "
                  f"({args.jacobian_source_lineage['reason']}).", flush=True)

    env = build_env(args, model, train_loader, device)
    env.fast_finalize = False

    k_max = args.k_max
    if k_max is None:
        k_max = max(int(args.max_theta_size) - len(args.theta0), 1)

    print("=== pruning and visualize saved RS-AC policy ===", flush=True)
    print(f"task={args.task}, checkpoint={checkpoint_path}, reuse_generator={reuse_generator}", flush=True)
    print(
        f"batch_size={args.final_vis_batch_size}, k_max={k_max}, "
        f"greedy={args.final_vis_greedy}, temperature={args.final_vis_temperature}, "
        f"sample_size={args.sample_size}, sample={args.sample}",
        flush=True,
    )
    print(
        f"spectral_threshold_min={args.spectral_threshold_min}, "
        f"spectral_threshold_max={args.spectral_threshold_max}, "
        f"use_threshold_bounds={args.use_threshold_bounds}, spectral_floor={args.spectral_floor}",
        flush=True,
    )
    if args.rank_selection == "validated":
        print(f"rank_selection=validated, fixed_fit_groups=3, fit_points_each={args.sample_size}, "
              f"scale_points={args.rank_scale_points}, validation_points={args.rank_validation_points}, "
              f"scale_and_validation_residual_tolerance={args.rank_residual_tolerance:g}, "
              f"stability_tolerance={args.rank_stability_tolerance:g}, "
              f"function_stability_tolerance={getattr(args, 'rank_function_stability_tolerance', 1e-2):g}",
              flush=True)
    else:
        print(
            "sample_window=" + describe_sample_window(
                len(train_dataset), seed=data_seed, sample=args.sample, sample_size=args.sample_size,
            ), flush=True,
        )

    with torch.no_grad():
        support_basis = None
        if reuse_generator:
            support_basis = _load_tensor(generator_path, device=torch.device("cpu"))
            print(
                f"reuse_generator_path={generator_path}, "
                f"generator_shape={tuple(support_basis.shape)}, "
                f"theta={','.join(theta_terms)}",
                flush=True,
            )
        elif fixed_theta_only:
            theta_terms = tuple(str(term) for term in theta_terms_override)
            print(
                f"fixed_theta={','.join(theta_terms)}",
                flush=True,
            )
        else:
            policy = build_policy(args, device)
            state_dict = checkpoint_state.get("policy_state_dict", checkpoint_state)
            policy.load_state_dict(state_dict)
            policy.eval()
            episodes = collect_rollouts(
                policy,
                env,
                batch_size=args.final_vis_batch_size,
                theta0=args.theta0,
                max_steps=int(k_max),
                max_len=args.max_tree_len,
                greedy=bool(args.final_vis_greedy),
                temperature=args.final_vis_temperature,
                strict_prefix_ban=bool(args.strict_prefix_ban),
                forbid_observed=not bool(args.allow_observed),
                pad_to=args.pad_to,
            )

            best, candidate_selection = select_final_episode(
                episodes,
                historical_best=checkpoint_state.get("best_candidate"),
                args=args,
                env=env,
                policy=policy,
            )
            best_result = best.final.result
            theta_terms = best_result.theta_terms
            print(
                f"best_final_episode={best.index}, "
                f"total_return={_episode_total_return(best):.6g}, "
                f"D0={best_result.D0}, D_best={best_result.D_best}, "
                f"DeltaD={best_result.delta_D}, "
                f"theta={','.join(theta_terms)}",
                flush=True,
            )

        pruning_metadata = None
        if bool(cli_args.no_prune):
            evaluation = evaluate_theta_terms(env, theta_terms)
            evaluation, _, _ = prepare_theta_evaluation(
                env, evaluation, _split_terms(cli_args.mandatory_terms) or tuple(args.theta0),
                reference_state=env.state,
            )
        else:
            mandatory_terms = _split_terms(cli_args.mandatory_terms) or tuple(args.theta0)
            prune_result = post_rl_backward_prune(
                env=env,
                theta_terms=theta_terms,
                mandatory_terms=mandatory_terms,
                device=device,
                sparsify_epsilon1=args.sparsify_epsilon1,
                sparsify_epsilon2=args.sparsify_epsilon2,
                sparsify_log_interval=args.sparsify_log_interval,
                sparsify_max_iter=args.sparsify_max_iter,
                sparsify_verbose=bool(args.sparsify_verbose),
                support_basis=support_basis,
                condition_tolerance=float(cli_args.prune_condition_tolerance),
                subspace_tolerance=cli_args.prune_subspace_tolerance,
                max_kappa=cli_args.prune_max_kappa,
                ignore_d_change_below_support=cli_args.prune_ignore_d_change_below_support,
            )
            evaluation = prune_result.pruned
            pruning_metadata = {
                "reference_theta_terms": list(prune_result.reference.theta_terms),
                "reference_nullity": prune_result.reference.D_final,
                "condition_tolerance": float(cli_args.prune_condition_tolerance),
                "max_kappa": cli_args.prune_max_kappa,
                "subspace_tolerance": prune_result.subspace_tolerance,
                "subspace_error": prune_result.subspace_error,
                "comparison_coordinates": "original",
                "reference": "before_pruning",
            }

        visualize_theta_evaluation(
            evaluation=evaluation,
            args=args,
            device=device,
            pruning_metadata=pruning_metadata,
            source_search_samples=source_search_samples,
            inherit_search_samples=bool(reuse_generator or not fixed_theta_only),
            candidate_selection=candidate_selection,
        )


if __name__ == "__main__":
    main()
