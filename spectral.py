"""Adaptive relative spectral cutoffs for an already normalized system."""

import math
from collections import namedtuple
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import torch


SpectralCut = namedtuple("SpectralCut", ["threshold", "gap", "confidence", "rank_cut", "null_dim"])
# Retained only for explicit use_threshold_bounds=True comparisons.
DEFAULT_THRESHOLD_MIN = 1e-7
DEFAULT_THRESHOLD_MAX = 1e-2
DEFAULT_SPECTRAL_FLOOR = 1e-12


@dataclass(frozen=True)
class RankSelectionConfig:
    """Frozen validation tolerances; no reference dimension enters selection."""

    top_k: int = 3
    residual_tolerance: float = 1e-3
    stability_tolerance: float = 1e-2
    spectral_floor: float = DEFAULT_SPECTRAL_FLOOR
    function_stability_tolerance: float = 1e-2

    def __post_init__(self):
        if isinstance(self.top_k, bool) or int(self.top_k) != self.top_k or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        for name in ("residual_tolerance", "stability_tolerance", "function_stability_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly between zero and one.")
        _validate_floor(self.spectral_floor)


class RankUnresolvedError(ValueError):
    """No validated dimension was found; this does not mean zero nullity."""

    def __init__(self, decision):
        self.decision = decision
        message = (
            f"Dimension unresolved: {decision.reason}; "
            f"required Rs<={decision.config.residual_tolerance:.6g}, "
            f"Rv<={decision.config.residual_tolerance:.6g}, "
            f"S<={decision.config.stability_tolerance:.6g}, "
            f"S_function<={decision.config.function_stability_tolerance:.6g}"
        )
        if decision.candidates:
            details = "; ".join(
                f"d={item['dimension']}: Rs={item['scale_residual']:.6g}, "
                f"Rv={item['residual']:.6g}, S={item['stability']:.6g}, "
                f"S_function={item['function_stability']}"
                for item in decision.candidates[:9]
            )
            message += f"; candidates [{details}]"
        super().__init__(message)


@dataclass(frozen=True)
class RankDecision:
    resolved: bool
    dimension: Optional[int]
    basis: torch.Tensor
    spectra: tuple
    vhs: tuple
    kappa: float
    residual: float
    scale_residual: float
    stability: float
    function_stability: float
    reason: str
    candidates: tuple
    proposed_dimensions: tuple
    threshold: float
    sigma_max: float
    sigma_min_active: float
    config: RankSelectionConfig

    def metadata(self):
        def finite(value):
            return float(value) if math.isfinite(float(value)) else None

        return {
            "method": "validated_top_gap", "resolved": self.resolved,
            "rule_version": "function_space_guard_v1",
            "selection_checks": ["scale_reference_residual", "validation_residual", "subset_stability",
                                 "function_rank", "function_subset_stability"],
            "scale_residual_source": "training_scale_reference_pool",
            "function_stability_source": "training_scale_reference_pool",
            "dimension": self.dimension, "reason": self.reason,
            "top_k": int(self.config.top_k),
            "residual_tolerance": float(self.config.residual_tolerance),
            "stability_tolerance": float(self.config.stability_tolerance),
            "function_stability_tolerance": float(self.config.function_stability_tolerance),
            "spectral_floor": float(self.config.spectral_floor),
            "residual": finite(self.residual), "stability": finite(self.stability),
            "function_stability": finite(self.function_stability),
            "scale_residual": finite(self.scale_residual),
            "kappa": finite(self.kappa), "candidates": list(self.candidates),
            "proposed_dimensions": [list(group) for group in self.proposed_dimensions],
            "basis_subset": 0, "threshold_role": "display_only",
            "display_threshold": finite(self.threshold),
            "condition_number": "max_subset_unfloored_active_spectrum",
        }


def top_gap_dimensions(S, top_k=3, spectral_floor=DEFAULT_SPECTRAL_FLOOR):
    """Return distinct dimensions at the largest positive logarithmic gaps."""
    if isinstance(top_k, bool) or int(top_k) != top_k or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    spectral_floor = _validate_floor(spectral_floor)
    values = _singular_values(S)
    if values.numel() < 2 or float(values[0]) == 0:
        return ()
    if bool((values[:-1] < values[1:]).any()):
        raise ValueError("Singular values must be sorted in descending order.")
    relative = relative_singular_values(values, floor=spectral_floor)
    logs = torch.log10(relative)
    gaps = logs[:-1] - logs[1:]
    gap_values = gaps.detach().cpu().tolist()
    selected = []
    for index in sorted(range(len(gap_values)), key=lambda j: (-gap_values[j], j)):
        threshold = float(torch.pow(10., .5 * (logs[index] + logs[index + 1])))
        if gap_values[index] <= 0 or not spectral_floor < threshold < 1:
            continue
        selected.append(values.numel() - index - 1)
        if len(selected) == top_k:
            break
    return tuple(selected)


def _function_space_diagnostics(function_rows, bases, tolerance):
    """Compare whole generator spaces without squaring their condition numbers.

    The common function matrix evaluates all generator components in the same
    coefficient coordinates as the constraints. A numerically rank-deficient
    image is unresolved, not an instruction to discard individual features.
    """
    spaces, singular_values, ranks, rank_tolerances = [], [], [], []
    dimension = bases[0].shape[1]
    for basis in bases:
        values = function_rows @ basis
        U, singular, _ = torch.linalg.svd(values, full_matrices=False)
        rank_tolerance = max(values.shape) * torch.finfo(values.dtype).eps * singular[0]
        rank = int((singular > rank_tolerance).sum().item())
        spaces.append(U)
        singular_values.append(singular.detach().cpu().tolist())
        ranks.append(rank)
        rank_tolerances.append(float(rank_tolerance))
    resolved = all(rank == dimension for rank in ranks)
    distances = []
    if resolved:
        # Equal-rank projection distance, using thin matrices rather than
        # allocating a square projector over all sample/component rows.
        distances = [float(torch.linalg.matrix_norm(right - left @ (left.T @ right), ord=2))
                     for left, right in combinations(spaces, 2)]
    stability = max(distances) if resolved else None
    return {
        "function_stability": stability,
        "function_stability_pairs": distances,
        "function_rank_resolved": resolved,
        "function_ranks": ranks,
        "function_singular_values": singular_values,
        "function_rank_tolerances": rank_tolerances,
        "function_checks_passed": resolved and stability <= tolerance,
    }


def select_validated_nullspace(spectra, vhs, validation_rows, config=None, *, scale_rows,
                              function_rows):
    """Validate the union of top-gap candidates on three fixed sample subsets.

    Inputs already use the same prolonged-column scales. Gram factorization is
    performed by the caller, unchanged from the search path. The selected basis
    is taken directly from subset zero; a diagnostic cutoff never reselects it.
    Both the training scale-reference pool and the independent validation pool
    must satisfy the same relative residual tolerance. Neither pool is refit.
    Generator functions are evaluated on the common scale pool to check that
    the whole candidate function space is numerically resolved and stable.
    This adds no new constraint-residual cutoff.
    """
    config = RankSelectionConfig() if config is None else config
    if len(spectra) != 3 or len(vhs) != 3:
        raise ValueError("Validated selection requires exactly three fixed subsets.")
    spectra = tuple(_singular_values(values) for values in spectra)
    n = spectra[0].numel()
    if n == 0 or any(values.numel() != n for values in spectra):
        raise ValueError("All subset spectra must have the same positive length.")
    device = vhs[0].device
    vhs = tuple(vh.to(device=device, dtype=torch.float64) for vh in vhs)
    for values, vh in zip(spectra, vhs):
        if bool((values[:-1] < values[1:]).any()):
            raise ValueError("Subset spectra must be sorted in descending order.")
        if vh.shape != (n, n) or not bool(torch.isfinite(vh).all()):
            raise ValueError("Each Vh must be a finite square matrix matching its spectrum.")
        orthogonality = vh @ vh.T - torch.eye(n, dtype=torch.float64, device=device)
        if float(torch.max(torch.abs(orthogonality))) > 1e-7:
            raise ValueError("Gram decomposition returned non-orthonormal right singular vectors.")
    cv = torch.as_tensor(validation_rows, device=device, dtype=torch.float64)
    if cv.ndim != 2 or cv.shape[0] == 0 or cv.shape[1] != n or not bool(torch.isfinite(cv).all()):
        raise ValueError("Validation rows must be a nonempty finite matrix with matching columns.")
    cs = torch.as_tensor(scale_rows, device=device, dtype=torch.float64)
    if cs.ndim != 2 or cs.shape[0] == 0 or cs.shape[1] != n or not bool(torch.isfinite(cs).all()):
        raise ValueError("Scale-reference rows must be a nonempty finite matrix with matching columns.")
    function_rows = torch.as_tensor(function_rows, device=device, dtype=torch.float64)
    if (function_rows.ndim != 2 or function_rows.shape[0] == 0
            or function_rows.shape[1] != n or not bool(torch.isfinite(function_rows).all())):
        raise ValueError("Function-reference rows must be a nonempty finite matrix with matching columns.")
    proposed = tuple(top_gap_dimensions(values, config.top_k, config.spectral_floor)
                     for values in spectra)
    denominator = float(torch.linalg.matrix_norm(cv, ord=2))
    if not math.isfinite(denominator):
        raise ValueError("Validation operator norm is nonfinite.")
    scale_denominator = float(torch.linalg.matrix_norm(cs, ord=2))
    if not math.isfinite(scale_denominator):
        raise ValueError("Scale-reference operator norm is nonfinite.")
    zero_spectra = tuple(float(values[0]) == 0 for values in spectra)

    def result(*, dimension=None, reason, candidates=(), residual=float("nan"),
               scale_residual=float("nan"),
               stability=float("nan"), function_stability=float("nan"),
               kappa=float("inf"), sigma_max=float("nan"),
               sigma_min_active=float("nan")):
        resolved = dimension is not None
        threshold = float("nan")
        if resolved and 0 < dimension < n:
            relative = relative_singular_values(spectra[0], floor=config.spectral_floor)
            cut = n - dimension
            threshold = float(torch.sqrt(relative[cut - 1] * relative[cut]))
        return RankDecision(
            resolved=resolved, dimension=dimension,
            basis=vhs[0][-dimension:].detach().clone() if resolved and dimension > 0 else vhs[0][:0].clone(),
            spectra=spectra, vhs=vhs, kappa=kappa, residual=residual,
            scale_residual=scale_residual,
            stability=stability, function_stability=function_stability,
            reason=reason, candidates=tuple(candidates),
            proposed_dimensions=proposed, threshold=threshold, sigma_max=sigma_max,
            sigma_min_active=sigma_min_active, config=config,
        )

    if all(zero_spectra) and denominator == 0 and scale_denominator == 0:
        function = _function_space_diagnostics(
            function_rows, [vh.T for vh in vhs], config.function_stability_tolerance)
        metric = {
            "dimension": n, "residual": 0., "stability": 0., "scale_residual": 0.,
            "residual_by_subset": [0.] * 3, "stability_pairs": [0.] * 3,
            "scale_residual_by_subset": [0.] * 3, **function,
            "accepted": function["function_checks_passed"],
        }
        if not metric["accepted"]:
            return result(reason="no_candidate_passed_function_validation", candidates=(metric,))
        return result(dimension=n, reason="all_systems_zero", residual=0., scale_residual=0., stability=0.,
                      function_stability=function["function_stability"], candidates=(metric,),
                      kappa=1., sigma_max=0., sigma_min_active=0.)
    if any(zero_spectra) or denominator == 0 or scale_denominator == 0:
        return result(reason="inconsistent_zero_systems")
    dimensions = sorted(set().union(*map(set, proposed)))
    if not dimensions:
        return result(reason="no_positive_gap_candidates")
    metrics = []
    for dimension in dimensions:
        bases = [vh[-dimension:].T for vh in vhs]
        residual_norms = [torch.linalg.matrix_norm(cv @ basis, ord=2) for basis in bases]
        scale_norms = [torch.linalg.matrix_norm(cs @ basis, ord=2) for basis in bases]
        projectors = [basis @ basis.T for basis in bases]
        distance_norms = [torch.linalg.matrix_norm(left - right, ord=2)
                          for left, right in combinations(projectors, 2)]
        # Read the nine scalar diagnostics together; keep the original Python
        # float divisions and the individual matrix products/norms unchanged.
        norms = torch.stack(residual_norms + scale_norms + distance_norms).detach().cpu().tolist()
        residuals = [norm / denominator for norm in norms[:3]]
        scale_residuals = [norm / scale_denominator for norm in norms[3:6]]
        distances = norms[6:]
        residual, stability = max(residuals), max(distances)
        scale_residual = max(scale_residuals)
        function = _function_space_diagnostics(function_rows, bases, config.function_stability_tolerance)
        metrics.append({
            "dimension": dimension, "residual": residual, "stability": stability,
            "scale_residual": scale_residual,
            "residual_by_subset": residuals, "stability_pairs": distances,
            "scale_residual_by_subset": scale_residuals,
            **function,
            "accepted": residual <= config.residual_tolerance
                        and scale_residual <= config.residual_tolerance
                        and stability <= config.stability_tolerance
                        and function["function_checks_passed"],
        })
    feasible = [metric for metric in metrics if metric["accepted"]]
    if not feasible:
        return result(reason="no_candidate_passed_validation", candidates=metrics)
    selected = feasible[-1]
    dimension = selected["dimension"]
    active = n - dimension
    kappas = [float(values[0] / values[active - 1]) if float(values[active - 1]) > 0
              else float("inf") for values in spectra]
    worst = max(range(3), key=kappas.__getitem__)
    return result(dimension=dimension, reason="largest_validated_dimension", candidates=metrics,
                  residual=selected["residual"], stability=selected["stability"],
                  function_stability=selected["function_stability"],
                  scale_residual=selected["scale_residual"],
                  kappa=kappas[worst], sigma_max=float(spectra[worst][0]),
                  sigma_min_active=float(spectra[worst][active - 1]))


def _singular_values(S):
    values = torch.as_tensor(S).flatten().to(dtype=torch.float64).abs()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Singular values must be finite.")
    return values


def _validate_floor(floor):
    floor = float(floor)
    if not math.isfinite(floor) or not 0 < floor < 1:
        raise ValueError("The relative spectral floor must lie strictly between 0 and 1.")
    return floor


def _validate_threshold(threshold, floor):
    threshold = float(threshold)
    if not math.isfinite(threshold) or not floor < threshold < 1:
        raise ValueError("A relative threshold must satisfy spectral_floor < threshold < 1.")
    return threshold


def relative_singular_values(S, floor=DEFAULT_SPECTRAL_FLOOR):
    """Return max(abs(S) / max(abs(S)), floor) without changing its order."""
    floor = _validate_floor(floor)
    values = _singular_values(S)
    if values.numel() == 0:
        return values
    maximum = values.max()
    if float(maximum.item()) == 0:
        return torch.full_like(values, floor)
    return torch.clamp(values / maximum, min=floor)


def estimate_spectral_cut(
    S,
    fixed_threshold=None,
    threshold_min=None,
    threshold_max=None,
    spectral_floor=DEFAULT_SPECTRAL_FLOOR,
    use_threshold_bounds=False,
    **legacy_kwargs,
):
    """Select the largest valid log gap; all thresholds are relative.

    Former confidence options remain available for diagnostics. Former absolute
    floors, tail preferences, and minimum-active options do not alter selection.
    By default the largest positive log gap is selected over the entire floored
    relative spectrum, ignoring both manual bounds. Set use_threshold_bounds=True
    only to reproduce the former bounded selection.
    Without a positive gap, a threshold just above the numerical floor keeps a
    nonzero equal spectrum active and an identically zero spectrum all null.
    """
    supported = {
        "min_log10_gap", "confidence_center", "confidence_sharpness",
        "tail_relative_cap", "relative_floor", "absolute_floor", "min_active",
    }
    unknown = set(legacy_kwargs) - supported
    if unknown:
        raise TypeError(f"Unexpected spectral options: {sorted(unknown)}")
    floor = _validate_floor(spectral_floor)
    if use_threshold_bounds:
        lower = DEFAULT_THRESHOLD_MIN if threshold_min is None else float(threshold_min)
        upper = DEFAULT_THRESHOLD_MAX if threshold_max is None else float(threshold_max)
        if not (math.isfinite(lower) and math.isfinite(upper) and floor < lower <= upper < 1):
            raise ValueError("Relative bounds must satisfy spectral_floor < min <= max < 1.")
        threshold = lower
    else:
        threshold = _validate_threshold(math.nextafter(floor, 1.0), floor)
    values = torch.sort(_singular_values(S), descending=True).values
    gap = 0.0
    confidence = 0.0
    if fixed_threshold is not None:
        threshold = _validate_threshold(fixed_threshold, floor)
        if use_threshold_bounds and not lower <= threshold <= upper:
            raise ValueError("The fixed relative threshold must lie within the configured bounds.")
        confidence = 1.0
    relative = relative_singular_values(values, floor=floor)
    if fixed_threshold is None and values.numel() > 1 and float(values[0].item()) > 0:
        log_values = torch.log10(relative)
        gaps = log_values[:-1] - log_values[1:]
        if use_threshold_bounds:
            candidates = torch.sqrt(relative[:-1] * relative[1:])
            valid = (candidates >= lower) & (candidates <= upper)
        else:
            # The log midpoint avoids underflow for unusually small floors.
            candidates = torch.pow(10., 0.5 * (log_values[:-1] + log_values[1:]))
            valid = (gaps > 0) & (candidates > floor) & (candidates < 1)
        if bool(valid.any()):
            indices = torch.nonzero(valid, as_tuple=False).flatten()
            chosen = indices[torch.argmax(gaps[indices])]
            threshold = float(candidates[chosen].item())
            gap = float(gaps[chosen].item())
            center = float(legacy_kwargs.get("confidence_center", 0.8))
            sharpness = float(legacy_kwargs.get("confidence_sharpness", 8.0))
            confidence = float(torch.sigmoid(values.new_tensor(sharpness * (gap - center))).item())
            if gap >= float(legacy_kwargs.get("min_log10_gap", 1.0)):
                confidence = max(confidence, 0.95)
    null_dim = int((relative < threshold).sum().item())
    return SpectralCut(
        threshold=threshold, gap=gap, confidence=confidence,
        rank_cut=int(values.numel()) - null_dim, null_dim=null_dim,
    )


def select_null_basis(S, Vh, threshold, spectral_floor=DEFAULT_SPECTRAL_FLOOR):
    """Select right-singular-vector rows using the same relative cutoff."""
    if Vh is None:
        return None
    floor = _validate_floor(spectral_floor)
    threshold = _validate_threshold(threshold, floor)
    relative = relative_singular_values(S, floor=floor).to(device=Vh.device)
    if Vh.ndim != 2 or Vh.shape[0] != relative.numel():
        raise ValueError("Vh must have one row per singular value.")
    return Vh.to(dtype=torch.float64)[relative < threshold]


def active_singular_condition_number(
    S,
    threshold,
    eps=1e-12,
    spectral_floor=DEFAULT_SPECTRAL_FLOOR,
):
    """Return normalized-system active kappa and raw singular-value diagnostics.

    The compatibility argument eps does not replace the relative cutoff. For a
    zero spectrum the full coefficient space is null and kappa is 1 by convention.
    """
    floor = _validate_floor(spectral_floor)
    threshold = _validate_threshold(threshold, floor)
    values = torch.sort(_singular_values(S), descending=True).values
    if values.numel() == 0:
        return float("inf"), float("nan"), float("nan")
    if float(values[0].item()) == 0:
        return 1.0, 0.0, 0.0
    relative = relative_singular_values(values, floor=floor)
    active = values[relative >= threshold]
    sigma_max = float(active[0].item())
    sigma_min = float(active[-1].item())
    active_relative = relative[relative >= threshold]
    kappa = float((active_relative[0] / active_relative[-1]).item())
    return kappa, sigma_max, sigma_min
