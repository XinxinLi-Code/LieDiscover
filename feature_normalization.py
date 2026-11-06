"""Column RMS scaling, kept separate from spectral cutoff selection.

Coefficient vectors stack variable blocks, with all feature coefficients inside
each block. Symbolic terms remain unchanged; numerical columns use fixed scales.
"""

import math
from typing import Mapping, Sequence

import torch


def validate_feature_scales(scales, num_terms, device):
    """Return a finite, positive float64 scale vector in feature order."""
    values = torch.as_tensor(scales, dtype=torch.float64, device=device)
    if values.ndim != 1 or values.numel() != int(num_terms):
        raise ValueError(
            f"Expected {int(num_terms)} feature scales, got shape {tuple(values.shape)}."
        )
    if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError("Feature scales must be finite and strictly positive.")
    return values


def validate_coefficient_scales(scales, num_terms, variable_dim, device):
    """Expand legacy per-term scales or validate per-coefficient column scales.

    New scales have ``variable_dim * num_terms`` entries in variable-block
    order. Legacy vectors have ``num_terms`` entries shared by all blocks.
    """
    num_terms, variable_dim = int(num_terms), int(variable_dim)
    if num_terms <= 0 or variable_dim <= 0:
        raise ValueError("Term count and variable dimension must be positive.")
    values = torch.as_tensor(scales, dtype=torch.float64, device=device)
    if values.ndim != 1:
        raise ValueError("Coefficient scales must be a one-dimensional vector.")
    if values.numel() == num_terms:
        values = validate_feature_scales(values, num_terms, device).repeat(variable_dim)
    else:
        values = validate_feature_scales(values, num_terms * variable_dim, device)
    return values


def restrict_library_scales(scales, retained_indices, *, num_terms, variable_dim, device=None):
    """Restrict fixed scales to retained terms in every coefficient block.

    Retained terms may be reordered. Legacy per-term vectors stay per-term;
    prolonged-column vectors stay in variable-block order. No scale is refit.
    """
    values = torch.as_tensor(scales, dtype=torch.float64, device=device)
    num_terms, variable_dim = int(num_terms), int(variable_dim)
    expanded = validate_coefficient_scales(values, num_terms, variable_dim, values.device)
    positions = torch.as_tensor(retained_indices, dtype=torch.long, device=values.device)
    if positions.ndim != 1 or positions.numel() == 0:
        raise ValueError("Retained term indices must be a nonempty vector.")
    if bool((positions < 0).any()) or bool((positions >= num_terms).any()):
        raise ValueError("Retained term indices are outside the reference library.")
    if positions.unique().numel() != positions.numel():
        raise ValueError("Retained term indices must be unique.")
    if values.numel() == num_terms:
        return values.index_select(0, positions)
    return expanded.reshape(variable_dim, num_terms).index_select(1, positions).reshape(-1)


def feature_scales(
    samples: Sequence[Mapping],
    *,
    task,
    device,
    theta_terms,
    eps_s=1e-12,
):
    """Compute feature RMS on a fixed sample set; near-zero RMS uses scale 1.

    Values are evaluated in float64. Top-task momentum components are counted
    individually. A scaled sum of squares avoids overflow when forming the RMS.
    """
    from discovery import compute_theta

    eps_s = float(eps_s)
    if not math.isfinite(eps_s) or eps_s <= 0:
        raise ValueError("eps_s must be finite and strictly positive.")
    maximum = None
    sum_squares = None
    count = 0
    with torch.no_grad():
        for sample in samples:
            double_data = {
                key: torch.as_tensor(value, device=device, dtype=torch.float64)
                for key, value in sample.items()
            }
            values = compute_theta(double_data, task, device, theta_terms=theta_terms)["th"]
            if task != "top":
                values = values.movedim(0, -1)
            num_terms = values.shape[-1]
            if num_terms == 0:
                raise ValueError("A feature library must contain at least one term.")
            values = values.reshape(-1, num_terms).to(dtype=torch.float64)
            if values.shape[0] == 0:
                continue
            if not bool(torch.isfinite(values).all()):
                raise ValueError("Feature values must be finite on the reference samples.")
            batch_maximum = values.abs().amax(dim=0)
            if maximum is None:
                maximum = torch.zeros_like(batch_maximum)
                sum_squares = torch.zeros_like(batch_maximum)
            elif maximum.numel() != num_terms:
                raise ValueError("Reference samples produced inconsistent feature counts.")
            new_maximum = torch.maximum(maximum, batch_maximum)
            denominator = torch.where(new_maximum > 0, new_maximum, torch.ones_like(new_maximum))
            sum_squares = (
                sum_squares * (maximum / denominator).square()
                + (values / denominator).square().sum(dim=0)
            )
            maximum = new_maximum
            count += values.shape[0]
    if count == 0:
        raise ValueError("Reference samples must contain at least one feature observation.")
    scales = maximum * torch.sqrt(sum_squares / count)
    scales = torch.where(scales <= eps_s, torch.ones_like(scales), scales)
    return validate_feature_scales(scales, scales.numel(), device)


def normalize_constraint_rows(c_rows, scales, variable_dim):
    """Divide raw columns by prolonged-column or legacy per-term scales."""
    if not c_rows:
        raise ValueError("Constraint rows must be non-empty.")
    variable_dim = int(variable_dim)
    if variable_dim <= 0:
        raise ValueError("variable_dim must be positive.")
    first = c_rows[0]
    if first.ndim != 2 or first.shape[1] % variable_dim:
        raise ValueError("Constraint column count must match the variable blocks.")
    column_scales = validate_coefficient_scales(
        scales, first.shape[1] // variable_dim, variable_dim, first.device,
    )
    result = []
    for row in c_rows:
        if row.ndim != 2 or row.shape[1] != column_scales.numel():
            raise ValueError("Constraint rows have inconsistent coefficient dimensions.")
        row = row.to(device=first.device, dtype=torch.float64)
        result.append(row / column_scales.unsqueeze(0))
    return result


def restore_coefficient_basis(basis, scales):
    """Restore row-wise coefficients from column or legacy per-term scales.

    A full column scale vector divides each coefficient once. Legacy vectors
    are repeated over variable blocks for compatibility with saved results.
    """
    if basis.ndim != 2:
        raise ValueError("The coefficient basis must have shape [generators, coefficients].")
    num_terms = torch.as_tensor(scales).numel()
    if num_terms == 0 or basis.shape[1] % num_terms:
        raise ValueError("The basis columns must be complete feature blocks.")
    scales = validate_feature_scales(scales, num_terms, basis.device)
    variable_dim = basis.shape[1] // num_terms
    return basis.to(dtype=torch.float64) / scales.repeat(variable_dim).unsqueeze(0)


def prolonged_column_scales(prolonged_rows, *, eps_s=1e-12, row_weights=None):
    """Return one fixed float64 RMS scale per prolonged coefficient column.

    Each matrix is one sample's prolonged feature matrix B_i. The scale is
    sqrt(mean_i(sum_r((row_weights[r] * B_i[r, column])**2))). Optional positive
    jet-row weights affect this norm only: normalization still divides every
    entry in a coefficient column by the same constant. A scaled sum of squares
    avoids overflow when squaring large, finite entries. Near-zero RMS uses 1.
    Scales are computed before multiplication by the surrogate Jacobians.
    """
    eps_s = float(eps_s)
    if not math.isfinite(eps_s) or eps_s <= 0:
        raise ValueError("eps_s must be finite and strictly positive.")
    maximum = None
    sum_squares = None
    shape = None
    device = None
    weights = None
    count = 0
    finite_checks = []
    finite_errors = []

    def check_finite():
        # Read the device once for the whole pool, rather than once per sample.
        # The ordered flags also preserve the original first failing check.
        if finite_checks:
            flags = torch.stack(finite_checks)
            if not bool(flags.all()):
                for valid, message in zip(flags.cpu().tolist(), finite_errors):
                    if not valid:
                        raise ValueError(message)

    with torch.no_grad():
        try:
            for matrix in prolonged_rows:
                values = torch.as_tensor(matrix, dtype=torch.float64, device=device)
                if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
                    raise ValueError("Prolonged matrices must be non-empty two-dimensional matrices.")
                if shape is None:
                    shape = tuple(values.shape)
                    device = values.device
                    if row_weights is not None:
                        weights = torch.as_tensor(row_weights, dtype=torch.float64, device=device)
                        if weights.ndim != 1 or weights.numel() != shape[0]:
                            raise ValueError("Jet-row weights must have one entry per prolonged row.")
                        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
                            raise ValueError("Jet-row weights must be finite and strictly positive.")
                elif tuple(values.shape) != shape:
                    raise ValueError("Prolonged matrices must have a consistent shape.")
                finite_checks.append(torch.isfinite(values).all())
                finite_errors.append("Prolonged features must be finite on the reference samples.")
                if weights is not None:
                    values = values * weights.unsqueeze(1)
                    finite_checks.append(torch.isfinite(values).all())
                    finite_errors.append("Weighted prolonged features must be representable in float64.")
                batch_maximum = values.abs().amax(dim=0)
                if maximum is None:
                    maximum = torch.zeros_like(batch_maximum)
                    sum_squares = torch.zeros_like(batch_maximum)
                new_maximum = torch.maximum(maximum, batch_maximum)
                denominator = torch.where(new_maximum > 0, new_maximum, torch.ones_like(new_maximum))
                sum_squares = (
                    sum_squares * (maximum / denominator).square()
                    + (values / denominator).square().sum(dim=0)
                )
                maximum = new_maximum
                count += 1
        except Exception:
            # An earlier nonfinite sample takes precedence over a later shape,
            # conversion, or iterator error, as in the original serial checks.
            check_finite()
            raise
        check_finite()
    if count == 0:
        raise ValueError("Reference samples must contain at least one prolonged matrix.")
    scales = maximum * torch.sqrt(sum_squares / count)
    scales = torch.where(scales <= eps_s, torch.ones_like(scales), scales)
    return validate_feature_scales(scales, shape[1], device)


def normalize_columns(rows, column_scales):
    """Divide numerical columns by fixed per-coefficient scales in float64.

    Applicable to both prolonged feature matrices and constraint matrices.
    Jet-row weights used to estimate scales are not applied to these matrices.
    """
    result = []
    scales = None
    device = None
    for matrix in rows:
        values = torch.as_tensor(matrix, dtype=torch.float64, device=device)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("Matrices must have shape [rows, coefficients] with coefficients non-empty.")
        if scales is None:
            device = values.device
            scales = validate_feature_scales(column_scales, values.shape[1], device)
        elif values.shape[1] != scales.numel():
            raise ValueError("Matrices have inconsistent coefficient dimensions.")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Matrices must contain finite entries.")
        normalized = values / scales.unsqueeze(0)
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("Normalized matrices must be representable in float64.")
        result.append(normalized)
    if not result:
        raise ValueError("Matrices must be non-empty.")
    return result


def restore_column_coefficient_basis(basis, column_scales):
    """Restore row-wise generator vectors using one scale per coefficient."""
    values = torch.as_tensor(basis, dtype=torch.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("The coefficient basis must have shape [generators, coefficients].")
    scales = validate_feature_scales(column_scales, values.shape[1], values.device)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("The coefficient basis must contain finite entries.")
    restored = values / scales.unsqueeze(0)
    if not bool(torch.isfinite(restored).all()):
        raise ValueError("Restored coefficients must be representable in float64.")
    return restored


# Search arithmetic uses float64 independently of policy training.
SEARCH_DTYPE = torch.float64


def prepare_search_model(model, device):
    """Promote parameters and buffers before differentiating the surrogate."""
    target = torch.device(device)
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    if any(
        tensor.device != target
        or (tensor.is_floating_point() and tensor.dtype != SEARCH_DTYPE)
        for tensor in tensors
    ):
        model = model.to(device=target, dtype=SEARCH_DTYPE)
    # Explicit WSINDy coefficients have a full-precision JSON source. A mere
    # .double() would retain any earlier float32 rounding of these buffers.
    with torch.no_grad():
        for module in model.modules():
            restore = getattr(module, "restore_coefficients", None)
            if restore is not None:
                restore()
    return model


def search_data(data, device):
    """Convert jet values before nonlinear features or derivatives are evaluated."""
    return {
        key: torch.as_tensor(value, device=device, dtype=SEARCH_DTYPE)
        for key, value in data.items()
    }


# Compare generator subspaces after restoring their original coefficients.
DEFAULT_SUBSPACE_TOLERANCE = 1e-2


def _orthonormal_columns(matrix):
    """Compute a thin QR without overflowing column norms or changing rank."""
    rows, columns = matrix.shape
    if columns == 0:
        return matrix
    if columns > rows:
        raise ValueError("A coefficient basis cannot have more vectors than coordinates.")
    maxima = matrix.abs().amax(dim=0)
    if bool((maxima == 0).any()):
        raise ValueError("A restored coefficient basis contains a zero vector.")
    scaled = matrix / maxima
    scaled = scaled / torch.linalg.vector_norm(scaled, dim=0)
    Q, R = torch.linalg.qr(scaled, mode="reduced")
    singular_values = torch.linalg.svdvals(R)
    rank_tolerance = max(rows, columns) * torch.finfo(matrix.dtype).eps * singular_values[0]
    if singular_values[-1] <= rank_tolerance:
        raise ValueError("The restored coefficient basis is numerically rank deficient.")
    if not bool(torch.isfinite(Q).all()):
        raise ValueError("Orthonormalization produced non-finite coefficients.")
    return Q


def restored_subspace_basis(basis, scales, theta_terms, reference_terms):
    """Restore normalized row vectors, zero-embed each variable block, then QR.

    Returns float64 orthonormal columns in the original reference-library order.
    Full scales follow variable blocks; legacy per-term scales are also accepted.
    The input sparse/normalized coefficient basis is never modified.
    """
    terms = tuple(str(term) for term in theta_terms)
    reference_terms = tuple(str(term) for term in reference_terms)
    if not terms or not reference_terms:
        raise ValueError("A coefficient library must contain at least one term.")
    if len(set(terms)) != len(terms) or len(set(reference_terms)) != len(reference_terms):
        raise ValueError("Coefficient libraries must contain unique terms.")
    positions = {term: index for index, term in enumerate(reference_terms)}
    if any(term not in positions for term in terms):
        raise ValueError("The trial library must be a subset of the reference library.")
    basis = torch.as_tensor(basis).detach().to(device="cpu", dtype=torch.float64)
    if basis.ndim != 2 or basis.shape[1] == 0 or basis.shape[1] % len(terms):
        raise ValueError("The row basis must contain complete, nonempty variable blocks.")
    if not bool(torch.isfinite(basis).all()):
        raise ValueError("The coefficient basis must contain finite values.")
    variable_dim = basis.shape[1] // len(terms)
    column_scales = validate_coefficient_scales(
        scales, len(terms), variable_dim, basis.device,
    )
    restored = restore_coefficient_basis(basis, column_scales)
    if not bool(torch.isfinite(restored).all()):
        raise ValueError("Restoring coefficients produced non-finite values.")
    embedded = torch.zeros(
        (basis.shape[0], variable_dim * len(reference_terms)), dtype=torch.float64,
    )
    indices = [
        block * len(reference_terms) + positions[term]
        for block in range(variable_dim) for term in terms
    ]
    embedded[:, indices] = restored
    return _orthonormal_columns(embedded.T)


def projector_subspace_distance(reference_basis, trial_basis):
    """Return the projector distance for orthonormal column bases.

    At equal dimension, the projection residual has norm sin(theta_max), equal
    to the projector-difference norm. Different dimensions give distance 1.
    """
    bases = []
    for basis in (reference_basis, trial_basis):
        basis = torch.as_tensor(basis).detach().to(device="cpu", dtype=torch.float64)
        if basis.ndim != 2 or basis.shape[1] > basis.shape[0]:
            raise ValueError("Subspace bases must be matrices with vectors in their columns.")
        if not bool(torch.isfinite(basis).all()):
            raise ValueError("Subspace bases must contain finite values.")
        identity = torch.eye(basis.shape[1], dtype=torch.float64)
        if not torch.allclose(basis.T @ basis, identity, rtol=0, atol=1e-10):
            raise ValueError("Subspace bases must have orthonormal columns.")
        bases.append(basis)
    reference_basis, trial_basis = bases
    if reference_basis.shape[0] != trial_basis.shape[0]:
        raise ValueError("Subspace bases must use the same coefficient coordinates.")
    if reference_basis.shape[1] != trial_basis.shape[1]:
        return 1.0
    if reference_basis.shape[1] == 0:
        return 0.0
    residual = trial_basis - reference_basis @ (reference_basis.T @ trial_basis)
    error = float(torch.linalg.matrix_norm(residual, ord=2).item())
    return min(1.0, max(0.0, error))
