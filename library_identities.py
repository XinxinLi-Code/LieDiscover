"""Remove proven constant-coefficient identities before numerical postprocessing.

The columns are symbolic library functions, not sampled data.  An exact change
of coordinates preserves every generator function; deciding which mapped
generator vectors are numerically independent remains the caller's job.
"""

from copy import copy
from dataclasses import dataclass, is_dataclass, replace
from typing import Tuple

import numpy as np
import sympy as sp
import torch

from feature_normalization import (
    DEFAULT_SUBSPACE_TOLERANCE, projector_subspace_distance, restored_subspace_basis,
    restore_coefficient_basis, restrict_library_scales, validate_coefficient_scales,
)


@dataclass(frozen=True)
class ExactLibraryReduction:
    """A retained library and ``Theta_original = transform * Theta_retained``.

    ``transform`` has shape [original terms, retained terms].  Retained terms
    have their original spelling and order.  ``identities`` describes each
    removed term, including whether it was requested as mandatory.
    """

    original_terms: Tuple[str, ...]
    theta_terms: Tuple[str, ...]
    kept_indices: Tuple[int, ...]
    transform: sp.ImmutableMatrix
    identities: Tuple[dict, ...]

    @property
    def changed(self):
        return len(self.theta_terms) != len(self.original_terms)


def _canonical_exact(expression):
    """Expand algebra and cosine powers without fitting or rounding numbers.

    A stored SymPy Float is replaced by its exact binary rational value, not by
    a nearby small rational.  This prevents floating-point cancellation from
    becoming a proof of a symbolic identity.  Cosine powers use the exact rule
    cos(a)**2 = 1 - sin(a)**2, with no trigonometric argument expansion.
    """
    expression = sp.sympify(expression)
    expression = expression.xreplace({value: sp.Rational(value)
                                      for value in expression.atoms(sp.Float)})
    if expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise ValueError("Library terms must be finite symbolic expressions.")
    expression = expression.replace(
        lambda term: (term.is_Pow and term.base.func == sp.cos
                      and term.exp.is_Integer and term.exp >= 2),
        lambda term: (term.base ** (term.exp % 2)
                      * (1 - sp.sin(term.base.args[0]) ** 2) ** (term.exp // 2)),
    )
    return sp.expand(expression)


def _unchanged(terms):
    return ExactLibraryReduction(
        terms, terms, tuple(range(len(terms))),
        sp.ImmutableMatrix(sp.eye(len(terms))), (),
    )


def _is_cosine_square(term):
    """Prefer sine squares when a Pythagorean identity allows either choice."""
    _, expression = sp.sympify(term).as_coeff_Mul()
    return bool(expression.is_Pow and expression.base.func == sp.cos
                and expression.exp == 2)


def find_library_reduction(terms, mandatory_terms=()):
    """Find only exact constant-coefficient relations in a symbolic library.

    Mandatory terms are considered first.  Other terms keep original priority,
    except cosine squares are considered last, preferring 1 and sine squares
    when both Pythagorean representations are available.
    Redundant mandatory terms may still be removed; their exact retained-term
    support is recorded so the caller can protect that support during pruning.
    This conservative algebraic expansion is not a complete identity prover:
    functions that cannot be related exactly remain separate symbolic atoms.
    No samples, tolerance, ``nsimplify``, or numerical rank decision are used.
    """
    source_terms = tuple(terms)
    terms = tuple(str(term) for term in source_terms)
    if not terms:
        return _unchanged(terms)
    expressions = tuple(_canonical_exact(term) for term in source_terms)
    mandatory = {_canonical_exact(term) for term in mandatory_terms}
    mandatory_indices = tuple(index for index, term in enumerate(expressions)
                              if term in mandatory)
    optional = tuple(index for index in range(len(terms))
                     if index not in mandatory_indices)
    priority = (mandatory_indices
                + tuple(index for index in optional if not _is_cosine_square(source_terms[index]))
                + tuple(index for index in optional if _is_cosine_square(source_terms[index])))

    symbols = tuple(sorted(set().union(*(term.free_symbols for term in expressions)),
                           key=sp.default_sort_key))
    expansions = []
    atoms = set()
    for expression in expressions:
        coefficients = {}
        for summand in sp.Add.make_args(expression):
            coefficient, atom = summand.as_independent(*symbols, as_Add=False)
            if not symbols:
                coefficient, atom = summand, sp.S.One
            coefficients[atom] = coefficients.get(atom, sp.S.Zero) + coefficient
        expansions.append(coefficients)
        atoms.update(coefficients)
    atoms = tuple(sorted(atoms, key=sp.default_sort_key))
    matrix = sp.Matrix([[coefficients.get(atom, sp.S.Zero)
                         for coefficients in expansions] for atom in atoms])

    # Exact arithmetic determines which original columns span the function
    # library.  Reorder only for pivot preference, then restore library order.
    try:
        _, pivots = matrix[:, list(priority)].rref()
        kept = tuple(sorted(priority[pivot] for pivot in pivots))
        if len(kept) == len(terms):
            return _unchanged(terms)
        if kept:
            coordinates, parameters = matrix[:, list(kept)].gauss_jordan_solve(matrix)
            if parameters.rows:
                return _unchanged(terms)
            transform = coordinates.T
        else:
            transform = sp.zeros(len(terms), 0)
        # Every deletion must have a separately checked exact identity.  An
        # undecidable expression produces a no-op, never an approximate drop.
        for index, expression in enumerate(expressions):
            mapped = sum((transform[index, column] * expressions[original_index]
                          for column, original_index in enumerate(kept)), sp.S.Zero)
            if sp.cancel(sp.expand(expression - mapped)) != 0:
                return _unchanged(terms)
        if any(value.free_symbols or value.is_real is not True
               or value.is_finite is not True for value in transform):
            return _unchanged(terms)
    except (ValueError, ZeroDivisionError, NotImplementedError):
        return _unchanged(terms)

    identities = []
    for index, term in enumerate(terms):
        if index in kept:
            continue
        support = tuple(kept[column] for column in range(len(kept))
                        if transform[index, column] != 0)
        identities.append({
            "removed_index": index,
            "term": term,
            "mandatory": index in mandatory_indices,
            "retained_indices": list(support),
            "coefficients": [str(transform[index, kept.index(original_index)])
                             for original_index in support],
            "expression": str(sum((transform[index, column] * sp.sympify(terms[original_index])
                                    for column, original_index in enumerate(kept)), sp.S.Zero)),
        })
    return ExactLibraryReduction(
        terms, tuple(terms[index] for index in kept), kept,
        sp.ImmutableMatrix(transform), tuple(identities),
    )


def map_original_basis(basis, reduction):
    """Map row-wise original-coordinate coefficients to the retained library.

    Input shape is [generators, variable_dim * original_term_count], with one
    contiguous term block per vector-field component.  Output is the same
    Torch/NumPy type, floating dtype, and (for Torch) device.  No rescaling,
    sparsification, orthogonalization, or rank cutoff is performed here.
    """
    if not isinstance(basis, (torch.Tensor, np.ndarray)):
        raise TypeError("The basis must be a Torch tensor or NumPy array.")
    if basis.ndim != 2:
        raise ValueError("The basis must have shape [generators, coefficients].")
    original_count, retained_count = len(reduction.original_terms), len(reduction.theta_terms)
    if original_count == 0 or basis.shape[1] % original_count:
        raise ValueError("The basis columns must contain complete original term blocks.")
    variable_dim = basis.shape[1] // original_count
    if variable_dim == 0:
        raise ValueError("The basis must contain at least one vector-field component.")
    values = [[float(reduction.transform[row, column]) for column in range(retained_count)]
              for row in range(original_count)]
    if isinstance(basis, torch.Tensor):
        dtype = basis.dtype if basis.is_floating_point() or basis.is_complex() else torch.float64
        transform = torch.tensor(values, dtype=dtype, device=basis.device).reshape(
            original_count, retained_count)
        mapped = basis.to(dtype=dtype).reshape(basis.shape[0], variable_dim, original_count) @ transform
    else:
        dtype = basis.dtype if basis.dtype.kind in "fc" else np.float64
        transform = np.asarray(values, dtype=dtype).reshape(original_count, retained_count)
        mapped = basis.astype(dtype, copy=False).reshape(basis.shape[0], variable_dim, original_count) @ transform
    return mapped.reshape(basis.shape[0], variable_dim * retained_count)


@dataclass(frozen=True)
class PreparedGeneratorLibrary:
    result: object
    final_S: torch.Tensor
    final_Vh: object
    reference_state: object
    mandatory_terms: tuple


def replace_result(result, **changes):
    """Copy immutable production results and lightweight diagnostic fixtures."""
    if is_dataclass(result):
        return replace(result, **changes)
    updated = copy(result)
    for name, value in changes.items():
        setattr(updated, name, value)
    return updated


def retain_identity_metadata(source, target):
    """Keep the exact-coordinate-change audit through subsequent pruning."""
    identity = getattr(source, "normalization_metadata", {}).get("identity_reduction")
    if identity is None:
        return target
    metadata = dict(getattr(target, "normalization_metadata", {}))
    metadata["identity_reduction"] = identity
    return replace_result(target, normalization_metadata=metadata)


def prepare_generator_library(
    *, env, result, final_S, final_Vh=None, mandatory_terms=(),
    reference_state=None, subspace_tolerance=DEFAULT_SUBSPACE_TOLERANCE,
):
    """Map a library quotient, extract its independent image, and revalidate.

Without a proven symbolic identity this is a strict no-op. No observations,
models, SVD cutoffs, or retained-column scales are changed. A failed check stops
postprocessing instead of silently discarding a physical generator direction.
    """
    terms = tuple(result.theta_terms)
    mandatory_terms = tuple(str(term) for term in mandatory_terms)
    reduction = find_library_reduction(terms, mandatory_terms=mandatory_terms)
    if reduction.theta_terms == terms:
        return PreparedGeneratorLibrary(result, final_S, final_Vh, reference_state, mandatory_terms)
    if not reduction.theta_terms:
        raise ValueError("Exact identity reduction left an empty library.")

    # Final policy selection need not leave the selected episode in env.state.
    if reference_state is None or tuple(reference_state.theta_terms) != terms:
        source = env.evaluate_library(terms)
        result = source.result
        reference_state = env.state
    old_basis = result.null_basis.detach().to(device="cpu", dtype=torch.float64)
    if old_basis.ndim != 2 or old_basis.shape[1] % len(terms):
        raise ValueError("Identity reduction requires complete coefficient blocks.")
    if old_basis.shape[0] != result.D_final:
        raise ValueError("Identity reduction requires the complete discovered basis.")
    variable_dim = old_basis.shape[1] // len(terms)
    old_scales = validate_coefficient_scales(result.feature_scales, len(terms), variable_dim, "cpu")
    new_scales = restrict_library_scales(
        old_scales, reduction.kept_indices, num_terms=len(terms), variable_dim=variable_dim,
    )
    original_basis = restore_coefficient_basis(old_basis, old_scales)
    mapped = map_original_basis(original_basis, reduction).to(dtype=torch.float64)
    if not bool(torch.isfinite(mapped).all()):
        raise ValueError("Exact coefficient mapping produced non-finite values.")

    # Rank must be extracted BEFORE multiplying by the retained column scales:
    # that multiplication can amplify cancellation roundoff into false vectors.
    _, values, vh = torch.linalg.svd(mapped, full_matrices=False)
    tolerance = (max(mapped.shape) * torch.finfo(torch.float64).eps * float(values[0])
                 if values.numel() else 0.0)
    dimension = int((values > tolerance).sum())
    removed = old_basis.shape[0] - dimension
    algebraic_redundancy = variable_dim * (len(terms) - len(reduction.theta_terms))
    if removed < 0 or removed > algebraic_redundancy:
        raise ValueError("Numerical rank loss exceeds the proven library identity redundancy.")
    raw_basis = vh[:dimension]
    projection_error = (float(torch.linalg.matrix_norm(mapped - (mapped @ raw_basis.T) @ raw_basis, ord=2))
                        / float(values[0]) if values.numel() and float(values[0]) > 0 else 0.0)
    normalized_columns = (raw_basis * new_scales).T
    normalized_basis = torch.linalg.qr(normalized_columns, mode="reduced").Q.T
    if not bool(torch.isfinite(normalized_basis).all()):
        raise ValueError("Identity-reduced basis normalization produced non-finite values.")

    final = env.evaluate_library(
        reduction.theta_terms, feature_scales=new_scales, reference_state=reference_state,
    )
    if int(final.result.D_final) != dimension:
        raise ValueError(
            f"Identity-reduced system selected {final.result.D_final} directions, "
            f"but the mapped generator space has {dimension}; refusing to change its span."
        )
    recomputed = restored_subspace_basis(
        final.result.null_basis, final.result.feature_scales,
        reduction.theta_terms, reduction.theta_terms,
    )
    subspace_error = projector_subspace_distance(raw_basis.T, recomputed)
    if subspace_error > float(subspace_tolerance):
        raise ValueError(f"Identity-reduced recomputation changed the generator subspace ({subspace_error:.6g}).")

    mapped_checks = {}
    rank_reference = getattr(reference_state, "rank_reference", None)
    if rank_reference is not None:
        indices = [block * len(terms) + j for block in range(variable_dim)
                   for j in reduction.kept_indices]
        q = normalized_basis.T.to(reference_state.device)

        def relative_residual(sample_indices):
            if q.shape[1] == 0:
                return 0.0
            matrix = torch.cat([reference_state.c_rows[i][:, indices] for i in sample_indices], dim=0)
            denominator = float(torch.linalg.matrix_norm(matrix, ord=2))
            return float(torch.linalg.matrix_norm(matrix @ q, ord=2)) / denominator if denominator else 0.0

        mapped_checks = {
            "scale_residual": relative_residual(rank_reference.scale_indices),
            "validation_residual": relative_residual(rank_reference.validation_indices),
        }
        tolerance_r = reference_state.rank_config.residual_tolerance
        if any(value > tolerance_r for value in mapped_checks.values()):
            raise ValueError(f"Mapped generator basis failed the residual checks: {mapped_checks}")
        mapped_checks["residual_tolerance"] = tolerance_r

    # A mandatory function may itself be redundant. Protect every retained term
    # needed to represent it exactly during the subsequent ordinary pruning.
    required = set(mandatory_terms)
    for i, term in enumerate(terms):
        if term in mandatory_terms:
            required.update(reduction.theta_terms[j] for j in range(len(reduction.theta_terms))
                            if reduction.transform[i, j] != 0)
    required = tuple(term for term in reduction.theta_terms if term in required)
    metadata = dict(final.result.normalization_metadata)
    metadata["identity_reduction"] = {
        "method": "exact_symbolic_library_quotient",
        "original_theta_terms": list(terms),
        "reduced_theta_terms": list(reduction.theta_terms),
        "kept_indices": list(reduction.kept_indices),
        "identities": list(reduction.identities),
        "transform": [[str(reduction.transform[i, j]) for j in range(len(reduction.theta_terms))]
                      for i in range(len(terms))],
        "original_coefficient_nullity": int(old_basis.shape[0]),
        "independent_generator_dimension": dimension,
        "removed_representation_directions": removed,
        "library_identity_count": len(terms) - len(reduction.theta_terms),
        "mapped_original_singular_values": values.tolist(),
        "rank_tolerance": tolerance,
        "mapping_relative_projection_error": projection_error,
        "recomputed_subspace_error": subspace_error,
        "subspace_tolerance": float(subspace_tolerance),
        "mapped_basis_checks": mapped_checks,
        "source_rank_selection": getattr(result, "normalization_metadata", {}).get("rank_selection"),
    }
    reduced_result = replace_result(
        final.result, null_basis=normalized_basis, normalization_metadata=metadata,
        basis_source="identity_mapped_original_basis",
    )
    print(
        f"exact_identity_reduction: terms={len(terms)}->{len(reduction.theta_terms)}, "
        f"coefficient_nullity={old_basis.shape[0]}, independent_generators={dimension}, "
        f"removed_representation_directions={removed}, subspace_error={subspace_error:.6g}",
        flush=True,
    )
    return PreparedGeneratorLibrary(reduced_result, final.final_S, final.final_Vh, reference_state, required)
