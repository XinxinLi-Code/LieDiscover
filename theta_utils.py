"""Utilities for turning generated symbolic trees into Theta terms.

The RL environment should not have to know about DSO prefix syntax, SymPy
canonicalization, or duplicate handling. This module is the small adapter layer:

    decoder tree / prefix tokens -> discovery.py expression string -> stable key

By default, duplicate keys are scale invariant and algebraically simplified. For
example, ``t*(t**2)``, ``t**3`` and ``t*t*t`` share the same key.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import sympy as sp
import torch
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


TokenLike = Union[int, str]
TreeLike = object
TermLike = Union[str, Sequence[TokenLike], TreeLike]

DEFAULT_VARIABLE_NAMES = ("t", "x", "y", "u", "v")

# Keep only deterministic symbolic preprocessing here. Neural encoder outputs and
# token-id tensors still depend on the current model/token map and are not cached.
_SYMBOLIC_CACHE_SIZE = 4096

_PARSER_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_BINARY_TOKENS = {
    "add": "+",
    "+": "+",
    "sub": "-",
    "-": "-",
    "mul": "*",
    "*": "*",
    "div": "/",
    "/": "/",
}

_UNARY_FUNCTION_TOKENS = {
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sqrt",
}

_POWER_TOKENS = {
    "n2": 2,
    "n3": 3,
    "n4": 4,
}

_POWER_VALUES_TO_TOKENS = {value: token for token, value in _POWER_TOKENS.items()}

_UNARY_SPECIAL_TOKENS = set(_UNARY_FUNCTION_TOKENS) | set(_POWER_TOKENS) | {
    "neg",
    "inv",
}

_ALLOWED_SYMPY_FUNCTIONS = {
    sp.sin,
    sp.cos,
    sp.tan,
    sp.exp,
    sp.log,
}


class ThetaTermError(ValueError):
    """Raised when a generated term cannot be used as a discovery theta term."""


def _library_token_names(library) -> Optional[set]:
    if library is None:
        return None
    names = getattr(library, "names", None)
    if names is None:
        return None
    return {str(name) for name in names}


def _library_token_signature(library) -> Optional[Tuple[str, ...]]:
    """Snapshot the vocabulary by value, including its order and later changes."""

    if library is None:
        return None
    names = getattr(library, "names", None)
    return None if names is None else tuple(str(name) for name in names)


def _has_token(token: str, available_tokens: Optional[set]) -> bool:
    return available_tokens is None or token in available_tokens


@dataclass(frozen=True)
class NormalizedThetaTerm:
    """Stable representation used by the RL environment."""

    expression: str
    canonical: str
    key: str
    complexity: int

    @property
    def is_zero(self) -> bool:
        return self.key == "0"


def split_theta_terms(theta_terms: Union[str, Iterable[str]]) -> List[str]:
    """Split a comma-separated theta string while respecting parentheses."""

    if not isinstance(theta_terms, str):
        return [str(term).strip() for term in theta_terms if str(term).strip()]

    text = theta_terms.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    parts: List[str] = []
    start = 0
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            part = text[start:idx].strip()
            if part:
                parts.append(part)
            start = idx + 1

    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _is_numeric_token(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def _library_token_name(library, token_id: int) -> str:
    try:
        return str(library[int(token_id)].name)
    except Exception as exc:
        raise ThetaTermError(f"Cannot resolve token id {token_id!r} from DSO library.") from exc


def _token_arity(name: str, variable_names: Sequence[str]) -> int:
    if name in _BINARY_TOKENS:
        return 2
    if name in _UNARY_SPECIAL_TOKENS:
        return 1
    if name in variable_names or _is_numeric_token(name) or name in {"pi", "E"}:
        return 0
    raise ThetaTermError(
        f"Unsupported token {name!r}. Add it to theta_utils.py and discovery.py "
        "before letting the decoder sample it."
    )


def _compose_expression(name: str, child_exprs: Sequence[str]) -> str:
    arity = len(child_exprs)

    if arity == 0:
        return name

    if arity == 1:
        child = child_exprs[0]
        if name == "neg":
            return f"(-({child}))"
        if name == "inv":
            return f"(1/({child}))"
        if name in _POWER_TOKENS:
            return f"(({child})**{_POWER_TOKENS[name]})"
        if name in _UNARY_FUNCTION_TOKENS:
            return f"{name}({child})"
        raise ThetaTermError(f"Unsupported unary token {name!r}.")

    if arity == 2:
        if name not in _BINARY_TOKENS:
            raise ThetaTermError(f"Unsupported binary token {name!r}.")
        left, right = child_exprs
        op = _BINARY_TOKENS[name]
        return f"(({left}) {op} ({right}))"

    raise ThetaTermError(f"Unsupported arity {arity} for token {name!r}.")


def tree_to_expression(tree: TreeLike) -> str:
    """Convert decoder.SymbolicTree-like objects to an infix expression string."""

    if tree is None or not hasattr(tree, "name") or not hasattr(tree, "children"):
        raise ThetaTermError("tree_to_expression expects a decoder.SymbolicTree-like object.")

    name = str(getattr(tree, "name"))
    children = tuple(getattr(tree, "children"))
    child_exprs = [tree_to_expression(child) for child in children]
    return _compose_expression(name, child_exprs)


def prefix_to_expression(
    prefix: Sequence[TokenLike],
    *,
    library=None,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
) -> str:
    """Convert a DSO prefix traversal into a discovery.py expression string."""

    names: List[str] = []
    for token in prefix:
        if isinstance(token, str):
            name = token.strip()
            if not name:
                continue
            names.append(name)
        else:
            if library is None:
                raise ThetaTermError("Integer token ids require a DSO library.")
            names.append(_library_token_name(library, int(token)))

    def parse_at(pos: int) -> Tuple[str, int]:
        if pos >= len(names):
            raise ThetaTermError(f"Prefix ended before the tree closed: {names}")
        name = names[pos]
        arity = _token_arity(name, variable_names)
        child_exprs: List[str] = []
        next_pos = pos + 1
        for _ in range(arity):
            child_expr, next_pos = parse_at(next_pos)
            child_exprs.append(child_expr)
        return _compose_expression(name, child_exprs), next_pos

    expression, final_pos = parse_at(0)
    if final_pos != len(names):
        raise ThetaTermError(f"Prefix has extra tokens after a closed tree: {names}")
    return expression


def term_to_expression(
    term: TermLike,
    *,
    library=None,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
) -> str:
    """Accept a decoder tree, prefix token sequence, or expression string."""

    if isinstance(term, NormalizedThetaTerm):
        return term.expression
    if isinstance(term, str):
        return term.strip()
    if hasattr(term, "name") and hasattr(term, "children"):
        return tree_to_expression(term)
    if hasattr(term, "detach") and hasattr(term, "cpu"):
        values = term.detach().cpu().view(-1).tolist()
        return prefix_to_expression(values, library=library, variable_names=variable_names)
    if hasattr(term, "reshape") and hasattr(term, "tolist"):
        values = term.reshape(-1).tolist()
        return prefix_to_expression(values, library=library, variable_names=variable_names)
    if isinstance(term, Sequence):
        return prefix_to_expression(term, library=library, variable_names=variable_names)
    raise ThetaTermError(f"Unsupported theta term input type: {type(term)!r}")


def _local_dict(variable_names: Sequence[str]) -> dict:
    symbols = {name: sp.Symbol(name) for name in variable_names}
    symbols.update(
        {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "pi": sp.pi,
            "E": sp.E,
        }
    )
    return symbols


def parse_theta_expression(
    expression: str,
    *,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
    evaluate: bool = True,
) -> sp.Expr:
    """Parse one expression using the same broad syntax as discovery.py."""

    try:
        return parse_expr(
            expression,
            local_dict=_local_dict(variable_names),
            transformations=_PARSER_TRANSFORMS,
            evaluate=evaluate,
        )
    except Exception as exc:
        raise ThetaTermError(f"Cannot parse theta expression {expression!r}.") from exc


def _validate_supported_sympy(expr: sp.Expr) -> None:
    for node in sp.preorder_traversal(expr):
        if node.is_Function and node.func not in _ALLOWED_SYMPY_FUNCTIONS:
            raise ThetaTermError(
                f"Expression {expr!s} contains unsupported function {node.func!s}. "
                "discovery.py currently supports sin/cos/tan/exp/log/sqrt and algebraic powers."
            )


def _expr_to_string(expr: sp.Expr) -> str:
    return sp.sstr(expr).replace(" ", "")


def canonicalize_expression(
    expression: str,
    *,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
    rationalize_constants: bool = True,
) -> str:
    """Return a stable discovery.py expression string."""

    return _canonicalize_expression_cached(
        expression, tuple(variable_names), rationalize_constants
    )


@lru_cache(maxsize=_SYMBOLIC_CACHE_SIZE)
def _canonicalize_expression_cached(
    expression: str,
    variable_names: Tuple[str, ...],
    rationalize_constants: bool,
) -> str:

    expr = parse_theta_expression(expression, variable_names=variable_names)
    if rationalize_constants:
        expr = sp.nsimplify(expr, rational=True)
    expr = sp.simplify(expr)
    _validate_supported_sympy(expr)
    return _expr_to_string(expr)


def canonical_key(
    expression: str,
    *,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
    scale_invariant: bool = True,
) -> str:
    """Return a duplicate-detection key.

    SymPy simplification catches structural duplicates such as ``t*(t**2)`` and
    ``t**3``. When ``scale_invariant`` is true, nonzero constant multiples also
    share the same key.
    """

    return _canonical_key_cached(expression, tuple(variable_names), scale_invariant)


@lru_cache(maxsize=_SYMBOLIC_CACHE_SIZE)
def _canonical_key_cached(
    expression: str,
    variable_names: Tuple[str, ...],
    scale_invariant: bool,
) -> str:

    expr = parse_theta_expression(expression, variable_names=variable_names)
    expr = sp.simplify(sp.nsimplify(expr, rational=True))
    if scale_invariant and (expr.is_number or expr in (sp.pi, sp.E)) and expr != 0:
        expr = sp.Integer(1)
    if scale_invariant and expr != 0:
        factored = sp.factor(expr)
        coeff, rest = factored.as_coeff_Mul()
        if coeff != 0:
            expr = sp.simplify(rest)
    _validate_supported_sympy(expr)
    return _expr_to_string(expr)


def _strip_numeric_factors(expr: sp.Expr, *, preserve_sign: bool = False) -> sp.Expr:
    """Remove numeric multiplicative factors for structural prefix fallbacks."""

    expr = sp.sympify(expr)
    if expr.is_Number:
        if expr == 0:
            return sp.Integer(0)
        if preserve_sign and expr < 0:
            return -sp.Integer(1)
        return sp.Integer(1)
    if expr.is_Add:
        return sp.Add(*[_strip_numeric_factors(arg, preserve_sign=True) for arg in expr.args])
    if expr.is_Mul:
        coeff, rest = expr.as_coeff_Mul()
        if rest != expr:
            stripped = _strip_numeric_factors(rest, preserve_sign=False)
            if preserve_sign and coeff < 0:
                return -stripped
            return stripped

        stripped_factors = []
        negative = False
        for factor in expr.args:
            if factor.is_number:
                if preserve_sign and factor < 0:
                    negative = not negative
                continue
            stripped_factors.append(_strip_numeric_factors(factor, preserve_sign=False))
        stripped = sp.Mul(*stripped_factors) if stripped_factors else sp.Integer(1)
        if negative:
            return -stripped
        if preserve_sign and coeff < 0:
            return -stripped
        return stripped
    if expr.is_Pow:
        base, exponent = expr.as_base_exp()
        if exponent.is_Integer:
            return sp.Pow(_strip_numeric_factors(base, preserve_sign=False), exponent)
        return expr
    return expr


def _structural_prefix_fallbacks(expression: str, *, variable_names: Sequence[str]) -> List[str]:
    """Return scale-invariant expressions that may be easier for the decoder to encode."""

    expr = parse_theta_expression(expression, variable_names=variable_names)
    candidates = []
    for candidate in (
        sp.expand_trig(expr),
        sp.expand_trig(sp.expand(expr)),
    ):
        stripped = sp.expand(_strip_numeric_factors(candidate))
        if stripped != expr:
            candidates.append(_expr_to_string(stripped))
    out = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            out.append(candidate)
            seen.add(candidate)
    return out


def _expression_tree_node_count(expr: sp.Expr) -> int:
    if expr.is_Atom:
        return 1

    if expr.is_Pow:
        base, exponent = expr.as_base_exp()
        if exponent.is_number:
            return 1 + _expression_tree_node_count(base)
        return 1 + _expression_tree_node_count(base) + _expression_tree_node_count(exponent)

    if expr.is_Add or expr.is_Mul:
        return max(len(expr.args) - 1, 1) + sum(
            _expression_tree_node_count(arg)
            for arg in expr.args
        )

    return 1 + sum(_expression_tree_node_count(arg) for arg in expr.args)


def expression_complexity(
    expression: str,
    *,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
) -> int:
    """Count expression tree nodes for RL length penalties.

    This mirrors decoder token length more closely than string length:
    ``t**2`` has complexity 2, ``t*x`` has complexity 3, and
    ``t*(t**2)`` has complexity 4.
    """

    return _expression_complexity_cached(expression, tuple(variable_names))


@lru_cache(maxsize=_SYMBOLIC_CACHE_SIZE)
def _expression_complexity_cached(
    expression: str, variable_names: Tuple[str, ...]
) -> int:

    expr = parse_theta_expression(
        expression,
        variable_names=variable_names,
        evaluate=False,
    )
    return int(_expression_tree_node_count(expr))


def normalize_theta_term(
    term: TermLike,
    *,
    library=None,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
    scale_invariant: bool = True,
) -> NormalizedThetaTerm:
    """Convert any supported term representation into a stable theta record."""

    expression = term_to_expression(term, library=library, variable_names=variable_names)
    return _normalize_expression_cached(expression, tuple(variable_names), scale_invariant)


@lru_cache(maxsize=_SYMBOLIC_CACHE_SIZE)
def _normalize_expression_cached(
    expression: str,
    variable_names: Tuple[str, ...],
    scale_invariant: bool,
) -> NormalizedThetaTerm:
    canonical = canonicalize_expression(expression, variable_names=variable_names)
    key = canonical_key(
        canonical,
        variable_names=variable_names,
        scale_invariant=scale_invariant,
    )
    complexity = expression_complexity(expression, variable_names=variable_names)
    return NormalizedThetaTerm(
        expression=expression,
        canonical=canonical,
        key=key,
        complexity=complexity,
    )


def theta_terms_to_string(
    terms: Iterable[Union[str, NormalizedThetaTerm]],
    *,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
) -> str:
    """Create the comma-separated theta string expected by discovery.py."""

    out = []
    for term in terms:
        if isinstance(term, NormalizedThetaTerm):
            out.append(term.canonical)
        else:
            out.append(canonicalize_expression(str(term), variable_names=variable_names))
    return ",".join(out)


def _flatten_left_associative(
    token: str,
    args: Sequence[sp.Expr],
    *,
    available_tokens: Optional[set] = None,
) -> List[str]:
    if not args:
        raise ThetaTermError(f"Cannot encode empty {token!r} expression.")
    if not _has_token(token, available_tokens):
        raise ThetaTermError(f"Token {token!r} is not available in the decoder library.")
    prefix = sympy_expr_to_prefix(args[0], available_tokens=available_tokens)
    for arg in args[1:]:
        prefix = [token] + prefix + sympy_expr_to_prefix(arg, available_tokens=available_tokens)
    return prefix


def _encode_integer_power(
    base: sp.Expr,
    exponent: int,
    *,
    available_tokens: Optional[set],
) -> List[str]:
    if exponent == 1:
        return sympy_expr_to_prefix(base, available_tokens=available_tokens)

    direct_token = _POWER_VALUES_TO_TOKENS.get(int(exponent))
    if direct_token is not None and _has_token(direct_token, available_tokens):
        return [direct_token] + sympy_expr_to_prefix(base, available_tokens=available_tokens)

    available_power_tokens = [
        (power, token)
        for token, power in _POWER_TOKENS.items()
        if power > 1 and power < exponent and _has_token(token, available_tokens)
    ]
    for power, token in sorted(available_power_tokens, reverse=True):
        if exponent % power == 0:
            return [token] + _encode_integer_power(
                base,
                exponent // power,
                available_tokens=available_tokens,
            )

    if _has_token("mul", available_tokens):
        return _flatten_left_associative(
            "mul",
            [base] * int(exponent),
            available_tokens=available_tokens,
        )

    raise ThetaTermError(
        f"Cannot encode power {base!s}**{exponent} with available decoder tokens."
    )


def sympy_expr_to_prefix(
    expr: sp.Expr,
    *,
    available_tokens: Optional[set] = None,
) -> List[str]:
    """Convert a supported SymPy expression into DSO prefix token names.

    The output uses token names expected by ``decoder.make_dso_library``. When
    ``available_tokens`` is provided, powers are decomposed into tokens that
    the current decoder library can actually emit. For example, ``x**4``
    becomes ``n2 -> n2 -> x`` if ``n4`` is absent but ``n2`` is available.
    """

    expr = sp.sympify(expr)

    if expr.is_Symbol:
        return [str(expr)]

    if expr.is_Number:
        if expr == 0:
            raise ThetaTermError("Zero constants should not be encoded as theta terms.")
        if expr == 1:
            if not _has_token("1", available_tokens):
                raise ThetaTermError("Token '1' is not available in the decoder library.")
            return ["1"]
        if expr == -1:
            if not _has_token("neg", available_tokens):
                raise ThetaTermError("Token 'neg' is not available in the decoder library.")
            return ["neg", "1"]
        if expr.is_Integer:
            token = str(int(expr))
            if _has_token(token, available_tokens):
                return [token]
            raise ThetaTermError(f"Integer token {token!r} is not available in the decoder library.")
        value = float(expr)
        token = str(int(value)) if value.is_integer() else str(value)
        if _has_token(token, available_tokens):
            return [token]
        raise ThetaTermError(f"Constant token {token!r} is not available in the decoder library.")

    if expr.is_Add:
        terms = list(expr.as_ordered_terms())
        if not terms:
            raise ThetaTermError("Cannot encode an empty Add expression.")
        positive_terms = []
        negative_terms = []
        for term in terms:
            if term.could_extract_minus_sign():
                negative_terms.append(sp.simplify(-term))
            else:
                positive_terms.append(term)

        if not positive_terms:
            if not _has_token("neg", available_tokens):
                raise ThetaTermError("Token 'neg' is not available in the decoder library.")
            positive_sum = sp.simplify(sp.Add(*negative_terms))
            return ["neg"] + sympy_expr_to_prefix(
                positive_sum,
                available_tokens=available_tokens,
            )

        prefix = sympy_expr_to_prefix(positive_terms[0], available_tokens=available_tokens)
        for term in positive_terms[1:]:
            if not _has_token("add", available_tokens):
                raise ThetaTermError("Token 'add' is not available in the decoder library.")
            prefix = ["add"] + prefix + sympy_expr_to_prefix(
                term,
                available_tokens=available_tokens,
            )
        for term in negative_terms:
            if not _has_token("sub", available_tokens):
                raise ThetaTermError("Token 'sub' is not available in the decoder library.")
            prefix = ["sub"] + prefix + sympy_expr_to_prefix(
                term,
                available_tokens=available_tokens,
            )
        return prefix

    if expr.is_Mul:
        numerator, denominator = sp.fraction(expr)
        if denominator != 1:
            if _has_token("div", available_tokens):
                return (
                    ["div"]
                    + sympy_expr_to_prefix(numerator, available_tokens=available_tokens)
                    + sympy_expr_to_prefix(denominator, available_tokens=available_tokens)
                )
            if _has_token("inv", available_tokens):
                inv_denominator = ["inv"] + sympy_expr_to_prefix(
                    denominator,
                    available_tokens=available_tokens,
                )
                if numerator == 1:
                    return inv_denominator
                if numerator == -1 and _has_token("neg", available_tokens):
                    return ["neg"] + inv_denominator
                if not _has_token("mul", available_tokens):
                    raise ThetaTermError(
                        "Token 'div' is unavailable, and encoding a/b as a*inv(b) "
                        "also requires token 'mul'."
                    )
                return (
                    ["mul"]
                    + sympy_expr_to_prefix(numerator, available_tokens=available_tokens)
                    + inv_denominator
                )
            raise ThetaTermError("Neither 'div' nor 'inv' is available in the decoder library.")

        coeff, rest = expr.as_coeff_Mul()
        if coeff == -1:
            if not _has_token("neg", available_tokens):
                raise ThetaTermError("Token 'neg' is not available in the decoder library.")
            return ["neg"] + sympy_expr_to_prefix(rest, available_tokens=available_tokens)

        factors = list(expr.as_ordered_factors())
        return _flatten_left_associative("mul", factors, available_tokens=available_tokens)

    if expr.is_Pow:
        base, exponent = expr.as_base_exp()
        if exponent.is_Integer and exponent > 0:
            return _encode_integer_power(
                base,
                int(exponent),
                available_tokens=available_tokens,
            )
        if exponent.is_Integer and exponent < 0:
            positive_power = _encode_integer_power(
                base,
                int(-exponent),
                available_tokens=available_tokens,
            )
            if _has_token("inv", available_tokens):
                return ["inv"] + positive_power
            if _has_token("div", available_tokens):
                return ["div", "1"] + positive_power
            raise ThetaTermError("Neither 'inv' nor 'div' is available in the decoder library.")
        if exponent == -1:
            if _has_token("inv", available_tokens):
                return ["inv"] + sympy_expr_to_prefix(base, available_tokens=available_tokens)
            if _has_token("div", available_tokens):
                return ["div", "1"] + sympy_expr_to_prefix(base, available_tokens=available_tokens)
            raise ThetaTermError("Neither 'inv' nor 'div' is available in the decoder library.")
        raise ThetaTermError(
            f"Unsupported power {expr!s}. Add n{exponent} to the decoder library "
            "or restrict generated powers to n2/n3/n4."
        )

    if expr.is_Function:
        name = expr.func.__name__
        if name not in _UNARY_FUNCTION_TOKENS:
            raise ThetaTermError(f"Unsupported function {name!r}.")
        if not _has_token(name, available_tokens):
            raise ThetaTermError(f"Token {name!r} is not available in the decoder library.")
        if len(expr.args) != 1:
            raise ThetaTermError(f"Function {name!r} must have arity 1.")
        if name == "exp":
            if expr.args[0].is_Add and _has_token("mul", available_tokens):
                factor_prefixes = []
                for term in expr.args[0].as_ordered_terms():
                    if term.could_extract_minus_sign():
                        positive = sp.simplify(-term)
                        positive_prefix = sympy_expr_to_prefix(
                            sp.exp(positive),
                            available_tokens=available_tokens,
                        )
                        if _has_token("inv", available_tokens):
                            factor_prefixes.append(["inv"] + positive_prefix)
                        elif _has_token("div", available_tokens):
                            factor_prefixes.append(["div", "1"] + positive_prefix)
                        else:
                            raise ThetaTermError(
                                "Neither 'inv' nor 'div' is available in the decoder library."
                            )
                    else:
                        factor_prefixes.append(
                            sympy_expr_to_prefix(
                                sp.exp(term),
                                available_tokens=available_tokens,
                            )
                        )
                prefix = factor_prefixes[0]
                for factor_prefix in factor_prefixes[1:]:
                    prefix = ["mul"] + prefix + factor_prefix
                return prefix
            coeff, rest = expr.args[0].as_coeff_Mul()
            if coeff.is_Integer and coeff != 1 and rest != 1:
                positive_power = _encode_integer_power(
                    sp.exp(rest),
                    abs(int(coeff)),
                    available_tokens=available_tokens,
                )
                if coeff > 0:
                    return positive_power
                if _has_token("inv", available_tokens):
                    return ["inv"] + positive_power
                if _has_token("div", available_tokens):
                    return ["div", "1"] + positive_power
                raise ThetaTermError("Neither 'inv' nor 'div' is available in the decoder library.")
        return [name] + sympy_expr_to_prefix(expr.args[0], available_tokens=available_tokens)

    raise ThetaTermError(f"Cannot encode expression {expr!s} as DSO prefix tokens.")


def expression_to_prefix(
    expression: str,
    *,
    library=None,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
    canonicalize: bool = True,
) -> List[str]:
    """Convert an expression string into DSO prefix token names."""

    return list(_expression_to_prefix_cached(
        expression, tuple(variable_names), _library_token_signature(library), canonicalize
    ))


@lru_cache(maxsize=_SYMBOLIC_CACHE_SIZE)
def _expression_to_prefix_cached(
    expression: str,
    variable_names: Tuple[str, ...],
    token_signature: Optional[Tuple[str, ...]],
    canonicalize: bool,
) -> Tuple[str, ...]:

    if canonicalize:
        expression = canonicalize_expression(expression, variable_names=variable_names)
    expr = parse_theta_expression(expression, variable_names=variable_names)
    return tuple(sympy_expr_to_prefix(
        expr,
        available_tokens=None if token_signature is None else set(token_signature),
    ))


def theta_term_to_prefix(
    term: Union[TermLike, NormalizedThetaTerm],
    *,
    library=None,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
) -> List[str]:
    """Convert any supported theta term into DSO prefix token names."""

    normalized = (
        term
        if isinstance(term, NormalizedThetaTerm)
        else normalize_theta_term(term, library=library, variable_names=variable_names)
    )
    # Cache immutable prefixes, then return a fresh list to callers that may pad
    # or otherwise mutate traversals. The forbidden set and encoder share this.
    return list(_theta_term_to_prefix_cached(
        normalized.canonical,
        normalized.key,
        tuple(variable_names),
        _library_token_signature(library),
    ))


@lru_cache(maxsize=_SYMBOLIC_CACHE_SIZE)
def _theta_term_to_prefix_cached(
    canonical: str,
    key: str,
    variable_names: Tuple[str, ...],
    token_signature: Optional[Tuple[str, ...]],
) -> Tuple[str, ...]:
    try:
        return _expression_to_prefix_cached(
            canonical, variable_names, token_signature, False
        )
    except ThetaTermError:
        candidates = []
        if key != canonical:
            candidates.append(key)
        candidates.extend(
            _structural_prefix_fallbacks(
                canonical,
                variable_names=variable_names,
            )
        )
        if key != canonical:
            candidates.extend(
                _structural_prefix_fallbacks(
                    key,
                    variable_names=variable_names,
                )
            )
        last_error = None
        for candidate in candidates:
            try:
                return _expression_to_prefix_cached(
                    candidate, variable_names, token_signature, False
                )
            except ThetaTermError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise


def symbolic_cache_info() -> dict:
    """Return bounded symbolic cache counters without retaining input objects."""

    return {
        name: function.cache_info()._asdict()
        for name, function in _symbolic_cached_functions()
    }


def clear_symbolic_caches() -> None:
    """Clear symbolic preprocessing entries and counters, e.g. for a benchmark."""

    for _, function in _symbolic_cached_functions():
        function.cache_clear()


def _symbolic_cached_functions():
    return (
        ("canonical", _canonicalize_expression_cached),
        ("key", _canonical_key_cached),
        ("complexity", _expression_complexity_cached),
        ("normalized", _normalize_expression_cached),
        ("expression_prefix", _expression_to_prefix_cached),
        ("term_prefix", _theta_term_to_prefix_cached),
    )


def theta_terms_to_prefixes(
    terms: Iterable[Union[TermLike, NormalizedThetaTerm]],
    *,
    library=None,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
    skip_unencodable: bool = False,
) -> List[List[str]]:
    """Convert a batch of terms into DSO prefix token-name sequences."""

    prefixes = []
    for term in terms:
        try:
            prefixes.append(
                theta_term_to_prefix(term, library=library, variable_names=variable_names)
            )
        except ThetaTermError:
            if not skip_unencodable:
                raise
    return prefixes


def theta_terms_to_encoder_tensor(
    terms: Iterable[Union[TermLike, NormalizedThetaTerm]],
    token_map,
    *,
    library=None,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
    pad_to: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert theta terms into a padded encoder-id tensor.

    ``token_map`` should be ``decoder.EncoderDSOTokenMap`` or any object with
    a compatible ``dso_batch_to_encoder`` method.
    """

    prefixes = theta_terms_to_prefixes(
        terms,
        library=library,
        variable_names=variable_names,
    )
    if not prefixes:
        if pad_to is None:
            pad_to = 1
        return torch.empty((0, pad_to), dtype=torch.long, device=device)
    return token_map.dso_batch_to_encoder(prefixes, pad_to=pad_to, device=device)


def theta_sets_to_encoder_inputs(
    theta_acc: Iterable[Union[TermLike, NormalizedThetaTerm]],
    theta_rej: Iterable[Union[TermLike, NormalizedThetaTerm]],
    token_map,
    *,
    library=None,
    variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
    pad_to: Optional[int] = None,
    device: Optional[torch.device] = None,
    add_batch_dim: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode ``Theta_acc`` and ``Theta_rej`` for ``DualSetEncoder``.

    Returns ``acc_tensor, acc_mask, rej_tensor, rej_mask``. Empty sets are
    represented by one all-PAD row with mask 0 so the BiLSTM never receives a
    zero-sized batch. With ``add_batch_dim=True`` the tensor shapes are:

        acc_tensor: [1, max(n_acc, 1), seq_len]
        acc_mask:   [1, max(n_acc, 1)]
        rej_tensor: [1, max(n_rej, 1), seq_len]
        rej_mask:   [1, max(n_rej, 1)]
    """

    acc_terms = list(theta_acc)
    rej_terms = list(theta_rej)
    acc_prefixes = theta_terms_to_prefixes(
        acc_terms,
        library=library,
        variable_names=variable_names,
        skip_unencodable=True,
    )
    rej_prefixes = theta_terms_to_prefixes(
        rej_terms,
        library=library,
        variable_names=variable_names,
        skip_unencodable=True,
    )

    if pad_to is None:
        lengths = [len(seq) for seq in acc_prefixes + rej_prefixes]
        pad_to = max(lengths) if lengths else 1

    acc_tensor = (
        token_map.dso_batch_to_encoder(acc_prefixes, pad_to=pad_to, device=device)
        if acc_prefixes
        else torch.zeros((1, pad_to), dtype=torch.long, device=device)
    )
    rej_tensor = (
        token_map.dso_batch_to_encoder(rej_prefixes, pad_to=pad_to, device=device)
        if rej_prefixes
        else torch.zeros((1, pad_to), dtype=torch.long, device=device)
    )
    acc_mask = (
        torch.ones(acc_tensor.shape[0], dtype=torch.float32, device=device)
        if acc_prefixes
        else torch.zeros(acc_tensor.shape[0], dtype=torch.float32, device=device)
    )
    rej_mask = (
        torch.ones(rej_tensor.shape[0], dtype=torch.float32, device=device)
        if rej_prefixes
        else torch.zeros(rej_tensor.shape[0], dtype=torch.float32, device=device)
    )

    if add_batch_dim:
        acc_tensor = acc_tensor.unsqueeze(0)
        rej_tensor = rej_tensor.unsqueeze(0)
        acc_mask = acc_mask.unsqueeze(0)
        rej_mask = rej_mask.unsqueeze(0)

    return acc_tensor, acc_mask, rej_tensor, rej_mask


class ThetaRegistry:
    """Track accepted/rejected theta terms with canonical duplicate detection."""

    def __init__(
        self,
        theta_acc: Optional[Iterable[TermLike]] = None,
        theta_rej: Optional[Iterable[TermLike]] = None,
        *,
        library=None,
        variable_names: Sequence[str] = DEFAULT_VARIABLE_NAMES,
        scale_invariant: bool = True,
    ):
        self.library = library
        self.variable_names = tuple(variable_names)
        self.scale_invariant = bool(scale_invariant)
        self.acc: List[NormalizedThetaTerm] = []
        self.rej: List[NormalizedThetaTerm] = []
        self._acc_keys = set()
        self._rej_keys = set()

        for term in theta_acc or ():
            self.add_acc(term, allow_duplicate=False)
        for term in theta_rej or ():
            self.add_rej(term, allow_duplicate=False)

    def normalize(self, term: TermLike) -> NormalizedThetaTerm:
        return normalize_theta_term(
            term,
            library=self.library,
            variable_names=self.variable_names,
            scale_invariant=self.scale_invariant,
        )

    def duplicate_location(self, term: Union[TermLike, NormalizedThetaTerm]) -> Optional[str]:
        normalized = term if isinstance(term, NormalizedThetaTerm) else self.normalize(term)
        if normalized.key in self._acc_keys:
            return "acc"
        if normalized.key in self._rej_keys:
            return "rej"
        return None

    def add_acc(
        self,
        term: TermLike,
        *,
        allow_duplicate: bool = False,
    ) -> Tuple[bool, NormalizedThetaTerm, Optional[str]]:
        normalized = self.normalize(term)
        duplicate = self.duplicate_location(normalized)
        if duplicate is not None and not allow_duplicate:
            return False, normalized, duplicate
        self.acc.append(normalized)
        self._acc_keys.add(normalized.key)
        return True, normalized, None

    def add_rej(
        self,
        term: TermLike,
        *,
        allow_duplicate: bool = False,
    ) -> Tuple[bool, NormalizedThetaTerm, Optional[str]]:
        normalized = self.normalize(term)
        duplicate = self.duplicate_location(normalized)
        if duplicate is not None and not allow_duplicate:
            return False, normalized, duplicate
        self.rej.append(normalized)
        self._rej_keys.add(normalized.key)
        return True, normalized, None

    @property
    def theta_acc_string(self) -> str:
        return theta_terms_to_string(self.acc, variable_names=self.variable_names)

    @property
    def theta_rej_string(self) -> str:
        return theta_terms_to_string(self.rej, variable_names=self.variable_names)

    @property
    def acc_keys(self) -> Tuple[str, ...]:
        return tuple(term.key for term in self.acc)

    @property
    def rej_keys(self) -> Tuple[str, ...]:
        return tuple(term.key for term in self.rej)


if __name__ == "__main__":
    print("=== theta_utils smoke test ===")

    registry = ThetaRegistry(theta_acc=["1", "t", "x", "u"], variable_names=("t", "x", "u"))

    generated_prefixes = [
        ["mul", "t", "n2", "t"],          # t * (t**2)
        ["n3", "t"],                       # t**3, duplicate of the first term
        ["mul", "mul", "t", "t", "t"],    # (t*t)*t, duplicate again
        ["mul", "t", "x"],                # a genuinely new term
    ]

    for prefix in generated_prefixes:
        normalized = normalize_theta_term(prefix, variable_names=("t", "x", "u"))
        location = registry.duplicate_location(normalized)
        print(f"prefix={prefix}")
        print(f"  expression: {normalized.expression}")
        print(f"  canonical : {normalized.canonical}")
        print(f"  key       : {normalized.key}")
        print(f"  complexity: {normalized.complexity}")
        print(f"  duplicate : {location}")
        if location is None and not normalized.is_zero:
            registry.add_acc(prefix)

    print(f"Theta_acc for discovery.py: {registry.theta_acc_string}")

    duplicate = normalize_theta_term("t**3", variable_names=("t", "x", "u"))
    print(f"t**3 key: {duplicate.key}; duplicate location: {registry.duplicate_location(duplicate)}")

    print("\n=== expression -> DSO prefix check ===")
    for expr in ["t*x", "t**2", "t*(t**2)", "x/u"]:
        print(f"{expr:10s} -> {expression_to_prefix(expr, variable_names=('t', 'x', 'u'))}")

    print("\n=== theta set -> encoder tensor check ===")
    try:
        from decoder import DSORNNDecoder, make_dso_library

        library = make_dso_library(
            input_names=("t", "x", "u"),
            function_set=("1", "add", "sub", "mul", "div", "n2", "n3"),
            protected=True,
        )
        decoder = DSORNNDecoder(
            library=library,
            embed_size=16,
            hidden_size=32,
            max_length=10,
            never_sample_tokens=("1",),
        )
        acc_tensor, acc_mask, rej_tensor, rej_mask = theta_sets_to_encoder_inputs(
            ["1", "t", "x", "u", "t*x", "x**4"],
            [],
            decoder.token_map,
            library=library,
            variable_names=("t", "x", "u"),
        )
        print(f"library names: {library.names}")
        print(f"x**4 with n2/n3 library: {theta_term_to_prefix('x**4', library=library, variable_names=('t', 'x', 'u'))}")
        print(f"acc_tensor shape: {tuple(acc_tensor.shape)}, acc_mask={acc_mask.tolist()}")
        print(f"rej_tensor shape: {tuple(rej_tensor.shape)}, rej_mask={rej_mask.tolist()}")
    except Exception as exc:
        print(f"encoder tensor check skipped: {exc}")
