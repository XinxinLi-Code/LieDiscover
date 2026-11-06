try:
    import torch
    from torch.autograd.functional import jvp
    _TORCH_IMPORT_ERROR = None
except ImportError as exc:
    torch = None
    jvp = None
    _TORCH_IMPORT_ERROR = exc
import sympy as sp
from functools import lru_cache
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from orgin_theta_computer import (
    apply_concrete_theta, build_theta_n_matrix, get_sym, incremental_expand_matrix,
)


_PARSER_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_TASK_CONFIGS = {
    'heat': {
        'indep_vars': ('t', 'x'),
        'dep_vars': ('u',),
        'target_terms': ('u', 'u_x', 'u_xx', 'u_t'),
        'theta_derivatives': ('x', 'xx', 't'),
        'theta': ('1', 't', 'x', 'u', 't**2', 'x**2', 'u**2', 't*x', 't*u', 'x*u'),
    },
    'burger': {
        'indep_vars': ('t', 'x'),
        'dep_vars': ('u',),
        'target_terms': ('u', 'u_x', 'u_xx', 'u_t'),
        'theta_derivatives': ('x', 'xx', 't'),
        'theta': ('1', 't', 'x', 'u', 't**2', 'x**2', 'u**2', 't*x', 't*u', 'x*u'),
    },
    'damped_burger': {
        'indep_vars': ('t', 'x'),
        'dep_vars': ('u',),
        'target_terms': ('u', 'u_x', 'u_xx', 'u_t'),
        'theta_derivatives': ('x', 'xx', 't'),
        'theta': (
            '1', 't', 'x', 'u', 't**2', 'x**2', 'u**2', 't*x', 't*u', 'x*u',
        ),
    },
    'forced_transport': {
        'indep_vars': ('t', 'x'),
        'dep_vars': ('u',),
        'target_terms': ('x', 'u', 'u_x', 'u_xx', 'u_t'),
        'theta_derivatives': ('x', 'xx', 't'),
        'theta': (
            '1', 't', 'x', 'u', 't**2', 'x**2', 'u**2',
            't*x', 't*u', 'x*u',
        ),
    },
    'kdv': {
        'indep_vars': ('t', 'x'),
        'dep_vars': ('u',),
        'target_terms': ('u', 'u_x', 'u_xx', 'u_xxx', 'u_t'),
        'theta_derivatives': ('x', 'xx', 'xxx', 't'),
        'theta': ('1', 't', 'x', 'u', 't**2', 'x**2', 'u**2', 't*x', 't*u', 'x*u'),
    },
    'wave': {
        'indep_vars': ('t', 'x', 'y'),
        'dep_vars': ('u',),
        'target_terms': ('u', 'u_x', 'u_y', 'u_xx', 'u_yy', 'u_xy', 'u_tt'),
        'theta_derivatives': ('t', 'x', 'y', 'tt', 'xx', 'yy', 'xy'),
        'theta': (
            '1', 't', 'x', 'y', 'u', 't**2', 'x**2', 'y**2', 'u**2',
            't*x', 't*y', 't*u', 'x*y', 'x*u', 'y*u',
        ),
    },
    'rd': {
        'indep_vars': ('t', 'x', 'y'),
        'dep_vars': ('u', 'v'),
        'target_terms': (
            'u', 'u_x', 'u_y', 'u_xx', 'u_yy', 'u_xy',
            'v', 'v_x', 'v_y', 'v_xx', 'v_yy', 'v_xy',
            'u_t', 'v_t',
        ),
        'theta_derivatives': ('t', 'x', 'y', 'xx', 'yy', 'xy'),
        'theta': (
            '1', 't', 'x', 'y', 'u', 'v', 't**2', 'x**2', 'y**2',
            'u**2', 'v**2', 't*x', 't*y', 't*u', 't*v', 'x*y',
            'x*u', 'x*v', 'y*u', 'y*v', 'u*v',
        ),
    },
    'schrodinger': {
        'indep_vars': ('t', 'x', 'y'),
        'dep_vars': ('u', 'v'),
        'target_terms': (
            'u', 'u_x', 'u_y', 'u_xx', 'u_yy', 'u_xy',
            'v', 'v_x', 'v_y', 'v_xx', 'v_yy', 'v_xy',
            'u_t', 'v_t',
        ),
        'theta_derivatives': ('t', 'x', 'y', 'xx', 'yy', 'xy'),
        'theta': (
            '1', 't', 'x', 'y', 'u', 'v', 't**2', 'x**2', 'y**2',
            'u**2', 'v**2', 't*x', 't*y', 't*u', 't*v', 'x*y',
            'x*u', 'x*v', 'y*u', 'y*v', 'u*v',
        ),
    },
}

_TOP_VARIABLES = ('p0', 'p1', 'p2', 'p3')
_TOP_DEFAULT_THETA = _TOP_VARIABLES


def _split_theta_string(theta):
    theta = theta.strip()
    if theta.startswith('[') and theta.endswith(']'):
        theta = theta[1:-1]
    return tuple(term.strip() for term in theta.split(',') if term.strip())


def _theta_key(task, theta_terms=None):
    if task == 'top':
        if theta_terms is None:
            theta_terms = _TOP_DEFAULT_THETA
        if isinstance(theta_terms, str):
            theta_terms = _split_theta_string(theta_terms)
        return tuple(str(expr) for expr in _parse_theta_terms(task, theta_terms))
    if theta_terms is None:
        theta_terms = _TASK_CONFIGS[task]['theta']
    if isinstance(theta_terms, str):
        theta_terms = _split_theta_string(theta_terms)
    return tuple(str(expr) for expr in _parse_theta_terms(task, theta_terms))


def _validate_theta_key(theta_key):
    if len(set(theta_key)) != len(theta_key):
        raise ValueError(f'Theta terms must be unique after simplification: {theta_key}')


def _symbol_dict(config):
    names = config['indep_vars'] + config['dep_vars']
    symbols = {name: sp.Symbol(name) for name in names}
    symbols.update({
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'exp': sp.exp,
        'log': sp.log,
        'sqrt': sp.sqrt,
        'pi': sp.pi,
        'E': sp.E,
    })
    return symbols


def _parse_theta_terms(task, theta_key):
    if task == 'top':
        local_dict = _top_symbol_dict()
        terms = []
        for term in theta_key:
            if isinstance(term, str):
                terms.append(parse_expr(term, local_dict=local_dict, transformations=_PARSER_TRANSFORMS))
            else:
                terms.append(sp.sympify(term))
        return tuple(terms)
    config = _TASK_CONFIGS[task]
    local_dict = _symbol_dict(config)
    terms = []
    for term in theta_key:
        if isinstance(term, str):
            terms.append(parse_expr(term, local_dict=local_dict, transformations=_PARSER_TRANSFORMS))
        else:
            terms.append(sp.sympify(term))
    return tuple(terms)


@lru_cache(maxsize=None)
def _cached_task_symbols(task):
    config = _TASK_CONFIGS[task]
    indep_vars = tuple(sp.Symbol(name) for name in config['indep_vars'])
    dep_vars = tuple(sp.Symbol(name) for name in config['dep_vars'])
    return indep_vars, dep_vars


@lru_cache(maxsize=None)
def _cached_abstract_theta_n(task):
    indep_vars, dep_vars = _cached_task_symbols(task)
    return build_theta_n_matrix(list(indep_vars), list(dep_vars), _TASK_CONFIGS[task]['target_terms'])


def _total_derivative(expr, var_sym, dep_vars):
    expr = sp.sympify(expr)
    if expr.is_Number:
        return sp.S.Zero

    res = sp.diff(expr, var_sym)
    for sym in expr.free_symbols:
        name = sym.name
        if any(name == dep_var.name for dep_var in dep_vars):
            res += get_sym(name, var_sym.name) * sp.diff(expr, sym)
        elif any(name.startswith(f"{dep_var.name}_{{") for dep_var in dep_vars):
            base, sub_part = name.split('_')
            sub = sub_part.strip('{}')
            res += get_sym(base, sub + var_sym.name) * sp.diff(expr, sym)
    return sp.simplify(res)


def _differentiate_theta(theta_terms, deriv_str, indep_vars, dep_vars):
    result = tuple(theta_terms)
    for var_name in deriv_str:
        var_sym = next(var for var in indep_vars if var.name == var_name)
        result = tuple(_total_derivative(expr, var_sym, dep_vars) for expr in result)
    return result


def _theta_derivative_key(deriv_str):
    return 'dth_' + ''.join(f'd{var_name}' for var_name in deriv_str)


def _build_theta_expression_dict(task, theta_terms, indep_vars, dep_vars):
    config = _TASK_CONFIGS[task]
    theta = {'th': theta_terms}
    for deriv_str in config['theta_derivatives']:
        theta[_theta_derivative_key(deriv_str)] = _differentiate_theta(
            theta_terms, deriv_str, indep_vars, dep_vars
        )
    return theta


def _select_expr_tuple(exprs, source_key, target_key):
    indices = [source_key.index(term) for term in target_key]
    return tuple(exprs[i] for i in indices)


def _select_theta_expressions(theta, source_key, target_key):
    return {key: _select_expr_tuple(exprs, source_key, target_key) for key, exprs in theta.items()}


def _merge_theta_expressions(old_theta, old_key, delta_theta, delta_key, target_key):
    source_key = old_key + delta_key
    merged_theta = {
        key: old_theta[key] + delta_theta[key]
        for key in old_theta
    }
    return _select_theta_expressions(merged_theta, source_key, target_key)


def _select_sympy_theta_n_columns(matrix, source_key, target_key, num_vars):
    r = len(source_key)
    indices = [source_key.index(term) for term in target_key]
    blocks = []
    for block_idx in range(num_vars):
        block = matrix[:, block_idx * r:(block_idx + 1) * r]
        blocks.append(block[:, indices])
    return sp.Matrix.hstack(*blocks)


def _best_cached_theta_entry(task, theta_key):
    best_key = None
    best_entry = None
    best_reuse_count = 0
    target_terms = set(theta_key)
    for (cached_task, cached_key), cached_entry in _THETA_EXPRESSION_CACHE.items():
        if cached_task != task:
            continue
        reuse_count = len(target_terms.intersection(cached_key))
        if reuse_count > best_reuse_count:
            best_key = cached_key
            best_entry = cached_entry
            best_reuse_count = reuse_count
    return best_key, best_entry


_THETA_EXPRESSION_CACHE = {}


def _cached_theta_expressions(task, theta_key):
    _validate_theta_key(theta_key)
    cache_key = (task, theta_key)
    if cache_key in _THETA_EXPRESSION_CACHE:
        return _THETA_EXPRESSION_CACHE[cache_key]

    indep_vars, dep_vars = _cached_task_symbols(task)
    abstract_matrix = _cached_abstract_theta_n(task)
    num_vars = len(indep_vars) + len(dep_vars)

    old_key, old_entry = _best_cached_theta_entry(task, theta_key)
    missing_key = tuple(term for term in theta_key if old_key is None or term not in old_key)

    if old_entry is not None and len(missing_key) < len(theta_key):
        old_theta, old_theta_n = old_entry
        delta_terms = _parse_theta_terms(task, missing_key)
        delta_theta = _build_theta_expression_dict(task, delta_terms, indep_vars, dep_vars)
        if missing_key:
            expanded_theta_n = incremental_expand_matrix(
                abstract_matrix,
                old_theta_n,
                list(indep_vars),
                list(dep_vars),
                len(old_key),
                list(delta_terms),
            )
            source_key = old_key + missing_key
        else:
            expanded_theta_n = old_theta_n
            source_key = old_key
        theta = _merge_theta_expressions(old_theta, old_key, delta_theta, missing_key, theta_key)
        theta_n = _select_sympy_theta_n_columns(expanded_theta_n, source_key, theta_key, num_vars)
    else:
        theta_terms = _parse_theta_terms(task, theta_key)
        theta = _build_theta_expression_dict(task, theta_terms, indep_vars, dep_vars)
        theta_n = apply_concrete_theta(abstract_matrix, list(indep_vars), list(dep_vars), theta_terms)

    _THETA_EXPRESSION_CACHE[cache_key] = (theta, theta_n)
    return theta, theta_n


def _sympy_derivative_to_data_key(base, sub):
    return 'd' + base + ''.join(f'd{var_name}' for var_name in sub)


def _lookup_symbol_value(sym, data):
    name = sym.name
    if name in data:
        return data[name]
    if '_{' in name:
        base, sub_part = name.split('_')
        key = _sympy_derivative_to_data_key(base, sub_part.strip('{}'))
        if key in data:
            return data[key]
    raise KeyError(f"Symbol {name} cannot be resolved from data keys: {sorted(data.keys())}")


def _reference_tensor(data, device):
    for value in data.values():
        if torch.is_tensor(value):
            return value.to(device)
    raise ValueError('data must contain at least one torch.Tensor value')


def _as_scalar_tensor(value, data, device):
    ref = _reference_tensor(data, device)
    if torch.is_tensor(value):
        value = value.to(device=device, dtype=ref.dtype)
        return value.reshape(()) if value.numel() == 1 else value
    return ref.new_tensor(value)


def _eval_sympy_expr(expr, data, device):
    expr = sp.sympify(expr)
    if expr.is_Number:
        return _as_scalar_tensor(float(expr), data, device)
    if expr in (sp.pi, sp.E):
        return _as_scalar_tensor(float(sp.N(expr)), data, device)
    if expr.is_number:
        return _as_scalar_tensor(float(sp.N(expr)), data, device)
    if expr.is_Symbol:
        return _as_scalar_tensor(_lookup_symbol_value(expr, data), data, device)
    if expr.is_Add:
        terms = [_eval_sympy_expr(arg, data, device) for arg in expr.args]
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result
    if expr.is_Mul:
        factors = [_eval_sympy_expr(arg, data, device) for arg in expr.args]
        result = factors[0]
        for factor in factors[1:]:
            result = result * factor
        return result
    if expr.is_Pow:
        base, exponent = expr.args
        base_value = _eval_sympy_expr(base, data, device)
        if exponent.is_Number:
            return base_value ** float(exponent)
        return base_value ** _eval_sympy_expr(exponent, data, device)
    if expr.func == sp.sin:
        return torch.sin(_eval_sympy_expr(expr.args[0], data, device))
    if expr.func == sp.cos:
        return torch.cos(_eval_sympy_expr(expr.args[0], data, device))
    if expr.func == sp.tan:
        return torch.tan(_eval_sympy_expr(expr.args[0], data, device))
    if expr.func == sp.exp:
        return torch.exp(_eval_sympy_expr(expr.args[0], data, device))
    if expr.func == sp.log:
        return torch.log(_eval_sympy_expr(expr.args[0], data, device))
    if expr.func == sp.sqrt:
        return torch.sqrt(_eval_sympy_expr(expr.args[0], data, device))
    raise NotImplementedError(f"Unsupported theta expression: {expr}")


def _eval_expr_vector(exprs, data, device):
    return torch.stack([_eval_sympy_expr(expr, data, device) for expr in exprs])


def _eval_expr_matrix(matrix, data, device):
    values = [_eval_sympy_expr(matrix[i, j], data, device) for i in range(matrix.rows) for j in range(matrix.cols)]
    return torch.stack(values).reshape(matrix.rows, matrix.cols)


class _CompiledExpressionMatrix:
    """Evaluate a fixed SymPy tree without repeating symbolic interpretation.

    Identical nodes are reused, but every addition, multiplication and power
    keeps the interpreter's operation order. There is no algebraic CSE,
    simplification, batching, or fused arithmetic here.
    """

    def __init__(self, matrix):
        nodes, positions, constants = [], {}, []

        def visit(expression):
            expression = sp.sympify(expression)
            if expression in positions:
                return positions[expression]
            if expression.is_Number:
                value = float(expression)
                node = ('constant', len(constants))
                constants.append(value)
            elif expression in (sp.pi, sp.E) or expression.is_number:
                value = float(sp.N(expression))
                node = ('constant', len(constants))
                constants.append(value)
            elif expression.is_Symbol:
                node = ('symbol', expression)
            elif expression.is_Add:
                node = ('add', tuple(visit(arg) for arg in expression.args))
            elif expression.is_Mul:
                node = ('multiply', tuple(visit(arg) for arg in expression.args))
            elif expression.is_Pow:
                base, exponent = expression.args
                node = (('power', visit(base), float(exponent)) if exponent.is_Number
                        else ('variable_power', visit(base), visit(exponent)))
            elif expression.func in (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sqrt):
                node = ('function', getattr(torch, expression.func.__name__),
                        visit(expression.args[0]))
            else:
                raise NotImplementedError(f'Unsupported theta expression: {expression}')
            positions[expression] = len(nodes)
            nodes.append(node)
            return len(nodes) - 1

        self.indices = tuple(visit(expression) for expression in matrix)
        self.nodes = tuple(nodes)
        self.shape = (matrix.rows, matrix.cols)

        @lru_cache(maxsize=8)
        def constant_tensors(dtype, device):
            return tuple(torch.tensor(value, dtype=dtype, device=device) for value in constants)

        self._constant_tensors = constant_tensors

    def __call__(self, data, device):
        reference = _reference_tensor(data, device)
        constants = self._constant_tensors(reference.dtype, reference.device)
        values = []
        for node in self.nodes:
            operation = node[0]
            if operation == 'constant':
                value = constants[node[1]]
            elif operation == 'symbol':
                value = _lookup_symbol_value(node[1], data)
                if torch.is_tensor(value):
                    value = value.to(device=reference.device, dtype=reference.dtype)
                    value = value.reshape(()) if value.numel() == 1 else value
                else:
                    value = reference.new_tensor(value)
            elif operation in ('add', 'multiply'):
                value = values[node[1][0]]
                for index in node[1][1:]:
                    value = value + values[index] if operation == 'add' else value * values[index]
            elif operation == 'power':
                value = values[node[1]] ** node[2]
            elif operation == 'variable_power':
                value = values[node[1]] ** values[node[2]]
            else:
                value = node[1](values[node[2]])
            values.append(value)
        return torch.stack([values[index] for index in self.indices]).reshape(self.shape)


@lru_cache(maxsize=256)
def _cached_theta_n_plan(task, theta_key):
    _, matrix = _cached_theta_expressions(task, theta_key)
    return _CompiledExpressionMatrix(matrix)


def get_theta_n_evaluator(task, device, theta_terms=None):
    """Prepare one pointwise prolongation evaluator outside the sample loop.

    The result matches ``compute_theta_n`` while avoiding repeated parsing and
    SymPy traversal. Constants follow each input's dtype/device, so one callable
    can safely serve float32 and float64 samples. ``compute_theta_n`` retains
    the original interpreter for reference and compatibility.
    """
    if task == 'top':
        parsed_terms = _parse_theta_terms(task, _theta_key(task, theta_terms))

        def evaluate_top(data):
            return _build_top_theta_n(_eval_top_theta_values(parsed_terms, data, device))

        return evaluate_top
    if task not in _TASK_CONFIGS:
        raise NotImplementedError(f'Unsupported task: {task}')
    plan = _cached_theta_n_plan(task, _theta_key(task, theta_terms))
    return lambda data: plan(data, device)


def _top_symbol_dict():
    symbols = {name: sp.Symbol(name) for name in _TOP_VARIABLES}
    symbols.update({
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'exp': sp.exp,
        'log': sp.log,
        'sqrt': sp.sqrt,
        'pi': sp.pi,
        'E': sp.E,
    })
    return symbols


def _top_momentum_tensor(data, device):
    x = data['x'].to(device)
    if x.dim() == 3:
        if x.shape[0] != 1:
            raise ValueError(f'top discovery expects batch_size=1, got x shape {tuple(x.shape)}')
        x = x[0]
    elif x.dim() == 1:
        if x.numel() % 4 != 0:
            raise ValueError(f'top flattened x length must be divisible by 4, got {x.numel()}')
        x = x.reshape(-1, 4)
    if x.dim() != 2 or x.shape[1] != 4:
        raise ValueError(f'top x must have shape [n_component, 4], got {tuple(x.shape)}')
    return x


def _eval_top_theta_values(theta_terms, data, device):
    momenta = _top_momentum_tensor(data, device)
    top_data = {name: momenta[:, idx] for idx, name in enumerate(_TOP_VARIABLES)}
    values = []
    for expr in theta_terms:
        value = _eval_sympy_expr(expr, top_data, device)
        if value.ndim == 0:
            value = value.expand(momenta.shape[0])
        values.append(value.reshape(momenta.shape[0]))
    return torch.stack(values, dim=1)


def _build_top_theta_n(theta_values):
    n_component, num_terms = theta_values.shape
    theta_n = torch.zeros(
        (n_component * 4, 4 * num_terms),
        device=theta_values.device,
        dtype=theta_values.dtype,
    )
    for i in range(n_component):
        for j in range(4):
            theta_n[4 * i + j, j * num_terms:(j + 1) * num_terms] = theta_values[i]
    return theta_n


def _concat_torch_theta_n_blocks(old_theta_n, delta_theta_n, old_r, delta_r, num_vars):
    blocks = []
    for block_idx in range(num_vars):
        old_block = old_theta_n[:, block_idx * old_r:(block_idx + 1) * old_r]
        delta_block = delta_theta_n[:, block_idx * delta_r:(block_idx + 1) * delta_r]
        blocks.append(torch.cat([old_block, delta_block], dim=1))
    return torch.cat(blocks, dim=1)


def _select_torch_theta_n_columns(matrix, source_key, target_key, num_vars):
    r = len(source_key)
    indices = torch.tensor([source_key.index(term) for term in target_key], device=matrix.device)
    blocks = []
    for block_idx in range(num_vars):
        block = matrix[:, block_idx * r:(block_idx + 1) * r]
        blocks.append(torch.index_select(block, dim=1, index=indices))
    return torch.cat(blocks, dim=1)


def compute_theta(data, task, device, theta_terms=None):
    if task == 'top':
        theta_key = _theta_key(task, theta_terms)
        theta_terms = _parse_theta_terms(task, theta_key)
        theta = {'th': _eval_top_theta_values(theta_terms, data, device)}
    elif task in _TASK_CONFIGS:
        theta_key = _theta_key(task, theta_terms)
        theta_exprs, _ = _cached_theta_expressions(task, theta_key)
        theta = {key: _eval_expr_vector(exprs, data, device) for key, exprs in theta_exprs.items()}
    else:
        raise NotImplementedError(f'Unsupported task: {task}')
    return theta


def compute_theta_n(data, task, device, theta_terms=None):
    if task == 'top':
        theta = compute_theta(data, task, device, theta_terms=theta_terms)
        th = theta['th']
        theta_n = _build_top_theta_n(th)
    elif task in _TASK_CONFIGS:
        theta_key = _theta_key(task, theta_terms)
        _, theta_n_expr = _cached_theta_expressions(task, theta_key)
        theta_n = _eval_expr_matrix(theta_n_expr, data, device)
    else:
        raise NotImplementedError(f'Unsupported task: {task}')
    return theta_n


def compute_theta_n_incremental(data, task, device, old_theta_terms, added_theta_terms, old_theta_n=None):
    """
    Compute Theta_n after appending new Theta terms.

    If old_theta_n is provided, only the columns induced by added_theta_terms
    are evaluated for this data point, then they are stitched into each
    variable block of the old matrix.
    """
    if task == 'top':
        old_key = _theta_key(task, old_theta_terms)
        added_key = _theta_key(task, added_theta_terms)
        target_key = old_key + tuple(term for term in added_key if term not in old_key)
        return compute_theta_n(data, task, device, theta_terms=target_key)

    old_key = _theta_key(task, old_theta_terms)
    added_key = _theta_key(task, added_theta_terms)
    _validate_theta_key(old_key)

    delta_key = tuple(term for term in added_key if term not in old_key)
    target_key = old_key + delta_key
    if not delta_key:
        if old_theta_n is not None:
            return old_theta_n
        return compute_theta_n(data, task, device, theta_terms=old_key)

    if old_theta_n is None:
        old_theta_n = compute_theta_n(data, task, device, theta_terms=old_key)

    indep_vars, dep_vars = _cached_task_symbols(task)
    num_vars = len(indep_vars) + len(dep_vars)
    expected_old_cols = len(old_key) * num_vars
    if old_theta_n.shape[1] != expected_old_cols:
        raise ValueError(
            f'old_theta_n has {old_theta_n.shape[1]} columns, '
            f'but old_theta_terms require {expected_old_cols}.'
        )
    abstract_matrix = _cached_abstract_theta_n(task)
    delta_terms = _parse_theta_terms(task, delta_key)
    delta_theta_n_expr = apply_concrete_theta(abstract_matrix, list(indep_vars), list(dep_vars), delta_terms)
    delta_theta_n = _eval_expr_matrix(delta_theta_n_expr, data, device)
    combined = _concat_torch_theta_n_blocks(
        old_theta_n,
        delta_theta_n,
        old_r=len(old_key),
        delta_r=len(delta_key),
        num_vars=num_vars,
    )
    return _select_torch_theta_n_columns(combined, target_key, target_key, num_vars)

def symmetry_discovery(
    model, train_loader, task, device, threshold=None, sample=0,
    return_full_vh=False, return_scales=False, **kwargs,
):
    """Discover a normalized coefficient basis using a relative spectral cutoff.

    Prolonged-column scaling is a separate preprocessing step before multiplying
    by the Jacobians. Supplied ``column_scales`` (or legacy ``feature_scales``)
    and ``reference_samples`` can be reused for a fixed library evaluation. A
    manual threshold is relative and must lie within the configured bounds.
    ``return_scales=True`` appends the per-coefficient column scale vector;
    returned Vh rows use normalized coefficients, including ``return_full_vh``.
    """
    if torch is None:
        raise ImportError(f'Torch is required for symmetry_discovery: {_TORCH_IMPORT_ERROR}')
    from feature_normalization import (
        normalize_columns, prolonged_column_scales, validate_coefficient_scales,
    )
    from spectral import (
        DEFAULT_SPECTRAL_FLOOR,
        estimate_spectral_cut, select_null_basis,
    )
    from feature_normalization import SEARCH_DTYPE, prepare_search_model, search_data

    model = prepare_search_model(model, device)
    model.eval()
    sample_size = int(kwargs.get('sample_size', 100))
    if sample_size <= 0:
        raise ValueError('sample_size must be positive.')
    reference_samples = kwargs.get('reference_samples')
    if reference_samples is None:
        reference_samples = []
        start = int(sample) * sample_size
        stop = (int(sample) + 1) * sample_size
        for j, data in enumerate(train_loader):
            if j < start:
                continue
            if j >= stop:
                break
            reference_samples.append(search_data(data, device))
    else:
        reference_samples = list(reference_samples)
    if not reference_samples:
        raise ValueError('No samples were processed; check train_loader length and sample index.')
    theta_terms = kwargs.get('theta_terms', kwargs.get('theta'))
    if task == 'top':
        variable_dim = 4
    elif task in _TASK_CONFIGS:
        config = _TASK_CONFIGS[task]
        variable_dim = len(config['indep_vars']) + len(config['dep_vars'])
    else:
        raise NotImplementedError(f'Unsupported task: {task}')
    model_dtype = SEARCH_DTYPE

    with torch.no_grad():
        prolonged_rows, jacobians = [], []
        for sample_data in reference_samples:
            data = search_data(sample_data, device)
            if task == 'top':
                feature = data['x'].reshape(data['x'].shape[0], -1)
            elif task in ['heat', 'burger', 'damped_burger']:
                feature = torch.stack([data['u'], data['dudx'], data['dudxdx']], dim=1)
            elif task == 'forced_transport':
                feature = torch.stack([data['x'], data['u'], data['dudx'], data['dudxdx']], dim=1)
            elif task == 'kdv':
                feature = torch.stack([data['u'], data['dudx'], data['dudxdx'], data['dudxdxdx']], dim=1)
            elif task == 'wave':
                feature = torch.stack([data['u'], data['dudx'], data['dudy'], data['dudxdx'], data['dudydy'], data['dudxdy']], dim=1)
            elif task in ['rd', 'schrodinger']:
                feature = torch.stack([data['u'], data['dudx'], data['dudy'], data['dudxdx'], data['dudydy'], data['dudxdy'], data['v'], data['dvdx'], data['dvdy'], data['dvdxdx'], data['dvdydy'], data['dvdxdy']], dim=1)
            theta_n = compute_theta_n(data, task, device, theta_terms=theta_terms)
            if task == 'top':
                df = torch.zeros((kwargs['output_dim'], kwargs['input_dim']), device=device, dtype=torch.float64)
            else:
                df = torch.zeros(
                    (kwargs['output_dim'], kwargs['input_dim'] + kwargs['output_dim']),
                    device=device, dtype=torch.float64,
                )
                df[:, kwargs['input_dim']:] = -torch.eye(kwargs['output_dim'], device=device, dtype=torch.float64)
            for i in range(kwargs['input_dim']):
                v = torch.zeros(kwargs['input_dim'], device=device, dtype=model_dtype)
                v[i] = 1
                df[:, i] = jvp(model, feature[0], v=v)[1]
            prolonged_rows.append(theta_n.to(dtype=torch.float64))
            jacobians.append(df)
        num_terms = prolonged_rows[0].shape[1] // variable_dim
        scales = kwargs.get('column_scales', kwargs.get('feature_scales'))
        if kwargs.get('column_scales') is not None and kwargs.get('feature_scales') is not None:
            raise ValueError('Supply column_scales or feature_scales, not both.')
        if scales is None:
            scales = prolonged_column_scales(
                prolonged_rows, eps_s=kwargs.get('feature_scale_eps', 1e-12),
            )
        scales = validate_coefficient_scales(scales, num_terms, variable_dim, device)
        normalized_rows = [
            df @ matrix for df, matrix in zip(jacobians, normalize_columns(prolonged_rows, scales))
        ]
        columns = prolonged_rows[0].shape[1]
        gram = torch.zeros((columns, columns), device=device, dtype=torch.float64)
        for ci in normalized_rows:
            gram += ci.T @ ci
        if not bool(torch.isfinite(gram).all()):
            raise ValueError('The normalized Gram matrix contains non-finite values.')
        _, S2, Vh = torch.linalg.svd(gram)
        S = torch.sqrt(torch.clamp(S2, min=0.0))
        floor = kwargs.get('spectral_floor', DEFAULT_SPECTRAL_FLOOR)
        cut = estimate_spectral_cut(
            S, fixed_threshold=threshold,
            threshold_min=kwargs.get('threshold_min'),
            threshold_max=kwargs.get('threshold_max'),
            use_threshold_bounds=kwargs.get('use_threshold_bounds', False),
            spectral_floor=floor,
        )
        basis = Vh if return_full_vh else select_null_basis(S, Vh, cut.threshold, spectral_floor=floor)
        return (S, basis, scales) if return_scales else (S, basis)


def check_incremental_theta_n():
    task = 'burger'
    old_theta_terms = '1,t,x,u'
    added_theta_terms = 't^2,sin(x)'
    target_theta_terms = '1,t,x,u,t^2,sin(x)'

    print('=== Symbolic incremental Theta_n check ===')
    old_key = _theta_key(task, old_theta_terms)
    delta_key = tuple(term for term in _theta_key(task, added_theta_terms) if term not in old_key)
    target_key = _theta_key(task, target_theta_terms)
    indep_vars, dep_vars = _cached_task_symbols(task)
    abstract_matrix = _cached_abstract_theta_n(task)
    num_vars = len(indep_vars) + len(dep_vars)

    old_terms = _parse_theta_terms(task, old_key)
    delta_terms = _parse_theta_terms(task, delta_key)
    target_terms = _parse_theta_terms(task, target_key)
    old_symbolic = apply_concrete_theta(abstract_matrix, list(indep_vars), list(dep_vars), old_terms)
    expanded_symbolic = incremental_expand_matrix(
        abstract_matrix,
        old_symbolic,
        list(indep_vars),
        list(dep_vars),
        len(old_key),
        list(delta_terms),
    )
    incremental_symbolic = _select_sympy_theta_n_columns(
        expanded_symbolic,
        old_key + delta_key,
        target_key,
        num_vars,
    )
    full_symbolic = apply_concrete_theta(abstract_matrix, list(indep_vars), list(dep_vars), target_terms)
    symbolic_equal = sp.simplify(incremental_symbolic - full_symbolic).is_zero_matrix
    print(f'old Theta terms: {old_key}')
    print(f'added Theta terms: {delta_key}')
    print(f'symbolic shape: {incremental_symbolic.shape}')
    print(f'symbolic equals full recomputation: {symbolic_equal}')

    print('\n=== Numeric incremental Theta_n check ===')
    if torch is None:
        print(f'skipped: torch import failed: {_TORCH_IMPORT_ERROR}')
        return
    device = torch.device('cpu')
    data = {
        't': torch.tensor(0.7, device=device),
        'x': torch.tensor(-0.3, device=device),
        'u': torch.tensor(1.2, device=device),
        'dudx': torch.tensor(-0.5, device=device),
        'dudxdx': torch.tensor(0.25, device=device),
        'dudt': torch.tensor(0.9, device=device),
        'dudtdx': torch.tensor(-0.15, device=device),
    }
    old_numeric = compute_theta_n(data, task, device, theta_terms=old_theta_terms)
    incremental_numeric = compute_theta_n_incremental(
        data,
        task,
        device,
        old_theta_terms=old_theta_terms,
        added_theta_terms=added_theta_terms,
        old_theta_n=old_numeric,
    )
    full_numeric = compute_theta_n(data, task, device, theta_terms=target_theta_terms)
    max_abs_diff = torch.max(torch.abs(incremental_numeric - full_numeric)).item()
    print(f'numeric shape: {tuple(incremental_numeric.shape)}')
    print(f'max abs diff: {max_abs_diff:.3e}')
    print(f'numeric allclose: {torch.allclose(incremental_numeric, full_numeric, atol=1e-6, rtol=1e-6)}')


if __name__ == '__main__':
    check_incremental_theta_n()
