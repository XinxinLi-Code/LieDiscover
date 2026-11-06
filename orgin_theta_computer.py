import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


def get_sym(base, sub):
    if not sub:
        return sp.Symbol(base)
    sorted_sub = "".join(sorted(sub))
    if "Theta" in base:
        return sp.Symbol(f"D_{{{sorted_sub}}}\\Theta^T")
    else:
        return sp.Symbol(f"{base}_{{{sorted_sub}}}")


def get_abstract_coeffs(indep_vars, dep_var, target_deriv_str):
    
    def abstract_D(expr, var):
        expr = sp.sympify(expr)
        if expr.is_Number: 
            return sp.S.Zero
        elif expr.is_Symbol:
            name = expr.name
            if name in [v.name for v in indep_vars]:  
                return sp.S.One if name == var.name else sp.S.Zero
            elif name == dep_var.name:                
                return get_sym(dep_var.name, var.name)
            elif name == "\\Theta^T":                 
                return get_sym("\\Theta^T", var.name)
            elif name.startswith(f"{dep_var.name}_{{"):  
                sub = name.split('_')[1].strip('{}')
                return get_sym(dep_var.name, sub + var.name)
            elif name.startswith("D_{"):              
                sub = name.split('}')[0].replace("D_{", "")
                return get_sym("\\Theta^T", sub + var.name)
            else: 
                return sp.S.Zero
                
        elif expr.is_Add: 
            return sp.Add(*[abstract_D(arg, var) for arg in expr.args])
        elif expr.is_Mul: 
            args = expr.args
            res = sp.S.Zero
            for i, arg in enumerate(args):
                term = abstract_D(arg, var)
                for j, other in enumerate(args):
                    if i != j: term *= other
                res += term
            return res
        else: 
            return sp.S.Zero

    def compute_recursive(deriv_str):
        deriv_str = "".join(sorted(deriv_str))
        if not deriv_str:
            A = {v.name: sp.S.Zero for v in indep_vars}
            B = sp.Symbol("\\Theta^T")
            return A, B
        
        last_var_str = deriv_str[-1]
        prev_str = deriv_str[:-1]
        last_var_sym = [v for v in indep_vars if v.name == last_var_str][0]
        
        prev_A, prev_B = compute_recursive(prev_str)
        D_k_Theta = abstract_D(sp.Symbol("\\Theta^T"), last_var_sym)
        
        new_B = abstract_D(prev_B, last_var_sym)
        new_A = {}
        for v in indep_vars:
            u_J_i_str = prev_str + v.name
            u_J_i = get_sym(dep_var.name, u_J_i_str)
            new_A[v.name] = sp.simplify(abstract_D(prev_A[v.name], last_var_sym) - u_J_i * D_k_Theta)
            
        return new_A, new_B

    A_dict, B = compute_recursive(target_deriv_str)
    coeffs = {f"W_{v}": A_dict[v] for v in [var.name for var in indep_vars]}
    coeffs[f"W_{dep_var.name}"] = B
    return coeffs

def build_theta_n_matrix(indep_vars, dep_vars, target_terms):
    rows = []
    for term in target_terms:
        parts = term.split('_')
        base_var_name = parts[0]
        deriv_str = parts[1] if len(parts) > 1 else ""
        if any(v.name == base_var_name for v in indep_vars):
            if deriv_str:
                raise ValueError(f"Independent-variable target {term!r} cannot have derivatives.")
            row_blocks = []
            for v in indep_vars:
                row_blocks.append(sp.Symbol("\\Theta^T") if v.name == base_var_name else sp.S.Zero)
            for _ in dep_vars:
                row_blocks.append(sp.S.Zero)
            rows.append(row_blocks)
            continue
        current_dep_var = next(v for v in dep_vars if v.name == base_var_name)
        
        coeffs = get_abstract_coeffs(indep_vars, current_dep_var, deriv_str)
        
        row_blocks = []
        for v in indep_vars:
            row_blocks.append(coeffs[f"W_{v.name}"])
        for v in dep_vars:
            if v.name == base_var_name:
                row_blocks.append(coeffs[f"W_{v.name}"])
            else:
                row_blocks.append(sp.S.Zero)
        rows.append(row_blocks)
    return sp.Matrix(rows)


def apply_concrete_theta(abstract_matrix, indep_vars, dep_vars, concrete_theta_list):
    r = len(concrete_theta_list)
    
    def concrete_D(expr_matrix, var_sym):
        def D_scalar(expr):
            expr = sp.sympify(expr)
            if expr.is_Number: 
                return sp.S.Zero
            
            res = sp.diff(expr, var_sym)
            for sym in expr.free_symbols:
                name = sym.name
                if any(name == dv.name for dv in dep_vars):
                    next_sym = get_sym(name, var_sym.name)
                    res += next_sym * sp.diff(expr, sym)
                elif any(name.startswith(f"{dv.name}_{{") for dv in dep_vars):
                    base, sub_part = name.split('_')
                    sub = sub_part.strip('{}')
                    next_sym = get_sym(base, sub + var_sym.name)
                    res += next_sym * sp.diff(expr, sym)
            return sp.simplify(res)
            
        return sp.Matrix([D_scalar(e) for e in expr_matrix]).reshape(1, r)

    theta_syms = set()
    for expr in abstract_matrix:
        for sym in expr.free_symbols:
            if "\\Theta^T" in sym.name:
                theta_syms.add(sym.name)
                
    concrete_dict = {}
    base_theta_matrix = sp.Matrix(concrete_theta_list).reshape(1, r)
    
    for sym_name in theta_syms:
        if sym_name == "\\Theta^T":
            concrete_dict[sym_name] = base_theta_matrix
        else:
            sub = sym_name.split('}')[0].replace("D_{", "")
            current_val = base_theta_matrix
            for var_str in sub:
                var_sym = next(v for v in indep_vars if v.name == var_str)
                current_val = concrete_D(current_val, var_sym)
            concrete_dict[sym_name] = current_val

    expanded_rows = []
    for i in range(abstract_matrix.rows):
        row_blocks = []
        for j in range(abstract_matrix.cols):
            expr = abstract_matrix[i, j]
            if expr == 0:
                row_blocks.append(sp.zeros(1, r))
            else:
                cell_matrix = sp.zeros(1, r)
                for sym_name in theta_syms:
                    sym = sp.Symbol(sym_name)
                    coeff = expr.coeff(sym)
                    if coeff != 0:
                        cell_matrix += coeff * concrete_dict[sym_name]
                row_blocks.append(sp.simplify(cell_matrix))
        expanded_rows.append(sp.Matrix.hstack(*row_blocks))
        
    return sp.Matrix.vstack(*expanded_rows)


_PARSER_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def _term_key(term):
    if isinstance(term, str):
        return str(parse_expr(term, transformations=_PARSER_TRANSFORMS))
    return str(sp.sympify(term))


def _validate_unique(keys, name):
    if len(set(keys)) != len(keys):
        raise ValueError(f'{name} must not contain duplicate terms after simplification: {keys}')


def _select_matrix_blocks(matrix, source_keys, target_keys, num_vars):
    r = len(source_keys)
    indices = [source_keys.index(term) for term in target_keys]
    blocks = []
    for j in range(num_vars):
        block = matrix[:, j * r:(j + 1) * r]
        blocks.append(block[:, indices])
    return sp.Matrix.hstack(*blocks)


def incremental_expand_matrix(abstract_matrix, old_matrix, indep_vars, dep_vars, old_r, new_theta_terms):
    delta_matrix = apply_concrete_theta(abstract_matrix, indep_vars, dep_vars, new_theta_terms)
    
    num_vars = len(indep_vars) + len(dep_vars) 
    new_r = len(new_theta_terms)
    
    new_blocks = []
    
    for j in range(num_vars):
        old_block = old_matrix[:, j * old_r : (j + 1) * old_r]
        delta_block = delta_matrix[:, j * new_r : (j + 1) * new_r]
        combined_block = sp.Matrix.hstack(old_block, delta_block)
        new_blocks.append(combined_block)
        
    final_matrix = sp.Matrix.hstack(*new_blocks)
    
    return final_matrix


def incremental_update_matrix(abstract_matrix, old_matrix, indep_vars, dep_vars, old_theta_terms, target_theta_terms):
    old_keys = tuple(_term_key(term) for term in old_theta_terms)
    target_keys = tuple(_term_key(term) for term in target_theta_terms)
    _validate_unique(old_keys, 'old_theta_terms')
    _validate_unique(target_keys, 'target_theta_terms')

    new_terms = []
    new_keys = []
    for term, key in zip(target_theta_terms, target_keys):
        if key not in old_keys:
            new_terms.append(term)
            new_keys.append(key)

    if new_terms:
        expanded_matrix = incremental_expand_matrix(
            abstract_matrix,
            old_matrix,
            indep_vars,
            dep_vars,
            len(old_keys),
            new_terms,
        )
        source_keys = old_keys + tuple(new_keys)
    else:
        expanded_matrix = old_matrix
        source_keys = old_keys

    num_vars = len(indep_vars) + len(dep_vars)
    return _select_matrix_blocks(expanded_matrix, source_keys, target_keys, num_vars)
