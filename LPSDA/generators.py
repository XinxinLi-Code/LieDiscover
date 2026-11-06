"""Sources and coordinate adapters for the pinned LieNLSD solver."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .coordinates import PROTOCOL, physical_points, pushforward
else:
    from coordinates import PROTOCOL, physical_points, pushforward


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = '7d85a987ac3c04ba9d202ffd0a7c5304f10a8ef9'
DEFAULT_TERMS = ('1', 't', 'x', 'u', 't**2', 'x**2', 'u**2', 't*x', 't*u', 'x*u')


def add_source_arguments(parser):
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--output-root', default=str(ROOT / 'result' / 'pde_solving'))
    parser.add_argument('--generator-path', default=None)
    parser.add_argument('--theta-terms-path', default=None)






def read_generator(path, terms_path=None):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / 'generator.pt'
    weights = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(weights, torch.Tensor) or weights.ndim != 3 or weights.shape[1] != 3 or weights.shape[0] < 1:
        raise ValueError(f'Expected nonempty [generators, 3, terms] coefficients: {path}')
    if not torch.isfinite(weights).all():
        raise ValueError(f'Nonfinite saved generator: {path}')
    terms_path = Path(terms_path).resolve() if terms_path else path.with_name('theta_terms.txt')
    if terms_path.is_file():
        terms = tuple(s.strip() for s in terms_path.read_text().strip().strip('[]').replace('\n', ',').split(',') if s.strip())
    elif weights.shape[2] == len(DEFAULT_TERMS):
        terms = DEFAULT_TERMS
    else:
        raise FileNotFoundError(f'Missing dynamic function library: {terms_path}')
    if len(terms) != weights.shape[2]:
        raise ValueError('Generator columns and theta_terms.txt disagree.')
    return weights, terms, path, terms_path






def configure_sources(args):
    if args.pde not in ('burger', 'heat', 'kdv'):
        raise ValueError('PDE solving supports only burger, heat, and kdv.')
    args.data_dir = str(Path(args.data_dir).resolve())
    args.output_root = str(Path(args.output_root).resolve())
    args.upstream_commit = UPSTREAM_COMMIT
    args.solving_protocol = PROTOCOL
    args.spatial_sampling = 'first_10_saved_points'
    args.metric_version = 'official_time_sum_configured_batch_divisor'
    if args.n_transform < 1 or not 0 <= args.p_original <= 1:
        raise ValueError('Require n_transform >= 1 and 0 <= p_original <= 1.')
    path = args.generator_path or ROOT / 'vis' / f'{args.pde}_{args.sample}_pruned'
    weights, terms, path, terms_path = read_generator(path, args.theta_terms_path)
    if args.n_delta is not None and args.n_delta != weights.shape[0]:
        raise ValueError('Requested n_delta does not match the saved generator count.')
    args.n_delta = weights.shape[0]
    args.generator_path = str(path)
    args.theta_terms_path = str(terms_path) if terms_path.is_file() else None
    args.theta_terms = ','.join(terms)
    args.generator_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    args.generator_storage_dtype = str(weights.dtype)
    args.generator_evaluation_dtype = 'torch.float32'
    if len(args.sigma) == 1:
        args.sigma = args.sigma * args.n_delta
    elif len(args.sigma) != args.n_delta:
        raise ValueError('Supply one common sigma or one value per generator.')
    if any(not np.isfinite(v) or v < 0 for v in args.sigma):
        raise ValueError('sigma must be finite and nonnegative.')
    return args


def build_generator(args, vfop):
    weights, terms, _, _ = read_generator(args.generator_path, args.theta_terms_path)
    weights = weights.to(device=args.device, dtype=torch.float32)
    trees = tuple(ast.parse(term.replace('^', '**'), mode='eval') for term in terms)
    frame = args.coordinate_map

    def field(xtu):
        points = physical_points(xtu, frame)
        t, x, u = points[:, 0], points[:, 1], points[:, 2]
        theta = torch.stack([_eval_theta_ast(tree, t, x, u) for tree in trees], dim=1)
        delta = torch.einsum('kjl,il->ijk', weights, theta)
        delta = pushforward(delta, frame)
        if not torch.isfinite(delta).all():
            raise FloatingPointError('Nonfinite explicit generator evaluation; no generator was discarded.')
        delta = vfop.normalize_delta(delta)
        # Use the upstream normalization and orthogonalization.
        for i in range(delta.shape[-1]):
            for j in range(i):
                delta[..., i] = delta[..., i] - vfop.inner_product(delta[..., j], delta[..., i]) * delta[..., j]
        return delta

    return field


def record_augmentation(args, reject_list, print_out):
    """Observe official rejection/fallback without changing samples or RNG."""
    rejected = torch.stack(reject_list).detach().cpu().bool()
    accepted = (~rejected[1:]).sum(dim=0)
    fallback_probability = args.p_original + (1 - args.p_original) * (
        1 + rejected[1:].sum(dim=0).double()) / args.n_transform
    report = {
        'protocol': args.solving_protocol,
        'method': args.delta_exp,
        'generators_used': getattr(args, 'n_delta', None),
        'generator_selection': 'all_saved_channels',
        'augmented_attempts': int(rejected[1:].numel()),
        'accepted_augmented_trajectories': int(accepted.sum()),
        'accepted_versions_per_trajectory': accepted.tolist(),
        'effective_original_probability_per_trajectory': fallback_probability.tolist(),
    }
    (Path(args.exp_path) / 'augmentation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print_out(f"Accepted augmentation: {report['accepted_augmented_trajectories']}/{report['augmented_attempts']}; per trajectory: {accepted.tolist()}")
    if args.n_transform > 1 and not bool(accepted.any()):
        print_out('All augmented trajectories were rejected by the official rule; sampling falls back to original trajectories.')


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record_inputs(args, train_loader, val_loader, test_loader):
    """Freeze input provenance for later comparisons; do not alter the data."""
    datasets = {}
    for split, loader in (('train', train_loader), ('val', val_loader), ('test', test_loader)):
        data = loader.dataset.data
        directory = Path(args.data_dir) / args.pde
        keys = ('t', 'x', 'u', 'dudt', 'dudx', 'dudxdx') + (('dudxdxdx',) if args.pde == 'kdv' else ())
        datasets[split] = {
            'shape_after_official_crop': list(data['u'].shape),
            'coordinate_map': loader.dataset.coordinate_map,
            'coordinate_values': 'normalized_index_grid',
            'first_last_t': [float(data['t'][0, 0, 0]), float(data['t'][0, -1, 0])],
            'first_last_x': [float(data['x'][0, 0, 0]), float(data['x'][0, 0, -1])],
            'physical_first_last_t': [loader.dataset.coordinate_map['t_origin'],
                                      loader.dataset.coordinate_map['t_origin'] +
                                      (loader.dataset.coordinate_map['nt'] - 1) * loader.dataset.coordinate_map['dt']],
            'physical_first_last_x': [loader.dataset.coordinate_map['x_origin'],
                                      loader.dataset.coordinate_map['x_origin'] +
                                      (loader.dataset.coordinate_map['nx'] - 1) * loader.dataset.coordinate_map['dx']],
            'source_sha256': {key: _file_sha256(directory / f'{split}-{key}.pt') for key in keys},
        }
    (Path(args.exp_path) / 'input_provenance.json').write_text(json.dumps(datasets, indent=2) + '\n')


def _theta_name_value(name, t, x, u):
    if name == 't':
        return t
    if name == 'x':
        return x
    if name == 'u':
        return u
    if name == 'pi':
        return torch.ones_like(t) * np.pi
    if name == 'E':
        return torch.ones_like(t) * np.e
    raise ValueError(f'Unsupported theta symbol {name!r}; LPSDA dynamic theta supports only t, x, u.')


def _eval_theta_ast(node, t, x, u):
    if isinstance(node, ast.Expression):
        return _eval_theta_ast(node.body, t, x, u)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return torch.ones_like(t) * float(node.value)
        raise ValueError(f'Unsupported theta constant {node.value!r}.')
    if isinstance(node, ast.Name):
        return _theta_name_value(node.id, t, x, u)
    if isinstance(node, ast.UnaryOp):
        value = _eval_theta_ast(node.operand, t, x, u)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = _eval_theta_ast(node.left, t, x, u)
        right = _eval_theta_ast(node.right, t, x, u)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if len(node.args) != 1:
            raise ValueError(f'Theta function {node.func.id!r} expects one argument.')
        arg = _eval_theta_ast(node.args[0], t, x, u)
        if node.func.id == 'sin':
            return torch.sin(arg)
        if node.func.id == 'cos':
            return torch.cos(arg)
        if node.func.id == 'tan':
            return torch.tan(arg)
        if node.func.id == 'exp':
            return torch.exp(arg)
        if node.func.id == 'log':
            return torch.log(arg)
        if node.func.id == 'sqrt':
            return torch.sqrt(arg)
        if node.func.id == 'abs':
            return torch.abs(arg)
        if node.func.id == 'inv':
            return 1.0 / arg
    raise ValueError(f'Unsupported theta expression node: {ast.dump(node)}')


def eval_theta_term(term, t, x, u):
    term = str(term).strip().replace('^', '**')
    tree = ast.parse(term, mode='eval')
    return _eval_theta_ast(tree, t, x, u)


def evaluate_theta_terms(theta_terms, t, x, u):
    return torch.stack([eval_theta_term(term, t, x, u) for term in theta_terms], dim=1)

