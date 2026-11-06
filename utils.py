import math
import torch
import numpy as np
import os

def basis_sparsification(
    Q,
    epsilon1,
    epsilon2,
    device,
    log_interval,
    max_iter=1000,
    verbose=True,
    **kwargs,
):

    def soft_threshold(x, epsilon):
        return torch.sign(x) * torch.clamp(torch.abs(x) - epsilon, min=0.0)

    Q = Q.to(device)
    if Q.ndim != 2:
        raise ValueError("Q must be a matrix with basis vectors in its columns.")
    if not Q.is_floating_point():
        Q = Q.to(dtype=torch.float64)
    n, d = Q.shape
    if n == 0 or d == 0:
        return Q
    if not bool(torch.isfinite(Q).all()):
        raise ValueError("Q must contain finite coefficients.")
    beta = min(n, d)
    beta_max = 1e10
    rho0 = 1.9
    etaA = 1.02 * torch.linalg.norm(Q, ord=2) ** 2
    etaB = 1.02
    R = torch.eye(d, device=Q.device, dtype=Q.dtype)
    Z = Q @ R
    Lambda = torch.zeros_like(Q)
    k = 0
    if max_iter is not None and max_iter <= 0:
        raise ValueError("max_iter must be positive or None.")

    while True:
        R_pre = R.clone()
        Z_pre = Z.clone()
        tempR = R - (Q.T @ (Lambda + beta * (Q @ R - Z))) / (beta * etaA)
        U, S, Vh = torch.linalg.svd(tempR.to(torch.float64))
        U, Vh = U.to(Q.dtype), Vh.to(Q.dtype)
        R = U @ Vh
        tempZ = Z + (Lambda + beta * (Q @ R - Z)) / (beta * etaB)
        Z = soft_threshold(tempZ, 1.0 / (beta * etaB))
        Lambda = Lambda + beta * (Q @ R - Z)
        condition = beta * max(math.sqrt(etaA) * torch.linalg.norm(R - R_pre, ord=float('inf')).item(), math.sqrt(etaB) * torch.linalg.norm(Z - Z_pre, ord=float('inf')).item())
        if condition < epsilon2:
            rho = rho0
        else:
            rho = 1.0
        beta = min(beta_max, rho * beta)
        loss = torch.sum(torch.abs(Q @ R)).item()
        if verbose and log_interval and (k + 1) % log_interval == 0:
            print(f'Sparsify iter {k}, loss: {loss:.4f}')
        k += 1
        if torch.linalg.norm(Q @ R - Z, ord=float('inf')) < epsilon1 and condition <= epsilon2:
            return Q @ R
        if max_iter is not None and k >= max_iter:
            if verbose:
                print(f'Sparsify reached max_iter={max_iter}, returning current basis.')
            return Q @ R

save_vis_path = './vis'

BURGERS_THETA_LABELS = [
    r'$1$', r'$t$', r'$x$', r'$u$', r'$t^2$', r'$x^2$', r'$u^2$',
    r'$tx$', r'$tu$', r'$xu$',
]
DAMPED_BURGER_THETA_LABELS = [
    r'$1$', r'$t$', r'$x$', r'$u$', r'$e^{-t}$', r'$t^2$', r'$x^2$',
    r'$u^2$', r'$tx$', r'$tu$', r'$xu$',
]
FORCED_TRANSPORT_THETA_LABELS = [
    r'$1$', r'$t$', r'$x$', r'$u$', r'$\sin x$', r'$\cos x$',
    r'$\sin(x-t)$', r'$\cos(x-t)$', r'$t^2$', r'$x^2$', r'$u^2$',
    r'$tx$', r'$tu$', r'$xu$',
]


def _fallback_theta_labels(num_terms):
    return [rf'$\theta_{{{i}}}$' for i in range(num_terms)]


def _theta_labels_for_vis(task, num_terms):
    if task in ['heat', 'burger', 'kdv', 'damped_burger', 'forced_transport'] and num_terms == 10:
        return BURGERS_THETA_LABELS
    if task == 'damped_burger' and num_terms == 11:
        return DAMPED_BURGER_THETA_LABELS
    if task == 'forced_transport' and num_terms == 14:
        return FORCED_TRANSPORT_THETA_LABELS
    if task == 'wave' and num_terms == 15:
        return [
            r'$1$', r'$t$', r'$x$', r'$y$', r'$u$', r'$t^2$', r'$x^2$',
            r'$y^2$', r'$u^2$', r'$tx$', r'$ty$', r'$tu$', r'$xy$',
            r'$xu$', r'$yu$',
        ]
    if task in ['rd', 'schrodinger'] and num_terms == 21:
        return [
            r'$1$', r'$t$', r'$x$', r'$y$', r'$u$', r'$v$', r'$t^2$',
            r'$x^2$', r'$y^2$', r'$u^2$', r'$v^2$', r'$tx$', r'$ty$',
            r'$tu$', r'$tv$', r'$xy$', r'$xu$', r'$xv$', r'$yu$',
            r'$yv$', r'$uv$',
        ]
    return _fallback_theta_labels(num_terms)


def _reshape_generators_for_vis(Vh, task):
    n = Vh.shape[0]
    if task == 'top':
        variable_dim = 4
    elif task in ['heat', 'burger', 'damped_burger', 'forced_transport', 'kdv']:
        variable_dim = 3
    elif task == 'wave':
        variable_dim = 4
    elif task in ['rd', 'schrodinger']:
        variable_dim = 5
    else:
        raise NotImplementedError(f'Unsupported visualization task: {task}')
    if Vh.shape[1] % variable_dim != 0:
        raise ValueError(
            f'Cannot reshape Vh with shape {tuple(Vh.shape)} for task {task}: '
            f'column count is not divisible by variable_dim={variable_dim}.'
        )
    return Vh.reshape(n, variable_dim, Vh.shape[1] // variable_dim)


def vis(S, Vh, task, sample):
    import matplotlib.pyplot as plt

    S = S.cpu().numpy()
    plt.plot(range(S.shape[0]-Vh.shape[0], S.shape[0]+1), S[S.shape[0]-Vh.shape[0]-1:S.shape[0]])
    plt.xticks(range(S.shape[0]-Vh.shape[0], S.shape[0]+1))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Singular Value')
    if not os.path.exists(f'{save_vis_path}/{task}_{sample}'):
        os.makedirs(f'{save_vis_path}/{task}_{sample}')
    plt.savefig(f'{save_vis_path}/{task}_{sample}/singular_value.png', bbox_inches='tight')
    plt.clf()

    n = Vh.shape[0]
    W = _reshape_generators_for_vis(Vh, task)
    theta_labels = _theta_labels_for_vis(task, W.shape[2])
    torch.save(W, f'{save_vis_path}/{task}_{sample}/generator.pt')
    W = W.cpu().numpy()
    for i in range(n):
        plt.imshow(W[i])
        if task == 'top':
            plt.colorbar()
            plt.xticks(ticks=np.arange(4), labels=[r'$p^0$', r'$p^1$', r'$p^2$', r'$p^3$'])
            plt.yticks(ticks=np.arange(4), labels=[r'$\partial_{p^0}$', r'$\partial_{p^1}$', r'$\partial_{p^2}$', r'$\partial_{p^3}$'])
        elif task in ['heat', 'burger', 'damped_burger', 'forced_transport', 'kdv']:
            plt.colorbar(shrink=0.5)
            plt.xticks(ticks=np.arange(W.shape[2]), labels=theta_labels)
            plt.yticks(ticks=np.arange(3), labels=[r'$\partial_t$', r'$\partial_x$', r'$\partial_u$'])
        elif task == 'wave':
            plt.colorbar(shrink=0.5)
            plt.xticks(ticks=np.arange(W.shape[2]), labels=theta_labels)
            plt.yticks(ticks=np.arange(4), labels=[r'$\partial_t$', r'$\partial_x$', r'$\partial_y$', r'$\partial_u$'])
        elif task in ['rd', 'schrodinger']:
            plt.colorbar(shrink=0.5)
            plt.xticks(ticks=np.arange(W.shape[2]), labels=theta_labels)
            plt.yticks(ticks=np.arange(5), labels=[r'$\partial_t$', r'$\partial_x$', r'$\partial_y$', r'$\partial_u$', r'$\partial_v$'])
        plt.tick_params(axis='both', which='both', length=0)
        plt.savefig(f'{save_vis_path}/{task}_{sample}/generator_{S.shape[0] - n + i + 1}.png', bbox_inches='tight')
        plt.clf()
