import os
from functools import partial
import argparse

import numpy as np
import torch
from tqdm import trange


def generate_random_ic_params(n_ics=200, L=2 * np.pi, n_terms=10, amplitude=0.5):
    """Generate Fourier coefficients for smooth periodic initial conditions."""
    coeffs = np.random.uniform(-1, 1, size=(n_ics, n_terms))
    phases = np.random.uniform(0, 2 * np.pi, size=(n_ics, n_terms))
    return coeffs, phases, amplitude


def evaluate_random_ics(x, coeffs, phases, amplitude=0.5, L=2 * np.pi):
    """Evaluate u0, u0_x, and u0_xx at x for all initial conditions."""
    x = np.asarray(x)
    n_ics, n_terms = coeffs.shape
    u0 = np.zeros((n_ics,) + x.shape)
    du0dx = np.zeros_like(u0)
    du0dxdx = np.zeros_like(u0)

    for k in range(n_terms):
        wave_number = (k + 1) * 2 * np.pi / L
        arg = wave_number * x[None, ...] + phases[:, k].reshape(n_ics, *([1] * x.ndim))
        coeff = coeffs[:, k].reshape(n_ics, *([1] * x.ndim))
        u0 += coeff * np.sin(arg)
        du0dx += coeff * wave_number * np.cos(arg)
        du0dxdx += -coeff * wave_number**2 * np.sin(arg)

    return amplitude * u0, amplitude * du0dx, amplitude * du0dxdx


def forced_transport_rhs(x, dudx, c=1.0):
    """
    Forced linear transport equation:

        u_t + c * u_x = sin(x)

    Evolution form:

        u_t = sin(x) - c * u_x
    """
    return np.sin(x) - c * dudx


def gen_data(
    pde,
    init_fn,
    n_ics=200,
    L=2 * np.pi,
    n=100,
    n_terms=10,
    c=1.0,
    amplitude=0.5,
    dt=0.002,
    num_steps=1000,
):
    coeffs, phases, amplitude = init_fn(
        n_ics=n_ics,
        L=L,
        n_terms=n_terms,
        amplitude=amplitude,
    )
    x = np.linspace(-L / 2, L / 2, n)
    t = np.arange(num_steps) * dt

    u = np.zeros((n_ics, num_steps, n))
    dudx = np.zeros_like(u)
    dudxdx = np.zeros_like(u)
    dudt = np.zeros_like(u)
    dudtdx = np.zeros_like(u)

    for i in trange(num_steps):
        shifted_x = x[None, :] - c * t[i]
        u0, du0dx, du0dxdx = evaluate_random_ics(
            shifted_x,
            coeffs,
            phases,
            amplitude=amplitude,
            L=L,
        )
        if abs(c) > 1e-12:
            forcing_part = (np.cos(shifted_x) - np.cos(x)[None, :]) / c
            forcing_x = (-np.sin(shifted_x) + np.sin(x)[None, :]) / c
            forcing_xx = (-np.cos(shifted_x) + np.cos(x)[None, :]) / c
            forcing_t = np.sin(shifted_x)
            forcing_tx = np.cos(shifted_x)
        else:
            forcing_part = t[i] * np.sin(x)[None, :]
            forcing_x = t[i] * np.cos(x)[None, :]
            forcing_xx = -t[i] * np.sin(x)[None, :]
            forcing_t = np.sin(x)[None, :]
            forcing_tx = np.cos(x)[None, :]

        u[:, i, :] = u0[:, 0, :] + forcing_part
        dudx[:, i, :] = du0dx[:, 0, :] + forcing_x
        dudxdx[:, i, :] = du0dxdx[:, 0, :] + forcing_xx
        dudt[:, i, :] = -c * du0dx[:, 0, :] + forcing_t
        dudtdx[:, i, :] = -c * du0dxdx[:, 0, :] + forcing_tx

    return t, x, u, dudx, dudxdx, dudt, dudtdx


def compute_derivative(t, x, u, dudx, dudxdx, dudt, dudtdx, c=1.0):
    """Crop analytic fields to match the existing 1D PDE data layout."""
    t_crop = t[1:-1]
    x_crop = x[1:-1]
    u_crop = u[:, 1:-1, 1:-1]
    dudx_crop = dudx[:, 1:-1, 1:-1]
    dudxdx_crop = dudxdx[:, 1:-1, 1:-1]
    dudt_crop = dudt[:, 1:-1, 1:-1]
    dudtdx_crop = dudtdx[:, 1:-1, 1:-1]

    residual = dudt_crop + c * dudx_crop - np.sin(x_crop)[None, None, :]
    residual_l2 = np.linalg.norm(residual.reshape(residual.shape[0], -1), axis=1)
    rhs = np.sin(x_crop)[None, None, :] - c * dudx_crop
    rhs_l2 = np.linalg.norm(rhs.reshape(rhs.shape[0], -1), axis=1) + 1e-12
    residual_ratio = residual_l2 / rhs_l2

    return (
        t_crop,
        x_crop,
        u_crop,
        dudx_crop,
        dudxdx_crop,
        dudt_crop,
        dudtdx_crop,
        residual_ratio,
    )


get_forced_transport_data = partial(
    gen_data,
    pde=forced_transport_rhs,
    init_fn=generate_random_ic_params,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ics", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--L", type=float, default=2 * np.pi)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--n_terms", type=int, default=10)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--save_dir", type=str, default="data")
    parser.add_argument("--save_name", type=str, default="train")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t, x, u, dudx, dudxdx, dudt, dudtdx = get_forced_transport_data(
        n_ics=args.n_ics,
        L=args.L,
        n=args.n,
        n_terms=args.n_terms,
        c=args.c,
        amplitude=args.amplitude,
        dt=args.dt,
        num_steps=args.num_steps,
    )
    t, x, u, dudx, dudxdx, dudt, dudtdx, residual_ratio = compute_derivative(
        t,
        x,
        u,
        dudx,
        dudxdx,
        dudt,
        dudtdx,
        c=args.c,
    )

    print(t.shape)
    print(x.shape)
    print(u.shape)
    print(dudx.shape)
    print(dudxdx.shape)
    print(dudt.shape)
    print(dudtdx.shape)
    print(f"c={args.c}")
    print(f"residual_ratio_mean={residual_ratio.mean():.6g}")
    print(f"residual_ratio_max={residual_ratio.max():.6g}")

    t = torch.tensor(t, dtype=torch.float32)
    x = torch.tensor(x, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    dudx = torch.tensor(dudx, dtype=torch.float32)
    dudxdx = torch.tensor(dudxdx, dtype=torch.float32)
    dudt = torch.tensor(dudt, dtype=torch.float32)
    dudtdx = torch.tensor(dudtdx, dtype=torch.float32)

    save_dir = f"{args.save_dir}/forced_transport"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(t, f"{save_dir}/{args.save_name}-t.pt")
    torch.save(x, f"{save_dir}/{args.save_name}-x.pt")
    torch.save(u, f"{save_dir}/{args.save_name}-u.pt")
    torch.save(dudx, f"{save_dir}/{args.save_name}-dudx.pt")
    torch.save(dudxdx, f"{save_dir}/{args.save_name}-dudxdx.pt")
    torch.save(dudt, f"{save_dir}/{args.save_name}-dudt.pt")
    torch.save(dudtdx, f"{save_dir}/{args.save_name}-dudtdx.pt")
