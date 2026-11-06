import os
from functools import partial
import argparse

import numpy as np
import torch
from tqdm import trange


def generate_random_ics(n_ics=200, L=20, n=100, n_terms=10, amplitude=0.5):
    """
    Generate smooth periodic initial conditions.

    The structure follows data_utils/burger.py, with an optional amplitude
    factor to keep the damped-convective dynamics numerically stable.
    """
    x = np.linspace(-L / 2, L / 2, n)
    u0 = np.zeros((n_ics, n))
    for i in range(n_ics):
        coeffs = np.random.uniform(-1, 1, n_terms)
        phase = np.random.uniform(0, 2 * np.pi, n_terms)
        for k in range(n_terms):
            u0[i] += coeffs[k] * np.sin((k + 1) * 2 * np.pi * x / L + phase[k])
    return amplitude * u0, x


def damped_burger(u, dudx, dudxdx, alpha=0.5):
    """
    Damped Burgers equation in evolution form:

        u_t = u_xx - u * u_x - alpha * u

    This corresponds to u_t + u u_x = u_xx - alpha u.
    """
    dudt = dudxdx - u * dudx - alpha * u
    return dudt


def space_derivative(u, dx):
    """Compute u_x and u_xx with periodic finite differences."""
    dudx = (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2 * dx)
    dudxdx = (np.roll(u, -1, axis=-1) - 2 * u + np.roll(u, 1, axis=-1)) / (dx**2)
    return dudx, dudxdx


def gen_data(
    pde,
    init_fn,
    n_ics=200,
    L=20,
    n=100,
    n_terms=10,
    alpha=0.5,
    amplitude=0.5,
    dt=0.002,
    num_steps=1000,
):
    u0, x = init_fn(
        n_ics=n_ics,
        L=L,
        n=n,
        n_terms=n_terms,
        amplitude=amplitude,
    )
    dx = x[1] - x[0]

    u = np.zeros((n_ics, num_steps, n))
    u[:, 0, :] = u0

    for i in trange(1, num_steps):
        dudx, dudxdx = space_derivative(u[:, i - 1, :], dx)
        k1 = dt * pde(u[:, i - 1, :], dudx, dudxdx, alpha=alpha)

        dudx, dudxdx = space_derivative(u[:, i - 1, :] + k1 / 2, dx)
        k2 = dt * pde(u[:, i - 1, :] + k1 / 2, dudx, dudxdx, alpha=alpha)

        dudx, dudxdx = space_derivative(u[:, i - 1, :] + k2 / 2, dx)
        k3 = dt * pde(u[:, i - 1, :] + k2 / 2, dudx, dudxdx, alpha=alpha)

        dudx, dudxdx = space_derivative(u[:, i - 1, :] + k3, dx)
        k4 = dt * pde(u[:, i - 1, :] + k3, dudx, dudxdx, alpha=alpha)

        u[:, i, :] = u[:, i - 1, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    t = np.arange(num_steps) * dt
    return t, x, u


def compute_derivative(t, x, u, alpha=0.5):
    dt = t[1] - t[0]
    dx = x[1] - x[0]

    dudx, dudxdx = space_derivative(u, dx)
    dudt = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    dudtdx, _ = space_derivative(dudt, dx)

    t = t[1:-1]
    x = x[1:-1]
    u = u[:, 1:-1, 1:-1]
    dudx = dudx[:, 1:-1, 1:-1]
    dudxdx = dudxdx[:, 1:-1, 1:-1]
    dudt = dudt[:, :, 1:-1]
    dudtdx = dudtdx[:, :, 1:-1]

    rhs = damped_burger(u, dudx, dudxdx, alpha=alpha)
    residual = dudt - rhs
    residual_l2 = np.linalg.norm(residual.reshape(residual.shape[0], -1), axis=1)
    rhs_l2 = np.linalg.norm(rhs.reshape(rhs.shape[0], -1), axis=1) + 1e-12
    residual_ratio = residual_l2 / rhs_l2

    return t, x, u, dudx, dudxdx, dudt, dudtdx, residual_ratio


get_damped_burger_data = partial(
    gen_data,
    pde=damped_burger,
    init_fn=generate_random_ics,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ics", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--L", type=float, default=20)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--n_terms", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--save_dir", type=str, default="data")
    parser.add_argument("--save_name", type=str, default="train")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t, x, u = get_damped_burger_data(
        n_ics=args.n_ics,
        L=args.L,
        n=args.n,
        n_terms=args.n_terms,
        alpha=args.alpha,
        amplitude=args.amplitude,
        dt=args.dt,
        num_steps=args.num_steps,
    )
    t, x, u, dudx, dudxdx, dudt, dudtdx, residual_ratio = compute_derivative(
        t,
        x,
        u,
        alpha=args.alpha,
    )

    print(t.shape)
    print(x.shape)
    print(u.shape)
    print(dudx.shape)
    print(dudxdx.shape)
    print(dudt.shape)
    print(dudtdx.shape)
    print(f"alpha={args.alpha}")
    print(f"residual_ratio_mean={residual_ratio.mean():.6g}")
    print(f"residual_ratio_max={residual_ratio.max():.6g}")

    t = torch.tensor(t, dtype=torch.float32)
    x = torch.tensor(x, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    dudx = torch.tensor(dudx, dtype=torch.float32)
    dudxdx = torch.tensor(dudxdx, dtype=torch.float32)
    dudt = torch.tensor(dudt, dtype=torch.float32)
    dudtdx = torch.tensor(dudtdx, dtype=torch.float32)

    save_dir = f"{args.save_dir}/damped_burger"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(t, f"{save_dir}/{args.save_name}-t.pt")
    torch.save(x, f"{save_dir}/{args.save_name}-x.pt")
    torch.save(u, f"{save_dir}/{args.save_name}-u.pt")
    torch.save(dudx, f"{save_dir}/{args.save_name}-dudx.pt")
    torch.save(dudxdx, f"{save_dir}/{args.save_name}-dudxdx.pt")
    torch.save(dudt, f"{save_dir}/{args.save_name}-dudt.pt")
    torch.save(dudtdx, f"{save_dir}/{args.save_name}-dudtdx.pt")
