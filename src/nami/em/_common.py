"""Shared helpers for the EM propagators (em2d_tm / em3d and Born variants)."""

import math

import torch

from ..common.fd import check_accuracy

# vacuum permittivity / permeability / speed of light (SI)
EPS0 = 8.8541878128e-12  # F/m
MU0 = 1.2566370614359173e-06  # H/m


def _compile_material_coefficients(epsilon_r, sigma_r, mu_r, dt):
    """Compile material models into Maxwell update coefficients.

    Same formulas as ``compile_material_coefficients`` (and the
    FDTD convention used by the eager reference): ca/cb for the E update
    with a conductivity loss averaged over the time step, cq = dt/mu for
    the H update.  Differentiable, so gradients flow from ca/cb/cq back to
    epsilon/sigma/mu.
    """
    eps = epsilon_r * EPS0
    mu = mu_r * MU0
    denom = 1.0 + sigma_r * dt / (2.0 * eps)
    ca = (1.0 - sigma_r * dt / (2.0 * eps)) / denom
    cb = (dt / eps) / denom
    cq = dt / mu
    return ca, cb, cq


def _pml_profile_1d(
    pml_width,
    pml_start,
    dt,
    n,
    dtype,
    device,
    half,
    accuracy=2,
    n_power=4,
    eps=1e-9,
    grid_spacing=1.0,
    eps_scale=EPS0,
):
    """a, b, k CPML profiles along one dimension.

    ``accuracy`` is validated for API parity but does not affect the EM
    a/b/k profiles (the CPML recursion is FD-order independent).
    """
    check_accuracy(accuracy)
    k_max_cpml = 5.0  # maximum coordinate stretching factor
    alpha_max_cpml = 0.008  # maximum frequency shift

    a = torch.zeros(n, dtype=dtype, device=device)
    b = torch.zeros(n, dtype=dtype, device=device)
    k = torch.ones(n, dtype=dtype, device=device)

    if pml_width[0] == 0 and pml_width[1] == 0:
        return a, b, k

    sigma0 = (n_power + 1) / (150.0 * math.pi * grid_spacing)

    x = torch.arange(n, dtype=dtype, device=device)
    if half:
        x = x + 0.5

    for side in range(2):
        if pml_width[side] == 0:
            continue
        abscissa = pml_start[0] - x if side == 0 else x - pml_start[1]
        mask = abscissa >= 0
        # Normalized distance into the PML (0 at inner edge, 1 at outer edge)
        abscissa_norm = torch.clamp(abscissa / pml_width[side], 0, 1)

        sigma = sigma0 * (abscissa_norm**n_power)
        k_side = 1.0 + (k_max_cpml - 1.0) * (abscissa_norm**n_power)
        alpha = alpha_max_cpml * (1.0 - abscissa_norm) + 0.1 * alpha_max_cpml

        k = torch.where(mask, k_side, k)

        b_side = torch.exp(-(sigma / k_side + alpha) * dt / eps_scale)
        b = torch.where(mask, b_side, b)

        denom = k_side * (sigma + k_side * alpha) + eps
        a_side = sigma * (b_side - 1.0) / denom
        a_side = torch.where(sigma > 1e-6, a_side, torch.zeros_like(a_side))
        a = torch.where(mask, a_side, a)

    return a, b, k
