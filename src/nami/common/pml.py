"""Acoustic/elastic C-PML profiles (Pasalic & McGarry).

Torch implementation of the C-PML profiles (bit-for-bit reproducible in
float64).
"""

import math
from collections.abc import Sequence

import torch

from .fd import DIFF1, check_accuracy


def set_acoustic_pml_profiles(
    pml_width: Sequence[int],
    fd_pad: Sequence[int],
    dt: float,
    grid_spacing: Sequence[float],
    max_vel: float,
    pml_freq: float,
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
    accuracy: int = 2,
) -> list[torch.Tensor]:
    """Acoustic CPML profiles: [ay, by, dbydy, ax, bx, dbxdx].

    ``shape`` is the padded grid shape [ny, nx]; returned 1-D profiles are
    that length (the kernels index them directly by cell coordinate).
    ``db = diff1(b, accuracy, 1/grid_spacing)`` matches the reference
    ``regular_grid.set_pml_profiles`` for accuracy 2/4/6/8.
    """
    accuracy = check_accuracy(accuracy)
    ndim = len(shape)
    pml_start = [
        [
            float(fd_pad[dim * 2] + pml_width[dim * 2]),
            float(shape[dim] - 1 - fd_pad[dim * 2 + 1] - pml_width[dim * 2 + 1]),
        ]
        for dim in range(ndim)
    ]
    physical_widths = [pml_width[i] * grid_spacing[i // 2] for i in range(2 * ndim)]
    max_pml = max(physical_widths) if physical_widths else 0.0

    profiles: list[torch.Tensor] = []
    for dim in range(ndim):
        n = shape[dim]
        a, b = _setup_pml_acoustic(
            pml_width[2 * dim : 2 * dim + 2],
            pml_start[dim],
            max_pml,
            dt,
            n,
            max_vel,
            dtype,
            device,
            pml_freq,
        )
        db = _diff1_pml(b, accuracy, 1.0 / grid_spacing[dim])
        profiles.extend([a, b, db])
    return profiles


def _diff1_pml(b: torch.Tensor, accuracy: int, scale: float) -> torch.Tensor:
    """First derivative of a 1-D profile."""
    coeffs = DIFF1[accuracy]
    pad = accuracy // 2
    terms = []
    for k, c in enumerate(coeffs, start=1):
        if k == pad:
            terms.append(c * (b[pad + k :] - b[: -pad - k]))
        else:
            terms.append(
                c * (b[pad + k : -(pad - k)] - b[pad - k : -(pad + k)])
            )
    return torch.nn.functional.pad(sum(terms) * scale, (pad, pad))


def _setup_pml_acoustic(
    pml_width: Sequence[int],
    pml_start: Sequence[float],
    max_pml: float,
    dt: float,
    n: int,
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """a, b profiles for the acoustic C-PML."""
    alpha0 = math.pi * pml_freq
    if max_pml == 0:  # no PML: all-zero profiles (avoid -inf/NaN sigma0)
        return torch.zeros(n, dtype=dtype, device=device), torch.zeros(
            n, dtype=dtype, device=device
        )
    sigma0 = -(1 + 2) * max_vel * math.log(0.001) / (2 * max_pml)
    x = torch.arange(n, device=device, dtype=dtype)
    pml_frac0 = (
        torch.zeros_like(x) if pml_width[0] == 0 else (pml_start[0] - x) / pml_width[0]
    )
    pml_frac1 = (
        torch.zeros_like(x) if pml_width[1] == 0 else (x - pml_start[1]) / pml_width[1]
    )
    pml_frac = torch.clamp(torch.maximum(pml_frac0, pml_frac1), 0, 1)
    sigma = sigma0 * pml_frac**2
    alpha = alpha0 * (1 - pml_frac)
    sigmaalpha = sigma + alpha
    a = torch.exp(-sigmaalpha * abs(dt))
    b = sigma / sigmaalpha * (a - 1)
    a[pml_frac == 0] = 0
    return a, b
