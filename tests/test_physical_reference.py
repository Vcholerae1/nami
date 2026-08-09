"""Independent homogeneous-medium travel-time references."""

import math

import pytest
import torch

from nami.elastic.elastic2d import elastic2d
from nami.em.em2d_tm import em2d_tm
from nami.em.em3d import em3d
from nami.scalar.scalar2d import scalar2d
from nami.scalar.scalar3d import scalar3d
from nami.wavelets import ricker


def _assert_peak_time_difference(trace, dt, expected):
    peaks = trace.abs().argmax(dim=0)
    observed = abs(int(peaks[1]) - int(peaks[0])) * dt
    assert observed == pytest.approx(expected, abs=2.5 * dt, rel=0.03)


def _subsample_peak(trace):
    magnitude = trace.abs()
    index = int(magnitude.argmax())
    left, centre, right = (
        magnitude[index - 1].item(),
        magnitude[index].item(),
        magnitude[index + 1].item(),
    )
    return index + 0.5 * (left - right) / (left - 2 * centre + right)


def _subsample_time_difference(trace, dt):
    return abs(_subsample_peak(trace[:, 1]) - _subsample_peak(trace[:, 0])) * dt


def test_scalar2d_matches_deepwave_reference_trace():
    """Golden trace generated with Deepwave 0.0.27's scalar propagator."""
    dtype, device = torch.float64, torch.device("cuda")
    ny, nx, nt, dx, dt = 24, 28, 12, 5.0, 5.0e-4
    model = (
        1500
        + torch.arange(ny * nx, dtype=dtype).reshape(ny, nx) % 17
    ).to(device)
    amplitude = torch.zeros(1, 1, nt, dtype=dtype, device=device)
    amplitude[..., 1:4] = torch.tensor(
        [1.0, -0.5, 0.2], dtype=dtype, device=device,
    )
    source = torch.tensor([[[12, 10]]], device=device)
    receivers = torch.tensor([[[12, 13], [10, 14]]], device=device)
    expected = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [-6.668031144824673e-06, 0.0],
            [-4.7585676116040256e-05, 0.0],
            [-0.000192314403503164, 0.0],
            [-0.0005762747157694518, -1.1831322415686242e-09],
            [-0.0014208454965022911, -1.5218568710838913e-08],
            [-0.0030419817337340115, -1.0479359840256936e-07],
            [-0.005840512509581442, -5.121569014407743e-07],
        ],
        dtype=dtype,
        device=device,
    )
    actual = scalar2d(
        model, dx, dt, amplitude, source, receivers, accuracy=2,
        pml_width=4, pml_freq=25.0, storage="none",
    )[:, 0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-15)


def test_scalar2d_homogeneous_travel_time():
    dtype, device = torch.float64, torch.device("cuda")
    ny, nx, nt, dx, dt, velocity = 80, 110, 400, 5.0, 5.0e-4, 2000.0
    model = torch.full((ny, nx), velocity, dtype=dtype, device=device)
    source = torch.tensor([[[40, 20]]], device=device)
    receivers = torch.tensor([[[40, 50], [40, 70]]], device=device)
    amplitude = ricker(20.0, nt, dt, 0.03, dtype)[None, None].to(device)
    trace = scalar2d(
        model, dx, dt, amplitude, source, receivers, accuracy=4,
        pml_width=10, pml_freq=20.0, storage="none",
    )[:, 0]
    _assert_peak_time_difference(trace, dt, 20 * dx / velocity)


def test_scalar3d_homogeneous_travel_time():
    dtype, device = torch.float64, torch.device("cuda")
    shape, nt, dx, dt, velocity = (28, 36, 90), 430, 5.0, 4.0e-4, 2000.0
    model = torch.full(shape, velocity, dtype=dtype, device=device)
    source = torch.tensor([[[14, 18, 15]]], device=device)
    receivers = torch.tensor([[[14, 18, 45], [14, 18, 65]]], device=device)
    amplitude = ricker(20.0, nt, dt, 0.03, dtype)[None, None].to(device)
    trace = scalar3d(
        model, dx, dt, amplitude, source, receivers, accuracy=4,
        pml_width=8, pml_freq=20.0, storage="none",
    )[:, 0]
    _assert_peak_time_difference(trace, dt, 20 * dx / velocity)


def test_elastic2d_homogeneous_p_wave_travel_time():
    dtype, device = torch.float64, torch.device("cuda")
    ny, nx, nt, dx, dt = 80, 110, 500, 5.0, 4.0e-4
    vp, vs, density = 2000.0, 1100.0, 2000.0
    rho = torch.full((ny, nx), density, dtype=dtype, device=device)
    lamb = rho * (vp**2 - 2 * vs**2)
    mu = rho * vs**2
    source = torch.tensor([[[40, 20]]], device=device)
    receivers = torch.tensor([[[40, 50], [40, 70]]], device=device)
    amplitude = (1.0e8 * ricker(20.0, nt, dt, 0.03, dtype))[None, None].to(device)
    trace = elastic2d(
        lamb, mu, rho.reciprocal(), dx, dt, amplitude, source, receivers,
        accuracy=4, pml_width=10, pml_freq=20.0, storage="none",
    )[:, 0]
    _assert_peak_time_difference(trace, dt, 20 * dx / vp)


@pytest.mark.parametrize("dimension", (2, 3))
def test_em_homogeneous_travel_time(dimension):
    dtype, device = torch.float64, torch.device("cuda")
    nt, dx, dt, epsilon_r = 450, 0.01, 2.0e-11, 4.0
    speed = 299_792_458.0 / math.sqrt(epsilon_r)
    if dimension == 2:
        shape = (80, 110)
        source = torch.tensor([[[40, 20]]], device=device)
        receivers = torch.tensor([[[40, 50], [40, 70]]], device=device)
        fn = em2d_tm
    else:
        shape = (24, 28, 90)
        source = torch.tensor([[[12, 14, 15]]], device=device)
        receivers = torch.tensor(
            [[[12, 14, 45], [12, 14, 65]]], device=device,
        )
        fn = em3d
    epsilon = torch.full(shape, epsilon_r, dtype=dtype, device=device)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    amplitude = ricker(5.0e8, nt, dt, 2.0e-9, dtype)[None, None].to(device)
    trace = fn(
        epsilon, sigma, mu, dx, dt, amplitude, source, receivers,
        accuracy=4, pml_width=8 if dimension == 3 else 10, storage="none",
    )[:, 0]
    _assert_peak_time_difference(trace, dt, 20 * dx / speed)


def test_scalar2d_grid_refinement_reduces_travel_time_error():
    dtype, device, velocity = torch.float64, torch.device("cuda"), 2100.0

    def error(dx):
        dt = dx * 1.0e-4
        ny, nx, nt = round(400 / dx), round(600 / dx), round(0.22 / dt)
        model = torch.full((ny, nx), velocity, dtype=dtype, device=device)
        source = torch.tensor([[[ny // 2, round(100 / dx)]]], device=device)
        receivers = torch.tensor(
            [[[ny // 2, round(250 / dx)], [ny // 2, round(350 / dx)]]],
            device=device,
        )
        amplitude = ricker(18.0, nt, dt, 0.04, dtype)[None, None].to(device)
        trace = scalar2d(
            model, dx, dt, amplitude, source, receivers, accuracy=2,
            pml_width=max(6, round(50 / dx)), pml_freq=18.0, storage="none",
        )[:, 0]
        return abs(_subsample_time_difference(trace, dt) - 100 / velocity)

    assert error(5.0) < 0.5 * error(10.0)


def test_elastic2d_grid_refinement_reduces_p_wave_time_error():
    dtype, device = torch.float64, torch.device("cuda")
    vp, vs, density = 2100.0, 1150.0, 2000.0

    def error(dx):
        dt = dx * 8.0e-5
        ny, nx, nt = round(400 / dx), round(600 / dx), round(0.3 / dt)
        rho = torch.full((ny, nx), density, dtype=dtype, device=device)
        lamb = rho * (vp**2 - 2 * vs**2)
        mu = rho * vs**2
        source = torch.tensor([[[ny // 2, round(100 / dx)]]], device=device)
        receivers = torch.tensor(
            [[[ny // 2, round(250 / dx)], [ny // 2, round(350 / dx)]]],
            device=device,
        )
        amplitude = (
            1.0e8 * ricker(10.0, nt, dt, 0.08, dtype)
        )[None, None].to(device)
        trace = elastic2d(
            lamb, mu, rho.reciprocal(), dx, dt, amplitude, source, receivers,
            accuracy=2, pml_width=max(6, round(50 / dx)), pml_freq=10.0,
            storage="none",
        )[:, 0]
        return abs(_subsample_time_difference(trace, dt) - 100 / vp)

    assert error(5.0) < 0.5 * error(10.0)


def test_em3d_grid_refinement_reduces_travel_time_error():
    dtype, device = torch.float64, torch.device("cuda")
    epsilon_r = 4.0
    speed = 299_792_458.0 / math.sqrt(epsilon_r)

    def error(dx):
        dt = dx * 2.0e-9
        nz, ny, nx = round(0.32 / dx), round(0.36 / dx), round(1.0 / dx)
        nt = round(12.0e-9 / dt)
        epsilon = torch.full(
            (nz, ny, nx), epsilon_r, dtype=dtype, device=device,
        )
        sigma, mu = torch.zeros_like(epsilon), torch.ones_like(epsilon)
        source = torch.tensor(
            [[[nz // 2, ny // 2, round(0.2 / dx)]]], device=device,
        )
        receivers = torch.tensor(
            [[
                [nz // 2, ny // 2, round(0.4 / dx)],
                [nz // 2, ny // 2, round(0.6 / dx)],
            ]],
            device=device,
        )
        amplitude = ricker(
            5.0e8, nt, dt, 4.0e-9, dtype,
        )[None, None].to(device)
        trace = em3d(
            epsilon, sigma, mu, dx, dt, amplitude, source, receivers,
            accuracy=2, pml_width=max(5, round(0.1 / dx)), storage="none",
        )[:, 0]
        return abs(_subsample_time_difference(trace, dt) - 0.2 / speed)

    assert error(0.01) < 0.1 * error(0.02)
