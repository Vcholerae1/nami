"""nami scalar3d Born: Born linearity, autograd gradcheck, and storage-mode tests."""

import torch

from nami.scalar.scalar3d import scalar3d
from nami.scalar.scalar3d_born import scalar3d_born


def build_case(
    dtype=torch.float64,
    nz=10,
    ny=14,
    nx=14,
    nt=12,
    pml=3,
    n_shots=1,
    device="cuda:0",
    seed=0,
):
    """Small three-layer random model; one source near the PML, receivers
    nearby.

    grid_spacing = 5 m, dt = 0.5 ms.  dt satisfies the CFL limit
    max_dt ~= 8.2e-4 s for v_max ~= 2100 m/s.
    """
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    v = 1500 + 300 * torch.rand(nz, ny, nx, generator=g)
    v[nz // 2 :] += 300  # two layers in z
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(seed + 2)
    scatter = 0.005 * (torch.rand(nz, ny, nx, generator=g2) - 0.5)
    grid_spacing, dt = 5.0, 5.0e-4
    # near the z/y/x = 0 PML so the CPML branch is exercised
    srcs = torch.tensor([[[1, 2, 2]]])[:n_shots]
    recs = torch.tensor(
        [
            [
                [1, 3, 3],
                [1, 4, 2],
                [2, 2, 4],
                [2, 4, 3],
            ]
        ]
    )[:n_shots]
    bg_recs = torch.tensor(
        [
            [
                [2, 3, 3],
                [3, 4, 4],
                [3, 2, 2],
            ]
        ]
    )[:n_shots]
    amp = torch.zeros(n_shots, srcs.shape[1], nt, dtype=dtype)
    amp[:, :, 0] = 1.0
    amp[:, :, 1] = -0.5
    amp[:, :, 2] = 0.2
    return {
        "v": v,
        "scatter": scatter,
        "grid_spacing": grid_spacing,
        "dt": dt,
        "nt": nt,
        "pml": pml,
        "srcs": srcs,
        "recs": recs,
        "bg_recs": bg_recs,
        "amp": amp,
        "n_shots": n_shots,
        "dtype": dtype,
        "device": device,
    }


def _run_nami(c, v, scatter, amp, accuracy=2, **kw):
    dev = c["device"]
    return scalar3d_born(
        v,
        scatter,
        c["grid_spacing"],
        c["dt"],
        source_amplitudes=amp,
        source_locations=c["srcs"].to(dev),
        receiver_locations=c["recs"].to(dev),
        bg_receiver_locations=c["bg_recs"].to(dev),
        accuracy=accuracy,
        pml_width=c["pml"],
        pml_freq=25.0,
        nt=c["nt"],
        **kw,
    )
def test_born_linearity():
    """r_born(delta) ~= r_full(delta) - r_bg with O(delta^2) error."""
    dtype = torch.float64
    c = build_case(dtype=dtype, nz=8, ny=10, nx=10, nt=10, pml=3,
                   device="cuda:0", seed=1)
    dev = c["device"]
    v = c["v"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)
    srcs = c["srcs"].to(dev)
    recs = c["recs"].to(dev)
    g = torch.Generator(device="cpu")
    g.manual_seed(9)
    base = (torch.rand(c["v"].shape, generator=g, dtype=dtype) - 0.5).to(dev)

    def full(delta):
        return scalar3d(
            v + delta,
            c["grid_spacing"],
            c["dt"],
            source_amplitudes=amp,
            source_locations=srcs,
            receiver_locations=recs,
            accuracy=2,
            pml_width=c["pml"],
            pml_freq=25.0,
            nt=c["nt"],
        )

    def born(delta):
        return scalar3d_born(
            v,
            delta,
            c["grid_spacing"],
            c["dt"],
            source_amplitudes=amp,
            source_locations=srcs,
            receiver_locations=recs,
            accuracy=2,
            pml_width=c["pml"],
            pml_freq=25.0,
            nt=c["nt"],
        )

    r_bg = full(torch.zeros_like(base))
    r1 = born(base)
    r2 = born(2 * base)
    signal = r1.abs().max().item()
    # the scattered field is exactly linear in the scattering potential:
    # doubling scatter doubles the scattered field (bit-for-bit, since all
    # update coefficients are scatter-independent)
    lin = (r2 - 2 * r1).abs().max().item()
    print(f"scatter exact linearity (2*r1 - r2) max: {lin:.3e}")
    assert lin == 0.0, f"scattered field not exactly linear in scatter: {lin}"
    resid_1 = (r1 - (full(base) - r_bg)).abs().max().item()
    resid_2 = (r2 - (full(2 * base) - r_bg)).abs().max().item()
    print(f"signal(delta) {signal:.3e}, residual(delta) {resid_1:.3e}, "
          f"residual(2*delta) {resid_2:.3e}")
    assert resid_1 < 1e-2 * signal, f"Born linearity residual(delta): {resid_1}"
    # O(delta^2): doubling delta multiplies the residual by ~4
    ratio = resid_2 / (resid_1 + 1e-300)
    print(f"residual ratio 2x: {ratio:.3f}")
    assert 2.5 < ratio < 6.0, f"residual did not scale quadratically: {ratio}"


def test_gradcheck():
    """Numerical gradient check for v, scatter, and source_amplitudes."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, nz=8, ny=10, nx=10, nt=8, pml=3, device="cuda:0", seed=1
    )
    dev = c["device"]

    def fn(v, scatter, amp):
        return _run_nami(c, v, scatter, amp)

    ok = torch.autograd.gradcheck(
        fn,
        (
            c["v"].to(dev, dtype).requires_grad_(True),
            c["scatter"].to(dev, dtype).requires_grad_(True),
            c["amp"].to(dev, dtype).requires_grad_(True),
        ),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        fast_mode=True,
        nondet_tol=1e-8,
        raise_exception=False,
    )
    print("gradcheck nami scalar3d_born (v, scatter, amp):", ok)
    assert ok


def test_storage_false_is_forward_only():
    """storage='none' runs the forward but backward raises RuntimeError."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, nz=8, ny=10, nx=10, nt=8, pml=3, device="cuda:0", seed=1
    )
    dev = c["device"]
    v = c["v"].to(dev, dtype).requires_grad_(True)
    scatter = c["scatter"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype)
    out, out_bg = _run_nami(c, v, scatter, amp, storage="none")
    assert out.shape == (c["nt"], 1, c["recs"].shape[1])
    assert out_bg.shape == (c["nt"], 1, c["bg_recs"].shape[1])
    try:
        torch.autograd.grad(out.square().sum() + out_bg.square().sum(), v)
    except RuntimeError:
        return
    raise AssertionError("storage='none' backward should raise RuntimeError")


def test_checkpoint_forward_and_gradient_parity():
    """ckpt_steps=N reproduces full-storage forward and gradients."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, nz=8, ny=10, nx=10, nt=12, pml=3, device="cuda:0",
        seed=1,
    )
    dev = c["device"]
    v = c["v"].to(dev, dtype).requires_grad_(True)
    scatter = c["scatter"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype)

    def run(ckpt):
        return _run_nami(
            c, v, scatter, amp, storage="auto", ckpt_steps=ckpt,
        )

    r0, r0_bg = run(0)
    r_full, r_bg_full = r0.detach(), r0_bg.detach()
    g_full = torch.autograd.grad(r0.square().sum() + r0_bg.square().sum(), v)[0]
    for ckpt in (2, 4, 7, 11, 12):
        r, r_bg = run(ckpt)
        assert (r.detach() - r_full).abs().max().item() == 0.0, (
            f"ckpt_steps={ckpt} fwd r"
        )
        assert (r_bg.detach() - r_bg_full).abs().max().item() == 0.0, (
            f"ckpt_steps={ckpt} fwd r_bg"
        )
        g = torch.autograd.grad(r.square().sum() + r_bg.square().sum(), v)[0]
        rel = (g - g_full).abs().max().item() / (
            g_full.abs().max().item() + 1e-300
        )
        assert rel == 0.0, f"ckpt_steps={ckpt} grad rel err {rel}"
