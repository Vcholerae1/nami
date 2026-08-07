"""nami elastic2d Born: linearity vs the full elastic solve, autograd
gradcheck, and a small forward/backward sanity check."""

import torch

from nami.elastic.elastic2d import elastic2d
from nami.elastic.elastic2d_born import elastic2d_born


def build_case(
    dtype=torch.float64,
    ny=24,
    nx=24,
    nt=12,
    pml=4,
    n_shots=1,
    device="cuda:0",
    seed=0,
):
    """Small mild-heterogeneity elastic model (two layers in rho); one
    source at the centre, 3 receivers nearby.

    dx = 1 mm, dt = 1e-7 s (CFL ~ 0.27 for vp <= ~2700 m/s).  Scatter
    perturbations are ~1e-3 relative for lamb/mu and ~1e-4 relative for
    buoyancy so the Born residual is within the O(delta^2) regime.
    """
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    rho = 2000 + 150 * torch.rand(ny, nx, generator=g)
    rho[ny // 2 :] += 100
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(seed + 2)
    vp = 2000 + 200 * torch.rand(ny, nx, generator=g2)
    vs = 1100 + 100 * torch.rand(ny, nx, generator=g2)
    lamb = rho * (vp ** 2 - 2 * vs ** 2)
    mu = rho * vs ** 2
    buoyancy = 1 / rho
    dx, dt = 1e-3, 1e-7
    srcs = torch.tensor(
        [
            [
                [ny // 2, nx // 2],
                [ny // 2, nx // 2 + 1],
                [ny // 2 + 1, nx // 2],
            ]
        ]
    )[:n_shots]
    recs = torch.tensor(
        [
            [
                [ny // 2, nx // 2],
                [ny // 2, nx // 2 + 1],
                [ny // 2 + 1, nx // 2],
            ]
        ]
    )[:n_shots]
    amp = torch.zeros(n_shots, srcs.shape[1], nt, dtype=dtype)
    amp[:, 0, :] = 1.0
    amp[:, 1, :] = -0.5
    amp[:, 2, :] = 0.2
    g3 = torch.Generator(device="cpu")
    g3.manual_seed(seed + 3)
    dlamb = lamb.abs().mean().item() * 2e-3 * (
        torch.rand(ny, nx, generator=g3) - 0.5
    )
    g4 = torch.Generator(device="cpu")
    g4.manual_seed(seed + 4)
    dmu = mu.abs().mean().item() * 2e-3 * (
        torch.rand(ny, nx, generator=g4) - 0.5
    )
    g5 = torch.Generator(device="cpu")
    g5.manual_seed(seed + 5)
    dbuoy = buoyancy.abs().mean().item() * 2e-4 * (
        torch.rand(ny, nx, generator=g5) - 0.5
    )
    return {
        "lamb": lamb,
        "mu": mu,
        "buoyancy": buoyancy,
        "dlamb": dlamb,
        "dmu": dmu,
        "dbuoy": dbuoy,
        "dx": dx,
        "dt": dt,
        "nt": nt,
        "pml": pml,
        "srcs": srcs,
        "recs": recs,
        "amp": amp,
        "n_shots": n_shots,
        "dtype": dtype,
        "device": device,
    }


def _run_full(c, lamb, mu, buoyancy, amp, accuracy=2):
    dev = c["device"]
    return elastic2d(
        lamb,
        mu,
        buoyancy,
        c["dx"],
        c["dt"],
        source_amplitudes=amp,
        source_locations=c["srcs"].to(dev),
        receiver_locations=c["recs"].to(dev),
        accuracy=accuracy,
        pml_width=c["pml"],
        nt=c["nt"],
    )


def _run_born(c, lamb, mu, buoyancy, dlamb, dmu, dbuoy, amp, accuracy=2,
              storage="auto", ckpt_steps=0, sample_steps=1):
    dev = c["device"]
    return elastic2d_born(
        lamb,
        mu,
        buoyancy,
        dlamb,
        dmu,
        dbuoy,
        c["dx"],
        c["dt"],
        source_amplitudes=amp,
        source_locations=c["srcs"].to(dev),
        receiver_locations=c["recs"].to(dev),
        accuracy=accuracy,
        pml_width=c["pml"],
        nt=c["nt"],
        storage=storage,
        ckpt_steps=ckpt_steps,
        sample_steps=sample_steps,
    )


def test_born_linearity():
    """r_born(delta) ~= r_full(delta) - r_bg with O(delta^2) error; the full
    solve uses the perturbed (lamb, mu, buoyancy) models."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=24, nx=24, nt=12, pml=4,
                   device="cuda:0", seed=1)
    dev = c["device"]
    lamb = c["lamb"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    buoyancy = c["buoyancy"].to(dev, dtype)
    dlamb = c["dlamb"].to(dev, dtype)
    dmu = c["dmu"].to(dev, dtype)
    dbuoy = c["dbuoy"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    def full(delta):
        return _run_full(c, lamb + delta * dlamb, mu + delta * dmu,
                         buoyancy + delta * dbuoy, amp)

    def born(delta):
        return _run_born(c, lamb, mu, buoyancy, delta * dlamb,
                         delta * dmu, delta * dbuoy, amp)

    r_bg = full(torch.zeros_like(dlamb))
    signal = born(torch.tensor(1.0, device=dev)).abs().max().item()
    resid_1 = (
        born(torch.tensor(1.0, device=dev))
        - (full(torch.tensor(1.0, device=dev)) - r_bg)
    ).abs().max().item()
    resid_2 = (
        born(torch.tensor(2.0, device=dev))
        - (full(torch.tensor(2.0, device=dev)) - r_bg)
    ).abs().max().item()
    print(f"signal(delta) {signal:.3e}, residual(delta) {resid_1:.3e}, "
          f"residual(2*delta) {resid_2:.3e}")
    assert resid_1 < 1e-2 * signal, f"Born linearity residual(delta): {resid_1}"
    # O(delta^2): doubling delta multiplies the residual by ~4
    ratio = resid_2 / (resid_1 + 1e-300)
    print(f"residual ratio 2x: {ratio:.3f}")
    assert 3.0 < ratio < 5.0, f"residual did not scale quadratically: {ratio}"


def test_gradcheck():
    """Numerical gradient check for bg + scatter models and amplitudes."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=24, nx=24, nt=10, pml=4,
                   device="cuda:0", seed=1)
    dev = c["device"]

    def fn(lamb, mu, buoyancy, dlamb, dmu, dbuoy, amp):
        return _run_born(c, lamb, mu, buoyancy, dlamb, dmu, dbuoy, amp)

    ok = torch.autograd.gradcheck(
        fn,
        (
            c["lamb"].to(dev, dtype).requires_grad_(True),
            c["mu"].to(dev, dtype).requires_grad_(True),
            c["buoyancy"].to(dev, dtype).requires_grad_(True),
            c["dlamb"].to(dev, dtype).requires_grad_(True),
            c["dmu"].to(dev, dtype).requires_grad_(True),
            c["dbuoy"].to(dev, dtype).requires_grad_(True),
            c["amp"].to(dev, dtype).requires_grad_(True),
        ),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        fast_mode=True,
        nondet_tol=1e-8,
        raise_exception=False,
    )
    print("gradcheck nami elastic2d_born (lamb, mu, buoy, scatter, amp):", ok)
    assert ok


def test_gradcheck_accuracy4():
    """Same gradcheck at accuracy 4 (coefficient-driven stencils)."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=20, nx=20, nt=8, pml=4,
                   device="cuda:0", seed=2)
    dev = c["device"]

    def fn(lamb, mu, buoyancy, dlamb, dmu, dbuoy, amp):
        return _run_born(c, lamb, mu, buoyancy, dlamb, dmu, dbuoy, amp,
                         accuracy=4)

    ok = torch.autograd.gradcheck(
        fn,
        (
            c["lamb"].to(dev, dtype).requires_grad_(True),
            c["mu"].to(dev, dtype).requires_grad_(True),
            c["buoyancy"].to(dev, dtype).requires_grad_(True),
            c["dlamb"].to(dev, dtype).requires_grad_(True),
            c["dmu"].to(dev, dtype).requires_grad_(True),
            c["dbuoy"].to(dev, dtype).requires_grad_(True),
            c["amp"].to(dev, dtype).requires_grad_(True),
        ),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        fast_mode=True,
        nondet_tol=1e-8,
        raise_exception=False,
    )
    print("gradcheck nami elastic2d_born accuracy 4:", ok)
    assert ok


def test_forward_sanity():
    """Tiny grid forward + backward: finite traces, finite grads."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=16, nx=16, nt=10, pml=4,
                   device="cuda:0", seed=3)
    dev = c["device"]
    lamb = c["lamb"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype).requires_grad_(True)
    buoyancy = c["buoyancy"].to(dev, dtype).requires_grad_(True)
    dlamb = c["dlamb"].to(dev, dtype).requires_grad_(True)
    dmu = c["dmu"].to(dev, dtype).requires_grad_(True)
    dbuoy = c["dbuoy"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype).requires_grad_(True)
    out = _run_born(c, lamb, mu, buoyancy, dlamb, dmu, dbuoy, amp)
    assert out.shape == (c["nt"], c["n_shots"], 3)
    assert torch.isfinite(out).all()
    assert out.abs().max().item() > 0.0
    out.sum().backward()
    for name, x in (("lamb", lamb), ("mu", mu), ("buoyancy", buoyancy),
                    ("dlamb", dlamb), ("dmu", dmu), ("dbuoy", dbuoy),
                    ("amp", amp)):
        g = x.grad
        assert g is not None and torch.isfinite(g).all(), f"{name} grad"
        assert g.abs().max().item() > 0.0, f"{name} grad zero"


def test_checkpoint_forward_and_gradient_parity():
    """checkpoint_every=N reproduces full-storage forward and gradients."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=24, nx=24, nt=30, pml=4,
                   device="cuda:0", seed=1)
    dev = c["device"]
    lamb = c["lamb"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype).requires_grad_(True)
    buoyancy = c["buoyancy"].to(dev, dtype).requires_grad_(True)
    dlamb = c["dlamb"].to(dev, dtype)
    dmu = c["dmu"].to(dev, dtype)
    dbuoy = c["dbuoy"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    def run(ckpt):
        return _run_born(c, lamb, mu, buoyancy, dlamb, dmu, dbuoy, amp,
                         ckpt_steps=ckpt)

    r_full = run(0).detach()
    g_full = torch.autograd.grad(run(0).square().sum(), (lamb, mu, buoyancy))[0]
    for ckpt in (2, 4, 7, 30, 31):
        r = run(ckpt).detach()
        g = torch.autograd.grad(run(ckpt).square().sum(), (lamb, mu, buoyancy))[0]
        assert (r - r_full).abs().max().item() == 0.0, f"ckpt_steps={ckpt} fwd"
        rel = (g - g_full).abs().max().item() / (
            g_full.abs().max().item() + 1e-300
        )
        # Forward is bit-identical; the gradient can differ at the ULP level
        # because torch's F.pad(mode="replicate") backward has a
        # data-dependent CUDA race at the padded edge (reproducible in pure
        # torch, unrelated to checkpointing). Observed max ~1e-39.
        assert rel < 1e-30, f"ckpt_steps={ckpt} grad rel err {rel}"


def test_checkpoint_parity_with_sampling_interval():
    """checkpoint_every=N is bit-identical even when snapshots are decimated
    (sample_steps=2)."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=24, nx=24, nt=30, pml=4,
                   device="cuda:0", seed=2)
    dev = c["device"]
    lamb = c["lamb"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype).requires_grad_(True)
    buoyancy = c["buoyancy"].to(dev, dtype).requires_grad_(True)
    dlamb = c["dlamb"].to(dev, dtype)
    dmu = c["dmu"].to(dev, dtype)
    dbuoy = c["dbuoy"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    def run(ckpt, stride):
        return _run_born(c, lamb, mu, buoyancy, dlamb, dmu, dbuoy, amp,
                         ckpt_steps=ckpt, sample_steps=stride)

    r_full = run(0, 2).detach()
    g_full = torch.autograd.grad(run(0, 2).square().sum(), (lamb, mu, buoyancy))[0]
    for ckpt in (2, 4, 7, 15):
        r = run(ckpt, 2).detach()
        g = torch.autograd.grad(run(ckpt, 2).square().sum(), (lamb, mu, buoyancy))[0]
        assert (r - r_full).abs().max().item() == 0.0, f"ckpt_steps={ckpt} fwd"
        rel = (g - g_full).abs().max().item() / (
            g_full.abs().max().item() + 1e-300
        )
        # See test_checkpoint_forward_and_gradient_parity: F.pad replicate
        # backward ULP nondeterminism, not a checkpointing issue.
        assert rel < 1e-30, f"ckpt_steps={ckpt} grad rel err {rel}"


def test_checkpoint_auto_selection():
    """storage='auto' enables snapshots when any input requires grad."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=24, nx=24, nt=25, pml=4,
                   device="cuda:0", seed=1)
    dev = c["device"]
    lamb = c["lamb"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype).requires_grad_(True)
    buoyancy = c["buoyancy"].to(dev, dtype).requires_grad_(True)
    dlamb = c["dlamb"].to(dev, dtype)
    dmu = c["dmu"].to(dev, dtype)
    dbuoy = c["dbuoy"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)
    out = _run_born(c, lamb, mu, buoyancy, dlamb, dmu, dbuoy, amp)
    g_auto = torch.autograd.grad(out.square().sum(), lamb)[0]
    g_full = torch.autograd.grad(
        _run_born(c, lamb, mu, buoyancy, dlamb, dmu, dbuoy, amp,
                  ckpt_steps=0).square().sum(),
        lamb,
    )[0]
    rel = (g_auto - g_full).abs().max().item() / (
        g_full.abs().max().item() + 1e-300
    )
    assert rel < 1e-30, f"auto grad rel err {rel}"


def test_storage_false_is_forward_only():
    """storage='none' runs the forward but backward raises RuntimeError."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=16, nx=16, nt=10, pml=4,
                   device="cuda:0", seed=3)
    dev = c["device"]
    lamb = c["lamb"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype)
    buoyancy = c["buoyancy"].to(dev, dtype)
    dlamb = c["dlamb"].to(dev, dtype)
    dmu = c["dmu"].to(dev, dtype)
    dbuoy = c["dbuoy"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)
    out = _run_born(c, lamb, mu, buoyancy, dlamb, dmu, dbuoy, amp,
                    storage="none")
    assert out.shape == (c["nt"], 1, 3)
    try:
        torch.autograd.grad(out.square().sum(), lamb)
    except RuntimeError:
        return
    raise AssertionError(
        "storage='none' backward should raise RuntimeError"
    )
