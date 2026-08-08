"""nami em2d_tm Born: linearity vs the full EM solve, autograd gradcheck,
and a small forward/backward sanity check."""

import torch

from nami.em.em2d_tm import em2d_tm
from nami.em.em2d_tm_born import em2d_tm_born


def build_case(
    dtype=torch.float64,
    ny=24,
    nx=24,
    nt=10,
    pml=4,
    n_shots=1,
    device="cuda:0",
    seed=0,
):
    """Small mild-heterogeneity TM model; sources at the centre + 2
    neighbours, 3 receivers nearby.

    dx = 1 mm, dt = 1 ps.  Relative permittivity 4-5.5, conductivity up to
    1e-3 S/m, mu = 1.  Scatter perturbations are ~1e-3 relative so the
    Born residual is well within the O(delta^2) regime.
    """
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    eps = 4.0 + 1.5 * torch.rand(ny, nx, generator=g)
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(seed + 2)
    sigma = 1e-3 * torch.rand(ny, nx, generator=g2)
    mu = torch.ones(ny, nx)
    dx, dt = 1e-3, 1e-12
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
    deps = 1e-3 * eps * torch.rand(ny, nx, generator=g3)
    g4 = torch.Generator(device="cpu")
    g4.manual_seed(seed + 4)
    dsig = 1e-6 * torch.rand(ny, nx, generator=g4)
    g5 = torch.Generator(device="cpu")
    g5.manual_seed(seed + 5)
    dmu = 1e-3 * torch.rand(ny, nx, generator=g5)
    return {
        "eps": eps,
        "sigma": sigma,
        "mu": mu,
        "deps": deps,
        "dsig": dsig,
        "dmu": dmu,
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


def _run_full(c, eps, sigma, mu, amp, accuracy=2):
    dev = c["device"]
    return em2d_tm(
        eps,
        sigma,
        mu,
        c["dx"],
        c["dt"],
        source_amplitudes=amp,
        source_locations=c["srcs"].to(dev),
        receiver_locations=c["recs"].to(dev),
        accuracy=accuracy,
        pml_width=c["pml"],
        nt=c["nt"],
    )


def _run_born(c, eps, sigma, mu, deps, dsig, dmu, amp, accuracy=2):
    dev = c["device"]
    return em2d_tm_born(
        eps,
        sigma,
        mu,
        deps,
        dsig,
        dmu,
        c["dx"],
        c["dt"],
        source_amplitudes=amp,
        source_locations=c["srcs"].to(dev),
        receiver_locations=c["recs"].to(dev),
        accuracy=accuracy,
        pml_width=c["pml"],
        nt=c["nt"],
    )


def test_born_linearity():
    """r_born(delta) ~= r_full(delta) - r_bg with O(delta^2) error; the full
    solve recompiles ca/cb/cq from the perturbed epsilon/sigma/mu."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=24, nx=24, nt=10, pml=4,
                   device="cuda:0", seed=1)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype)
    sigma = c["sigma"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    deps = c["deps"].to(dev, dtype)
    dsig = c["dsig"].to(dev, dtype)
    dmu = c["dmu"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    def full(delta):
        return _run_full(c, eps + delta * deps, sigma + delta * dsig,
                         mu + delta * dmu, amp)

    def born(delta):
        return _run_born(c, eps, sigma, mu, delta * deps, delta * dsig,
                         delta * dmu, amp)

    r_bg = full(torch.zeros_like(deps))
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

    def fn(eps, sigma, mu, deps, dsig, dmu, amp):
        return _run_born(c, eps, sigma, mu, deps, dsig, dmu, amp)

    ok = torch.autograd.gradcheck(
        fn,
        (
            c["eps"].to(dev, dtype).requires_grad_(True),
            c["sigma"].to(dev, dtype).requires_grad_(True),
            c["mu"].to(dev, dtype).requires_grad_(True),
            c["deps"].to(dev, dtype).requires_grad_(True),
            c["dsig"].to(dev, dtype).requires_grad_(True),
            c["dmu"].to(dev, dtype).requires_grad_(True),
            c["amp"].to(dev, dtype).requires_grad_(True),
        ),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        fast_mode=True,
        nondet_tol=1e-8,
        raise_exception=False,
    )
    print("gradcheck nami em2d_tm_born (eps, sigma, mu, scatter, amp):", ok)
    assert ok


def test_gradcheck_accuracy4():
    """Same gradcheck at accuracy 4 (coefficient-driven stencils)."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=20, nx=20, nt=8, pml=4,
                   device="cuda:0", seed=2)
    dev = c["device"]

    def fn(eps, sigma, mu, deps, dsig, dmu, amp):
        return _run_born(c, eps, sigma, mu, deps, dsig, dmu, amp, accuracy=4)

    ok = torch.autograd.gradcheck(
        fn,
        (
            c["eps"].to(dev, dtype).requires_grad_(True),
            c["sigma"].to(dev, dtype).requires_grad_(True),
            c["mu"].to(dev, dtype).requires_grad_(True),
            c["deps"].to(dev, dtype).requires_grad_(True),
            c["dsig"].to(dev, dtype).requires_grad_(True),
            c["dmu"].to(dev, dtype).requires_grad_(True),
            c["amp"].to(dev, dtype).requires_grad_(True),
        ),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        fast_mode=True,
        nondet_tol=1e-8,
        raise_exception=False,
    )
    print("gradcheck nami em2d_tm_born accuracy 4:", ok)
    assert ok


def test_forward_sanity():
    """Tiny grid forward + backward: finite traces, finite grads."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=16, nx=16, nt=10, pml=4,
                   device="cuda:0", seed=3)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sigma = c["sigma"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype).requires_grad_(True)
    deps = c["deps"].to(dev, dtype).requires_grad_(True)
    dsig = c["dsig"].to(dev, dtype).requires_grad_(True)
    dmu = c["dmu"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype).requires_grad_(True)
    out = _run_born(c, eps, sigma, mu, deps, dsig, dmu, amp)
    assert out.shape == (c["nt"], c["n_shots"], 3)
    assert torch.isfinite(out).all()
    assert out.abs().max().item() > 0.0
    out.sum().backward()
    for name, x in (("eps", eps), ("sigma", sigma), ("mu", mu),
                    ("deps", deps), ("dsig", dsig), ("dmu", dmu),
                    ("amp", amp)):
        g = x.grad
        assert g is not None and torch.isfinite(g).all(), f"{name} grad"
        assert g.abs().max().item() > 0.0, f"{name} grad zero"


def _run_born_storage(c, eps, sigma, mu, deps, dsig, dmu, amp, storage,
                      ckpt_steps=None, accuracy=2):
    dev = c["device"]
    return em2d_tm_born(
        eps,
        sigma,
        mu,
        deps,
        dsig,
        dmu,
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
    )


def test_checkpoint_forward_and_gradient_parity():
    """ckpt_steps=N reproduces full-storage forward and gradients
    (bit-identical forward, max-relative gradient error 0.0)."""
    dtype = torch.float64
    c = build_case(dtype=dtype, device="cuda:0", seed=1)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sigma = c["sigma"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    deps = c["deps"].to(dev, dtype).requires_grad_(True)
    dsig = c["dsig"].to(dev, dtype)
    dmu = c["dmu"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    def run(ckpt):
        return _run_born_storage(c, eps, sigma, mu, deps, dsig, dmu, amp,
                                 "auto", ckpt)

    r_full = run(0).detach()
    g_full = torch.autograd.grad(run(0).square().sum(), (eps, deps))
    for ckpt in (2, 4, 7, c["nt"]):
        r = run(ckpt).detach()
        g = torch.autograd.grad(run(ckpt).square().sum(), (eps, deps))
        assert (r - r_full).abs().max().item() == 0.0, f"ckpt_steps={ckpt} fwd"
        for ga, gb in zip(g, g_full, strict=True):
            rel = (ga - gb).abs().max().item() / (
                gb.abs().max().item() + 1e-300
            )
            assert rel == 0.0, f"ckpt_steps={ckpt} grad rel err {rel}"


def test_checkpoint_auto_selection():
    """default storage auto-enables snapshots; ckpt_steps=None picks
    ~sqrt(nt) and reproduces explicit full storage."""
    dtype = torch.float64
    c = build_case(dtype=dtype, device="cuda:0", seed=1)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sigma = c["sigma"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    deps = c["deps"].to(dev, dtype).requires_grad_(True)
    dsig = c["dsig"].to(dev, dtype)
    dmu = c["dmu"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    out = _run_born(c, eps, sigma, mu, deps, dsig, dmu, amp)
    g_auto = torch.autograd.grad(out.square().sum(), (eps, deps))
    g_full = torch.autograd.grad(
        _run_born_storage(c, eps, sigma, mu, deps, dsig, dmu, amp,
                          "auto", 0).square().sum(),
        (eps, deps),
    )
    for ga, gb in zip(g_auto, g_full, strict=True):
        rel = (ga - gb).abs().max().item() / (gb.abs().max().item() + 1e-300)
        assert rel == 0.0, f"auto grad rel err {rel}"


def test_storage_false_is_forward_only():
    """storage='none' runs the forward but backward raises RuntimeError."""
    dtype = torch.float64
    c = build_case(dtype=dtype, device="cuda:0", seed=1)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sigma = c["sigma"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    deps = c["deps"].to(dev, dtype)
    dsig = c["dsig"].to(dev, dtype)
    dmu = c["dmu"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)
    out = _run_born_storage(c, eps, sigma, mu, deps, dsig, dmu, amp, "none")
    assert out.shape == (c["nt"], c["n_shots"], 3)
    try:
        torch.autograd.grad(out.square().sum(), eps)
    except RuntimeError:
        return
    raise AssertionError("storage='none' backward should raise RuntimeError")


def _multi_shot_survey(ny, nx, nt, dtype):
    srcs = torch.tensor(
        [
            [[ny // 2, nx // 2], [ny // 2, nx // 2 + 1]],
            [[ny // 2 + 1, nx // 2 + 1], [ny // 2 + 1, nx // 2]],
        ]
    )
    recs = torch.tensor(
        [
            [
                [ny // 2, nx // 2],
                [ny // 2, nx // 2 + 1],
                [ny // 2 + 1, nx // 2],
            ],
            [
                [ny // 2 + 1, nx // 2 + 1],
                [ny // 2 + 1, nx // 2],
                [ny // 2, nx // 2 + 1],
            ],
        ]
    )
    amp = torch.zeros(2, 2, nt, dtype=dtype)
    amp[0, 0, :3] = torch.tensor([1.0, -0.5, 0.2], dtype=dtype)
    amp[0, 1, :3] = torch.tensor([0.5, 0.1, -0.2], dtype=dtype)
    amp[1, 0, :3] = torch.tensor([0.7, 0.3, -0.1], dtype=dtype)
    amp[1, 1, :3] = torch.tensor([-0.4, 0.2, 0.1], dtype=dtype)
    return srcs, recs, amp


def test_multi_shot_shared_model():
    """n_shots=2 shared models: Born fwd matches singles; grads are sums."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=20, nx=20, nt=10, pml=3, device="cuda:0", seed=1)
    dev = c["device"]
    ny, nx = c["eps"].shape
    nt = c["nt"]
    srcs, recs, amp = _multi_shot_survey(ny, nx, nt, dtype)
    keys = ("eps", "sigma", "mu", "deps", "dsig", "dmu")

    def run(models, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return em2d_tm_born(
            *models, c["dx"], c["dt"],
            source_amplitudes=amp[sl].to(dev),
            source_locations=srcs[sl].to(dev),
            receiver_locations=recs[sl].to(dev),
            accuracy=2, pml_width=c["pml"], nt=nt,
        )

    params = {k: c[k].to(dev, dtype) for k in keys}
    p = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    p0 = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    p1 = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    r = run([p[k] for k in keys])
    r0 = run([p0[k] for k in keys], 0)
    r1 = run([p1[k] for k in keys], 1)
    assert r.shape == (nt, 2, 3)
    assert (r[:, 0].detach() - r0[:, 0].detach()).abs().max().item() == 0.0
    assert (r[:, 1].detach() - r1[:, 0].detach()).abs().max().item() == 0.0
    gs = torch.autograd.grad(r.square().sum(), [p[k] for k in keys])
    g0s = torch.autograd.grad(r0.square().sum(), [p0[k] for k in keys])
    g1s = torch.autograd.grad(r1.square().sum(), [p1[k] for k in keys])
    for g, g0, g1, name in zip(gs, g0s, g1s, keys, strict=True):
        rel = (g - (g0 + g1)).abs().max().item() / (
            (g0 + g1).abs().max().item() + 1e-300
        )
        # float64: measured worst rel err ~5.2e-16 over 7 runs.
        assert rel < 1e-12, f"shared-model grad_{name} rel err {rel}"


def test_multi_shot_batched_model():
    """n_shots=2 batched models: shot i uses slice i."""
    dtype = torch.float64
    c0 = build_case(dtype=dtype, ny=20, nx=20, nt=10, pml=3, device="cuda:0", seed=1)
    c1 = build_case(dtype=dtype, ny=20, nx=20, nt=10, pml=3, device="cuda:0", seed=2)
    dev = c0["device"]
    ny, nx = c0["eps"].shape
    nt = c0["nt"]
    srcs, recs, amp = _multi_shot_survey(ny, nx, nt, dtype)
    keys = ("eps", "sigma", "mu", "deps", "dsig", "dmu")

    def run(models, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return em2d_tm_born(
            *models, c0["dx"], c0["dt"],
            source_amplitudes=amp[sl].to(dev),
            source_locations=srcs[sl].to(dev),
            receiver_locations=recs[sl].to(dev),
            accuracy=2, pml_width=c0["pml"], nt=nt,
        )

    batch = [
        torch.stack([c0[k], c1[k]]).to(dev, dtype).requires_grad_(True)
        for k in keys
    ]
    m0 = [c0[k].to(dev, dtype).requires_grad_(True) for k in keys]
    m1 = [c1[k].to(dev, dtype).requires_grad_(True) for k in keys]
    r = run(batch)
    r0, r1 = run(m0, 0), run(m1, 1)
    assert r.shape == (nt, 2, 3)
    assert (r[:, 0].detach() - r0[:, 0].detach()).abs().max().item() == 0.0
    assert (r[:, 1].detach() - r1[:, 0].detach()).abs().max().item() == 0.0
    gs = torch.autograd.grad(r.square().sum(), batch)
    g0s = torch.autograd.grad(r0.square().sum(), m0)
    g1s = torch.autograd.grad(r1.square().sum(), m1)
    for gb, g0, g1, name in zip(gs, g0s, g1s, keys, strict=True):
        rel0 = (gb[0] - g0).abs().max().item() / (g0.abs().max().item() + 1e-300)
        rel1 = (gb[1] - g1).abs().max().item() / (g1.abs().max().item() + 1e-300)
        # float64: measured worst rel err 0 (bit-identical) over 7 runs.
        assert rel0 < 1e-12 and rel1 < 1e-12, (
            f"batched-model grad_{name} rel err ({rel0}, {rel1})"
        )
