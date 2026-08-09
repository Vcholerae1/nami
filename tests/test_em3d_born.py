"""nami em3d_born: Born linearity vs the full EM solve, autograd gradcheck,
finite-difference gradient consistency, and storage/forward sanity.

Self-contained tests (same approach as test_em2d_tm_born.py / test_em3d.py):
correctness is verified via numerical gradients
(``torch.autograd.gradcheck``), the first-order Born linearity relation (the
scattered solve matches the perturbed full solve to O(delta^2)), and central
finite differences of the analytic gradients.
"""

import torch

from nami.em.em3d import em3d
from nami.em.em3d_born import em3d_born


def build_case(
    dtype=torch.float64,
    nz=8,
    ny=8,
    nx=8,
    nt=6,
    pml=2,
    n_shots=1,
    device="cuda:0",
    seed=0,
):
    """Small random 3D EM Born model; sources at the centre + 2 neighbours,
    3 nearby scattered receivers, 2 background receivers.

    dx = 1e-3 m, dt = 1e-12 s: the wave travels ~0.14 cells/step
    (c_max = 1.0 CFL check passes).  eps ~ 4-5.5, sigma ~ 0-1e-3 S/m,
    mu = 1.  Scatter perturbations are ~1e-3 relative so the Born residual
    is well within the O(delta^2) regime.
    """
    torch.manual_seed(seed)
    dx, dt = 1e-3, 1.0e-12
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    eps = 4.0 + 1.5 * torch.rand(nz, ny, nx, generator=g)
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(seed + 2)
    sig = 0.001 * torch.rand(nz, ny, nx, generator=g2)
    mu = torch.ones(nz, ny, nx)
    srcs = torch.tensor(
        [
            [
                [nz // 2, ny // 2, nx // 2],
                [nz // 2, ny // 2, nx // 2 + 1],
                [nz // 2 + 1, ny // 2, nx // 2],
            ]
        ]
    )[:n_shots]
    recs = torch.tensor(
        [
            [
                [nz // 2, ny // 2, nx // 2],
                [nz // 2, ny // 2, nx // 2 + 1],
                [nz // 2 + 1, ny // 2, nx // 2],
            ]
        ]
    )[:n_shots]
    bg_recs = torch.tensor(
        [[[nz // 2, ny // 2, nx // 2], [nz // 2, ny // 2 + 1, nx // 2]]]
    )[:n_shots]
    amp = torch.zeros(n_shots, srcs.shape[1], nt, dtype=dtype)
    amp[:, :, 0] = 1.0
    amp[:, :, 1] = -0.5
    amp[:, :, 2] = 0.2
    g3 = torch.Generator(device="cpu")
    g3.manual_seed(seed + 3)
    deps = 1e-3 * eps * torch.rand(nz, ny, nx, generator=g3)
    g4 = torch.Generator(device="cpu")
    g4.manual_seed(seed + 4)
    dsig = 1e-6 * torch.rand(nz, ny, nx, generator=g4)
    g5 = torch.Generator(device="cpu")
    g5.manual_seed(seed + 5)
    dmu = 1e-3 * torch.rand(nz, ny, nx, generator=g5)
    return {
        "eps": eps,
        "sig": sig,
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
        "bg_recs": bg_recs,
        "amp": amp,
        "n_shots": n_shots,
        "dtype": dtype,
        "device": device,
    }


def _run_full(c, eps, sigma, mu, amp, accuracy=2, recs=None):
    """Full (non-Born) em3d solve at the given models."""
    dev = c["device"]
    return em3d(
        eps,
        sigma,
        mu,
        c["dx"],
        c["dt"],
        source_amplitudes=amp,
        source_locations=c["srcs"].to(dev),
        receiver_locations=(c["recs"] if recs is None else recs).to(dev),
        accuracy=accuracy,
        pml_width=c["pml"],
        nt=c["nt"],
    )


def _run_born(c, eps, sigma, mu, deps, dsig, dmu, amp, accuracy=2, **kwargs):
    """Runs nami's em3d_born."""
    dev = c["device"]
    return em3d_born(
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
        bg_receiver_locations=c["bg_recs"].to(dev),
        accuracy=accuracy,
        pml_width=c["pml"],
        nt=c["nt"],
        **kwargs,
    )


def test_born_linearity():
    """r_born(delta) ~= r_full(delta) - r_bg with O(delta^2) error; the full
    solve recompiles ca/cb/cq from the perturbed epsilon/sigma/mu.  Also
    checks the Born background traces match the em3d background solve at
    the bg receiver locations exactly."""
    dtype = torch.float64
    c = build_case(dtype=dtype, nz=8, ny=8, nx=8, nt=6, pml=2,
                   device="cuda:0", seed=1)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype)
    sigma = c["sig"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    deps = c["deps"].to(dev, dtype)
    dsig = c["dsig"].to(dev, dtype)
    dmu = c["dmu"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    def full(delta):
        return _run_full(
            c, eps + delta * deps, sigma + delta * dsig, mu + delta * dmu, amp
        )

    def born(delta):
        r, r_bg = _run_born(
            c, eps, sigma, mu, delta * deps, delta * dsig, delta * dmu, amp
        )
        return r, r_bg

    r_bg_full = full(torch.zeros_like(deps))
    # background propagation parity: Born bg receivers == em3d bg solve
    full_bg = _run_full(
        c, eps, sigma, mu, amp, recs=c["bg_recs"]
    )
    r_bg_1 = born(torch.tensor(1.0, device=dev))[1]
    assert r_bg_1.shape == (c["nt"], c["n_shots"], 2)
    assert torch.allclose(r_bg_1, full_bg, rtol=1e-9, atol=0.0), (
        "Born background traces do not match the em3d background solve"
    )

    signal = born(torch.tensor(1.0, device=dev))[0].abs().max().item()
    resid_1 = (
        born(torch.tensor(1.0, device=dev))[0]
        - (full(torch.tensor(1.0, device=dev)) - r_bg_full)
    ).abs().max().item()
    resid_2 = (
        born(torch.tensor(2.0, device=dev))[0]
        - (full(torch.tensor(2.0, device=dev)) - r_bg_full)
    ).abs().max().item()
    print(f"signal(delta) {signal:.3e}, residual(delta) {resid_1:.3e}, "
          f"residual(2*delta) {resid_2:.3e}")
    assert resid_1 < 1e-2 * signal, f"Born linearity residual(delta): {resid_1}"
    # O(delta^2): doubling delta multiplies the residual by ~4
    ratio = resid_2 / (resid_1 + 1e-300)
    print(f"residual ratio 2x: {ratio:.3f}")
    assert 3.0 < ratio < 5.0, f"residual did not scale quadratically: {ratio}"


def test_background_receiver_matches_full_solve_and_gradient():
    c = build_case(
        dtype=torch.float64, nz=8, ny=8, nx=8, nt=6, pml=2,
        device="cuda:0", seed=7,
    )
    dev = c["device"]
    eps = c["eps"].to(dev).requires_grad_(True)
    sigma = c["sig"].to(dev).requires_grad_(True)
    mu = c["mu"].to(dev).requires_grad_(True)
    amp = c["amp"].to(dev).requires_grad_(True)
    scatter = [c[name].to(dev) for name in ("deps", "dsig", "dmu")]
    _, r_bg = _run_born(c, eps, sigma, mu, *scatter, amp)

    full_models = [
        model.detach().clone().requires_grad_(True)
        for model in (eps, sigma, mu)
    ]
    full_amp = amp.detach().clone().requires_grad_(True)
    r_full = _run_full(c, *full_models, full_amp, recs=c["bg_recs"])
    torch.testing.assert_close(r_bg, r_full, rtol=0, atol=0)
    grads = torch.autograd.grad(r_bg.square().sum(), (eps, sigma, mu, amp))
    refs = torch.autograd.grad(r_full.square().sum(), (*full_models, full_amp))
    for grad, ref in zip(grads, refs, strict=True):
        torch.testing.assert_close(grad, ref, rtol=1e-6, atol=1e-12)


def test_gradcheck():
    """Numerical gradient check for bg + scatter models and amplitudes."""
    dtype = torch.float64
    c = build_case(dtype=dtype, nz=8, ny=8, nx=8, nt=6, pml=2,
                   device="cuda:0", seed=1)
    dev = c["device"]

    def fn(eps, sigma, mu, deps, dsig, dmu, amp):
        return _run_born(c, eps, sigma, mu, deps, dsig, dmu, amp)

    ok = torch.autograd.gradcheck(
        fn,
        (
            c["eps"].to(dev, dtype).requires_grad_(True),
            c["sig"].to(dev, dtype).requires_grad_(True),
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
    print("gradcheck nami em3d_born (eps, sigma, mu, scatter, amp):", ok)
    assert ok


def test_gradcheck_higher_order():
    """Same gradcheck at accuracy 4, 6, and 8."""
    dtype = torch.float64
    c = build_case(dtype=dtype, nz=8, ny=8, nx=8, nt=6, pml=2,
                   device="cuda:0", seed=2)
    dev = c["device"]

    for accuracy in (4, 6, 8):
        def fn(eps, sigma, mu, deps, dsig, dmu, amp, accuracy=accuracy):
            return _run_born(
                c, eps, sigma, mu, deps, dsig, dmu, amp, accuracy=accuracy,
            )

        args = tuple(
            c[name].to(dev, dtype).requires_grad_(True)
            for name in ("eps", "sig", "mu", "deps", "dsig", "dmu", "amp")
        )
        ok = torch.autograd.gradcheck(
            fn, args, eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True,
            nondet_tol=1e-8, raise_exception=False,
        )
        assert ok, f"em3d_born gradcheck failed at accuracy={accuracy}"


def test_gradient_consistency_finite_difference():
    """Analytic gradients (torch.autograd.grad) vs central finite
    differences for all six models on a tiny grid.

    Compares the strongest-response cells (largest analytic gradient) so
    the numeric reference stays cheap: central FD of a large-magnitude
    scalar loss is only resolvable in float64 where the induced loss change
    (2*h*|g_i|) is well above the loss' ULP.
    """
    dtype = torch.float64
    c = build_case(dtype=dtype, nz=6, ny=8, nx=8, nt=6, pml=2,
                   device="cuda:0", seed=3)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sig = c["sig"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype).requires_grad_(True)
    deps = c["deps"].to(dev, dtype).requires_grad_(True)
    dsig = c["dsig"].to(dev, dtype).requires_grad_(True)
    dmu = c["dmu"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype)

    out_r, out_rb = _run_born(c, eps, sig, mu, deps, dsig, dmu, amp)
    loss = (out_r**2).sum() + (out_rb**2).sum()
    g_eps, g_sig, g_mu, g_deps, g_dsig, g_dmu = torch.autograd.grad(
        loss, (eps, sig, mu, deps, dsig, dmu)
    )

    base = {
        "eps": eps.detach(),
        "sig": sig.detach(),
        "mu": mu.detach(),
        "deps": deps.detach(),
        "dsig": dsig.detach(),
        "dmu": dmu.detach(),
    }
    for name, model, grad, h in (
        ("eps", eps, g_eps, 1e-4),
        ("sig", sig, g_sig, 1e-7),
        ("mu", mu, g_mu, 1e-4),
        ("deps", deps, g_deps, 1e-4),
        ("dsig", dsig, g_dsig, 1e-7),
        ("dmu", dmu, g_dmu, 1e-4),
    ):
        flat_grad = grad.reshape(-1)
        idx = torch.topk(flat_grad.abs(), 8).indices
        analytic = flat_grad[idx]
        numeric = torch.zeros_like(analytic)
        for j, flat in enumerate(idx.tolist()):
            loss_p = 0.0
            loss_m = 0.0
            for sign in (+1.0, -1.0):
                models = dict(base)
                p = model.detach().reshape(-1).clone()
                p[flat] += sign * h
                models[name] = p.reshape(model.shape)
                r, r_bg = _run_born(
                    c, models["eps"], models["sig"], models["mu"],
                    models["deps"], models["dsig"], models["dmu"], amp,
                )
                loss_val = (r**2).sum() + (r_bg**2).sum()
                if sign > 0:
                    loss_p = loss_val
                else:
                    loss_m = loss_val
            numeric[j] = (loss_p - loss_m) / (2 * h)
        denom = analytic.abs().max().item() + 1e-300
        rel = (analytic - numeric).abs().max().item() / denom
        print(f"grad-consistency {name}: max rel err {rel:.3e}")
        assert rel < 5e-4, f"{name} gradient mismatch: {rel:.3e}"


def test_forward_sanity():
    """Tiny grid forward + backward: finite traces, finite grads."""
    dtype = torch.float64
    c = build_case(dtype=dtype, nz=8, ny=8, nx=8, nt=6, pml=2,
                   device="cuda:0", seed=3)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sig = c["sig"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype).requires_grad_(True)
    deps = c["deps"].to(dev, dtype).requires_grad_(True)
    dsig = c["dsig"].to(dev, dtype).requires_grad_(True)
    dmu = c["dmu"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype).requires_grad_(True)
    r, r_bg = _run_born(c, eps, sig, mu, deps, dsig, dmu, amp)
    assert r.shape == (c["nt"], c["n_shots"], 3)
    assert r_bg.shape == (c["nt"], c["n_shots"], 2)
    assert torch.isfinite(r).all() and torch.isfinite(r_bg).all()
    assert r.abs().max().item() > 0.0 and r_bg.abs().max().item() > 0.0
    (r**2).sum().backward()
    for name, x in (("eps", eps), ("sig", sig), ("mu", mu),
                    ("deps", deps), ("dsig", dsig), ("dmu", dmu),
                    ("amp", amp)):
        g = x.grad
        assert g is not None and torch.isfinite(g).all(), f"{name} grad"
        assert g.abs().max().item() > 0.0, f"{name} grad zero"


def test_checkpoint_forward_and_gradient_parity():
    """ckpt_steps=N reproduces full-storage forward and gradients."""
    dtype = torch.float64
    c = build_case(dtype=dtype, nz=8, ny=8, nx=8, nt=10, pml=2,
                   device="cuda:0", seed=4)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sig = c["sig"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype).requires_grad_(True)
    deps = c["deps"].to(dev, dtype).requires_grad_(True)
    dsig = c["dsig"].to(dev, dtype).requires_grad_(True)
    dmu = c["dmu"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype).requires_grad_(True)

    def run(ckpt):
        return _run_born(
            c, eps, sig, mu, deps, dsig, dmu, amp,
            storage="auto", ckpt_steps=ckpt,
        )

    r_full, r_bg_full = run(0)
    r_full = r_full.detach()
    r_bg_full = r_bg_full.detach()
    loss = lambda out: out[0].square().sum() + out[1].square().sum()  # noqa: E731
    g_full = torch.autograd.grad(
        loss(run(0)), (eps, sig, mu, deps, dsig, dmu, amp)
    )
    for ckpt in (2, 4, 7, 10, 15):
        r, r_bg = run(ckpt)
        assert (r - r_full).abs().max().item() == 0.0, f"ckpt_steps={ckpt} fwd"
        assert (r_bg - r_bg_full).abs().max().item() == 0.0, f"ckpt_steps={ckpt} bg fwd"
        g = torch.autograd.grad(
            loss(run(ckpt)), (eps, sig, mu, deps, dsig, dmu, amp)
        )
        for name, ga, gb in zip(
            ("eps", "sig", "mu", "deps", "dsig", "dmu", "amp"),
            g, g_full,
            strict=True,
        ):
            rel = (ga - gb).abs().max().item() / (
                gb.abs().max().item() + 1e-300
            )
            # em3d CUDA reductions can differ by ~1 ULP of the largest term
            assert rel < 1e-12, f"ckpt_steps={ckpt} {name} grad rel err {rel}"


def test_storage_false_is_forward_only():
    """storage='none' runs the forward but backward raises RuntimeError."""
    dtype = torch.float64
    c = build_case(dtype=dtype, nz=8, ny=8, nx=8, nt=6, pml=2,
                   device="cuda:0", seed=4)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sig = c["sig"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype).requires_grad_(True)
    deps = c["deps"].to(dev, dtype).requires_grad_(True)
    dsig = c["dsig"].to(dev, dtype).requires_grad_(True)
    dmu = c["dmu"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype).requires_grad_(True)
    r, r_bg = _run_born(c, eps, sig, mu, deps, dsig, dmu, amp,
                        storage="none")
    assert r.shape == (c["nt"], c["n_shots"], 3)
    assert r_bg.shape == (c["nt"], c["n_shots"], 2)
    assert torch.isfinite(r).all() and torch.isfinite(r_bg).all()
    try:
        torch.autograd.grad(r.square().sum(), eps)
    except RuntimeError:
        return
    raise AssertionError("storage='none' backward should raise RuntimeError")


def _multi_shot_survey(nz, ny, nx, nt, dtype):
    srcs = torch.tensor(
        [
            [
                [nz // 2, ny // 2, nx // 2],
                [nz // 2, ny // 2, nx // 2 + 1],
            ],
            [
                [nz // 2, ny // 2 + 1, nx // 2 + 1],
                [nz // 2 + 1, ny // 2, nx // 2],
            ],
        ]
    )
    recs = torch.tensor(
        [
            [
                [nz // 2, ny // 2, nx // 2],
                [nz // 2, ny // 2, nx // 2 + 1],
                [nz // 2 + 1, ny // 2, nx // 2],
            ],
            [
                [nz // 2, ny // 2 + 1, nx // 2 + 1],
                [nz // 2 + 1, ny // 2, nx // 2],
                [nz // 2, ny // 2 + 1, nx // 2],
            ],
        ]
    )
    bg_recs = torch.tensor(
        [
            [[nz // 2, ny // 2, nx // 2], [nz // 2, ny // 2 + 1, nx // 2]],
            [[nz // 2, ny // 2 + 1, nx // 2 + 1], [nz // 2 + 1, ny // 2, nx // 2]],
        ]
    )
    amp = torch.zeros(2, 2, nt, dtype=dtype)
    amp[0, 0, :3] = torch.tensor([1.0, -0.5, 0.2], dtype=dtype)
    amp[0, 1, :3] = torch.tensor([0.5, 0.1, -0.2], dtype=dtype)
    amp[1, 0, :3] = torch.tensor([0.7, 0.3, -0.1], dtype=dtype)
    amp[1, 1, :3] = torch.tensor([-0.4, 0.2, 0.1], dtype=dtype)
    return srcs, recs, bg_recs, amp


def test_multi_shot_shared_model():
    """n_shots=2 shared models: Born fwd matches singles; grads are sums."""
    dtype = torch.float64
    c = build_case(dtype=dtype, nz=8, ny=8, nx=8, nt=6, pml=2,
                   device="cuda:0", seed=1)
    dev = c["device"]
    nz, ny, nx = c["eps"].shape
    nt = c["nt"]
    srcs, recs, bg_recs, amp = _multi_shot_survey(nz, ny, nx, nt, dtype)
    keys = ("eps", "sig", "mu", "deps", "dsig", "dmu")

    def run(models, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return em3d_born(
            *models, c["dx"], c["dt"],
            source_amplitudes=amp[sl].to(dev),
            source_locations=srcs[sl].to(dev),
            receiver_locations=recs[sl].to(dev),
            bg_receiver_locations=bg_recs[sl].to(dev),
            accuracy=2, pml_width=c["pml"], nt=nt,
        )

    params = {k: c[k].to(dev, dtype) for k in keys}
    p = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    p0 = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    p1 = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    r, r_bg = run([p[k] for k in keys])
    r0, r0_bg = run([p0[k] for k in keys], 0)
    r1, r1_bg = run([p1[k] for k in keys], 1)
    assert r.shape == (nt, 2, 3) and r_bg.shape == (nt, 2, 2)
    for out, ref, name in (
        (r[:, 0], r0[:, 0], "shot 0 fwd r"),
        (r[:, 1], r1[:, 0], "shot 1 fwd r"),
        (r_bg[:, 0], r0_bg[:, 0], "shot 0 fwd r_bg"),
        (r_bg[:, 1], r1_bg[:, 0], "shot 1 fwd r_bg"),
    ):
        assert (out.detach() - ref.detach()).abs().max().item() == 0.0, name
    loss = r.square().sum() + r_bg.square().sum()
    gs = torch.autograd.grad(loss, [p[k] for k in keys])
    g0s = torch.autograd.grad(
        r0.square().sum() + r0_bg.square().sum(), [p0[k] for k in keys]
    )
    g1s = torch.autograd.grad(
        r1.square().sum() + r1_bg.square().sum(), [p1[k] for k in keys]
    )
    for g, g0, g1, name in zip(gs, g0s, g1s, keys, strict=True):
        rel = (g - (g0 + g1)).abs().max().item() / (
            (g0 + g1).abs().max().item() + 1e-300
        )
        # Deterministic ~1-ulp residual: the shared-model gradient sums the
        # per-shot gradients before the padding backward, the reference sums
        # the two single-shot gradients after it (measured <= 5e-16).
        assert rel < 1e-12, f"shared-model grad_{name} rel err {rel}"


def test_multi_shot_batched_model():
    """n_shots=2 batched models: shot i uses slice i."""
    dtype = torch.float64
    c0 = build_case(dtype=dtype, nz=8, ny=8, nx=8, nt=6, pml=2,
                    device="cuda:0", seed=1)
    c1 = build_case(dtype=dtype, nz=8, ny=8, nx=8, nt=6, pml=2,
                    device="cuda:0", seed=2)
    dev = c0["device"]
    nz, ny, nx = c0["eps"].shape
    nt = c0["nt"]
    srcs, recs, bg_recs, amp = _multi_shot_survey(nz, ny, nx, nt, dtype)
    keys = ("eps", "sig", "mu", "deps", "dsig", "dmu")

    def run(models, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return em3d_born(
            *models, c0["dx"], c0["dt"],
            source_amplitudes=amp[sl].to(dev),
            source_locations=srcs[sl].to(dev),
            receiver_locations=recs[sl].to(dev),
            bg_receiver_locations=bg_recs[sl].to(dev),
            accuracy=2, pml_width=c0["pml"], nt=nt,
        )

    batch = [
        torch.stack([c0[k], c1[k]]).to(dev, dtype).requires_grad_(True)
        for k in keys
    ]
    m0 = [c0[k].to(dev, dtype).requires_grad_(True) for k in keys]
    m1 = [c1[k].to(dev, dtype).requires_grad_(True) for k in keys]
    r, r_bg = run(batch)
    r0, r0_bg = run(m0, 0)
    r1, r1_bg = run(m1, 1)
    assert r.shape == (nt, 2, 3) and r_bg.shape == (nt, 2, 2)
    for out, ref, name in (
        (r[:, 0], r0[:, 0], "shot 0 fwd r"),
        (r[:, 1], r1[:, 0], "shot 1 fwd r"),
        (r_bg[:, 0], r0_bg[:, 0], "shot 0 fwd r_bg"),
        (r_bg[:, 1], r1_bg[:, 0], "shot 1 fwd r_bg"),
    ):
        assert (out.detach() - ref.detach()).abs().max().item() == 0.0, name
    loss = r.square().sum() + r_bg.square().sum()
    gs = torch.autograd.grad(loss, batch)
    g0s = torch.autograd.grad(r0.square().sum() + r0_bg.square().sum(), m0)
    g1s = torch.autograd.grad(r1.square().sum() + r1_bg.square().sum(), m1)
    for gb, g0, g1, name in zip(gs, g0s, g1s, keys, strict=True):
        rel0 = (gb[0] - g0).abs().max().item() / (g0.abs().max().item() + 1e-300)
        rel1 = (gb[1] - g1).abs().max().item() / (g1.abs().max().item() + 1e-300)
        assert rel0 < 1e-12 and rel1 < 1e-12, (
            f"batched-model grad_{name} rel err ({rel0}, {rel1})"
        )


def test_multi_shot_mixed_derived_coefficients():
    dtype = torch.float64
    c0 = build_case(dtype=dtype, nz=8, ny=8, nx=8, nt=6, pml=2,
                    device="cuda:0", seed=1)
    c1 = build_case(dtype=dtype, nz=8, ny=8, nx=8, nt=6, pml=2,
                    device="cuda:0", seed=2)
    dev = c0["device"]
    nz, ny, nx = c0["eps"].shape
    nt = c0["nt"]
    srcs, recs, bg_recs, amp = _multi_shot_survey(nz, ny, nx, nt, dtype)
    keys = ("eps", "sig", "mu", "deps", "dsig", "dmu")
    batched = {"sig", "dsig"}

    def run(models, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return em3d_born(
            *models, c0["dx"], c0["dt"],
            source_amplitudes=amp[sl].to(dev),
            source_locations=srcs[sl].to(dev),
            receiver_locations=recs[sl].to(dev),
            bg_receiver_locations=bg_recs[sl].to(dev),
            accuracy=2, pml_width=c0["pml"], nt=nt,
        )

    models = [
        (torch.stack([c0[k], c1[k]]) if k in batched else c0[k])
        .to(dev, dtype).requires_grad_(True)
        for k in keys
    ]
    singles = [
        [
            (m[i] if k in batched else m).detach().clone().requires_grad_(True)
            for k, m in zip(keys, models, strict=True)
        ]
        for i in range(2)
    ]
    out = run(models)
    refs = [run(singles[i], i) for i in range(2)]
    for i in range(2):
        for value, ref in zip(out, refs[i], strict=True):
            torch.testing.assert_close(value[:, i], ref[:, 0], rtol=0, atol=0)
    loss = sum(value.square().sum() for value in out)
    grads = torch.autograd.grad(loss, models)
    ref_grads = [
        torch.autograd.grad(sum(value.square().sum() for value in ref), single)
        for ref, single in zip(refs, singles, strict=True)
    ]
    for k, grad, g0, g1 in zip(keys, grads, *ref_grads, strict=True):
        expected = torch.stack([g0, g1]) if k in batched else g0 + g1
        torch.testing.assert_close(grad, expected, rtol=1e-12, atol=1e-20)
