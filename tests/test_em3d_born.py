"""nami em3d_born: Born linearity vs the full EM solve, autograd gradcheck,
finite-difference gradient consistency, and storage/forward sanity.

Self-contained tests (same approach as test_em2d_tm_born.py / test_em3d.py):
correctness is verified via numerical gradients
(``torch.autograd.gradcheck``), the first-order Born linearity relation (the
scattered solve matches the perturbed full solve to O(delta^2)), and central
finite differences of the analytic gradients.

nami imports are deferred to inside the test functions so that merely
importing this module never triggers compilation of the nami_em3d_born CUDA
extension.
"""

import torch


def _ensure_extension_importable():
    """Make ``nami_em3d_born`` and ``nami_em3d`` importable when the build
    recipe only compiles the born extension into /tmp
    (``torch.utils.cpp_extension.load`` returns the module without
    registering it on ``sys.path``; the recipe sets ``sys.modules``
    explicitly).  No-op when the extensions are already importable; does not
    compile anything itself.
    """
    import os
    import sys

    try:
        import nami_em3d_born  # noqa: F401
    except ModuleNotFoundError:
        candidates = ["/tmp/nami_agent_em3d_born", "/tmp/nami_agent_em3d"]
        build = os.path.join(os.path.dirname(__file__), "..", "build")
        if os.path.isdir(build):
            candidates += [
                os.path.join(build, d)
                for d in sorted(os.listdir(build))
                if d.startswith("lib.")
            ]
        for d in candidates:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.startswith("nami_em3d_born") and f.endswith((".so", ".pyd")):
                    if d not in sys.path:
                        sys.path.insert(0, d)
                    break
            else:
                continue
            break
    try:
        import nami_em3d  # noqa: F401
    except ModuleNotFoundError:
        candidates = ["/tmp/nami_agent_em3d", "/tmp/nami_agent_em3d_clean_v2"]
        build = os.path.join(os.path.dirname(__file__), "..", "build")
        if os.path.isdir(build):
            candidates += [
                os.path.join(build, d)
                for d in sorted(os.listdir(build))
                if d.startswitZXsdfghjkl.
                h("lib.")
            ]
        for d in candidates:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.startswith("nami_em3d") and f.endswith((".so", ".pyd")):
                    if d not in sys.path:
                        sys.path.insert(0, d)
                    return


_ensure_extension_importable()


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
    from nami.em.em3d import em3d

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
    """Runs nami's em3d_born (import deferred: never compiled at import)."""
    from nami.em.em3d_born import em3d_born

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


def test_gradcheck_accuracy4():
    """Same gradcheck at accuracy 4 (coefficient-driven stencils)."""
    dtype = torch.float64
    c = build_case(dtype=dtype, nz=8, ny=8, nx=8, nt=6, pml=2,
                   device="cuda:0", seed=2)
    dev = c["device"]

    def fn(eps, sigma, mu, deps, dsig, dmu, amp):
        return _run_born(c, eps, sigma, mu, deps, dsig, dmu, amp, accuracy=4)

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
    print("gradcheck nami em3d_born accuracy 4:", ok)
    assert ok


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
            # em3d CUDA backward reduces in nondeterministic order, so even
            # ckpt_steps=0 vs 0 differs by ~1 ULP of the largest term.
            # 1e-12 keeps parity semantics while absorbing that noise.
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
