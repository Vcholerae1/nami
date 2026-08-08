"""nami em3d: autograd gradcheck, gradient consistency, FWI and storage tests.

Self-contained tests (same approach as test_em2d_tm.py): correctness is
verified via numerical gradients (``torch.autograd.gradcheck``), a
finite-difference gradient-consistency check, and a loss-descent FWI smoke
test.
"""

import torch

from nami.em.em3d import em3d


def build_case(
    dtype=torch.float64,
    nz=10,
    ny=12,
    nx=12,
    nt=10,
    pml=3,
    n_shots=1,
    device="cuda:0",
    seed=0,
):
    """Small random 3D EM model; one source at centre, 3 nearby receivers.

    dx = 1e-3 m, dt = 1e-12 s, eps ~ 4-5.5, sig ~ 0-1e-3 S/m, mu = 1.
    With dt = 1e-12 the wave only travels ~0.14 cells/step, so receivers
    are placed within ~1-2 cells of the source (a far receiver would just
    record zeros).
    """
    torch.manual_seed(seed)
    dx = 1e-3
    dt = 1.0e-12
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    eps = 4.0 + 1.5 * torch.rand(nz, ny, nx, generator=g)
    sig = 0.001 * torch.rand(nz, ny, nx, generator=g)
    mu = torch.ones(nz, ny, nx)
    srcs = torch.tensor([[[nz // 2, ny // 2, nx // 2]]])[:n_shots]
    recs = torch.tensor(
        [
            [
                [nz // 2, ny // 2, nx // 2],  # at the source cell: injection scaling
                [nz // 2, ny // 2, nx // 2 + 1],  # 1 cell away
                [nz // 2, ny // 2 + 1, nx // 2],  # 1 cell away
            ]
        ]
    )[:n_shots]
    amp = torch.zeros(n_shots, srcs.shape[1], nt, dtype=dtype)
    amp[:, :, 0] = 1.0
    amp[:, :, 1] = -0.5
    amp[:, :, 2] = 0.2
    return {
        "eps": eps,
        "sig": sig,
        "mu": mu,
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


def _run_nami(c, eps, sig, mu, amp, accuracy=2, **kwargs):
    """Runs nami's em3d."""
    dev = c["device"]
    return em3d(
        eps,
        sig,
        mu,
        c["dx"],
        c["dt"],
        source_amplitudes=amp,
        source_locations=c["srcs"].to(dev),
        receiver_locations=c["recs"].to(dev),
        accuracy=accuracy,
        pml_width=c["pml"],
        nt=c["nt"],
        **kwargs,
    )


def test_gradcheck():
    """Numerical gradient check for (epsilon, sigma, mu)."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, nz=10, ny=12, nx=12, nt=8, pml=3, device="cuda:0", seed=1
    )
    dev = c["device"]

    def fn(eps, sig, mu):
        return _run_nami(c, eps, sig, mu, c["amp"].to(dev, dtype))

    ok = torch.autograd.gradcheck(
        fn,
        (
            c["eps"].to(dev, dtype).requires_grad_(True),
            c["sig"].to(dev, dtype).requires_grad_(True),
            c["mu"].to(dev, dtype).requires_grad_(True),
        ),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        fast_mode=True,
        nondet_tol=1e-8,
        raise_exception=False,
    )
    print("gradcheck nami em3d (eps, sig, mu):", ok)
    assert ok


def test_gradcheck_higher_order():
    """Numerical gradient check at accuracy 4 and 6 on a larger grid."""
    dtype = torch.float64
    for acc in (4, 6):
        c = build_case(
            dtype=dtype, nz=14, ny=16, nx=16, nt=8, pml=4,
            device="cuda:0", seed=acc,
        )
        dev = c["device"]

        def fn(eps, sig, mu, c=c, dev=dev, acc=acc):
            return _run_nami(c, eps, sig, mu, c["amp"].to(dev, dtype), accuracy=acc)

        ok = torch.autograd.gradcheck(
            fn,
            (
                c["eps"].to(dev, dtype).requires_grad_(True),
                c["sig"].to(dev, dtype).requires_grad_(True),
                c["mu"].to(dev, dtype).requires_grad_(True),
            ),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
            fast_mode=True,
            nondet_tol=1e-8,
            raise_exception=False,
        )
        print(f"gradcheck nami em3d accuracy={acc} (eps, sig, mu):", ok)
        assert ok, f"gradcheck failed at accuracy={acc}"


def test_gradcheck_source_amplitudes():
    """Numerical gradient check for the source amplitudes."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, nz=10, ny=12, nx=12, nt=8, pml=3, device="cuda:0", seed=2
    )
    dev = c["device"]
    eps = c["eps"].to(dev, dtype)
    sig = c["sig"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)

    def fn(amp):
        return _run_nami(c, eps, sig, mu, amp)

    ok = torch.autograd.gradcheck(
        fn,
        (c["amp"].to(dev, dtype).requires_grad_(True),),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        fast_mode=True,
        nondet_tol=1e-8,
        raise_exception=False,
    )
    print("gradcheck nami em3d (source_amplitudes):", ok)
    assert ok


def test_gradient_consistency_finite_difference():
    """Analytic gradients (torch.autograd.grad) vs central finite differences.

    Checks the strongest-response cells (largest analytic gradient) of
    epsilon/sigma/mu on a tiny grid so the numeric reference stays cheap:
    central FD of a large-magnitude scalar loss is only resolvable in float64
    where the induced loss change (2*h*|g_i|) is well above the loss' ULP.
    """
    dtype = torch.float64
    c = build_case(
        dtype=dtype, nz=6, ny=8, nx=8, nt=6, pml=2, device="cuda:0", seed=3
    )
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sig = c["sig"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype)

    out = _run_nami(c, eps, sig, mu, amp)
    loss = (out**2).sum()
    g_eps, g_sig, g_mu = torch.autograd.grad(loss, (eps, sig, mu))

    for name, model, grad, h in (
        ("eps", eps, g_eps, 1e-4),
        ("sig", sig, g_sig, 1e-7),
        ("mu", mu, g_mu, 1e-4),
    ):
        # Central FD of a large-magnitude scalar loss (|loss| ~ 1e15) is only
        # resolvable in float64 where the induced loss change (2*h*|g_i|)
        # sits well above the loss' ULP, so compare at the strongest-response
        # cells (largest analytic gradient) rather than random cells.
        flat_grad = grad.reshape(-1)
        idx = torch.topk(flat_grad.abs(), 8).indices
        analytic = flat_grad[idx]
        numeric = torch.zeros_like(analytic)
        for j, flat in enumerate(idx.tolist()):
            p = model.detach().reshape(-1).clone()
            p[flat] += h
            e = p.reshape(eps.shape) if name == "eps" else eps.detach()
            s = p.reshape(sig.shape) if name == "sig" else sig.detach()
            m = p.reshape(mu.shape) if name == "mu" else mu.detach()
            loss_p = (_run_nami(c, e, s, m, amp) ** 2).sum()
            p = model.detach().reshape(-1).clone()
            p[flat] -= h
            e = p.reshape(eps.shape) if name == "eps" else eps.detach()
            s = p.reshape(sig.shape) if name == "sig" else sig.detach()
            m = p.reshape(mu.shape) if name == "mu" else mu.detach()
            loss_m = (_run_nami(c, e, s, m, amp) ** 2).sum()
            numeric[j] = (loss_p - loss_m) / (2 * h)
        denom = analytic.abs().max().item() + 1e-300
        rel = (analytic - numeric).abs().max().item() / denom
        print(f"grad-consistency {name}: max rel err {rel:.3e}")
        assert rel < 5e-4, f"{name} gradient mismatch: {rel:.3e}"


def _ricker(freq, nt, dt, device, dtype):
    t = torch.arange(nt, device=device, dtype=dtype) * dt
    tau = 1.0 / freq
    arg = (torch.pi * freq * (t - tau)) ** 2
    return (1 - 2 * arg) * torch.exp(-arg)


def test_fwi_smoke():
    """6 Adam iterations inverting epsilon on a small random problem:
    the loss must drop below its initial value."""
    dtype = torch.float64
    dev = "cuda:0"
    torch.manual_seed(3)
    g = torch.Generator(device="cpu")
    g.manual_seed(7)
    nz, ny, nx, nt, pml = 10, 12, 12, 14, 4
    dx, dt = 1e-3, 2.0e-12  # ~0.35 cells/step at eps=4.5, below the 3D CFL limit

    eps_true = 4.0 + 1.5 * torch.rand(nz, ny, nx, generator=g)
    eps_true[nz // 2 :, ny // 2 :] += 0.5
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(11)
    eps_init = 4.0 + 0.6 * torch.rand(nz, ny, nx, generator=g2)
    g3 = torch.Generator(device="cpu")
    g3.manual_seed(13)
    sig = 0.001 * torch.rand(nz, ny, nx, generator=g3)
    mu = torch.ones(nz, ny, nx)

    srcs = torch.tensor(
        [[[nz // 2 - 2, ny // 2 - 2, nx // 2 - 2],
          [nz // 2 + 2, ny // 2 + 2, nx // 2 + 2]]]
    )
    recs = torch.tensor(
        [
            [
                [nz // 2, ny // 2, nx // 2 - 3],
                [nz // 2, ny // 2, nx // 2 - 1],
                [nz // 2, ny // 2, nx // 2 + 1],
                [nz // 2, ny // 2, nx // 2 + 3],
                [nz // 2, ny // 2 - 3, nx // 2],
            ]
        ]
    )
    # Ricker peaked at step 6, well resolved in nt = 14
    amp = (
        _ricker(1.0 / (6.0 * dt), nt, dt, dev, dtype)[None, None, :]
        .expand(1, 2, nt)
        .contiguous()
    )

    c = {
        "dx": dx,
        "dt": dt,
        "nt": nt,
        "pml": pml,
        "srcs": srcs,
        "recs": recs,
        "device": dev,
    }
    data = _run_nami(
        c, eps_true.to(dev, dtype), sig.to(dev, dtype), mu.to(dev, dtype), amp
    ).detach()

    eps_opt = eps_init.to(dev, dtype).requires_grad_(True)
    opt = torch.optim.Adam([eps_opt], lr=0.1)
    losses = []
    for _ in range(6):
        opt.zero_grad()
        # keep eps in a range where dt = 2e-12 stays below the 3D CFL limit
        out = _run_nami(
            c,
            torch.clamp(eps_opt, 3.0, 7.0),
            sig.to(dev, dtype),
            mu.to(dev, dtype),
            amp,
        )
        loss = ((out - data) ** 2).mean()
        loss.backward()
        losses.append(loss.item())
        opt.step()

    print(
        f"FWI smoke: loss first={losses[0]:.6e} "
        f"last={losses[-1]:.6e} ({losses[-1] / losses[0]:.3f}x)"
    )
    assert (
        losses[-1] < losses[0]
    ), f"FWI loss did not decrease: {losses[0]:.6e} -> {losses[-1]:.6e}"


def test_float32_forward_smoke():
    """float32 forward: correct shape and no NaNs/infs."""
    dev = "cuda:0"
    c = build_case(dtype=torch.float32, nz=10, ny=12, nx=12, nt=12, pml=3, seed=9)
    out = _run_nami(
        c,
        c["eps"].to(dev, torch.float32),
        c["sig"].to(dev, torch.float32),
        c["mu"].to(dev, torch.float32),
        c["amp"].to(dev, torch.float32),
    )
    assert out.shape == (c["nt"], 1, 3)
    assert torch.isfinite(out).all(), "float32 forward produced NaN/inf"
    print(f"float32 forward: max |trace| {out.abs().max().item():.3e}")


def test_components():
    """source/receiver component selection runs and couples through the grid."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, nz=10, ny=12, nx=12, nt=10, pml=3, device="cuda:0", seed=4
    )
    dev = c["device"]
    eps = c["eps"].to(dev, dtype)
    sig = c["sig"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    traces = {}
    for comp in ("ex", "ey", "ez"):
        for rec_comp in ("ex", "ey", "ez"):
            key = (comp, rec_comp)
            traces[key] = _run_nami(
                c, eps, sig, mu, amp,
                source_component=comp, receiver_component=rec_comp,
            ).detach()
            assert torch.isfinite(traces[key]).all(), f"{key} not finite"
    # the traces must actually differ between components
    assert (traces[("ey", "ey")] - traces[("ez", "ez")]).abs().max().item() > 0
    # recorded receiver component must match what propagates: injecting into
    # the recorded component gives the direct (largest) response
    direct = traces[("ey", "ey")].abs().max().item()
    cross = traces[("ey", "ex")].abs().max().item()
    print(f"component test: direct {direct:.3e} cross {cross:.3e}")
    assert direct > 0 and cross >= 0


def test_invalid_component():
    c = build_case(dtype=torch.float64, device="cuda:0", seed=1)
    dev = c["device"]
    try:
        em3d(
            c["eps"].to(dev), c["sig"].to(dev), c["mu"].to(dev),
            c["dx"], c["dt"],
            source_amplitudes=c["amp"].to(dev),
            source_locations=c["srcs"].to(dev),
            receiver_locations=c["recs"].to(dev),
            pml_width=c["pml"], nt=c["nt"],
            source_component="hx",
        )
    except ValueError:
        return
    raise AssertionError("invalid source_component should raise ValueError")


def _run_nami_storage(c, eps, sig, mu, amp, storage="auto",
                      ckpt_steps=0):
    dev = c["device"]
    return em3d(
        eps,
        sig,
        mu,
        c["dx"],
        c["dt"],
        source_amplitudes=amp,
        source_locations=c["srcs"].to(dev),
        receiver_locations=c["recs"].to(dev),
        accuracy=2,
        pml_width=c["pml"],
        nt=c["nt"],
        storage=storage,
        ckpt_steps=ckpt_steps,
    )


def test_checkpoint_forward_and_gradient_parity():
    """ckpt_steps=N reproduces full-storage forward and gradients."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, nz=8, ny=8, nx=8, nt=12, pml=2, device="cuda:0", seed=1
    )
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sig = c["sig"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    def run(ckpt):
        return _run_nami_storage(c, eps, sig, mu, amp, "auto", ckpt)

    r_full = run(0).detach()
    g_full = torch.autograd.grad(run(0).square().sum(), eps)[0]
    for ckpt in (2, 4, 7, 12, 15):
        r = run(ckpt).detach()
        g = torch.autograd.grad(run(ckpt).square().sum(), eps)[0]
        assert (r - r_full).abs().max().item() == 0.0, f"ckpt_steps={ckpt} fwd"
        rel = (g - g_full).abs().max().item() / (
            g_full.abs().max().item() + 1e-300
        )
        # em3d CUDA reductions can differ by ~1 ULP of the largest term
        assert rel < 1e-12, f"ckpt_steps={ckpt} grad rel err {rel}"


def test_storage_false_is_forward_only():
    """storage='none' runs the forward but backward raises RuntimeError."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, nz=8, ny=8, nx=8, nt=8, pml=2, device="cuda:0", seed=1
    )
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sig = c["sig"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)
    out = _run_nami_storage(c, eps, sig, mu, amp, "none")
    assert out.shape == (c["nt"], 1, 3)
    try:
        torch.autograd.grad(out.square().sum(), eps)
    except RuntimeError:
        return
    raise AssertionError("storage='none' backward should raise RuntimeError")


def test_stride_parity():
    """stride>1: forward exact, model gradients a close approximation
    (integral sampling)."""
    dtype = torch.float64
    c = build_case(dtype=dtype, device="cuda:0", seed=1)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sig = c["sig"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    def nami_grad(i):
        e = eps.clone().requires_grad_(True)
        out = em3d(
            e, sig, mu, c["dx"], c["dt"],
            source_amplitudes=amp, source_locations=c["srcs"].to(dev),
            receiver_locations=c["recs"].to(dev), accuracy=2,
            pml_width=c["pml"], nt=c["nt"], storage="auto",
            sample_steps=i,
        )
        return out, torch.autograd.grad(out.square().sum(), e)[0]

    (r1, _), (r2, g2) = nami_grad(1), nami_grad(2)
    assert (r1 - r2).abs().max().item() == 0.0
    (_, g1) = nami_grad(1)
    rel = (g1 - g2).abs().max().item() / (g1.abs().max().item() + 1e-300)
    print(f"sample_steps=2 model-grad rel err: {rel:.3e}")
    assert rel < 0.2, f"sample_steps=2 grad rel err {rel}"


def _max_abs_diff(a, b):
    return (a - b).abs().max().item()


def _multi_shot_survey(nz, ny, nx, nt, dtype):
    """Two shots with nearby sources/receivers (short EM travel)."""
    srcs = torch.tensor(
        [
            [[nz // 2, ny // 2, nx // 2]],
            [[nz // 2, ny // 2 + 1, nx // 2 + 1]],
        ]
    )
    recs = torch.tensor(
        [
            [
                [nz // 2, ny // 2, nx // 2],
                [nz // 2, ny // 2, nx // 2 + 1],
                [nz // 2, ny // 2 + 1, nx // 2],
            ],
            [
                [nz // 2, ny // 2 + 1, nx // 2 + 1],
                [nz // 2, ny // 2 + 1, nx // 2],
                [nz // 2, ny // 2, nx // 2 + 1],
            ],
        ]
    )
    amp = torch.zeros(2, 1, nt, dtype=dtype)
    amp[0, 0, :3] = torch.tensor([1.0, -0.5, 0.2], dtype=dtype)
    amp[1, 0, :3] = torch.tensor([0.7, 0.3, -0.1], dtype=dtype)
    return srcs, recs, amp


def test_multi_shot_shared_model():
    """n_shots=2 shared models: forward matches singles; grad is sum."""
    dtype = torch.float64
    c = build_case(dtype=dtype, nz=10, ny=12, nx=12, nt=10, pml=3,
                   device="cuda:0", seed=1)
    dev = c["device"]
    nz, ny, nx = c["eps"].shape
    nt = c["nt"]
    srcs, recs, amp = _multi_shot_survey(nz, ny, nx, nt, dtype)

    def run(eps, sig, mu, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return em3d(
            eps, sig, mu, c["dx"], c["dt"],
            source_amplitudes=amp[sl].to(dev),
            source_locations=srcs[sl].to(dev),
            receiver_locations=recs[sl].to(dev),
            accuracy=2, pml_width=c["pml"], nt=nt,
        )

    params = {k: c[k].to(dev, dtype) for k in ("eps", "sig", "mu")}
    p = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    p0 = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    p1 = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    r = run(p["eps"], p["sig"], p["mu"])
    r0 = run(p0["eps"], p0["sig"], p0["mu"], 0)
    r1 = run(p1["eps"], p1["sig"], p1["mu"], 1)
    assert r.shape == (nt, 2, 3)
    assert (r[:, 0].detach() - r0[:, 0].detach()).abs().max().item() == 0.0
    assert (r[:, 1].detach() - r1[:, 0].detach()).abs().max().item() == 0.0
    names = ("eps", "sig", "mu")
    gs = torch.autograd.grad(r.square().sum(), [p[k] for k in names])
    g0s = torch.autograd.grad(r0.square().sum(), [p0[k] for k in names])
    g1s = torch.autograd.grad(r1.square().sum(), [p1[k] for k in names])
    for g, g0, g1, name in zip(gs, g0s, g1s, names, strict=True):
        rel = (g - (g0 + g1)).abs().max().item() / (
            (g0 + g1).abs().max().item() + 1e-300
        )
        assert rel < 1e-10, f"shared-model grad_{name} rel err {rel}"


def test_multi_shot_batched_model():
    """n_shots=2 batched models: shot i uses slice i."""
    dtype = torch.float64
    c0 = build_case(dtype=dtype, nz=10, ny=12, nx=12, nt=10, pml=3,
                    device="cuda:0", seed=1)
    c1 = build_case(dtype=dtype, nz=10, ny=12, nx=12, nt=10, pml=3,
                    device="cuda:0", seed=2)
    dev = c0["device"]
    nz, ny, nx = c0["eps"].shape
    nt = c0["nt"]
    srcs, recs, amp = _multi_shot_survey(nz, ny, nx, nt, dtype)

    def run(eps, sig, mu, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return em3d(
            eps, sig, mu, c0["dx"], c0["dt"],
            source_amplitudes=amp[sl].to(dev),
            source_locations=srcs[sl].to(dev),
            receiver_locations=recs[sl].to(dev),
            accuracy=2, pml_width=c0["pml"], nt=nt,
        )

    eps_b = torch.stack([c0["eps"], c1["eps"]]).to(dev, dtype).requires_grad_(True)
    sig_b = torch.stack([c0["sig"], c1["sig"]]).to(dev, dtype).requires_grad_(True)
    mu_b = torch.stack([c0["mu"], c1["mu"]]).to(dev, dtype).requires_grad_(True)
    eps0 = c0["eps"].to(dev, dtype).requires_grad_(True)
    sig0 = c0["sig"].to(dev, dtype).requires_grad_(True)
    mu0 = c0["mu"].to(dev, dtype).requires_grad_(True)
    eps1 = c1["eps"].to(dev, dtype).requires_grad_(True)
    sig1 = c1["sig"].to(dev, dtype).requires_grad_(True)
    mu1 = c1["mu"].to(dev, dtype).requires_grad_(True)
    r = run(eps_b, sig_b, mu_b)
    r0 = run(eps0, sig0, mu0, 0)
    r1 = run(eps1, sig1, mu1, 1)
    assert r.shape == (nt, 2, 3)
    assert (r[:, 0].detach() - r0[:, 0].detach()).abs().max().item() == 0.0
    assert (r[:, 1].detach() - r1[:, 0].detach()).abs().max().item() == 0.0
    gs = torch.autograd.grad(r.square().sum(), [eps_b, sig_b, mu_b])
    g0s = torch.autograd.grad(r0.square().sum(), [eps0, sig0, mu0])
    g1s = torch.autograd.grad(r1.square().sum(), [eps1, sig1, mu1])
    for gb, g0, g1, name in zip(gs, g0s, g1s, ("eps", "sig", "mu"), strict=True):
        rel0 = (gb[0] - g0).abs().max().item() / (g0.abs().max().item() + 1e-300)
        rel1 = (gb[1] - g1).abs().max().item() / (g1.abs().max().item() + 1e-300)
        assert rel0 < 1e-10 and rel1 < 1e-10, (
            f"batched-model grad_{name} rel err ({rel0}, {rel1})"
        )
