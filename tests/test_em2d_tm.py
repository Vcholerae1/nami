"""nami em2d_tm: autograd gradcheck, grid convergence and a small FWI smoke test.

Self-contained tests: correctness is verified via numerical gradients
(``torch.autograd.gradcheck``), a two-grid convergence study of the spatial
FD order, and a loss-descent FWI smoke test.

nami imports are deferred to inside the test functions so that merely
importing this module (or running ``py_compile``) never triggers compilation
of the nami_em2d_tm CUDA extension.
"""

import math

import torch


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
    """Small random EM model; one source at centre, 3 nearby receivers.

    dx = 1e-3 m, dt = 1e-12 s, eps ~ 4-5.5, sig ~ 0-1e-3 S/m, mu = 1.
    With dt = 1e-12 the wave only travels ~0.14 cells/step, so receivers
    are placed within ~1.4 cells of the source (a far receiver would just
    record zeros).
    """
    torch.manual_seed(seed)
    dx = 1e-3
    dt = 1.0e-12
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    eps = 4.0 + 1.5 * torch.rand(ny, nx, generator=g)
    sig = 0.001 * torch.rand(ny, nx, generator=g)
    mu = torch.ones(ny, nx)
    srcs = torch.tensor([[[ny // 2, nx // 2]]])[:n_shots]
    recs = torch.tensor(
        [
            [
                [ny // 2, nx // 2],  # at the source cell: injection scaling
                [ny // 2, nx // 2 + 1],  # 1 cell away
                [ny // 2 + 1, nx // 2],  # 1 cell away
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


def _run_nami(c, eps, sig, mu, amp, accuracy=2):
    """Runs nami's em2d_tm (import deferred: never compiled at import time)."""
    from nami.em.em2d_tm import em2d_tm

    dev = c["device"]
    return em2d_tm(
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
    )


def test_gradcheck():
    """Numerical gradient check for (epsilon, sigma, mu)."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=24, nt=10, pml=4, device="cuda:0", seed=1
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
    print("gradcheck nami em2d_tm (eps, sig, mu):", ok)
    assert ok


def test_gradcheck_higher_order():
    """Numerical gradient check for (epsilon, sigma, mu) at accuracy 4/6/8.

    Same problem family as ``test_gradcheck`` but on a slightly larger grid
    (28 x 28) so the accuracy-8 stencil (fd_pad [4, 3, 4, 3]) fits inside
    the padded model region with room to spare.
    """
    dtype = torch.float64
    for acc in (4, 6, 8):
        c = build_case(
            dtype=dtype, ny=28, nx=28, nt=8, pml=4, device="cuda:0", seed=acc,
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
        print(f"gradcheck nami em2d_tm accuracy={acc} (eps, sig, mu):", ok)
        assert ok, f"gradcheck failed at accuracy={acc}"


def _ricker(freq, nt, dt, device, dtype):
    t = torch.arange(nt, device=device, dtype=dtype) * dt
    tau = 1.0 / freq
    arg = (torch.pi * freq * (t - tau)) ** 2
    return (1 - 2 * arg) * torch.exp(-arg)


def test_fwi_smoke():
    """20 Adam iterations inverting epsilon on a small random problem:
    loss must drop below half its initial value."""
    dtype = torch.float64
    dev = "cuda:0"
    torch.manual_seed(3)
    g = torch.Generator(device="cpu")
    g.manual_seed(7)
    ny, nx, nt, pml = 28, 30, 24, 5
    dx, dt = 1e-3, 3.0e-12  # ~0.64 x the 2D CFL limit for eps ~ 4-5.5

    eps_true = 4.0 + 1.5 * torch.rand(ny, nx, generator=g)
    eps_true[ny // 2 :] += 0.5
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(11)
    eps_init = 4.0 + 0.6 * torch.rand(ny, nx, generator=g2)
    g3 = torch.Generator(device="cpu")
    g3.manual_seed(13)
    sig = 0.001 * torch.rand(ny, nx, generator=g3)
    mu = torch.ones(ny, nx)

    srcs = torch.tensor(
        [[[ny // 2 - 3, nx // 2 - 3], [ny // 2 + 3, nx // 2 + 3]]]
    )
    recs = torch.tensor(
        [
            [
                [ny // 2, nx // 2 - 4],
                [ny // 2, nx // 2 - 2],
                [ny // 2, nx // 2 + 2],
                [ny // 2, nx // 2 + 4],
                [ny // 2 - 4, nx // 2],
            ]
        ]
    )
    # Ricker peaked at step 5, well resolved in nt = 24
    amp = (
        _ricker(1.0 / (5.0 * dt), nt, dt, dev, dtype)[None, None, :]
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
    opt = torch.optim.Adam([eps_opt], lr=0.05)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        # keep eps in a range where dt = 3e-12 stays below the CFL limit
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
        f"mid={losses[len(losses) // 2]:.6e} last={losses[-1]:.6e} "
        f"({losses[-1] / losses[0]:.3f}x)"
    )
    assert (
        losses[-1] < 0.5 * losses[0]
    ), f"FWI loss did not halve: {losses[0]:.6e} -> {losses[-1]:.6e}"


def _max_abs_diff(a, b):
    return (a - b).abs().max().item()


def _mod_gauss(freq, nt, dt, device, dtype, ncycles=5.0):
    """Narrow-band modulated Gaussian pulse (well-defined wavelength)."""
    t = torch.arange(nt, device=device, dtype=dtype) * dt
    t0 = ncycles / freq
    sigma = ncycles / (2.0 * torch.pi * freq)
    carrier = torch.sin(2 * torch.pi * freq * (t - t0))
    env = torch.exp(-0.5 * ((t - t0) / sigma) ** 2)
    return carrier * env


def test_grid_convergence():
    """Spatial FD order measured from a two-grid convergence study.

    A narrow-band point source in a homogeneous medium; on each grid the
    accuracy-2/4/6/8 runs share the identical source injection, time
    integration and CPML, so the order-to-order trace gaps
    (d24 = |tr2 - tr4|, d46 = |tr4 - tr6|, d68 = |tr6 - tr8|) isolate the
    spatial discretisation error: the common source/time/PML errors cancel
    in each gap.  Running the same physical setup at dx and dx/2 (dt scaled
    with dx), the gap between accuracy p and p+2 must shrink by ~2^p, i.e.
    the estimated order log2(gap(dx)/gap(dx/2)) is ~p.  Accuracy 2 uses the
    [1] staggered coefficient, 4 uses [9/8, -1/24], 6 uses
    [75/64, -25/384, 3/640], 8 uses [1225/1024, -245/3072, 49/5120,
    -5/7168]; the measured orders (2.0 / 3.8-4.0 / 5.7-5.9 here) match.
    """
    dtype = torch.float64
    dev = "cuda:0"
    # ~10 cells per wavelength at the coarse grid, CFL fraction 0.1
    dx, cpl, alpha, dist, pml, ncyc = 2.0e-3, 10, 0.1, 2.0, 15, 3.0
    c0 = 2.99792458e8
    v = c0 / (5.0 ** 0.5)
    lam = cpl * dx
    freq = v / lam
    half = 4.0  # half-domain size in wavelengths (source centred)
    eps = 5.0 * torch.ones(1, 1, device=dev, dtype=dtype)
    sig = torch.zeros(1, 1, device=dev, dtype=dtype)
    mu = torch.ones(1, 1, device=dev, dtype=dtype)

    def run(acc, h):
        ny = int(round(2 * half * lam / h))
        if ny % 2:
            ny += 1
        nx = ny
        dt = alpha * h / v
        r_cells = int(round(dist * lam / h))
        t_travel = dist * lam / v
        nt = int(math.ceil((1.6 * t_travel + ncyc / freq) / dt))
        srcs = torch.tensor([[[ny // 2, nx // 2]]], device=dev)
        recs = torch.tensor([[[ny // 2, nx // 2 + r_cells]]], device=dev)
        amp = _mod_gauss(freq, nt, dt, dev, dtype, ncycles=ncyc)[
            None, None, :
        ].contiguous()
        c = {
            "dx": h, "dt": dt, "nt": nt, "pml": pml,
            "srcs": srcs, "recs": recs, "device": dev,
        }
        return _run_nami(
            c, eps.expand(ny, nx), sig.expand(ny, nx), mu.expand(ny, nx),
            amp, accuracy=acc,
        ).detach().squeeze(1).squeeze(0)

    def gaps(h):
        tr = {acc: run(acc, h) for acc in (2, 4, 6, 8)}
        return {
            "g24": _max_abs_diff(tr[2], tr[4]),
            "g46": _max_abs_diff(tr[4], tr[6]),
            "g68": _max_abs_diff(tr[6], tr[8]),
        }

    g_coarse = gaps(dx)
    g_fine = gaps(dx / 2)
    orders = {
        "g24": math.log2(g_coarse["g24"] / g_fine["g24"]),
        "g46": math.log2(g_coarse["g46"] / g_fine["g46"]),
        "g68": math.log2(g_coarse["g68"] / g_fine["g68"]),
    }
    print("grid convergence gaps (dx):", g_coarse)
    print("grid convergence gaps (dx/2):", g_fine)
    print("measured orders:", {k: round(v, 2) for k, v in orders.items()})
    # error decreases with accuracy on the coarse grid
    assert g_coarse["g46"] < g_coarse["g24"]
    assert g_coarse["g68"] < g_coarse["g46"]
    # estimated order ~ accuracy: gap p -> p+2 shrinks by ~2^p
    assert 1.6 <= orders["g24"] <= 2.4, orders["g24"]
    assert 3.2 <= orders["g46"] <= 4.8, orders["g46"]
    assert 4.8 <= orders["g68"] <= 7.0, orders["g68"]


def test_float32_float64_parity():
    """Forward traces in float32 vs float64 agree to ~1e-4 relative."""
    dev = "cuda:0"
    c64 = build_case(dtype=torch.float64, ny=24, nx=24, nt=20, pml=4, seed=9)
    tr64 = _run_nami(
        c64,
        c64["eps"].to(dev, torch.float64),
        c64["sig"].to(dev, torch.float64),
        c64["mu"].to(dev, torch.float64),
        c64["amp"].to(dev, torch.float64),
    )
    c32 = build_case(dtype=torch.float32, ny=24, nx=24, nt=20, pml=4, seed=9)
    tr32 = _run_nami(
        c32,
        c32["eps"].to(dev, torch.float32),
        c32["sig"].to(dev, torch.float32),
        c32["mu"].to(dev, torch.float32),
        c32["amp"].to(dev, torch.float32),
    ).double()
    rel = _max_abs_diff(tr32, tr64) / tr64.abs().max().item()
    print(f"float32/float64 max relative diff: {rel:.3e}")
    assert rel < 1e-4, f"float32/float64 mismatch: {rel:.3e}"


def _run_nami_storage(c, eps, sig, mu, amp, storage="auto",
                      ckpt_steps=None):
    from nami.em.em2d_tm import em2d_tm

    dev = c["device"]
    return em2d_tm(
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
    """ckpt_steps=N reproduces full-storage forward and gradients
    (bit-identical forward, max-relative gradient error 0.0)."""
    dtype = torch.float64
    c = build_case(dtype=dtype, device="cuda:0", seed=1)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sig = c["sig"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    r_full = _run_nami_storage(c, eps, sig, mu, amp, "auto", 0).detach()
    g_full = torch.autograd.grad(
        _run_nami_storage(c, eps, sig, mu, amp, "auto", 0).square().sum(), eps
    )[0]
    for ckpt in (2, 4, 7, c["nt"]):
        r = _run_nami_storage(c, eps, sig, mu, amp, "auto", ckpt).detach()
        g = torch.autograd.grad(
            _run_nami_storage(c, eps, sig, mu, amp, "auto", ckpt).square().sum(),
            eps,
        )[0]
        assert (r - r_full).abs().max().item() == 0.0, f"ckpt_steps={ckpt} fwd"
        rel = (g - g_full).abs().max().item() / (
            g_full.abs().max().item() + 1e-300
        )
        assert rel == 0.0, f"ckpt_steps={ckpt} grad rel err {rel}"


def test_checkpoint_auto_selection():
    """Default call auto-enables snapshots (input requires grad);
    ckpt_steps=None picks ~sqrt(nt) and reproduces explicit full
    storage."""
    dtype = torch.float64
    c = build_case(dtype=dtype, device="cuda:0", seed=1)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sig = c["sig"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    out = _run_nami_storage(c, eps, sig, mu, amp)
    g_auto = torch.autograd.grad(out.square().sum(), eps)[0]
    g_full = torch.autograd.grad(
        _run_nami_storage(c, eps, sig, mu, amp, "auto", 0).square().sum(), eps
    )[0]
    rel = (g_auto - g_full).abs().max().item() / (
        g_full.abs().max().item() + 1e-300
    )
    assert rel == 0.0, f"auto grad rel err {rel}"


def test_storage_false_is_forward_only():
    """storage='none' runs the forward but backward raises RuntimeError."""
    dtype = torch.float64
    c = build_case(dtype=dtype, device="cuda:0", seed=1)
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
    (integral sampling: each sampled snapshot represents `stride` time
    steps)."""
    dtype = torch.float64
    c = build_case(dtype=dtype, device="cuda:0", seed=1)
    dev = c["device"]
    eps = c["eps"].to(dev, dtype).requires_grad_(True)
    sig = c["sig"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    from nami.em.em2d_tm import em2d_tm

    def nami_grad(i):
        e = eps.clone().requires_grad_(True)
        out = em2d_tm(
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


def _multi_shot_survey(ny, nx, nt, dtype):
    """Two shots with distinct near-source receivers (short travel distance)."""
    srcs = torch.tensor([[[ny // 2, nx // 2 - 1]], [[ny // 2 + 1, nx // 2 + 1]]])
    recs = torch.tensor(
        [
            [[ny // 2, nx // 2 - 1], [ny // 2, nx // 2], [ny // 2 + 1, nx // 2 - 1]],
            [[ny // 2 + 1, nx // 2 + 1], [ny // 2 + 1, nx // 2],
             [ny // 2, nx // 2 + 1]],
        ]
    )
    amp = torch.zeros(2, 1, nt, dtype=dtype)
    amp[0, 0, :3] = torch.tensor([1.0, -0.5, 0.2], dtype=dtype)
    amp[1, 0, :3] = torch.tensor([0.7, 0.3, -0.1], dtype=dtype)
    return srcs, recs, amp


def test_multi_shot_shared_model():
    """n_shots=2 with shared [ny, nx] models: forward matches single-shot runs
    and shared-model gradients are the sums of the single-shot gradients."""
    from nami.em.em2d_tm import em2d_tm

    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=28, nt=12, pml=4, device="cuda:0", seed=1
    )
    dev = c["device"]
    ny, nx = c["eps"].shape
    nt = c["nt"]
    srcs, recs, amp = _multi_shot_survey(ny, nx, nt, dtype)

    def run(eps, sig, mu, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return em2d_tm(
            eps, sig, mu, c["dx"], c["dt"],
            source_amplitudes=amp[sl].to(dev),
            source_locations=srcs[sl].to(dev),
            receiver_locations=recs[sl].to(dev),
            accuracy=2, pml_width=c["pml"], nt=nt,
        )

    params = {
        "eps": c["eps"].to(dev, dtype),
        "sig": c["sig"].to(dev, dtype),
        "mu": c["mu"].to(dev, dtype),
    }
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
        print(f"em2d multi-shot shared-model grad_{name} rel err {rel:.3e}")
        assert rel < 1e-12, f"shared-model grad_{name} rel err {rel}"


def test_multi_shot_batched_model():
    """n_shots=2 with batched [2, ny, nx] models: shot i uses slice i."""
    from nami.em.em2d_tm import em2d_tm

    dtype = torch.float64
    c0 = build_case(
        dtype=dtype, ny=24, nx=28, nt=12, pml=4, device="cuda:0", seed=1
    )
    c1 = build_case(
        dtype=dtype, ny=24, nx=28, nt=12, pml=4, device="cuda:0", seed=2
    )
    dev = c0["device"]
    ny, nx = c0["eps"].shape
    nt = c0["nt"]
    srcs, recs, amp = _multi_shot_survey(ny, nx, nt, dtype)

    def run(eps, sig, mu, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return em2d_tm(
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
        print(f"em2d multi-shot batched-model grad_{name} rel err "
              f"({rel0:.3e}, {rel1:.3e})")
        assert rel0 < 1e-12 and rel1 < 1e-12, (
            f"batched-model grad_{name} rel err ({rel0}, {rel1})"
        )


def test_multi_shot_mixed_model_batching():
    """Derived coefficients must follow their actual broadcasted batch shape."""
    from nami.em.em2d_tm import em2d_tm

    dtype = torch.float64
    c0 = build_case(dtype=dtype, ny=24, nx=28, nt=12, pml=4,
                    device="cuda:0", seed=1)
    c1 = build_case(dtype=dtype, ny=24, nx=28, nt=12, pml=4,
                    device="cuda:0", seed=2)
    dev = c0["device"]
    ny, nx = c0["eps"].shape
    nt = c0["nt"]
    srcs, recs, amp = _multi_shot_survey(ny, nx, nt, dtype)

    def run(eps, sig, mu, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return em2d_tm(
            eps, sig, mu, c0["dx"], c0["dt"],
            source_amplitudes=amp[sl].to(dev),
            source_locations=srcs[sl].to(dev),
            receiver_locations=recs[sl].to(dev),
            accuracy=2, pml_width=c0["pml"], nt=nt,
        )

    eps = c0["eps"].to(dev, dtype).requires_grad_(True)
    sig = torch.stack([c0["sig"], c1["sig"]]).to(dev, dtype).requires_grad_(True)
    mu = c0["mu"].to(dev, dtype).requires_grad_(True)
    eps0 = eps.detach().clone().requires_grad_(True)
    eps1 = eps.detach().clone().requires_grad_(True)
    sig0 = sig[0].detach().clone().requires_grad_(True)
    sig1 = sig[1].detach().clone().requires_grad_(True)
    mu0 = mu.detach().clone().requires_grad_(True)
    mu1 = mu.detach().clone().requires_grad_(True)
    r = run(eps, sig, mu)
    r0 = run(eps0, sig0, mu0, 0)
    r1 = run(eps1, sig1, mu1, 1)
    torch.testing.assert_close(r[:, 0], r0[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(r[:, 1], r1[:, 0], rtol=0, atol=0)
    geps, gsig, gmu = torch.autograd.grad(r.square().sum(), (eps, sig, mu))
    g0 = torch.autograd.grad(r0.square().sum(), (eps0, sig0, mu0))
    g1 = torch.autograd.grad(r1.square().sum(), (eps1, sig1, mu1))
    torch.testing.assert_close(geps, g0[0] + g1[0], rtol=1e-12, atol=1e-20)
    torch.testing.assert_close(gsig[0], g0[1], rtol=1e-12, atol=1e-20)
    torch.testing.assert_close(gsig[1], g1[1], rtol=1e-12, atol=1e-20)
    torch.testing.assert_close(gmu, g0[2] + g1[2], rtol=1e-12, atol=1e-20)
