"""nami elastic2d: autograd gradcheck, a small FWI smoke test, and
checkpoint-storage checks.

nami imports are deferred to inside the test functions so that merely
importing this module (or running py_compile) never triggers compilation of
the nami_elastic2d CUDA extension.
"""

import math

import torch


def build_case(
    dtype=torch.float64,
    ny=40,
    nx=48,
    nt=100,
    pml=8,
    n_shots=1,
    device="cuda:0",
    seed=0,
    rec_offsets=(3, 3, 3),
):
    """Small random vp/vs/rho model converted to lamb/mu/buoyancy.

    dx = 1e-3 m, dt = 1e-7 s.  With vp <= 2500 m/s the CFL limit is
    max_dt = 0.6 * dx / (sqrt(2) * vp_max) ~= 1.70e-7 s, so dt is ~1.7x
    below it: nami elastic2d does not sub-step and stays stable on this
    time grid.
    """
    torch.manual_seed(seed)
    dx, dt = 1e-3, 1e-7
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    vp = 2000 + 500 * torch.rand(ny, nx, generator=g)
    vs = 1100 + 300 * torch.rand(ny, nx, generator=g)
    rho = 2000 + 300 * torch.rand(ny, nx, generator=g)
    # vp_min = 2000 > sqrt(2) * vs_max ~= 1980 keeps vp^2 - 2*vs^2 > 0.
    lamb = rho * (vp**2 - 2 * vs**2)
    mu = rho * vs**2
    buoy = 1.0 / rho
    srcs = torch.tensor([[[ny // 2, nx // 2], [ny // 2, nx // 2 + 2]]])[:n_shots]
    recs = torch.tensor(
        [
            [
                [ny // 2, nx // 2 - rec_offsets[0]],
                [ny // 2, nx // 2 + rec_offsets[1]],
                [ny // 2 - rec_offsets[2], nx // 2],
            ]
        ]
    )[:n_shots]
    amp = torch.zeros(n_shots, srcs.shape[1], nt, dtype=dtype)
    amp[:, :, 0] = 1.0
    amp[:, :, 1] = -0.5
    amp[:, :, 2] = 0.2
    return {
        "lamb": lamb,
        "mu": mu,
        "buoy": buoy,
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


def _run_nami(c, lamb, mu, buoy, amp, accuracy=2):
    from nami.elastic.elastic2d import elastic2d

    dev = c["device"]
    return elastic2d(
        lamb,
        mu,
        buoy,
        c["dx"],
        c["dt"],
        source_amplitudes=amp,
        source_locations=c["srcs"].to(dev),
        receiver_locations=c["recs"].to(dev),
        accuracy=accuracy,
        pml_width=c["pml"],
        pml_freq=25.0,
        nt=c["nt"],
    )


def test_gradcheck():
    """Numerical gradient check for (lamb, mu, buoyancy) at every accuracy."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype,
        ny=24,
        nx=24,
        nt=10,
        pml=4,
        device="cuda:0",
        seed=1,
        rec_offsets=(1, 1, 1),
    )
    dev = c["device"]

    for accuracy in (2, 4, 6, 8):
        def fn(la, m, b, accuracy=accuracy):
            return _run_nami(c, la, m, b, c["amp"].to(dev, dtype), accuracy)

        ok = torch.autograd.gradcheck(
            fn,
            (
                c["lamb"].to(dev, dtype).requires_grad_(True),
                c["mu"].to(dev, dtype).requires_grad_(True),
                c["buoy"].to(dev, dtype).requires_grad_(True),
            ),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
            fast_mode=True,
            nondet_tol=1e-8,
            raise_exception=False,
        )
        print(f"accuracy={accuracy} gradcheck (lamb, mu, buoyancy):", ok)
        assert ok


def test_fwi_smoke():
    """20 Adam iterations on a small random elastic problem: loss must drop."""
    dtype = torch.float64
    dev = "cuda:0"
    torch.manual_seed(3)
    g = torch.Generator(device="cpu")
    g.manual_seed(7)
    ny, nx, nt, pml = 28, 30, 24, 5
    dx, dt = 1e-3, 1e-7

    vp_true = 1800 + 300 * torch.rand(ny, nx, generator=g)
    vs_true = 1000 + 200 * torch.rand(ny, nx, generator=g)
    rho_true = 2000 + 200 * torch.rand(ny, nx, generator=g)
    lamb_true = rho_true * (vp_true**2 - 2 * vs_true**2)
    mu_true = rho_true * vs_true**2
    buoy_true = 1.0 / rho_true

    g2 = torch.Generator(device="cpu")
    g2.manual_seed(11)
    vp_init = 1900 + 100 * torch.rand(ny, nx, generator=g2)
    vs_init = 1050 + 30 * torch.rand(ny, nx, generator=g2)
    rho_init = 2100 + 100 * torch.rand(ny, nx, generator=g2)

    srcs = torch.tensor([[[ny // 2 - 3, nx // 3], [ny // 2 + 3, 2 * nx // 3]]])
    recs = torch.tensor([[[ny // 2, k] for k in range(4, nx - 4, 2)]])
    amp = torch.zeros(1, srcs.shape[1], nt, dtype=dtype)
    amp[:, :, 0] = 1.0
    amp[:, :, 1] = -0.5
    amp[:, :, 2] = 0.2

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
        c,
        lamb_true.to(dev, dtype),
        mu_true.to(dev, dtype),
        buoy_true.to(dev, dtype),
        amp.to(dev, dtype),
    ).detach()

    # Organise the inversion around vp/vs/rho: convert the true
    # lamb/mu/buoyancy back to vp/vs/rho to check the CFL margin.
    from nami.elastic.elastic2d import lambmubuoyancy_to_vpvsrho

    vp0, vs0, _ = lambmubuoyancy_to_vpvsrho(lamb_true, mu_true, buoy_true)
    vmax = max(vp0.abs().max().item(), vs0.abs().max().item())
    max_dt = 0.6 * dx / (math.sqrt(2) * vmax)
    print(f"FWI CFL: v_max={vmax:.1f} m/s, max_dt={max_dt:.3e}, dt={dt:.3e}")
    assert dt <= max_dt, f"FWI dt {dt} exceeds CFL max_dt {max_dt}"

    # Optimise in km/s (vp, vs) and 10^3 kg/m^3 (rho) so Adam lr is well scaled.
    vp_opt = (vp_init / 1000.0).to(dev, dtype).requires_grad_(True)
    vs_opt = (vs_init / 1000.0).to(dev, dtype).requires_grad_(True)
    rho_opt = (rho_init / 1000.0).to(dev, dtype).requires_grad_(True)
    opt = torch.optim.Adam([vp_opt, vs_opt, rho_opt], lr=0.1)

    def forward(vp, vs, rho):
        # keep vp^2 - 2*vs^2 > 0 (Poisson stability) during optimisation
        vs = torch.clamp(vs, max=0.7 * vp)
        lamb = rho * (vp**2 - 2 * vs**2)
        mu = rho * vs**2
        buoy = 1.0 / rho
        return _run_nami(c, lamb, mu, buoy, amp.to(dev, dtype))

    losses = []
    for _ in range(20):
        opt.zero_grad()
        out = forward(vp_opt * 1000.0, vs_opt * 1000.0, rho_opt * 1000.0)
        loss = ((out - data) ** 2).mean()
        loss.backward()
        losses.append(loss.item())
        opt.step()

    print(
        f"FWI smoke: loss first={losses[0]:.6e} last={losses[-1]:.6e} "
        f"({losses[-1] / losses[0]:.3f}x)"
    )
    assert losses[-1] < losses[0], (
        f"FWI loss did not drop: {losses[0]} -> {losses[-1]}"
    )


def _run_nami_storage(c, lamb, mu, buoy, amp, storage="auto",
                      ckpt_steps=0, sample_steps=1):
    from nami.elastic.elastic2d import elastic2d

    dev = c["device"]
    return elastic2d(
        lamb,
        mu,
        buoy,
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
        sample_steps=sample_steps,
    )


def test_checkpoint_forward_and_gradient_parity():
    """checkpoint_every=N reproduces full-storage forward and gradients."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=24, nx=24, nt=30, pml=4, device="cuda:0",
                   seed=1)
    dev = c["device"]
    lamb = c["lamb"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype)
    buoy = c["buoy"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    def run(ckpt):
        return _run_nami_storage(c, lamb, mu, buoy, amp,
                                 ckpt_steps=ckpt)

    r_full = run(0).detach()
    g_full = torch.autograd.grad(run(0).square().sum(), lamb)[0]
    for ckpt in (2, 4, 7, 30, 31):
        r = run(ckpt).detach()
        g = torch.autograd.grad(run(ckpt).square().sum(), lamb)[0]
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
    c = build_case(dtype=dtype, ny=24, nx=24, nt=30, pml=4, device="cuda:0",
                   seed=2)
    dev = c["device"]
    lamb = c["lamb"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype)
    buoy = c["buoy"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    def run(ckpt, stride):
        return _run_nami_storage(
            c, lamb, mu, buoy, amp, ckpt_steps=ckpt,
            sample_steps=stride,
        )

    r_full = run(0, 2).detach()
    g_full = torch.autograd.grad(run(0, 2).square().sum(), lamb)[0]
    for ckpt in (2, 4, 7, 15):
        r = run(ckpt, 2).detach()
        g = torch.autograd.grad(run(ckpt, 2).square().sum(), lamb)[0]
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
    c = build_case(dtype=dtype, ny=24, nx=24, nt=25, pml=4, device="cuda:0",
                   seed=1)
    dev = c["device"]
    lamb = c["lamb"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype)
    buoy = c["buoy"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)
    out = _run_nami_storage(c, lamb, mu, buoy, amp)
    g_auto = torch.autograd.grad(out.square().sum(), lamb)[0]
    g_full = torch.autograd.grad(
        _run_nami_storage(c, lamb, mu, buoy, amp, ckpt_steps=0)
        .square().sum(),
        lamb,
    )[0]
    rel = (g_auto - g_full).abs().max().item() / (
        g_full.abs().max().item() + 1e-300
    )
    assert rel < 1e-30, f"auto grad rel err {rel}"


def test_storage_false_is_forward_only():
    """storage='none' runs the forward but backward raises RuntimeError."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=24, nx=24, nt=24, pml=4, device="cuda:0",
                   seed=1)
    dev = c["device"]
    lamb = c["lamb"].to(dev, dtype).requires_grad_(True)
    mu = c["mu"].to(dev, dtype)
    buoy = c["buoy"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)
    out = _run_nami_storage(c, lamb, mu, buoy, amp, storage="none")
    assert out.shape == (c["nt"], 1, 3)
    try:
        torch.autograd.grad(out.square().sum(), lamb)
    except RuntimeError:
        return
    raise AssertionError(
        "storage='none' backward should raise RuntimeError"
    )


def test_storage_sampling_interval_consistency():
    """stride>1 keeps the forward exactly on the sample_steps=1 grid and
    approximates the model-gradient integral by sampling every `stride`
    steps and scaling each contribution by `stride`."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=24, nx=24, nt=24, pml=4, device="cuda:0",
                   seed=1)
    dev = c["device"]
    lamb = c["lamb"].to(dev, dtype)
    mu = c["mu"].to(dev, dtype)
    buoy = c["buoy"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)

    def nami_grad(stride):
        lam = lamb.clone().requires_grad_(True)
        out = _run_nami_storage(c, lam, mu, buoy, amp, ckpt_steps=0,
                                sample_steps=stride)
        return torch.autograd.grad(out.square().sum(), lam)[0]

    r1 = _run_nami_storage(c, lamb, mu, buoy, amp, ckpt_steps=0,
                           sample_steps=1).detach()
    r2 = _run_nami_storage(c, lamb, mu, buoy, amp, ckpt_steps=0,
                           sample_steps=2).detach()
    assert (r1 - r2).abs().max().item() == 0.0

    # sample_steps=2 samples the model-gradient integral -> an approximation, but
    # it must stay close on a smooth random model.
    g1, g2 = nami_grad(1), nami_grad(2)
    rel = (g1 - g2).abs().max().item() / (g1.abs().max().item() + 1e-300)
    print(f"sample_steps=2 model-grad rel err: {rel:.3e}")
    assert rel < 0.2, f"sample_steps=2 grad rel err {rel}"
