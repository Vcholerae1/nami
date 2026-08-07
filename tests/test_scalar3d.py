"""nami scalar3d: autograd gradcheck, FWI smoke, and checkpointing tests."""

import torch

from nami.scalar.scalar3d import scalar3d


def build_case(
    dtype=torch.float64,
    nz=12,
    ny=16,
    nx=18,
    nt=16,
    pml=4,
    n_shots=1,
    device="cuda:0",
    seed=0,
    near_pml=True,
):
    """Small three-layer random model; sources near the PML by default.

    grid_spacing = 5 m, dt = 0.5 ms.  dt must satisfy the 3D CFL limit
    max_dt = 0.6 * v_max / (v_max^2 * sqrt(sum(1/dx_i^2))) ~= 8.2e-4 s for
    v_max ~= 2100 m/s.
    """
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    v = 1500 + 300 * torch.rand(nz, ny, nx, generator=g)
    v[nz // 2 :] += 300  # two layers in z
    grid_spacing, dt = 5.0, 5.0e-4
    if near_pml:
        # close to the z=0 / y=0 / x=0 PML so the CPML branch is exercised
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
    else:
        srcs = torch.tensor([[[nz // 2, ny // 2, nx // 2]]])[:n_shots]
        recs = torch.tensor(
            [
                [
                    [nz // 2, ny // 2, nx // 2 - 3],
                    [nz // 2, ny // 2 - 3, nx // 2],
                    [nz // 2 - 3, ny // 2, nx // 2],
                ]
            ]
        )[:n_shots]
    amp = torch.zeros(n_shots, srcs.shape[1], nt, dtype=dtype)
    amp[:, :, 0] = 1.0
    amp[:, :, 1] = -0.5
    amp[:, :, 2] = 0.2
    return {
        "v": v,
        "grid_spacing": grid_spacing,
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


def _run_nami(c, v, amp, accuracy=2, **kw):
    dev = c["device"]
    return scalar3d(
        v,
        c["grid_spacing"],
        c["dt"],
        source_amplitudes=amp,
        source_locations=c["srcs"].to(dev),
        receiver_locations=c["recs"].to(dev),
        accuracy=accuracy,
        pml_width=c["pml"],
        pml_freq=25.0,
        nt=c["nt"],
        **kw,
    )
def test_gradcheck():
    """Numerical gradient check for v and source_amplitudes."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, nz=10, ny=12, nx=14, nt=10, pml=4, device="cuda:0",
        seed=1,
    )
    dev = c["device"]

    def fn(v, amp):
        return _run_nami(c, v, amp)

    ok = torch.autograd.gradcheck(
        fn,
        (
            c["v"].to(dev, dtype).requires_grad_(True),
            c["amp"].to(dev, dtype).requires_grad_(True),
        ),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        fast_mode=True,
        nondet_tol=1e-8,
        raise_exception=False,
    )
    print("gradcheck nami scalar3d (v, amp):", ok)
    assert ok


def _ricker(freq, nt, dt, device, dtype):
    t = torch.arange(nt, device=device, dtype=dtype) * dt
    tau = 1.0 / freq
    arg = (torch.pi * freq * (t - tau)) ** 2
    return (1 - 2 * arg) * torch.exp(-arg)


def test_fwi_smoke():
    """Tiny FWI problem: loss must drop."""
    dtype = torch.float64
    dev = "cuda:0"
    torch.manual_seed(3)
    g = torch.Generator(device="cpu")
    g.manual_seed(7)
    # nt=24 is long enough for the wave to reach the receivers (so the loss is
    # non-trivial) but short enough that the wavefront stays well clear of the
    # PML.  dt=5e-4 is CFL-safe for v_max ~= 2150.
    nz, ny, nx, nt, pml = 10, 12, 14, 24, 4
    grid_spacing, dt = 5.0, 5.0e-4

    v_true = 1500 + 400 * torch.rand(nz, ny, nx, generator=g)
    v_true[nz // 2 :] += 250
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(11)
    v_init = 1600 + 200 * torch.rand(nz, ny, nx, generator=g2)

    srcs = torch.tensor(
        [[[nz // 2 - 3, ny // 3, nx // 3], [nz // 2 + 3, 2 * ny // 3, 2 * nx // 3]]]
    )
    recs = torch.tensor(
        [
            [
                [nz // 2, y, x]
                for y in range(2, ny - 2, 2)
                for x in range(2, nx - 2, 2)
            ]
        ]
    )
    amp = _ricker(40.0, nt, dt, dev, dtype)[None, None, :].expand(1, 2, nt).contiguous()

    c = {
        "grid_spacing": grid_spacing,
        "dt": dt,
        "nt": nt,
        "pml": pml,
        "srcs": srcs,
        "recs": recs,
        "device": dev,
    }
    data = _run_nami(c, v_true.to(dev, dtype), amp).detach()

    # optimise in km/s so Adam lr is well scaled
    v_opt = (v_init / 1000.0).to(dev, dtype).requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=0.1)
    losses = []
    for _ in range(6):
        opt.zero_grad()
        out = _run_nami(c, v_opt * 1000.0, amp)
        loss = ((out - data) ** 2).mean()
        loss.backward()
        losses.append(loss.item())
        opt.step()

    print(
        f"FWI smoke: loss first={losses[0]:.6e} last={losses[-1]:.6e} "
        f"({losses[-1] / losses[0]:.3f}x)"
    )
    # the loss must drop by a real factor, not just roundoff
    assert losses[-1] < 0.5 * losses[0], (
        f"FWI loss did not drop: {losses[0]} -> {losses[-1]}"
    )


def test_checkpoint_forward_and_gradient_parity():
    """ckpt_steps=N reproduces full-storage forward and gradients."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, nz=10, ny=12, nx=14, nt=30, pml=4, device="cuda:0",
        seed=1,
    )
    dev = c["device"]
    v = c["v"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype)

    def run(ckpt):
        out = scalar3d(
            v, c["grid_spacing"], c["dt"],
            source_amplitudes=amp,
            source_locations=c["srcs"].to(dev),
            receiver_locations=c["recs"].to(dev),
            accuracy=2, pml_width=c["pml"], pml_freq=25.0, nt=c["nt"],
            storage="auto", ckpt_steps=ckpt,
        )
        # padded model saved by the autograd Function (first saved tensor);
        # its gradient is the adjoint model gradient without the (racy)
        # replicate-padding backward, so it is bit-exact under checkpointing.
        v_p = out.grad_fn.saved_tensors[0]
        return out, v_p

    out_full, v_p_full = run(0)
    r_full = out_full.detach()
    g_full = torch.autograd.grad(
        out_full.square().sum(), v_p_full, retain_graph=True
    )[0]
    g_v_full = torch.autograd.grad(out_full.square().sum(), v)[0]
    for ckpt in (2, 4, 7, 30, 31):
        out, v_p = run(ckpt)
        r = out.detach()
        g = torch.autograd.grad(out.square().sum(), v_p, retain_graph=True)[0]
        g_v = torch.autograd.grad(out.square().sum(), v)[0]
        assert (r - r_full).abs().max().item() == 0.0, f"ckpt_steps={ckpt} fwd"
        rel = (g - g_full).abs().max().item() / (
            g_full.abs().max().item() + 1e-300
        )
        assert rel == 0.0, f"ckpt_steps={ckpt} grad rel err {rel}"
        # The gradient w.r.t. the unpadded v flows through
        # replication_pad3d_backward, whose CUDA kernel has no deterministic
        # implementation (boundary cells accumulate with atomics), so it
        # carries ~1-ulp run-to-run noise even for full storage.  A real
        # checkpointing error would be orders of magnitude larger.
        rel_v = (g_v - g_v_full).abs().max().item() / (
            g_v_full.abs().max().item() + 1e-300
        )
        assert rel_v <= 1e-12, f"ckpt_steps={ckpt} grad_v rel err {rel_v}"


def test_stride_forward_traces_exact():
    """stride decimates snapshot storage but receiver traces stay exact."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, nz=10, ny=12, nx=14, nt=12, pml=4, device="cuda:0",
        seed=1,
    )
    dev = c["device"]
    v = c["v"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype)

    r_ref = _run_nami(c, v, amp, sample_steps=1).detach()
    for gs in (2, 3):
        r = _run_nami(c, v, amp, sample_steps=gs).detach()
        assert (r - r_ref).abs().max().item() == 0.0, f"sample_steps={gs} fwd"


def test_storage_false_is_forward_only():
    """storage='none' runs the forward but backward raises RuntimeError."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, nz=10, ny=12, nx=14, nt=10, pml=4, device="cuda:0", seed=1
    )
    dev = c["device"]
    v = c["v"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype)
    out = _run_nami(c, v, amp, storage="none")
    assert out.shape == (c["nt"], 1, c["recs"].shape[1])
    try:
        torch.autograd.grad(out.square().sum(), v)
    except RuntimeError:
        return
    raise AssertionError("storage='none' backward should raise RuntimeError")
