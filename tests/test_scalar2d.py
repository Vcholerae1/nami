"""nami scalar2d: autograd gradcheck, FWI smoke, and storage-mode tests."""

import torch

from nami.scalar.scalar2d import scalar2d


def build_case(
    dtype=torch.float64,
    ny=36,
    nx=40,
    nt=18,
    pml=5,
    n_shots=1,
    device="cuda:0",
    seed=0,
):
    """Small two-layer random model; one source at centre, 3 receivers nearby.

    dx = 5 m, dt = 1.0 ms. dt must satisfy the CFL limit max_dt =
    dx / (2 * v_max) ~= 1.19 ms for v_max ~= 2100 m/s.
    """
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    v = 1500 + 300 * torch.rand(ny, nx, generator=g)
    v[ny // 2 :] += 300  # two layers
    dx, dt = 5.0, 1.0e-3
    srcs = torch.tensor([[[ny // 2, nx // 2]]])[:n_shots]
    recs = torch.tensor(
        [
            [
                [ny // 2, nx // 2 - 3],
                [ny // 2, nx // 2 + 3],
                [ny // 2 - 3, nx // 2],
            ]
        ]
    )[:n_shots]
    amp = torch.zeros(n_shots, srcs.shape[1], nt, dtype=dtype)
    amp[:, :, 0] = 1.0
    amp[:, :, 1] = -0.5
    amp[:, :, 2] = 0.2
    return {
        "v": v,
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


def _run_nami(c, v, amp, accuracy=2):
    dev = c["device"]
    return scalar2d(
        v,
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
    """Numerical gradient check for v and source_amplitudes."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=24, nt=10, pml=4, device="cuda:0", seed=1
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
    print("gradcheck nami scalar2d (v, amp):", ok)
    assert ok


def _ricker(freq, nt, dt, device, dtype):
    t = torch.arange(nt, device=device, dtype=dtype) * dt
    tau = 1.0 / freq
    arg = (torch.pi * freq * (t - tau)) ** 2
    return (1 - 2 * arg) * torch.exp(-arg)


def test_fwi_smoke():
    """20 Adam iterations on a small random problem: loss must drop."""
    dtype = torch.float64
    dev = "cuda:0"
    torch.manual_seed(3)
    g = torch.Generator(device="cpu")
    g.manual_seed(7)
    ny, nx, nt, pml = 28, 30, 24, 5
    dx, dt = 5.0, 8.5e-4  # CFL-safe for v_max ~= 2150 (max_dt ~= 9.87e-4)

    v_true = 1500 + 400 * torch.rand(ny, nx, generator=g)
    v_true[ny // 2 :] += 250
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(11)
    v_init = 1600 + 200 * torch.rand(ny, nx, generator=g2)

    srcs = torch.tensor([[[ny // 2 - 5, nx // 3], [ny // 2 + 5, 2 * nx // 3]]])
    recs = torch.tensor([[[ny // 2, k] for k in range(4, nx - 4, 2)]])
    amp = _ricker(20.0, nt, dt, dev, dtype)[None, None, :].expand(1, 2, nt).contiguous()

    c = {
        "dx": dx,
        "dt": dt,
        "nt": nt,
        "pml": pml,
        "srcs": srcs,
        "recs": recs,
        "device": dev,
    }
    data = _run_nami(c, v_true.to(dev, dtype), amp).detach()

    # optimise in km/s so Adam lr ~0.1 is well scaled
    v_opt = (v_init / 1000.0).to(dev, dtype).requires_grad_(True)
    opt = torch.optim.Adam([v_opt], lr=0.1)
    losses = []
    for _ in range(20):
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
    assert losses[-1] < losses[0], f"FWI loss did not drop: {losses[0]} -> {losses[-1]}"


def test_checkpoint_forward_and_gradient_parity():
    """ckpt_steps=N reproduces full-storage forward and gradients."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=24, nt=30, pml=4, device="cuda:0", seed=1
    )
    dev = c["device"]
    v = c["v"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype)

    def run(ckpt):
        return scalar2d(
            v, c["dx"], c["dt"],
            source_amplitudes=amp,
            source_locations=c["srcs"].to(dev),
            receiver_locations=c["recs"].to(dev),
            accuracy=2, pml_width=c["pml"], pml_freq=25.0, nt=c["nt"],
            storage="auto", ckpt_steps=ckpt,
        )

    r_full = run(0).detach()
    g_full = torch.autograd.grad(run(0).square().sum(), v)[0]
    for ckpt in (2, 5, 7, 13, 29, 30):
        r = run(ckpt).detach()
        g = torch.autograd.grad(run(ckpt).square().sum(), v)[0]
        assert (r - r_full).abs().max().item() == 0.0, f"ckpt_steps={ckpt} fwd"
        rel = (g - g_full).abs().max().item() / (
            g_full.abs().max().item() + 1e-300
        )
        assert rel == 0.0, f"ckpt_steps={ckpt} grad rel err {rel}"


def test_checkpoint_auto_selection():
    """Default storage auto-enables snapshots; ckpt_steps=None auto-selects."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=24, nt=25, pml=4, device="cuda:0", seed=1
    )
    dev = c["device"]
    v = c["v"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype)
    out = scalar2d(
        v, c["dx"], c["dt"],
        source_amplitudes=amp,
        source_locations=c["srcs"].to(dev),
        receiver_locations=c["recs"].to(dev),
        accuracy=2, pml_width=c["pml"], pml_freq=25.0, nt=c["nt"],
    )
    # default (auto) must be backward-compatible with explicit full storage
    g_auto = torch.autograd.grad(out.square().sum(), v)[0]
    g_full = torch.autograd.grad(
        scalar2d(
            v, c["dx"], c["dt"],
            source_amplitudes=amp,
            source_locations=c["srcs"].to(dev),
            receiver_locations=c["recs"].to(dev),
            accuracy=2, pml_width=c["pml"], pml_freq=25.0, nt=c["nt"],
            storage="auto", ckpt_steps=0,
        ).square().sum(),
        v,
    )[0]
    rel = (g_auto - g_full).abs().max().item() / (
        g_full.abs().max().item() + 1e-300
    )
    assert rel == 0.0, f"auto grad rel err {rel}"


def test_stride_forward_traces_exact():
    """stride decimates snapshot storage but receiver traces stay exact."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=24, nt=12, pml=4, device="cuda:0", seed=1
    )
    dev = c["device"]
    v = c["v"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype)

    def run(gs):
        return scalar2d(
            v, c["dx"], c["dt"],
            source_amplitudes=amp,
            source_locations=c["srcs"].to(dev),
            receiver_locations=c["recs"].to(dev),
            accuracy=2, pml_width=c["pml"], pml_freq=25.0, nt=c["nt"],
            sample_steps=gs,
        )

    r_ref = run(1).detach()
    for gs in (2, 3):
        r = run(gs).detach()
        assert (r - r_ref).abs().max().item() == 0.0, f"sample_steps={gs} fwd"


def test_storage_false_is_forward_only():
    """storage='none' runs the forward but backward raises RuntimeError."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=24, nt=10, pml=4, device="cuda:0", seed=1
    )
    dev = c["device"]
    v = c["v"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype)
    out = scalar2d(
        v, c["dx"], c["dt"],
        source_amplitudes=amp,
        source_locations=c["srcs"].to(dev),
        receiver_locations=c["recs"].to(dev),
        accuracy=2, pml_width=c["pml"], pml_freq=25.0, nt=c["nt"],
        storage="none",
    )
    assert out.shape == (c["nt"], 1, 3)
    try:
        torch.autograd.grad(out.square().sum(), v)
    except RuntimeError:
        return
    raise AssertionError("storage='none' backward should raise RuntimeError")
