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
    """Numerical gradient check for v and source amplitudes at every order."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=24, nt=10, pml=4, device="cuda:0", seed=1
    )
    dev = c["device"]

    for accuracy in (2, 4, 6, 8):
        def fn(v, amp, accuracy=accuracy):
            return _run_nami(c, v, amp, accuracy=accuracy)

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
        assert ok, f"scalar2d gradcheck failed at accuracy={accuracy}"


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
        # bit-exact: deterministic cat/expand replicate pad in survey.py
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


def _multi_shot_survey(ny, nx, nt, dtype):
    """Two shots with distinct source/receiver locations and amplitudes."""
    srcs = torch.tensor([[[ny // 2, nx // 2 - 4]], [[ny // 2 - 3, nx // 2 + 4]]])
    recs = torch.tensor(
        [
            [[ny // 2, nx // 2 + 3], [ny // 2 + 3, nx // 2]],
            [[ny // 2 - 4, nx // 2], [ny // 2, nx // 2 - 5]],
        ]
    )
    amp = torch.zeros(2, 1, nt, dtype=dtype)
    amp[0, 0, :3] = torch.tensor([1.0, -0.5, 0.2], dtype=dtype)
    amp[1, 0, :3] = torch.tensor([0.7, 0.3, -0.1], dtype=dtype)
    return srcs, recs, amp


def test_multi_shot_shared_model():
    """n_shots=2 with a shared [ny, nx] model: the forward matches the two
    single-shot runs and the shared-model gradient is the sum of the two
    single-shot gradients."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=28, nt=12, pml=4, device="cuda:0", seed=1
    )
    dev = c["device"]
    ny, nx = c["v"].shape
    nt = c["nt"]
    srcs, recs, amp = _multi_shot_survey(ny, nx, nt, dtype)

    def run(v, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return scalar2d(
            v, c["dx"], c["dt"],
            source_amplitudes=amp[sl].to(dev),
            source_locations=srcs[sl].to(dev),
            receiver_locations=recs[sl].to(dev),
            accuracy=2, pml_width=c["pml"], pml_freq=25.0, nt=nt,
        )

    v = c["v"].to(dev, dtype).requires_grad_(True)
    v0 = c["v"].to(dev, dtype).requires_grad_(True)
    v1 = c["v"].to(dev, dtype).requires_grad_(True)
    r = run(v)
    r0, r1 = run(v0, 0), run(v1, 1)
    assert r.shape == (nt, 2, 2)
    assert (r[:, 0].detach() - r0[:, 0].detach()).abs().max().item() == 0.0, (
        "shot 0 fwd"
    )
    assert (r[:, 1].detach() - r1[:, 0].detach()).abs().max().item() == 0.0, (
        "shot 1 fwd"
    )
    g = torch.autograd.grad(r.square().sum(), v)[0]
    g0 = torch.autograd.grad(r0.square().sum(), v0)[0]
    g1 = torch.autograd.grad(r1.square().sum(), v1)[0]
    rel = (g - (g0 + g1)).abs().max().item() / (
        (g0 + g1).abs().max().item() + 1e-300
    )
    print(f"multi-shot shared-model grad rel err {rel:.3e}")
    assert rel < 1e-12, f"shared-model grad rel err {rel}"


def test_multi_shot_batched_model():
    """n_shots=2 with a batched [2, ny, nx] model: shot i uses model slice i
    and slice i of the batched gradient matches the single-shot gradient."""
    dtype = torch.float64
    c0 = build_case(
        dtype=dtype, ny=24, nx=28, nt=12, pml=4, device="cuda:0", seed=1
    )
    c1 = build_case(
        dtype=dtype, ny=24, nx=28, nt=12, pml=4, device="cuda:0", seed=2
    )
    dev = c0["device"]
    ny, nx = c0["v"].shape
    nt = c0["nt"]
    srcs, recs, amp = _multi_shot_survey(ny, nx, nt, dtype)

    def run(v, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return scalar2d(
            v, c0["dx"], c0["dt"],
            source_amplitudes=amp[sl].to(dev),
            source_locations=srcs[sl].to(dev),
            receiver_locations=recs[sl].to(dev),
            accuracy=2, pml_width=c0["pml"], pml_freq=25.0, nt=nt,
        )

    v_batch = torch.stack([c0["v"], c1["v"]]).to(dev, dtype).requires_grad_(True)
    v0 = c0["v"].to(dev, dtype).requires_grad_(True)
    v1 = c1["v"].to(dev, dtype).requires_grad_(True)
    r = run(v_batch)
    r0, r1 = run(v0, 0), run(v1, 1)
    assert r.shape == (nt, 2, 2)
    assert (r[:, 0].detach() - r0[:, 0].detach()).abs().max().item() == 0.0, (
        "shot 0 fwd"
    )
    assert (r[:, 1].detach() - r1[:, 0].detach()).abs().max().item() == 0.0, (
        "shot 1 fwd"
    )
    g = torch.autograd.grad(r.square().sum(), v_batch)[0]
    g0 = torch.autograd.grad(r0.square().sum(), v0)[0]
    g1 = torch.autograd.grad(r1.square().sum(), v1)[0]
    rel0 = (g[0] - g0).abs().max().item() / (g0.abs().max().item() + 1e-300)
    rel1 = (g[1] - g1).abs().max().item() / (g1.abs().max().item() + 1e-300)
    print(f"multi-shot batched-model grad rel err ({rel0:.3e}, {rel1:.3e})")
    assert rel0 < 1e-12 and rel1 < 1e-12, (
        f"batched-model grad rel err ({rel0}, {rel1})"
    )
