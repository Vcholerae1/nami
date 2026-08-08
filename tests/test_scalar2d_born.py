"""nami scalar2d Born: Born linearity and autograd gradcheck."""

import pytest
import torch

from nami.scalar.scalar2d import scalar2d
from nami.scalar.scalar2d_born import scalar2d_born


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

    dx = 5 m, dt = 1.0 ms. dt satisfies the CFL limit max_dt =
    dx / (2 * v_max) ~= 1.47 ms for v_max ~= 1700 m/s.
    """
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    v = 1500 + 200 * torch.rand(ny, nx, generator=g)
    v[ny // 2 :] += 150  # two layers
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(seed + 2)
    scatter = 0.005 * (torch.rand(ny, nx, generator=g2) - 0.5)
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
    bg_recs = torch.tensor(
        [
            [
                [ny // 2 - 5, nx // 2 - 5],
                [ny // 2 + 5, nx // 2 + 5],
            ]
        ]
    )[:n_shots]
    amp = torch.zeros(n_shots, srcs.shape[1], nt, dtype=dtype)
    amp[:, :, 0] = 1.0
    amp[:, :, 1] = -0.5
    amp[:, :, 2] = 0.2
    return {
        "v": v,
        "scatter": scatter,
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


def _run_nami(c, v, scatter, amp, accuracy=2, **kw):
    dev = c["device"]
    return scalar2d_born(
        v,
        scatter,
        c["dx"],
        c["dt"],
        source_amplitudes=amp,
        source_locations=c["srcs"].to(dev),
        receiver_locations=c["recs"].to(dev),
        bg_receiver_locations=c["bg_recs"].to(dev),
        accuracy=accuracy,
        pml_width=c["pml"],
        pml_freq=25.0,
        nt=c["nt"],
        **kw,
    )
def test_born_linearity():
    """r_born(delta) ~= r_full(delta) - r_bg with O(delta^2) error."""
    dtype = torch.float64
    c = build_case(dtype=dtype, ny=24, nx=28, nt=12, pml=4,
                   device="cuda:0", seed=1)
    dev = c["device"]
    v = c["v"].to(dev, dtype)
    amp = c["amp"].to(dev, dtype)
    srcs = c["srcs"].to(dev)
    recs = c["recs"].to(dev)
    torch.manual_seed(5)
    g = torch.Generator(device="cpu")
    g.manual_seed(9)
    base = (torch.rand(c["v"].shape, generator=g, dtype=dtype) - 0.5).to(dev)

    def full(delta):
        return scalar2d(
            v + delta,
            c["dx"],
            c["dt"],
            source_amplitudes=amp,
            source_locations=srcs,
            receiver_locations=recs,
            accuracy=2,
            pml_width=c["pml"],
            pml_freq=25.0,
            nt=c["nt"],
        )

    def born(delta):
        return scalar2d_born(
            v,
            delta,
            c["dx"],
            c["dt"],
            source_amplitudes=amp,
            source_locations=srcs,
            receiver_locations=recs,
            accuracy=2,
            pml_width=c["pml"],
            pml_freq=25.0,
            nt=c["nt"],
        )

    r_bg = full(torch.zeros_like(base))
    signal = born(base).abs().max().item()
    resid_1 = (born(base) - (full(base) - r_bg)).abs().max().item()
    resid_2 = (born(2 * base) - (full(2 * base) - r_bg)).abs().max().item()
    print(f"signal(delta) {signal:.3e}, residual(delta) {resid_1:.3e}, "
          f"residual(2*delta) {resid_2:.3e}")
    assert resid_1 < 1e-2 * signal, f"Born linearity residual(delta): {resid_1}"
    # O(delta^2): doubling delta multiplies the residual by ~4
    ratio = resid_2 / (resid_1 + 1e-300)
    print(f"residual ratio 2x: {ratio:.3f}")
    assert 3.0 < ratio < 5.0, f"residual did not scale quadratically: {ratio}"


def test_gradcheck():
    """Numerical gradient check for v, scatter, and source_amplitudes."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=24, nt=10, pml=4, device="cuda:0", seed=1
    )
    dev = c["device"]

    def fn(v, scatter, amp):
        return _run_nami(c, v, scatter, amp)

    ok = torch.autograd.gradcheck(
        fn,
        (
            c["v"].to(dev, dtype).requires_grad_(True),
            c["scatter"].to(dev, dtype).requires_grad_(True),
            c["amp"].to(dev, dtype).requires_grad_(True),
        ),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        fast_mode=True,
        nondet_tol=1e-8,
        raise_exception=False,
    )
    print("gradcheck nami scalar2d_born (v, scatter, amp):", ok)
    assert ok


def test_checkpoint_forward_and_gradient_parity():
    """ckpt_steps=N reproduces full-storage forward and gradients."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=24, nt=30, pml=4, device="cuda:0", seed=1
    )
    dev = c["device"]
    v = c["v"].to(dev, dtype).requires_grad_(True)
    scatter = c["scatter"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype)

    def run(ckpt):
        return _run_nami(
            c, v, scatter, amp, storage="auto", ckpt_steps=ckpt,
        )

    r0, r0_bg = run(0)
    r_full, r_bg_full = r0.detach(), r0_bg.detach()
    g_full = torch.autograd.grad(r0.square().sum() + r0_bg.square().sum(), v)[0]
    for ckpt in (2, 4, 7, 29, 30):
        r, r_bg = run(ckpt)
        assert (r.detach() - r_full).abs().max().item() == 0.0, (
            f"ckpt_steps={ckpt} fwd r"
        )
        assert (r_bg.detach() - r_bg_full).abs().max().item() == 0.0, (
            f"ckpt_steps={ckpt} fwd r_bg"
        )
        g = torch.autograd.grad(r.square().sum() + r_bg.square().sum(), v)[0]
        rel = (g - g_full).abs().max().item() / (
            g_full.abs().max().item() + 1e-300
        )
        assert rel == 0.0, f"ckpt_steps={ckpt} grad rel err {rel}"


def test_storage_false_is_forward_only():
    """storage='none' runs the forward but backward raises RuntimeError."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=24, nt=10, pml=4, device="cuda:0", seed=1
    )
    dev = c["device"]
    v = c["v"].to(dev, dtype).requires_grad_(True)
    scatter = c["scatter"].to(dev, dtype).requires_grad_(True)
    amp = c["amp"].to(dev, dtype)
    out, out_bg = _run_nami(c, v, scatter, amp, storage="none")
    assert out.shape == (c["nt"], 1, c["recs"].shape[1])
    assert out_bg.shape == (c["nt"], 1, c["bg_recs"].shape[1])
    try:
        torch.autograd.grad(out.square().sum() + out_bg.square().sum(), v)
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
    bg_recs = torch.tensor(
        [
            [[ny // 2 - 5, nx // 2 - 5], [ny // 2 + 5, nx // 2 + 5]],
            [[ny // 2 + 4, nx // 2 - 6], [ny // 2 - 6, nx // 2 + 6]],
        ]
    )
    amp = torch.zeros(2, 1, nt, dtype=dtype)
    amp[0, 0, :3] = torch.tensor([1.0, -0.5, 0.2], dtype=dtype)
    amp[1, 0, :3] = torch.tensor([0.7, 0.3, -0.1], dtype=dtype)
    return srcs, recs, bg_recs, amp


def test_multi_shot_shared_model():
    """n_shots=2 with shared [ny, nx] v and scatter: the forward matches the
    two single-shot runs and the shared-model gradients are the sums of the
    two single-shot gradients."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=28, nt=12, pml=4, device="cuda:0", seed=1
    )
    dev = c["device"]
    ny, nx = c["v"].shape
    nt = c["nt"]
    srcs, recs, bg_recs, amp = _multi_shot_survey(ny, nx, nt, dtype)

    def run(v, scatter, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return scalar2d_born(
            v, scatter, c["dx"], c["dt"],
            source_amplitudes=amp[sl].to(dev),
            source_locations=srcs[sl].to(dev),
            receiver_locations=recs[sl].to(dev),
            bg_receiver_locations=bg_recs[sl].to(dev),
            accuracy=2, pml_width=c["pml"], pml_freq=25.0, nt=nt,
        )

    v = c["v"].to(dev, dtype).requires_grad_(True)
    scatter = c["scatter"].to(dev, dtype).requires_grad_(True)
    v0 = c["v"].to(dev, dtype).requires_grad_(True)
    sc0 = c["scatter"].to(dev, dtype).requires_grad_(True)
    v1 = c["v"].to(dev, dtype).requires_grad_(True)
    sc1 = c["scatter"].to(dev, dtype).requires_grad_(True)
    r, r_bg = run(v, scatter)
    r0, r0_bg = run(v0, sc0, 0)
    r1, r1_bg = run(v1, sc1, 1)
    assert r.shape == (nt, 2, 2) and r_bg.shape == (nt, 2, 2)
    for out, ref, name in (
        (r[:, 0], r0[:, 0], "shot 0 fwd r"),
        (r[:, 1], r1[:, 0], "shot 1 fwd r"),
        (r_bg[:, 0], r0_bg[:, 0], "shot 0 fwd r_bg"),
        (r_bg[:, 1], r1_bg[:, 0], "shot 1 fwd r_bg"),
    ):
        assert (out.detach() - ref.detach()).abs().max().item() == 0.0, name
    loss = r.square().sum() + r_bg.square().sum()
    loss0 = r0.square().sum() + r0_bg.square().sum()
    loss1 = r1.square().sum() + r1_bg.square().sum()
    g_v, g_sc = torch.autograd.grad(loss, [v, scatter])
    g0_v, g0_sc = torch.autograd.grad(loss0, [v0, sc0])
    g1_v, g1_sc = torch.autograd.grad(loss1, [v1, sc1])
    for g, g0, g1, name in (
        (g_v, g0_v, g1_v, "v"),
        (g_sc, g0_sc, g1_sc, "scatter"),
    ):
        rel = (g - (g0 + g1)).abs().max().item() / (
            (g0 + g1).abs().max().item() + 1e-300
        )
        print(f"multi-shot shared-model grad_{name} rel err {rel:.3e}")
        assert rel < 1e-12, f"shared-model grad_{name} rel err {rel}"


def test_multi_shot_batched_model():
    """n_shots=2 with batched [2, ny, nx] v and scatter: shot i uses slice i
    and slice i of the batched gradients matches the single-shot gradient."""
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
    srcs, recs, bg_recs, amp = _multi_shot_survey(ny, nx, nt, dtype)

    def run(v, scatter, shot=None):
        sl = slice(None) if shot is None else slice(shot, shot + 1)
        return scalar2d_born(
            v, scatter, c0["dx"], c0["dt"],
            source_amplitudes=amp[sl].to(dev),
            source_locations=srcs[sl].to(dev),
            receiver_locations=recs[sl].to(dev),
            bg_receiver_locations=bg_recs[sl].to(dev),
            accuracy=2, pml_width=c0["pml"], pml_freq=25.0, nt=nt,
        )

    v_batch = torch.stack([c0["v"], c1["v"]]).to(dev, dtype).requires_grad_(True)
    sc_batch = (
        torch.stack([c0["scatter"], c1["scatter"]]).to(dev, dtype).requires_grad_(True)
    )
    v0 = c0["v"].to(dev, dtype).requires_grad_(True)
    sc0 = c0["scatter"].to(dev, dtype).requires_grad_(True)
    v1 = c1["v"].to(dev, dtype).requires_grad_(True)
    sc1 = c1["scatter"].to(dev, dtype).requires_grad_(True)
    r, r_bg = run(v_batch, sc_batch)
    r0, r0_bg = run(v0, sc0, 0)
    r1, r1_bg = run(v1, sc1, 1)
    assert r.shape == (nt, 2, 2) and r_bg.shape == (nt, 2, 2)
    for out, ref, name in (
        (r[:, 0], r0[:, 0], "shot 0 fwd r"),
        (r[:, 1], r1[:, 0], "shot 1 fwd r"),
        (r_bg[:, 0], r0_bg[:, 0], "shot 0 fwd r_bg"),
        (r_bg[:, 1], r1_bg[:, 0], "shot 1 fwd r_bg"),
    ):
        assert (out.detach() - ref.detach()).abs().max().item() == 0.0, name
    loss = r.square().sum() + r_bg.square().sum()
    loss0 = r0.square().sum() + r0_bg.square().sum()
    loss1 = r1.square().sum() + r1_bg.square().sum()
    g_v, g_sc = torch.autograd.grad(loss, [v_batch, sc_batch])
    g0_v, g0_sc = torch.autograd.grad(loss0, [v0, sc0])
    g1_v, g1_sc = torch.autograd.grad(loss1, [v1, sc1])
    for gb, gs, name in (
        ((g_v[0], g_v[1]), (g0_v, g1_v), "v"),
        ((g_sc[0], g_sc[1]), (g0_sc, g1_sc), "scatter"),
    ):
        rels = [
            (gb[i] - gs[i]).abs().max().item() / (gs[i].abs().max().item() + 1e-300)
            for i in range(2)
        ]
        print(f"multi-shot batched-model grad_{name} rel err {rels}")
        assert all(rel < 1e-12 for rel in rels), f"batched grad_{name} rel err {rels}"


def test_cfl_limit_raises():
    """dt above the CFL limit raises NotImplementedError."""
    dtype = torch.float64
    c = build_case(
        dtype=dtype, ny=24, nx=24, nt=10, pml=4, device="cuda:0", seed=1
    )
    dev = c["device"]
    v = c["v"].to(dev, dtype)
    scatter = c["scatter"].to(dev, dtype)
    with pytest.raises(NotImplementedError):
        scalar2d_born(
            v,
            scatter,
            c["dx"],
            5.0e-3,  # way above the ~1.2 ms CFL limit
            source_amplitudes=c["amp"].to(dev, dtype),
            source_locations=c["srcs"].to(dev),
            receiver_locations=c["recs"].to(dev),
            accuracy=2,
            pml_width=c["pml"],
            pml_freq=25.0,
            nt=c["nt"],
        )
