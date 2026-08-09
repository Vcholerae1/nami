"""nami class-based FWI API (``Scalar`` / ``Elastic`` / ``TM2D``).

Each model class is compared against the functional primitive it wraps:
forward outputs and ``backward`` gradients must match exactly, and a tiny
FWI loop must reduce the data-misfit loss.
"""

import pytest
import torch


def test_normalized_scalar_public_signatures():
    """Scalar APIs use grid_spacing and derive Born max velocity internally."""
    from inspect import signature

    from nami import Scalar
    from nami.scalar.scalar2d import scalar2d
    from nami.scalar.scalar2d_born import scalar2d_born
    from nami.scalar.scalar3d_born import scalar3d_born

    assert "grid_spacing" in signature(scalar2d).parameters
    assert "grid_spacing" in signature(Scalar).parameters
    assert "dx" not in signature(scalar2d).parameters
    assert "dx" not in signature(Scalar).parameters
    assert "max_vel" not in signature(scalar2d_born).parameters
    assert "max_vel" not in signature(scalar3d_born).parameters


def _scalar_case(device="cuda:0"):
    from tests.test_scalar2d import build_case

    return build_case(
        dtype=torch.float64, ny=24, nx=24, nt=10, pml=4,
        device=device, seed=1,
    )


def _elastic_case(device="cuda:0"):
    from tests.test_elastic2d import build_case

    return build_case(
        dtype=torch.float64, ny=24, nx=24, nt=20, pml=4,
        device=device, seed=1,
    )


def _em_case(device="cuda:0"):
    from tests.test_em2d_tm import build_case

    return build_case(
        dtype=torch.float64, ny=24, nx=24, nt=10, pml=4,
        device=device, seed=1,
    )


def test_scalar3d_class_matches_functional():
    from nami.models import Scalar3D
    from nami.scalar.scalar3d import scalar3d

    c = _scalar3d_case()
    dev = c["device"]
    v = c["v"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    model = Scalar3D(v.clone(), c["grid_spacing"], c["dt"], pml_width=c["pml"])
    rec_cls = model.forward(amp, srcs, recs)
    assert rec_cls.shape == (c["nt"], 1, recs.shape[1])

    rec_fn = scalar3d(
        v.clone(), c["grid_spacing"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
    )
    assert torch.equal(rec_cls, rec_fn)

    g_cls = model.backward(rec_cls.square().sum())
    v_ref = v.clone().requires_grad_(True)
    rec_ref = scalar3d(
        v_ref, c["grid_spacing"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
    )
    (g_ref,) = torch.autograd.grad(rec_ref.square().sum(), v_ref)
    assert torch.equal(g_cls, g_ref)


def test_em3d_class_matches_functional():
    from nami.em.em3d import em3d
    from nami.models import EM3D

    c = _em3d_case()
    dev = c["device"]
    eps = c["eps"].to(dev)
    sig = c["sig"].to(dev)
    mu = c["mu"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    model = EM3D(eps.clone(), sig.clone(), mu.clone(), c["dx"], c["dt"],
                 pml_width=c["pml"])
    rec_cls = model.forward(amp, srcs, recs)

    rec_fn = em3d(
        eps.clone(), sig.clone(), mu.clone(), c["dx"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
    )
    assert torch.equal(rec_cls, rec_fn)

    g_cls = model.backward(rec_cls.square().sum())
    assert isinstance(g_cls, tuple) and len(g_cls) == 3

    eps_r = eps.clone().requires_grad_(True)
    sig_r = sig.clone().requires_grad_(True)
    mu_r = mu.clone().requires_grad_(True)
    rec_ref = em3d(
        eps_r, sig_r, mu_r, c["dx"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
    )
    g_ref = torch.autograd.grad(rec_ref.square().sum(), (eps_r, sig_r, mu_r))
    for gc, gr in zip(g_cls, g_ref, strict=True):
        assert torch.equal(gc, gr)


def _scalar3d_case(device="cuda:0"):
    from tests.test_scalar3d import build_case as b3

    return b3(dtype=torch.float64, nz=8, ny=10, nx=10, nt=6, pml=3,
              device=device, seed=1)


def _em3d_case(device="cuda:0"):
    from tests.test_em3d import build_case as b3

    return b3(dtype=torch.float64, nz=8, ny=8, nx=8, nt=6, pml=2,
              device=device, seed=1)


def test_scalar_class_matches_functional():
    from nami.models import Scalar
    from nami.scalar.scalar2d import scalar2d

    c = _scalar_case()
    dev = c["device"]
    v = c["v"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    model = Scalar(v.clone(), c["dx"], c["dt"], pml_width=c["pml"])
    rec_cls = model.forward(amp, srcs, recs)
    assert rec_cls.shape == (c["nt"], 1, recs.shape[1])

    rec_fn = scalar2d(
        v.clone(), c["dx"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
    )
    assert torch.equal(rec_cls, rec_fn)
    assert model.receiver_amplitudes is rec_cls
    assert model.v.requires_grad
    assert list(model.parameters()) == [model.v]

    g_cls = model.backward(rec_cls.square().sum())
    assert model.grad is g_cls

    v_ref = v.clone().requires_grad_(True)
    rec_ref = scalar2d(
        v_ref, c["dx"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
    )
    (g_ref,) = torch.autograd.grad(rec_ref.square().sum(), v_ref)
    assert torch.equal(g_cls, g_ref)


def test_scalar_fwi_converges():
    from nami.models import Scalar

    c = _scalar_case(device="cuda:0")
    # the perturbed FWI models push v to ~2300, above the CFL limit for
    # build_case's dt=1e-3 (max_dt ~= 9.2e-4); use a CFL-safe dt.
    c["dt"] = 8.5e-4
    dev = c["device"]
    torch.manual_seed(5)
    v_true = (c["v"] + 200 * torch.rand_like(c["v"])).to(dev)
    v_init = (c["v"] - 200 * torch.rand_like(c["v"])).to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    model = Scalar(v_init.clone(), c["dx"], c["dt"], pml_width=c["pml"])
    data = scalar_data_from_true(v_true, c, amp, srcs, recs)

    losses = []
    for _ in range(12):
        rec = model.forward(amp, srcs, recs)
        loss = (rec - data).square().mean()
        losses.append(loss.item())
        g = model.backward(loss)
        with torch.no_grad():
            model.v -= 0.5 * g / (g.norm() + 1e-12)
    print("scalar FWI loss:", [f"{x:.3e}" for x in losses])
    assert losses[-1] < losses[0]


def scalar_data_from_true(v_true, c, amp, srcs, recs):
    from nami.scalar.scalar2d import scalar2d

    return scalar2d(
        v_true, c["dx"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
    ).detach()


def test_elastic_class_matches_functional():
    from nami.elastic.elastic2d import elastic2d
    from nami.models import Elastic

    c = _elastic_case()
    dev = c["device"]
    lamb = c["lamb"].to(dev)
    mu = c["mu"].to(dev)
    buoy = c["buoy"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    model = Elastic(
        lamb.clone(), mu.clone(), buoy.clone(), c["dx"], c["dt"],
        pml_width=c["pml"],
    )
    rec_cls = model.forward(amp, srcs, recs)

    rec_fn = elastic2d(
        lamb.clone(), mu.clone(), buoy.clone(), c["dx"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
    )
    assert torch.equal(rec_cls, rec_fn)
    assert [p.requires_grad for p in model.parameters()] == [True, True, True]

    g_cls = model.backward(rec_cls.square().sum())
    assert isinstance(g_cls, tuple) and len(g_cls) == 3

    lamb_r = lamb.clone().requires_grad_(True)
    mu_r = mu.clone().requires_grad_(True)
    buoy_r = buoy.clone().requires_grad_(True)
    rec_ref = elastic2d(
        lamb_r, mu_r, buoy_r, c["dx"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
    )
    g_ref = torch.autograd.grad(rec_ref.square().sum(), (lamb_r, mu_r, buoy_r))
    for gc, gr in zip(g_cls, g_ref, strict=True):
        assert torch.equal(gc, gr)


def test_elastic_fwi_converges():
    from nami.models import Elastic

    c = _elastic_case(device="cuda:0")
    dev = c["device"]
    torch.manual_seed(6)
    # build_case models come from torch.rand (float32); FWI needs float64 so
    # the O(1e-28) lamb gradients survive the norm (float32 underflows).
    lamb64 = c["lamb"].double()
    lamb_true = (lamb64 * (1 + 0.15 * torch.rand_like(lamb64))).to(dev)
    lamb_init = (lamb64 * (1 - 0.15 * torch.rand_like(lamb64))).to(dev)
    mu = c["mu"].double().to(dev)
    buoy = c["buoy"].double().to(dev)
    amp = c["amp"].double().to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    from nami.elastic.elastic2d import elastic2d

    data = elastic2d(
        lamb_true, mu, buoy, c["dx"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
    ).detach()

    model = Elastic(
        lamb_init.clone(), mu.clone(), buoy.clone(), c["dx"], c["dt"],
        pml_width=c["pml"],
    )
    losses = []
    for _ in range(12):
        rec = model.forward(amp, srcs, recs)
        loss = (rec - data).square().mean()
        losses.append(loss.item())
        (gl, gm, gb) = model.backward(loss)
        with torch.no_grad():
            # lamb is O(1e9) in SI units; scale the step to the model so
            # a unit gradient move is a small *relative* change.
            step = gl / gl.norm()
            model.lamb -= 1e-3 * model.lamb.abs().mean() * step
    print("elastic FWI loss:", [f"{x:.3e}" for x in losses])
    assert losses[-1] < losses[0]


def test_em_class_matches_functional():
    from nami.em.em2d_tm import em2d_tm
    from nami.models import TM2D

    c = _em_case()
    dev = c["device"]
    eps = c["eps"].to(dev)
    sig = c["sig"].to(dev)
    mu = c["mu"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    model = TM2D(
        eps.clone(), sig.clone(), mu.clone(), c["dx"], c["dt"],
        pml_width=c["pml"],
    )
    rec_cls = model.forward(amp, srcs, recs)

    rec_fn = em2d_tm(
        eps.clone(), sig.clone(), mu.clone(), c["dx"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
    )
    assert torch.equal(rec_cls, rec_fn)

    g_cls = model.backward(rec_cls.square().sum())
    assert isinstance(g_cls, tuple) and len(g_cls) == 3

    eps_r = eps.clone().requires_grad_(True)
    sig_r = sig.clone().requires_grad_(True)
    mu_r = mu.clone().requires_grad_(True)
    rec_ref = em2d_tm(
        eps_r, sig_r, mu_r, c["dx"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
    )
    g_ref = torch.autograd.grad(rec_ref.square().sum(), (eps_r, sig_r, mu_r))
    for gc, gr in zip(g_cls, g_ref, strict=True):
        assert torch.equal(gc, gr)


def test_em_fwi_converges():
    from nami.em.em2d_tm import em2d_tm
    from nami.models import TM2D

    c = _em_case(device="cuda:0")
    dev = c["device"]
    torch.manual_seed(8)
    eps_true = (c["eps"] * (1 + 0.1 * torch.rand_like(c["eps"]))).to(dev)
    eps_init = (c["eps"] * (1 - 0.1 * torch.rand_like(c["eps"]))).to(dev)
    sig = c["sig"].to(dev)
    mu = c["mu"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    data = em2d_tm(
        eps_true, sig, mu, c["dx"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
    ).detach()

    model = TM2D(
        eps_init.clone(), sig.clone(), mu.clone(), c["dx"], c["dt"],
        pml_width=c["pml"],
    )
    losses = []
    for _ in range(12):
        rec = model.forward(amp, srcs, recs)
        loss = (rec - data).square().mean()
        losses.append(loss.item())
        (ge, gs, gm) = model.backward(loss)
        with torch.no_grad():
            model.epsilon -= 1e-3 * ge / (ge.norm() + 1e-12)
    print("em FWI loss:", [f"{x:.3e}" for x in losses])
    assert losses[-1] < losses[0]


def test_backward_writes_param_grad_scalar():
    from nami.models import Scalar

    c = _scalar_case()
    dev = c["device"]
    v = c["v"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    model = Scalar(v.clone(), c["dx"], c["dt"], pml_width=c["pml"])
    assert model.v.grad is None
    rec = model.forward(amp, srcs, recs)
    g = model.backward(rec.square().sum())
    assert model.v.grad is not None
    assert torch.equal(model.v.grad, g)


def test_backward_writes_param_grad_elastic():
    from nami.models import Elastic

    c = _elastic_case()
    dev = c["device"]
    lamb = c["lamb"].to(dev)
    mu = c["mu"].to(dev)
    buoy = c["buoy"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    model = Elastic(
        lamb.clone(), mu.clone(), buoy.clone(), c["dx"], c["dt"],
        pml_width=c["pml"],
    )
    rec = model.forward(amp, srcs, recs)
    gl, gm, gb = model.backward(rec.square().sum())
    assert torch.equal(model.lamb.grad, gl)
    assert torch.equal(model.mu.grad, gm)
    assert torch.equal(model.buoyancy.grad, gb)


def test_backward_clears_stale_param_grad():
    from nami.models import Elastic

    c = _elastic_case()
    dev = c["device"]
    lamb = c["lamb"].to(dev)
    mu = c["mu"].to(dev)
    buoy = c["buoy"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    model = Elastic(
        lamb.clone(), mu.clone(), buoy.clone(), c["dx"], c["dt"],
        pml_width=c["pml"],
    )
    rec = model.forward(amp, srcs, recs)
    gl, gm, gb = model.backward(rec.square().sum())
    assert model.mu.grad is not None

    # Drop mu from the gradient set; the next backward must not leave the
    # previous gradient on model.mu.grad.
    model.mu.requires_grad_(False)
    rec = model.forward(amp, srcs, recs)
    gl2, gm2, gb2 = model.backward(rec.square().sum())
    assert gm2 is None
    assert model.mu.grad is None
    assert torch.equal(model.lamb.grad, gl2)
    assert torch.equal(model.buoyancy.grad, gb2)


def test_adam_step_updates_parameters():
    from nami.models import Scalar

    c = _scalar_case()
    dev = c["device"]
    v = c["v"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    model = Scalar(v.clone(), c["dx"], c["dt"], pml_width=c["pml"])
    opt = torch.optim.Adam(model.parameters())
    rec = model.forward(amp, srcs, recs)
    model.backward(rec.square().sum())
    v_before = model.v.detach().clone()
    opt.step()
    assert not torch.equal(model.v.detach(), v_before)


def test_storage_none_forward_ok_backward_raises():
    from nami.models import Scalar

    c = _scalar_case()
    dev = c["device"]
    v = c["v"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    model = Scalar(v.clone(), c["dx"], c["dt"], pml_width=c["pml"],
                   storage="none")
    rec = model.forward(amp, srcs, recs)
    assert rec.shape == (c["nt"], 1, recs.shape[1])
    with pytest.raises(RuntimeError):
        model.backward(rec.square().sum())


def test_ckpt_steps_matches_full_storage():
    from nami.models import Scalar

    c = _scalar_case()
    dev = c["device"]
    v = c["v"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    grads = {}
    for name, ckpt in (("full", 0), ("ckpt", 3)):
        model = Scalar(v.clone(), c["dx"], c["dt"], pml_width=c["pml"],
                       ckpt_steps=ckpt)
        rec = model.forward(amp, srcs, recs)
        grads[name] = model.backward(rec.square().sum())
    torch.testing.assert_close(grads["ckpt"], grads["full"])


def test_em3d_components():
    from nami.em.em3d import em3d
    from nami.models import EM3D

    c = _em3d_case()
    dev = c["device"]
    eps = c["eps"].to(dev)
    sig = c["sig"].to(dev)
    mu = c["mu"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    model = EM3D(eps.clone(), sig.clone(), mu.clone(), c["dx"], c["dt"],
                 pml_width=c["pml"], source_component="ex",
                 receiver_component="ez")
    rec_cls = model.forward(amp, srcs, recs)
    assert rec_cls.shape == (c["nt"], 1, recs.shape[1])

    rec_fn = em3d(
        eps.clone(), sig.clone(), mu.clone(), c["dx"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
        source_component="ex", receiver_component="ez",
    )
    assert torch.equal(rec_cls, rec_fn)


def test_class_forward_callback_and_return_state():
    """The class API forwards callback/state args to the propagator, caches
    the state on ``.last_state``, and ``backward`` still works."""
    from nami.models import Scalar
    from nami.scalar.scalar2d import scalar2d

    c = _scalar_case()
    dev = c["device"]
    v = c["v"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)

    steps = []

    def cb(state):
        steps.append(state.step)
        state.get_wavefield("u")

    model = Scalar(v.clone(), c["dx"], c["dt"], pml_width=c["pml"])
    rec_cls, state = model.forward(
        amp, srcs, recs,
        forward_callback=cb, callback_frequency=2, return_state=True,
    )
    assert steps == list(range(0, c["nt"], 2))
    assert model.last_state is state
    assert model.receiver_amplitudes is rec_cls
    assert isinstance(state, dict) and state

    rec_fn, state_fn = scalar2d(
        v.clone(), c["dx"], c["dt"],
        source_amplitudes=amp, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"], nt=c["nt"],
        return_state=True,
    )
    assert torch.equal(rec_cls, rec_fn)
    assert state.keys() == state_fn.keys()
    for key in state:
        assert torch.equal(state[key], state_fn[key])

    g = model.backward(rec_cls.square().sum())
    assert g is not None


def test_class_forward_continuation_matches_one_shot():
    """A class-API split run (return_state -> initial_state) bitwise matches
    the one-shot functional run."""
    from nami.models import Scalar
    from nami.scalar.scalar2d import scalar2d

    c = _scalar_case()
    dev = c["device"]
    v = c["v"].to(dev)
    amp = c["amp"].to(dev)
    srcs, recs = c["srcs"].to(dev), c["recs"].to(dev)
    amp2 = torch.cat((amp, amp), dim=-1)

    rec_full = scalar2d(
        v.clone(), c["dx"], c["dt"],
        source_amplitudes=amp2, source_locations=srcs,
        receiver_locations=recs, accuracy=2, pml_width=c["pml"],
    )

    model = Scalar(v.clone(), c["dx"], c["dt"], pml_width=c["pml"])
    rec1, state = model.forward(amp, srcs, recs, return_state=True)
    rec2 = model.forward(amp, srcs, recs, initial_state=state)
    assert model.last_state is None
    assert torch.equal(torch.cat((rec1, rec2), dim=0), rec_full)


def _class_smoke_case(name):
    """Return ``(model, amp, srcs, recs, nt)`` for each of the five wrappers."""
    if name == "scalar":
        from nami.models import Scalar

        c = _scalar_case()
        dev = c["device"]
        model = Scalar(c["v"].to(dev), c["dx"], c["dt"], pml_width=c["pml"])
    elif name == "scalar3d":
        from nami.models import Scalar3D

        c = _scalar3d_case()
        dev = c["device"]
        model = Scalar3D(
            c["v"].to(dev), c["grid_spacing"], c["dt"], pml_width=c["pml"]
        )
    elif name == "elastic":
        from nami.models import Elastic

        c = _elastic_case()
        dev = c["device"]
        model = Elastic(
            c["lamb"].to(dev), c["mu"].to(dev), c["buoy"].to(dev),
            c["dx"], c["dt"], pml_width=c["pml"],
        )
    elif name == "tm2d":
        from nami.models import TM2D

        c = _em_case()
        dev = c["device"]
        model = TM2D(
            c["eps"].to(dev), c["sig"].to(dev), c["mu"].to(dev),
            c["dx"], c["dt"], pml_width=c["pml"],
        )
    else:
        from nami.models import EM3D

        c = _em3d_case()
        dev = c["device"]
        model = EM3D(
            c["eps"].to(dev), c["sig"].to(dev), c["mu"].to(dev),
            c["dx"], c["dt"], pml_width=c["pml"],
        )
    return (
        model,
        c["amp"].to(dev),
        c["srcs"].to(dev),
        c["recs"].to(dev),
        c["nt"],
    )


@pytest.mark.parametrize(
    "name", ["scalar", "scalar3d", "elastic", "tm2d", "em3d"]
)
def test_all_classes_forward_callback_and_state_smoke(name):
    """Every wrapper forwards callback/state args identically: the callback
    fires every ``callback_frequency`` steps, ``return_state=True`` returns
    ``(rec, state)`` with the state cached on ``.last_state``, and the
    returned state can be passed back as ``initial_state``."""
    model, amp, srcs, recs, nt = _class_smoke_case(name)

    steps = []
    rec, state = model.forward(
        amp, srcs, recs,
        forward_callback=lambda s: steps.append(s.step),
        callback_frequency=2,
        return_state=True,
    )
    assert steps == list(range(0, nt, 2))
    assert model.last_state is state
    assert model.receiver_amplitudes is rec
    assert isinstance(state, dict) and state

    rec_plain = model.forward(amp, srcs, recs)
    assert model.last_state is None
    assert torch.equal(rec, rec_plain)

    rec_cont = model.forward(amp, srcs, recs, initial_state=state)
    assert rec_cont.shape == rec.shape
