"""Unified propagator callbacks, state validation, and continuation."""

import inspect

import pytest
import torch

from nami.elastic.elastic2d import elastic2d
from nami.elastic.elastic2d_born import elastic2d_born
from nami.em.em2d_tm import em2d_tm
from nami.em.em2d_tm_born import em2d_tm_born
from nami.em.em3d import em3d
from nami.em.em3d_born import em3d_born
from nami.scalar.scalar2d import scalar2d
from nami.scalar.scalar2d_born import scalar2d_born
from nami.scalar.scalar3d import scalar3d
from nami.scalar.scalar3d_born import scalar3d_born

PROPAGATORS = (
    scalar2d,
    scalar2d_born,
    scalar3d,
    scalar3d_born,
    elastic2d,
    elastic2d_born,
    em2d_tm,
    em2d_tm_born,
    em3d,
    em3d_born,
)

BORN_PROPAGATORS = (
    scalar2d_born,
    scalar3d_born,
    elastic2d_born,
    em2d_tm_born,
    em3d_born,
)


def build_case(dtype=torch.float64, ny=36, nx=40, nt=18, pml=5, n_shots=1, seed=0):
    """Small two-layer random model; one source at centre, 3 receivers nearby."""
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    v = 1500 + 300 * torch.rand(ny, nx, generator=g)
    v[ny // 2 :] += 300  # two layers
    dx, dt = 5.0, 1.0e-3
    srcs = torch.tensor([[[ny // 2, nx // 2]]])[:n_shots]
    recs = torch.tensor(
        [[[ny // 2, nx // 2 - 3], [ny // 2, nx // 2 + 4], [ny // 2 + 3, nx // 2]]]
    )[:n_shots]
    amp = torch.zeros(n_shots, 1, nt, dtype=dtype)
    amp[:, 0, 1] = 0.1
    return {
        "v": v.to(dtype=dtype),
        "dx": dx,
        "dt": dt,
        "nt": nt,
        "pml_width": pml,
        "source_amplitudes": amp,
        "source_locations": srcs.to(dtype=torch.long),
        "receiver_locations": recs.to(dtype=torch.long),
        "accuracy": 2,
        "pml_freq": 25.0,
        "device": "cuda:0",
        "dtype": dtype,
    }


def build_elastic_case(dtype=torch.float64, ny=24, nx=28, nt=10, pml=4, seed=2):
    """Small elastic model (vp/vs/rho with a mild random layer), pressure
    source at centre, 2 nearby pressure receivers."""
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    vp = 2000.0 + 100.0 * torch.rand(ny, nx, generator=g)
    vs = 1150.0 + 50.0 * torch.rand(ny, nx, generator=g)
    rho = 2000.0 + 100.0 * torch.rand(ny, nx, generator=g)
    lamb = rho * (vp**2 - 2 * vs**2)
    mu = rho * vs**2
    buoyancy = 1.0 / rho
    dx, dt = 5.0, 1.0e-3
    srcs = torch.tensor([[[ny // 2, nx // 2]]])
    recs = torch.tensor([[[ny // 2, nx // 2 - 3], [ny // 2 + 3, nx // 2]]])
    amp = torch.zeros(1, 1, nt, dtype=dtype)
    amp[:, 0, 1] = 1.0e8
    return {
        "lamb": lamb.to(dtype=dtype),
        "mu": mu.to(dtype=dtype),
        "buoyancy": buoyancy.to(dtype=dtype),
        "grid_spacing": (dx, dx),
        "dt": dt,
        "nt": nt,
        "pml_width": pml,
        "source_amplitudes": amp,
        "source_locations": srcs.to(dtype=torch.long),
        "receiver_locations": recs.to(dtype=torch.long),
        "accuracy": 2,
        "pml_freq": 25.0,
        "device": "cuda:0",
        "dtype": dtype,
    }


def build_em3d_case(dtype=torch.float64, nz=6, ny=8, nx=8, nt=8, pml=2, seed=3):
    """Small 3D EM model; 'ey' source at centre, 2 nearby 'ey' receivers."""
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    eps = 2.0 + 0.3 * torch.rand(nz, ny, nx, generator=g)
    sig = 0.01 * torch.ones(nz, ny, nx)
    mu = torch.ones(nz, ny, nx)
    dx, dt = 5.0, 2.0e-9
    srcs = torch.tensor([[[nz // 2, ny // 2, nx // 2]]])
    recs = torch.tensor(
        [[[nz // 2, ny // 2, nx // 2 - 2], [nz // 2, ny // 2 + 2, nx // 2]]]
    )
    amp = torch.zeros(1, 1, nt, dtype=dtype)
    amp[:, 0, 1] = 1.0
    return {
        "epsilon": eps.to(dtype=dtype),
        "sigma": sig.to(dtype=dtype),
        "mu": mu.to(dtype=dtype),
        "grid_spacing": (dx, dx, dx),
        "dt": dt,
        "nt": nt,
        "pml_width": pml,
        "source_amplitudes": amp,
        "source_locations": srcs.to(dtype=torch.long),
        "receiver_locations": recs.to(dtype=torch.long),
        "accuracy": 2,
        "source_component": "ey",
        "receiver_component": "ey",
        "device": "cuda:0",
        "dtype": dtype,
    }


def _run_scalar(c, **kw):
    c = dict(c)
    v = c.pop("v")
    dev = c.pop("device")
    c.pop("dtype")
    c.pop("nt", None)
    return scalar2d(v.to(dev), c.pop("dx"), c.pop("dt"), **c, **kw)


def _run_elastic(c, **kw):
    c = dict(c)
    lamb = c.pop("lamb")
    mu = c.pop("mu")
    buoyancy = c.pop("buoyancy")
    dev = c.pop("device")
    c.pop("dtype")
    c.pop("nt", None)
    return elastic2d(
        lamb.to(dev), mu.to(dev), buoyancy.to(dev),
        c.pop("grid_spacing"), c.pop("dt"), **c, **kw,
    )


def _run_em3d(c, **kw):
    c = dict(c)
    eps = c.pop("epsilon")
    sig = c.pop("sigma")
    mu = c.pop("mu")
    dev = c.pop("device")
    c.pop("dtype")
    c.pop("nt", None)
    return em3d(
        eps.to(dev), sig.to(dev), mu.to(dev),
        c.pop("grid_spacing"), c.pop("dt"), **c, **kw,
    )


def _build_continuation_case(
    name, dtype=torch.float64, *, accuracy=2, pml_width=2,
):
    """Return ``(run, amplitudes)`` for each public propagator."""
    device = torch.device("cuda:0")
    nt = 6
    amp_scale = 1.0e8 if name.startswith("elastic") else 1.0
    amp = torch.zeros(1, 1, nt, dtype=dtype)
    amp[..., 1] = amp_scale

    if name.startswith("scalar"):
        dx, dt = 5.0, 5.0e-4
        if "3d" in name:
            shape = (7, 8, 9)
            v = torch.full(shape, 1800.0, dtype=dtype, device=device)
            scatter = torch.full(shape, 10.0, dtype=dtype, device=device)
            src = torch.tensor([[[3, 4, 4]]])
            rec = torch.tensor([[[3, 4, 5]]])
            fn = scalar3d_born if name.endswith("born") else scalar3d
            models = (v, scatter) if name.endswith("born") else (v,)
            spacing = (dx, dx, dx)
        else:
            shape = (12, 14)
            v = torch.full(shape, 1800.0, dtype=dtype, device=device)
            scatter = torch.full(shape, 10.0, dtype=dtype, device=device)
            src = torch.tensor([[[6, 7]]])
            rec = torch.tensor([[[6, 8]]])
            fn = scalar2d_born if name.endswith("born") else scalar2d
            models = (v, scatter) if name.endswith("born") else (v,)
            spacing = dx if name == "scalar2d" else (dx, dx)

        def run(source_amplitudes, **state_args):
            return fn(
                *models,
                spacing,
                dt,
                source_amplitudes=source_amplitudes,
                source_locations=src,
                receiver_locations=rec,
                accuracy=accuracy,
                pml_width=pml_width,
                pml_freq=25.0,
                storage="none",
                **state_args,
            )

        return run, amp

    if name.startswith("elastic"):
        shape = (12, 14)
        rho = torch.full(shape, 2000.0, dtype=dtype, device=device)
        vp = torch.full(shape, 2000.0, dtype=dtype, device=device)
        vs = torch.full(shape, 1100.0, dtype=dtype, device=device)
        lamb = rho * (vp.square() - 2 * vs.square())
        mu = rho * vs.square()
        buoyancy = rho.reciprocal()
        src = torch.tensor([[[6, 7]]])
        rec = torch.tensor([[[6, 8]]])
        if name.endswith("born"):
            fn = elastic2d_born
            models = (
                lamb,
                mu,
                buoyancy,
                0.01 * lamb,
                0.01 * mu,
                0.01 * buoyancy,
            )
        else:
            fn = elastic2d
            models = (lamb, mu, buoyancy)

        def run(source_amplitudes, **state_args):
            return fn(
                *models,
                (5.0, 5.0),
                2.0e-4,
                source_amplitudes=source_amplitudes,
                source_locations=src,
                receiver_locations=rec,
                accuracy=accuracy,
                pml_width=pml_width,
                pml_freq=25.0,
                storage="none",
                **state_args,
            )

        return run, amp

    dx, dt = 1.0, 1.0e-9
    if "3d" in name:
        shape = (7, 8, 9)
        src = torch.tensor([[[3, 4, 4]]])
        rec = torch.tensor([[[3, 4, 5]]])
        spacing = (dx, dx, dx)
        fn = em3d_born if name.endswith("born") else em3d
    else:
        shape = (12, 14)
        src = torch.tensor([[[6, 7]]])
        rec = torch.tensor([[[6, 8]]])
        spacing = (dx, dx)
        fn = em2d_tm_born if name.endswith("born") else em2d_tm
    epsilon = torch.full(shape, 2.0, dtype=dtype, device=device)
    sigma = torch.full(shape, 0.01, dtype=dtype, device=device)
    mu = torch.ones(shape, dtype=dtype, device=device)
    models = (epsilon, sigma, mu)
    if name.endswith("born"):
        models += (
            torch.full_like(epsilon, 0.1),
            torch.full_like(sigma, 0.001),
            torch.full_like(mu, 0.01),
        )

    def run(source_amplitudes, **state_args):
        return fn(
            *models,
            spacing,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=src,
            receiver_locations=rec,
            accuracy=accuracy,
            pml_width=pml_width,
            storage="none",
            **state_args,
        )

    return run, amp


def _split_result(result, has_state):
    """Normalize propagator returns into ``(receiver_tensors, state)``."""
    if has_state:
        assert isinstance(result, tuple)
        *outputs, state = result
        return tuple(outputs), state
    if isinstance(result, tuple):
        return result, None
    return (result,), None


def test_forward_callback_fires_and_views():
    """Callback fires every N steps with live padded wavefields; inner view
    returns the unpadded model region."""
    c = build_case()
    seen = []

    def cb(state):
        seen.append(
            (
                state.step,
                state.nt,
                state.dt,
                tuple(state.get_wavefield(n).shape for n in ("u", "u_prev")),
                tuple(state.get_wavefield(n, view="full").shape for n in ("u",)),
            )
        )

    _run_scalar(c, storage="none", forward_callback=cb, callback_frequency=4)
    assert [s[0] for s in seen] == [0, 4, 8, 12, 16], [s[0] for s in seen]
    assert seen[0][1] == c["nt"] and seen[0][2] == c["dt"]
    ny, nx = c["v"].shape
    assert seen[0][3] == ((1, ny, nx), (1, ny, nx)), seen[0][3]
    fd_pad, pml = 1, c["pml_width"]
    assert seen[0][4] == ((1, ny + 2 * (fd_pad + pml), nx + 2 * (fd_pad + pml)),)


def test_return_state_padded_shapes():
    """return_state returns the final padded state with the N_STATE keys."""
    c = build_case()
    r, st = _run_scalar(c, storage="none", return_state=True)
    assert r.shape == (c["nt"], 1, 3)
    assert set(st) == {"u", "u_prev", "psi_y", "psi_x", "zeta_y", "zeta_x"}
    ny, nx = c["v"].shape
    fd_pad, pml = 1, c["pml_width"]
    for t in st.values():
        assert tuple(t.shape) == (1, ny + 2 * (fd_pad + pml), nx + 2 * (fd_pad + pml))


def test_initial_state_empty_equals_none():
    """initial_state={} behaves exactly like None (zero start)."""
    c = build_case()
    r0 = _run_scalar(c, storage="none")
    r1 = _run_scalar(c, storage="none", initial_state={})
    assert torch.equal(r0, r1)


def test_initial_state_partial_zero_fills_missing_keys():
    """A partial initial_state (physics fields only) is equivalent to the
    same dict with every missing key zero-filled."""
    c = build_case()
    nt = c["nt"]
    ca = dict(c)
    ca["source_amplitudes"] = c["source_amplitudes"][:, :, : nt // 2]
    cb_ = dict(c)
    cb_["source_amplitudes"] = c["source_amplitudes"][:, :, nt // 2 :]
    _, st = _run_scalar(ca, storage="none", return_state=True)
    partial = {k: st[k] for k in ("u", "u_prev")}
    zero_filled = {
        k: (st[k] if k in ("u", "u_prev") else torch.zeros_like(st[k]))
        for k in st
    }
    r_partial = _run_scalar(cb_, storage="none", initial_state=partial)
    r_zero_filled = _run_scalar(cb_, storage="none", initial_state=zero_filled)
    assert torch.equal(r_partial, r_zero_filled)
    # the restored physics fields make the continuation differ from a cold
    # (zero) start of the second half
    r_zero = _run_scalar(cb_, storage="none")
    assert not torch.equal(r_partial, r_zero)


def test_callback_illumination_accumulation():
    """A callback can accumulate source illumination during the forward run
    (the deepwave custom-imaging-condition pattern)."""
    c = build_case()
    illum = None

    def cb(state):
        nonlocal illum
        u = state.get_wavefield("u")
        acc = u * u
        illum = acc if illum is None else illum + acc

    _run_scalar(c, storage="none", forward_callback=cb)
    assert illum is not None
    assert illum.shape == (1,) + c["v"].shape, illum.shape
    assert illum.sum().item() > 0  # energy accumulated somewhere


@pytest.mark.parametrize("propagator", PROPAGATORS, ids=lambda fn: fn.__name__)
def test_all_propagators_share_state_callback_signature(propagator):
    """Every public full-wave and Born primitive exposes the same controls."""
    parameters = inspect.signature(propagator).parameters
    expected = {
        "forward_callback": None,
        "callback_frequency": 1,
        "return_state": False,
        "initial_state": None,
    }
    for name, default in expected.items():
        assert name in parameters
        assert parameters[name].default == default


@pytest.mark.parametrize("propagator", BORN_PROPAGATORS, ids=lambda fn: fn.__name__)
def test_all_born_propagators_accept_background_receivers(propagator):
    parameter = inspect.signature(propagator).parameters["bg_receiver_locations"]
    assert parameter.default is None


@pytest.mark.parametrize(
    ("case_name", "expected_fields"),
    (
        ("scalar2d", ("u", "u_prev")),
        ("scalar2d_born", ("u", "u_sc")),
        ("scalar3d", ("u", "u_prev")),
        ("scalar3d_born", ("u", "u_sc")),
        ("elastic2d", ("vy", "vx", "syy", "sxx", "sxy")),
        (
            "elastic2d_born",
            ("vy", "vx", "syy", "sxx", "sxy",
             "dvy", "dvx", "dsyy", "dsxx", "dsxy"),
        ),
        ("em2d_tm", ("ey", "hx", "hz")),
        ("em2d_tm_born", ("ey", "hx", "hz", "d_ey", "d_hx", "d_hz")),
        ("em3d", ("ex", "ey", "ez", "hx", "hy", "hz")),
        (
            "em3d_born",
            ("ex", "ey", "ez", "hx", "hy", "hz",
             "d_ex", "d_ey", "d_ez", "d_hx", "d_hy", "d_hz"),
        ),
    ),
)
def test_callback_exposes_all_physics_fields(case_name, expected_fields):
    """Callbacks expose physical fields consistently and keep PML memory private."""
    run, amp = _build_continuation_case(case_name)
    seen = []

    def callback(state):
        seen.append(state.wavefield_names)
        for name in expected_fields:
            assert state.get_wavefield(name).is_cuda

    run(amp, forward_callback=callback, callback_frequency=amp.shape[-1])
    assert seen == [expected_fields]


@pytest.mark.parametrize(
    "case_name",
    (
        "scalar2d",
        "scalar2d_born",
        "scalar3d",
        "scalar3d_born",
        "elastic2d",
        "elastic2d_born",
        "em2d_tm",
        "em2d_tm_born",
        "em3d",
        "em3d_born",
    ),
)
def test_all_propagators_reject_short_source_amplitudes(case_name):
    """Native loops must never read beyond a short source time axis."""
    run, amp = _build_continuation_case(case_name)
    with pytest.raises(ValueError, match="at least nt steps"):
        run(amp[..., :-1], nt=amp.shape[-1])


@pytest.mark.parametrize(
    "case_name",
    (
        "scalar2d",
        "scalar2d_born",
        "scalar3d",
        "scalar3d_born",
        "elastic2d",
        "elastic2d_born",
        "em2d_tm",
        "em2d_tm_born",
        "em3d",
        "em3d_born",
    ),
)
def test_locations_without_amplitudes_are_explicit_zero_sources(case_name):
    """Located sources with no amplitudes are safe no-ops on every path."""
    run, amp = _build_continuation_case(case_name)
    implicit, _ = _split_result(run(None, nt=amp.shape[-1]), has_state=False)
    explicit, _ = _split_result(run(torch.zeros_like(amp)), has_state=False)
    assert len(implicit) == len(explicit)
    for actual, expected in zip(implicit, explicit, strict=True):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "case_name",
    (
        "scalar2d",
        "scalar2d_born",
        "scalar3d",
        "scalar3d_born",
        "elastic2d",
        "elastic2d_born",
        "em2d_tm",
        "em2d_tm_born",
        "em3d",
        "em3d_born",
    ),
)
def test_all_propagators_continuation_bitwise_matches_one_shot(case_name):
    """A complete returned state gives exact forward continuation."""
    run, amp = _build_continuation_case(case_name)
    full, _ = _split_result(run(amp), has_state=False)
    split = amp.shape[-1] // 2
    first, state = _split_result(
        run(amp[..., :split], return_state=True), has_state=True,
    )
    second, _ = _split_result(
        run(amp[..., split:], initial_state=state), has_state=False,
    )

    assert state
    assert all(not value.requires_grad for value in state.values())
    assert any(torch.count_nonzero(value).item() for value in state.values())
    assert len(full) == len(first) == len(second)
    for expected, part1, part2 in zip(full, first, second, strict=True):
        assert torch.equal(torch.cat((part1, part2), dim=0), expected)


@pytest.mark.parametrize(
    "case_name",
    (
        "scalar2d", "scalar2d_born", "scalar3d", "scalar3d_born",
        "elastic2d", "elastic2d_born", "em2d_tm", "em2d_tm_born",
        "em3d", "em3d_born",
    ),
)
def test_all_propagators_accept_asymmetric_and_disabled_pml_sides(case_name):
    pml_width = (0, 1, 2, 0, 1, 2) if "3d" in case_name else (0, 1, 2, 0)
    run, amp = _build_continuation_case(case_name, pml_width=pml_width)
    outputs, _ = _split_result(run(amp), has_state=False)
    assert outputs
    assert all(torch.isfinite(output).all() for output in outputs)


@pytest.mark.parametrize(
    "case_name",
    (
        "scalar2d", "scalar2d_born", "scalar3d", "scalar3d_born",
        "elastic2d", "elastic2d_born", "em2d_tm", "em2d_tm_born",
        "em3d", "em3d_born",
    ),
)
def test_all_propagators_float32_float64_forward_parity(case_name):
    run64, amp64 = _build_continuation_case(case_name, torch.float64)
    run32, amp32 = _build_continuation_case(case_name, torch.float32)
    outputs64, _ = _split_result(run64(amp64), has_state=False)
    outputs32, _ = _split_result(run32(amp32), has_state=False)
    for output32, output64 in zip(outputs32, outputs64, strict=True):
        scale = output64.abs().max().clamp_min(1e-30)
        relative_error = (output32.double() - output64).abs().max() / scale
        assert relative_error < 1e-5


def test_initial_state_rejects_unknown_field():
    c = build_case()
    with pytest.raises(ValueError, match="unknown initial_state field"):
        _run_scalar(
            c,
            storage="none",
            initial_state={"u_prevv": torch.zeros(1)},
        )


@pytest.mark.parametrize("initial_state", ([], {"u": torch.zeros(2, 3)}))
def test_initial_state_rejects_invalid_type_or_shape(initial_state):
    c = build_case()
    error = TypeError if isinstance(initial_state, list) else ValueError
    with pytest.raises(error):
        _run_scalar(c, storage="none", initial_state=initial_state)


@pytest.mark.parametrize("frequency", (0, -1, 1.5, True))
def test_callback_frequency_must_be_positive_integer(frequency):
    c = build_case()
    with pytest.raises(ValueError, match="positive integer"):
        _run_scalar(c, storage="none", callback_frequency=frequency)


def test_forward_callback_must_be_callable():
    c = build_case()
    with pytest.raises(TypeError, match="must be callable"):
        _run_scalar(c, storage="none", forward_callback=object())
