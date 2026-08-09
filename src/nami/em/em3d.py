"""nami em3d: 3D electromagnetic FDTD (staggered Yee grid, C-PML) with a
native CUDA adjoint.

Forward/adjoint discretisation is the 3D generalisation of ``em2d_tm``,
following the maxwell3d conventions for the staggered-grid equations:

    per step: H half-step (Hx/Hy/Hz from curls of Ex/Ey/Ez with the
    half-integer CPML memory), E integer-step (snapshotting the pre-update
    Ex/Ey/Ez and the PML-modified curls for the adjoint), source injection
    (pre-scaled by ``cb * -1/(dx dy dz)`` into the source component),
    receiver recording (post-injection, from the receiver component).

The backward pass is the exact discrete transpose: ``record_grad_r`` /
``record_grad_f`` at the start of each adjoint step, ``coeff_grad`` and
``cq_grad`` model-gradient accumulators on the snapshot stride, and the
two-stage E/H transpose kernels with the time-reversed CPML memory
recursions.

Kernels are intentionally unoptimised (one flat launch per step, naive
stencil) -- the correctness baseline; performance work lands on top of it
later.

The spatial FD order is user-selectable (accuracy 2/4/6/8) with the
``fd.STAGGERED_DIFF1`` coefficient tables (staggered-grid convention);
the kernels are driven by coefficient arrays (fixed max radius
4, zero-padded for lower orders) and the per-side FD padding
``fd_pad = [accuracy // 2, accuracy // 2 - 1] * 3``.  Snapshots (six
streams: pre-update Ex/Ey/Ez and the PML-modified curls) live in device
memory, with optional checkpointing of the full wavefield state.
"""

import math

import nami_em3d as _ext
import torch

from ..common.callback import validate_callback_frequency, wrap_forward_callback
from ..common.cfl import check_cfl
from ..common.fd import check_accuracy, staggered_diff1_coeffs
from ..common.pml import set_pml_width
from ..common.state import allocate_final_state, prepare_initial_state, unpack_state
from ..common.storage import (
    SnapshotStorage,
    check_sample_steps,
    resolve_storage,
    storage_plan,
)
from ..common.survey import (
    extract_survey_3d,
    is_shot_batched,
    prepare_source_amplitudes,
)
from ._common import EPS0, MU0, _compile_material_coefficients, _pml_profile_1d

_COMPONENTS = ("ex", "ey", "ez")

# Checkpoint state layout (saved at time t BEFORE step t):
#   [0:3]   ex, ey, ez
#   [3:6]   hx, hy, hz
#   [6:12]  H-step memory variables (m_ey_z, m_ez_y, m_ez_x, m_ex_z,
#           m_ex_y, m_ey_x)
#   [12:18] E-step memory variables (m_hy_z, m_hz_y, m_hz_x, m_hx_z,
#           m_hx_y, m_hy_x)
N_STATE = 18
N_STREAMS = 6

# Full wavefield state layout (N_STATE order, saved at time t BEFORE step t):
#   [0:3]   ex, ey, ez
#   [3:6]   hx, hy, hz
#   [6:12]  H-step memory variables (m_ey_z, m_ez_y, m_ez_x, m_ex_z,
#           m_ex_y, m_ey_x)
#   [12:18] E-step memory variables (m_hy_z, m_hz_y, m_hz_x, m_hx_z,
#           m_hx_y, m_hy_x)
_WAVEFIELD_NAMES = (
    "ex", "ey", "ez", "hx", "hy", "hz",
    "m_ey_z", "m_ez_y", "m_ez_x", "m_ex_z", "m_ex_y", "m_ey_x",
    "m_hy_z", "m_hz_y", "m_hz_x", "m_hx_z", "m_hx_y", "m_hy_x",
)
_CALLBACK_FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz")


def _normalize_component(component, name):
    """Maps 'ex'/'ey'/'ez' to 0/1/2 (the maxwell3d component convention)."""
    if component not in _COMPONENTS:
        raise ValueError(
            f"{name} must be one of {_COMPONENTS}; got {component!r}."
        )
    return _COMPONENTS.index(component)


class EM3DFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ca_p,            # [n_shots, nz, ny, nx] or [1, nz, ny, nx] padded E coeffs
        cb_p,            # (batched when the first dim equals n_shots)
        cq_p,
        f,               # [nt, n_shots, n_src] pre-scaled source amplitudes
        src_i,           # [n_shots, n_src] flat indices
        rec_i,           # [n_shots, n_rec] flat indices
        profs,           # [az, azh, ay, ayh, ax, axh, bz, bzh, by, byh, bx,
                         #  bxh, kz, kzh, ky, kyh, kx, kxh]
        c,               # [4] staggered FD coefficients (zero-padded, max radius 4)
        fd_pad,          # [z0, z1, y0, y1, x0, x1] FD padding
        rdz, rdy, rdx,
        nt, pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
        ca_batched, cb_batched, cq_batched,
        grad_stride,
        ex_storage,      # SnapshotStorage streams (Ex/Ey/Ez + curls) or None
        ey_storage,
        ez_storage,
        curl_x_storage,
        curl_y_storage,
        curl_z_storage,
        source_component,    # 0/1/2 -> ex/ey/ez
        receiver_component,  # 0/1/2 -> ex/ey/ez
        ckpt_state,          # [n_ckpt, N_STATE, n_shots, nz, ny, nx] or None
        checkpoint_every,    # 0 = full storage; N = checkpoint every N steps
        segments,            # [(s0, s1)] replay segments; [] = full storage
        init_state,          # [N_STATE, n_shots, nz, ny, nx] initial state or None
        final_state,         # [N_STATE, n_shots, nz, ny, nx] output buffer or None
        forward_callback,    # cb(t, nt, ex, ey, ez, hx, hy, hz) or None
        callback_frequency,  # call the callback every N steps
    ):
        ext = _ext
        device = ca_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = ca_p.dtype
        n_shots = int(src_i.shape[0])
        nz, ny, nx = ca_p.shape[-3:]
        n_rec = rec_i.shape[1]

        ex = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        ey = torch.zeros_like(ex)
        ez = torch.zeros_like(ex)
        hx = torch.zeros_like(ex)
        hy = torch.zeros_like(ex)
        hz = torch.zeros_like(ex)
        # H-step memory variables (half-integer profiles)
        m_ey_z = torch.zeros_like(ex)
        m_ez_y = torch.zeros_like(ex)
        m_ez_x = torch.zeros_like(ex)
        m_ex_z = torch.zeros_like(ex)
        m_ex_y = torch.zeros_like(ex)
        m_ey_x = torch.zeros_like(ex)
        # E-step memory variables (integer profiles)
        m_hy_z = torch.zeros_like(ex)
        m_hz_y = torch.zeros_like(ex)
        m_hz_x = torch.zeros_like(ex)
        m_hx_z = torch.zeros_like(ex)
        m_hx_y = torch.zeros_like(ex)
        m_hy_x = torch.zeros_like(ex)
        r = torch.zeros(nt, n_shots, n_rec, device=device, dtype=dtype)
        if ex_storage is not None:
            ex_store = ex_storage.snap
            ey_store = ey_storage.snap
            ez_store = ez_storage.snap
            curl_x_store = curl_x_storage.snap
            curl_y_store = curl_y_storage.snap
            curl_z_store = curl_z_storage.snap
        else:
            ex_store = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            ey_store = torch.zeros_like(ex_store)
            ez_store = torch.zeros_like(ex_store)
            curl_x_store = torch.zeros_like(ex_store)
            curl_y_store = torch.zeros_like(ex_store)
            curl_z_store = torch.zeros_like(ex_store)

        az, azh, ay, ayh, ax, axh, bz, bzh, by, byh, bx, bxh, \
            kz, kzh, ky, kyh, kx, kxh = [p.contiguous() for p in profs]
        # checkpointed forward stores no snapshots (the backward replay
        # regenerates them); full storage writes every sampled step.
        store = 0 if segments else (1 if ex_storage is not None else 0)
        ext.forward_loop(
            ca_p, cb_p, cq_p,
            [ex, ey, ez, hx, hy, hz,
             m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x,
             m_hy_z, m_hz_y, m_hz_x, m_hx_z, m_hx_y, m_hy_x],
            az, bz, ay, by, ax, bx, kz, ky, kx,
            azh, bzh, ayh, byh, axh, bxh, kzh, kyh, kxh,
            ex_store, ey_store, ez_store,
            curl_x_store, curl_y_store, curl_z_store,
            f, src_i, r, rec_i,
            c,
            rdz, rdy, rdx,
            nt, grad_stride,
            pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
            ca_batched, cb_batched, cq_batched,
            fd_pad[0], fd_pad[1], fd_pad[2], fd_pad[3], fd_pad[4], fd_pad[5],
            source_component, receiver_component,
            store,
            checkpoint_every, ckpt_state,
            init_state, final_state, forward_callback, callback_frequency,
        )

        ctx.ext = ext
        ctx.save_for_backward(ca_p, cb_p, cq_p, f, src_i, rec_i)
        ctx.ex_storage = ex_storage
        ctx.ey_storage = ey_storage
        ctx.ez_storage = ez_storage
        ctx.curl_x_storage = curl_x_storage
        ctx.curl_y_storage = curl_y_storage
        ctx.curl_z_storage = curl_z_storage
        ctx.profs = profs
        ctx.c = c
        ctx.fd_pad = fd_pad
        ctx.rdz = rdz
        ctx.rdy = rdy
        ctx.rdx = rdx
        ctx.nt = nt
        ctx.pml_z0 = pml_z0
        ctx.pml_z1 = pml_z1
        ctx.pml_y0 = pml_y0
        ctx.pml_y1 = pml_y1
        ctx.pml_x0 = pml_x0
        ctx.pml_x1 = pml_x1
        ctx.ca_batched = ca_batched
        ctx.cb_batched = cb_batched
        ctx.cq_batched = cq_batched
        ctx.grad_stride = grad_stride
        ctx.source_component = source_component
        ctx.receiver_component = receiver_component
        ctx.ckpt_state = ckpt_state
        ctx.checkpoint_every = checkpoint_every
        ctx.segments = segments
        return r

    @staticmethod
    def backward(ctx, grad_r):
        ext = ctx.ext
        ca_p, cb_p, cq_p, f, src_i, rec_i = ctx.saved_tensors
        if ctx.ex_storage is None:
            raise RuntimeError(
                "em3d backward() requires snapshot storage: run the forward "
                "with an input requiring grad (and not under torch.no_grad())."
            )
        ex_storage = ctx.ex_storage
        ey_storage = ctx.ey_storage
        ez_storage = ctx.ez_storage
        curl_x_storage = ctx.curl_x_storage
        curl_y_storage = ctx.curl_y_storage
        curl_z_storage = ctx.curl_z_storage
        ex_store, ey_store, ez_store = (
            ex_storage.snap, ey_storage.snap, ez_storage.snap,
        )
        curl_x_store, curl_y_store, curl_z_store = (
            curl_x_storage.snap, curl_y_storage.snap, curl_z_storage.snap,
        )
        device = ca_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = ca_p.dtype
        n_shots = int(src_i.shape[0])
        nz, ny, nx = ca_p.shape[-3:]
        n_src, n_rec = src_i.shape[1], rec_i.shape[1]
        nt = ctx.nt

        if grad_r is None:
            grad_r = torch.zeros(nt, n_shots, n_rec, device=device, dtype=dtype)
        # grad_r arrives from autograd and may be a non-contiguous broadcast
        # view; the kernels use flat row-major indexing, so materialise it.
        grad_r = grad_r.contiguous()
        grad_f = torch.zeros(nt, n_shots, n_src, device=device, dtype=dtype)
        grad_ca = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        grad_cb = torch.zeros_like(grad_ca)
        grad_cq = torch.zeros_like(grad_ca)

        lam_ex = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        lam_ey = torch.zeros_like(lam_ex)
        lam_ez = torch.zeros_like(lam_ex)
        lam_hx = torch.zeros_like(lam_ex)
        lam_hy = torch.zeros_like(lam_ex)
        lam_hz = torch.zeros_like(lam_ex)
        # E-stage-1 memory (integer profiles) and work arrays
        m_lambda_hy_z = torch.zeros_like(lam_ex)
        m_lambda_hz_y = torch.zeros_like(lam_ex)
        m_lambda_hz_x = torch.zeros_like(lam_ex)
        m_lambda_hx_z = torch.zeros_like(lam_ex)
        m_lambda_hx_y = torch.zeros_like(lam_ex)
        m_lambda_hy_x = torch.zeros_like(lam_ex)
        work_hy_z = torch.zeros_like(lam_ex)
        work_hz_y = torch.zeros_like(lam_ex)
        work_hz_x = torch.zeros_like(lam_ex)
        work_hx_z = torch.zeros_like(lam_ex)
        work_hx_y = torch.zeros_like(lam_ex)
        work_hy_x = torch.zeros_like(lam_ex)
        # H-stage-1 memory (half-integer profiles) and work arrays
        m_lambda_ey_z = torch.zeros_like(lam_ex)
        m_lambda_ez_y = torch.zeros_like(lam_ex)
        m_lambda_ez_x = torch.zeros_like(lam_ex)
        m_lambda_ex_z = torch.zeros_like(lam_ex)
        m_lambda_ex_y = torch.zeros_like(lam_ex)
        m_lambda_ey_x = torch.zeros_like(lam_ex)
        work2_ey_z = torch.zeros_like(lam_ex)
        work2_ez_y = torch.zeros_like(lam_ex)
        work2_ez_x = torch.zeros_like(lam_ex)
        work2_ex_z = torch.zeros_like(lam_ex)
        work2_ex_y = torch.zeros_like(lam_ex)
        work2_ey_x = torch.zeros_like(lam_ex)

        az, azh, ay, ayh, ax, axh, bz, bzh, by, byh, bx, bxh, \
            kz, kzh, ky, kyh, kx, kxh = [p.contiguous() for p in ctx.profs]
        # integral sampling: each snapshot represents
        # `grad_stride` time steps of the model-gradient integral.
        scale = float(ctx.grad_stride)
        segments = ctx.segments
        if segments:
            # Checkpointed backward replays each segment's forward pass in
            # C++; these buffers hold the replayed wavefield state (the
            # adjoint state, lam + memory variables, carries across segments).
            ex_f = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            ey_f = torch.zeros_like(ex_f)
            ez_f = torch.zeros_like(ex_f)
            hx_f = torch.zeros_like(ex_f)
            hy_f = torch.zeros_like(ex_f)
            hz_f = torch.zeros_like(ex_f)
            m_ey_z_f = torch.zeros_like(ex_f)
            m_ez_y_f = torch.zeros_like(ex_f)
            m_ez_x_f = torch.zeros_like(ex_f)
            m_ex_z_f = torch.zeros_like(ex_f)
            m_ex_y_f = torch.zeros_like(ex_f)
            m_ey_x_f = torch.zeros_like(ex_f)
            m_hy_z_f = torch.zeros_like(ex_f)
            m_hz_y_f = torch.zeros_like(ex_f)
            m_hz_x_f = torch.zeros_like(ex_f)
            m_hx_z_f = torch.zeros_like(ex_f)
            m_hx_y_f = torch.zeros_like(ex_f)
            m_hy_x_f = torch.zeros_like(ex_f)
            state_f = [
                ex_f, ey_f, ez_f, hx_f, hy_f, hz_f,
                m_ey_z_f, m_ez_y_f, m_ez_x_f, m_ex_z_f, m_ex_y_f, m_ey_x_f,
                m_hy_z_f, m_hz_y_f, m_hz_x_f, m_hx_z_f, m_hx_y_f, m_hy_x_f,
            ]
            segments_t = torch.tensor(segments, dtype=torch.int64)
        else:
            state_f = []
            segments_t = torch.empty(0, 2, dtype=torch.int64)
        ext.adjoint_loop(
            ca_p, cb_p, cq_p,
            lam_ex, lam_ey, lam_ez, lam_hx, lam_hy, lam_hz,
            m_lambda_hy_z, m_lambda_hz_y, m_lambda_hz_x,
            m_lambda_hx_z, m_lambda_hx_y, m_lambda_hy_x,
            work_hy_z, work_hz_y, work_hz_x,
            work_hx_z, work_hx_y, work_hy_x,
            m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x,
            m_lambda_ex_z, m_lambda_ex_y, m_lambda_ey_x,
            work2_ey_z, work2_ez_y, work2_ez_x,
            work2_ex_z, work2_ex_y, work2_ey_x,
            az, bz, ay, by, ax, bx, kz, ky, kx,
            azh, bzh, ayh, byh, axh, bxh, kzh, kyh, kxh,
            ex_store, ey_store, ez_store,
            curl_x_store, curl_y_store, curl_z_store,
            grad_ca, grad_cb, grad_cq,
            grad_r, rec_i, grad_f, src_i, f,
            state_f,
            ctx.c,
            ctx.rdz, ctx.rdy, ctx.rdx, scale,
            nt, ctx.grad_stride,
            ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
            ctx.pml_x0, ctx.pml_x1,
            ctx.ca_batched, ctx.cb_batched, ctx.cq_batched,
            ctx.fd_pad[0], ctx.fd_pad[1], ctx.fd_pad[2],
            ctx.fd_pad[3], ctx.fd_pad[4], ctx.fd_pad[5],
            ctx.source_component, ctx.receiver_component,
            segments_t, ctx.ckpt_state,
        )

        grad_ca = grad_ca if ctx.ca_batched else grad_ca.sum(0, keepdim=True)
        grad_cb = grad_cb if ctx.cb_batched else grad_cb.sum(0, keepdim=True)
        grad_cq = grad_cq if ctx.cq_batched else grad_cq.sum(0, keepdim=True)

        # 38 forward() inputs: 3 model grads + grad_f + 34 x None
        return (
            grad_ca, grad_cb, grad_cq, grad_f,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            None, None,
        )


def em3d(
    epsilon,
    sigma,
    mu,
    grid_spacing,
    dt,
    source_amplitudes=None,
    source_locations=None,
    receiver_locations=None,
    accuracy=2,
    pml_width=20,
    nt=None,
    storage="auto",
    sample_steps=1,
    ckpt_steps=None,
    source_component="ey",
    receiver_component="ey",
    forward_callback=None,
    callback_frequency=1,
    return_state=False,
    initial_state=None,
):
    """3D electromagnetic (Ex/Ey/Ez-Hx/Hy/Hz) forward modelling / FWI primitive.

    Discretisation follows the maxwell3d formulation (the 3D generalisation of
    the 2D TM scheme mirrored by ``em2d_tm``): ``ca/cb/cq`` material
    coefficients are compiled from the padded epsilon/sigma/mu models
    (differentiable), sources are pre-scaled by ``cb * -1/(dx dy dz)`` and
    injected into ``source_component``, and receivers record the selected
    ``receiver_component`` after injection.

    Args:
        epsilon: Relative permittivity model [nz, ny, nx] (or [1, nz, ny, nx]).
        sigma: Conductivity model (S/m).
        mu: Relative permeability model.
        grid_spacing: Cell size (scalar or [dz, dy, dx]).
        dt: Time step interval (s).
        source_amplitudes: [n_shots, n_src, nt] source amplitudes, or None.
        source_locations: [n_shots, n_src, 3] (z, y, x) source locations.
        receiver_locations: [n_shots, n_rec, 3] receiver locations.
        accuracy: Finite-difference accuracy order (2, 4, 6 or 8).
        pml_width: PML width (int or [z0, z1, y0, y1, x0, x1]).
        nt: Number of time steps (defaults to source_amplitudes length).
        storage: 'auto' (default) keeps wavefield snapshots when any input
            requires grad (outside torch.no_grad()); 'none' runs forward-only
            (backward raises RuntimeError).
        sample_steps: sample model gradients and snapshots every N time steps
            (default 1); larger values trade gradient accuracy for memory.
            Receiver traces stay exact.
        ckpt_steps: checkpoint every N time steps; None (default) auto-selects
            ~sqrt(nt), 0 = full storage (every sampled step), N = save wavefield
            state every N steps and replay on backward.  Same gradients as full
            storage at the same sample_steps.
        source_component: Component to inject the source into ('ex', 'ey'
            or 'ez'; default 'ey', matching the maxwell3d default).
        receiver_component: Component recorded at the receivers (default
            'ey').
        forward_callback: called every ``callback_frequency`` steps with a
            ``CallbackState`` (deepwave-style) exposing the current padded
            wavefields via ``state.get_wavefield(name, view)`` — useful for
            RTM imaging conditions, illumination accumulation, monitoring.
        callback_frequency: call ``forward_callback`` every N time steps.
        return_state: if True, return ``(r, state)`` where ``state`` is a
            dict of the FINAL padded wavefield state (keys ``ex``, ``ey``,
            ``ez``, ``hx``, ``hy``, ``hz`` and the twelve PML memory
            variables) suitable for continuation via ``initial_state``.
        initial_state: a dict of initial wavefield state (padded grid, keys
            as in the ``return_state=True`` output) to continue a previous
            run.  The staggered Yee state before step 0 holds the E fields
            at t = 0 (integer time) and the H fields at t = -1/2 (half
            integer time).  Missing keys are zero-filled; a complete state
            dict (every key, including the PML memory variables) makes a
            split run bitwise match a one-shot run, while a partial dict
            restores only the given fields with the remaining state starting
            from zero.
            State I/O is for forward continuation; autograd does not propagate
            across the boundary between runs.  State dicts are ephemeral
            runtime snapshots: they may be passed back only to the same
            propagator with the same model layout and nami version, and are
            not a stable long-term checkpoint format.

    Returns:
        receiver_amplitudes [nt, n_shots, n_rec] (or ``(r, state)``
        when ``return_state=True``).
    """
    accuracy = check_accuracy(accuracy)
    if not isinstance(grid_spacing, (list, tuple)):
        grid_spacing = [float(grid_spacing)] * 3
    grid_spacing = [float(g) for g in grid_spacing]
    pml_w = set_pml_width(pml_width, 3)
    fd_pad = [
        accuracy // 2, accuracy // 2 - 1,
        accuracy // 2, accuracy // 2 - 1,
        accuracy // 2, accuracy // 2 - 1,
    ]
    device = epsilon.device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = epsilon.dtype
    source_component = _normalize_component(source_component, "source_component")
    receiver_component = _normalize_component(receiver_component, "receiver_component")

    if source_amplitudes is not None:
        n_shots = source_amplitudes.shape[0]
        nt_inner = nt or source_amplitudes.shape[2]
    elif source_locations is not None:
        n_shots = source_locations.shape[0]
        nt_inner = nt
    else:
        n_shots = 1
        nt_inner = nt
    if nt_inner is None:
        raise ValueError("nt must be provided when source_amplitudes is None.")

    c0 = 1.0 / math.sqrt(EPS0 * MU0)
    max_vel = float(
        (c0 / torch.sqrt((epsilon * mu).abs().clamp_min(1e-30))).max().item()
    )
    check_cfl(grid_spacing, dt, max_vel, "em3d", c_max=1.0)

    (epsilon_p, sigma_p, mu_p), src_i, rec_i = extract_survey_3d(
        [epsilon, sigma, mu],
        source_locations,
        receiver_locations,
        fd_pad,
        pml_w,
        n_shots,
        device,
        dtype,
    )

    ca_p, cb_p, cq_p = _compile_material_coefficients(epsilon_p, sigma_p, mu_p, dt)
    nz, ny, nx = epsilon_p.shape[-3:]
    profs = _set_em_pml_profiles_3d(
        pml_w, fd_pad, dt, grid_spacing, nz, ny, nx, dtype, device,
        accuracy=accuracy,
    )

    c = staggered_diff1_coeffs(accuracy, dtype, device)

    amp = prepare_source_amplitudes(
        source_amplitudes, n_shots, src_i.shape[1], nt_inner,
        device=device, dtype=dtype,
    )

    dz, dy, dx = grid_spacing
    source_coeff = -1.0 / (dz * dy * dx)
    if amp.numel() > 0:
        src_mask = src_i != -1
        src_i_masked = src_i.masked_fill(~src_mask, 0)
        cb_flat = cb_p.reshape(-1, nz * ny * nx).expand(n_shots, -1)
        cb_at_src = cb_flat.gather(1, src_i_masked)
        f = (amp.permute(2, 0, 1) * cb_at_src.unsqueeze(0) * source_coeff).contiguous()
    else:
        f = torch.empty(0, device=device, dtype=dtype)

    rdz, rdy, rdx = 1.0 / dz, 1.0 / dy, 1.0 / dx
    pml_z0, pml_z1 = fd_pad[0] + pml_w[0], nz - fd_pad[1] - pml_w[1]
    pml_y0, pml_y1 = fd_pad[2] + pml_w[2], ny - fd_pad[3] - pml_w[3]
    pml_x0, pml_x1 = fd_pad[4] + pml_w[4], nx - fd_pad[5] - pml_w[5]
    # ca/cb both depend on epsilon and sigma; use the compiled tensor shapes
    # after broadcasting to select the CUDA model slab.
    ca_batched = int(is_shot_batched(ca_p, n_shots, spatial_ndim=3))
    cb_batched = int(is_shot_batched(cb_p, n_shots, spatial_ndim=3))
    cq_batched = int(is_shot_batched(cq_p, n_shots, spatial_ndim=3))

    storage_mode = resolve_storage(storage)
    grad_stride = check_sample_steps(sample_steps)
    storage_enabled = storage_mode == "auto" and torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (epsilon, sigma, mu, source_amplitudes)
    )
    checkpoint_every, segments, n_snap, n_ckpt = storage_plan(
        nt_inner, N_STATE, grad_stride, N_STREAMS, storage_enabled,
        ckpt_steps=ckpt_steps,
    )

    ex_storage = ey_storage = ez_storage = None
    curl_x_storage = curl_y_storage = curl_z_storage = None
    ckpt_state = None
    if storage_enabled:
        # SnapshotStorage is 2D-only ([n_snap, n_shots, ny, nx] with
        # shot_count = n_shots*ny*nx); passing ny=nz, nx=ny*nx gives each
        # snapshot a contiguous flat [n_shots, nz*ny*nx] block.
        ex_storage = SnapshotStorage(
            _ext, n_snap, n_shots, nz, ny * nx, dtype, device,
        )
        ey_storage = SnapshotStorage(
            _ext, n_snap, n_shots, nz, ny * nx, dtype, device,
        )
        ez_storage = SnapshotStorage(
            _ext, n_snap, n_shots, nz, ny * nx, dtype, device,
        )
        curl_x_storage = SnapshotStorage(
            _ext, n_snap, n_shots, nz, ny * nx, dtype, device,
        )
        curl_y_storage = SnapshotStorage(
            _ext, n_snap, n_shots, nz, ny * nx, dtype, device,
        )
        curl_z_storage = SnapshotStorage(
            _ext, n_snap, n_shots, nz, ny * nx, dtype, device,
        )
        if n_ckpt > 0:
            ckpt_state = torch.zeros(
                n_ckpt, N_STATE, n_shots, nz, ny, nx, device=device, dtype=dtype,
            )

    # Wavefield I/O (deepwave-style): continuation initial state, optional
    # final-state output, and the per-step forward callback.
    state_shape = (n_shots, nz, ny, nx)
    init_state = prepare_initial_state(
        initial_state, _WAVEFIELD_NAMES, state_shape, device=device, dtype=dtype,
    )
    final_state = allocate_final_state(
        return_state, _WAVEFIELD_NAMES, state_shape, device=device, dtype=dtype,
    )
    callback_frequency = validate_callback_frequency(callback_frequency)
    cb = (
        wrap_forward_callback(
            forward_callback,
            _CALLBACK_FIELDS,
            float(dt), fd_pad, list(pml_w),
        )
        if forward_callback is not None
        else None
    )

    r = EM3DFunc.apply(
        ca_p, cb_p, cq_p, f, src_i, rec_i, profs, c, fd_pad,
        rdz, rdy, rdx, nt_inner,
        pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
        ca_batched, cb_batched, cq_batched,
        grad_stride,
        ex_storage, ey_storage, ez_storage,
        curl_x_storage, curl_y_storage, curl_z_storage,
        source_component, receiver_component,
        ckpt_state, checkpoint_every, segments,
        init_state, final_state, cb, callback_frequency,
    )
    if return_state:
        return r, unpack_state(final_state, _WAVEFIELD_NAMES)
    return r


def _set_em_pml_profiles_3d(
    pml_width,
    fd_pad,
    dt,
    grid_spacing,
    nz,
    ny,
    nx,
    dtype,
    device,
    accuracy=2,
    eps_scale=EPS0,
):
    """Sets up the eighteen flat 1D CPML profiles for the 3D staggered grid.

    Returns [az, azh, ay, ayh, ax, axh, bz, bzh, by, byh, bx, bxh,
    kz, kzh, ky, kyh, kx, kxh] where the ``_h`` profiles are evaluated at
    half-integer grid points, matching the kernel argument order (and
    the maxwell3d profile order).  Same formula as em2d_tm's
    ``_set_em_pml_profiles`` (and ``setup_pml``).

    ``accuracy`` is accepted for API parity with the scalar/staggered PML
    setup and validated, but the EM a/b/k profiles do not depend on the FD
    order.
    """
    check_accuracy(accuracy)
    dz, dy, dx = grid_spacing
    pml_start = [
        fd_pad[0] + pml_width[0],
        nz - 1 - fd_pad[1] - pml_width[1],
        fd_pad[2] + pml_width[2],
        ny - 1 - fd_pad[3] - pml_width[3],
        fd_pad[4] + pml_width[4],
        nx - 1 - fd_pad[5] - pml_width[5],
    ]

    profiles = []
    for half in (False, True):
        az, bz, kz = _pml_profile_1d(
            pml_width[:2], pml_start[:2], dt, nz, dtype, device, half,
            accuracy=accuracy, grid_spacing=dz, eps_scale=eps_scale,
        )
        ay, by, ky = _pml_profile_1d(
            pml_width[2:4], pml_start[2:4], dt, ny, dtype, device, half,
            accuracy=accuracy, grid_spacing=dy, eps_scale=eps_scale,
        )
        ax, bx, kx = _pml_profile_1d(
            pml_width[4:], pml_start[4:], dt, nx, dtype, device, half,
            accuracy=accuracy, grid_spacing=dx, eps_scale=eps_scale,
        )
        profiles.extend([az, ay, ax, bz, by, bx, kz, ky, kx])

    # Reorder to [az, azh, ay, ayh, ax, axh, bz, bzh, by, byh, bx, bxh,
    #              kz, kzh, ky, kyh, kx, kxh]
    return [
        profiles[0], profiles[9], profiles[1], profiles[10],
        profiles[2], profiles[11], profiles[3], profiles[12],
        profiles[4], profiles[13], profiles[5], profiles[14],
        profiles[6], profiles[15], profiles[7], profiles[16],
        profiles[8], profiles[17],
    ]
