"""nami scalar2d: 2D acoustic FDTD with native CUDA adjoint.

Forward/adjoint discretisation follows the reference compiled scalar
backend (bit-for-bit reproducible in float64): CPML (Pasalic &
McGarry) with split-field memory variables, pressure-source convention
``f = -amp * v^2 dt^2`` and receivers recording the pre-update field.
The spatial FD order is user-selectable (accuracy 2/4/6/8) with the
``regular_grid`` coefficient tables; the kernels are driven by
coefficient arrays (fixed max radius 4, zero-padded for lower orders).

Kernels are intentionally unoptimised (naive stencil, one launch per
operator) — that is the correctness baseline; performance work
(interior/PML fusion, shared-memory tiling) lands on top of it later.
The time-stepping loop itself runs inside the extension
(``forward_loop`` / ``adjoint_loop``): one pybind call per pass, with
checkpoint save/restore and snapshot offsets computed in C++.
"""

import nami_scalar2d as _ext
import torch

from ..common.callback import validate_callback_frequency, wrap_forward_callback
from ..common.cfl import check_cfl
from ..common.fd import diff1_coeffs, diff2_coeffs
from ..common.pml import set_acoustic_pml_profiles, set_pml_width
from ..common.state import allocate_final_state, prepare_initial_state, unpack_state
from ..common.storage import (
    SnapshotStorage,
    check_sample_steps,
    resolve_storage,
    storage_plan,
)
from ..common.survey import extract_survey_2d, prepare_source_amplitudes

# Checkpoint state layout for the acoustic field:
#   [0] u[t % 3]      field at time t
#   [1] u[(t - 1)%3]  field at time t - 1
#   [2] psi_y[t % 2]  split-field memory (y)
#   [3] psi_x[t % 2]  split-field memory (x)
#   [4] zeta_y[t % 2] auxiliary memory (y)
#   [5] zeta_x[t % 2] auxiliary memory (x)
N_STATE = 6
N_STREAMS = 1

_WAVEFIELD_NAMES = ("u", "u_prev", "psi_y", "psi_x", "zeta_y", "zeta_x")
_CALLBACK_FIELDS = ("u", "u_prev")


class Scalar2DFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v_p,             # [n_shots, ny, nx] or [1, ny, nx] padded model
        f,               # [nt, n_shots, n_src] pre-scaled source amplitudes
        src_i,           # [n_shots, n_src] flat indices
        rec_i,           # [n_shots, n_rec] flat indices
        profs,           # [ay, by, dbydy, ax, bx, dbxdx]
        c1, c2,          # FD coefficient arrays (regular grid)
        rdy, rdx, rdy2, rdx2, dt2,
        nt,
        pml_y0, pml_y1, pml_x0, pml_x1,          # forward PML
        pml_y0_b, pml_y1_b, pml_x0_b, pml_x1_b,  # backward PML
        v_batched, grad_stride, fd_pad,
        storage,         # SnapshotStorage or None (forward-only)
        ckpt_state,      # [n_ckpt, N_STATE, n_shots, ny, nx] or None
        checkpoint_every,  # 0 = full storage; N = checkpoint every N steps
        segments,        # [(s0, s1)] replay segments; [] = full storage
        init_state,      # [N_STATE, n_shots, ny, nx] initial wavefield or None
        final_state,     # [N_STATE, n_shots, ny, nx] output buffer or None
        forward_callback,  # cb(t, nt, u, u_prev) or None
        callback_frequency,  # call the callback every N steps
    ):
        ext = _ext
        device = v_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = v_p.dtype
        # n_shots from survey (sources), not model batch: shared [1, ny, nx]
        # still runs n_shots wavefields (v_batched selects model slab 0).
        n_shots = int(src_i.shape[0])
        ny, nx = v_p.shape[-2:]
        n_rec = rec_i.shape[1]

        u = [torch.zeros(n_shots, ny, nx, device=device, dtype=dtype) for _ in range(3)]
        psi_y = [torch.zeros_like(u[0]) for _ in range(2)]
        psi_x = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_y = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_x = [torch.zeros_like(u[0]) for _ in range(2)]
        r = torch.zeros(nt, n_shots, n_rec, device=device, dtype=dtype)

        ay, by, dbydy, ax, bx, dbxdx = [p.contiguous() for p in profs]
        w_store = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        if storage is not None:
            w_store = storage.snap
        # checkpointed forward stores no snapshots (the backward replay
        # regenerates them); full storage writes every sampled step.
        store = 0 if segments else (1 if storage is not None else 0)
        ext.forward_loop(
            v_p, u, psi_y, psi_x, zeta_y, zeta_x,
            ay, by, dbydy, ax, bx, dbxdx, w_store,
            f, src_i, r, rec_i,
            c1, c2,
            rdy, rdx, rdy2, rdx2, dt2,
            nt, grad_stride,
            pml_y0, pml_y1, pml_x0, pml_x1,
            v_batched, store, fd_pad,
            checkpoint_every, ckpt_state,
            init_state, final_state, forward_callback, callback_frequency,
        )

        ctx.ext = ext
        ctx.save_for_backward(v_p, f, src_i, rec_i, w_store, c1, c2)
        ctx.n_shots = n_shots
        ctx.rdy, ctx.rdx, ctx.rdy2, ctx.rdx2, ctx.dt2 = rdy, rdx, rdy2, rdx2, dt2
        ctx.nt = nt
        ctx.pml_y0, ctx.pml_y1, ctx.pml_x0, ctx.pml_x1 = (
            pml_y0, pml_y1, pml_x0, pml_x1,
        )
        ctx.pml_y0_b, ctx.pml_y1_b, ctx.pml_x0_b, ctx.pml_x1_b = (
            pml_y0_b, pml_y1_b, pml_x0_b, pml_x1_b,
        )
        ctx.v_batched = v_batched
        ctx.grad_stride = grad_stride
        ctx.profs = profs
        ctx.fd_pad = fd_pad
        ctx.storage = storage
        ctx.ckpt_state = ckpt_state
        ctx.segments = segments
        return r

    @staticmethod
    def backward(ctx, grad_r):
        ext = ctx.ext
        v_p, f, src_i, rec_i, w_store, c1, c2 = ctx.saved_tensors
        if ctx.storage is None:
            raise RuntimeError(
                "scalar2d backward() requires snapshot storage: run the forward "
                "with an input requiring grad (and not under torch.no_grad())."
            )
        device = v_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = v_p.dtype
        n_shots = ctx.n_shots
        ny, nx = v_p.shape[-2:]
        n_src, n_rec = src_i.shape[1], rec_i.shape[1]
        nt = ctx.nt
        grad_stride = ctx.grad_stride

        if grad_r is None:
            grad_r = torch.zeros(nt, n_shots, n_rec, device=device, dtype=dtype)
        # grad_r arrives from autograd and may be a non-contiguous broadcast
        # view; the kernels use flat row-major indexing, so materialise it.
        grad_r = grad_r.contiguous()
        grad_f = torch.zeros(nt, n_shots, n_src, device=device, dtype=dtype)
        # Per-shot grads; summed below when the model is shared (not batched).
        grad_v = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)

        lam = [
            torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
            for _ in range(3)
        ]
        psi_y = [torch.zeros_like(lam[0]) for _ in range(2)]
        psi_x = [torch.zeros_like(lam[0]) for _ in range(2)]
        zeta_y = [torch.zeros_like(lam[0]) for _ in range(2)]
        zeta_x = [torch.zeros_like(lam[0]) for _ in range(2)]

        ay, by, dbydy, ax, bx, dbxdx = [p.contiguous() for p in ctx.profs]
        # integral sampling: each snapshot represents
        # `grad_stride` time steps of the model-gradient integral.
        scale = float(grad_stride)
        segments = ctx.segments
        if segments:
            # Checkpointed backward replays each segment's forward pass in
            # C++; these rings hold the replayed wavefield state.
            u = [
                torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
                for _ in range(3)
            ]
            psi_y_f = [torch.zeros_like(lam[0]) for _ in range(2)]
            psi_x_f = [torch.zeros_like(lam[0]) for _ in range(2)]
            zeta_y_f = [torch.zeros_like(lam[0]) for _ in range(2)]
            zeta_x_f = [torch.zeros_like(lam[0]) for _ in range(2)]
            segments_t = torch.tensor(segments, dtype=torch.int64)
        else:
            u, psi_y_f, psi_x_f, zeta_y_f, zeta_x_f = [], [], [], [], []
            segments_t = torch.empty(0, 2, dtype=torch.int64)
        ext.adjoint_loop(
            v_p, lam, psi_y, psi_x, zeta_y, zeta_x,
            ay, by, dbydy, ax, bx, dbxdx,
            w_store, grad_v,
            grad_r, rec_i, grad_f, src_i, f,
            u, psi_y_f, psi_x_f, zeta_y_f, zeta_x_f,
            c1, c2,
            ctx.rdy, ctx.rdx, ctx.rdy2, ctx.rdx2, scale, ctx.dt2,
            nt, grad_stride,
            ctx.pml_y0, ctx.pml_y1, ctx.pml_x0, ctx.pml_x1,
            ctx.pml_y0_b, ctx.pml_y1_b, ctx.pml_x0_b, ctx.pml_x1_b,
            ctx.v_batched, ctx.fd_pad,
            segments_t, ctx.ckpt_state,
        )

        # Shared model: sum per-shot grads to match v_p shape [1, ny, nx]
        # (same contract as elastic2d / em2d_tm). Batched model keeps
        # [n_shots, ...].
        if not ctx.v_batched:
            grad_v = grad_v.sum(0, keepdim=True)
        # 32 forward() inputs: grad_v + grad_f + 30 x None
        return (
            grad_v,
            grad_f,      # grad w.r.t. pre-scaled f; scaled to amp in scalar2d()
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
        )


def scalar2d(
    v,
    grid_spacing,
    dt,
    source_amplitudes=None,
    source_locations=None,
    receiver_locations=None,
    accuracy=2,
    pml_width=20,
    pml_freq=25.0,
    nt=None,
    storage="auto",
    sample_steps=1,
    ckpt_steps=None,
    forward_callback=None,
    callback_frequency=1,
    return_state=False,
    initial_state=None,
):
    """2D acoustic forward + adjoint (torch in/out).

    Pressure-source convention: pre-scaled by
    ``-amp * v^2 dt^2``; receivers record the pre-update field.

    Args:
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
        forward_callback: called every ``callback_frequency`` steps with a
            ``CallbackState`` (deepwave-style) exposing the current padded
            wavefields via ``state.get_wavefield(name, view)`` — useful for
            RTM imaging conditions, illumination accumulation, monitoring.
        callback_frequency: call ``forward_callback`` every N time steps.
        return_state: if True, return ``(r, state)`` where ``state`` is a
            dict of the FINAL padded wavefield state (keys ``u``, ``u_prev``,
            ``psi_y``, ``psi_x``, ``zeta_y``, ``zeta_x``) suitable for
            continuation via ``initial_state``.
        initial_state: a dict of initial wavefield state (padded grid, keys
            as in the ``return_state=True`` output) to continue a previous
            run.  Missing keys are zero-filled; a complete state dict (every
            key, including the PML memory variables) makes a split run
            bitwise match a one-shot run, while a partial dict restores only
            the given fields with the remaining state starting from zero.
            State I/O is for forward continuation; autograd does not propagate
            across the boundary between runs.  State dicts are ephemeral
            runtime snapshots: they may be passed back only to the same
            propagator with the same model layout and nami version, and are
            not a stable long-term checkpoint format.

    Returns receiver amplitudes [nt, n_shots, n_rec] (or ``(r, state)``
    when ``return_state=True``).
    """
    if not isinstance(grid_spacing, (list, tuple)):
        grid_spacing = [float(grid_spacing)] * 2
    grid_spacing = [float(d) for d in grid_spacing]
    if len(grid_spacing) != 2:
        raise ValueError(
            "grid_spacing must be a scalar or length 2 [dy, dx]."
        )
    pml_w = set_pml_width(pml_width, 2)
    fd_pad = [accuracy // 2] * 4
    device = v.device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = v.dtype

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

    (v_p,), src_i, rec_i = extract_survey_2d(
        [v],
        source_locations,
        receiver_locations,
        fd_pad,
        pml_w,
        n_shots,
        device,
        dtype,
    )
    # Shared [ny, nx] / [1, ny, nx] models stay rank-1 in the batch dim;
    # kernels index slab 0 (v_batched=0) while wavefields are n_shots-wide.
    # Per-shot model grads are summed in backward (elastic/EM style).
    ny, nx = v_p.shape[-2:]
    max_vel = float(v.detach().abs().max())
    check_cfl(grid_spacing, dt, max_vel, "scalar2d")
    profs = set_acoustic_pml_profiles(
        pml_w, fd_pad, dt, grid_spacing, max_vel, pml_freq, (ny, nx), dtype,
        device,
        accuracy=accuracy,
    )

    amp = prepare_source_amplitudes(
        source_amplitudes, n_shots, src_i.shape[1], nt_inner,
        device=device, dtype=dtype,
    )
    if amp.numel() > 0:
        src_mask = src_i != -1
        src_i_masked = src_i.masked_fill(~src_mask, 0)
        # per-shot v at each source: expand the flat rows for gather only
        # (not the full model into the Function).
        v_flat = v_p.reshape(-1, ny * nx).expand(n_shots, -1)
        v_at_src = v_flat.gather(1, src_i_masked)
        f = (
            -amp.permute(2, 0, 1) * (v_at_src.unsqueeze(0) ** 2 * dt * dt)
        ).contiguous()
    else:
        f = torch.empty(0, device=device, dtype=dtype)

    c1 = diff1_coeffs(accuracy, dtype, device)
    c2 = diff2_coeffs(accuracy, dtype, device)
    rdy, rdx = 1.0 / grid_spacing[0], 1.0 / grid_spacing[1]
    rdy2, rdx2 = rdy * rdy, rdx * rdx
    dt2 = float(dt) * float(dt)
    pml_y0, pml_y1 = fd_pad[0] + pml_w[0], ny - fd_pad[1] - pml_w[1]
    pml_x0, pml_x1 = fd_pad[2] + pml_w[2], nx - fd_pad[3] - pml_w[3]
    # the backward pass widens the PML region by one fd_pad (forward
    # boundary + fd_pad, clamped to n - fd_pad): the transpose of the
    # forward PML stencil reads one cell further into the interior, so the
    # adjoint CPML branch must cover those cells.
    pml_y0_b = min(pml_y0 + fd_pad[0], ny - fd_pad[0])
    pml_y1_b = max(pml_y0_b, pml_y1 - fd_pad[1])
    pml_x0_b = min(pml_x0 + fd_pad[2], nx - fd_pad[2])
    pml_x1_b = max(pml_x0_b, pml_x1 - fd_pad[3])
    v_batched = 1 if (v.ndim == 3 and v.shape[0] == n_shots and v.shape[0] > 1) else 0

    storage_mode = resolve_storage(storage)
    grad_stride = check_sample_steps(sample_steps)
    storage_enabled = storage_mode == "auto" and torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (v, source_amplitudes)
    )
    checkpoint_every, segments, n_snap, n_ckpt = storage_plan(
        nt_inner, N_STATE, grad_stride, N_STREAMS, storage_enabled,
        ckpt_steps=ckpt_steps,
    )
    store_obj = None
    ckpt_state = None
    if storage_enabled:
        store_obj = SnapshotStorage(
            _ext, n_snap, n_shots, ny, nx, dtype, device,
        )
        if n_ckpt > 0:
            ckpt_state = torch.zeros(
                n_ckpt, N_STATE, n_shots, ny, nx, device=device, dtype=dtype,
            )

    # Wavefield I/O (deepwave-style): continuation initial state, optional
    # final-state output, and the per-step forward callback.
    state_shape = (n_shots, ny, nx)
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

    r = Scalar2DFunc.apply(
        v_p, f, src_i, rec_i, profs, c1, c2,
        rdy, rdx, rdy2, rdx2, dt2,
        nt_inner,
        pml_y0, pml_y1, pml_x0, pml_x1,
        pml_y0_b, pml_y1_b, pml_x0_b, pml_x1_b,
        v_batched, grad_stride, fd_pad[0],
        store_obj, ckpt_state, checkpoint_every, segments,
        init_state, final_state, cb, callback_frequency,
    )

    if return_state:
        return r, unpack_state(final_state, _WAVEFIELD_NAMES)
    return r
