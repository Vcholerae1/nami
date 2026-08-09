"""nami scalar3d Born: 3D acoustic Born FDTD with native CUDA adjoint.

Direct translation of the compiled 3D scalar_born backend (bit-for-bit
reproducible in float64): a background wavefield is propagated as in
:mod:`scalar3d`, and
a scattered wavefield is driven by the source
``2 * v * scatter * dt^2 * (Laplacian of the background wavefield)``
(equivalently ``2/v * scatter * dt^2 * d2(u_bg)/dt2``).  The adjoint is the
exact discrete transpose: the background adjoint evolves with the Laplacian
of ``v^2 dt^2 lam_bg + 2 v dt^2 scatter lam_sc``, and velocity/scatter
gradients accumulate from stored Laplacian snapshots of both fields.

Return convention: ``(receiver_amplitudes, bg_receiver_amplitudes)`` with
shapes ``[nt, n_shots, n_rec]`` (the reference returns these as the last
two elements of its output tuple in the opposite order, with time moved
last).
``receiver_amplitudes`` is the scattered wavefield recorded at
``receiver_locations``; ``bg_receiver_amplitudes`` is the background
wavefield recorded at ``bg_receiver_locations``.
"""

import nami_scalar3d_born as _ext
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
from ..common.survey import extract_survey_3d, prepare_source_amplitudes

# Checkpoint state layout for the Born wavefield (saved at t before step t):
#   [0]  u[t % 3]         background field at time t
#   [1]  u[(t - 1) % 3]   background field at time t - 1
#   [2]  u_sc[t % 3]      scattered field at time t
#   [3]  u_sc[(t - 1)%3]  scattered field at time t - 1
#   [4]  psi_z[t % 2]     split-field memory (background, z)
#   [5]  psi_y[t % 2]     split-field memory (background, y)
#   [6]  psi_x[t % 2]     split-field memory (background, x)
#   [7]  zeta_z[t % 2]    auxiliary memory (background, z)
#   [8]  zeta_y[t % 2]    auxiliary memory (background, y)
#   [9]  zeta_x[t % 2]    auxiliary memory (background, x)
#   [10] psi_z_sc[t % 2]  split-field memory (scattered, z)
#   [11] psi_y_sc[t % 2]  split-field memory (scattered, y)
#   [12] psi_x_sc[t % 2]  split-field memory (scattered, x)
#   [13] zeta_z_sc[t % 2] auxiliary memory (scattered, z)
#   [14] zeta_y_sc[t % 2] auxiliary memory (scattered, y)
#   [15] zeta_x_sc[t % 2] auxiliary memory (scattered, x)
N_STATE = 16
N_STREAMS = 2

# Full wavefield-state layout (N_STATE order, mirrors the C++ checkpoint /
# init_state / final_state layout): the forward callback exposes only the
# physics fields (u, u_sc); return_state / initial_state use all slots.
_WAVEFIELD_NAMES = (
    "u", "u_prev", "u_sc", "u_sc_prev",
    "psi_z", "psi_y", "psi_x", "zeta_z", "zeta_y", "zeta_x",
    "psi_z_sc", "psi_y_sc", "psi_x_sc", "zeta_z_sc", "zeta_y_sc", "zeta_x_sc",
)
_CALLBACK_FIELDS = ("u", "u_sc")


class Scalar3DBornFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v_p,             # [n_shots, nz, ny, nx] or [1, nz, ny, nx] padded model
        scatter_p,       # [n_shots, nz, ny, nx] or [1, nz, ny, nx] padded scatter
        f_bg,            # [nt, n_shots, n_src] pre-scaled background sources
        f_sc,            # [nt, n_shots, n_src] pre-scaled scattered sources
        src_i,           # [n_shots, n_src] flat indices
        rec_i,           # [n_shots, n_rec] flat indices (scattered field)
        bg_rec_i,        # [n_shots, n_bg_rec] flat indices (background field)
        profs,           # [az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx]
        c1, c2,          # FD coefficient arrays (regular grid)
        rdz, rdy, rdx, rdz2, rdy2, rdx2, dt2,
        nt,
        pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,          # forward PML
        pml_z0_b, pml_z1_b, pml_y0_b, pml_y1_b,                  # backward PML
        pml_x0_b, pml_x1_b,
        v_batched, scatter_batched, grad_stride, fd_pad,
        storage,         # SnapshotStorage (bg Laplacian) or None (forward-only)
        storage_sc,      # SnapshotStorage (scattered Laplacian) or None
        ckpt_state,      # [n_ckpt, N_STATE, n_shots, nz, ny, nx] or None
        checkpoint_every,  # 0 = full storage; N = checkpoint every N steps
        segments,        # [(s0, s1)] replay segments; [] = full storage
        init_state,      # [N_STATE, n_shots, nz, ny, nx] initial wavefield or None
        final_state,     # [N_STATE, n_shots, nz, ny, nx] output buffer or None
        forward_callback,  # cb(t, nt, u, u_sc) or None
        callback_frequency,  # call the callback every N steps
    ):
        ext = _ext
        device = v_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = v_p.dtype
        # n_shots from survey, not model batch (shared [1, nz, ny, nx] ok).
        n_shots = int(src_i.shape[0])
        nz, ny, nx = v_p.shape[-3:]
        n_rec = rec_i.shape[1]
        n_bg_rec = bg_rec_i.shape[1]

        u = [
            torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            for _ in range(3)
        ]
        u_sc = [torch.zeros_like(u[0]) for _ in range(3)]
        psi_z = [torch.zeros_like(u[0]) for _ in range(2)]
        psi_y = [torch.zeros_like(u[0]) for _ in range(2)]
        psi_x = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_z = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_y = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_x = [torch.zeros_like(u[0]) for _ in range(2)]
        psi_z_sc = [torch.zeros_like(u[0]) for _ in range(2)]
        psi_y_sc = [torch.zeros_like(u[0]) for _ in range(2)]
        psi_x_sc = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_z_sc = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_y_sc = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_x_sc = [torch.zeros_like(u[0]) for _ in range(2)]
        r = torch.zeros(nt, n_shots, n_rec, device=device, dtype=dtype)
        r_bg = torch.zeros(nt, n_shots, n_bg_rec, device=device, dtype=dtype)
        nz_ny_nx = nz * ny * nx

        az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx = [p.contiguous() for p in profs]
        w_store = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        wsc_store = torch.zeros_like(w_store)
        if storage is not None:
            w_store = storage.snap
        if storage_sc is not None:
            wsc_store = storage_sc.snap
        # checkpointed forward stores no snapshots (the backward replay
        # regenerates them); full storage writes every sampled step.
        store = 0 if segments else (1 if storage is not None else 0)
        ext.forward_loop(
            v_p, scatter_p,
            u, u_sc,
            psi_z, psi_y, psi_x, zeta_z, zeta_y, zeta_x,
            psi_z_sc, psi_y_sc, psi_x_sc, zeta_z_sc, zeta_y_sc, zeta_x_sc,
            az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx,
            w_store, wsc_store,
            f_bg, f_sc, src_i, r_bg, r, bg_rec_i, rec_i,
            c1, c2,
            rdz, rdy, rdx, rdz2, rdy2, rdx2, dt2,
            nt, grad_stride,
            pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
            v_batched, scatter_batched, store, fd_pad,
            checkpoint_every, ckpt_state,
            init_state, final_state, forward_callback, callback_frequency,
        )

        ctx.ext = ext
        ctx.save_for_backward(
            v_p, scatter_p, f_bg, f_sc, src_i, rec_i, bg_rec_i,
            w_store, wsc_store, c1, c2,
        )
        ctx.n_shots = n_shots
        ctx.nz_ny_nx = nz_ny_nx
        ctx.rdz, ctx.rdy, ctx.rdx = rdz, rdy, rdx
        ctx.rdz2, ctx.rdy2, ctx.rdx2, ctx.dt2 = rdz2, rdy2, rdx2, dt2
        ctx.nt = nt
        ctx.pml_z0, ctx.pml_z1 = pml_z0, pml_z1
        ctx.pml_y0, ctx.pml_y1 = pml_y0, pml_y1
        ctx.pml_x0, ctx.pml_x1 = pml_x0, pml_x1
        ctx.pml_z0_b, ctx.pml_z1_b = pml_z0_b, pml_z1_b
        ctx.pml_y0_b, ctx.pml_y1_b = pml_y0_b, pml_y1_b
        ctx.pml_x0_b, ctx.pml_x1_b = pml_x0_b, pml_x1_b
        ctx.v_batched = v_batched
        ctx.scatter_batched = scatter_batched
        ctx.grad_stride = grad_stride
        ctx.profs = profs
        ctx.fd_pad = fd_pad
        ctx.storage = storage
        ctx.storage_sc = storage_sc
        ctx.ckpt_state = ckpt_state
        ctx.checkpoint_every = checkpoint_every
        ctx.segments = segments
        return r, r_bg

    @staticmethod
    def backward(ctx, grad_r, grad_r_bg):
        ext = ctx.ext
        (
            v_p, scatter_p, f_bg, f_sc, src_i, rec_i, bg_rec_i,
            w_store, wsc_store, c1, c2,
        ) = ctx.saved_tensors
        if ctx.storage is None:
            raise RuntimeError(
                "scalar3d_born backward() requires snapshot storage: run the "
                "forward with an input requiring grad (and not under "
                "torch.no_grad())."
            )
        device = v_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = v_p.dtype
        n_shots = ctx.n_shots
        nz, ny, nx = v_p.shape[-3:]
        n_src, n_rec, n_bg_rec = src_i.shape[1], rec_i.shape[1], bg_rec_i.shape[1]
        nt = ctx.nt
        grad_stride = ctx.grad_stride

        if grad_r is None:
            grad_r = torch.zeros(nt, n_shots, n_rec, device=device, dtype=dtype)
        if grad_r_bg is None:
            grad_r_bg = torch.zeros(nt, n_shots, n_bg_rec, device=device, dtype=dtype)
        # grad_r arrives from autograd and may be a non-contiguous broadcast
        # view; the kernels use flat row-major indexing, so materialise it.
        grad_r = grad_r.contiguous()
        grad_r_bg = grad_r_bg.contiguous()
        grad_f_bg = torch.zeros(nt, n_shots, n_src, device=device, dtype=dtype)
        grad_f_sc = torch.zeros(nt, n_shots, n_src, device=device, dtype=dtype)
        grad_v = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        grad_scatter = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)

        lam_bg = [
            torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            for _ in range(3)
        ]
        lam_sc = [torch.zeros_like(lam_bg[0]) for _ in range(3)]
        psi_z = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        psi_y = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        psi_x = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        zeta_z = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        zeta_y = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        zeta_x = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        psi_z_sc = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        psi_y_sc = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        psi_x_sc = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        zeta_z_sc = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        zeta_y_sc = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        zeta_x_sc = [torch.zeros_like(lam_bg[0]) for _ in range(2)]

        az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx = [
            p.contiguous() for p in ctx.profs
        ]
        # integral sampling: each snapshot represents
        # `grad_stride` time steps of the model-gradient integral.
        scale = float(grad_stride)
        segments = ctx.segments
        if segments:
            # Checkpointed backward replays each segment's forward pass in
            # C++; these rings hold the replayed wavefield state.
            u = [
                torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
                for _ in range(3)
            ]
            u_sc = [torch.zeros_like(u[0]) for _ in range(3)]
            psi_z_f = [torch.zeros_like(u[0]) for _ in range(2)]
            psi_y_f = [torch.zeros_like(u[0]) for _ in range(2)]
            psi_x_f = [torch.zeros_like(u[0]) for _ in range(2)]
            zeta_z_f = [torch.zeros_like(u[0]) for _ in range(2)]
            zeta_y_f = [torch.zeros_like(u[0]) for _ in range(2)]
            zeta_x_f = [torch.zeros_like(u[0]) for _ in range(2)]
            psi_z_sc_f = [torch.zeros_like(u[0]) for _ in range(2)]
            psi_y_sc_f = [torch.zeros_like(u[0]) for _ in range(2)]
            psi_x_sc_f = [torch.zeros_like(u[0]) for _ in range(2)]
            zeta_z_sc_f = [torch.zeros_like(u[0]) for _ in range(2)]
            zeta_y_sc_f = [torch.zeros_like(u[0]) for _ in range(2)]
            zeta_x_sc_f = [torch.zeros_like(u[0]) for _ in range(2)]
            segments_t = torch.tensor(segments, dtype=torch.int64)
        else:
            u, u_sc = [], []
            psi_z_f, psi_y_f, psi_x_f = [], [], []
            zeta_z_f, zeta_y_f, zeta_x_f = [], [], []
            psi_z_sc_f, psi_y_sc_f, psi_x_sc_f = [], [], []
            zeta_z_sc_f, zeta_y_sc_f, zeta_x_sc_f = [], [], []
            segments_t = torch.empty(0, 2, dtype=torch.int64)
        ext.adjoint_loop(
            v_p, scatter_p,
            lam_bg, lam_sc,
            psi_z, psi_y, psi_x, zeta_z, zeta_y, zeta_x,
            psi_z_sc, psi_y_sc, psi_x_sc, zeta_z_sc, zeta_y_sc, zeta_x_sc,
            az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx,
            w_store, wsc_store, grad_v, grad_scatter,
            grad_r_bg, grad_r, bg_rec_i, rec_i,
            grad_f_bg, grad_f_sc, src_i,
            f_bg, f_sc,
            u, u_sc,
            psi_z_f, psi_y_f, psi_x_f, zeta_z_f, zeta_y_f, zeta_x_f,
            psi_z_sc_f, psi_y_sc_f, psi_x_sc_f,
            zeta_z_sc_f, zeta_y_sc_f, zeta_x_sc_f,
            c1, c2,
            ctx.rdz, ctx.rdy, ctx.rdx, ctx.rdz2, ctx.rdy2, ctx.rdx2,
            scale, ctx.dt2,
            nt, grad_stride,
            ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
            ctx.pml_x0, ctx.pml_x1,
            ctx.pml_z0_b, ctx.pml_z1_b, ctx.pml_y0_b, ctx.pml_y1_b,
            ctx.pml_x0_b, ctx.pml_x1_b,
            ctx.v_batched, ctx.scatter_batched, ctx.fd_pad,
            segments_t, ctx.ckpt_state,
        )

        if not ctx.v_batched:
            grad_v = grad_v.sum(0, keepdim=True)
        if not ctx.scatter_batched:
            grad_scatter = grad_scatter.sum(0, keepdim=True)
        # 43 forward() inputs: 4 grads + 39 x None
        return (
            grad_v,
            grad_scatter,
            grad_f_bg,      # grads w.r.t. pre-scaled f_bg/f_sc; scaled to amp
            grad_f_sc,      # in scalar3d_born()
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None,
        )


def scalar3d_born(
    v,
    scatter,
    grid_spacing,
    dt,
    source_amplitudes=None,
    source_locations=None,
    receiver_locations=None,
    bg_receiver_locations=None,
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
    """3D acoustic Born forward + adjoint (torch in/out).

    ``v`` is the background wavespeed, ``scatter`` the scattering potential
    (padded with zeros).  Sources are scaled as:
    ``-amp * v^2 dt^2`` for the background field and
    ``-2 * amp * v * scatter * dt^2`` for the scattered field; receivers
    record the pre-update fields.

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
            wavefields via ``state.get_wavefield(name, view)`` (fields ``u``,
            ``u_sc``) — useful for RTM imaging conditions, illumination
            accumulation, monitoring.
        callback_frequency: call ``forward_callback`` every N time steps.
        return_state: if True, return the final padded wavefield state as a
            dict of N_STATE fields (``u``, ``u_prev``, ``u_sc``,
            ``u_sc_prev``, ``psi_*``, ``zeta_*`` for bg and scattered)
            suitable for continuation via ``initial_state``.  With
            ``bg_receiver_locations`` the return becomes ``(r, r_bg, state)``,
            else ``(r, state)``.
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

    Returns ``(receiver_amplitudes, bg_receiver_amplitudes)``
    ``[nt, n_shots, n_rec]`` when ``bg_receiver_locations`` is given, else
    just ``receiver_amplitudes``.
    """
    if not isinstance(grid_spacing, (list, tuple)):
        grid_spacing = [float(grid_spacing)] * 3
    grid_spacing = [float(g) for g in grid_spacing]
    if len(grid_spacing) != 3:
        raise ValueError("grid_spacing must be a scalar or length 3 [dz, dy, dx].")
    pml_w = set_pml_width(pml_width, 3)
    fd_pad = [accuracy // 2] * 6
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

    (v_p, scatter_p), src_i, rec_i = extract_survey_3d(
        [v, scatter],
        source_locations,
        receiver_locations,
        fd_pad,
        pml_w,
        n_shots,
        device,
        dtype,
        pad_modes=["replicate", "constant"],
    )
    # Shared models stay [1, nz, ny, nx]; kernels use *_batched=0 and
    # backward sums per-shot grads (elastic/EM style).
    nz, ny, nx = v_p.shape[-3:]
    max_vel = float(v.detach().abs().max())
    check_cfl(grid_spacing, dt, max_vel, "scalar3d_born")
    profs = set_acoustic_pml_profiles(
        pml_w, fd_pad, dt, grid_spacing, max_vel, pml_freq, (nz, ny, nx), dtype,
        device, accuracy=accuracy,
    )

    if bg_receiver_locations is not None:
        _, _, bg_rec_i = extract_survey_3d(
            [v, scatter],
            None,
            bg_receiver_locations,
            fd_pad,
            pml_w,
            n_shots,
            device,
            dtype,
            pad_modes=["replicate", "constant"],
        )
    else:
        bg_rec_i = torch.empty(n_shots, 0, dtype=torch.int64, device=device)

    amp = prepare_source_amplitudes(
        source_amplitudes, n_shots, src_i.shape[1], nt_inner,
        device=device, dtype=dtype,
    )
    if amp.numel() > 0:
        src_mask = src_i != -1
        src_i_masked = src_i.masked_fill(~src_mask, 0)
        # expand flat rows for gather only (not the full model into Func).
        v_flat = v_p.reshape(-1, nz * ny * nx).expand(n_shots, -1)
        sc_flat = scatter_p.reshape(-1, nz * ny * nx).expand(n_shots, -1)
        v_at_src = v_flat.gather(1, src_i_masked)
        sc_at_src = sc_flat.gather(1, src_i_masked)
        f_bg = (
            -amp.permute(2, 0, 1) * (v_at_src.unsqueeze(0) ** 2 * dt * dt)
        ).contiguous()
        f_sc = (
            -amp.permute(2, 0, 1)
            * (2 * v_at_src.unsqueeze(0) * sc_at_src.unsqueeze(0) * dt * dt)
        ).contiguous()
    else:
        f_bg = torch.empty(0, device=device, dtype=dtype)
        f_sc = torch.empty(0, device=device, dtype=dtype)

    c1 = diff1_coeffs(accuracy, dtype, device)
    c2 = diff2_coeffs(accuracy, dtype, device)
    rdz, rdy, rdx = (1.0 / g for g in grid_spacing)
    rdz2, rdy2, rdx2 = rdz * rdz, rdy * rdy, rdx * rdx
    dt2 = float(dt) * float(dt)
    # the exact PML boundaries (see csrc/scalar3d.cu header comment).
    pml_z0 = min(pml_w[0] + 2 * fd_pad[0], nz - fd_pad[0])
    pml_z1 = max(pml_z0, nz - pml_w[1] - 2 * fd_pad[1])
    pml_y0 = min(pml_w[2] + 2 * fd_pad[2], ny - fd_pad[2])
    pml_y1 = max(pml_y0, ny - pml_w[3] - 2 * fd_pad[3])
    pml_x0 = min(pml_w[4] + 2 * fd_pad[4], nx - fd_pad[4])
    pml_x1 = max(pml_x0, nx - pml_w[5] - 2 * fd_pad[5])
    # the backward pass widens the PML region by one fd_pad (3*fd_pad
    # total): the transpose of the forward PML stencil reads one cell further
    # into the interior, so the adjoint CPML branch must cover those cells.
    pml_z0_b = min(pml_w[0] + 3 * fd_pad[0], nz - fd_pad[0])
    pml_z1_b = max(pml_z0_b, nz - pml_w[1] - 3 * fd_pad[1])
    pml_y0_b = min(pml_w[2] + 3 * fd_pad[2], ny - fd_pad[2])
    pml_y1_b = max(pml_y0_b, ny - pml_w[3] - 3 * fd_pad[3])
    pml_x0_b = min(pml_w[4] + 3 * fd_pad[4], nx - fd_pad[4])
    pml_x1_b = max(pml_x0_b, nx - pml_w[5] - 3 * fd_pad[5])
    v_batched = 1 if (v.ndim == 4 and v.shape[0] == n_shots and v.shape[0] > 1) else 0
    scatter_batched = (
        1
        if (scatter.ndim == 4 and scatter.shape[0] == n_shots and scatter.shape[0] > 1)
        else 0
    )

    storage_mode = resolve_storage(storage)
    grad_stride = check_sample_steps(sample_steps)
    storage_enabled = storage_mode == "auto" and torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (v, scatter, source_amplitudes)
    )
    checkpoint_every, segments, n_snap, n_ckpt = storage_plan(
        nt_inner, N_STATE, grad_stride, N_STREAMS, storage_enabled,
        ckpt_steps=ckpt_steps,
    )
    store_obj = None
    store_sc_obj = None
    ckpt_state = None
    if storage_enabled:
        # The C++ SnapshotStore is shaped [n_shots, ny, nx]; passing
        # ny=nz*ny keeps the flat per-shot stride nz*ny*nx (the kernels index
        # the snapshot buffer flatly, so the 2D-shaped storage view is fine).
        # One store per snapshot stream (bg and scattered Laplacians).
        store_obj = SnapshotStorage(
            _ext, n_snap, n_shots, nz * ny, nx, dtype, device,
        )
        store_sc_obj = SnapshotStorage(
            _ext, n_snap, n_shots, nz * ny, nx, dtype, device,
        )
        if n_ckpt > 0:
            ckpt_state = torch.zeros(
                n_ckpt, N_STATE, n_shots, nz, ny, nx,
                device=device, dtype=dtype,
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

    r, r_bg = Scalar3DBornFunc.apply(
        v_p, scatter_p, f_bg, f_sc, src_i, rec_i, bg_rec_i,
        profs, c1, c2,
        rdz, rdy, rdx, rdz2, rdy2, rdx2, dt2,
        nt_inner,
        pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
        pml_z0_b, pml_z1_b, pml_y0_b, pml_y1_b, pml_x0_b, pml_x1_b,
        v_batched, scatter_batched, grad_stride, fd_pad[0],
        store_obj, store_sc_obj, ckpt_state, checkpoint_every, segments,
        init_state, final_state, cb, callback_frequency,
    )

    if return_state:
        state = unpack_state(final_state, _WAVEFIELD_NAMES)
        if bg_receiver_locations is not None:
            return r, r_bg, state
        return r, state
    if bg_receiver_locations is not None:
        return r, r_bg
    return r
