"""nami scalar2d Born: 2D acoustic Born FDTD with native CUDA adjoint.

Direct translation of the compiled scalar_born backend (bit-for-bit
reproducible in float64): a background wavefield is propagated as in :mod:`scalar2d`,
and a scattered wavefield is driven by the source
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

import nami_born as _ext
import torch

from ..common.fd import diff1_coeffs, diff2_coeffs
from ..common.pml import set_acoustic_pml_profiles
from ..common.storage import (
    SnapshotStorage,
    check_sample_steps,
    resolve_storage,
    storage_plan,
)
from ..common.survey import extract_survey_2d
from .scalar2d import _set_pml_width

# Checkpoint state layout for the Born wavefield (saved at t before step t):
#   [0]  u[t % 3]         background field at time t
#   [1]  u[(t - 1) % 3]   background field at time t - 1
#   [2]  u_sc[t % 3]      scattered field at time t
#   [3]  u_sc[(t - 1)%3]  scattered field at time t - 1
#   [4]  psi_y[t % 2]     split-field memory (background, y)
#   [5]  psi_x[t % 2]     split-field memory (background, x)
#   [6]  zeta_y[t % 2]    auxiliary memory (background, y)
#   [7]  zeta_x[t % 2]    auxiliary memory (background, x)
#   [8]  psi_y_sc[t % 2]  split-field memory (scattered, y)
#   [9]  psi_x_sc[t % 2]  split-field memory (scattered, x)
#   [10] zeta_y_sc[t % 2] auxiliary memory (scattered, y)
#   [11] zeta_x_sc[t % 2] auxiliary memory (scattered, x)
N_STATE = 12
N_STREAMS = 2



def _replicate_pad_2d(model, top, bottom, left, right, device, dtype):
    """Replicate-pad ``[ny, nx]``/``[n, ny, nx]`` with a deterministic backward.

    Produces exactly the values of ``F.pad(mode="replicate")`` (edge copies)
    but builds the gradient from ``cat``/``expand`` (deterministic reductions)
    instead of torch's atomicAdd-based ``replication_pad2d_backward`` CUDA
    kernel, so gradients through the model padding are reproducible.
    """
    m = model.to(device=device, dtype=dtype)
    if m.ndim == 2:
        m = m[None]
    if top or bottom:
        m = torch.cat(
            [m[:, :1].expand(-1, top, -1), m, m[:, -1:].expand(-1, bottom, -1)],
            dim=1,
        )
    if left or right:
        m = torch.cat(
            [
                m[:, :, :1].expand(-1, -1, left),
                m,
                m[:, :, -1:].expand(-1, -1, right),
            ],
            dim=2,
        )
    return m


class Scalar2DBornFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v_p,             # [n_shots, ny, nx] or [1, ny, nx] padded model
        scatter_p,       # [n_shots, ny, nx] or [1, ny, nx] padded scatter
        f_bg,            # [nt, n_shots, n_src] pre-scaled background sources
        f_sc,            # [nt, n_shots, n_src] pre-scaled scattered sources
        src_i,           # [n_shots, n_src] flat indices
        rec_i,           # [n_shots, n_rec] flat indices (scattered field)
        bg_rec_i,        # [n_shots, n_bg_rec] flat indices (background field)
        profs,           # [ay, by, dbydy, ax, bx, dbxdx]
        c1, c2,          # FD coefficient arrays (regular grid)
        rdy, rdx, rdy2, rdx2, dt2,
        nt, pml_y0, pml_y1, pml_x0, pml_x1,
        v_batched, scatter_batched, grad_stride, fd_pad,
        storage,         # SnapshotStorage (bg Laplacian) or None (forward-only)
        storage_sc,      # SnapshotStorage (scattered Laplacian) or None
        ckpt_state,      # [n_ckpt, N_STATE, n_shots, ny, nx] or None
        checkpoint_every,  # 0 = full storage; N = checkpoint every N steps
        segments,        # [(s0, s1)] replay segments; [] = full storage
    ):
        ext = _ext
        device = v_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = v_p.dtype
        n_shots, ny, nx = v_p.shape
        n_src = src_i.shape[1]
        n_rec = rec_i.shape[1]
        n_bg_rec = bg_rec_i.shape[1]

        u = [torch.zeros(n_shots, ny, nx, device=device, dtype=dtype) for _ in range(3)]
        u_sc = [torch.zeros_like(u[0]) for _ in range(3)]
        psi_y = [torch.zeros_like(u[0]) for _ in range(2)]
        psi_x = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_y = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_x = [torch.zeros_like(u[0]) for _ in range(2)]
        psi_y_sc = [torch.zeros_like(u[0]) for _ in range(2)]
        psi_x_sc = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_y_sc = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_x_sc = [torch.zeros_like(u[0]) for _ in range(2)]
        r = torch.zeros(nt, n_shots, n_rec, device=device, dtype=dtype)
        r_bg = torch.zeros(nt, n_shots, n_bg_rec, device=device, dtype=dtype)
        ny_nx = ny * nx

        ay, by, dbydy, ax, bx, dbxdx = [p.contiguous() for p in profs]
        ckpt = ckpt_state is not None
        w_store = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        wsc_store = torch.zeros_like(w_store)
        if storage is not None:
            w_store = storage.snap
        if storage_sc is not None:
            wsc_store = storage_sc.snap
        for t in range(nt):
            if ckpt and t > 0 and t % checkpoint_every == 0:
                k = t // checkpoint_every - 1
                ckpt_state[k, 0].copy_(u[t % 3])
                ckpt_state[k, 1].copy_(u[(t - 1) % 3])
                ckpt_state[k, 2].copy_(u_sc[t % 3])
                ckpt_state[k, 3].copy_(u_sc[(t - 1) % 3])
                ckpt_state[k, 4].copy_(psi_y[t % 2])
                ckpt_state[k, 5].copy_(psi_x[t % 2])
                ckpt_state[k, 6].copy_(zeta_y[t % 2])
                ckpt_state[k, 7].copy_(zeta_x[t % 2])
                ckpt_state[k, 8].copy_(psi_y_sc[t % 2])
                ckpt_state[k, 9].copy_(psi_x_sc[t % 2])
                ckpt_state[k, 10].copy_(zeta_y_sc[t % 2])
                ckpt_state[k, 11].copy_(zeta_x_sc[t % 2])
            store = 0 if segments else 1 if storage is not None else 0
            ext.forward_step(
                v_p, scatter_p,
                u[t % 3], u[(t - 1) % 3], u_sc[t % 3], u_sc[(t - 1) % 3],
                psi_y[t % 2], psi_x[t % 2], zeta_y[t % 2], zeta_x[t % 2],
                psi_y_sc[t % 2], psi_x_sc[t % 2], zeta_y_sc[t % 2], zeta_x_sc[t % 2],
                u[(t + 1) % 3], u_sc[(t + 1) % 3],
                psi_y[(t + 1) % 2], psi_x[(t + 1) % 2],
                zeta_y[(t + 1) % 2], zeta_x[(t + 1) % 2],
                psi_y_sc[(t + 1) % 2], psi_x_sc[(t + 1) % 2],
                zeta_y_sc[(t + 1) % 2], zeta_x_sc[(t + 1) % 2],
                ay, by, dbydy, ax, bx, dbxdx, w_store, wsc_store,
                c1, c2,
                rdy, rdx, rdy2, rdx2, t, grad_stride, dt2,
                n_shots, ny, nx,
                pml_y0, pml_y1, pml_x0, pml_x1,
                v_batched, scatter_batched, store, fd_pad,
            )
            if n_src > 0:
                ext.inject(
                    u[(t + 1) % 3], u_sc[(t + 1) % 3], f_bg, f_sc, src_i,
                    t, n_shots, n_src, ny_nx,
                )
            if n_rec > 0 or n_bg_rec > 0:
                ext.record(
                    u[t % 3], u_sc[t % 3], r_bg, r, bg_rec_i, rec_i,
                    t, n_shots, n_bg_rec, n_rec, ny_nx,
                )

        ctx.ext = ext
        ctx.save_for_backward(
            v_p, scatter_p, f_bg, f_sc, src_i, rec_i, bg_rec_i,
            w_store, wsc_store, c1, c2,
        )
        ctx.ny_nx = ny_nx
        ctx.rdy, ctx.rdx, ctx.rdy2, ctx.rdx2, ctx.dt2 = rdy, rdx, rdy2, rdx2, dt2
        ctx.nt = nt
        ctx.pml_y0, ctx.pml_y1, ctx.pml_x0, ctx.pml_x1 = (
            pml_y0, pml_y1, pml_x0, pml_x1,
        )
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
    def _replay_segment(ext, v_p, scatter_p, f_bg, f_sc, src_i, n_shots,
                        n_src, u, u_sc, psi_y, psi_x, zeta_y, zeta_x,
                        psi_y_sc, psi_x_sc, zeta_y_sc, zeta_x_sc, ay, by,
                        dbydy, ax, bx, dbxdx, w_store, wsc_store, c1, c2,
                        rdy, rdx, rdy2, rdx2, grad_stride, dt2, pml_y0,
                        pml_y1, pml_x0, pml_x1, v_batched, scatter_batched,
                        ny, nx, fd_pad, ny_nx, s0, s1):
        """Re-run forward steps [s0, s1), writing the w snapshots.

        The 2D Born kernels index the snapshot buffer by ``t / grad_stride``
        directly, so the replay passes the segment-local step number and the
        snapshots land at local offsets inside the segment-sized store.
        """
        for t in range(s0, s1):
            t_local = t - s0
            ext.forward_step(
                v_p, scatter_p,
                u[t % 3], u[(t - 1) % 3], u_sc[t % 3], u_sc[(t - 1) % 3],
                psi_y[t % 2], psi_x[t % 2], zeta_y[t % 2], zeta_x[t % 2],
                psi_y_sc[t % 2], psi_x_sc[t % 2], zeta_y_sc[t % 2], zeta_x_sc[t % 2],
                u[(t + 1) % 3], u_sc[(t + 1) % 3],
                psi_y[(t + 1) % 2], psi_x[(t + 1) % 2],
                zeta_y[(t + 1) % 2], zeta_x[(t + 1) % 2],
                psi_y_sc[(t + 1) % 2], psi_x_sc[(t + 1) % 2],
                zeta_y_sc[(t + 1) % 2], zeta_x_sc[(t + 1) % 2],
                ay, by, dbydy, ax, bx, dbxdx, w_store, wsc_store,
                c1, c2,
                rdy, rdx, rdy2, rdx2, t_local, grad_stride, dt2,
                n_shots, ny, nx,
                pml_y0, pml_y1, pml_x0, pml_x1,
                v_batched, scatter_batched, 1, fd_pad,
            )
            if n_src > 0:
                ext.inject(
                    u[(t + 1) % 3], u_sc[(t + 1) % 3], f_bg, f_sc, src_i,
                    t, n_shots, n_src, ny_nx,
                )

    @staticmethod
    def backward(ctx, grad_r, grad_r_bg):
        ext = ctx.ext
        (
            v_p, scatter_p, f_bg, f_sc, src_i, rec_i, bg_rec_i,
            w_store, wsc_store, c1, c2,
        ) = ctx.saved_tensors
        if ctx.storage is None:
            raise RuntimeError(
                "scalar2d_born backward() requires snapshot storage: run the "
                "forward with an input requiring grad (and not under "
                "torch.no_grad())."
            )
        device = v_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = v_p.dtype
        n_shots, ny, nx = v_p.shape
        n_src, n_rec, n_bg_rec = src_i.shape[1], rec_i.shape[1], bg_rec_i.shape[1]
        nt = ctx.nt
        grad_stride = ctx.grad_stride
        ny_nx = ctx.ny_nx

        if grad_r is None:
            grad_r = torch.zeros(nt, n_shots, n_rec, device=device, dtype=dtype)
        if grad_r_bg is None:
            grad_r_bg = torch.zeros(nt, n_shots, n_bg_rec, device=device, dtype=dtype)
        grad_f_bg = torch.zeros(nt, n_shots, n_src, device=device, dtype=dtype)
        grad_f_sc = torch.zeros(nt, n_shots, n_src, device=device, dtype=dtype)
        grad_v = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        grad_scatter = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)

        lam_bg = [
            torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
            for _ in range(3)
        ]
        lam_sc = [torch.zeros_like(lam_bg[0]) for _ in range(3)]
        psi_y = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        psi_x = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        zeta_y = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        zeta_x = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        psi_y_sc = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        psi_x_sc = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        zeta_y_sc = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
        zeta_x_sc = [torch.zeros_like(lam_bg[0]) for _ in range(2)]

        ay, by, dbydy, ax, bx, dbxdx = [p.contiguous() for p in ctx.profs]
        scale = 1.0
        segments = ctx.segments
        if segments:
            # Checkpointed backward: per segment, restore the wavefield
            # state, replay the forward steps to regenerate the w
            # snapshots, then run the adjoint steps.  The adjoint state
            # (lam + memory variables) carries across segments.
            u = [
                torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
                for _ in range(3)
            ]
            u_sc = [torch.zeros_like(u[0]) for _ in range(3)]
            psi_y_f = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
            psi_x_f = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
            zeta_y_f = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
            zeta_x_f = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
            psi_y_sc_f = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
            psi_x_sc_f = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
            zeta_y_sc_f = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
            zeta_x_sc_f = [torch.zeros_like(lam_bg[0]) for _ in range(2)]
            ckpt = ctx.ckpt_state
            for k in range(len(segments) - 1, -1, -1):
                s0, s1 = segments[k]
                if s0 > 0:
                    c = ckpt[k - 1]
                    u[s0 % 3].copy_(c[0])
                    u[(s0 - 1) % 3].copy_(c[1])
                    u_sc[s0 % 3].copy_(c[2])
                    u_sc[(s0 - 1) % 3].copy_(c[3])
                    psi_y_f[s0 % 2].copy_(c[4])
                    psi_x_f[s0 % 2].copy_(c[5])
                    zeta_y_f[s0 % 2].copy_(c[6])
                    zeta_x_f[s0 % 2].copy_(c[7])
                    psi_y_sc_f[s0 % 2].copy_(c[8])
                    psi_x_sc_f[s0 % 2].copy_(c[9])
                    zeta_y_sc_f[s0 % 2].copy_(c[10])
                    zeta_x_sc_f[s0 % 2].copy_(c[11])
                else:
                    for buf in u:
                        buf.zero_()
                    for buf in u_sc:
                        buf.zero_()
                    for bufs in (
                        psi_y_f, psi_x_f, zeta_y_f, zeta_x_f,
                        psi_y_sc_f, psi_x_sc_f, zeta_y_sc_f, zeta_x_sc_f,
                    ):
                        for buf in bufs:
                            buf.zero_()
                Scalar2DBornFunc._replay_segment(
                    ext, v_p, scatter_p, f_bg, f_sc, src_i, n_shots, n_src,
                    u, u_sc, psi_y_f, psi_x_f, zeta_y_f, zeta_x_f,
                    psi_y_sc_f, psi_x_sc_f, zeta_y_sc_f, zeta_x_sc_f,
                    ay, by, dbydy, ax, bx, dbxdx, w_store, wsc_store,
                    c1, c2, ctx.rdy, ctx.rdx, ctx.rdy2, ctx.rdx2, grad_stride,
                    ctx.dt2, ctx.pml_y0, ctx.pml_y1, ctx.pml_x0, ctx.pml_x1,
                    ctx.v_batched, ctx.scatter_batched, ny, nx, ctx.fd_pad,
                    ny_nx, s0, s1,
                )
                for t in range(s1 - 1, s0 - 1, -1):
                    if n_src > 0:
                        ext.record_grad_f(
                            lam_bg[(t + 1) % 3], lam_sc[(t + 1) % 3],
                            grad_f_bg, grad_f_sc, src_i, t, n_shots, n_src,
                            ny_nx,
                        )
                    t_local = t - s0
                    ext.adjoint_step(
                        v_p, scatter_p,
                        lam_bg[(t + 1) % 3], lam_bg[(t + 2) % 3],
                        lam_sc[(t + 1) % 3], lam_sc[(t + 2) % 3],
                        lam_bg[t % 3], lam_sc[t % 3],
                        psi_y[t % 2], psi_x[t % 2], zeta_y[t % 2], zeta_x[t % 2],
                        psi_y_sc[t % 2], psi_x_sc[t % 2],
                        zeta_y_sc[t % 2], zeta_x_sc[t % 2],
                        psi_y[(t + 1) % 2], psi_x[(t + 1) % 2],
                        zeta_y[(t + 1) % 2], zeta_x[(t + 1) % 2],
                        psi_y_sc[(t + 1) % 2], psi_x_sc[(t + 1) % 2],
                        zeta_y_sc[(t + 1) % 2], zeta_x_sc[(t + 1) % 2],
                    ay, by, dbydy, ax, bx, dbxdx, w_store, wsc_store,
                    grad_v, grad_scatter,
                    c1, c2,
                    ctx.rdy, ctx.rdx, ctx.rdy2, ctx.rdx2,
                    t_local, grad_stride, scale, ctx.dt2,
                        n_shots, ny, nx,
                        ctx.pml_y0, ctx.pml_y1, ctx.pml_x0, ctx.pml_x1,
                        ctx.v_batched, ctx.scatter_batched, ctx.fd_pad,
                    )
                    if n_rec > 0 or n_bg_rec > 0:
                        ext.record_grad_r(
                            lam_bg[t % 3], lam_sc[t % 3],
                            grad_r_bg, grad_r, bg_rec_i, rec_i,
                            t, n_shots, n_bg_rec, n_rec, ny_nx,
                        )
        else:
            for t in range(nt - 1, -1, -1):
                if n_src > 0:
                    ext.record_grad_f(
                        lam_bg[(t + 1) % 3], lam_sc[(t + 1) % 3],
                        grad_f_bg, grad_f_sc, src_i, t, n_shots, n_src, ny_nx,
                    )
                ext.adjoint_step(
                    v_p, scatter_p,
                    lam_bg[(t + 1) % 3], lam_bg[(t + 2) % 3],
                    lam_sc[(t + 1) % 3], lam_sc[(t + 2) % 3],
                    lam_bg[t % 3], lam_sc[t % 3],
                    psi_y[t % 2], psi_x[t % 2], zeta_y[t % 2], zeta_x[t % 2],
                    psi_y_sc[t % 2], psi_x_sc[t % 2],
                    zeta_y_sc[t % 2], zeta_x_sc[t % 2],
                    psi_y[(t + 1) % 2], psi_x[(t + 1) % 2],
                    zeta_y[(t + 1) % 2], zeta_x[(t + 1) % 2],
                    psi_y_sc[(t + 1) % 2], psi_x_sc[(t + 1) % 2],
                    zeta_y_sc[(t + 1) % 2], zeta_x_sc[(t + 1) % 2],
                    ay, by, dbydy, ax, bx, dbxdx, w_store, wsc_store,
                    grad_v, grad_scatter,
                    c1, c2,
                    ctx.rdy, ctx.rdx, ctx.rdy2, ctx.rdx2,
                    t, grad_stride, scale, ctx.dt2,
                    n_shots, ny, nx,
                    ctx.pml_y0, ctx.pml_y1, ctx.pml_x0, ctx.pml_x1,
                    ctx.v_batched, ctx.scatter_batched, ctx.fd_pad,
                )
                if n_rec > 0 or n_bg_rec > 0:
                    ext.record_grad_r(
                        lam_bg[t % 3], lam_sc[t % 3],
                        grad_r_bg, grad_r, bg_rec_i, rec_i,
                        t, n_shots, n_bg_rec, n_rec, ny_nx,
                    )

        # 29 forward() inputs: 4 grads + 25 x None
        return (
            grad_v,
            grad_scatter,
            grad_f_bg,      # grads w.r.t. pre-scaled f_bg/f_sc; scaled to amp
            grad_f_sc,      # in scalar2d_born()
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
        )


def scalar2d_born(
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
    max_vel=None,
    nt=None,
    storage="auto",
    sample_steps=1,
    ckpt_steps=None,
):
    """2D acoustic Born forward + adjoint (torch in/out).

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

    Returns ``(receiver_amplitudes, bg_receiver_amplitudes)``
    ``[nt, n_shots, n_rec]`` when ``bg_receiver_locations`` is given, else
    just ``receiver_amplitudes``.
    """
    if not isinstance(grid_spacing, (list, tuple)):
        grid_spacing = [float(grid_spacing)] * 2
    grid_spacing = [float(d) for d in grid_spacing]
    pml_w = _set_pml_width(pml_width, 2)
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

    (v_p, scatter_p), src_i, rec_i = extract_survey_2d(
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
    # deterministic replicate pad for the velocity model (see
    # _replicate_pad_2d): same values as extract's F.pad, but a
    # reproducible gradient through the model padding.
    v_p = _replicate_pad_2d(
        v,
        fd_pad[0] + pml_w[0],
        fd_pad[1] + pml_w[1],
        fd_pad[2] + pml_w[2],
        fd_pad[3] + pml_w[3],
        device,
        dtype,
    )
    ny, nx = v_p.shape[-2:]
    if max_vel is None:
        max_vel = float(v.detach().abs().max())
    profs = set_acoustic_pml_profiles(
        pml_w, fd_pad, dt, grid_spacing, max_vel, pml_freq, (ny, nx), dtype,
        device, accuracy=accuracy,
    )

    if bg_receiver_locations is not None:
        _, _, bg_rec_i = extract_survey_2d(
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

    if source_amplitudes is not None and source_amplitudes.numel() > 0:
        amp = source_amplitudes.to(device=device, dtype=dtype)[:, :, :nt_inner]
        src_mask = src_i != -1
        src_i_masked = src_i.masked_fill(~src_mask, 0)
        v_flat = v_p.reshape(1, -1).expand(n_shots, -1)
        v_at_src = v_flat.gather(1, src_i_masked)
        sc_flat = scatter_p.reshape(1, -1).expand(n_shots, -1)
        sc_at_src = sc_flat.gather(1, src_i_masked)
        f_bg = (
            -amp.permute(2, 0, 1) * (v_at_src.unsqueeze(0) ** 2 * dt * dt)
        ).contiguous()
        f_sc = (
            -amp.permute(2, 0, 1)
            * (2 * v_at_src.unsqueeze(0) * sc_at_src.unsqueeze(0) * dt * dt)
        ).contiguous()
    else:
        amp = torch.zeros(n_shots, 0, nt_inner, device=device, dtype=dtype)
        f_bg = torch.empty(0, device=device, dtype=dtype)
        f_sc = torch.empty(0, device=device, dtype=dtype)

    c1 = diff1_coeffs(accuracy, dtype, device)
    c2 = diff2_coeffs(accuracy, dtype, device)
    rdy, rdx = 1.0 / grid_spacing[0], 1.0 / grid_spacing[1]
    rdy2, rdx2 = rdy * rdy, rdx * rdx
    dt2 = float(dt) * float(dt)
    pml_y0, pml_y1 = fd_pad[0] + pml_w[0], ny - fd_pad[1] - pml_w[1]
    pml_x0, pml_x1 = fd_pad[2] + pml_w[2], nx - fd_pad[3] - pml_w[3]
    v_batched = 1 if (v.ndim == 3 and v.shape[0] == n_shots and v.shape[0] > 1) else 0
    scatter_batched = (
        1
        if (scatter.ndim == 3 and scatter.shape[0] == n_shots and scatter.shape[0] > 1)
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
        store_obj = SnapshotStorage(
            _ext, n_snap, n_shots, ny, nx, dtype, device,
        )
        store_sc_obj = SnapshotStorage(
            _ext, n_snap, n_shots, ny, nx, dtype, device,
        )
        if n_ckpt > 0:
            ckpt_state = torch.zeros(
                n_ckpt, N_STATE, n_shots, ny, nx, device=device, dtype=dtype,
            )

    r, r_bg = Scalar2DBornFunc.apply(
        v_p, scatter_p, f_bg, f_sc, src_i, rec_i, bg_rec_i,
        profs, c1, c2,
        rdy, rdx, rdy2, rdx2, dt2,
        nt_inner, pml_y0, pml_y1, pml_x0, pml_x1,
        v_batched, scatter_batched, grad_stride, fd_pad[0],
        store_obj, store_sc_obj, ckpt_state, checkpoint_every, segments,
    )

    if bg_receiver_locations is not None:
        return r, r_bg
    return r
