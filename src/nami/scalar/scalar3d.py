"""nami scalar3d: 3D acoustic FDTD with native CUDA adjoint.

Forward/adjoint discretisation follows the reference compiled 3D scalar
backend (bit-for-bit reproducible in float64): CPML (Pasalic &
McGarry) with three split-field memory pairs (psi_z/zeta_z, psi_y/zeta_y,
psi_x/zeta_x), pressure-source convention ``f = -amp * v^2 dt^2`` and
receivers recording the pre-update field.  The spatial FD order is
user-selectable (accuracy 2/4/6/8) with the ``regular_grid``
coefficient tables; the kernels are driven by coefficient arrays (fixed
max radius 4, zero-padded for lower orders).

Kernels are intentionally unoptimised (one launch per step, naive stencil)
— that is the correctness baseline; performance work (persistent kernels,
interior/PML splitting, shared-memory tiling) lands on top of it later.

PML boundary convention: exact boundaries
``pml_z0 = min(pml_width + 2*fd_pad, nz - fd_pad)`` etc. (see the comment
in ``csrc/scalar3d.cu``).  This differs from scalar2d's
``fd_pad + pml_width`` convention so that parity with the reference holds
even
after the wavefront has entered the PML.
"""

import nami_scalar3d as _ext
import torch

from ..common.cfl import check_cfl
from ..common.fd import diff1_coeffs, diff2_coeffs
from ..common.pml import set_acoustic_pml_profiles
from ..common.storage import (
    SnapshotStorage,
    check_sample_steps,
    resolve_storage,
    storage_plan,
)
from ..common.survey import extract_survey_3d

# Checkpoint state layout for the 3D acoustic field:
#   [0] u[t % 3]      field at time t
#   [1] u[(t - 1)%3]  field at time t - 1
#   [2] psi_z[t % 2]  split-field memory (z)
#   [3] psi_y[t % 2]  split-field memory (y)
#   [4] psi_x[t % 2]  split-field memory (x)
#   [5] zeta_z[t % 2] auxiliary memory (z)
#   [6] zeta_y[t % 2] auxiliary memory (y)
#   [7] zeta_x[t % 2] auxiliary memory (x)
N_STATE = 8


class Scalar3DFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v_p,             # [n_shots, nz, ny, nx] or [1, nz, ny, nx] padded model
        f,               # [nt, n_shots, n_src] pre-scaled source amplitudes
        src_i,           # [n_shots, n_src] flat indices
        rec_i,           # [n_shots, n_rec] flat indices
        profs,           # [az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx]
        c1, c2,          # FD coefficient arrays (regular grid)
        rdz, rdy, rdx, rdz2, rdy2, rdx2, dt2,
        nt,
        pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,          # forward PML
        pml_z0_b, pml_z1_b, pml_y0_b, pml_y1_b,                  # backward PML
        pml_x0_b, pml_x1_b,
        v_batched, grad_stride, fd_pad,
        storage,         # SnapshotStorage or None (forward-only)
        ckpt_state,      # [n_ckpt, N_STATE, n_shots, nz, ny, nx] or None
        checkpoint_every,  # 0 = full storage; N = checkpoint every N steps
        segments,        # [(s0, s1)] replay segments; [] = full storage
    ):
        ext = _ext
        device = v_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = v_p.dtype
        n_shots, nz, ny, nx = v_p.shape
        n_src = src_i.shape[1]
        n_rec = rec_i.shape[1]

        u = [
            torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            for _ in range(3)
        ]
        psi_z = [torch.zeros_like(u[0]) for _ in range(2)]
        psi_y = [torch.zeros_like(u[0]) for _ in range(2)]
        psi_x = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_z = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_y = [torch.zeros_like(u[0]) for _ in range(2)]
        zeta_x = [torch.zeros_like(u[0]) for _ in range(2)]
        r = torch.zeros(nt, n_shots, n_rec, device=device, dtype=dtype)
        nz_ny_nx = nz * ny * nx

        az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx = [p.contiguous() for p in profs]
        ckpt = ckpt_state is not None
        w_store = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        if storage is not None:
            w_store = storage.snap
        for t in range(nt):
            if ckpt and t > 0 and t % checkpoint_every == 0:
                k = t // checkpoint_every - 1
                ckpt_state[k, 0].copy_(u[t % 3])
                ckpt_state[k, 1].copy_(u[(t - 1) % 3])
                ckpt_state[k, 2].copy_(psi_z[t % 2])
                ckpt_state[k, 3].copy_(psi_y[t % 2])
                ckpt_state[k, 4].copy_(psi_x[t % 2])
                ckpt_state[k, 5].copy_(zeta_z[t % 2])
                ckpt_state[k, 6].copy_(zeta_y[t % 2])
                ckpt_state[k, 7].copy_(zeta_x[t % 2])
            if segments:
                store = 0
                snap_off = 0
            else:
                store = 1 if storage is not None else 0
                snap_off = storage.snap_offset(t // grad_stride) if storage else 0
            ext.forward_step(
                v_p, u[t % 3], u[(t - 1) % 3],
                psi_z[t % 2], psi_y[t % 2], psi_x[t % 2],
                zeta_z[t % 2], zeta_y[t % 2], zeta_x[t % 2],
                u[(t + 1) % 3],
                psi_z[(t + 1) % 2], psi_y[(t + 1) % 2], psi_x[(t + 1) % 2],
                zeta_z[(t + 1) % 2], zeta_y[(t + 1) % 2], zeta_x[(t + 1) % 2],
                az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx, w_store,
                c1, c2,
                rdz, rdy, rdx, rdz2, rdy2, rdx2, t, grad_stride, dt2,
                pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
                v_batched, store, snap_off, fd_pad,
            )
            if n_src > 0:
                ext.inject(u[(t + 1) % 3], f, src_i, t, n_shots, n_src, nz_ny_nx)
            if n_rec > 0:
                ext.record(u[t % 3], r, rec_i, t, n_shots, n_rec, nz_ny_nx)

        ctx.ext = ext
        ctx.save_for_backward(v_p, f, src_i, rec_i, w_store, c1, c2)
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
        ctx.grad_stride = grad_stride
        ctx.profs = profs
        ctx.fd_pad = fd_pad
        ctx.storage = storage
        ctx.ckpt_state = ckpt_state
        ctx.checkpoint_every = checkpoint_every
        ctx.segments = segments
        return r

    @staticmethod
    def _replay_segment(ext, v_p, f, src_i, n_shots, n_src, u, psi_z, psi_y,
                        psi_x, zeta_z, zeta_y, zeta_x, az, bz, dbzdz, ay, by,
                        dbydy, ax, bx, dbxdx, w_store, c1, c2, rdz, rdy, rdx,
                        rdz2, rdy2, rdx2, grad_stride, dt2, pml_z0, pml_z1,
                        pml_y0, pml_y1, pml_x0, pml_x1, v_batched, fd_pad,
                        nz_ny_nx, shot_count, s0, s1):
        """Re-run forward steps [s0, s1), writing the w snapshots."""
        for t in range(s0, s1):
            snap_off = ((t - s0) // grad_stride) * shot_count
            ext.forward_step(
                v_p, u[t % 3], u[(t - 1) % 3],
                psi_z[t % 2], psi_y[t % 2], psi_x[t % 2],
                zeta_z[t % 2], zeta_y[t % 2], zeta_x[t % 2],
                u[(t + 1) % 3],
                psi_z[(t + 1) % 2], psi_y[(t + 1) % 2], psi_x[(t + 1) % 2],
                zeta_z[(t + 1) % 2], zeta_y[(t + 1) % 2], zeta_x[(t + 1) % 2],
                az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx, w_store,
                c1, c2,
                rdz, rdy, rdx, rdz2, rdy2, rdx2, t, grad_stride, dt2,
                pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
                v_batched, 1, snap_off, fd_pad,
            )
            if n_src > 0:
                ext.inject(u[(t + 1) % 3], f, src_i, t, n_shots, n_src, nz_ny_nx)

    @staticmethod
    def backward(ctx, grad_r):
        ext = ctx.ext
        v_p, f, src_i, rec_i, w_store, c1, c2 = ctx.saved_tensors
        if ctx.storage is None:
            raise RuntimeError(
                "scalar3d backward() requires snapshot storage: run the forward "
                "with an input requiring grad (and not under torch.no_grad())."
            )
        storage = ctx.storage
        device = v_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = v_p.dtype
        # n_shots is the survey shot count: for a non-batched model shared by
        # several shots, v_p is [1, nz, ny, nx] but n_shots > 1.
        n_shots = ctx.n_shots
        nz, ny, nx = v_p.shape[-3:]
        n_src, n_rec = src_i.shape[1], rec_i.shape[1]
        nt = ctx.nt
        grad_stride = ctx.grad_stride
        nz_ny_nx = ctx.nz_ny_nx

        if grad_r is None:
            grad_r = torch.zeros(nt, n_shots, n_rec, device=device, dtype=dtype)
        grad_f = torch.zeros(nt, n_shots, n_src, device=device, dtype=dtype)
        grad_v = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)

        lam = [
            torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            for _ in range(3)
        ]
        psi_z = [torch.zeros_like(lam[0]) for _ in range(2)]
        psi_y = [torch.zeros_like(lam[0]) for _ in range(2)]
        psi_x = [torch.zeros_like(lam[0]) for _ in range(2)]
        zeta_z = [torch.zeros_like(lam[0]) for _ in range(2)]
        zeta_y = [torch.zeros_like(lam[0]) for _ in range(2)]
        zeta_x = [torch.zeros_like(lam[0]) for _ in range(2)]

        az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx = [
            p.contiguous() for p in ctx.profs
        ]
        scale = 1.0
        segments = ctx.segments
        if segments:
            # Checkpointed backward: per segment, restore the wavefield
            # state, replay the forward steps to regenerate the w
            # snapshots, then run the adjoint steps.  The adjoint state
            # (lam + memory variables) carries across segments.
            shot_count = n_shots * nz_ny_nx
            u = [
                torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
                for _ in range(3)
            ]
            psi_z_f = [torch.zeros_like(lam[0]) for _ in range(2)]
            psi_y_f = [torch.zeros_like(lam[0]) for _ in range(2)]
            psi_x_f = [torch.zeros_like(lam[0]) for _ in range(2)]
            zeta_z_f = [torch.zeros_like(lam[0]) for _ in range(2)]
            zeta_y_f = [torch.zeros_like(lam[0]) for _ in range(2)]
            zeta_x_f = [torch.zeros_like(lam[0]) for _ in range(2)]
            ckpt = ctx.ckpt_state
            for k in range(len(segments) - 1, -1, -1):
                s0, s1 = segments[k]
                if s0 > 0:
                    c = ckpt[k - 1]
                    u[s0 % 3].copy_(c[0])
                    u[(s0 - 1) % 3].copy_(c[1])
                    psi_z_f[s0 % 2].copy_(c[2])
                    psi_y_f[s0 % 2].copy_(c[3])
                    psi_x_f[s0 % 2].copy_(c[4])
                    zeta_z_f[s0 % 2].copy_(c[5])
                    zeta_y_f[s0 % 2].copy_(c[6])
                    zeta_x_f[s0 % 2].copy_(c[7])
                else:
                    for buf in u:
                        buf.zero_()
                    for bufs in (
                        psi_z_f, psi_y_f, psi_x_f, zeta_z_f, zeta_y_f,
                        zeta_x_f,
                    ):
                        for buf in bufs:
                            buf.zero_()
                Scalar3DFunc._replay_segment(
                    ext, v_p, f, src_i, n_shots, n_src, u, psi_z_f, psi_y_f,
                    psi_x_f, zeta_z_f, zeta_y_f, zeta_x_f,
                    az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx,
                    w_store, c1, c2, ctx.rdz, ctx.rdy, ctx.rdx, ctx.rdz2,
                    ctx.rdy2, ctx.rdx2, grad_stride, ctx.dt2,
                    ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                    ctx.pml_x0, ctx.pml_x1, ctx.v_batched, ctx.fd_pad,
                    nz_ny_nx, shot_count, s0, s1,
                )
                for t in range(s1 - 1, s0 - 1, -1):
                    if n_src > 0:
                        ext.record_grad_f(
                            lam[(t + 1) % 3], grad_f, src_i, t,
                            n_shots, n_src, nz_ny_nx,
                        )
                    snap_off = ((t - s0) // grad_stride) * shot_count
                    ext.adjoint_step(
                        v_p, lam[(t + 1) % 3], lam[(t + 2) % 3], lam[t % 3],
                        psi_z[t % 2], psi_y[t % 2], psi_x[t % 2],
                        zeta_z[t % 2], zeta_y[t % 2], zeta_x[t % 2],
                        psi_z[(t + 1) % 2], psi_y[(t + 1) % 2],
                        psi_x[(t + 1) % 2],
                        zeta_z[(t + 1) % 2], zeta_y[(t + 1) % 2],
                        zeta_x[(t + 1) % 2],
                        az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx,
                        w_store, grad_v,
                        c1, c2,
                        ctx.rdz, ctx.rdy, ctx.rdx, ctx.rdz2, ctx.rdy2,
                        ctx.rdx2, t, grad_stride, scale, ctx.dt2,
                        ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                        ctx.pml_x0, ctx.pml_x1,
                        ctx.pml_z0_b, ctx.pml_z1_b, ctx.pml_y0_b,
                        ctx.pml_y1_b, ctx.pml_x0_b, ctx.pml_x1_b,
                        ctx.v_batched, snap_off, ctx.fd_pad,
                    )
                    if n_rec > 0:
                        ext.record_grad_r(
                            lam[t % 3], grad_r, rec_i, t, n_shots, n_rec,
                            nz_ny_nx,
                        )
        else:
            for t in range(nt - 1, -1, -1):
                if n_src > 0:
                    ext.record_grad_f(
                        lam[(t + 1) % 3], grad_f, src_i, t,
                        n_shots, n_src, nz_ny_nx,
                    )
                snap_off = storage.snap_offset(t // grad_stride)
                ext.adjoint_step(
                    v_p, lam[(t + 1) % 3], lam[(t + 2) % 3], lam[t % 3],
                    psi_z[t % 2], psi_y[t % 2], psi_x[t % 2],
                    zeta_z[t % 2], zeta_y[t % 2], zeta_x[t % 2],
                    psi_z[(t + 1) % 2], psi_y[(t + 1) % 2],
                    psi_x[(t + 1) % 2],
                    zeta_z[(t + 1) % 2], zeta_y[(t + 1) % 2],
                    zeta_x[(t + 1) % 2],
                    az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx,
                    w_store, grad_v,
                    c1, c2,
                    ctx.rdz, ctx.rdy, ctx.rdx, ctx.rdz2, ctx.rdy2, ctx.rdx2,
                    t, grad_stride, scale, ctx.dt2,
                    ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                    ctx.pml_x0, ctx.pml_x1,
                    ctx.pml_z0_b, ctx.pml_z1_b, ctx.pml_y0_b,
                    ctx.pml_y1_b, ctx.pml_x0_b, ctx.pml_x1_b,
                    ctx.v_batched, snap_off, ctx.fd_pad,
                )
                if n_rec > 0:
                    ext.record_grad_r(
                        lam[t % 3], grad_r, rec_i, t, n_shots, n_rec,
                        nz_ny_nx,
                    )

        # grad_v is per-shot [n_shots, nz, ny, nx]; for a model shared by
        # several shots (v_p is an expand of a [1, ...] model) the autograd
        # graph sums the per-shot gradients back through the expand node.
        # The storage (and its temp files) is released when the autograd graph
        # is garbage collected; it must outlive repeated
        # backward calls, so it is not closed here.
        # 34 forward() inputs: grad_v + grad_f + 32 x None
        return (
            grad_v,
            grad_f,      # grad w.r.t. pre-scaled f; scaled to amp in scalar3d()
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None,
        )


def scalar3d(
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
):
    """3D acoustic forward + adjoint (torch in/out).

    Pressure-source convention: pre-scaled by
    ``-amp * v^2 dt^2``; receivers record the pre-update field.
    ``v`` is [nz, ny, nx] (or [1, ...] / [n_shots, ...]).

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

    Returns receiver amplitudes [nt, n_shots, n_rec].
    """
    if not isinstance(grid_spacing, (list, tuple)):
        grid_spacing = [float(grid_spacing)] * 3
    grid_spacing = [float(g) for g in grid_spacing]
    if len(grid_spacing) != 3:
        raise ValueError("grid_spacing must be a scalar or length 3 [dz, dy, dx].")
    pml_w = _set_pml_width(pml_width, 3)
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

    (v_p,), src_i, rec_i = extract_survey_3d(
        [v],
        source_locations,
        receiver_locations,
        fd_pad,
        pml_w,
        n_shots,
        device,
        dtype,
    )
    # a model shared by several shots is replicated to [n_shots, ...]
    # (the autograd graph then sums per-shot gradients back through the
    # expand node). Without this, the kernels would propagate only shot 0.
    if v_p.shape[0] == 1 and n_shots > 1:
        v_p = v_p.expand(n_shots, -1, -1, -1)
    nz, ny, nx = v_p.shape[-3:]
    max_vel = float(v.detach().abs().max())
    check_cfl(grid_spacing, dt, max_vel, "scalar3d")
    profs = set_acoustic_pml_profiles(
        pml_w, fd_pad, dt, grid_spacing, max_vel, pml_freq, (nz, ny, nx), dtype, device,
        accuracy=accuracy,
    )

    v_at_src = None
    if source_amplitudes is not None and source_amplitudes.numel() > 0:
        amp = source_amplitudes.to(device=device, dtype=dtype)[:, :, :nt_inner]
        src_mask = src_i != -1
        src_i_masked = src_i.masked_fill(~src_mask, 0)
        # per-shot v at each source (exact: [n_shots, nz*ny*nx] rows;
        # flattening all shots into one row would gather shot 0's v for every
        # shot and corrupt the source-scaling contribution to grad_v).
        v_flat = v_p.reshape(-1, nz * ny * nx).expand(n_shots, -1)
        v_at_src = v_flat.gather(1, src_i_masked)
        f = (
            -amp.permute(2, 0, 1) * (v_at_src.unsqueeze(0) ** 2 * dt * dt)
        ).contiguous()
    else:
        amp = torch.zeros(n_shots, 0, nt_inner, device=device, dtype=dtype)
        f = torch.empty(0, device=device, dtype=dtype)

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

    storage_mode = resolve_storage(storage)
    grad_stride = check_sample_steps(sample_steps)
    storage_enabled = storage_mode == "auto" and torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (v, source_amplitudes)
    )
    checkpoint_every, segments, n_snap, n_ckpt = storage_plan(
        nt_inner, N_STATE, grad_stride, 1, storage_enabled,
        ckpt_steps=ckpt_steps,
    )
    store_obj = None
    ckpt_state = None
    if storage_enabled:
        # The C++ SnapshotStore is shaped [n_shots, ny, nx]; passing
        # ny=nz*ny keeps the flat per-shot stride nz*ny*nx (the kernels index
        # the snapshot buffer flatly, so the 2D-shaped storage view is fine).
        store_obj = SnapshotStorage(
            _ext, n_snap, n_shots, nz * ny, nx, dtype, device,
        )
        if n_ckpt > 0:
            ckpt_state = torch.zeros(
                n_ckpt, N_STATE, n_shots, nz, ny, nx,
                device=device, dtype=dtype,
            )

    r = Scalar3DFunc.apply(
        v_p, f, src_i, rec_i, profs, c1, c2,
        rdz, rdy, rdx, rdz2, rdy2, rdx2, dt2,
        nt_inner,
        pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
        pml_z0_b, pml_z1_b, pml_y0_b, pml_y1_b, pml_x0_b, pml_x1_b,
        v_batched, grad_stride, fd_pad[0],
        store_obj, ckpt_state, checkpoint_every, segments,
    )

    # gradients returned by backward() need post-processing (done in a helper
    # so autograd users can call .backward() on `r` and get model grads via
    # the saved `v_p`; here we just return the receiver amplitudes).
    return r


def _set_pml_width(pml_width, ndim):
    if isinstance(pml_width, int):
        return [pml_width] * (2 * ndim)
    pml_width = list(pml_width)
    if len(pml_width) != 2 * ndim:
        raise ValueError(f"pml_width must be int or length {2 * ndim}.")
    return pml_width
