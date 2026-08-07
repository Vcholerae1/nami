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

from ..common.cfl import check_cfl
from ..common.fd import check_accuracy, staggered_diff1_coeffs
from ..common.storage import (
    SnapshotStorage,
    check_sample_steps,
    resolve_storage,
    storage_plan,
)
from ..common.survey import extract_survey_3d

# vacuum permittivity / permeability / speed of light (SI)
EPS0 = 8.8541878128e-12  # F/m
MU0 = 1.2566370614359173e-06  # H/m

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
    ):
        ext = _ext
        device = ca_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = ca_p.dtype
        n_shots, nz, ny, nx = ca_p.shape
        n_src = src_i.shape[1]
        n_rec = rec_i.shape[1]
        numel = nz * ny * nx

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
        ckpt = ckpt_state is not None
        for t in range(nt):
            if ckpt and t > 0 and t % checkpoint_every == 0:
                k = t // checkpoint_every - 1
                ckpt_state[k, 0].copy_(ex)
                ckpt_state[k, 1].copy_(ey)
                ckpt_state[k, 2].copy_(ez)
                ckpt_state[k, 3].copy_(hx)
                ckpt_state[k, 4].copy_(hy)
                ckpt_state[k, 5].copy_(hz)
                ckpt_state[k, 6].copy_(m_ey_z)
                ckpt_state[k, 7].copy_(m_ez_y)
                ckpt_state[k, 8].copy_(m_ez_x)
                ckpt_state[k, 9].copy_(m_ex_z)
                ckpt_state[k, 10].copy_(m_ex_y)
                ckpt_state[k, 11].copy_(m_ey_x)
                ckpt_state[k, 12].copy_(m_hy_z)
                ckpt_state[k, 13].copy_(m_hz_y)
                ckpt_state[k, 14].copy_(m_hz_x)
                ckpt_state[k, 15].copy_(m_hx_z)
                ckpt_state[k, 16].copy_(m_hx_y)
                ckpt_state[k, 17].copy_(m_hy_x)
            if segments:
                snap_off = 0
            else:
                snap_off = (
                    ex_storage.snap_offset(t // grad_stride) if ex_storage else 0
                )
            ext.step_h(
                cq_p, ex, ey, ez, hx, hy, hz,
                m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x,
                azh, bzh, ayh, byh, axh, bxh, kzh, kyh, kxh,
                c,
                rdz, rdy, rdx,
                n_shots, nz, ny, nx,
                pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
                *fd_pad[0::2], cq_batched,
            )
            ext.step_e_storage(
                ca_p, cb_p, hx, hy, hz, ex, ey, ez,
                m_hy_z, m_hz_y, m_hz_x, m_hx_z, m_hx_y, m_hy_x,
                ex_store, ey_store, ez_store,
                curl_x_store, curl_y_store, curl_z_store,
                az, bz, ay, by, ax, bx, kz, ky, kx,
                c,
                rdz, rdy, rdx, t, grad_stride, snap_off,
                n_shots, nz, ny, nx,
                pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
                *fd_pad, ca_batched, cb_batched,
            )
            if n_src > 0:
                ext.inject(
                    (ex, ey, ez)[source_component], f, src_i,
                    t, n_shots, n_src, numel,
                )
            if n_rec > 0:
                ext.record(
                    (ex, ey, ez)[receiver_component], r, rec_i,
                    t, n_shots, n_rec, numel,
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
    def _replay_segment(ctx, ext, ca_p, cb_p, cq_p, f, src_i, n_shots, n_src,
                        ex, ey, ez, hx, hy, hz, m_ey_z, m_ez_y, m_ez_x,
                        m_ex_z, m_ex_y, m_ey_x, m_hy_z, m_hz_y, m_hz_x,
                        m_hx_z, m_hx_y, m_hy_x, ex_store, ey_store, ez_store,
                        curl_x_store, curl_y_store, curl_z_store, numel,
                        nz, ny, nx, s0, s1):
        """Re-run forward steps [s0, s1), writing the w snapshots."""
        az, azh, ay, ayh, ax, axh, bz, bzh, by, byh, bx, bxh, \
            kz, kzh, ky, kyh, kx, kxh = [p.contiguous() for p in ctx.profs]
        c = ctx.c
        for t in range(s0, s1):
            snap_off = ((t - s0) // ctx.grad_stride) * (n_shots * numel)
            ext.step_h(
                cq_p, ex, ey, ez, hx, hy, hz,
                m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x,
                azh, bzh, ayh, byh, axh, bxh, kzh, kyh, kxh,
                c,
                ctx.rdz, ctx.rdy, ctx.rdx,
                n_shots, nz, ny, nx,
                ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                ctx.pml_x0, ctx.pml_x1,
                *ctx.fd_pad[0::2], ctx.cq_batched,
            )
            ext.step_e_storage(
                ca_p, cb_p, hx, hy, hz, ex, ey, ez,
                m_hy_z, m_hz_y, m_hz_x, m_hx_z, m_hx_y, m_hy_x,
                ex_store, ey_store, ez_store,
                curl_x_store, curl_y_store, curl_z_store,
                az, bz, ay, by, ax, bx, kz, ky, kx,
                c,
                ctx.rdz, ctx.rdy, ctx.rdx, t, ctx.grad_stride, snap_off,
                n_shots, nz, ny, nx,
                ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                ctx.pml_x0, ctx.pml_x1,
                *ctx.fd_pad, ctx.ca_batched, ctx.cb_batched,
            )
            if n_src > 0:
                ext.inject(
                    (ex, ey, ez)[ctx.source_component], f, src_i,
                    t, n_shots, n_src, numel,
                )

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
        n_shots, nz, ny, nx = ca_p.shape
        n_src, n_rec = src_i.shape[1], rec_i.shape[1]
        nt = ctx.nt
        numel = nz * ny * nx
        shot_count = n_shots * numel

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
            # Checkpointed backward: per segment, restore the wavefield
            # state, replay the forward steps to regenerate the w
            # snapshots at segment-local offsets, then run the adjoint
            # steps.  The adjoint state (lam + memory variables) carries
            # across segments.
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
            ckpt = ctx.ckpt_state
            for k in range(len(segments) - 1, -1, -1):
                s0, s1 = segments[k]
                if s0 > 0:
                    c = ckpt[k - 1]
                    ex_f.copy_(c[0])
                    ey_f.copy_(c[1])
                    ez_f.copy_(c[2])
                    hx_f.copy_(c[3])
                    hy_f.copy_(c[4])
                    hz_f.copy_(c[5])
                    m_ey_z_f.copy_(c[6])
                    m_ez_y_f.copy_(c[7])
                    m_ez_x_f.copy_(c[8])
                    m_ex_z_f.copy_(c[9])
                    m_ex_y_f.copy_(c[10])
                    m_ey_x_f.copy_(c[11])
                    m_hy_z_f.copy_(c[12])
                    m_hz_y_f.copy_(c[13])
                    m_hz_x_f.copy_(c[14])
                    m_hx_z_f.copy_(c[15])
                    m_hx_y_f.copy_(c[16])
                    m_hy_x_f.copy_(c[17])
                else:
                    for buf in (
                        ex_f, ey_f, ez_f, hx_f, hy_f, hz_f,
                        m_ey_z_f, m_ez_y_f, m_ez_x_f, m_ex_z_f, m_ex_y_f,
                        m_ey_x_f, m_hy_z_f, m_hz_y_f, m_hz_x_f, m_hx_z_f,
                        m_hx_y_f, m_hy_x_f,
                    ):
                        buf.zero_()
                EM3DFunc._replay_segment(
                    ctx, ext, ca_p, cb_p, cq_p, f, src_i, n_shots, n_src,
                    ex_f, ey_f, ez_f, hx_f, hy_f, hz_f,
                    m_ey_z_f, m_ez_y_f, m_ez_x_f, m_ex_z_f, m_ex_y_f,
                    m_ey_x_f, m_hy_z_f, m_hz_y_f, m_hz_x_f, m_hx_z_f,
                    m_hx_y_f, m_hy_x_f,
                    ex_store, ey_store, ez_store,
                    curl_x_store, curl_y_store, curl_z_store,
                    numel, nz, ny, nx, s0, s1,
                )
                for t in range(s1 - 1, s0 - 1, -1):
                    if n_rec > 0:
                        ext.record_grad_r(
                            (lam_ex, lam_ey, lam_ez)[ctx.receiver_component],
                            grad_r, rec_i, t, n_shots, n_rec, numel,
                        )
                    if n_src > 0:
                        ext.record_grad_f(
                            (lam_ex, lam_ey, lam_ez)[ctx.source_component],
                            grad_f, src_i, t, n_shots, n_src, numel,
                        )
                    snap_off = ((t - s0) // ctx.grad_stride) * shot_count
                    if t % ctx.grad_stride == 0:
                        ext.coeff_grad(
                            lam_ex, lam_ey, lam_ez,
                            ex_store, ey_store, ez_store,
                            curl_x_store, curl_y_store, curl_z_store,
                            grad_ca, grad_cb,
                            scale, snap_off,
                            n_shots, nz, ny, nx,
                            *ctx.fd_pad,
                        )
                    ext.adjoint_e_stage1(
                        ca_p, cb_p, lam_ex, lam_ey, lam_ez,
                        m_lambda_hy_z, m_lambda_hz_y, m_lambda_hz_x,
                        m_lambda_hx_z, m_lambda_hx_y, m_lambda_hy_x,
                        work_hy_z, work_hz_y, work_hz_x,
                        work_hx_z, work_hx_y, work_hy_x,
                        az, bz, ay, by, ax, bx, kz, ky, kx,
                        n_shots, nz, ny, nx,
                        ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                        ctx.pml_x0, ctx.pml_x1,
                        *ctx.fd_pad, ctx.ca_batched, ctx.cb_batched,
                    )
                    ext.adjoint_e_stage2(
                        work_hy_z, work_hz_y, work_hz_x,
                        work_hx_z, work_hx_y, work_hy_x,
                        lam_hy, lam_hz, lam_hx, ctx.c,
                        ctx.rdz, ctx.rdy, ctx.rdx,
                        n_shots, nz, ny, nx,
                        ctx.fd_pad[0], ctx.fd_pad[2], ctx.fd_pad[4],
                    )
                    ext.adjoint_h_stage1(
                        cq_p, lam_hx, lam_hy, lam_hz,
                        m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x,
                        m_lambda_ex_z, m_lambda_ex_y, m_lambda_ey_x,
                        work2_ey_z, work2_ez_y, work2_ez_x,
                        work2_ex_z, work2_ex_y, work2_ey_x,
                        azh, bzh, ayh, byh, axh, bxh, kzh, kyh, kxh,
                        n_shots, nz, ny, nx,
                        ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                        ctx.pml_x0, ctx.pml_x1,
                        *ctx.fd_pad, ctx.cq_batched,
                    )
                    if t % ctx.grad_stride == 0:
                        ext.cq_grad(
                            cq_p, lam_hx, lam_hy, lam_hz,
                            ex_store, ey_store, ez_store,
                            m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x,
                            m_lambda_ex_z, m_lambda_ex_y, m_lambda_ey_x,
                            grad_cq,
                            azh, kzh, ayh, kyh, axh, kxh,
                            ctx.c,
                            ctx.rdz, ctx.rdy, ctx.rdx,
                            scale, snap_off,
                            n_shots, nz, ny, nx,
                            *ctx.fd_pad, ctx.cq_batched,
                        )
                    ext.adjoint_h_stage2(
                        work2_ey_z, work2_ez_y, work2_ez_x,
                        work2_ex_z, work2_ex_y, work2_ey_x,
                        lam_ex, lam_ey, lam_ez, ctx.c,
                        ctx.rdz, ctx.rdy, ctx.rdx,
                        n_shots, nz, ny, nx,
                        ctx.fd_pad[0], ctx.fd_pad[2], ctx.fd_pad[4],
                    )
        else:
            for t in range(nt - 1, -1, -1):
                if n_rec > 0:
                    ext.record_grad_r(
                        (lam_ex, lam_ey, lam_ez)[ctx.receiver_component],
                        grad_r, rec_i, t, n_shots, n_rec, numel,
                    )
                if n_src > 0:
                    ext.record_grad_f(
                        (lam_ex, lam_ey, lam_ez)[ctx.source_component],
                        grad_f, src_i, t, n_shots, n_src, numel,
                    )
                snap_off = ex_storage.snap_offset(t // ctx.grad_stride)
                if t % ctx.grad_stride == 0:
                    ext.coeff_grad(
                        lam_ex, lam_ey, lam_ez,
                        ex_store, ey_store, ez_store,
                        curl_x_store, curl_y_store, curl_z_store,
                        grad_ca, grad_cb,
                        scale, snap_off,
                        n_shots, nz, ny, nx,
                        *ctx.fd_pad,
                    )
                ext.adjoint_e_stage1(
                    ca_p, cb_p, lam_ex, lam_ey, lam_ez,
                    m_lambda_hy_z, m_lambda_hz_y, m_lambda_hz_x,
                    m_lambda_hx_z, m_lambda_hx_y, m_lambda_hy_x,
                    work_hy_z, work_hz_y, work_hz_x,
                    work_hx_z, work_hx_y, work_hy_x,
                    az, bz, ay, by, ax, bx, kz, ky, kx,
                    n_shots, nz, ny, nx,
                    ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                    ctx.pml_x0, ctx.pml_x1,
                    *ctx.fd_pad, ctx.ca_batched, ctx.cb_batched,
                )
                ext.adjoint_e_stage2(
                    work_hy_z, work_hz_y, work_hz_x,
                    work_hx_z, work_hx_y, work_hy_x,
                    lam_hy, lam_hz, lam_hx, ctx.c,
                    ctx.rdz, ctx.rdy, ctx.rdx,
                    n_shots, nz, ny, nx,
                    ctx.fd_pad[0], ctx.fd_pad[2], ctx.fd_pad[4],
                )
                ext.adjoint_h_stage1(
                    cq_p, lam_hx, lam_hy, lam_hz,
                    m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x,
                    m_lambda_ex_z, m_lambda_ex_y, m_lambda_ey_x,
                    work2_ey_z, work2_ez_y, work2_ez_x,
                    work2_ex_z, work2_ex_y, work2_ey_x,
                    azh, bzh, ayh, byh, axh, bxh, kzh, kyh, kxh,
                    n_shots, nz, ny, nx,
                    ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                    ctx.pml_x0, ctx.pml_x1,
                    *ctx.fd_pad, ctx.cq_batched,
                )
                if t % ctx.grad_stride == 0:
                    ext.cq_grad(
                        cq_p, lam_hx, lam_hy, lam_hz,
                        ex_store, ey_store, ez_store,
                        m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x,
                        m_lambda_ex_z, m_lambda_ex_y, m_lambda_ey_x,
                        grad_cq,
                        azh, kzh, ayh, kyh, axh, kxh,
                        ctx.c,
                        ctx.rdz, ctx.rdy, ctx.rdx,
                        scale, snap_off,
                        n_shots, nz, ny, nx,
                        *ctx.fd_pad, ctx.cq_batched,
                    )
                ext.adjoint_h_stage2(
                    work2_ey_z, work2_ez_y, work2_ez_x,
                    work2_ex_z, work2_ex_y, work2_ey_x,
                    lam_ex, lam_ey, lam_ez, ctx.c,
                    ctx.rdz, ctx.rdy, ctx.rdx,
                    n_shots, nz, ny, nx,
                    ctx.fd_pad[0], ctx.fd_pad[2], ctx.fd_pad[4],
                )

        grad_ca = grad_ca if ctx.ca_batched else grad_ca.sum(0, keepdim=True)
        grad_cb = grad_cb if ctx.cb_batched else grad_cb.sum(0, keepdim=True)
        grad_cq = grad_cq if ctx.cq_batched else grad_cq.sum(0, keepdim=True)

        return (
            grad_ca, grad_cb, grad_cq, grad_f,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None,
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

    Returns:
        receiver_amplitudes [nt, n_shots, n_rec].
    """
    accuracy = check_accuracy(accuracy)
    if not isinstance(grid_spacing, (list, tuple)):
        grid_spacing = [float(grid_spacing)] * 3
    grid_spacing = [float(g) for g in grid_spacing]
    pml_w = _set_pml_width(pml_width, 3)
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

    if source_amplitudes is not None:
        amp = source_amplitudes.to(device=device, dtype=dtype)
        if amp.shape[0] != n_shots:
            raise ValueError("source_amplitudes must have n_shots batches.")
        if amp.shape[2] < nt_inner:
            raise ValueError("source_amplitudes must have at least nt steps.")
        amp = amp[:, :, :nt_inner]
    else:
        amp = torch.zeros(n_shots, 0, nt_inner, device=device, dtype=dtype)

    dz, dy, dx = grid_spacing
    source_coeff = -1.0 / (dz * dy * dx)
    if amp.numel() > 0:
        src_mask = src_i != -1
        src_i_masked = src_i.masked_fill(~src_mask, 0)
        cb_flat = cb_p.reshape(-1, nz * ny * nx).expand(n_shots, -1)
        cb_at_src = cb_flat.gather(1, src_i_masked)
        f = (amp.permute(2, 0, 1) * cb_at_src.unsqueeze(0) * source_coeff).contiguous()
    elif src_i.shape[1] > 0:
        # located sources without amplitudes: inject zero (no-op) amplitudes
        f = torch.zeros(nt_inner, n_shots, src_i.shape[1], device=device, dtype=dtype)
    else:
        f = torch.empty(0, device=device, dtype=dtype)

    rdz, rdy, rdx = 1.0 / dz, 1.0 / dy, 1.0 / dx
    pml_z0, pml_z1 = fd_pad[0] + pml_w[0], nz - fd_pad[1] - pml_w[1]
    pml_y0, pml_y1 = fd_pad[2] + pml_w[2], ny - fd_pad[3] - pml_w[3]
    pml_x0, pml_x1 = fd_pad[4] + pml_w[4], nx - fd_pad[5] - pml_w[5]
    ca_batched = 1 if ca_p.shape[0] == n_shots else 0
    cb_batched = 1 if cb_p.shape[0] == n_shots else 0
    cq_batched = 1 if cq_p.shape[0] == n_shots else 0

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
    )
    return r


def _compile_material_coefficients(epsilon_r, sigma_r, mu_r, dt):
    """Compile material models into 3D Maxwell update coefficients.

    Same formulas as em2d_tm (``prepare_parameters``): ca/cb for
    the E update with a conductivity loss averaged over the time step, cq =
    dt/mu for the H update.  Differentiable, so gradients flow from
    ca/cb/cq back to epsilon/sigma/mu.
    """
    eps = epsilon_r * EPS0
    mu = mu_r * MU0
    denom = 1.0 + sigma_r * dt / (2.0 * eps)
    ca = (1.0 - sigma_r * dt / (2.0 * eps)) / denom
    cb = (dt / eps) / denom
    cq = dt / mu
    return ca, cb, cq


def _pml_profile_1d(
    pml_width,
    pml_start,
    dt,
    n,
    dtype,
    device,
    half,
    accuracy=2,
    n_power=4,
    eps=1e-9,
    grid_spacing=1.0,
    eps_scale=EPS0,
):
    """a, b, k CPML profiles along one dimension (em2d_tm's _pml_profile_1d).

    ``accuracy`` is validated for API parity but does not affect the EM
    a/b/k profiles (the CPML recursion is FD-order independent).
    """
    check_accuracy(accuracy)
    k_max_cpml = 5.0  # maximum coordinate stretching factor
    alpha_max_cpml = 0.008  # maximum frequency shift

    a = torch.zeros(n, dtype=dtype, device=device)
    b = torch.zeros(n, dtype=dtype, device=device)
    k = torch.ones(n, dtype=dtype, device=device)

    if pml_width[0] == 0 and pml_width[1] == 0:
        return a, b, k

    sigma0 = (n_power + 1) / (150.0 * math.pi * grid_spacing)

    x = torch.arange(n, dtype=dtype, device=device)
    if half:
        x = x + 0.5

    for side in range(2):
        if pml_width[side] == 0:
            continue
        abscissa = pml_start[0] - x if side == 0 else x - pml_start[1]
        mask = abscissa >= 0
        # Normalized distance into the PML (0 at inner edge, 1 at outer edge)
        abscissa_norm = torch.clamp(abscissa / pml_width[side], 0, 1)

        sigma = sigma0 * (abscissa_norm**n_power)
        k_side = 1.0 + (k_max_cpml - 1.0) * (abscissa_norm**n_power)
        alpha = alpha_max_cpml * (1.0 - abscissa_norm) + 0.1 * alpha_max_cpml

        k = torch.where(mask, k_side, k)

        b_side = torch.exp(-(sigma / k_side + alpha) * dt / eps_scale)
        b = torch.where(mask, b_side, b)

        denom = k_side * (sigma + k_side * alpha) + eps
        a_side = sigma * (b_side - 1.0) / denom
        a_side = torch.where(sigma > 1e-6, a_side, torch.zeros_like(a_side))
        a = torch.where(mask, a_side, a)

    return a, b, k


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


def _set_pml_width(pml_width, ndim):
    if isinstance(pml_width, int):
        return [pml_width] * (2 * ndim)
    pml_width = list(pml_width)
    if len(pml_width) != 2 * ndim:
        raise ValueError(f"pml_width must be int or length {2 * ndim}.")
    return pml_width
