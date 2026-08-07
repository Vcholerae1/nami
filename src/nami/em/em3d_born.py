"""nami em3d_born: 3D electromagnetic first-order Born FDTD (staggered Yee
grid, C-PML) with a native exact-transpose CUDA adjoint.

A background field (Ex/Ey/Ez, Hx/Hy/Hz) is propagated with the
unperturbed ca/cb/cq coefficients exactly as :mod:`em3d`, and a scattered
field (d_ex/d_ey/d_ez, d_hx/d_hy/d_hz) is driven by the first-order Born
scattering sources

    d_ex = ca*d_ex + cb*dcurl_x + dca*ex_old + dcb*curl_x      (y, z anal.)
    d_hx -= cq*(dd_ey_dz - dd_ez_dy) + dcq*(d_ey_dz - d_ez_dy)    (y, z anal.)

where ``ex_old``/``curl_*`` are the pre-update background field/curl and
``dca/dcb/dcq`` are the exact linearizations of the material coefficients
w.r.t. ``(epsilon_scatter, sigma_scatter, mu_scatter)`` around the
background (the exact linearization of the material coefficients, plus
``dcq = -cq/mu*dmu`` when a ``mu_scatter`` is given; ``dcq`` is zero by
default).  Source pre-scaling follows nami's ``em3d``
(``cb * -1/(dx dy dz)`` for the background source and the ``dcb`` analogue
for the scattered source, matching the source linearization); receivers
record the post-injection scattered field from ``receiver_component`` (and
the background field from the same component for the optional
``bg_receiver_locations``).

The adjoint is the exact discrete transpose of the coupled
(background + scattered) system (the same two-stage E/H structure as
:mod:`em3d`, with separate adjoint fields for each wave component), so
gradients flow to both the background models (epsilon/sigma/mu) and the
scatter models (epsilon_scatter/sigma_scatter/mu_scatter), plus the source
amplitudes.

Return convention: ``(receiver_amplitudes, bg_receiver_amplitudes)`` each
``[nt, n_shots, n_rec]`` (the internal order is background-first;
nami returns the scattered traces first to mirror ``em2d_tm_born``).
"""

import math

import nami_em3d_born as _ext
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
from .em3d import (
    EPS0,
    MU0,
    _compile_material_coefficients,
    _normalize_component,
    _set_em_pml_profiles_3d,
    _set_pml_width,
)

# Checkpoint state layout (saved at time t BEFORE step t): the em3d 18
# followed by the scattered counterparts:
#   [0:3]   ex, ey, ez
#   [3:6]   hx, hy, hz
#   [6:12]  bg H-step memory variables
#   [12:18] bg E-step memory variables
#   [18:21] d_ex, d_ey, d_ez
#   [21:24] d_hx, d_hy, d_hz
#   [24:30] scatter H-step memory variables
#   [30:36] scatter E-step memory variables
N_STATE = 36
N_STREAMS = 24


class BornEM3DFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ca_p, cb_p, cq_p,           # [n_shots, nz, ny, nx] or [1, ...] padded bg
        dca_p, dcb_p, dcq_p,        # padded scatter coefficient linearizations
        f_bg, f_sc,                 # [nt, n_shots, n_src] pre-scaled sources
        src_i, rec_i, bg_rec_i,
        profs, c, fd_pad,
        rdz, rdy, rdx,
        nt, pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
        ca_batched, cb_batched, cq_batched,
        dca_batched, dcb_batched, dcq_batched,
        grad_stride,
        ex_storage, ey_storage, ez_storage,
        curl_x_storage, curl_y_storage, curl_z_storage,
        dex_storage, dey_storage, dez_storage,
        dcurl_x_storage, dcurl_y_storage, dcurl_z_storage,
        dey_dz_storage, dez_dy_storage, dez_dx_storage,
        dex_dz_storage, dex_dy_storage, dey_dx_storage,
        ddey_dz_storage, ddez_dy_storage, ddez_dx_storage,
        ddex_dz_storage, ddex_dy_storage, ddey_dx_storage,
        source_component, receiver_component,
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
        n_bg_rec = bg_rec_i.shape[1]
        numel = nz * ny * nx
        shot_count = n_shots * numel

        z = lambda *shape: torch.zeros(shape, device=device, dtype=dtype)  # noqa: E731
        ex, ey, ez = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx)
        )
        d_ex, d_ey, d_ez = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx)
        )
        hx, hy, hz = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx)
        )
        d_hx, d_hy, d_hz = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx)
        )
        # H-step memory variables (half-integer profiles), bg + scattered
        m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        dm_ey_z, dm_ez_y, dm_ez_x, dm_ex_z, dm_ex_y, dm_ey_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        # E-step memory variables (integer profiles), bg + scattered
        m_hy_z, m_hz_y, m_hz_x, m_hx_z, m_hx_y, m_hy_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        dm_hy_z, dm_hz_y, dm_hz_x, dm_hx_z, dm_hx_y, dm_hy_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        r = z(nt, n_shots, n_rec)
        r_bg = z(nt, n_shots, n_bg_rec)

        if ex_storage is not None:
            dey_dz_store, dez_dy_store, dez_dx_store = (
                dey_dz_storage.snap, dez_dy_storage.snap, dez_dx_storage.snap,
            )
            dex_dz_store, dex_dy_store, dey_dx_store = (
                dex_dz_storage.snap, dex_dy_storage.snap, dey_dx_storage.snap,
            )
            ddey_dz_store, ddez_dy_store, ddez_dx_store = (
                ddey_dz_storage.snap, ddez_dy_storage.snap, ddez_dx_storage.snap,
            )
            ddex_dz_store, ddex_dy_store, ddey_dx_store = (
                ddex_dz_storage.snap, ddex_dy_storage.snap, ddey_dx_storage.snap,
            )
            ex_store, ey_store, ez_store = (
                ex_storage.snap, ey_storage.snap, ez_storage.snap,
            )
            curl_x_store, curl_y_store, curl_z_store = (
                curl_x_storage.snap, curl_y_storage.snap, curl_z_storage.snap,
            )
            dex_store, dey_store, dez_store = (
                dex_storage.snap, dey_storage.snap, dez_storage.snap,
            )
            dcurl_x_store, dcurl_y_store, dcurl_z_store = (
                dcurl_x_storage.snap, dcurl_y_storage.snap, dcurl_z_storage.snap,
            )
        else:
            # forward-only: skip the H-step snapshot writes via store=0 with
            # empty tensors; born_step_e has no store flag, so give its E-step
            # stores full-size scratch buffers (never read).
            n_snap = (nt + grad_stride - 1) // grad_stride
            empty = torch.empty(0, device=device, dtype=dtype)
            dey_dz_store = dez_dy_store = dez_dx_store = empty
            dex_dz_store = dex_dy_store = dey_dx_store = empty
            ddey_dz_store = ddez_dy_store = ddez_dx_store = empty
            ddex_dz_store = ddex_dy_store = ddey_dx_store = empty
            scratch = z(n_snap, shot_count)
            ex_store = ey_store = ez_store = scratch
            curl_x_store = curl_y_store = curl_z_store = scratch
            dex_store = dey_store = dez_store = scratch
            dcurl_x_store = dcurl_y_store = dcurl_z_store = scratch

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
                ckpt_state[k, 18].copy_(d_ex)
                ckpt_state[k, 19].copy_(d_ey)
                ckpt_state[k, 20].copy_(d_ez)
                ckpt_state[k, 21].copy_(d_hx)
                ckpt_state[k, 22].copy_(d_hy)
                ckpt_state[k, 23].copy_(d_hz)
                ckpt_state[k, 24].copy_(dm_ey_z)
                ckpt_state[k, 25].copy_(dm_ez_y)
                ckpt_state[k, 26].copy_(dm_ez_x)
                ckpt_state[k, 27].copy_(dm_ex_z)
                ckpt_state[k, 28].copy_(dm_ex_y)
                ckpt_state[k, 29].copy_(dm_ey_x)
                ckpt_state[k, 30].copy_(dm_hy_z)
                ckpt_state[k, 31].copy_(dm_hz_y)
                ckpt_state[k, 32].copy_(dm_hz_x)
                ckpt_state[k, 33].copy_(dm_hx_z)
                ckpt_state[k, 34].copy_(dm_hx_y)
                ckpt_state[k, 35].copy_(dm_hy_x)
            if segments:
                store = 0
                snap_off = 0
            elif ex_storage is not None:
                store = 1
                snap_off = ex_storage.snap_offset(t // grad_stride)
            else:
                store = 0
                snap_off = (t // grad_stride) * shot_count
            ext.born_step_h(
                cq_p, dcq_p, ex, ey, ez, d_ex, d_ey, d_ez,
                hx, hy, hz, d_hx, d_hy, d_hz,
                m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x,
                dm_ey_z, dm_ez_y, dm_ez_x, dm_ex_z, dm_ex_y, dm_ey_x,
                dey_dz_store, dez_dy_store, dez_dx_store,
                dex_dz_store, dex_dy_store, dey_dx_store,
                ddey_dz_store, ddez_dy_store, ddez_dx_store,
                ddex_dz_store, ddex_dy_store, ddey_dx_store,
                azh, bzh, ayh, byh, axh, bxh, kzh, kyh, kxh,
                c, rdz, rdy, rdx, t, grad_stride, snap_off,
                n_shots, nz, ny, nx,
                pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
                fd_pad[0], fd_pad[2], fd_pad[4],
                cq_batched, dcq_batched, store,
            )
            ext.born_step_e(
                ca_p, cb_p, dca_p, dcb_p,
                hx, hy, hz, d_hx, d_hy, d_hz,
                ex, ey, ez, d_ex, d_ey, d_ez,
                m_hy_z, m_hz_y, m_hz_x, m_hx_z, m_hx_y, m_hy_x,
                dm_hy_z, dm_hz_y, dm_hz_x, dm_hx_z, dm_hx_y, dm_hy_x,
                ex_store, ey_store, ez_store,
                curl_x_store, curl_y_store, curl_z_store,
                dex_store, dey_store, dez_store,
                dcurl_x_store, dcurl_y_store, dcurl_z_store,
                az, bz, ay, by, ax, bx, kz, ky, kx,
                c, rdz, rdy, rdx, t, grad_stride, snap_off,
                n_shots, nz, ny, nx,
                pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
                *fd_pad,
                ca_batched, cb_batched, dca_batched, dcb_batched,
            )
            if n_src > 0:
                ext.born_inject(
                    (ex, ey, ez)[source_component],
                    (d_ex, d_ey, d_ez)[source_component],
                    f_bg, f_sc, src_i, t, n_shots, n_src, numel,
                )
            if n_rec > 0:
                ext.born_record(
                    (d_ex, d_ey, d_ez)[receiver_component], r, rec_i,
                    t, n_shots, n_rec, numel,
                )
            if n_bg_rec > 0:
                ext.born_record(
                    (ex, ey, ez)[receiver_component], r_bg, bg_rec_i,
                    t, n_shots, n_bg_rec, numel,
                )

        ctx.ext = ext
        ctx.save_for_backward(
            ca_p, cb_p, cq_p, dca_p, dcb_p, dcq_p,
            f_bg, f_sc, src_i, rec_i, bg_rec_i,
        )
        ctx.ex_storage = ex_storage
        ctx.ey_storage = ey_storage
        ctx.ez_storage = ez_storage
        ctx.curl_x_storage = curl_x_storage
        ctx.curl_y_storage = curl_y_storage
        ctx.curl_z_storage = curl_z_storage
        ctx.dex_storage = dex_storage
        ctx.dey_storage = dey_storage
        ctx.dez_storage = dez_storage
        ctx.dcurl_x_storage = dcurl_x_storage
        ctx.dcurl_y_storage = dcurl_y_storage
        ctx.dcurl_z_storage = dcurl_z_storage
        ctx.dey_dz_storage = dey_dz_storage
        ctx.dez_dy_storage = dez_dy_storage
        ctx.dez_dx_storage = dez_dx_storage
        ctx.dex_dz_storage = dex_dz_storage
        ctx.dex_dy_storage = dex_dy_storage
        ctx.dey_dx_storage = dey_dx_storage
        ctx.ddey_dz_storage = ddey_dz_storage
        ctx.ddez_dy_storage = ddez_dy_storage
        ctx.ddez_dx_storage = ddez_dx_storage
        ctx.ddex_dz_storage = ddex_dz_storage
        ctx.ddex_dy_storage = ddex_dy_storage
        ctx.ddey_dx_storage = ddey_dx_storage
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
        ctx.dca_batched = dca_batched
        ctx.dcb_batched = dcb_batched
        ctx.dcq_batched = dcq_batched
        ctx.grad_stride = grad_stride
        ctx.source_component = source_component
        ctx.receiver_component = receiver_component
        ctx.ckpt_state = ckpt_state
        ctx.checkpoint_every = checkpoint_every
        ctx.segments = segments
        return r, r_bg

    @staticmethod
    def _replay_segment(ctx, ext, ca_p, cb_p, cq_p, dca_p, dcb_p, dcq_p,
                        f_bg, f_sc, src_i, n_shots, n_src,
                        ex, ey, ez, d_ex, d_ey, d_ez, hx, hy, hz,
                        d_hx, d_hy, d_hz, m_ey_z, m_ez_y, m_ez_x, m_ex_z,
                        m_ex_y, m_ey_x, dm_ey_z, dm_ez_y, dm_ez_x, dm_ex_z,
                        dm_ex_y, dm_ey_x, m_hy_z, m_hz_y, m_hz_x, m_hx_z,
                        m_hx_y, m_hy_x, dm_hy_z, dm_hz_y, dm_hz_x, dm_hx_z,
                        dm_hx_y, dm_hy_x,
                        dey_dz_store, dez_dy_store, dez_dx_store,
                        dex_dz_store, dex_dy_store, dey_dx_store,
                        ddey_dz_store, ddez_dy_store, ddez_dx_store,
                        ddex_dz_store, ddex_dy_store, ddey_dx_store,
                        ex_store, ey_store, ez_store,
                        curl_x_store, curl_y_store, curl_z_store,
                        dex_store, dey_store, dez_store,
                        dcurl_x_store, dcurl_y_store, dcurl_z_store,
                        numel, nz, ny, nx, s0, s1):
        """Re-run forward steps [s0, s1), writing the w snapshots."""
        az, azh, ay, ayh, ax, axh, bz, bzh, by, byh, bx, bxh, \
            kz, kzh, ky, kyh, kx, kxh = [p.contiguous() for p in ctx.profs]
        for t in range(s0, s1):
            snap_off = ((t - s0) // ctx.grad_stride) * (n_shots * numel)
            ext.born_step_h(
                cq_p, dcq_p, ex, ey, ez, d_ex, d_ey, d_ez,
                hx, hy, hz, d_hx, d_hy, d_hz,
                m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x,
                dm_ey_z, dm_ez_y, dm_ez_x, dm_ex_z, dm_ex_y, dm_ey_x,
                dey_dz_store, dez_dy_store, dez_dx_store,
                dex_dz_store, dex_dy_store, dey_dx_store,
                ddey_dz_store, ddez_dy_store, ddez_dx_store,
                ddex_dz_store, ddex_dy_store, ddey_dx_store,
                azh, bzh, ayh, byh, axh, bxh, kzh, kyh, kxh,
                ctx.c, ctx.rdz, ctx.rdy, ctx.rdx, t, ctx.grad_stride, snap_off,
                n_shots, nz, ny, nx,
                ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                ctx.pml_x0, ctx.pml_x1,
                ctx.fd_pad[0], ctx.fd_pad[2], ctx.fd_pad[4],
                ctx.cq_batched, ctx.dcq_batched, 1,
            )
            ext.born_step_e(
                ca_p, cb_p, dca_p, dcb_p,
                hx, hy, hz, d_hx, d_hy, d_hz,
                ex, ey, ez, d_ex, d_ey, d_ez,
                m_hy_z, m_hz_y, m_hz_x, m_hx_z, m_hx_y, m_hy_x,
                dm_hy_z, dm_hz_y, dm_hz_x, dm_hx_z, dm_hx_y, dm_hy_x,
                ex_store, ey_store, ez_store,
                curl_x_store, curl_y_store, curl_z_store,
                dex_store, dey_store, dez_store,
                dcurl_x_store, dcurl_y_store, dcurl_z_store,
                az, bz, ay, by, ax, bx, kz, ky, kx,
                ctx.c, ctx.rdz, ctx.rdy, ctx.rdx, t, ctx.grad_stride, snap_off,
                n_shots, nz, ny, nx,
                ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                ctx.pml_x0, ctx.pml_x1,
                *ctx.fd_pad,
                ctx.ca_batched, ctx.cb_batched,
                ctx.dca_batched, ctx.dcb_batched,
            )
            if n_src > 0:
                ext.born_inject(
                    (ex, ey, ez)[ctx.source_component],
                    (d_ex, d_ey, d_ez)[ctx.source_component],
                    f_bg, f_sc, src_i, t, n_shots, n_src, numel,
                )

    @staticmethod
    def backward(ctx, grad_r, grad_r_bg):
        ext = ctx.ext
        (ca_p, cb_p, cq_p, dca_p, dcb_p, dcq_p,
         f_bg, f_sc, src_i, rec_i, bg_rec_i) = ctx.saved_tensors
        if ctx.ex_storage is None:
            raise RuntimeError(
                "em3d_born backward() requires snapshot storage: run the "
                "forward with an input requiring grad (and not under "
                "torch.no_grad())."
            )
        ex_storage = ctx.ex_storage
        ey_storage = ctx.ey_storage
        ez_storage = ctx.ez_storage
        curl_x_storage = ctx.curl_x_storage
        curl_y_storage = ctx.curl_y_storage
        curl_z_storage = ctx.curl_z_storage
        dex_storage = ctx.dex_storage
        dey_storage = ctx.dey_storage
        dez_storage = ctx.dez_storage
        dcurl_x_storage = ctx.dcurl_x_storage
        dcurl_y_storage = ctx.dcurl_y_storage
        dcurl_z_storage = ctx.dcurl_z_storage
        dey_dz_storage = ctx.dey_dz_storage
        dez_dy_storage = ctx.dez_dy_storage
        dez_dx_storage = ctx.dez_dx_storage
        dex_dz_storage = ctx.dex_dz_storage
        dex_dy_storage = ctx.dex_dy_storage
        dey_dx_storage = ctx.dey_dx_storage
        ddey_dz_storage = ctx.ddey_dz_storage
        ddez_dy_storage = ctx.ddez_dy_storage
        ddez_dx_storage = ctx.ddez_dx_storage
        ddex_dz_storage = ctx.ddex_dz_storage
        ddex_dy_storage = ctx.ddex_dy_storage
        ddey_dx_storage = ctx.ddey_dx_storage
        ex_store, ey_store, ez_store = (
            ex_storage.snap, ey_storage.snap, ez_storage.snap,
        )
        curl_x_store, curl_y_store, curl_z_store = (
            curl_x_storage.snap, curl_y_storage.snap, curl_z_storage.snap,
        )
        dex_store, dey_store, dez_store = (
            dex_storage.snap, dey_storage.snap, dez_storage.snap,
        )
        dcurl_x_store, dcurl_y_store, dcurl_z_store = (
            dcurl_x_storage.snap, dcurl_y_storage.snap, dcurl_z_storage.snap,
        )
        dey_dz_store, dez_dy_store, dez_dx_store = (
            dey_dz_storage.snap, dez_dy_storage.snap, dez_dx_storage.snap,
        )
        dex_dz_store, dex_dy_store, dey_dx_store = (
            dex_dz_storage.snap, dex_dy_storage.snap, dey_dx_storage.snap,
        )
        ddey_dz_store, ddez_dy_store, ddez_dx_store = (
            ddey_dz_storage.snap, ddez_dy_storage.snap, ddez_dx_storage.snap,
        )
        ddex_dz_store, ddex_dy_store, ddey_dx_store = (
            ddex_dz_storage.snap, ddex_dy_storage.snap, ddey_dx_storage.snap,
        )
        device = ca_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = ca_p.dtype
        n_shots, nz, ny, nx = ca_p.shape
        n_src, n_rec, n_bg_rec = (
            src_i.shape[1], rec_i.shape[1], bg_rec_i.shape[1],
        )
        nt = ctx.nt
        numel = nz * ny * nx
        shot_count = n_shots * numel

        if grad_r is None:
            grad_r = torch.zeros(nt, n_shots, n_rec, device=device, dtype=dtype)
        grad_r = grad_r.contiguous()
        if grad_r_bg is None:
            grad_r_bg = torch.zeros(
                nt, n_shots, n_bg_rec, device=device, dtype=dtype
            )
        grad_r_bg = grad_r_bg.contiguous()
        grad_f_bg = torch.zeros(nt, n_shots, n_src, device=device, dtype=dtype)
        grad_f_sc = torch.zeros_like(grad_f_bg)
        grad_ca = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        grad_cb = torch.zeros_like(grad_ca)
        grad_cq = torch.zeros_like(grad_ca)
        grad_dca = torch.zeros_like(grad_ca)
        grad_dcb = torch.zeros_like(grad_ca)
        grad_dcq = torch.zeros_like(grad_ca)

        z = lambda *shape: torch.zeros(shape, device=device, dtype=dtype)  # noqa: E731
        lam_ex, lam_ey, lam_ez = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx)
        )
        lam_d_ex, lam_d_ey, lam_d_ez = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx)
        )
        lam_hx, lam_hy, lam_hz = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx)
        )
        lam_dhx, lam_dhy, lam_dhz = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx)
        )
        # E-stage-1 memory (integer profiles) and work arrays, bg + scattered
        m_lambda_hy_z, m_lambda_hz_y, m_lambda_hz_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        m_lambda_hx_z, m_lambda_hx_y, m_lambda_hy_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        dm_lambda_hy_z, dm_lambda_hz_y, dm_lambda_hz_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        dm_lambda_hx_z, dm_lambda_hx_y, dm_lambda_hy_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        work_hy_z, work_hz_y, work_hz_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        work_hx_z, work_hx_y, work_hy_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        work_dhy_z, work_dhz_y, work_dhz_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        work_dhx_z, work_dhx_y, work_dhy_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        # H-stage-1 memory (half-integer profiles) and work arrays, bg + sc
        m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        m_lambda_ex_z, m_lambda_ex_y, m_lambda_ey_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        dm_lambda_ey_z, dm_lambda_ez_y, dm_lambda_ez_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        dm_lambda_ex_z, dm_lambda_ex_y, dm_lambda_ey_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        work2_ey_z, work2_ez_y, work2_ez_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        work2_ex_z, work2_ex_y, work2_ey_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        work2_d_ey_z, work2_d_ez_y, work2_d_ez_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )
        work2_d_ex_z, work2_d_ex_y, work2_d_ey_x = (
            z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx), z(n_shots, nz, ny, nx),
        )

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
            d_ex_f = torch.zeros_like(ex_f)
            d_ey_f = torch.zeros_like(ex_f)
            d_ez_f = torch.zeros_like(ex_f)
            hx_f = torch.zeros_like(ex_f)
            hy_f = torch.zeros_like(ex_f)
            hz_f = torch.zeros_like(ex_f)
            d_hx_f = torch.zeros_like(ex_f)
            d_hy_f = torch.zeros_like(ex_f)
            d_hz_f = torch.zeros_like(ex_f)
            m_ey_z_f = torch.zeros_like(ex_f)
            m_ez_y_f = torch.zeros_like(ex_f)
            m_ez_x_f = torch.zeros_like(ex_f)
            m_ex_z_f = torch.zeros_like(ex_f)
            m_ex_y_f = torch.zeros_like(ex_f)
            m_ey_x_f = torch.zeros_like(ex_f)
            dm_ey_z_f = torch.zeros_like(ex_f)
            dm_ez_y_f = torch.zeros_like(ex_f)
            dm_ez_x_f = torch.zeros_like(ex_f)
            dm_ex_z_f = torch.zeros_like(ex_f)
            dm_ex_y_f = torch.zeros_like(ex_f)
            dm_ey_x_f = torch.zeros_like(ex_f)
            m_hy_z_f = torch.zeros_like(ex_f)
            m_hz_y_f = torch.zeros_like(ex_f)
            m_hz_x_f = torch.zeros_like(ex_f)
            m_hx_z_f = torch.zeros_like(ex_f)
            m_hx_y_f = torch.zeros_like(ex_f)
            m_hy_x_f = torch.zeros_like(ex_f)
            dm_hy_z_f = torch.zeros_like(ex_f)
            dm_hz_y_f = torch.zeros_like(ex_f)
            dm_hz_x_f = torch.zeros_like(ex_f)
            dm_hx_z_f = torch.zeros_like(ex_f)
            dm_hx_y_f = torch.zeros_like(ex_f)
            dm_hy_x_f = torch.zeros_like(ex_f)
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
                    d_ex_f.copy_(c[18])
                    d_ey_f.copy_(c[19])
                    d_ez_f.copy_(c[20])
                    d_hx_f.copy_(c[21])
                    d_hy_f.copy_(c[22])
                    d_hz_f.copy_(c[23])
                    dm_ey_z_f.copy_(c[24])
                    dm_ez_y_f.copy_(c[25])
                    dm_ez_x_f.copy_(c[26])
                    dm_ex_z_f.copy_(c[27])
                    dm_ex_y_f.copy_(c[28])
                    dm_ey_x_f.copy_(c[29])
                    dm_hy_z_f.copy_(c[30])
                    dm_hz_y_f.copy_(c[31])
                    dm_hz_x_f.copy_(c[32])
                    dm_hx_z_f.copy_(c[33])
                    dm_hx_y_f.copy_(c[34])
                    dm_hy_x_f.copy_(c[35])
                else:
                    for buf in (
                        ex_f, ey_f, ez_f, hx_f, hy_f, hz_f,
                        m_ey_z_f, m_ez_y_f, m_ez_x_f, m_ex_z_f, m_ex_y_f,
                        m_ey_x_f, m_hy_z_f, m_hz_y_f, m_hz_x_f, m_hx_z_f,
                        m_hx_y_f, m_hy_x_f,
                        d_ex_f, d_ey_f, d_ez_f, d_hx_f, d_hy_f, d_hz_f,
                        dm_ey_z_f, dm_ez_y_f, dm_ez_x_f, dm_ex_z_f,
                        dm_ex_y_f, dm_ey_x_f, dm_hy_z_f, dm_hz_y_f,
                        dm_hz_x_f, dm_hx_z_f, dm_hx_y_f, dm_hy_x_f,
                    ):
                        buf.zero_()
                BornEM3DFunc._replay_segment(
                    ctx, ext, ca_p, cb_p, cq_p, dca_p, dcb_p, dcq_p,
                    f_bg, f_sc, src_i, n_shots, n_src,
                    ex_f, ey_f, ez_f, d_ex_f, d_ey_f, d_ez_f, hx_f, hy_f, hz_f,
                    d_hx_f, d_hy_f, d_hz_f,
                    m_ey_z_f, m_ez_y_f, m_ez_x_f, m_ex_z_f, m_ex_y_f,
                    m_ey_x_f, dm_ey_z_f, dm_ez_y_f, dm_ez_x_f, dm_ex_z_f,
                    dm_ex_y_f, dm_ey_x_f, m_hy_z_f, m_hz_y_f, m_hz_x_f,
                    m_hx_z_f, m_hx_y_f, m_hy_x_f, dm_hy_z_f, dm_hz_y_f,
                    dm_hz_x_f, dm_hx_z_f, dm_hx_y_f, dm_hy_x_f,
                    dey_dz_store, dez_dy_store, dez_dx_store,
                    dex_dz_store, dex_dy_store, dey_dx_store,
                    ddey_dz_store, ddez_dy_store, ddez_dx_store,
                    ddex_dz_store, ddex_dy_store, ddey_dx_store,
                    ex_store, ey_store, ez_store,
                    curl_x_store, curl_y_store, curl_z_store,
                    dex_store, dey_store, dez_store,
                    dcurl_x_store, dcurl_y_store, dcurl_z_store,
                    numel, nz, ny, nx, s0, s1,
                )
                for t in range(s1 - 1, s0 - 1, -1):
                    if n_rec > 0:
                        ext.born_record_grad_r(
                            (lam_d_ex, lam_d_ey, lam_d_ez)[ctx.receiver_component],
                            grad_r, rec_i, t, n_shots, n_rec, numel,
                        )
                    if n_bg_rec > 0:
                        ext.born_record_grad_r(
                            (lam_ex, lam_ey, lam_ez)[ctx.receiver_component],
                            grad_r_bg, bg_rec_i, t, n_shots, n_bg_rec, numel,
                        )
                    if n_src > 0:
                        ext.born_record_grad_f(
                            (lam_ex, lam_ey, lam_ez)[ctx.source_component],
                            (lam_d_ex, lam_d_ey, lam_d_ez)[ctx.source_component],
                            grad_f_bg, grad_f_sc, src_i, t, n_shots, n_src, numel,
                        )
                    snap_off = ((t - s0) // ctx.grad_stride) * shot_count
                    if t % ctx.grad_stride == 0:
                        ext.born_coeff_grad(
                            lam_ex, lam_ey, lam_ez, lam_d_ex, lam_d_ey, lam_d_ez,
                            ex_store, ey_store, ez_store,
                            curl_x_store, curl_y_store, curl_z_store,
                            dex_store, dey_store, dez_store,
                            dcurl_x_store, dcurl_y_store, dcurl_z_store,
                            grad_ca, grad_cb, grad_dca, grad_dcb,
                            scale, snap_off,
                            n_shots, nz, ny, nx,
                            *ctx.fd_pad,
                        )
                    ext.born_adjoint_e_stage1(
                        ca_p, cb_p, dca_p, dcb_p,
                        lam_ex, lam_ey, lam_ez, lam_d_ex, lam_d_ey, lam_d_ez,
                        m_lambda_hy_z, m_lambda_hz_y, m_lambda_hz_x,
                        m_lambda_hx_z, m_lambda_hx_y, m_lambda_hy_x,
                        dm_lambda_hy_z, dm_lambda_hz_y, dm_lambda_hz_x,
                        dm_lambda_hx_z, dm_lambda_hx_y, dm_lambda_hy_x,
                        work_hy_z, work_hz_y, work_hz_x,
                        work_hx_z, work_hx_y, work_hy_x,
                        work_dhy_z, work_dhz_y, work_dhz_x,
                        work_dhx_z, work_dhx_y, work_dhy_x,
                        az, bz, ay, by, ax, bx, kz, ky, kx,
                        n_shots, nz, ny, nx,
                        ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                        ctx.pml_x0, ctx.pml_x1,
                        *ctx.fd_pad,
                        ctx.ca_batched, ctx.cb_batched,
                        ctx.dca_batched, ctx.dcb_batched,
                    )
                    ext.born_adjoint_e_stage2(
                        work_hy_z, work_hz_y, work_hz_x,
                        work_hx_z, work_hx_y, work_hy_x,
                        work_dhy_z, work_dhz_y, work_dhz_x,
                        work_dhx_z, work_dhx_y, work_dhy_x,
                        lam_hy, lam_hz, lam_hx,
                        lam_dhy, lam_dhz, lam_dhx,
                        ctx.c, ctx.rdz, ctx.rdy, ctx.rdx,
                        n_shots, nz, ny, nx,
                        ctx.fd_pad[0], ctx.fd_pad[2], ctx.fd_pad[4],
                    )
                    ext.born_adjoint_h_stage1(
                        cq_p, dcq_p,
                        lam_hx, lam_hy, lam_hz,
                        lam_dhx, lam_dhy, lam_dhz,
                        m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x,
                        m_lambda_ex_z, m_lambda_ex_y, m_lambda_ey_x,
                        dm_lambda_ey_z, dm_lambda_ez_y, dm_lambda_ez_x,
                        dm_lambda_ex_z, dm_lambda_ex_y, dm_lambda_ey_x,
                        work2_ey_z, work2_ez_y, work2_ez_x,
                        work2_ex_z, work2_ex_y, work2_ey_x,
                        work2_d_ey_z, work2_d_ez_y, work2_d_ez_x,
                        work2_d_ex_z, work2_d_ex_y, work2_d_ey_x,
                        azh, bzh, ayh, byh, axh, bxh, kzh, kyh, kxh,
                        n_shots, nz, ny, nx,
                        ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                        ctx.pml_x0, ctx.pml_x1,
                        *ctx.fd_pad,
                        ctx.cq_batched, ctx.dcq_batched,
                    )
                    if t % ctx.grad_stride == 0:
                        ext.born_cq_grad(
                            lam_hx, lam_hy, lam_hz,
                            lam_dhx, lam_dhy, lam_dhz,
                            dey_dz_store, dez_dy_store, dez_dx_store,
                            dex_dz_store, dex_dy_store, dey_dx_store,
                            ddey_dz_store, ddez_dy_store, ddez_dx_store,
                            ddex_dz_store, ddex_dy_store, ddey_dx_store,
                            grad_cq, grad_dcq,
                            t, ctx.grad_stride, scale, snap_off,
                            n_shots, nz, ny, nx,
                            *ctx.fd_pad,
                        )
                    ext.born_adjoint_h_stage2(
                        work2_ey_z, work2_ez_y, work2_ez_x,
                        work2_ex_z, work2_ex_y, work2_ey_x,
                        work2_d_ey_z, work2_d_ez_y, work2_d_ez_x,
                        work2_d_ex_z, work2_d_ex_y, work2_d_ey_x,
                        lam_ex, lam_ey, lam_ez,
                        lam_d_ex, lam_d_ey, lam_d_ez,
                        ctx.c, ctx.rdz, ctx.rdy, ctx.rdx,
                        n_shots, nz, ny, nx,
                        ctx.fd_pad[0], ctx.fd_pad[2], ctx.fd_pad[4],
                    )

        else:
            for t in range(nt - 1, -1, -1):
                if n_rec > 0:
                    ext.born_record_grad_r(
                        (lam_d_ex, lam_d_ey, lam_d_ez)[ctx.receiver_component],
                        grad_r, rec_i, t, n_shots, n_rec, numel,
                    )
                if n_bg_rec > 0:
                    ext.born_record_grad_r(
                        (lam_ex, lam_ey, lam_ez)[ctx.receiver_component],
                        grad_r_bg, bg_rec_i, t, n_shots, n_bg_rec, numel,
                    )
                if n_src > 0:
                    ext.born_record_grad_f(
                        (lam_ex, lam_ey, lam_ez)[ctx.source_component],
                        (lam_d_ex, lam_d_ey, lam_d_ez)[ctx.source_component],
                        grad_f_bg, grad_f_sc, src_i, t, n_shots, n_src, numel,
                    )
                snap_off = ex_storage.snap_offset(t // ctx.grad_stride)
                if t % ctx.grad_stride == 0:
                    ext.born_coeff_grad(
                        lam_ex, lam_ey, lam_ez, lam_d_ex, lam_d_ey, lam_d_ez,
                        ex_store, ey_store, ez_store,
                        curl_x_store, curl_y_store, curl_z_store,
                        dex_store, dey_store, dez_store,
                        dcurl_x_store, dcurl_y_store, dcurl_z_store,
                        grad_ca, grad_cb, grad_dca, grad_dcb,
                        scale, snap_off,
                        n_shots, nz, ny, nx,
                        *ctx.fd_pad,
                    )
                ext.born_adjoint_e_stage1(
                    ca_p, cb_p, dca_p, dcb_p,
                    lam_ex, lam_ey, lam_ez, lam_d_ex, lam_d_ey, lam_d_ez,
                    m_lambda_hy_z, m_lambda_hz_y, m_lambda_hz_x,
                    m_lambda_hx_z, m_lambda_hx_y, m_lambda_hy_x,
                    dm_lambda_hy_z, dm_lambda_hz_y, dm_lambda_hz_x,
                    dm_lambda_hx_z, dm_lambda_hx_y, dm_lambda_hy_x,
                    work_hy_z, work_hz_y, work_hz_x,
                    work_hx_z, work_hx_y, work_hy_x,
                    work_dhy_z, work_dhz_y, work_dhz_x,
                    work_dhx_z, work_dhx_y, work_dhy_x,
                    az, bz, ay, by, ax, bx, kz, ky, kx,
                    n_shots, nz, ny, nx,
                    ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                    ctx.pml_x0, ctx.pml_x1,
                    *ctx.fd_pad,
                    ctx.ca_batched, ctx.cb_batched,
                    ctx.dca_batched, ctx.dcb_batched,
                )
                ext.born_adjoint_e_stage2(
                    work_hy_z, work_hz_y, work_hz_x,
                    work_hx_z, work_hx_y, work_hy_x,
                    work_dhy_z, work_dhz_y, work_dhz_x,
                    work_dhx_z, work_dhx_y, work_dhy_x,
                    lam_hy, lam_hz, lam_hx,
                    lam_dhy, lam_dhz, lam_dhx,
                    ctx.c, ctx.rdz, ctx.rdy, ctx.rdx,
                    n_shots, nz, ny, nx,
                    ctx.fd_pad[0], ctx.fd_pad[2], ctx.fd_pad[4],
                )
                ext.born_adjoint_h_stage1(
                    cq_p, dcq_p,
                    lam_hx, lam_hy, lam_hz,
                    lam_dhx, lam_dhy, lam_dhz,
                    m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x,
                    m_lambda_ex_z, m_lambda_ex_y, m_lambda_ey_x,
                    dm_lambda_ey_z, dm_lambda_ez_y, dm_lambda_ez_x,
                    dm_lambda_ex_z, dm_lambda_ex_y, dm_lambda_ey_x,
                    work2_ey_z, work2_ez_y, work2_ez_x,
                    work2_ex_z, work2_ex_y, work2_ey_x,
                    work2_d_ey_z, work2_d_ez_y, work2_d_ez_x,
                    work2_d_ex_z, work2_d_ex_y, work2_d_ey_x,
                    azh, bzh, ayh, byh, axh, bxh, kzh, kyh, kxh,
                    n_shots, nz, ny, nx,
                    ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
                    ctx.pml_x0, ctx.pml_x1,
                    *ctx.fd_pad,
                    ctx.cq_batched, ctx.dcq_batched,
                )
                if t % ctx.grad_stride == 0:
                    ext.born_cq_grad(
                        lam_hx, lam_hy, lam_hz,
                        lam_dhx, lam_dhy, lam_dhz,
                        dey_dz_store, dez_dy_store, dez_dx_store,
                        dex_dz_store, dex_dy_store, dey_dx_store,
                        ddey_dz_store, ddez_dy_store, ddez_dx_store,
                        ddex_dz_store, ddex_dy_store, ddey_dx_store,
                        grad_cq, grad_dcq,
                        t, ctx.grad_stride, scale, snap_off,
                        n_shots, nz, ny, nx,
                        *ctx.fd_pad,
                    )
                ext.born_adjoint_h_stage2(
                    work2_ey_z, work2_ez_y, work2_ez_x,
                    work2_ex_z, work2_ex_y, work2_ey_x,
                    work2_d_ey_z, work2_d_ez_y, work2_d_ez_x,
                    work2_d_ex_z, work2_d_ex_y, work2_d_ey_x,
                    lam_ex, lam_ey, lam_ez,
                    lam_d_ex, lam_d_ey, lam_d_ez,
                    ctx.c, ctx.rdz, ctx.rdy, ctx.rdx,
                    n_shots, nz, ny, nx,
                    ctx.fd_pad[0], ctx.fd_pad[2], ctx.fd_pad[4],
                )

        if not ctx.ca_batched:
            grad_ca = grad_ca.sum(0, keepdim=True)
        if not ctx.cb_batched:
            grad_cb = grad_cb.sum(0, keepdim=True)
        if not ctx.cq_batched:
            grad_cq = grad_cq.sum(0, keepdim=True)
        if not ctx.dca_batched:
            grad_dca = grad_dca.sum(0, keepdim=True)
        if not ctx.dcb_batched:
            grad_dcb = grad_dcb.sum(0, keepdim=True)
        if not ctx.dcq_batched:
            grad_dcq = grad_dcq.sum(0, keepdim=True)

        return (
            grad_ca, grad_cb, grad_cq, grad_dca, grad_dcb, grad_dcq,
            grad_f_bg, grad_f_sc,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None,
        )


def em3d_born(
    epsilon,
    sigma,
    mu,
    epsilon_scatter,
    sigma_scatter,
    mu_scatter,
    grid_spacing,
    dt,
    source_amplitudes=None,
    source_locations=None,
    receiver_locations=None,
    bg_receiver_locations=None,
    accuracy=2,
    pml_width=20,
    nt=None,
    storage="auto",
    sample_steps=1,
    ckpt_steps=None,
    source_component="ey",
    receiver_component="ey",
):
    """3D electromagnetic Born forward modelling / FWI primitive.

    A background field is propagated on the staggered Yee grid exactly as
    :func:`em3d` (same ca/cb/cq coefficients, PML, FD order, source
    injection and receiver recording), and a scattered field is driven by
    the first-order Born scattering sources linear in the parameter
    perturbations ``(epsilon_scatter, sigma_scatter, mu_scatter)``.  The
    scattered wavefield is linear in the perturbations and differentiable
    w.r.t. both the background and the scatter models.

    Args:
        epsilon: Relative permittivity background [nz, ny, nx]
            (or [1, nz, ny, nx]).
        sigma: Conductivity background (S/m).
        mu: Relative permeability background.
        epsilon_scatter: Permittivity perturbation model.
        sigma_scatter: Conductivity perturbation model.
        mu_scatter: Permeability perturbation model (None keeps the
            default: mu is fixed, ``dcq = 0``).
        grid_spacing: Cell size (scalar or [dz, dy, dx]).
        dt: Time step interval (s).
        source_amplitudes: [n_shots, n_src, nt] source amplitudes, or None.
        source_locations: [n_shots, n_src, 3] (z, y, x) source locations.
        receiver_locations: [n_shots, n_rec, 3] scattered-field receiver
            locations.
        bg_receiver_locations: Optional [n_shots, n_bg_rec, 3] receiver
            locations for the background field (recorded from the same
            ``receiver_component``).
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
        (receiver_amplitudes, bg_receiver_amplitudes): the scattered
        ``[nt, n_shots, n_rec]`` traces and the background
        ``[nt, n_shots, n_bg_rec]`` traces (empty when
        ``bg_receiver_locations`` is None).
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
    check_cfl(grid_spacing, dt, max_vel, "em3d_born", c_max=1.0)

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
    if bg_receiver_locations is not None:
        (_,), _, bg_rec_i = extract_survey_3d(
            [epsilon],
            None,
            bg_receiver_locations,
            fd_pad,
            pml_w,
            n_shots,
            device,
            dtype,
        )
    else:
        bg_rec_i = torch.empty((n_shots, 0), dtype=torch.int64, device=device)

    def _pad_scatter(model):
        if model is None:
            return None
        (m_p,), _, _ = extract_survey_3d(
            [model], None, None, fd_pad, pml_w, n_shots, device, dtype,
            pad_modes=["constant"],
        )
        return m_p

    eps_sc_p = _pad_scatter(epsilon_scatter)
    sig_sc_p = _pad_scatter(sigma_scatter)
    if eps_sc_p is None:
        eps_sc_p = torch.zeros_like(epsilon_p)
    if sig_sc_p is None:
        sig_sc_p = torch.zeros_like(sigma_p)

    ca_p, cb_p, cq_p = _compile_material_coefficients(epsilon_p, sigma_p, mu_p, dt)
    # first-order coefficient linearizations (linearize_material_coefficients)
    cb_sq = cb_p * cb_p
    dca_p = (
        (1.0 - ca_p) * cb_p / dt * EPS0 * eps_sc_p
        - 0.5 * (1.0 + ca_p) * cb_p * sig_sc_p
    )
    dcb_p = -cb_sq / dt * EPS0 * eps_sc_p - 0.5 * cb_sq * sig_sc_p
    if mu_scatter is not None:
        mu_sc_p = _pad_scatter(mu_scatter)
        if mu_sc_p is None:
            mu_sc_p = torch.zeros_like(mu_p)
        dcq_p = -cq_p * mu_sc_p / mu_p
    else:
        dcq_p = torch.zeros_like(cq_p)

    nz, ny, nx = epsilon_p.shape[-3:]
    numel = nz * ny * nx
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
        cb_flat = cb_p.reshape(-1, numel).expand(n_shots, -1)
        cb_at_src = cb_flat.gather(1, src_i_masked)
        dcb_flat = dcb_p.reshape(-1, numel).expand(n_shots, -1)
        dcb_at_src = dcb_flat.gather(1, src_i_masked)
        f_bg = (
            amp.permute(2, 0, 1) * cb_at_src.unsqueeze(0) * source_coeff
        ).contiguous()
        f_sc = (
            amp.permute(2, 0, 1) * dcb_at_src.unsqueeze(0) * source_coeff
        ).contiguous()
    elif src_i.shape[1] > 0:
        # located sources without amplitudes: inject zero (no-op) amplitudes
        f_bg = torch.zeros(
            nt_inner, n_shots, src_i.shape[1], device=device, dtype=dtype
        )
        f_sc = torch.zeros_like(f_bg)
    else:
        f_bg = torch.empty(0, device=device, dtype=dtype)
        f_sc = torch.empty(0, device=device, dtype=dtype)

    rdz, rdy, rdx = 1.0 / dz, 1.0 / dy, 1.0 / dx
    pml_z0, pml_z1 = fd_pad[0] + pml_w[0], nz - fd_pad[1] - pml_w[1]
    pml_y0, pml_y1 = fd_pad[2] + pml_w[2], ny - fd_pad[3] - pml_w[3]
    pml_x0, pml_x1 = fd_pad[4] + pml_w[4], nx - fd_pad[5] - pml_w[5]
    ca_batched = 1 if ca_p.shape[0] == n_shots else 0
    cb_batched = 1 if cb_p.shape[0] == n_shots else 0
    cq_batched = 1 if cq_p.shape[0] == n_shots else 0
    dca_batched = 1 if dca_p.shape[0] == n_shots else 0
    dcb_batched = 1 if dcb_p.shape[0] == n_shots else 0
    dcq_batched = 1 if dcq_p.shape[0] == n_shots else 0

    storage_mode = resolve_storage(storage)
    grad_stride = check_sample_steps(sample_steps)
    storage_enabled = storage_mode == "auto" and torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (
            epsilon,
            sigma,
            mu,
            epsilon_scatter,
            sigma_scatter,
            mu_scatter,
            source_amplitudes,
        )
    )
    checkpoint_every, segments, n_snap, n_ckpt = storage_plan(
        nt_inner, N_STATE, grad_stride, N_STREAMS, storage_enabled,
        ckpt_steps=ckpt_steps,
    )

    def _mk_storage():
        return SnapshotStorage(
            _ext, n_snap, n_shots, nz, ny * nx, dtype, device,
        )

    if storage_enabled:
        ex_storage = _mk_storage()
        ey_storage = _mk_storage()
        ez_storage = _mk_storage()
        curl_x_storage = _mk_storage()
        curl_y_storage = _mk_storage()
        curl_z_storage = _mk_storage()
        dex_storage = _mk_storage()
        dey_storage = _mk_storage()
        dez_storage = _mk_storage()
        dcurl_x_storage = _mk_storage()
        dcurl_y_storage = _mk_storage()
        dcurl_z_storage = _mk_storage()
        dey_dz_storage = _mk_storage()
        dez_dy_storage = _mk_storage()
        dez_dx_storage = _mk_storage()
        dex_dz_storage = _mk_storage()
        dex_dy_storage = _mk_storage()
        dey_dx_storage = _mk_storage()
        ddey_dz_storage = _mk_storage()
        ddez_dy_storage = _mk_storage()
        ddez_dx_storage = _mk_storage()
        ddex_dz_storage = _mk_storage()
        ddex_dy_storage = _mk_storage()
        ddey_dx_storage = _mk_storage()
        ckpt_state = (
            torch.zeros(
                n_ckpt, N_STATE, n_shots, nz, ny, nx,
                device=device, dtype=dtype,
            )
            if n_ckpt > 0
            else None
        )
    else:
        ex_storage = ey_storage = ez_storage = None
        curl_x_storage = curl_y_storage = curl_z_storage = None
        dex_storage = dey_storage = dez_storage = None
        dcurl_x_storage = dcurl_y_storage = dcurl_z_storage = None
        dey_dz_storage = dez_dy_storage = dez_dx_storage = None
        dex_dz_storage = dex_dy_storage = dey_dx_storage = None
        ddey_dz_storage = ddez_dy_storage = ddez_dx_storage = None
        ddex_dz_storage = ddex_dy_storage = ddey_dx_storage = None
        ckpt_state = None

    r, r_bg = BornEM3DFunc.apply(
        ca_p, cb_p, cq_p, dca_p, dcb_p, dcq_p,
        f_bg, f_sc,
        src_i, rec_i, bg_rec_i,
        profs, c, fd_pad,
        rdz, rdy, rdx,
        nt_inner,
        pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
        ca_batched, cb_batched, cq_batched,
        dca_batched, dcb_batched, dcq_batched,
        grad_stride,
        ex_storage, ey_storage, ez_storage,
        curl_x_storage, curl_y_storage, curl_z_storage,
        dex_storage, dey_storage, dez_storage,
        dcurl_x_storage, dcurl_y_storage, dcurl_z_storage,
        dey_dz_storage, dez_dy_storage, dez_dx_storage,
        dex_dz_storage, dex_dy_storage, dey_dx_storage,
        ddey_dz_storage, ddez_dy_storage, ddez_dx_storage,
        ddex_dz_storage, ddex_dy_storage, ddey_dx_storage,
        source_component, receiver_component,
        ckpt_state, checkpoint_every, segments,
    )
    return r, r_bg
