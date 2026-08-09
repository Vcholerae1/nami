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
from ._common import EPS0, MU0, _compile_material_coefficients
from .em3d import _normalize_component, _set_em_pml_profiles_3d

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

# Full N_STATE names: keys of the state dicts returned by ``return_state``
# and accepted by ``initial_state`` (for split-run continuation).
_WAVEFIELD_NAMES = (
    "ex", "ey", "ez",
    "hx", "hy", "hz",
    "m_ey_z", "m_ez_y", "m_ez_x", "m_ex_z", "m_ex_y", "m_ey_x",
    "m_hy_z", "m_hz_y", "m_hz_x", "m_hx_z", "m_hx_y", "m_hy_x",
    "d_ex", "d_ey", "d_ez",
    "d_hx", "d_hy", "d_hz",
    "dm_ey_z", "dm_ez_y", "dm_ez_x", "dm_ex_z", "dm_ex_y", "dm_ey_x",
    "dm_hy_z", "dm_hz_y", "dm_hz_x", "dm_hx_z", "dm_hx_y", "dm_hy_x",
)
_CALLBACK_FIELDS = (
    "ex", "ey", "ez", "hx", "hy", "hz",
    "d_ex", "d_ey", "d_ez", "d_hx", "d_hy", "d_hz",
)


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
        init_state,          # [N_STATE, n_shots, nz, ny, nx] initial state or None
        final_state,         # [N_STATE, n_shots, nz, ny, nx] output buffer or None
        forward_callback,    # cb(t, nt, background fields..., scattered fields...)
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

        profs_c = [p.contiguous() for p in profs]
        # checkpointed forward stores no snapshots (the backward replay
        # regenerates them); full storage writes every sampled step.
        store = 0 if segments else (1 if ex_storage is not None else 0)
        ext.forward_loop(
            ca_p, cb_p, cq_p, dca_p, dcb_p, dcq_p,
            [ex, ey, ez], [d_ex, d_ey, d_ez],
            [hx, hy, hz], [d_hx, d_hy, d_hz],
            [m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x],
            [dm_ey_z, dm_ez_y, dm_ez_x, dm_ex_z, dm_ex_y, dm_ey_x],
            [m_hy_z, m_hz_y, m_hz_x, m_hx_z, m_hx_y, m_hy_x],
            [dm_hy_z, dm_hz_y, dm_hz_x, dm_hx_z, dm_hx_y, dm_hy_x],
            [dey_dz_store, dez_dy_store, dez_dx_store,
             dex_dz_store, dex_dy_store, dey_dx_store,
             ddey_dz_store, ddez_dy_store, ddez_dx_store,
             ddex_dz_store, ddex_dy_store, ddey_dx_store],
            [ex_store, ey_store, ez_store,
             curl_x_store, curl_y_store, curl_z_store,
             dex_store, dey_store, dez_store,
             dcurl_x_store, dcurl_y_store, dcurl_z_store],
            profs_c, c,
            f_bg, f_sc, src_i, r, rec_i, r_bg, bg_rec_i,
            rdz, rdy, rdx,
            nt, grad_stride,
            pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
            fd_pad,
            ca_batched, cb_batched, cq_batched,
            dca_batched, dcb_batched, dcq_batched,
            store, source_component, receiver_component,
            checkpoint_every, ckpt_state,
            init_state, final_state, forward_callback, callback_frequency,
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
        n_shots = int(src_i.shape[0])
        nz, ny, nx = ca_p.shape[-3:]
        n_src, n_rec, n_bg_rec = (
            src_i.shape[1], rec_i.shape[1], bg_rec_i.shape[1],
        )
        nt = ctx.nt

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

        profs_c = [p.contiguous() for p in ctx.profs]
        # integral sampling: each snapshot represents
        # `grad_stride` time steps of the model-gradient integral.
        scale = float(ctx.grad_stride)
        segments = ctx.segments
        # Checkpointed backward: per segment the C++ loop restores the
        # wavefield state, replays the forward steps to regenerate the
        # snapshots, then runs the adjoint.  Full storage: straight adjoint.
        segments_t = (
            torch.tensor(segments, dtype=torch.int64)
            if segments
            else torch.empty(0, 2, dtype=torch.int64)
        )
        if segments:
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
            e_f = [ex_f, ey_f, ez_f]
            d_e_f = [d_ex_f, d_ey_f, d_ez_f]
            h_f = [hx_f, hy_f, hz_f]
            d_h_f = [d_hx_f, d_hy_f, d_hz_f]
            m_h_f = [m_ey_z_f, m_ez_y_f, m_ez_x_f, m_ex_z_f, m_ex_y_f, m_ey_x_f]
            dm_h_f = [dm_ey_z_f, dm_ez_y_f, dm_ez_x_f, dm_ex_z_f, dm_ex_y_f, dm_ey_x_f]
            m_e_f = [m_hy_z_f, m_hz_y_f, m_hz_x_f, m_hx_z_f, m_hx_y_f, m_hy_x_f]
            dm_e_f = [dm_hy_z_f, dm_hz_y_f, dm_hz_x_f, dm_hx_z_f, dm_hx_y_f, dm_hy_x_f]
        else:
            e_f = d_e_f = h_f = d_h_f = []
            m_h_f = dm_h_f = m_e_f = dm_e_f = []
        ext.adjoint_loop(
            ca_p, cb_p, cq_p, dca_p, dcb_p, dcq_p,
            lam_ex, lam_ey, lam_ez,
            lam_d_ex, lam_d_ey, lam_d_ez,
            lam_hx, lam_hy, lam_hz,
            lam_dhx, lam_dhy, lam_dhz,
            [m_lambda_hy_z, m_lambda_hz_y, m_lambda_hz_x,
             m_lambda_hx_z, m_lambda_hx_y, m_lambda_hy_x],
            [dm_lambda_hy_z, dm_lambda_hz_y, dm_lambda_hz_x,
             dm_lambda_hx_z, dm_lambda_hx_y, dm_lambda_hy_x],
            [work_hy_z, work_hz_y, work_hz_x, work_hx_z, work_hx_y, work_hy_x],
            [work_dhy_z, work_dhz_y, work_dhz_x, work_dhx_z, work_dhx_y, work_dhy_x],
            [m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x,
             m_lambda_ex_z, m_lambda_ex_y, m_lambda_ey_x],
            [dm_lambda_ey_z, dm_lambda_ez_y, dm_lambda_ez_x,
             dm_lambda_ex_z, dm_lambda_ex_y, dm_lambda_ey_x],
            [work2_ey_z, work2_ez_y, work2_ez_x, work2_ex_z, work2_ex_y, work2_ey_x],
            [work2_d_ey_z, work2_d_ez_y, work2_d_ez_x,
             work2_d_ex_z, work2_d_ex_y, work2_d_ey_x],
            [dey_dz_store, dez_dy_store, dez_dx_store,
             dex_dz_store, dex_dy_store, dey_dx_store,
             ddey_dz_store, ddez_dy_store, ddez_dx_store,
             ddex_dz_store, ddex_dy_store, ddey_dx_store],
            [ex_store, ey_store, ez_store,
             curl_x_store, curl_y_store, curl_z_store,
             dex_store, dey_store, dez_store,
             dcurl_x_store, dcurl_y_store, dcurl_z_store],
            profs_c,
            ctx.c,
            grad_r, rec_i,
            grad_r_bg, bg_rec_i,
            grad_f_bg, grad_f_sc, src_i,
            f_bg, f_sc,
            e_f, d_e_f, h_f, d_h_f, m_h_f, dm_h_f, m_e_f, dm_e_f,
            grad_ca, grad_cb, grad_cq, grad_dca, grad_dcb, grad_dcq,
            ctx.rdz, ctx.rdy, ctx.rdx, scale,
            nt, ctx.grad_stride,
            ctx.pml_z0, ctx.pml_z1, ctx.pml_y0, ctx.pml_y1,
            ctx.pml_x0, ctx.pml_x1,
            ctx.fd_pad,
            ctx.ca_batched, ctx.cb_batched, ctx.cq_batched,
            ctx.dca_batched, ctx.dcb_batched, ctx.dcq_batched,
            ctx.source_component, ctx.receiver_component,
            segments_t, ctx.ckpt_state,
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
            None, None, None, None, None, None,
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
    forward_callback=None,
    callback_frequency=1,
    return_state=False,
    initial_state=None,
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
        forward_callback: called every ``callback_frequency`` steps with a
            ``CallbackState`` (deepwave-style) exposing the current padded
            wavefields via ``state.get_wavefield(name, view)`` — useful for
            RTM imaging conditions, illumination accumulation, monitoring.
        callback_frequency: call ``forward_callback`` every N time steps.
        return_state: if True, append a dict of the FINAL padded wavefield
            state in N_STATE order
            (keys ``ex``, ``ey``, ``ez``, ``hx``, ``hy``, ``hz``, the
            bg/scatter PML memory variables, ``d_ex``, ``d_ey``, ``d_ez``,
            ``d_hx``, ``d_hy``, ``d_hz``) suitable for continuation via
            ``initial_state``.
        initial_state: a dict of initial wavefield state (padded grid, keys
            as in the ``return_state=True`` output) to continue a previous
            run.  The 3D Born layout is a flat-buffer (non-ring) staggered
            scheme with no "previous time" slot.  Missing keys are
            zero-filled; a complete state dict (every key, including the PML
            memory variables) makes a split run bitwise match a one-shot
            run, while a partial dict restores only the given fields with
            the remaining state starting from zero.
            State I/O is for forward continuation; autograd does not propagate
            across the boundary between runs.  State dicts are ephemeral
            runtime snapshots: they may be passed back only to the same
            propagator with the same model layout and nami version, and are
            not a stable long-term checkpoint format.

    Returns:
        Scattered ``[nt, n_shots, n_rec]`` traces.  When
        ``bg_receiver_locations`` is provided, returns ``(r, r_bg)``;
        with ``return_state=True`` the state dict is appended.
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

    amp = prepare_source_amplitudes(
        source_amplitudes, n_shots, src_i.shape[1], nt_inner,
        device=device, dtype=dtype,
    )

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
    else:
        f_bg = torch.empty(0, device=device, dtype=dtype)
        f_sc = torch.empty(0, device=device, dtype=dtype)

    rdz, rdy, rdx = 1.0 / dz, 1.0 / dy, 1.0 / dx
    pml_z0, pml_z1 = fd_pad[0] + pml_w[0], nz - fd_pad[1] - pml_w[1]
    pml_y0, pml_y1 = fd_pad[2] + pml_w[2], ny - fd_pad[3] - pml_w[3]
    pml_x0, pml_x1 = fd_pad[4] + pml_w[4], nx - fd_pad[5] - pml_w[5]
    # Linearised coefficients combine background and scatter tensors, so use
    # the post-broadcast tensors passed to CUDA to determine batching.
    ca_batched = int(is_shot_batched(ca_p, n_shots, spatial_ndim=3))
    cb_batched = int(is_shot_batched(cb_p, n_shots, spatial_ndim=3))
    cq_batched = int(is_shot_batched(cq_p, n_shots, spatial_ndim=3))
    dca_batched = int(is_shot_batched(dca_p, n_shots, spatial_ndim=3))
    dcb_batched = int(is_shot_batched(dcb_p, n_shots, spatial_ndim=3))
    dcq_batched = int(is_shot_batched(dcq_p, n_shots, spatial_ndim=3))

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
