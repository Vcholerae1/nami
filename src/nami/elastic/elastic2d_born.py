"""nami elastic2d Born: first-order Born scattering for the 2D velocity-stress grid.

Self-derived (no external reference): linearizing the velocity-stress system
(bg fields un-superscripted, scattered fields ``d``-prefixed)

    vy  += by*dt*(DIFFYH1(syy) + DIFFX1(sxy))
    vx  += bx*dt*(DIFFXH1(sxx) + DIFFY1(sxy))
    syy += dt*(lamb*(dvydy + dvxdx) + 2*mu*dvydy)
    sxx += dt*(lamb*(dvydy + dvxdx) + 2*mu*dvxdx)
    sxy += dt*mu_yx*(DIFFXH1(vy) + DIFFYH1(vx))

around the background gives the scattering sources

    dvy  += by*dt*dw_y  + dby*dt*w_y           dvx  += bx*dt*dw_x  + dbx*dt*w_x
    dsyy += dt*(lamb*dssum + 2*mu*ddvydy) + dt*(dlamb*ssum + 2*dmu*dvydy)
    dsxx += dt*(lamb*dssum + 2*mu*ddvxdx) + dt*(dlamb*ssum + 2*dmu*dvxdx)
    dsxy += dt*mu_yx*dw_sum + dt*dmu_yx*w_sum

where ``w_y/w_x`` are the background velocity-update sums (with their PML
memories), ``ssum = dvydy + dvxdx``, ``w_sum = DIFFXH1(vy) + DIFFYH1(vx)``,
and the scattered field propagates through the background operator with the
same CPML memories and profiles.  ``dmu_yx``/``dby``/``dbx`` are the exact
first-order linearizations of :func:`prepare_parameters` w.r.t. the
(``mu_scatter``, ``buoyancy_scatter``) perturbations (harmonic-mean mu and
inverse-density buoyancy), computed on the Python side.

The adjoint is the exact discrete transpose of the coupled
(background + scattered) system (same structure as :mod:`elastic2d` with
separate adjoints for each field), so gradients flow to both the background
and the scatter models.

Return convention: ``[nt, n_shots, n_rec]`` scattered pressure traces
``-(dsyy + dsxx) / 2``, matching elastic2d's pressure receivers.
"""

import nami_elastic2d_born as _ext
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
    extract_survey_2d,
    materialize_batched_group,
    prepare_source_amplitudes,
)
from .elastic2d import (
    _set_elastic_pml_profiles,
    lambmubuoyancy_to_vpvsrho,
    prepare_parameters,
)

# Checkpoint state layout: the elastic2d 13 buffers plus their scattered
# counterparts (all updated in place, so the buffers ARE the wavefield state):
#   [0]  vy            [13] dvy
#   [1]  vx            [14] dvx
#   [2]  syy           [15] dsyy
#   [3]  sxx           [16] dsxx
#   [4]  sxy           [17] dsxy
#   [5]  m_vyy         [18] dm_vyy
#   [6]  m_vxx         [19] dm_vxx
#   [7]  m_vxy         [20] dm_vxy
#   [8]  m_vyx         [21] dm_vyx
#   [9]  m_sigmayyy    [22] dm_sigmayyy
#   [10] m_sigmaxyx    [23] dm_sigmaxyx
#   [11] m_sigmaxyy    [24] dm_sigmaxyy
#   [12] m_sigmaxxx    [25] dm_sigmaxxx
N_STATE = 26
N_STREAMS = 10

# Full N_STATE names: keys of the state dicts returned by ``return_state``
# and accepted by ``initial_state`` (for split-run continuation).
_WAVEFIELD_NAMES = (
    "vy", "vx", "syy", "sxx", "sxy",
    "m_vyy", "m_vxx", "m_vxy", "m_vyx",
    "m_sigmayyy", "m_sigmaxyx", "m_sigmaxyy", "m_sigmaxxx",
    "dvy", "dvx", "dsyy", "dsxx", "dsxy",
    "dm_vyy", "dm_vxx", "dm_vxy", "dm_vyx",
    "dm_sigmayyy", "dm_sigmaxyx", "dm_sigmaxyy", "dm_sigmaxxx",
)
_CALLBACK_FIELDS = (
    "vy", "vx", "syy", "sxx", "sxy",
    "dvy", "dvx", "dsyy", "dsxx", "dsxy",
)


def _linearize_prepare_parameters(mu, buoyancy, dmu, db):
    """First-order linearization of :func:`prepare_parameters`.

    ``mu_yx`` is the harmonic mean of each 2x2 mu block, so
    ``dmu_yx = (mu_yx**2 / 4) * sum_i dmu_i / mu_i**2`` over the block;
    the buoyancy is ``1 / (arithmetic mean of rho = 1/b)``, so
    ``dbuoyancy_y = -buoyancy_y**2 * drho_y`` with ``drho = -db / b**2``
    (and the x analogue).  Zero where the base parameters are zero.
    """
    rfmax = 1 / torch.finfo(mu.dtype).max ** (1 / 2)

    mu_safe = torch.where(mu.abs() > rfmax, mu, torch.ones_like(mu))
    mask = (
        (rfmax < mu[..., 1:, 1:].abs())
        .logical_and(rfmax < mu[..., :-1, :-1].abs())
        .logical_and(rfmax < mu[..., 1:, :-1].abs())
        .logical_and(rfmax < mu[..., :-1, 1:].abs())
    )
    mu_yx_val = 4 / (
        1 / mu_safe[..., 1:, 1:]
        + 1 / mu_safe[..., :-1, :-1]
        + 1 / mu_safe[..., 1:, :-1]
        + 1 / mu_safe[..., :-1, 1:]
    )
    dmu_yx_val = (mu_yx_val ** 2 / 4) * (
        dmu[..., 1:, 1:] / mu_safe[..., 1:, 1:] ** 2
        + dmu[..., :-1, :-1] / mu_safe[..., :-1, :-1] ** 2
        + dmu[..., 1:, :-1] / mu_safe[..., 1:, :-1] ** 2
        + dmu[..., :-1, 1:] / mu_safe[..., :-1, 1:] ** 2
    )
    dmu_yx = torch.where(mask, dmu_yx_val, torch.zeros_like(dmu_yx_val))
    dmu_yx = torch.nn.functional.pad(dmu_yx, (0, 1, 0, 1))

    mask_b = rfmax < buoyancy.abs()
    buoyancy_safe = torch.where(mask_b, buoyancy, torch.ones_like(buoyancy))
    rho = torch.where(mask_b, 1 / buoyancy_safe, torch.zeros_like(buoyancy))
    drho = torch.where(mask_b, -db / buoyancy_safe ** 2, torch.zeros_like(buoyancy))

    rho_y = torch.nn.functional.pad(
        (rho[..., :-1, :] + rho[..., 1:, :]) / 2, (0, 0, 0, 1)
    )
    drho_y = torch.nn.functional.pad(
        (drho[..., :-1, :] + drho[..., 1:, :]) / 2, (0, 0, 0, 1)
    )
    mask_y = rfmax < rho_y.abs()
    rho_y_safe = torch.where(mask_y, rho_y, torch.ones_like(rho_y))
    buoyancy_y = torch.where(mask_y, 1 / rho_y_safe, torch.zeros_like(rho_y))
    dbuoyancy_y = torch.where(
        mask_y, -buoyancy_y ** 2 * drho_y, torch.zeros_like(drho_y)
    )

    rho_x = torch.nn.functional.pad((rho[..., :-1] + rho[..., 1:]) / 2, (0, 1))
    drho_x = torch.nn.functional.pad((drho[..., :-1] + drho[..., 1:]) / 2, (0, 1))
    mask_x = rfmax < rho_x.abs()
    rho_x_safe = torch.where(mask_x, rho_x, torch.ones_like(rho_x))
    buoyancy_x = torch.where(mask_x, 1 / rho_x_safe, torch.zeros_like(rho_x))
    dbuoyancy_x = torch.where(
        mask_x, -buoyancy_x ** 2 * drho_x, torch.zeros_like(drho_x)
    )

    return dmu_yx, dbuoyancy_y, dbuoyancy_x


class BornElasticFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        lamb_p, mu_p, mu_yx, buoyancy_y, buoyancy_x,
        dlamb_p, dmu_p, dmu_yx, dbuoyancy_y, dbuoyancy_x,
        amp,              # [n_shots, n_src, nt] pressure source amplitudes
        src_i, rec_i, bg_rec_i,
        profs, c,
        fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
        rdy, rdx, dtv,
        nt, grad_stride, n_shots, model_batched, scatter_batched,
        storage,         # list of 10 SnapshotStorage (one per stream) or None
        ckpt_state,      # [n_ckpt, N_STATE, n_shots, ny, nx] or None
        checkpoint_every,  # 0 = full storage; N = checkpoint every N steps
        segments,        # [(s0, s1)] replay segments; [] = full storage
        init_state,      # [N_STATE, n_shots, ny, nx] initial wavefield or None
        final_state,     # [N_STATE, n_shots, ny, nx] output buffer or None
        forward_callback,  # cb(t, nt, background fields..., scattered fields...)
        callback_frequency,  # call the callback every N steps
    ):
        ext = _ext
        device = lamb_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = lamb_p.dtype
        ny, nx = lamb_p.shape[-2:]
        n_src = src_i.shape[1]
        n_rec = rec_i.shape[1]
        n_bg_rec = bg_rec_i.shape[1]
        ny_nx = ny * nx

        # pressure-source pre-scaling: f = -amp * dt
        if amp.numel() > 0:
            f = (-amp.permute(2, 0, 1) * dtv).contiguous()
        else:
            f = torch.empty(0, device=device, dtype=dtype)

        z = lambda *shape: torch.zeros(shape, device=device, dtype=dtype)  # noqa: E731
        vy, vx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        dvy, dvx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        syy, sxx, sxy = z(n_shots, ny, nx), z(n_shots, ny, nx), z(n_shots, ny, nx)
        dsyy, dsxx, dsxy = z(n_shots, ny, nx), z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_sigmayyy, m_sigmaxyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_sigmaxyy, m_sigmaxxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        dm_sigmayyy, dm_sigmaxyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        dm_sigmaxyy, dm_sigmaxxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_vyy, m_vxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_vxy, m_vyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        dm_vyy, dm_vxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        dm_vxy, dm_vyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        r = z(nt, n_shots, n_rec)
        r_bg = z(nt, n_shots, n_bg_rec)
        if storage is not None:
            (dvydy_store, dvxdx_store, dvxy_store, dvydb_store, dvxdb_store,
             ddvydy_store, ddvxdx_store, ddvxy_store, ddvydb_store,
             ddvxdb_store) = [st.snap for st in storage]
        else:
            dummy = z(n_shots, ny, nx)
            (dvydy_store, dvxdx_store, dvxy_store, dvydb_store, dvxdb_store,
             ddvydy_store, ddvxdx_store, ddvxy_store, ddvydb_store,
             ddvxdb_store) = (dummy,) * 10

        ayh, byh, ay, by, axh, bxh, ax, bx = profs
        # N_STATE layout: [0] vy [1] vx [2] syy [3] sxx [4] sxy
        # [5] m_vyy [6] m_vxx [7] m_vxy [8] m_vyx
        # [9] m_sigmayyy [10] m_sigmaxyx [11] m_sigmaxyy [12] m_sigmaxxx
        # [13..25] scattered counterparts (dvy, dvx, dsyy, dsxx, dsxy, dm_*)
        state = [
            vy, vx, syy, sxx, sxy,
            m_vyy, m_vxx, m_vxy, m_vyx,
            m_sigmayyy, m_sigmaxyx, m_sigmaxyy, m_sigmaxxx,
            dvy, dvx, dsyy, dsxx, dsxy,
            dm_vyy, dm_vxx, dm_vxy, dm_vyx,
            dm_sigmayyy, dm_sigmaxyx, dm_sigmaxyy, dm_sigmaxxx,
        ]
        # Checkpointed forward skips global snapshot writes; replay
        # regenerates segment-local snaps via snap_off.
        store = 0 if segments else (1 if storage is not None else 0)
        ext.born_elastic_forward_loop(
            lamb_p, mu_p, mu_yx, buoyancy_y, buoyancy_x,
            dlamb_p, dmu_p, dmu_yx, dbuoyancy_y, dbuoyancy_x,
            state,
            dvydy_store, dvxdx_store, dvxy_store, dvydb_store, dvxdb_store,
            ddvydy_store, ddvxdx_store, ddvxy_store, ddvydb_store, ddvxdb_store,
            ayh, byh, ay, by, axh, bxh, ax, bx, c,
            f, src_i, r, rec_i, r_bg, bg_rec_i,
            fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
            rdy, rdx, dtv,
            nt, grad_stride,
            model_batched, scatter_batched,
            store, checkpoint_every, ckpt_state,
            init_state, final_state, forward_callback, callback_frequency,
        )

        ctx.ext = ext
        ctx.save_for_backward(
            lamb_p, mu_p, mu_yx, buoyancy_y, buoyancy_x,
            dlamb_p, dmu_p, dmu_yx, dbuoyancy_y, dbuoyancy_x,
            src_i, rec_i, bg_rec_i, f,
        )
        ctx.storage = storage
        ctx.profs = profs
        ctx.c = c
        ctx.fd_pad = (fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1)
        ctx.rdy, ctx.rdx, ctx.dtv = rdy, rdx, dtv
        ctx.nt, ctx.grad_stride = nt, grad_stride
        ctx.n_shots, ctx.ny, ctx.nx, ctx.ny_nx = n_shots, ny, nx, ny_nx
        ctx.n_src, ctx.n_rec, ctx.n_bg_rec = n_src, n_rec, n_bg_rec
        ctx.model_batched = model_batched
        ctx.scatter_batched = scatter_batched
        ctx.ckpt_state = ckpt_state
        ctx.checkpoint_every = checkpoint_every
        ctx.segments = segments
        return r, r_bg

    @staticmethod
    def backward(ctx, grad_r, grad_r_bg):
        if ctx.storage is None:
            raise RuntimeError(
                "elastic2d_born backward() requires snapshot storage: run the "
                "forward with an input requiring grad (and not under "
                "torch.no_grad())."
            )
        ext = ctx.ext
        storage = ctx.storage
        (lamb_p, mu_p, mu_yx, buoyancy_y, buoyancy_x,
         dlamb_p, dmu_p, dmu_yx, dbuoyancy_y, dbuoyancy_x,
         src_i, rec_i, bg_rec_i, f) = ctx.saved_tensors
        (dvydy_store, dvxdx_store, dvxy_store, dvydb_store, dvxdb_store,
         ddvydy_store, ddvxdx_store, ddvxy_store, ddvydb_store,
         ddvxdb_store) = [st.snap for st in storage]
        device = lamb_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = lamb_p.dtype
        n_shots, ny, nx = ctx.n_shots, ctx.ny, ctx.nx
        nt, grad_stride = ctx.nt, ctx.grad_stride
        # integral sampling: each snapshot represents
        # `grad_stride` time steps of the model-gradient integral.
        scale = float(grad_stride)

        if grad_r is None:
            grad_r = torch.zeros(nt, n_shots, ctx.n_rec, device=device, dtype=dtype)
        grad_r = grad_r.contiguous()
        if grad_r_bg is None:
            grad_r_bg = torch.zeros(
                nt, n_shots, ctx.n_bg_rec, device=device, dtype=dtype
            )
        grad_r_bg = grad_r_bg.contiguous()
        grad_f = torch.zeros(nt, n_shots, ctx.n_src, device=device, dtype=dtype)
        grad_lamb = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        grad_mu = torch.zeros_like(grad_lamb)
        grad_mu_yx = torch.zeros_like(grad_lamb)
        grad_dlamb = torch.zeros_like(grad_lamb)
        grad_dmu = torch.zeros_like(grad_lamb)
        grad_dmu_yx = torch.zeros_like(grad_lamb)
        grad_by = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        grad_bx = torch.zeros_like(grad_by)
        grad_dby = torch.zeros_like(grad_by)
        grad_dbx = torch.zeros_like(grad_by)

        z = lambda *shape: torch.zeros(shape, device=device, dtype=dtype)  # noqa: E731
        l_vy, l_vx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        l_dvy, l_dvx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        l_syy, l_sxx, l_sxy = z(n_shots, ny, nx), z(n_shots, ny, nx), z(n_shots, ny, nx)
        l_dsyy, l_dsxx, l_dsxy = (
            z(n_shots, ny, nx), z(n_shots, ny, nx), z(n_shots, ny, nx),
        )
        m_vyy, m_vxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_vxy, m_vyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        dm_vyy, dm_vxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        dm_vxy, dm_vyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        # the m_sigma* adjoint memories alternate (the C code's _t/_n trick)
        m_sig_a = [z(n_shots, ny, nx) for _ in range(4)]
        m_sig_b = [z(n_shots, ny, nx) for _ in range(4)]
        dm_sig_a = [z(n_shots, ny, nx) for _ in range(4)]
        dm_sig_b = [z(n_shots, ny, nx) for _ in range(4)]

        ayh, byh, ay, by, axh, bxh, ax, bx = ctx.profs
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
            f_vy, f_vx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            f_dvy, f_dvx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            f_syy, f_sxx, f_sxy = (
                z(n_shots, ny, nx), z(n_shots, ny, nx), z(n_shots, ny, nx),
            )
            f_dsyy, f_dsxx, f_dsxy = (
                z(n_shots, ny, nx), z(n_shots, ny, nx), z(n_shots, ny, nx),
            )
            f_m_sigmayyy, f_m_sigmaxyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            f_m_sigmaxyy, f_m_sigmaxxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            f_dm_sigmayyy, f_dm_sigmaxyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            f_dm_sigmaxyy, f_dm_sigmaxxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            f_m_vyy, f_m_vxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            f_m_vxy, f_m_vyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            f_dm_vyy, f_dm_vxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            f_dm_vxy, f_dm_vyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            state_f = [
                f_vy, f_vx, f_syy, f_sxx, f_sxy,
                f_m_vyy, f_m_vxx, f_m_vxy, f_m_vyx,
                f_m_sigmayyy, f_m_sigmaxyx, f_m_sigmaxyy, f_m_sigmaxxx,
                f_dvy, f_dvx, f_dsyy, f_dsxx, f_dsxy,
                f_dm_vyy, f_dm_vxx, f_dm_vxy, f_dm_vyx,
                f_dm_sigmayyy, f_dm_sigmaxyx, f_dm_sigmaxyy, f_dm_sigmaxxx,
            ]
        else:
            state_f = []
        ext.born_elastic_adjoint_loop(
            lamb_p, mu_p, mu_yx, buoyancy_y, buoyancy_x,
            dlamb_p, dmu_p, dmu_yx, dbuoyancy_y, dbuoyancy_x,
            dvydy_store, dvxdx_store, dvxy_store, dvydb_store, dvxdb_store,
            ddvydy_store, ddvxdx_store, ddvxy_store, ddvydb_store, ddvxdb_store,
            grad_lamb, grad_mu, grad_mu_yx, grad_dlamb, grad_dmu, grad_dmu_yx,
            grad_by, grad_bx, grad_dby, grad_dbx,
            grad_f, grad_r, grad_r_bg,
            src_i, rec_i, bg_rec_i, f,
            l_vy, l_vx, l_dvy, l_dvx,
            l_syy, l_sxx, l_sxy, l_dsyy, l_dsxx, l_dsxy,
            m_vyy, m_vxx, m_vxy, m_vyx,
            dm_vyy, dm_vxx, dm_vxy, dm_vyx,
            m_sig_a, m_sig_b, dm_sig_a, dm_sig_b,
            state_f,
            ayh, byh, ay, by, axh, bxh, ax, bx, ctx.c,
            ctx.fd_pad[0], ctx.fd_pad[1], ctx.fd_pad[2], ctx.fd_pad[3],
            ctx.rdy, ctx.rdx, ctx.dtv, scale,
            nt, grad_stride,
            ctx.model_batched, ctx.scatter_batched,
            segments_t, ctx.ckpt_state,
        )

        if not ctx.model_batched:
            grad_lamb = grad_lamb.sum(0, keepdim=True)
            grad_mu = grad_mu.sum(0, keepdim=True)
            grad_mu_yx = grad_mu_yx.sum(0, keepdim=True)
            grad_by = grad_by.sum(0, keepdim=True)
            grad_bx = grad_bx.sum(0, keepdim=True)
        if not ctx.scatter_batched:
            grad_dlamb = grad_dlamb.sum(0, keepdim=True)
            grad_dmu = grad_dmu.sum(0, keepdim=True)
            grad_dmu_yx = grad_dmu_yx.sum(0, keepdim=True)
            grad_dby = grad_dby.sum(0, keepdim=True)
            grad_dbx = grad_dbx.sum(0, keepdim=True)
        # grad through the f = -amp * dt pre-scaling
        grad_amp = (grad_f * (-ctx.dtv)).permute(1, 2, 0)
        # 36 forward() inputs: 11 model/source grads + 25 x None.
        return (
            grad_lamb, grad_mu, grad_mu_yx, grad_by, grad_bx,
            grad_dlamb, grad_dmu, grad_dmu_yx, grad_dby, grad_dbx,
            grad_amp,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None,
        )


def elastic2d_born(
    lamb,
    mu,
    buoyancy,
    lamb_scatter,
    mu_scatter,
    buoyancy_scatter,
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
    """2D elastic Born forward + adjoint (torch in/out).

    ``lamb``/``mu``/``buoyancy`` are the background Lamé/buoyancy models;
    ``lamb_scatter``/``mu_scatter``/``buoyancy_scatter`` are the parameter
    perturbations (any may be None for a zero perturbation).  The scattered
    wavefield is linear in the perturbations and differentiable w.r.t. both
    the background and the scatter models.

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
            The available wavefields are all background and scattered
            velocity/stress fields; PML memory variables stay internal.
        callback_frequency: call ``forward_callback`` every N time steps.
        return_state: if True, append the final state dict to the outputs.
            The state is a
            dict of the FINAL padded wavefield state (keys ``vy``, ``vx``,
            ``syy``, ``sxx``, ``sxy``, ``dvy``, ``dvx``, ``dsyy``, ``dsxx``,
            ``dsxy``, plus the PML memory variables) suitable for
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

    Returns scattered pressure traces ``[nt, n_shots, n_rec]``.  When
    ``bg_receiver_locations`` is provided, returns ``(r, r_bg)``; with
    ``return_state=True`` the state dict is appended.
    """
    accuracy = check_accuracy(accuracy)
    if not isinstance(grid_spacing, (list, tuple)):
        grid_spacing = [float(grid_spacing)] * 2
    grid_spacing = [float(g) for g in grid_spacing]
    pml_w = set_pml_width(pml_width, 2)
    fd_pad = [accuracy // 2, accuracy // 2 - 1] * 2  # [1, 0, 1, 0]
    device = lamb.device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = lamb.dtype
    c = staggered_diff1_coeffs(accuracy, dtype, device)

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

    (lamb_p, mu_p, buoy_p), src_i, rec_i = extract_survey_2d(
        [lamb, mu, buoyancy],
        source_locations,
        receiver_locations,
        fd_pad,
        pml_w,
        n_shots,
        device,
        dtype,
    )
    if bg_receiver_locations is not None:
        (_,), _, bg_rec_i = extract_survey_2d(
            [lamb], None, bg_receiver_locations, fd_pad, pml_w,
            n_shots, device, dtype,
        )
    else:
        bg_rec_i = torch.empty((n_shots, 0), dtype=torch.int64, device=device)
    ny, nx = lamb_p.shape[-2:]

    def _pad_scatter(model):
        if model is None:
            return None
        (m_p,), _, _ = extract_survey_2d(
            [model], None, None, fd_pad, pml_w, n_shots, device, dtype,
            pad_modes=["constant"],
        )
        return m_p

    dlamb_p = _pad_scatter(lamb_scatter)
    dmu_p = _pad_scatter(mu_scatter)
    db_p = _pad_scatter(buoyancy_scatter)
    if dlamb_p is None:
        dlamb_p = torch.zeros_like(lamb_p)
    if dmu_p is None:
        dmu_p = torch.zeros_like(mu_p)
    if db_p is None:
        db_p = torch.zeros_like(buoy_p)

    # max_vel is derived from the (unpadded) models and used both for the
    # CFL condition and the PML grading.
    vp, vs, _ = lambmubuoyancy_to_vpvsrho(lamb, mu, buoyancy)
    max_vel = max(vp.abs().max().item(), vs.abs().max().item())
    check_cfl(grid_spacing, dt, max_vel, "elastic2d_born")

    profiles = _set_elastic_pml_profiles(
        pml_w,
        fd_pad,
        float(dt),
        grid_spacing,
        max_vel,
        pml_freq,
        (ny, nx),
        dtype,
        device,
    )
    # [ay, by, ayh, byh, ax, bx, axh, bxh] -> kernel arg order
    profs = [
        profiles[2], profiles[3], profiles[0], profiles[1],
        profiles[6], profiles[7], profiles[4], profiles[5],
    ]

    # staggered parameters from the padded models (staggered-grid convention)
    mu_yx, buoyancy_y, buoyancy_x = prepare_parameters(mu_p, buoy_p)
    dmu_yx, dbuoyancy_y, dbuoyancy_x = _linearize_prepare_parameters(
        mu_p, buoy_p, dmu_p, db_p
    )

    amp = prepare_source_amplitudes(
        source_amplitudes, n_shots, src_i.shape[1], nt_inner,
        device=device, dtype=dtype,
    )

    storage_mode = resolve_storage(storage)
    grad_stride = check_sample_steps(sample_steps)
    storage_enabled = storage_mode == "auto" and torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (
            lamb, mu, buoyancy, lamb_scatter, mu_scatter,
            buoyancy_scatter, source_amplitudes,
        )
    )
    checkpoint_every, segments, n_snap, n_ckpt = storage_plan(
        nt_inner, N_STATE, grad_stride, N_STREAMS, storage_enabled,
        ckpt_steps=ckpt_steps,
    )
    # Each native flag governs a coefficient group.  Mixed public-model
    # batching is normalised by materialising only shared group members;
    # autograd still reduces those gradients to their original shapes.
    background_coeffs, model_batched = materialize_batched_group(
        [lamb_p, mu_p, mu_yx, buoyancy_y, buoyancy_x], n_shots
    )
    lamb_p, mu_p, mu_yx, buoyancy_y, buoyancy_x = background_coeffs
    scatter_coeffs = [
        dlamb_p, dmu_p, dmu_yx, dbuoyancy_y, dbuoyancy_x,
    ]
    scatter_coeffs, scatter_batched = materialize_batched_group(
        scatter_coeffs, n_shots
    )
    dlamb_p, dmu_p, dmu_yx, dbuoyancy_y, dbuoyancy_x = scatter_coeffs

    stores = None
    ckpt_state = None
    if storage_enabled:
        stores = [
            SnapshotStorage(_ext, n_snap, n_shots, ny, nx, dtype, device)
            for _ in range(N_STREAMS)
        ]
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

    r, r_bg = BornElasticFunc.apply(
        lamb_p, mu_p, mu_yx, buoyancy_y, buoyancy_x,
        dlamb_p, dmu_p, dmu_yx, dbuoyancy_y, dbuoyancy_x,
        amp, src_i, rec_i, bg_rec_i, profs, c,
        fd_pad[0], fd_pad[1], fd_pad[2], fd_pad[3],
        1.0 / grid_spacing[0],
        1.0 / grid_spacing[1],
        float(dt),
        nt_inner,
        grad_stride,
        n_shots,
        model_batched,
        scatter_batched,
        stores,
        ckpt_state,
        checkpoint_every,
        segments,
        init_state, final_state, cb, callback_frequency,
    )
    r = -r / 2
    r_bg = -r_bg / 2
    if return_state:
        state = unpack_state(final_state, _WAVEFIELD_NAMES)
        if bg_receiver_locations is not None:
            return r, r_bg, state
        return r, state
    if bg_receiver_locations is not None:
        return r, r_bg
    return r
