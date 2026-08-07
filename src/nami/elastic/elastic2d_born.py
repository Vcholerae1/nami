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

import math

import nami_born_em_el as _ext
import nami_elastic2d as _storage_ext
import torch

from ..common.fd import check_accuracy, staggered_diff1_coeffs
from ..common.storage import (
    SnapshotStorage,
    check_sample_steps,
    resolve_storage,
    storage_plan,
)
from ..common.survey import extract_survey_2d
from .elastic2d import (
    _set_elastic_pml_profiles,
    _set_pml_width,
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
        src_i, rec_i,
        profs, c,
        fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
        rdy, rdx, dtv,
        nt, grad_stride, n_shots, model_batched, scatter_batched,
        storage,         # list of 10 SnapshotStorage (one per stream) or None
        ckpt_state,      # [n_ckpt, N_STATE, n_shots, ny, nx] or None
        checkpoint_every,  # 0 = full storage; N = checkpoint every N steps
        segments,        # [(s0, s1)] replay segments; [] = full storage
    ):
        ext = _ext
        device = lamb_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = lamb_p.dtype
        ny, nx = lamb_p.shape[-2:]
        n_src = src_i.shape[1]
        n_rec = rec_i.shape[1]
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
        ckpt = ckpt_state is not None
        for t in range(nt):
            if ckpt and t > 0 and t % checkpoint_every == 0:
                k = t // checkpoint_every - 1
                ckpt_state[k, 0].copy_(vy)
                ckpt_state[k, 1].copy_(vx)
                ckpt_state[k, 2].copy_(syy)
                ckpt_state[k, 3].copy_(sxx)
                ckpt_state[k, 4].copy_(sxy)
                ckpt_state[k, 5].copy_(m_vyy)
                ckpt_state[k, 6].copy_(m_vxx)
                ckpt_state[k, 7].copy_(m_vxy)
                ckpt_state[k, 8].copy_(m_vyx)
                ckpt_state[k, 9].copy_(m_sigmayyy)
                ckpt_state[k, 10].copy_(m_sigmaxyx)
                ckpt_state[k, 11].copy_(m_sigmaxyy)
                ckpt_state[k, 12].copy_(m_sigmaxxx)
                ckpt_state[k, 13].copy_(dvy)
                ckpt_state[k, 14].copy_(dvx)
                ckpt_state[k, 15].copy_(dsyy)
                ckpt_state[k, 16].copy_(dsxx)
                ckpt_state[k, 17].copy_(dsxy)
                ckpt_state[k, 18].copy_(dm_vyy)
                ckpt_state[k, 19].copy_(dm_vxx)
                ckpt_state[k, 20].copy_(dm_vxy)
                ckpt_state[k, 21].copy_(dm_vyx)
                ckpt_state[k, 22].copy_(dm_sigmayyy)
                ckpt_state[k, 23].copy_(dm_sigmaxyx)
                ckpt_state[k, 24].copy_(dm_sigmaxyy)
                ckpt_state[k, 25].copy_(dm_sigmaxxx)
            if n_rec > 0:
                ext.born_record_pressure(
                    dsyy, dsxx, r, rec_i, t, n_shots, n_rec, ny_nx,
                )
            store = 0 if segments else (1 if storage is not None else 0)
            ext.born_step_velocity(
                vy, vx, syy, sxx, sxy, dvy, dvx, dsyy, dsxx, dsxy,
                m_sigmayyy, m_sigmaxyx, m_sigmaxyy, m_sigmaxxx,
                dm_sigmayyy, dm_sigmaxyx, dm_sigmaxyy, dm_sigmaxxx,
                buoyancy_y, buoyancy_x, dbuoyancy_y, dbuoyancy_x,
                dvydb_store, dvxdb_store, ddvydb_store, ddvxdb_store,
                ayh, byh, ay, by, axh, bxh, ax, bx, c,
                fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
                rdy, rdx, dtv, t, grad_stride,
                model_batched, scatter_batched, store,
            )
            ext.born_step_stress(
                vy, vx, dvy, dvx, syy, sxx, sxy, dsyy, dsxx, dsxy,
                m_vyy, m_vxx, m_vxy, m_vyx,
                dm_vyy, dm_vxx, dm_vxy, dm_vyx,
                lamb_p, mu_p, mu_yx, dlamb_p, dmu_p, dmu_yx,
                dvydy_store, dvxdx_store, dvxy_store,
                ddvydy_store, ddvxdx_store, ddvxy_store,
                ayh, byh, ay, by, axh, bxh, ax, bx, c,
                fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
                rdy, rdx, dtv, t, grad_stride,
                model_batched, scatter_batched, store,
            )
            if n_src > 0:
                ext.born_inject_pressure(syy, sxx, f, src_i, t, n_shots, n_src, ny_nx)

        ctx.ext = ext
        ctx.save_for_backward(
            lamb_p, mu_p, mu_yx, buoyancy_y, buoyancy_x,
            dlamb_p, dmu_p, dmu_yx, dbuoyancy_y, dbuoyancy_x,
            src_i, rec_i, f,
        )
        ctx.storage = storage
        ctx.profs = profs
        ctx.c = c
        ctx.fd_pad = (fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1)
        ctx.rdy, ctx.rdx, ctx.dtv = rdy, rdx, dtv
        ctx.nt, ctx.grad_stride = nt, grad_stride
        ctx.n_shots, ctx.ny, ctx.nx, ctx.ny_nx = n_shots, ny, nx, ny_nx
        ctx.n_src, ctx.n_rec = n_src, n_rec
        ctx.model_batched = model_batched
        ctx.scatter_batched = scatter_batched
        ctx.ckpt_state = ckpt_state
        ctx.checkpoint_every = checkpoint_every
        ctx.segments = segments
        return r

    @staticmethod
    def backward(ctx, grad_r):
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
         src_i, rec_i, f) = ctx.saved_tensors
        (dvydy_store, dvxdx_store, dvxy_store, dvydb_store, dvxdb_store,
         ddvydy_store, ddvxdx_store, ddvxy_store, ddvydb_store,
         ddvxdb_store) = [st.snap for st in storage]
        device = lamb_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = lamb_p.dtype
        n_shots, ny, nx, ny_nx = ctx.n_shots, ctx.ny, ctx.nx, ctx.ny_nx
        nt, grad_stride = ctx.nt, ctx.grad_stride

        if grad_r is None:
            grad_r = torch.zeros(nt, n_shots, ctx.n_rec, device=device, dtype=dtype)
        grad_r = grad_r.contiguous()
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
        if segments:
            # Checkpointed backward: per segment, restore the wavefield state,
            # replay the forward steps to regenerate the snapshots, then run
            # the adjoint steps.  The adjoint state carries across segments.
            # The born kernels index the snapshot streams by ``t/grad_stride``,
            # so the replay/adjoint calls use a segment-local time ``t - base``
            # with ``base`` the first multiple of ``grad_stride`` >= s0; the
            # record/inject/grad kernels keep the global ``t``.
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
            state_bufs = (
                f_vy, f_vx, f_syy, f_sxx, f_sxy,
                f_m_vyy, f_m_vxx, f_m_vxy, f_m_vyx,
                f_m_sigmayyy, f_m_sigmaxyx, f_m_sigmaxyy, f_m_sigmaxxx,
                f_dvy, f_dvx, f_dsyy, f_dsxx, f_dsxy,
                f_dm_vyy, f_dm_vxx, f_dm_vxy, f_dm_vyx,
                f_dm_sigmayyy, f_dm_sigmaxyx, f_dm_sigmaxyy, f_dm_sigmaxxx,
            )
            ckpt = ctx.ckpt_state
            for k in range(len(segments) - 1, -1, -1):
                s0, s1 = segments[k]
                base = (
                    s0 if s0 % grad_stride == 0
                    else ((s0 // grad_stride) + 1) * grad_stride
                )
                if s0 > 0:
                    c = ckpt[k - 1]
                    for i, buf in enumerate(state_bufs):
                        buf.copy_(c[i])
                else:
                    for buf in state_bufs:
                        buf.zero_()
                # replay the forward steps (regenerate the snapshots)
                for t in range(s0, s1):
                    t_step = t - base
                    ext.born_step_velocity(
                        f_vy, f_vx, f_syy, f_sxx, f_sxy,
                        f_dvy, f_dvx, f_dsyy, f_dsxx, f_dsxy,
                        f_m_sigmayyy, f_m_sigmaxyx, f_m_sigmaxyy, f_m_sigmaxxx,
                        f_dm_sigmayyy, f_dm_sigmaxyx, f_dm_sigmaxyy, f_dm_sigmaxxx,
                        buoyancy_y, buoyancy_x, dbuoyancy_y, dbuoyancy_x,
                        dvydb_store, dvxdb_store, ddvydb_store, ddvxdb_store,
                        ayh, byh, ay, by, axh, bxh, ax, bx, ctx.c,
                        ctx.fd_pad[0], ctx.fd_pad[1], ctx.fd_pad[2], ctx.fd_pad[3],
                        ctx.rdy, ctx.rdx, ctx.dtv, t_step, grad_stride,
                        ctx.model_batched, ctx.scatter_batched, 1,
                    )
                    ext.born_step_stress(
                        f_vy, f_vx, f_dvy, f_dvx,
                        f_syy, f_sxx, f_sxy, f_dsyy, f_dsxx, f_dsxy,
                        f_m_vyy, f_m_vxx, f_m_vxy, f_m_vyx,
                        f_dm_vyy, f_dm_vxx, f_dm_vxy, f_dm_vyx,
                        lamb_p, mu_p, mu_yx, dlamb_p, dmu_p, dmu_yx,
                        dvydy_store, dvxdx_store, dvxy_store,
                        ddvydy_store, ddvxdx_store, ddvxy_store,
                        ayh, byh, ay, by, axh, bxh, ax, bx, ctx.c,
                        ctx.fd_pad[0], ctx.fd_pad[1], ctx.fd_pad[2], ctx.fd_pad[3],
                        ctx.rdy, ctx.rdx, ctx.dtv, t_step, grad_stride,
                        ctx.model_batched, ctx.scatter_batched, 1,
                    )
                    if ctx.n_src > 0:
                        ext.born_inject_pressure(
                            f_syy, f_sxx, f, src_i, t, n_shots, ctx.n_src, ny_nx,
                        )
                # adjoint steps for the segment
                for t in range(s1 - 1, s0 - 1, -1):
                    t_step = t - base
                    parity = (nt - 1 - t) % 2
                    old = m_sig_b if parity else m_sig_a
                    new = m_sig_a if parity else m_sig_b
                    dold = dm_sig_b if parity else dm_sig_a
                    dnew = dm_sig_a if parity else dm_sig_b
                    if ctx.n_src > 0:
                        ext.born_record_grad_f_p(
                            l_syy, l_sxx, grad_f, src_i, t,
                            n_shots, ctx.n_src, ny_nx,
                        )
                    ext.born_adjoint_velocity(
                        lamb_p, mu_p, mu_yx, dlamb_p, dmu_p, dmu_yx,
                        l_vy, l_vx, l_dvy, l_dvx,
                        l_syy, l_sxx, l_sxy, l_dsyy, l_dsxx, l_dsxy,
                        m_vyy, m_vxx, m_vxy, m_vyx,
                        dm_vyy, dm_vxx, dm_vxy, dm_vyx,
                        old[0], old[1], old[2], old[3],
                        new[0], new[1], new[2], new[3],
                        dold[0], dold[1], dold[2], dold[3],
                        dnew[0], dnew[1], dnew[2], dnew[3],
                        buoyancy_y, buoyancy_x, dbuoyancy_y, dbuoyancy_x,
                        grad_by, grad_bx, grad_dby, grad_dbx,
                        dvydb_store, dvxdb_store, ddvydb_store, ddvxdb_store,
                        ayh, byh, ay, by, axh, bxh, ax, bx,
                        ctx.c, ctx.fd_pad[0], ctx.fd_pad[1],
                        ctx.fd_pad[2], ctx.fd_pad[3],
                        ctx.rdy, ctx.rdx, ctx.dtv, t_step, grad_stride,
                        ctx.model_batched, ctx.scatter_batched,
                    )
                    ext.born_adjoint_stress(
                        buoyancy_y, buoyancy_x, dbuoyancy_y, dbuoyancy_x,
                        l_vy, l_vx, l_dvy, l_dvx,
                        l_syy, l_sxx, l_sxy, l_dsyy, l_dsxx, l_dsxy,
                        m_vyy, m_vxx, m_vxy, m_vyx,
                        dm_vyy, dm_vxx, dm_vxy, dm_vyx,
                        lamb_p, mu_p, mu_yx, dlamb_p, dmu_p, dmu_yx,
                        old[0], old[1], old[2], old[3],
                        dold[0], dold[1], dold[2], dold[3],
                        grad_lamb, grad_mu, grad_mu_yx,
                        grad_dlamb, grad_dmu, grad_dmu_yx,
                        dvydy_store, dvxdx_store, dvxy_store,
                        ddvydy_store, ddvxdx_store, ddvxy_store,
                        ayh, byh, ay, by, axh, bxh, ax, bx,
                        ctx.c, ctx.fd_pad[0], ctx.fd_pad[1],
                        ctx.fd_pad[2], ctx.fd_pad[3],
                        ctx.rdy, ctx.rdx, ctx.dtv, t_step, grad_stride,
                        ctx.model_batched, ctx.scatter_batched,
                    )
                    if ctx.n_rec > 0:
                        ext.born_add_grad_r(
                            l_dsyy, l_dsxx, grad_r, rec_i, t,
                            n_shots, ctx.n_rec, ny_nx,
                        )
        else:
            for t in range(nt - 1, -1, -1):
                parity = (nt - 1 - t) % 2
                old = m_sig_b if parity else m_sig_a
                new = m_sig_a if parity else m_sig_b
                dold = dm_sig_b if parity else dm_sig_a
                dnew = dm_sig_a if parity else dm_sig_b
                if ctx.n_src > 0:
                    ext.born_record_grad_f_p(
                        l_syy, l_sxx, grad_f, src_i, t, n_shots, ctx.n_src, ny_nx,
                    )
                ext.born_adjoint_velocity(
                    lamb_p, mu_p, mu_yx, dlamb_p, dmu_p, dmu_yx,
                    l_vy, l_vx, l_dvy, l_dvx,
                    l_syy, l_sxx, l_sxy, l_dsyy, l_dsxx, l_dsxy,
                    m_vyy, m_vxx, m_vxy, m_vyx,
                    dm_vyy, dm_vxx, dm_vxy, dm_vyx,
                    old[0], old[1], old[2], old[3],
                    new[0], new[1], new[2], new[3],
                    dold[0], dold[1], dold[2], dold[3],
                    dnew[0], dnew[1], dnew[2], dnew[3],
                    buoyancy_y, buoyancy_x, dbuoyancy_y, dbuoyancy_x,
                    grad_by, grad_bx, grad_dby, grad_dbx,
                    dvydb_store, dvxdb_store, ddvydb_store, ddvxdb_store,
                    ayh, byh, ay, by, axh, bxh, ax, bx,
                    ctx.c, ctx.fd_pad[0], ctx.fd_pad[1], ctx.fd_pad[2], ctx.fd_pad[3],
                    ctx.rdy, ctx.rdx, ctx.dtv, t, grad_stride,
                    ctx.model_batched, ctx.scatter_batched,
                )
                ext.born_adjoint_stress(
                    buoyancy_y, buoyancy_x, dbuoyancy_y, dbuoyancy_x,
                    l_vy, l_vx, l_dvy, l_dvx,
                    l_syy, l_sxx, l_sxy, l_dsyy, l_dsxx, l_dsxy,
                    m_vyy, m_vxx, m_vxy, m_vyx,
                    dm_vyy, dm_vxx, dm_vxy, dm_vyx,
                    lamb_p, mu_p, mu_yx, dlamb_p, dmu_p, dmu_yx,
                    old[0], old[1], old[2], old[3],
                    dold[0], dold[1], dold[2], dold[3],
                    grad_lamb, grad_mu, grad_mu_yx,
                    grad_dlamb, grad_dmu, grad_dmu_yx,
                    dvydy_store, dvxdx_store, dvxy_store,
                    ddvydy_store, ddvxdx_store, ddvxy_store,
                    ayh, byh, ay, by, axh, bxh, ax, bx,
                    ctx.c, ctx.fd_pad[0], ctx.fd_pad[1], ctx.fd_pad[2], ctx.fd_pad[3],
                    ctx.rdy, ctx.rdx, ctx.dtv, t, grad_stride,
                    ctx.model_batched, ctx.scatter_batched,
                )
                if ctx.n_rec > 0:
                    ext.born_add_grad_r(
                        l_dsyy, l_dsxx, grad_r, rec_i, t, n_shots, ctx.n_rec, ny_nx,
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
        return (
            grad_lamb, grad_mu, grad_mu_yx, grad_by, grad_bx,
            grad_dlamb, grad_dmu, grad_dmu_yx, grad_dby, grad_dbx,
            grad_amp,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
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
    accuracy=2,
    pml_width=20,
    pml_freq=25.0,
    nt=None,
    storage="auto",
    sample_steps=1,
    ckpt_steps=None,
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

    Returns scattered pressure traces ``[nt, n_shots, n_rec]``.
    """
    accuracy = check_accuracy(accuracy)
    if not isinstance(grid_spacing, (list, tuple)):
        grid_spacing = [float(grid_spacing)] * 2
    grid_spacing = [float(g) for g in grid_spacing]
    pml_w = _set_pml_width(pml_width, 2)
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
    if max_vel == 0:  # empty model: no CFL restriction
        max_dt = float("inf")
    else:
        max_dt = (
            0.6
            / math.sqrt(sum(1 / g**2 for g in grid_spacing))
            / (max_vel**2 + 1e-15)
        ) * max_vel
    if math.ceil(abs(float(dt)) / max_dt) > 1:
        raise NotImplementedError(
            "nami elastic2d_born requires dt <= "
            f"{max_dt:.3e} to satisfy the CFL condition (step_ratio=1); "
            f"got dt={dt}."
        )

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

    if source_amplitudes is not None:
        amp = source_amplitudes.to(device=device, dtype=dtype)
        if amp.shape[0] != n_shots:
            raise ValueError("source_amplitudes must have n_shots batches.")
        if amp.shape[2] < nt_inner:
            raise ValueError("source_amplitudes must have at least nt steps.")
        amp = amp[:, :, :nt_inner].contiguous()
    else:
        amp = torch.zeros(n_shots, 0, nt_inner, device=device, dtype=dtype)

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
    model_batched = 1 if (lamb_p.ndim == 3 and lamb_p.shape[0] > 1) else 0
    scatter_batched = 1 if (dlamb_p.ndim == 3 and dlamb_p.shape[0] > 1) else 0

    stores = None
    ckpt_state = None
    if storage_enabled:
        stores = [
            SnapshotStorage(
                _storage_ext, n_snap, n_shots, ny, nx, dtype, device,
            )
            for _ in range(10)
        ]
        if n_ckpt > 0:
            ckpt_state = torch.zeros(
                n_ckpt, N_STATE, n_shots, ny, nx, device=device, dtype=dtype,
            )

    r = BornElasticFunc.apply(
        lamb_p, mu_p, mu_yx, buoyancy_y, buoyancy_x,
        dlamb_p, dmu_p, dmu_yx, dbuoyancy_y, dbuoyancy_x,
        amp, src_i, rec_i, profs, c,
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
    )
    return -r / 2
