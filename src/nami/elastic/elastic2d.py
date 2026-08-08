"""nami elastic2d: 2D elastic FDTD (velocity-stress staggered grid) with
native CUDA forward/adjoint.

Forward/adjoint discretisation follows the reference elastic compiled
backend (bit-for-bit reproducible in float64): Komatitsch-Martin C-PML
memory variables updated uniformly (zero a/b in the interior exactly
reproduces the PML-region split), the pressure-source convention
``f = -amp * dt`` injected into both normal stresses, and pressure
receivers recording ``sigmayy + sigmaxx`` (the returned trace is
``-(sigmayy + sigmaxx) / 2``).

Per forward step (staggered grid, any accuracy order):

    vy += buoyancy_y*dt*(DIFFYH1(sigmayy) + DIFFX1(sigmaxy))   (PML memories)
    vx += buoyancy_x*dt*(DIFFXH1(sigmaxx) + DIFFY1(sigmaxy))
    syy += dt*(lamb*(dvydy + dvxdx) + 2*mu*dvydy)              (dvydy = DIFFY1(vy))
    sxx += dt*(lamb*(dvydy + dvxdx) + 2*mu*dvxdx)              (dvxdx = DIFFX1(vx))
    sxy += dt*mu_yx*(DIFFXH1(vy) + DIFFYH1(vx))

with ``m = a*m + b*derivative`` memory recursions.  Five snapshots are
stored every ``grad_stride`` steps for the imaging condition: dt*dvydy,
dt*dvxdx, dt*(DIFFXH1(vy)+DIFFYH1(vx)), dt*(vy update sum),
dt*(vx update sum).

The adjoint is the exact discrete transpose of the forward step.  The
``m_sigma*`` adjoint memories need two alternating buffers
(the C code's ``_t``/``_n`` trick); the ``m_v*`` adjoint memories update in
place.

Kernels are intentionally unoptimised (one launch per step, naive stencil)
— that is the correctness baseline; performance work lands on top later.
"""

import math

import nami_elastic2d as _ext
import torch

from ..common.cfl import check_cfl
from ..common.fd import check_accuracy, staggered_diff1_coeffs
from ..common.pml import set_pml_width
from ..common.storage import (
    SnapshotStorage,
    check_sample_steps,
    resolve_storage,
    storage_plan,
)
from ..common.survey import check_model_batching, extract_survey_2d

# Checkpoint state layout for the elastic velocity-stress field (the buffers
# are updated in place, so they ARE the wavefield state at time t):
#   [0]  vy           velocity (y)
#   [1]  vx           velocity (x)
#   [2]  syy          normal stress (yy)
#   [3]  sxx          normal stress (xx)
#   [4]  sxy          shear stress (xy)
#   [5]  m_vyy        stress-memory (vy, y)
#   [6]  m_vxx        stress-memory (vx, x)
#   [7]  m_vxy        stress-memory (vx, y)
#   [8]  m_vyx        stress-memory (vy, x)
#   [9]  m_sigmayyy   velocity-memory (syy, y)
#   [10] m_sigmaxyx   velocity-memory (sxy, x)
#   [11] m_sigmaxyy   velocity-memory (sxy, y)
#   [12] m_sigmaxxx   velocity-memory (sxx, x)
N_STATE = 13
N_STREAMS = 5


class Elastic2DFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        lamb,             # [1, ny, nx] or [n_shots, ny, nx] padded model
        mu,               # [1, ny, nx] or [n_shots, ny, nx] padded model
        mu_yx,            # [1, ny, nx] staggered harmonic mean of mu
        buoyancy_y,       # [1, ny, nx] buoyancy at half y points
        buoyancy_x,       # [1, ny, nx] buoyancy at half x points
        amp,              # [n_shots, n_src, nt] pressure source amplitudes
        src_i,            # [n_shots, n_src] flat indices
        rec_i,            # [n_shots, n_rec] flat indices
        profs,            # [ayh, byh, ay, by, axh, bxh, ax, bx]
        c,                # staggered first-derivative coefficients (len >= radius)
        fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
        rdy, rdx, dtv,
        nt, grad_stride, n_shots, model_batched,
        storage,         # list of 5 SnapshotStorage (one per stream) or None
        ckpt_state,      # [n_ckpt, N_STATE, n_shots, ny, nx] or None
        checkpoint_every,  # 0 = full storage; N = checkpoint every N steps
        segments,        # [(s0, s1)] replay segments; [] = full storage
    ):
        ext = _ext
        device = lamb.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = lamb.dtype
        ny, nx = lamb.shape[-2:]
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
        syy, sxx, sxy = z(n_shots, ny, nx), z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_vyy, m_vxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_vxy, m_vyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_sigmayyy, m_sigmaxyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_sigmaxyy, m_sigmaxxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        r = z(nt, n_shots, n_rec)
        if storage is not None:
            (dvydy_store, dvxdx_store, dvxy_store,
             dvydb_store, dvxdb_store) = [st.snap for st in storage]
        else:
            dummy = z(n_shots, ny, nx)
            dvydy_store = dvxdx_store = dvxy_store = dvydb_store = dvxdb_store = dummy

        ayh, byh, ay, by, axh, bxh, ax, bx = profs
        pml_y0, pml_y1, pml_x0, pml_x1 = ext.get_pml_box(by, bx, ny, nx)
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
            if segments:
                store = 0
                snap_off = 0
            else:
                store = 1 if storage is not None else 0
                snap_off = storage[0].snap_offset(t // grad_stride) if storage else 0
            ext.forward_step(
                vy, vx, syy, sxx, sxy,
                m_sigmayyy, m_sigmaxyx, m_sigmaxyy, m_sigmaxxx,
                buoyancy_y, buoyancy_x, dvydb_store, dvxdb_store,
                m_vyy, m_vxx, m_vxy, m_vyx,
                lamb, mu, mu_yx, dvydy_store, dvxdx_store, dvxy_store,
                ayh, byh, ay, by, axh, bxh, ax, bx,
                c, f, src_i, rec_i, r,
                fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
                rdy, rdx, dtv, t, grad_stride, snap_off,
                n_shots, ny, nx, ny_nx, n_src, n_rec, model_batched, store, 1,
                pml_y0, pml_y1, pml_x0, pml_x1,
            )

        ctx.ext = ext
        ctx.save_for_backward(
            lamb, mu, mu_yx, buoyancy_y, buoyancy_x, src_i, rec_i, f,
        )
        ctx.storage = storage
        ctx.profs = profs
        ctx.pml = (pml_y0, pml_y1, pml_x0, pml_x1)
        ctx.c = c
        ctx.fd_pad = (fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1)
        ctx.rdy, ctx.rdx, ctx.dtv = rdy, rdx, dtv
        ctx.nt, ctx.grad_stride = nt, grad_stride
        ctx.n_shots, ctx.ny, ctx.nx, ctx.ny_nx = n_shots, ny, nx, ny_nx
        ctx.n_src, ctx.n_rec = n_src, n_rec
        ctx.model_batched = model_batched
        ctx.ckpt_state = ckpt_state
        ctx.checkpoint_every = checkpoint_every
        ctx.segments = segments
        return r

    @staticmethod
    def backward(ctx, grad_r):
        if ctx.storage is None:
            raise RuntimeError(
                "elastic2d backward() requires snapshot storage: run the forward "
                "with an input requiring grad (and not under torch.no_grad())."
            )
        ext = ctx.ext
        storage = ctx.storage
        (lamb, mu, mu_yx, buoyancy_y, buoyancy_x, src_i, rec_i, f) = ctx.saved_tensors
        (dvydy_store, dvxdx_store, dvxy_store,
         dvydb_store, dvxdb_store) = [st.snap for st in storage]
        device = lamb.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = lamb.dtype
        n_shots, ny, nx, ny_nx = ctx.n_shots, ctx.ny, ctx.nx, ctx.ny_nx
        nt, grad_stride = ctx.nt, ctx.grad_stride
        scale = float(grad_stride)

        if grad_r is None:
            grad_r = torch.zeros(nt, n_shots, ctx.n_rec, device=device, dtype=dtype)
        grad_f = torch.zeros(nt, n_shots, ctx.n_src, device=device, dtype=dtype)
        grad_lamb = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        grad_mu = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        grad_mu_yx = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        grad_by = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        grad_bx = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)

        z = lambda *shape: torch.zeros(shape, device=device, dtype=dtype)  # noqa: E731
        l_vy, l_vx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        l_syy, l_sxx, l_sxy = z(n_shots, ny, nx), z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_vyy, m_vxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_vxy, m_vyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
        # the m_sigma* adjoint memories alternate (the C code's _t/_n trick)
        m_sig_a = [z(n_shots, ny, nx) for _ in range(4)]
        m_sig_b = [z(n_shots, ny, nx) for _ in range(4)]

        ayh, byh, ay, by, axh, bxh, ax, bx = ctx.profs
        segments = ctx.segments
        if segments:
            # Checkpointed backward: per segment, restore the wavefield state,
            # replay the forward steps to regenerate the snapshots, then run
            # the adjoint steps.  The adjoint state carries across segments.
            f_vy, f_vx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            f_syy, f_sxx, f_sxy = (
                z(n_shots, ny, nx), z(n_shots, ny, nx), z(n_shots, ny, nx),
            )
            f_m_vyy, f_m_vxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            f_m_vxy, f_m_vyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            f_m_sigmayyy, f_m_sigmaxyx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            f_m_sigmaxyy, f_m_sigmaxxx = z(n_shots, ny, nx), z(n_shots, ny, nx)
            state_bufs = (
                f_vy, f_vx, f_syy, f_sxx, f_sxy,
                f_m_vyy, f_m_vxx, f_m_vxy, f_m_vyx,
                f_m_sigmayyy, f_m_sigmaxyx, f_m_sigmaxyy, f_m_sigmaxxx,
            )
            ckpt = ctx.ckpt_state
            for k in range(len(segments) - 1, -1, -1):
                s0, s1 = segments[k]
                if s0 > 0:
                    c = ckpt[k - 1]
                    for i, buf in enumerate(state_bufs):
                        buf.copy_(c[i])
                else:
                    for buf in state_bufs:
                        buf.zero_()
                # replay the forward steps (regenerate the snapshots)
                for t in range(s0, s1):
                    snap_off = ((t - s0) // grad_stride) * (n_shots * ny_nx)
                    ext.forward_step(
                        f_vy, f_vx, f_syy, f_sxx, f_sxy,
                        f_m_sigmayyy, f_m_sigmaxyx, f_m_sigmaxyy, f_m_sigmaxxx,
                        buoyancy_y, buoyancy_x, dvydb_store, dvxdb_store,
                        f_m_vyy, f_m_vxx, f_m_vxy, f_m_vyx,
                        lamb, mu, mu_yx, dvydy_store, dvxdx_store, dvxy_store,
                        ayh, byh, ay, by, axh, bxh, ax, bx,
                        ctx.c, f, src_i, rec_i, grad_r,
                        ctx.fd_pad[0], ctx.fd_pad[1],
                        ctx.fd_pad[2], ctx.fd_pad[3],
                        ctx.rdy, ctx.rdx, ctx.dtv, t, grad_stride, snap_off,
                        n_shots, ny, nx, ny_nx, ctx.n_src, ctx.n_rec,
                        ctx.model_batched, 1, 0,
                        ctx.pml[0], ctx.pml[1], ctx.pml[2], ctx.pml[3],
                    )
                # adjoint steps for the segment
                for t in range(s1 - 1, s0 - 1, -1):
                    parity = (nt - 1 - t) % 2
                    old = m_sig_b if parity else m_sig_a
                    new = m_sig_a if parity else m_sig_b
                    snap_off = ((t - s0) // grad_stride) * (n_shots * ny_nx)
                    ext.backward_step(
                        lamb, mu, mu_yx, buoyancy_y, buoyancy_x,
                        l_vy, l_vx, l_syy, l_sxx, l_sxy,
                        m_vyy, m_vxx, m_vxy, m_vyx,
                        old[0], old[1], old[2], old[3],
                        new[0], new[1], new[2], new[3],
                        grad_by, grad_bx, grad_lamb, grad_mu, grad_mu_yx,
                        dvydb_store, dvxdb_store,
                        dvydy_store, dvxdx_store, dvxy_store,
                        ayh, byh, ay, by, axh, bxh, ax, bx,
                        ctx.c, grad_f, src_i, grad_r, rec_i,
                        ctx.fd_pad[0], ctx.fd_pad[1],
                        ctx.fd_pad[2], ctx.fd_pad[3],
                        ctx.rdy, ctx.rdx, ctx.dtv, scale, t, grad_stride, snap_off,
                        n_shots, ny, nx, ny_nx, ctx.n_src, ctx.n_rec,
                        ctx.model_batched,
                        ctx.pml[0], ctx.pml[1], ctx.pml[2], ctx.pml[3],
                    )
        else:
            for t in range(nt - 1, -1, -1):
                parity = (nt - 1 - t) % 2
                old = m_sig_b if parity else m_sig_a
                new = m_sig_a if parity else m_sig_b
                snap_off = storage[0].snap_offset(t // grad_stride)
                ext.backward_step(
                    lamb, mu, mu_yx, buoyancy_y, buoyancy_x,
                    l_vy, l_vx, l_syy, l_sxx, l_sxy,
                    m_vyy, m_vxx, m_vxy, m_vyx,
                    old[0], old[1], old[2], old[3],
                    new[0], new[1], new[2], new[3],
                    grad_by, grad_bx, grad_lamb, grad_mu, grad_mu_yx,
                    dvydb_store, dvxdb_store,
                    dvydy_store, dvxdx_store, dvxy_store,
                    ayh, byh, ay, by, axh, bxh, ax, bx,
                    ctx.c, grad_f, src_i, grad_r, rec_i,
                    ctx.fd_pad[0], ctx.fd_pad[1], ctx.fd_pad[2], ctx.fd_pad[3],
                    ctx.rdy, ctx.rdx, ctx.dtv, scale, t, grad_stride, snap_off,
                    n_shots, ny, nx, ny_nx, ctx.n_src, ctx.n_rec,
                    ctx.model_batched,
                    ctx.pml[0], ctx.pml[1], ctx.pml[2], ctx.pml[3],
                )

        if not ctx.model_batched:
            grad_lamb = grad_lamb.sum(0, keepdim=True)
            grad_mu = grad_mu.sum(0, keepdim=True)
            grad_mu_yx = grad_mu_yx.sum(0, keepdim=True)
            grad_by = grad_by.sum(0, keepdim=True)
            grad_bx = grad_bx.sum(0, keepdim=True)
        # grad through the f = -amp * dt pre-scaling
        grad_amp = (grad_f * (-ctx.dtv)).permute(1, 2, 0)
        return (
            grad_lamb,
            grad_mu,
            grad_mu_yx,
            grad_by,
            grad_bx,
            grad_amp,
            None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
        )


def lambmubuoyancy_to_vpvsrho(lamb, mu, buoyancy, eps=1e-15):
    """Converts lambda, mu, buoyancy to vp, vs, rho."""
    vs = (mu * buoyancy).sqrt()
    vp = (lamb * buoyancy + 2 * vs**2).sqrt()
    rho = 1 / (buoyancy**2 + eps) * buoyancy
    return vp, vs, rho


def prepare_parameters(mu, buoyancy):
    """Prepares staggered elastic parameters (2D).

    Returns ``(mu_yx, buoyancy_y, buoyancy_x)`` where mu_yx is the harmonic
    mean of each 2x2 block of mu (zero on the first row/column), and
    buoyancy_y/buoyancy_x are the buoyancy at the half grid points obtained
    from the arithmetic mean of adjacent densities (zero on the last
    row/column).
    """
    ndim = mu.ndim - 1
    rfmax = 1 / torch.finfo(mu.dtype).max ** (1 / 2)
    parameters = []

    # Mu (harmonic mean)
    mu_safe = torch.where(mu.abs() > rfmax, mu, torch.ones_like(mu))
    if ndim >= 2:
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
        mu_yx = torch.where(mask, mu_yx_val, torch.zeros_like(mu_yx_val))
        mu_yx = torch.nn.functional.pad(mu_yx, (0, 1, 0, 1))
        parameters.append(mu_yx)

    # Buoyancy (inverse of arithmetic mean of density)
    mask = rfmax < buoyancy.abs()
    buoyancy_safe = torch.where(mask, buoyancy, torch.ones_like(buoyancy))
    rho = torch.where(mask, 1 / buoyancy_safe, torch.zeros_like(buoyancy))
    if ndim >= 2:
        rho_y = torch.nn.functional.pad(
            (rho[..., :-1, :] + rho[..., 1:, :]) / 2, (0, 0, 0, 1)
        )
        mask = rfmax < rho_y.abs()
        rho_y_safe = torch.where(mask, rho_y, torch.ones_like(rho_y))
        buoyancy_y = torch.where(mask, 1 / rho_y_safe, torch.zeros_like(rho_y))
        parameters.append(buoyancy_y)
    rho_x = torch.nn.functional.pad((rho[..., :-1] + rho[..., 1:]) / 2, (0, 1))
    mask = rfmax < rho_x.abs()
    rho_x_safe = torch.where(mask, rho_x, torch.ones_like(rho_x))
    buoyancy_x = torch.where(mask, 1 / rho_x_safe, torch.zeros_like(rho_x))
    parameters.append(buoyancy_x)

    return parameters


def _set_elastic_pml_profiles(
    pml_width, fd_pad, dt, grid_spacing, max_vel, pml_freq, shape, dtype, device,
):
    """C-PML profiles for the 2D elastic staggered grid.

    Sets the C-PML profiles per
    dimension a pair of (a, b) profiles at the integer grid points and at the
    half-integer grid points, with the polynomial grading ``pml_frac**2`` and
    the reflection coefficient ``r_val=0.001``.

    Returns ``[ay, by, ayh, byh, ax, bx, axh, bxh]``, flat 1-D profiles of
    length ``ny`` / ``nx`` (the kernels index them directly by coordinate).
    """
    ndim = len(shape)
    pml_start = [
        [
            float(fd_pad[dim * 2] + pml_width[dim * 2]),
            float(shape[dim] - 1 - fd_pad[dim * 2 + 1] - pml_width[dim * 2 + 1]),
        ]
        for dim in range(ndim)
    ]
    physical_widths = [pml_width[i] * grid_spacing[i // 2] for i in range(2 * ndim)]
    max_pml = max(physical_widths) if physical_widths else 0.0
    if max_pml == 0:  # no PML: all-zero profiles (avoid -inf/NaN sigma0)
        zero_y = torch.zeros(shape[0], device=device, dtype=dtype)
        zero_x = torch.zeros(shape[1], device=device, dtype=dtype)
        return [zero_y, zero_y, zero_y, zero_y, zero_x, zero_x, zero_x, zero_x]

    alpha0 = math.pi * pml_freq
    sigma0 = -(1 + 2) * max_vel * math.log(0.001) / (2 * max_pml)

    profiles: list[torch.Tensor] = []
    for dim in range(ndim):
        n = shape[dim]
        for half_start in (0.0, 0.5):
            x = torch.arange(half_start, half_start + n, device=device, dtype=dtype)
            if pml_width[2 * dim] == 0:
                pml_frac0 = torch.zeros_like(x)
            else:
                pml_frac0 = (pml_start[dim][0] - x) / pml_width[2 * dim]
            if pml_width[2 * dim + 1] == 0:
                pml_frac1 = torch.zeros_like(x)
            else:
                pml_frac1 = (x - pml_start[dim][1]) / pml_width[2 * dim + 1]
            pml_frac = torch.clamp(torch.maximum(pml_frac0, pml_frac1), min=0, max=1)
            sigma = sigma0 * pml_frac**2
            alpha = alpha0 * (1 - pml_frac)
            sigmaalpha = sigma + alpha
            a = torch.exp(-sigmaalpha * abs(dt))
            b = sigma / sigmaalpha * (a - 1)
            a[pml_frac == 0] = 0
            profiles.extend([a, b])
    return profiles


def elastic2d(
    lamb,
    mu,
    buoyancy,
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
    """2D elastic wave modelling / FWI primitive (CUDA backend).

    Input ``lamb`` and ``mu`` (Lamé parameters) and
    ``buoyancy`` (1/density), each [ny, nx] (or [1, ny, nx] /
    [n_shots, ny, nx]).  Pressure sources and pressure receivers follow the
    elastic convention.

    Args:
        lamb: First Lamé parameter model [ny, nx], Pa.
        mu: Second Lamé parameter model [ny, nx], Pa.
        buoyancy: Buoyancy (1/rho) model [ny, nx], m^3/kg.
        grid_spacing: Cell size (scalar or [dy, dx]).
        dt: Time step interval (s).  Must satisfy the CFL condition
            (a single internal step is used; resampling is not supported).
        source_amplitudes: Pressure source amplitudes [n_shots, n_src, nt].
        source_locations: [n_shots, n_src, 2] (y, x).
        receiver_locations: [n_shots, n_rec, 2] (y, x).
        accuracy: FD accuracy order (2, 4, 6 or 8).
        pml_width: PML width (int or [top, bottom, left, right]).
        pml_freq: PML design frequency (Hz).
        nt: Number of steps (defaults to source_amplitudes length).
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

    Returns:
        receiver_amplitudes [nt, n_shots, n_rec], pressure = -(sigmayy +
        sigmaxx) / 2, matching the elastic pressure receivers.
    """
    accuracy = check_accuracy(accuracy)
    if not isinstance(grid_spacing, (list, tuple)):
        grid_spacing = [float(grid_spacing)] * 2
    grid_spacing = [float(g) for g in grid_spacing]
    pml_width_list = set_pml_width(pml_width, 2)
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

    (lamb_p, mu_p, buoy_p), sources_i, receivers_i = extract_survey_2d(
        [lamb, mu, buoyancy],
        source_locations,
        receiver_locations,
        fd_pad,
        pml_width_list,
        n_shots,
        device,
        dtype,
    )
    padded_ny, padded_nx = lamb_p.shape[-2:]

    # max_vel is derived from the (unpadded) models and used both for the
    # CFL condition and the PML grading.
    vp, vs, _ = lambmubuoyancy_to_vpvsrho(lamb, mu, buoyancy)
    max_vel = max(vp.abs().max().item(), vs.abs().max().item())
    check_cfl(grid_spacing, dt, max_vel, "elastic2d")

    profiles = _set_elastic_pml_profiles(
        pml_width_list,
        fd_pad,
        float(dt),
        grid_spacing,
        max_vel,
        pml_freq,
        (padded_ny, padded_nx),
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
        for t in (lamb, mu, buoyancy, source_amplitudes)
    )
    checkpoint_every, segments, n_snap, n_ckpt = storage_plan(
        nt_inner, N_STATE, grad_stride, N_STREAMS, storage_enabled,
        ckpt_steps=ckpt_steps,
    )
    # Batched flag from the *user* models (before pad).  One flag governs
    # all three parameters, so they must share a batch form.
    check_model_batching(
        [lamb, mu, buoyancy], ("lamb", "mu", "buoyancy"), n_shots
    )
    model_batched = 1 if (
        lamb.ndim == 3 and lamb.shape[0] == n_shots and n_shots > 1
    ) else 0

    stores = None
    ckpt_state = None
    if storage_enabled:
        stores = [
            SnapshotStorage(
                _ext, n_snap, n_shots, padded_ny, padded_nx, dtype, device,
            )
            for _ in range(5)
        ]
        if n_ckpt > 0:
            ckpt_state = torch.zeros(
                n_ckpt, N_STATE, n_shots, padded_ny, padded_nx,
                device=device, dtype=dtype,
            )

    r = Elastic2DFunc.apply(
        lamb_p,
        mu_p,
        mu_yx,
        buoyancy_y,
        buoyancy_x,
        amp,
        sources_i,
        receivers_i,
        profs,
        c,
        fd_pad[0], fd_pad[1], fd_pad[2], fd_pad[3],
        1.0 / grid_spacing[0],
        1.0 / grid_spacing[1],
        float(dt),
        nt_inner,
        grad_stride,
        n_shots,
        model_batched,
        stores,
        ckpt_state,
        checkpoint_every,
        segments,
    )
    return -r / 2
