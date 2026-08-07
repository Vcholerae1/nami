"""nami em2d_tm: 2D TM Maxwell FDTD (Ey-Hx-Hz staggered grid) with
native CUDA adjoint.

Forward/adjoint discretisation follows the reference 2D TM FDTD
scheme (the pure-PyTorch eager reference in the project tests):

    per step: H half-step, E integer-step (snapshotting the pre-update Ey
    and the PML-modified curl for the adjoint), source injection (pre-scaled
    by ``cb * -1/(dx dy)``), receiver recording (post-injection field).

The backward pass is the exact discrete transpose: ``record_grad_r`` /
``record_grad_f`` at the start of each adjoint step, ``coeff_grad`` and
``cq_grad`` model-gradient accumulators on the snapshot stride, and the
two-stage E/H transpose kernels with the time-reversed CPML memory
recursions.

Kernels are intentionally unoptimised (one launch per step, naive stencil)
— that is the correctness baseline; performance work lands on top of it
later.

The spatial FD order is user-selectable (accuracy 2/4/6/8) with the
``fd.STAGGERED_DIFF1`` coefficient tables (staggered-grid
convention); the kernels are driven by coefficient arrays (fixed max radius
4, zero-padded for lower orders) and the per-side FD padding
``fd_pad = [accuracy // 2, accuracy // 2 - 1] * 2``, mirroring the scalar
backend.  Snapshots (pre-update Ey and the PML-modified curl, two streams)
live in C++-owned storage with device / pinned-cpu / disk
offload and optional bf16 compression.
"""

import math

import nami_em2d_tm as _ext
import torch

from ..common.cfl import check_cfl
from ..common.fd import check_accuracy, staggered_diff1_coeffs
from ..common.storage import (
    SnapshotStorage,
    check_sample_steps,
    resolve_storage,
    storage_plan,
)
from ..common.survey import extract_survey_2d

# vacuum permittivity / permeability / speed of light (SI)
EPS0 = 8.8541878128e-12  # F/m
MU0 = 1.2566370614359173e-06  # H/m

# Checkpoint state layout (saved at time t before step t):
#   [0] ey      pre-update E field
#   [1] hx      H field (y)
#   [2] hz      H field (x)
#   [3] m_ey_z  split-field memory (z) from the H half-step
#   [4] m_ey_x  split-field memory (x) from the H half-step
#   [5] m_hx_z  split-field memory (z) from the E integer-step
#   [6] m_hz_x  split-field memory (x) from the E integer-step
N_STATE = 7
N_STREAMS = 2


class TM2DFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ca_p,            # [n_shots, ny, nx] or [1, ny, nx] padded E-update coeffs
        cb_p,            # (batched when the first dim equals n_shots)
        cq_p,
        f,               # [nt, n_shots, n_src] pre-scaled source amplitudes
        src_i,           # [n_shots, n_src] flat indices
        rec_i,           # [n_shots, n_rec] flat indices
        profs,           # [ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh]
        c,               # [4] staggered FD coefficients (zero-padded, max radius 4)
        fd_pad,          # [y0, y1, x0, x1] FD padding
        rdy, rdx,
        nt, pml_y0, pml_y1, pml_x0, pml_x1,
        ca_batched, cb_batched, cq_batched,
        grad_stride,
        ey_storage,      # SnapshotStorage (ey stream) or None (forward-only)
        curl_storage,    # SnapshotStorage (curl stream) or None
        ckpt_state,      # [n_ckpt, N_STATE, n_shots, ny, nx] or None
        checkpoint_every,  # 0 = full storage; N = checkpoint every N steps
        segments,        # [(s0, s1)] replay segments; [] = full storage
    ):
        ext = _ext
        device = ca_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = ca_p.dtype
        n_shots, ny, nx = ca_p.shape
        n_src = src_i.shape[1]
        n_rec = rec_i.shape[1]
        ny_nx = ny * nx

        ey = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        hx = torch.zeros_like(ey)
        hz = torch.zeros_like(ey)
        m_ey_z = torch.zeros_like(ey)
        m_ey_x = torch.zeros_like(ey)
        m_hx_z = torch.zeros_like(ey)
        m_hz_x = torch.zeros_like(ey)
        r = torch.zeros(nt, n_shots, n_rec, device=device, dtype=dtype)
        if ey_storage is not None:
            ey_store = ey_storage.snap
            curl_store = curl_storage.snap
        else:
            ey_store = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
            curl_store = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)

        ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh = [
            p.contiguous() for p in profs
        ]
        ckpt = ckpt_state is not None
        for t in range(nt):
            # snapshot the full wavefield state before step t at every
            # checkpoint boundary (only in checkpointed mode)
            if ckpt and t > 0 and t % checkpoint_every == 0:
                k = t // checkpoint_every - 1
                ckpt_state[k, 0].copy_(ey)
                ckpt_state[k, 1].copy_(hx)
                ckpt_state[k, 2].copy_(hz)
                ckpt_state[k, 3].copy_(m_ey_z)
                ckpt_state[k, 4].copy_(m_ey_x)
                ckpt_state[k, 5].copy_(m_hx_z)
                ckpt_state[k, 6].copy_(m_hz_x)
            if segments:
                # checkpointed forward: intermediate snapshots are
                # regenerated during the backward replay, so skip the
                # global-offset writes (they would be ignored anyway)
                snap_off = 0
            else:
                snap_off = (
                    ey_storage.snap_offset(t // grad_stride) if ey_storage else 0
                )
            ext.forward_step(
                cq_p, ey, hx, hz, m_ey_z, m_ey_x,
                m_hx_z, m_hz_x,
                ca_p, cb_p,
                ey_store, curl_store,
                ayh, byh, axh, bxh, kyh, kxh,
                ay, by, ax, bx, ky, kx,
                c,
                f, src_i, r, rec_i,
                rdy, rdx, t, grad_stride, snap_off,
                pml_y0, pml_y1, pml_x0, pml_x1,
                *fd_pad, cq_batched, ca_batched, cb_batched,
                n_shots, ny, nx, ny_nx, n_src, n_rec, 1,
            )

        ctx.ext = ext
        ctx.need_cq = bool(ctx.needs_input_grad[2])
        ctx.need_f = bool(ctx.needs_input_grad[3])
        ctx.save_for_backward(ca_p, cb_p, cq_p, f, src_i, rec_i)
        ctx.ey_storage = ey_storage
        ctx.curl_storage = curl_storage
        ctx.rdy, ctx.rdx = rdy, rdx
        ctx.nt = nt
        ctx.pml_y0, ctx.pml_y1, ctx.pml_x0, ctx.pml_x1 = (
            pml_y0, pml_y1, pml_x0, pml_x1,
        )
        ctx.c = c
        ctx.fd_pad = fd_pad
        ctx.ca_batched, ctx.cb_batched, ctx.cq_batched = (
            ca_batched, cb_batched, cq_batched,
        )
        ctx.grad_stride = grad_stride
        ctx.profs = profs
        ctx.ckpt_state = ckpt_state
        ctx.checkpoint_every = checkpoint_every
        ctx.segments = segments
        return r

    @staticmethod
    def _replay_segment(ext, ctx, ca_p, cb_p, cq_p, f, src_i, n_shots, n_src,
                        ey_f, hx_f, hz_f, m_ey_z_f, m_ey_x_f, m_hx_z_f,
                        m_hz_x_f, ey_store, curl_store,
                        ayh, byh, axh, bxh, kyh, kxh,
                        ay, by, ax, bx, ky, kx,
                        grad_stride, shot_count, s0, s1):
        """Re-run forward steps [s0, s1), writing segment-local snapshots."""
        ny_nx = shot_count // n_shots
        ny, nx = ey_f.shape[-2:]
        for t in range(s0, s1):
            snap_off = ((t - s0) // grad_stride) * shot_count
            ext.forward_step(
                cq_p, ey_f, hx_f, hz_f, m_ey_z_f, m_ey_x_f,
                m_hx_z_f, m_hz_x_f,
                ca_p, cb_p,
                ey_store, curl_store,
                ayh, byh, axh, bxh, kyh, kxh,
                ay, by, ax, bx, ky, kx,
                ctx.c,
                f, src_i, src_i, src_i,
                ctx.rdy, ctx.rdx, t, grad_stride, snap_off,
                ctx.pml_y0, ctx.pml_y1, ctx.pml_x0, ctx.pml_x1,
                *ctx.fd_pad, ctx.cq_batched, ctx.ca_batched, ctx.cb_batched,
                n_shots, ny, nx, shot_count, n_src, 0, 0,
            )

    @staticmethod
    def backward(ctx, grad_r):
        ext = ctx.ext
        ca_p, cb_p, cq_p, f, src_i, rec_i = ctx.saved_tensors
        if ctx.ey_storage is None:
            raise RuntimeError(
                "em2d_tm backward() requires snapshot storage: run the forward "
                "with an input requiring grad (and not under torch.no_grad())."
            )
        ey_storage, curl_storage = ctx.ey_storage, ctx.curl_storage
        ey_store, curl_store = ey_storage.snap, curl_storage.snap
        device = ca_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = ca_p.dtype
        n_shots, ny, nx = ca_p.shape
        n_src, n_rec = src_i.shape[1], rec_i.shape[1]
        nt = ctx.nt
        ny_nx = ny * nx

        if grad_r is None:
            grad_r = torch.zeros(nt, n_shots, n_rec, device=device, dtype=dtype)
        # grad_r arrives from autograd and may be a non-contiguous broadcast
        # view; the kernels use flat row-major indexing, so materialise it.
        grad_r = grad_r.contiguous()
        need_cq, need_f = ctx.need_cq, ctx.need_f
        grad_f = (
            torch.zeros(nt, n_shots, n_src, device=device, dtype=dtype)
            if need_f else None
        )
        grad_ca = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        grad_cb = torch.zeros_like(grad_ca)
        grad_cq = torch.zeros_like(grad_ca) if need_cq else None

        lam_ey = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        lam_hx = torch.zeros_like(lam_ey)
        lam_hz = torch.zeros_like(lam_ey)
        m_lambda_ey_x = torch.zeros_like(lam_ey)
        m_lambda_ey_z = torch.zeros_like(lam_ey)
        m_lambda_hx_z = torch.zeros_like(lam_ey)
        m_lambda_hz_x = torch.zeros_like(lam_ey)
        work_x = torch.zeros_like(lam_ey)
        work_y = torch.zeros_like(lam_ey)
        work2_x = torch.zeros_like(lam_ey)
        work2_y = torch.zeros_like(lam_ey)

        ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh = [
            p.contiguous() for p in ctx.profs
        ]
        # integral sampling: each snapshot represents
        # `grad_stride` time steps of the model-gradient integral.
        scale = float(ctx.grad_stride)
        grad_stride = ctx.grad_stride
        segments = ctx.segments
        if segments:
            # Checkpointed backward: per segment, restore the wavefield
            # state, replay the forward steps to regenerate the snapshots,
            # then run the adjoint steps.  The adjoint state (lambda +
            # memory variables) carries across segments.
            ey_f = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
            hx_f = torch.zeros_like(ey_f)
            hz_f = torch.zeros_like(ey_f)
            m_ey_z_f = torch.zeros_like(ey_f)
            m_ey_x_f = torch.zeros_like(ey_f)
            m_hx_z_f = torch.zeros_like(ey_f)
            m_hz_x_f = torch.zeros_like(ey_f)
            ckpt = ctx.ckpt_state
            for k in range(len(segments) - 1, -1, -1):
                s0, s1 = segments[k]
                if s0 > 0:
                    c = ckpt[k - 1]
                    ey_f.copy_(c[0])
                    hx_f.copy_(c[1])
                    hz_f.copy_(c[2])
                    m_ey_z_f.copy_(c[3])
                    m_ey_x_f.copy_(c[4])
                    m_hx_z_f.copy_(c[5])
                    m_hz_x_f.copy_(c[6])
                else:
                    for buf in (
                        ey_f, hx_f, hz_f, m_ey_z_f, m_ey_x_f, m_hx_z_f, m_hz_x_f,
                    ):
                        buf.zero_()
                TM2DFunc._replay_segment(
                    ext, ctx, ca_p, cb_p, cq_p, f, src_i, n_shots, n_src,
                    ey_f, hx_f, hz_f, m_ey_z_f, m_ey_x_f, m_hx_z_f, m_hz_x_f,
                    ey_store, curl_store,
                    ayh, byh, axh, bxh, kyh, kxh,
                    ay, by, ax, bx, ky, kx,
                    grad_stride, n_shots * ny_nx, s0, s1,
                )
                for t in range(s1 - 1, s0 - 1, -1):
                    snap_off = ((t - s0) // grad_stride) * (n_shots * ny_nx)
                    ext.backward_step(
                        ca_p, cb_p, cq_p,
                        lam_ey,
                        m_lambda_hx_z, m_lambda_hz_x,
                        work_x, work_y,
                        work2_x, work2_y,
                        m_lambda_ey_x, m_lambda_ey_z,
                        lam_hx, lam_hz,
                        grad_r, rec_i,
                        grad_f, src_i,
                        ey_store, curl_store,
                        grad_ca, grad_cb,
                        grad_cq,
                        ay, by, ax, bx, ky, kx,
                        ayh, byh, axh, bxh, kyh, kxh,
                        ctx.c,
                        ctx.rdy, ctx.rdx, t, grad_stride, scale, snap_off,
                        ctx.pml_y0, ctx.pml_y1, ctx.pml_x0, ctx.pml_x1,
                        *ctx.fd_pad, ctx.cq_batched, ctx.ca_batched,
                        ctx.cb_batched,
                        n_shots, ny, nx, ny_nx, n_src, n_rec,
                        need_cq, need_f,
                    )
        else:
            for t in range(nt - 1, -1, -1):
                snap_off = ey_storage.snap_offset(t // grad_stride)
                ext.backward_step(
                    ca_p, cb_p, cq_p,
                    lam_ey,
                    m_lambda_hx_z, m_lambda_hz_x,
                    work_x, work_y,
                    work2_x, work2_y,
                    m_lambda_ey_x, m_lambda_ey_z,
                    lam_hx, lam_hz,
                    grad_r, rec_i,
                    grad_f, src_i,
                    ey_store, curl_store,
                    grad_ca, grad_cb,
                    grad_cq,
                    ay, by, ax, bx, ky, kx,
                    ayh, byh, axh, bxh, kyh, kxh,
                    ctx.c,
                    ctx.rdy, ctx.rdx, t, grad_stride, scale, snap_off,
                    ctx.pml_y0, ctx.pml_y1, ctx.pml_x0, ctx.pml_x1,
                    *ctx.fd_pad, ctx.cq_batched, ctx.ca_batched,
                    ctx.cb_batched,
                    n_shots, ny, nx, ny_nx, n_src, n_rec,
                    need_cq, need_f,
                )

        grad_ca = grad_ca if ctx.ca_batched else grad_ca.sum(0, keepdim=True)
        grad_cb = grad_cb if ctx.cb_batched else grad_cb.sum(0, keepdim=True)
        if need_cq:
            grad_cq = grad_cq if ctx.cq_batched else grad_cq.sum(0, keepdim=True)

        return (
            grad_ca, grad_cb, grad_cq, grad_f,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            None, None, None,
        )


def em2d_tm(
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
):
    """2D TM Maxwell (Ey-Hx-Hz) forward modelling / FWI primitive.

    Signature and discretisation follow the reference ``em2d_tm`` exactly:
    ``ca/cb/cq`` material coefficients are compiled from the padded
    epsilon/sigma/mu models (differentiable), sources are pre-scaled by
    ``cb * -1/(dx dy)`` and receivers record the post-injection Ey.

    Args:
        epsilon: Relative permittivity model [ny, nx] (or [1, ny, nx]).
        sigma: Conductivity model (S/m).
        mu: Relative permeability model.
        grid_spacing: Cell size (scalar or [dy, dx]).
        dt: Time step interval (s).
        source_amplitudes: [n_shots, n_src, nt] source amplitudes, or None.
        source_locations: [n_shots, n_src, 2] (y, x) source locations.
        receiver_locations: [n_shots, n_rec, 2] receiver locations.
        accuracy: Finite-difference accuracy order (2, 4, 6 or 8).
        pml_width: PML width (int or [top, bottom, left, right]).
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

    Returns:
        receiver_amplitudes [nt, n_shots, n_rec].
    """
    accuracy = check_accuracy(accuracy)
    if not isinstance(grid_spacing, (list, tuple)):
        grid_spacing = [float(grid_spacing)] * 2
    grid_spacing = [float(g) for g in grid_spacing]
    pml_w = _set_pml_width(pml_width, 2)
    fd_pad = [accuracy // 2, accuracy // 2 - 1, accuracy // 2, accuracy // 2 - 1]
    device = epsilon.device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = epsilon.dtype

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
    check_cfl(grid_spacing, dt, max_vel, "em2d_tm", c_max=1.0)

    (epsilon_p, sigma_p, mu_p), src_i, rec_i = extract_survey_2d(
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
    ny, nx = epsilon_p.shape[-2:]
    profs = _set_em_pml_profiles(
        pml_w, fd_pad, dt, grid_spacing, ny, nx, dtype, device, accuracy=accuracy
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

    source_coeff = -1.0 / (grid_spacing[0] * grid_spacing[1])
    if amp.numel() > 0:
        src_mask = src_i != -1
        src_i_masked = src_i.masked_fill(~src_mask, 0)
        cb_flat = cb_p.reshape(-1, ny * nx).expand(n_shots, -1)
        cb_at_src = cb_flat.gather(1, src_i_masked)
        f = (amp.permute(2, 0, 1) * cb_at_src.unsqueeze(0) * source_coeff).contiguous()
    elif src_i.shape[1] > 0:
        # located sources without amplitudes: inject zero (no-op) amplitudes
        f = torch.zeros(nt_inner, n_shots, src_i.shape[1], device=device, dtype=dtype)
    else:
        f = torch.empty(0, device=device, dtype=dtype)

    rdy, rdx = 1.0 / grid_spacing[0], 1.0 / grid_spacing[1]
    pml_y0, pml_y1 = fd_pad[0] + pml_w[0], ny - fd_pad[1] - pml_w[1]
    pml_x0, pml_x1 = fd_pad[2] + pml_w[2], nx - fd_pad[3] - pml_w[3]
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
    ey_storage = None
    curl_storage = None
    ckpt_state = None
    if storage_enabled:
        ey_storage = SnapshotStorage(
            _ext, n_snap, n_shots, ny, nx, dtype, device,
        )
        curl_storage = SnapshotStorage(
            _ext, n_snap, n_shots, ny, nx, dtype, device,
        )
        if n_ckpt > 0:
            ckpt_state = torch.zeros(
                n_ckpt, N_STATE, n_shots, ny, nx, device=device, dtype=dtype,
            )

    r = TM2DFunc.apply(
        ca_p, cb_p, cq_p, f, src_i, rec_i, profs, c, fd_pad,
        rdy, rdx, nt_inner,
        pml_y0, pml_y1, pml_x0, pml_x1,
        ca_batched, cb_batched, cq_batched,
        grad_stride,
        ey_storage, curl_storage,
        ckpt_state, checkpoint_every, segments,
    )
    return r


def _compile_material_coefficients(epsilon_r, sigma_r, mu_r, dt):
    """Compile material models into 2D TM Maxwell update coefficients.

    Same formulas as ``compile_material_coefficients`` (and the
    FDTD convention used by the eager reference): ca/cb for the E update
    with a conductivity loss averaged over the time step, cq = dt/mu for
    the H update.  Differentiable, so gradients flow from ca/cb/cq back to
    epsilon/sigma/mu.
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
    """a, b, k CPML profiles along one dimension.

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


def _set_em_pml_profiles(
    pml_width,
    fd_pad,
    dt,
    grid_spacing,
    ny,
    nx,
    dtype,
    device,
    accuracy=2,
    eps_scale=EPS0,
):
    """Sets up the twelve flat 1D CPML profiles for the 2D TM staggered grid.

    Returns [ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh] where the
    ``_h`` profiles are evaluated at half-integer grid points.  Same order as
    the ``set_pml_profiles``; the kernels index them directly by cell
    coordinate, so the broadcast shapes are flattened away.

    ``accuracy`` is accepted for API parity with the scalar/staggered PML
    setup and validated, but the EM a/b/k profiles do not depend on the FD
    order (staggered ``set_pml_profiles`` likewise take
    accuracy and ignore it).
    """
    check_accuracy(accuracy)
    dy, dx = grid_spacing
    pml_start = [
        fd_pad[0] + pml_width[0],
        ny - 1 - fd_pad[1] - pml_width[1],
        fd_pad[2] + pml_width[2],
        nx - 1 - fd_pad[3] - pml_width[3],
    ]

    profiles = []
    for half in (False, True):
        ay, by, ky = _pml_profile_1d(
            pml_width[:2], pml_start[:2], dt, ny, dtype, device, half,
            accuracy=accuracy, grid_spacing=dy, eps_scale=eps_scale,
        )
        ax, bx, kx = _pml_profile_1d(
            pml_width[2:], pml_start[2:], dt, nx, dtype, device, half,
            accuracy=accuracy, grid_spacing=dx, eps_scale=eps_scale,
        )
        profiles.extend([ay, ax, by, bx, ky, kx])

    # Reorder to [ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh]
    return [
        profiles[0], profiles[6], profiles[1], profiles[7],
        profiles[2], profiles[8], profiles[3], profiles[9],
        profiles[4], profiles[10], profiles[5], profiles[11],
    ]


def _set_pml_width(pml_width, ndim):
    if isinstance(pml_width, int):
        return [pml_width] * (2 * ndim)
    pml_width = list(pml_width)
    if len(pml_width) != 2 * ndim:
        raise ValueError(f"pml_width must be int or length {2 * ndim}.")
    return pml_width
