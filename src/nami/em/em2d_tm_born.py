"""nami em2d_tm Born: first-order Born scattering for the 2D TM Maxwell grid.

A background field is propagated with the
unperturbed ca/cb/cq coefficients and a scattered field is driven by the
first-order scattering sources

    d_ey = ca*d_ey + cb*dcurl + dca*Ey_old + dcb*curl
    d_hx -= cq*ddey_dy + dcq*dey_dy     (and the hz analogue)

where ``Ey_old``/``curl`` are the pre-update background field/curl, with
``dca/dcb`` the exact linearization of the material coefficients w.r.t.
``(depsilon, dsigma)``, and
``dcq = -cq/mu*dmu`` when a ``mu_scatter`` is given (``dcq`` is zero by
default).  Source pre-scaling follows nami's ``em2d_tm``
(``cb * -1/(dx dy)`` for the background source and the ``dcb`` analogue for
the scattered source, matching the source linearization); receivers
record the post-injection scattered ``d_ey``.

The adjoint is the exact discrete transpose of the coupled
(background + scattered) system (the same two-stage E/H structure as
:mod:`em2d_tm`, with separate adjoints for each field), so gradients flow
to both the background models and the scatter models.

Return convention: ``[nt, n_shots, n_rec]`` scattered receiver traces.
"""

import nami_em2d_tm_born as _ext
import torch

from ..common.callback import validate_callback_frequency, wrap_forward_callback
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
    is_shot_batched,
    prepare_source_amplitudes,
)
from ._common import EPS0, _compile_material_coefficients
from .em2d_tm import _set_em_pml_profiles

# Checkpoint state layout (saved at time t before step t):
#   [0] ey        background pre-update E field
#   [1] d_ey      scattered pre-update E field
#   [2] hx        background H field (y)
#   [3] hz        background H field (x)
#   [4] d_hx      scattered H field (y)
#   [5] d_hz      scattered H field (x)
#   [6] m_ey_z    split-field memory (z) from the bg H half-step
#   [7] m_ey_x    split-field memory (x) from the bg H half-step
#   [8] dm_ey_z   split-field memory (z) from the sc H half-step
#   [9] dm_ey_x   split-field memory (x) from the sc H half-step
#   [10] m_hx_z   split-field memory (z) from the bg E integer-step
#   [11] m_hz_x   split-field memory (x) from the bg E integer-step
#   [12] dm_hx_z  split-field memory (z) from the sc E integer-step
#   [13] dm_hz_x  split-field memory (x) from the sc E integer-step
N_STATE = 14
N_STREAMS = 8

# Full N_STATE names: keys of the state dicts returned by ``return_state``
# and accepted by ``initial_state`` (for split-run continuation).
_WAVEFIELD_NAMES = (
    "ey", "d_ey", "hx", "hz", "d_hx", "d_hz",
    "m_ey_z", "m_ey_x", "dm_ey_z", "dm_ey_x",
    "m_hx_z", "m_hz_x", "dm_hx_z", "dm_hz_x",
)
_CALLBACK_FIELDS = ("ey", "hx", "hz", "d_ey", "d_hx", "d_hz")


class BornEMFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ca_p, cb_p, cq_p,           # [n_shots, ny, nx] or [1, ny, nx] padded bg
        dca_p, dcb_p, dcq_p,        # padded scatter coefficient linearizations
        f_bg, f_sc,                 # [nt, n_shots, n_src] pre-scaled sources
        src_i, rec_i, bg_rec_i,
        profs, c, fd_pad,
        rdy, rdx,
        nt, pml_y0, pml_y1, pml_x0, pml_x1,
        ca_batched, cb_batched, cq_batched,
        dca_batched, dcb_batched, dcq_batched,
        grad_stride,
        storage,         # list of 8 SnapshotStorage or None
        ckpt_state,      # [n_ckpt, N_STATE, n_shots, ny, nx] or None
        checkpoint_every,  # 0 = full storage; N = checkpoint every N steps
        segments,        # [(s0, s1)] replay segments; [] = full storage
        init_state,      # [N_STATE, n_shots, ny, nx] initial wavefield or None
        final_state,     # [N_STATE, n_shots, ny, nx] output buffer or None
        forward_callback,  # cb(t, nt, ey, hx, hz, d_ey, d_hx, d_hz) or None
        callback_frequency,  # call the callback every N steps
    ):
        ext = _ext
        device = ca_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = ca_p.dtype
        # n_shots from survey (sources); shared [1, ny, nx] models still run
        # n_shots wavefields (*_batched selects model slab 0).
        n_shots = int(src_i.shape[0])
        ny, nx = ca_p.shape[-2:]
        n_src = src_i.shape[1]
        n_rec = rec_i.shape[1]
        n_bg_rec = bg_rec_i.shape[1]
        ny_nx = ny * nx
        shot_count = n_shots * ny_nx

        z = lambda *shape: torch.zeros(shape, device=device, dtype=dtype)  # noqa: E731
        ey, d_ey = z(n_shots, ny, nx), z(n_shots, ny, nx)
        hx, hz = z(n_shots, ny, nx), z(n_shots, ny, nx)
        d_hx, d_hz = z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_ey_z, m_ey_x = z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_hx_z, m_hz_x = z(n_shots, ny, nx), z(n_shots, ny, nx)
        dm_ey_z, dm_ey_x = z(n_shots, ny, nx), z(n_shots, ny, nx)
        dm_hx_z, dm_hz_x = z(n_shots, ny, nx), z(n_shots, ny, nx)
        r = z(nt, n_shots, n_rec)
        r_bg = z(nt, n_shots, n_bg_rec)
        if storage is not None:
            (ey_store, curl_store, d_ey_store, dcurl_store,
             dey_dy_store, dey_dx_store, ddey_dy_store, ddey_dx_store) = [
                st.snap for st in storage
            ]
        else:
            dummy = z(n_shots, ny, nx)
            (ey_store, curl_store, d_ey_store, dcurl_store,
             dey_dy_store, dey_dx_store, ddey_dy_store, ddey_dx_store) = (
                dummy,
            ) * 8

        ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh = [
            p.contiguous() for p in profs
        ]
        # N_STATE layout: [0] ey [1] d_ey [2] hx [3] hz [4] d_hx [5] d_hz
        # [6] m_ey_z [7] m_ey_x [8] dm_ey_z [9] dm_ey_x
        # [10] m_hx_z [11] m_hz_x [12] dm_hx_z [13] dm_hz_x
        state = [
            ey, d_ey, hx, hz, d_hx, d_hz,
            m_ey_z, m_ey_x, dm_ey_z, dm_ey_x,
            m_hx_z, m_hz_x, dm_hx_z, dm_hz_x,
        ]
        # Checkpointed forward skips global snapshot writes; replay
        # regenerates segment-local snaps via snap_off.
        store = 0 if segments else (1 if storage is not None else 0)
        ext.born_em_forward_loop(
            ca_p, cb_p, cq_p, dca_p, dcb_p, dcq_p,
            state,
            ey_store, curl_store, d_ey_store, dcurl_store,
            dey_dy_store, dey_dx_store, ddey_dy_store, ddey_dx_store,
            ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh,
            c,
            f_bg, f_sc, src_i, r, rec_i, r_bg, bg_rec_i,
            rdy, rdx, nt, grad_stride,
            pml_y0, pml_y1, pml_x0, pml_x1,
            fd_pad[0], fd_pad[1], fd_pad[2], fd_pad[3],
            ca_batched, cb_batched, cq_batched,
            dca_batched, dcb_batched, dcq_batched,
            store, checkpoint_every, ckpt_state,
            init_state, final_state, forward_callback, callback_frequency,
        )

        ctx.save_for_backward(
            ca_p, cb_p, cq_p, dca_p, dcb_p, dcq_p, f_bg, f_sc,
            src_i, rec_i, bg_rec_i,
        )
        ctx.profs = profs
        ctx.c = c
        ctx.fd_pad = fd_pad
        ctx.rdy, ctx.rdx = rdy, rdx
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.ny, ctx.nx, ctx.ny_nx = ny, nx, ny_nx
        ctx.shot_count = shot_count
        ctx.n_src, ctx.n_rec, ctx.n_bg_rec = n_src, n_rec, n_bg_rec
        ctx.pml_y0, ctx.pml_y1, ctx.pml_x0, ctx.pml_x1 = (
            pml_y0, pml_y1, pml_x0, pml_x1,
        )
        ctx.ca_batched, ctx.cb_batched, ctx.cq_batched = (
            ca_batched, cb_batched, cq_batched,
        )
        ctx.dca_batched, ctx.dcb_batched, ctx.dcq_batched = (
            dca_batched, dcb_batched, dcq_batched,
        )
        ctx.grad_stride = grad_stride
        ctx.storage = storage
        ctx.ckpt_state = ckpt_state
        ctx.checkpoint_every = checkpoint_every
        ctx.segments = segments
        return r, r_bg

    @staticmethod
    def backward(ctx, grad_r, grad_r_bg):
        ext = _ext
        (ca_p, cb_p, cq_p, dca_p, dcb_p, dcq_p, f_bg, f_sc,
         src_i, rec_i, bg_rec_i) = ctx.saved_tensors
        if ctx.storage is None:
            raise RuntimeError(
                "em2d_tm_born backward() requires snapshot storage: run the "
                "forward with an input requiring grad (and not under "
                "torch.no_grad())."
            )
        storage = ctx.storage
        (ey_store, curl_store, d_ey_store, dcurl_store,
         dey_dy_store, dey_dx_store, ddey_dy_store, ddey_dx_store) = [
            st.snap for st in storage
        ]
        device = ca_p.device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dtype = ca_p.dtype
        n_shots = ctx.n_shots
        ny, nx = ctx.ny, ctx.nx
        n_src, n_rec, n_bg_rec = ctx.n_src, ctx.n_rec, ctx.n_bg_rec
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
        grad_ca = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        grad_cb = torch.zeros_like(grad_ca)
        grad_cq = torch.zeros_like(grad_ca)
        grad_dca = torch.zeros_like(grad_ca)
        grad_dcb = torch.zeros_like(grad_ca)
        grad_dcq = torch.zeros_like(grad_ca)

        z = lambda *shape: torch.zeros(shape, device=device, dtype=dtype)  # noqa: E731
        lam_ey, lam_d_ey = z(n_shots, ny, nx), z(n_shots, ny, nx)
        lam_hx, lam_hz = z(n_shots, ny, nx), z(n_shots, ny, nx)
        lam_dhx, lam_dhz = z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_lambda_hx_z, m_lambda_hz_x = z(n_shots, ny, nx), z(n_shots, ny, nx)
        dm_lambda_hx_z, dm_lambda_hz_x = z(n_shots, ny, nx), z(n_shots, ny, nx)
        m_lambda_ey_x, m_lambda_ey_z = z(n_shots, ny, nx), z(n_shots, ny, nx)
        dm_lambda_ey_x, dm_lambda_ey_z = z(n_shots, ny, nx), z(n_shots, ny, nx)
        work_x, work_y = z(n_shots, ny, nx), z(n_shots, ny, nx)
        work_dx, work_dy = z(n_shots, ny, nx), z(n_shots, ny, nx)
        work2_x, work2_y = z(n_shots, ny, nx), z(n_shots, ny, nx)
        work2_dx, work2_dy = z(n_shots, ny, nx), z(n_shots, ny, nx)

        ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh = [
            p.contiguous() for p in ctx.profs
        ]
        grad_stride = ctx.grad_stride
        # integral sampling: each snapshot represents
        # `grad_stride` time steps of the model-gradient integral.
        scale = float(grad_stride)
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
            ey_f, d_ey_f = z(n_shots, ny, nx), z(n_shots, ny, nx)
            hx_f, hz_f = z(n_shots, ny, nx), z(n_shots, ny, nx)
            d_hx_f, d_hz_f = z(n_shots, ny, nx), z(n_shots, ny, nx)
            m_ey_z_f, m_ey_x_f = z(n_shots, ny, nx), z(n_shots, ny, nx)
            dm_ey_z_f, dm_ey_x_f = z(n_shots, ny, nx), z(n_shots, ny, nx)
            m_hx_z_f, m_hz_x_f = z(n_shots, ny, nx), z(n_shots, ny, nx)
            dm_hx_z_f, dm_hz_x_f = z(n_shots, ny, nx), z(n_shots, ny, nx)
            state_f = [
                ey_f, d_ey_f, hx_f, hz_f, d_hx_f, d_hz_f,
                m_ey_z_f, m_ey_x_f, dm_ey_z_f, dm_ey_x_f,
                m_hx_z_f, m_hz_x_f, dm_hx_z_f, dm_hz_x_f,
            ]
        else:
            state_f = []
        ext.born_em_adjoint_loop(
            ca_p, cb_p, cq_p, dca_p, dcb_p, dcq_p,
            ey_store, curl_store, d_ey_store, dcurl_store,
            dey_dy_store, dey_dx_store, ddey_dy_store, ddey_dx_store,
            grad_ca, grad_cb, grad_cq, grad_dca, grad_dcb, grad_dcq,
            grad_f_bg, grad_f_sc, grad_r, grad_r_bg,
            src_i, rec_i, bg_rec_i,
            f_bg, f_sc,
            lam_ey, lam_d_ey,
            lam_hx, lam_hz, lam_dhx, lam_dhz,
            m_lambda_hx_z, m_lambda_hz_x, dm_lambda_hx_z, dm_lambda_hz_x,
            m_lambda_ey_x, m_lambda_ey_z, dm_lambda_ey_x, dm_lambda_ey_z,
            work_x, work_y, work_dx, work_dy,
            work2_x, work2_y, work2_dx, work2_dy,
            state_f,
            ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh,
            ctx.c,
            ctx.rdy, ctx.rdx, scale,
            nt, grad_stride,
            ctx.pml_y0, ctx.pml_y1, ctx.pml_x0, ctx.pml_x1,
            *ctx.fd_pad,
            ctx.ca_batched, ctx.cb_batched, ctx.cq_batched,
            ctx.dca_batched, ctx.dcb_batched, ctx.dcq_batched,
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

        # 36 forward() inputs: coefficient/source grads + 28 x None.
        return (
            grad_ca, grad_cb, grad_cq, grad_dca, grad_dcb, grad_dcq,
            grad_f_bg, grad_f_sc,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None,
        )


def em2d_tm_born(
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
    forward_callback=None,
    callback_frequency=1,
    return_state=False,
    initial_state=None,
):
    """2D TM Maxwell Born forward + adjoint (torch in/out).

    ``epsilon``/``sigma``/``mu`` are the background (relative permittivity,
    conductivity, relative permeability) models; ``epsilon_scatter`` /
    ``sigma_scatter`` / ``mu_scatter`` are the parameter perturbations
    (``mu_scatter`` defaults to None because
    ``mu`` is fixed — pass it to include the ``dcq`` scattering term).
    The scattered wavefield is linear in the perturbations and
    differentiable w.r.t. both the background and the scatter models.

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
            The available wavefields are ``ey``, ``hx``, ``hz`` and their
            scattered ``d_*`` counterparts.
        callback_frequency: call ``forward_callback`` every N time steps.
        return_state: if True, append the final state dict to the outputs.
            The state is a
            dict of the FINAL padded wavefield state (keys ``ey``, ``d_ey``,
            ``hx``, ``hz``, ``d_hx``, ``d_hz``, plus the PML memory
            variables) suitable for continuation via ``initial_state``.
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

    Returns scattered receiver traces ``[nt, n_shots, n_rec]``.  When
    ``bg_receiver_locations`` is provided, returns ``(r, r_bg)``; with
    ``return_state=True`` the state dict is appended.
    """
    accuracy = check_accuracy(accuracy)
    if not isinstance(grid_spacing, (list, tuple)):
        grid_spacing = [float(grid_spacing)] * 2
    grid_spacing = [float(g) for g in grid_spacing]
    pml_w = set_pml_width(pml_width, 2)
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
    if bg_receiver_locations is not None:
        (_,), _, bg_rec_i = extract_survey_2d(
            [epsilon], None, bg_receiver_locations, fd_pad, pml_w,
            n_shots, device, dtype,
        )
    else:
        bg_rec_i = torch.empty((n_shots, 0), dtype=torch.int64, device=device)

    def _pad_scatter(model):
        if model is None:
            return None
        (m_p,), _, _ = extract_survey_2d(
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

    ny, nx = epsilon_p.shape[-2:]
    profs = _set_em_pml_profiles(
        pml_w, fd_pad, dt, grid_spacing, ny, nx, dtype, device, accuracy=accuracy
    )
    c = staggered_diff1_coeffs(accuracy, dtype, device)

    amp = prepare_source_amplitudes(
        source_amplitudes, n_shots, src_i.shape[1], nt_inner,
        device=device, dtype=dtype,
    )

    source_coeff = -1.0 / (grid_spacing[0] * grid_spacing[1])
    if amp.numel() > 0:
        src_mask = src_i != -1
        src_i_masked = src_i.masked_fill(~src_mask, 0)
        cb_flat = cb_p.reshape(-1, ny * nx).expand(n_shots, -1)
        cb_at_src = cb_flat.gather(1, src_i_masked)
        dcb_flat = dcb_p.reshape(-1, ny * nx).expand(n_shots, -1)
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

    rdy, rdx = 1.0 / grid_spacing[0], 1.0 / grid_spacing[1]
    pml_y0, pml_y1 = fd_pad[0] + pml_w[0], ny - fd_pad[1] - pml_w[1]
    pml_x0, pml_x1 = fd_pad[2] + pml_w[2], nx - fd_pad[3] - pml_w[3]
    # Linearised coefficients combine background and scatter tensors, so
    # their actual post-broadcast shapes are the only reliable batch flags.
    ca_batched = int(is_shot_batched(ca_p, n_shots))
    cb_batched = int(is_shot_batched(cb_p, n_shots))
    cq_batched = int(is_shot_batched(cq_p, n_shots))
    dca_batched = int(is_shot_batched(dca_p, n_shots))
    dcb_batched = int(is_shot_batched(dcb_p, n_shots))
    dcq_batched = int(is_shot_batched(dcq_p, n_shots))

    storage_mode = resolve_storage(storage)
    grad_stride = check_sample_steps(sample_steps)
    storage_enabled = storage_mode == "auto" and torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (
            epsilon, sigma, mu,
            epsilon_scatter, sigma_scatter, mu_scatter,
            source_amplitudes,
        )
    )
    checkpoint_every, segments, n_snap, n_ckpt = storage_plan(
        nt_inner, N_STATE, grad_stride, N_STREAMS, storage_enabled,
        ckpt_steps=ckpt_steps,
    )
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

    r, r_bg = BornEMFunc.apply(
        ca_p, cb_p, cq_p, dca_p, dcb_p, dcq_p, f_bg, f_sc,
        src_i, rec_i, bg_rec_i, profs, c, fd_pad,
        rdy, rdx, nt_inner,
        pml_y0, pml_y1, pml_x0, pml_x1,
        ca_batched, cb_batched, cq_batched,
        dca_batched, dcb_batched, dcq_batched,
        grad_stride,
        stores, ckpt_state, checkpoint_every, segments,
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
