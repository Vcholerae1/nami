"""Survey extraction: padded models + flat source/receiver indices (torch).

Survey conventions: models are edge-padded by ``(fd_pad + pml_width)``
on each side and locations become flat indices into the padded grid
(-1 = "ignored").

``"replicate"`` padding is built with ``cat``/``expand`` instead of
``F.pad``: same values, but a deterministic backward (torch's
``replication_pad*_backward`` CUDA kernels accumulate with atomicAdd in a
nondeterministic order, which makes model gradients irreproducible at the
ULP level).
"""

from collections.abc import Sequence

import torch


def is_shot_batched(model, n_shots, spatial_ndim=2):
    """True if ``model`` is an explicit per-shot batch.

    A tensor is shot-batched when its leading dim is ``n_shots > 1`` and
    ``ndim == spatial_ndim + 1``.  ``None`` and shared shapes
    (``[…spatial]`` or ``[1, …spatial]``) are not batched.
    """
    if model is None:
        return False
    return (
        model.ndim == spatial_ndim + 1
        and model.shape[0] == n_shots
        and n_shots > 1
    )


def check_model_batching(models, names, n_shots, spatial_ndim=2):
    """Require a uniform shot-batch form within a multi-parameter group.

    Some kernels take a single ``*_batched`` flag for several models (e.g.
    elastic ``lamb``/``mu``/``buoyancy``).  Mixing a batched tensor with a
    shared one would index the shared slab at ``s ≥ 1`` (OOB).  ``None``
    counts as shared (defaults to zeros of the background form).

    Call this on the *user* tensors before pad, never on padded buffers.
    """
    flags = [is_shot_batched(m, n_shots, spatial_ndim) for m in models]
    if any(flags) and not all(flags):
        parts = []
        for name, m in zip(names, models, strict=True):
            if m is None:
                parts.append(f"{name} None (shared zeros)")
            else:
                parts.append(f"{name} {tuple(m.shape)}")
        raise ValueError(
            "models must be all shared ([spatial]/[1, spatial] or None) "
            f"or all batched ([n_shots, spatial]); got {', '.join(parts)}."
        )


def extract_survey_2d(
    models: list[torch.Tensor],
    source_locations: torch.Tensor | None,
    receiver_locations: torch.Tensor | None,
    fd_pad: Sequence[int],
    pml_width: Sequence[int],
    n_shots: int,
    device: torch.device,
    dtype: torch.dtype,
    pad_modes: Sequence[str] | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Returns (padded_models, sources_i, receivers_i).

    ``sources_i``/``receivers_i`` are [n_shots, n_loc] int64 flat indices
    into the padded grid. ``pad_modes`` optionally gives the padding mode
    (``"replicate"`` or ``"constant"``) for each model; it defaults to
    ``"replicate"`` for all (matching the scatter padding, which is
    constant/zero).

    Batched models must have batch size 1 (shared) or ``n_shots``; any
    other batch size raises ``ValueError``.
    """
    spatial = tuple(models[0].shape[-2:])
    for i, m in enumerate(models):
        if tuple(m.shape[-2:]) != spatial:
            raise ValueError("All models must have the same spatial shape.")
        if m.ndim == 3 and m.shape[0] not in (1, n_shots):
            raise ValueError(
                f"models[{i}] batch size must be 1 (shared) or n_shots "
                f"({n_shots}), got shape {tuple(m.shape)}."
            )

    pad = [f + p for f, p in zip(fd_pad, pml_width, strict=False)]
    top, bottom, left, right = pad
    padded_shape = (spatial[0] + top + bottom, spatial[1] + left + right)

    if pad_modes is None:
        pad_modes = ["replicate"] * len(models)
    if len(pad_modes) != len(models):
        raise ValueError("pad_modes must have one entry per model.")
    padded_models = [
        _pad_model_2d(m.to(device, dtype), pad, mode)
        for m, mode in zip(models, pad_modes, strict=True)
    ]

    def _flatten(locations: torch.Tensor | None) -> torch.Tensor:
        if locations is None:
            return torch.empty((n_shots, 0), dtype=torch.int64, device=device)
        loc = locations.to(torch.int64)
        if loc.ndim != 3 or loc.shape[0] != n_shots:
            raise ValueError("locations must have shape [n_shots, n_loc, 2].")
        valid = (loc[..., 0] >= 0) & (loc[..., 1] >= 0)
        if bool((valid & (loc[..., 0] >= spatial[0])).any()) or bool(
            (valid & (loc[..., 1] >= spatial[1])).any()
        ):
            raise RuntimeError("Locations must be within model.")
        y = torch.where(valid, loc[..., 0] + top, 0)
        x = torch.where(valid, loc[..., 1] + left, 0)
        flat = y * padded_shape[1] + x
        return torch.where(valid, flat, -1).to(device)

    sources_i = _flatten(source_locations)
    receivers_i = _flatten(receiver_locations)
    return padded_models, sources_i, receivers_i


def _pad_model_2d(
    model: torch.Tensor, pad: Sequence[int], mode: str = "replicate"
) -> torch.Tensor:
    """Pads a [ny, nx] or [1, ny, nx] model by [top, bottom, left, right].

    ``mode`` is ``"replicate"`` (edge replication) or ``"constant"``
    (zero-filled, matching the scatter padding).  ``"replicate"`` uses
    ``cat``/``expand`` (deterministic backward); other modes fall through
    to ``F.pad``.
    """
    if model.ndim not in (2, 3):
        raise ValueError(f"model must be 2D or 3D, got {model.ndim}D.")
    if model.ndim == 2:
        model = model[None]
    top, bottom, left, right = pad
    if mode == "replicate":
        if top or bottom:
            model = torch.cat(
                [
                    model[:, :1].expand(-1, top, -1),
                    model,
                    model[:, -1:].expand(-1, bottom, -1),
                ],
                dim=1,
            )
        if left or right:
            model = torch.cat(
                [
                    model[:, :, :1].expand(-1, -1, left),
                    model,
                    model[:, :, -1:].expand(-1, -1, right),
                ],
                dim=2,
            )
        return model
    if top or bottom:
        model = torch.nn.functional.pad(model, (0, 0, top, bottom), mode=mode)
    if left or right:
        model = torch.nn.functional.pad(model, (left, right, 0, 0), mode=mode)
    return model


def extract_survey_3d(
    models: list[torch.Tensor],
    source_locations: torch.Tensor | None,
    receiver_locations: torch.Tensor | None,
    fd_pad: Sequence[int],
    pml_width: Sequence[int],
    n_shots: int,
    device: torch.device,
    dtype: torch.dtype,
    pad_modes: Sequence[str] | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """3D survey extraction (see :func:`extract_survey_2d`).

    Locations are ``[n_shots, n_loc, 3]`` (z, y, x); flat indices use the
    padded grid's row-major layout ``z * (ny*nx) + y * nx + x``.
    """
    spatial = tuple(models[0].shape[-3:])
    for i, m in enumerate(models):
        if tuple(m.shape[-3:]) != spatial:
            raise ValueError("All models must have the same spatial shape.")
        if m.ndim == 4 and m.shape[0] not in (1, n_shots):
            raise ValueError(
                f"models[{i}] batch size must be 1 (shared) or n_shots "
                f"({n_shots}), got shape {tuple(m.shape)}."
            )

    pad = [f + p for f, p in zip(fd_pad, pml_width, strict=False)]
    p0, p1, p2, p3, p4, p5 = pad
    padded_shape = (
        spatial[0] + p0 + p1,
        spatial[1] + p2 + p3,
        spatial[2] + p4 + p5,
    )
    starts = (p0, p2, p4)
    strides = (
        padded_shape[1] * padded_shape[2],
        padded_shape[2],
        1,
    )

    if pad_modes is None:
        pad_modes = ["replicate"] * len(models)
    if len(pad_modes) != len(models):
        raise ValueError("pad_modes must have one entry per model.")
    padded_models = [
        _pad_model_3d(m.to(device, dtype), pad, mode)
        for m, mode in zip(models, pad_modes, strict=True)
    ]

    def _flatten(locations: torch.Tensor | None) -> torch.Tensor:
        if locations is None:
            return torch.empty((n_shots, 0), dtype=torch.int64, device=device)
        loc = locations.to(torch.int64)
        if loc.ndim != 3 or loc.shape[0] != n_shots or loc.shape[2] != 3:
            raise ValueError("locations must have shape [n_shots, n_loc, 3].")
        valid = torch.ones(loc.shape[:2], dtype=torch.bool, device=loc.device)
        for dim in range(3):
            valid = valid & (loc[..., dim] >= 0)
            if bool((valid & (loc[..., dim] >= spatial[dim])).any()):
                raise RuntimeError("Locations must be within model.")
        flat = torch.zeros(loc.shape[:2], dtype=torch.int64, device=loc.device)
        for dim in range(3):
            shifted = torch.where(valid, loc[..., dim] + starts[dim], 0)
            flat = flat + shifted * strides[dim]
        return torch.where(valid, flat, -1).to(device)

    sources_i = _flatten(source_locations)
    receivers_i = _flatten(receiver_locations)
    return padded_models, sources_i, receivers_i


def _pad_model_3d(
    model: torch.Tensor, pad: Sequence[int], mode: str = "replicate"
) -> torch.Tensor:
    """Pads a [nz, ny, nx] or [1, nz, ny, nx] model.

    ``pad`` is ``[z0, z1, y0, y1, x0, x1]`` (matching the 3D survey
    convention); ``torch.nn.functional.pad`` wants the reverse order.
    ``"replicate"`` uses ``cat``/``expand`` (deterministic backward);
    other modes fall through to ``F.pad``.
    """
    if model.ndim not in (3, 4):
        raise ValueError(f"model must be 3D or batched 4D, got {model.ndim}D.")
    if model.ndim == 3:
        model = model[None]
    z0, z1, y0, y1, x0, x1 = pad
    if mode == "replicate":
        if z0 or z1:
            model = torch.cat(
                [
                    model[:, :1].expand(-1, z0, -1, -1),
                    model,
                    model[:, -1:].expand(-1, z1, -1, -1),
                ],
                dim=1,
            )
        if y0 or y1:
            model = torch.cat(
                [
                    model[:, :, :1].expand(-1, -1, y0, -1),
                    model,
                    model[:, :, -1:].expand(-1, -1, y1, -1),
                ],
                dim=2,
            )
        if x0 or x1:
            model = torch.cat(
                [
                    model[:, :, :, :1].expand(-1, -1, -1, x0),
                    model,
                    model[:, :, :, -1:].expand(-1, -1, -1, x1),
                ],
                dim=3,
            )
        return model
    return torch.nn.functional.pad(model, (x0, x1, y0, y1, z0, z1), mode=mode)
