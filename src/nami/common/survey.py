"""Survey extraction: padded models + flat source/receiver indices (torch).

Survey conventions: models are edge-padded by ``(fd_pad + pml_width)``
on each side and locations become flat indices into the padded grid
(-1 = "ignored").
"""

from collections.abc import Sequence

import torch


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
    """
    spatial = tuple(models[0].shape[-2:])
    for m in models:
        if tuple(m.shape[-2:]) != spatial:
            raise ValueError("All models must have the same spatial shape.")

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
    (zero-filled, matching the scatter padding).
    """
    if model.ndim not in (2, 3):
        raise ValueError(f"model must be 2D or 3D, got {model.ndim}D.")
    if model.ndim == 2:
        model = model[None]
    top, bottom, left, right = pad
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
    for m in models:
        if tuple(m.shape[-3:]) != spatial:
            raise ValueError("All models must have the same spatial shape.")

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
    """
    if model.ndim not in (3, 4):
        raise ValueError(f"model must be 3D or batched 4D, got {model.ndim}D.")
    if model.ndim == 3:
        model = model[None]
    z0, z1, y0, y1, x0, x1 = pad
    return torch.nn.functional.pad(
        model, (x0, x1, y0, y1, z0, z1), mode=mode
    )
