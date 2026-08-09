"""Shared validation and allocation for propagator state I/O.

State dicts are ephemeral runtime snapshots: they contain implementation
details (padded grids, PML memory variables) and may be passed back only to
the same propagator with the same model layout and the same nami version.
They are not a stable long-term checkpoint format.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def prepare_initial_state(initial_state, names, shape, *, device, dtype):
    """Pack a public state mapping into the dense tensor used by kernels.

    State values may be shared across shots (spatial shape or a leading
    singleton) or explicitly batched with the requested shot count. Missing
    fields start from zero. State values are detached because state I/O is a
    forward-continuation boundary, not an autograd connection between runs.
    """
    if initial_state is None:
        return None
    if not isinstance(initial_state, Mapping):
        raise TypeError("initial_state must be a mapping from field names to tensors")

    names = tuple(names)
    unknown = set(initial_state) - set(names)
    if unknown:
        available = ", ".join(names)
        bad = ", ".join(repr(name) for name in sorted(unknown, key=repr))
        raise ValueError(
            f"unknown initial_state field(s): {bad}; available fields: {available}"
        )
    if not initial_state:
        return None

    shape = tuple(shape)
    if len(shape) < 2:
        raise ValueError("state shape must contain shot and spatial dimensions")
    spatial_shape = shape[1:]
    allowed_shapes = {spatial_shape, (1, *spatial_shape), shape}
    packed = torch.zeros(len(names), *shape, device=device, dtype=dtype)
    for i, name in enumerate(names):
        if name not in initial_state:
            continue
        value = initial_state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"initial_state[{name!r}] must be a torch.Tensor")
        if tuple(value.shape) not in allowed_shapes:
            expected = ", ".join(str(s) for s in sorted(allowed_shapes))
            raise ValueError(
                f"initial_state[{name!r}] has shape {tuple(value.shape)}; "
                f"expected one of {expected}"
            )
        packed[i] = value.detach().to(device=device, dtype=dtype)
    return packed


def allocate_final_state(enabled, names, shape, *, device, dtype):
    """Allocate the dense final-state buffer requested by a propagator."""
    if not enabled:
        return None
    if not isinstance(names, Sequence) or not names:
        raise ValueError("state names must be a non-empty sequence")
    return torch.zeros(len(names), *shape, device=device, dtype=dtype)


def unpack_state(state, names):
    """Expose a dense final-state buffer as a detached field-name mapping."""
    if state is None:
        raise ValueError("cannot unpack an unallocated final state")
    names = tuple(names)
    if state.shape[0] != len(names):
        raise ValueError(
            f"state has {state.shape[0]} fields but {len(names)} names were provided"
        )
    return {name: state[i].detach() for i, name in enumerate(names)}
