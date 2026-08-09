"""Shared forward-callback infrastructure (deepwave-style).

Each propagator exposes a subset of its N_STATE fields to the optional
``forward_callback``; the C++ side invokes the wrapped callable once per
time step with one tensor argument per exposed field, in N_STATE order.
``CallbackState`` wraps those fields so users can read padded / PML /
inner views of any exposed wavefield.
"""

from __future__ import annotations

import operator


class CallbackState:
    """State passed to ``forward_callback`` (deepwave-style).

    The wavefields are the *padded* GPU tensors (fd_pad + PML padding on each
    side); use :meth:`get_wavefield` with ``view`` to select the region:
    ``"full"`` (padded grid), ``"pml"`` (model + PML, fd_pad removed) or
    ``"inner"`` (the unpadded model region).

    The returned tensors are **live views** of the CUDA buffers the
    propagator keeps reusing: later time steps overwrite them in place.  To
    keep a snapshot beyond the current callback invocation, clone it, e.g.
    ``state.get_wavefield("u").clone()``.
    """

    def __init__(self, step, nt, dt, wavefields, fd_pad, pml_width):
        self.step = step
        self.nt = nt
        self.dt = dt
        self._wavefields = wavefields
        self._fd_pad = fd_pad
        self._pml_width = pml_width

    def get_wavefield(self, name, view="inner"):
        """Return the named wavefield exposed to the callback.

        The result is a live view of a buffer the propagator reuses every
        time step; call ``.clone()`` to keep a snapshot.
        """
        if name not in self._wavefields:
            raise KeyError(
                f"unknown wavefield {name!r}; available: {list(self._wavefields)}"
            )
        wf = self._wavefields[name]
        if view == "full":
            return wf
        if view not in ("inner", "pml"):
            raise ValueError(f"view must be 'inner', 'pml' or 'full', got {view!r}")
        fd = self._fd_pad
        if view == "inner":
            pad = [f + p for f, p in zip(fd, self._pml_width, strict=True)]
        else:  # pml
            pad = fd
        # pad is [lo0, hi0, lo1, hi1, ...] over the spatial dims (4 entries
        # in 2D, 6 in 3D), so pair up (lo, hi) per dim.
        pairs = list(zip(pad[0::2], pad[1::2], strict=True))
        slices = [
            slice(lo, size - hi)
            for (lo, hi), size in zip(pairs, wf.shape[-len(pairs):], strict=True)
        ]
        return wf[(..., *slices)]

    @property
    def wavefield_names(self):
        """Names accepted by :meth:`get_wavefield`, in a stable order."""
        return tuple(self._wavefields)


def wrap_forward_callback(fn, names, dt, fd_pad, pml_width):
    """Adapt the C++ callback signature to the user-facing CallbackState API.

    ``names`` are the callback-exposed fields, in the order the C++ side
    passes them to the wrapped callable.
    """

    if not callable(fn):
        raise TypeError("forward_callback must be callable")

    def cb(t, nt, *fields):
        wf = dict(zip(names, fields, strict=True))
        fn(CallbackState(int(t), int(nt), dt, wf, fd_pad, pml_width))

    return cb


def validate_callback_frequency(value):
    """Return a positive integer callback frequency or raise ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError("callback_frequency must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("callback_frequency must be a positive integer") from exc
    if value < 1:
        raise ValueError("callback_frequency must be a positive integer")
    return int(value)
