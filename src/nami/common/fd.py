"""Finite-difference coefficient tables.

The values reproduce the reference ``regular_grid`` (scalar/acoustic)
and ``staggered_grid`` (elastic / EM staggered grids) tables exactly:
the
fractions are evaluated in IEEE double precision and cast to the model
dtype, matching nvcc's ``(DW_DTYPE)(a/b)`` constant folding.

- ``DIFF1``: symmetric first derivative on a regular grid
  (accuracy 2 -> [1/2], 4 -> [8/12, -1/12], 6 -> [3/4, -3/20, 1/60],
  8 -> [4/5, -1/5, 4/105, -1/280]).
- ``DIFF2``: second derivative on a regular grid; the first entry is the
  center weight (accuracy 2 -> [-2, 1], 4 -> [-5/2, 4/3, -1/12],
  6 -> [-49/18, 3/2, -3/20, 1/90], 8 -> [-205/72, 8/5, -1/5, 8/315, -1/560]).
- ``STAGGERED_DIFF1``: first derivative on a staggered grid (offset pairs
  (0,-1), (1,-2), ... for the integer-point operator and (1,0), (2,-1), ...
  for the half-integer-point operator; accuracy 2 -> [1],
  4 -> [9/8, -1/24], 6 -> [75/64, -25/384, 3/640],
  8 -> [1225/1024, -245/3072, 49/5120, -5/7168]).

``MAX_RADIUS = 4`` is the fixed stencil radius of the CUDA kernels; lower
accuracies zero-pad their coefficient arrays to that length.
"""

from __future__ import annotations

MAX_RADIUS = 4

DIFF1 = {
    2: [1 / 2],
    4: [8 / 12, -1 / 12],
    6: [3 / 4, -3 / 20, 1 / 60],
    8: [4 / 5, -1 / 5, 4 / 105, -1 / 280],
}

DIFF2 = {
    2: [-2, 1],
    4: [-5 / 2, 4 / 3, -1 / 12],
    6: [-49 / 18, 3 / 2, -3 / 20, 1 / 90],
    8: [-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560],
}

STAGGERED_DIFF1 = {
    2: [1],
    4: [9 / 8, -1 / 24],
    6: [75 / 64, -25 / 384, 3 / 640],
    8: [1225 / 1024, -245 / 3072, 49 / 5120, -5 / 7168],
}

_VALID = (2, 4, 6, 8)


def check_accuracy(accuracy: int) -> int:
    """Validates an FD accuracy order (2, 4, 6, 8) and returns it."""
    if accuracy not in _VALID:
        raise NotImplementedError(
            f"nami supports FD accuracy orders 2, 4, 6 and 8; got {accuracy!r}."
        )
    return int(accuracy)


def diff1_coeffs(accuracy: int, dtype=None, device=None):
    """First-derivative (regular grid) coefficients, padded to ``MAX_RADIUS``."""
    coeffs = DIFF1[check_accuracy(accuracy)]
    return _to_tensor(coeffs, MAX_RADIUS, dtype, device)


def diff2_coeffs(accuracy: int, dtype=None, device=None):
    """Second-derivative coefficients ``[center, offset1..offset4]``."""
    coeffs = DIFF2[check_accuracy(accuracy)]
    return _to_tensor(coeffs, MAX_RADIUS + 1, dtype, device)


def staggered_diff1_coeffs(accuracy: int, dtype=None, device=None):
    """Staggered-grid first-derivative coefficients, padded to ``MAX_RADIUS``."""
    coeffs = STAGGERED_DIFF1[check_accuracy(accuracy)]
    return _to_tensor(coeffs, MAX_RADIUS, dtype, device)


def _to_tensor(values, length, dtype, device):
    import torch

    # Coefficient tables depend only on (values, length, dtype, device) and
    # are read-only for the kernels; cache them instead of re-uploading the
    # same host list on every propagator call.
    key = (tuple(values), length, dtype, device)
    tensor = _COEFF_CACHE.get(key)
    if tensor is None:
        padded = list(values) + [0.0] * (length - len(values))
        tensor = torch.tensor(padded, dtype=dtype, device=device)
        _COEFF_CACHE[key] = tensor
    return tensor


_COEFF_CACHE: dict = {}
