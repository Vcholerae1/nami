"""Acoustic (scalar) waves (regular-grid FDTD + FWI + Born)."""

from .scalar2d import scalar2d
from .scalar2d_born import scalar2d_born
from .scalar3d import scalar3d
from .scalar3d_born import scalar3d_born

__all__ = ["scalar2d", "scalar3d", "scalar2d_born", "scalar3d_born"]
