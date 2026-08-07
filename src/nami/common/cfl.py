"""CFL time-step guard.

nami deliberately does NOT implement internal
sub-stepping (step_ratio > 1) with time-signal resampling: the user's
``dt`` is used as-is.  When it exceeds the CFL limit the modules raise a
``NotImplementedError`` telling the user to reduce ``dt`` or coarsen the
grid spacing, instead of silently resampling the source/receiver signals.
"""

import math

# default Courant number for seismic (c_max=0.6) and Maxwell FDTD (c_max=1.0).
C_MAX_SEISMIC = 0.6
C_MAX_EM = 1.0


def cfl_max_dt(grid_spacing, max_vel, c_max=0.6, eps=1e-15):
    """Largest dt satisfying the CFL condition."""
    if max_vel == 0:
        return float("inf")
    return (
        c_max
        / math.sqrt(sum(1 / g**2 for g in grid_spacing))
        / (max_vel**2 + eps)
        * max_vel
    )


def check_cfl(grid_spacing, dt, max_vel, name, c_max=0.6):
    """Raise ``NotImplementedError`` if ``dt`` violates the CFL condition."""
    max_dt = cfl_max_dt(grid_spacing, max_vel, c_max)
    if math.ceil(abs(float(dt)) / max_dt) > 1:
        raise NotImplementedError(
            f"nami {name} requires dt <= {max_dt:.3e} to satisfy the CFL "
            f"condition (step_ratio=1, no internal sub-stepping or "
            f"resampling); got dt={dt}. Reduce dt or coarsen the grid "
            f"spacing."
        )
