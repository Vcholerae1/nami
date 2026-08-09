"""nami (波): unified FDTD forward modelling and FWI.

One engine, three wave families — acoustic (scalar), elastic and
electromagnetic (2D TM / 3D) — with native CUDA adjoints behind a
PyTorch front end.  Compiled-only: no eager/torch-op backend; every time
step and its exact-transpose adjoint are custom CUDA kernels.

Layout (directory = physics, file = dimensionality):

    common/      survey / PML profiles / CFL / storage / FD coefficients
    scalar/      acoustic waves       scalar2d.py, scalar3d.py (+ Born)
    elastic/     elastic waves        elastic2d.py (+ Born)
    em/          electromagnetic      em2d_tm.py, em3d.py (+ Born)
    wavelets.py  Gaussian / Ricker / Ormsby / Klauder / swept sources
    models.py    class-based FWI wrappers
"""

from . import common  # noqa: F401
from .models import EM3D, TM2D, Elastic, Scalar, Scalar3D  # noqa: F401

__version__ = "0.1.0"

__all__ = ["common", "Scalar", "Scalar3D", "Elastic", "TM2D", "EM3D", "__version__"]
