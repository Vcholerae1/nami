"""Native CUDA extensions for nami (architecture A).

Package metadata lives in pyproject.toml. This file only lists the
CUDAExtension modules compiled at ``pip install`` / ``build_ext`` time.

Extension names are top-level (no dots) because PYBIND11_MODULE uses
TORCH_EXTENSION_NAME, which cannot contain dots.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_NVCC_FLAGS = ["-O3"]  # no --use_fast_math: keep reference-comparable numerics

_CSRC = "src/nami/csrc"

_EXTENSIONS = [
    ("nami_scalar2d", "scalar2d.cu"),
    ("nami_scalar3d", "scalar3d.cu"),
    ("nami_scalar3d_born", "scalar3d_born.cu"),
    ("nami_elastic2d", "elastic2d.cu"),
    ("nami_em2d_tm", "em2d_tm.cu"),
    ("nami_em3d", "em3d.cu"),
    ("nami_em3d_born", "em3d_born.cu"),
    ("nami_born", "born.cu"),
    ("nami_born_em_el", "born_em_el.cu"),
]

setup(
    ext_modules=[
        CUDAExtension(
            name,
            [f"{_CSRC}/{src}"],
            extra_compile_args={"nvcc": _NVCC_FLAGS},
        )
        for name, src in _EXTENSIONS
    ],
    cmdclass={"build_ext": BuildExtension},
)
