# AGENTS.md

Instructions for coding agents working on or with this repo. **all install and usage detail lives here**.

## Constraints

- **GNU/Linux + CUDA only** (no macOS/Windows, no CPU-only path for the kernels)
- **Source install only** — no prebuilt wheels; `nvcc` compiles extensions at install time
- **Python ≥ 3.12**, `torch` with CUDA, matching toolkit (`CUDA_HOME`)

## Install

```bash
export CUDA_HOME=/usr/local/cuda   # adjust for the machine
pip install "git+https://github.com/barkure/nami.git"
# or, from a local clone:
# pip install -e .
# python setup.py build_ext --inplace
```

After editing `src/nami/csrc/*.cu`, recompile. Verify:

```bash
python -c "import torch; assert torch.cuda.is_available(); import nami; print(nami.__version__)"
pytest   # needs pytest in the env
```

## Repository layout

```text
src/nami/           # Python package (src layout)
  common/           # survey, PML, CFL, storage, FD coefficients
  csrc/             # CUDA kernels + setup.py CUDAExtension list
  scalar/           # scalar2d, scalar3d, Born
  elastic/          # elastic2d, Born
  em/               # em2d_tm, em3d, Born
  models.py         # nn.Module-style FWI wrappers
  wavelets.py       # ricker
setup.py            # CUDAExtension modules only (metadata in pyproject.toml)
tests/
```

Native modules are **top-level** names (`nami_scalar2d`, `nami_elastic2d`, …) required by `TORCH_EXTENSION_NAME`; the Python package imports them. Do not rename without updating both `.cu` `PYBIND11_MODULE` and `setup.py`.

## What to import

| Need | Import |
|---|---|
| Acoustic 2D/3D | `from nami.scalar.scalar2d import scalar2d` / `scalar3d` |
| Elastic 2D | `from nami.elastic.elastic2d import elastic2d` |
| EM 2D TM / 3D | `from nami.em.em2d_tm import em2d_tm` / `from nami.em.em3d import em3d` |
| Born variants | `*_born` modules next to each propagator |
| Class API | `from nami import Scalar, Scalar3D, Elastic, TM2D, EM3D` |
| Ricker wavelet | `from nami.wavelets import ricker` |

## Minimal usage (acoustic 2D)

```python
import torch
from nami.scalar.scalar2d import scalar2d
from nami.wavelets import ricker

device = "cuda"
ny, nx, nt = 100, 100, 300
dx, dt = 5.0, 1e-3

v = 1500 * torch.ones(ny, nx, device=device)
v[ny // 2 :] = 2000
v.requires_grad_()

amp = ricker(freq=25.0, length=nt, dt=dt, peak_time=0.06).reshape(1, 1, -1).to(device)
srcs = torch.tensor([[[10, 10]]], device=device)
recs = torch.tensor([[[10, 90]]], device=device)

out = scalar2d(
    v, dx, dt,
    source_amplitudes=amp,
    source_locations=srcs,
    receiver_locations=recs,
    accuracy=2,
    pml_width=20,
    pml_freq=25.0,
)
# out: [nt, n_shots, n_rec]
loss = out.square().mean()
loss.backward()  # fills v.grad when storage allows
```

Class-style FWI: construct `Scalar(v, dx, dt, ...)`, then `.forward(...)` / `.backward(loss)`.

## Shared propagator parameters

| Parameter | Default | Meaning |
|---|---|---|
| `storage` | `"auto"` | `"auto"`: keep GPU snapshots when inputs need grad; `"none"`: forward-only (backward errors) |
| `sample_steps` | `1` | Sample snapshots / model gradients every N steps; receiver traces stay exact; `>1` approximates model grads |
| `ckpt_steps` | `None` | `None` auto $\sim\sqrt{n_t}$; `0` full storage; `N` checkpoint every N steps (same grads as full storage at same `sample_steps`) |
| `accuracy` | `2` | Spatial FD order: `2`, `4`, `6`, or `8` |
| `pml_width` | `20` | int or per-side list; `0` on a side disables PML there |
| `pml_freq` | `25.0` | Acoustic/elastic C-PML design frequency (Hz); **not used by EM** |
| `nt` | from sources | Time steps if not implied by `source_amplitudes` |

- **dtype** = input model dtype (`float32` / `float64`)
- **device** = CUDA device of the inputs

## Physics coverage (current)

| Physics | Forward + adjoint | Born |
|---|---|---|
| Scalar acoustic 2D/3D | yes | yes |
| Elastic 2D (pressure src/rec) | yes | yes |
| EM 2D TM / 3D | yes | yes |

CFL: oversize `dt` raises (no automatic substepping). See `nami.common.cfl`.

## Agent do / don't

**Do**

- Keep changes Linux/CUDA-oriented
- After `.cu` edits, rebuild extensions and run relevant `tests/test_*.py`
- Preserve exact-adjoint / checkpoint parity tests when touching storage or kernels
- Prefer `storage="none"` for forward-only runs

**Don't**

- Add CPU-only or non-Linux support without an explicit request
- Assume prebuilt wheels or PyPI publish flow
- Break top-level extension module naming without a coordinated rename

## Tests

```bash
pytest                          # full suite (needs CUDA)
pytest tests/test_scalar2d.py   # single module
```
