# AGENTS.md

Instructions for coding agents. Installation, public API, and engineering
contracts for this repository live here.

## Platform

- GNU/Linux with CUDA only (no macOS/Windows; no CPU-only kernels)
- Source install only: `nvcc` builds extensions at install or `build_ext` time
- Python ≥ 3.12, CUDA-enabled PyTorch, matching toolkit via `CUDA_HOME`

## Install

```bash
export CUDA_HOME=/usr/local/cuda   # machine-specific
pip install "git+https://github.com/barkure/nami.git"
# local editable: pip install -e . && python setup.py build_ext --inplace
```

After changing `src/nami/csrc/*.cu`, rebuild and sanity-check:

```bash
python -c "import torch; assert torch.cuda.is_available(); import nami; print(nami.__version__)"
```

Dev tooling and the full test command are under [Tests](#tests).

## Layout

```text
src/nami/
  common/      # survey, PML, CFL, storage, FD coefficients
  csrc/        # CUDA sources; listed in setup.py
  scalar/      # acoustic 2D/3D + Born
  elastic/     # elastic 2D + Born
  em/          # EM 2D TM, EM 3D + Born; _common.py shared helpers
  models.py    # nn.Module FWI wrappers
  wavelets.py  # ricker
setup.py       # CUDAExtension list only (metadata in pyproject.toml)
tests/
```

Native extensions use top-level names (`nami_scalar2d`, `nami_born_em_el`, …)
because `TORCH_EXTENSION_NAME` cannot contain dots. The `nami_` prefix avoids
collisions in the process-wide module table. Public imports go through the
Python package, not the extension names. Renames require coordinated updates
to the corresponding `.cu` `PYBIND11_MODULE`, `setup.py`, and Python `import`s.

Extension map (Born):

| Kernel package | Source | Python callers |
|---|---|---|
| `nami_born` | `born.cu` | `scalar2d_born` |
| `nami_scalar3d_born` | `scalar3d_born.cu` | `scalar3d_born` |
| `nami_born_em_el` | `born_em_el.cu` | `em2d_tm_born`, `elastic2d_born` |
| `nami_em3d_born` | `em3d_born.cu` | `em3d_born` |

## Public imports

| Capability | Import |
|---|---|
| Acoustic 2D / 3D | `from nami.scalar.scalar2d import scalar2d` / `from nami.scalar.scalar3d import scalar3d` |
| Elastic 2D | `from nami.elastic.elastic2d import elastic2d` |
| EM 2D TM / 3D | `from nami.em.em2d_tm import em2d_tm` / `from nami.em.em3d import em3d` |
| Born | e.g. `from nami.em import em3d_born` (each physics subpackage re-exports its `*_born`) |
| Class API | `from nami import Scalar, Scalar3D, Elastic, TM2D, EM3D` |
| Ricker | `from nami.wavelets import ricker` |

## Minimal example (acoustic 2D forward)

```python
import torch
from nami.scalar.scalar2d import scalar2d
from nami.wavelets import ricker

device, ny, nx, nt, dx, dt = "cuda", 100, 100, 300, 5.0, 1e-3
v = 1500 * torch.ones(ny, nx, device=device)
v[ny // 2 :] = 2000
amp = ricker(25.0, nt, dt, 0.06).reshape(1, 1, -1).to(device)
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
    storage="none",
)  # out: [nt, n_shots, n_rec]
```

Class API: `Scalar(v, dx, dt, ...)`, then `.forward(...)`. Wrappers also
accept `storage` / `sample_steps` / `ckpt_steps` (`EM3D` additionally takes
`source_component` / `receiver_component`). For FWI, enable grad on the
model, use `storage="auto"`, and call `.backward(loss)` (writes `param.grad`).

## Propagator parameters

| Parameter | Default | Role |
|---|---|---|
| `storage` | `"auto"` | `"auto"`: retain GPU snapshots when inputs need grad; `"none"`: forward-only (backward raises) |
| `sample_steps` | `1` | Sample snapshots / model grads every N steps (receivers stay exact). For `N > 1`, model grads use the rectangle rule: each sampled imaging term is scaled by `N` (`scale = float(sample_steps)` into every kernel). `N = 1` is bitwise identical on every path |
| `ckpt_steps` | `None` | `None` selects $\sim\sqrt{n_t}$; `0` full snapshot storage; `N > 0` checkpoint every N steps (same grads as full storage at the same `sample_steps`) |
| `accuracy` | `2` | Spatial FD order: 2, 4, 6, or 8 |
| `pml_width` | `20` | Scalar width or per-side list; `0` disables that side |
| `pml_freq` | `25.0` | Acoustic/elastic C-PML design frequency (Hz); unused for EM |
| `nt` | from sources | Time steps when not implied by `source_amplitudes` |

dtype and device follow the input models (CUDA `float32` / `float64`).

## Multi-shot and indexing

Contract for every full-wave and Born propagator (scalar, elastic, EM).

- **`n_shots`** comes from the survey (`source_amplitudes` /
  `source_locations`). Shared models stay `[spatial]` or `[1, spatial]`;
  wavefields are still `n_shots`-wide.
- **`*_batched`** is derived in Python from the *user* model before pad
  (`shape[0] == n_shots > 1`). Kernels use slab `s` when batched, else
  slab `0`. Unbatched model grads are `sum(0)`-reduced after the adjoint.
  Scalar/EM: one flag per tensor (mixed forms allowed). Elastic: one flag
  per group via `check_model_batching` on `lamb`/`mu`/`buoyancy` and on the
  scatter triple (`None` scatter = shared zeros).
- **Indexing:** wavefields use `off = s * n_cells + spatial`; models use
  `slab(s_m) + spatial` with `spatial = y*nx+x` (3D: `z*(ny*nx)+…`).
  `off - off_s` is the same spatial index (handy after a stencil step).
- **Snapshots:** `snap_off + spatial`. Full storage:
  `storage.snap_offset(t // sample_steps)`; checkpoint replay:
  `((t - s0) // sample_steps) * (n_shots * n_cells)`. Inject/record use
  global `t`.
- **Survey replicate pad** is `cat`/`expand` in `common.survey` (values
  match `F.pad(mode="replicate")`, deterministic backward).
- **Outputs:** receiver gathers are `[nt, n_shots, n_rec]`.

## Physics coverage

Full-wave forward + adjoint and Born scattering for scalar acoustic 2D/3D,
elastic 2D (pressure sources/receivers), and EM 2D TM / 3D. Oversized `dt`
raises a CFL error (`nami.common.cfl`).

## Agent guidelines

**Do**

- Keep changes Linux/CUDA-oriented
- Rebuild extensions after `.cu` edits and run the matching `tests/test_*.py`
- Preserve exact-adjoint and checkpoint-parity tests when touching storage or kernels
- Prefer `storage="none"` for forward-only work

**Do not**

- Add CPU-only or non-Linux support unless explicitly requested
- Assume prebuilt wheels or a PyPI release path
- Rename top-level extension modules without a coordinated update of `.cu`,
  `setup.py`, and Python imports

## Tests

```bash
# optional: pip install -e .[dev]  |  pixi install -e dev
pytest                          # full suite (CUDA required)
pytest tests/test_scalar2d.py   # single module
pixi run -e dev pytest
```

The dev extra also carries `ruff` — lint with `ruff check` (config in
`pyproject.toml`).
