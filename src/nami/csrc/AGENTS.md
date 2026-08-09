# CUDA extension contracts

These instructions apply to `src/nami/csrc/*.cu` and supplement the
repository-level `AGENTS.md`.

## Time loops and bindings

Every propagator runs its complete forward or adjoint time loop inside the
extension, so the Python wrapper makes one pybind call per pass
(`forward_loop` / `adjoint_loop`, or the corresponding Born EM/elastic loop).
Keep the physics-specific per-step and stage exports (for example,
`forward_step`, `adjoint_step`, `step_h`, `step_e`, and `inject`): the loop
functions call the same operations internally.

Ring and state buffers cross pybind as Python lists backed by
`std::vector<torch::Tensor>`; include `pybind11/stl.h` where needed. Optional
tensors use `c10::optional`, with Python `None` represented as `nullopt`.

## Ring buffers and indexing

- Python `%` is floor modulo, while C++ integer remainder truncates toward
  zero. Never use an expression such as `(t - 1) % 3` when `t` may be zero;
  use a non-negative equivalent such as `(t + 2) % 3`.
- Wavefields use `off = s * n_cells + spatial`. Models use their selected
  slab plus `spatial`, where `spatial = y * nx + x` in 2D and
  `spatial = z * (ny * nx) + y * nx + x` in 3D.
- Python derives model batching before padding. A batched model uses shot
  slab `s`; a shared model uses slab `0`. Wavefields are always per-shot.
- Inject and record operations use the global time index `t`, including
  checkpoint replay.

## Snapshots and checkpoints

- `segments` is an `int64` CPU tensor with shape `[n_segments, 2]`; an empty
  tensor selects full-storage adjoint execution.
- Snapshot indexing is `snap_off + spatial`. For full storage,
  `snap_off = (t / interval) * shot_count`. During replay of `[s0, s1)`,
  `snap_off = ((t - s0) / interval) * shot_count`, where
  `shot_count = n_shots * n_cells`.
- Shared `ckpt_save`, `ckpt_restore`, and `zero_buffers` implementations live
  in `storage.h`; they perform checkpoint copies or resets on
  `at::cuda::getCurrentCUDAStream()`.

Preserve kernel launch order, argument order, ring-buffer state, exact-adjoint
behaviour, checkpoint/full-storage parity, and continuation parity. Rebuild
all affected extensions and run the matching CUDA tests after changes.
