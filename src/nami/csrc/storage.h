// Snapshot storage for the FWI adjoints.
//
// Each module owns one `SnapshotStore` per snapshot stream (scalar2d: 1,
// em2d_tm: 2, elastic2d: 5).  The forward kernels write one snapshot per
// step into a device buffer handed out by `snap_tensor()`; the backward
// kernels read the same buffer.  All buffers are owned by the C++ store;
// the Python layer only holds a handle.
//
// Snapshots always live on the GPU.  Memory is controlled from the Python
// front ends: `checkpoint_every=0` keeps every step's snapshot; with
// `checkpoint_every=N` the full wavefield state is snapshotted every N
// steps and the intermediate snapshots are recomputed during the backward
// pass, so the stream only needs capacity for one segment of N steps.

#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace nami_storage {

class SnapshotStoreBase {
public:
    virtual ~SnapshotStoreBase() = default;
    virtual torch::Tensor snap_tensor() const = 0;
    virtual int64_t snap_offset(int64_t step_idx) const = 0;
};

template <typename T>
class SnapshotStore : public SnapshotStoreBase {
public:
    SnapshotStore(int64_t n_snap, int64_t n_shots, int64_t ny, int64_t nx,
                  torch::Device device)
        : n_snap_(n_snap),
          n_shots_(n_shots),
          ny_(ny),
          nx_(nx),
          shot_count_(n_shots * ny * nx)
    {
        TORCH_CHECK(shot_count_ > 0, "nami storage: empty snapshot grid");
        auto opts = torch::TensorOptions()
                        .dtype(torch::CppTypeToScalarType<T>::value)
                        .device(device);
        snap_ = torch::zeros({n_snap, n_shots, ny, nx}, opts);
    }

    torch::Tensor snap_tensor() const override { return snap_; }

    int64_t snap_offset(int64_t step_idx) const override
    {
        return step_idx * shot_count_;
    }

private:
    int64_t n_snap_, n_shots_, ny_, nx_, shot_count_;
    torch::Tensor snap_;
};

// ---------------- type-erased entry points (shared by all modules) ----------------

inline int64_t storage_create(int64_t n_snap, int64_t n_shots, int64_t ny,
                              int64_t nx, int64_t dtype_index,
                              int64_t device_index)
{
    const torch::Device device(torch::kCUDA, (int)device_index);
    SnapshotStoreBase* store = nullptr;
    if (dtype_index == 0)
        store = new SnapshotStore<float>(n_snap, n_shots, ny, nx, device);
    else
        store = new SnapshotStore<double>(n_snap, n_shots, ny, nx, device);
    return (int64_t)store;
}

inline torch::Tensor storage_snap_tensor(int64_t handle)
{
    auto* store = reinterpret_cast<SnapshotStoreBase*>(handle);
    TORCH_CHECK(store != nullptr, "nami storage: invalid handle");
    return store->snap_tensor();
}

inline int64_t storage_snap_offset(int64_t handle, int64_t step_idx)
{
    auto* store = reinterpret_cast<SnapshotStoreBase*>(handle);
    TORCH_CHECK(store != nullptr, "nami storage: invalid handle");
    return store->snap_offset(step_idx);
}

inline void storage_destroy(int64_t handle)
{
    auto* store = reinterpret_cast<SnapshotStoreBase*>(handle);
    if (store != nullptr)
        delete store;
}

// ---------------- checkpoint state transfers (shared by all modules) --------

inline void ckpt_save(const torch::Tensor& ckpt_state, int64_t k,
                      const std::vector<const torch::Tensor*>& state)
{
    // ckpt_state is contiguous [n_ckpt, N_STATE, n_shots, ...].  Copy one
    // flat device slab per state slot on the caller's current CUDA stream.
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t n_state = ckpt_state.size(1);
    const int64_t slab_bytes = state[0]->numel() * state[0]->element_size();
    char* base = static_cast<char*>(ckpt_state.data_ptr())
                 + k * n_state * slab_bytes;
    for (size_t j = 0; j < state.size(); ++j)
        cudaMemcpyAsync(base + static_cast<int64_t>(j) * slab_bytes,
                        state[j]->data_ptr(), slab_bytes,
                        cudaMemcpyDeviceToDevice, stream);
}

inline void ckpt_restore(const torch::Tensor& ckpt_state, int64_t k,
                         const std::vector<torch::Tensor*>& state)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t n_state = ckpt_state.size(1);
    const int64_t slab_bytes = state[0]->numel() * state[0]->element_size();
    const char* base = static_cast<const char*>(ckpt_state.data_ptr())
                       + k * n_state * slab_bytes;
    for (size_t j = 0; j < state.size(); ++j)
        cudaMemcpyAsync(state[j]->data_ptr(),
                        base + static_cast<int64_t>(j) * slab_bytes,
                        slab_bytes, cudaMemcpyDeviceToDevice, stream);
}

inline void zero_buffers(const std::vector<torch::Tensor*>& bufs)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    for (auto* buffer : bufs)
        cudaMemsetAsync(buffer->data_ptr(), 0,
                        buffer->numel() * buffer->element_size(), stream);
}

#define NAMI_STORAGE_PYBIND(m)                                                 \
    m.def("storage_create", &nami_storage::storage_create);                    \
    m.def("storage_snap_tensor", &nami_storage::storage_snap_tensor);          \
    m.def("storage_snap_offset", &nami_storage::storage_snap_offset);          \
    m.def("storage_destroy", &nami_storage::storage_destroy);

}  // namespace nami_storage
