
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <pybind11/stl.h>
#include "storage.h"

#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------- finite-difference helpers (regular grid) ----------------
// Coefficient arrays come from Python with the regular-grid convention:
//   c1[4] = symmetric first-derivative coefficients for offsets 1..4
//           (accuracy 2 -> [1/2, 0, 0, 0]; 4 -> [8/12, -1/12, 0, 0]; ...)
//   c2[5] = [center, offset1, offset2, offset3, offset4] second-derivative
//           coefficients (accuracy 2 -> [-2, 1, 0, 0, 0]; 4 ->
//           [-5/2, 4/3, -1/12, 0, 0]; ...).
// FD_PAD is a COMPILE-TIME constant (accuracy/2): the loops fully unroll with
// constant offsets (k*nx folds, c2[k] becomes a constant-offset load), which
// measurably beats a runtime fd_pad loop.  Coefficients still come from the
// same c1/c2 tables as the runtime version, so results are bitwise identical.
template <typename T, int FD_PAD>
__device__ __forceinline__ T diff2_y(const T* u, long off, const T* c2, T rdy2, int nx)
{
    T d = c2[0] * u[off];
#pragma unroll
    for (int k = 1; k <= FD_PAD; ++k)
        d += c2[k] * (u[off + k * nx] + u[off - k * nx]);
    return d * rdy2;
}

template <typename T, int FD_PAD>
__device__ __forceinline__ T diff2_x(const T* u, long off, const T* c2, T rdx2)
{
    T d = c2[0] * u[off];
#pragma unroll
    for (int k = 1; k <= FD_PAD; ++k)
        d += c2[k] * (u[off + k] + u[off - k]);
    return d * rdx2;
}

template <typename T, int FD_PAD>
__device__ __forceinline__ T diff1_y(const T* u, long off, const T* c1, T rdy, int nx)
{
    T d = (T)0;
#pragma unroll
    for (int k = 1; k <= FD_PAD; ++k)
        d += c1[k - 1] * (u[off + k * nx] - u[off - k * nx]);
    return d * rdy;
}

template <typename T, int FD_PAD>
__device__ __forceinline__ T diff1_x(const T* u, long off, const T* c1, T rdx)
{
    T d = (T)0;
#pragma unroll
    for (int k = 1; k <= FD_PAD; ++k)
        d += c1[k - 1] * (u[off + k] - u[off - k]);
    return d * rdx;
}

// ---------------- forward: one time step (single kernel: interior + PML frame) ----------------
// The old two-kernel split (interior: pure Laplacian, no PML; frame: CPML
// border) launched the frame kernel over the whole grid with interior threads
// returning early — most of the frame launch was wasted.  Merged into one
// full-grid kernel: interior threads take the fast path (no PML memory
// variables), frame threads the CPML path.  Both branches keep their exact
// original arithmetic so results are bitwise identical.
template <typename T, int FD_PAD>
__global__ void step_forward_kernel(
    const T* __restrict__ v,
    const T* __restrict__ u_cur, const T* __restrict__ u_prev,
    const T* __restrict__ psi_y, const T* __restrict__ psi_x,
    const T* __restrict__ zeta_y, const T* __restrict__ zeta_x,
    T* __restrict__ u_new,
    T* __restrict__ psi_y_new, T* __restrict__ psi_x_new,
    T* __restrict__ zeta_y_new, T* __restrict__ zeta_x_new,
    const T* __restrict__ ay, const T* __restrict__ by, const T* __restrict__ dbydy,
    const T* __restrict__ ax, const T* __restrict__ bx, const T* __restrict__ dbxdx,
    T* __restrict__ w_store,
    const T* __restrict__ c1, const T* __restrict__ c2,
    T rdy, T rdx, T rdy2, T rdx2,
    int t, int interval, T dt2,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int v_batched, int store, int64_t snap_off)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= FD_PAD && x >= FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
        const int s_v = v_batched ? s : 0;
        const T* vs = v + (long)s_v * ny * nx;
        const long off = ((long)s * ny + y) * nx + x;
        const T v_val = vs[(long)y * nx + x];
        const T v2dt2 = v_val * v_val * dt2;
        T w_sum;
        if (y < pml_y0 || y >= pml_y1 || x < pml_x0 || x >= pml_x1) {
            // PML frame (ex-step_forward_frame_kernel body)
            w_sum = (T)0;
            if (y < pml_y0 || y >= pml_y1) {
                const T dwfcdy = diff1_y<T, FD_PAD>(u_cur, off, c1, rdy, nx);
                const T d2y = diff2_y<T, FD_PAD>(u_cur, off, c2, rdy2, nx);
                T dpsi = (T)0;
#pragma unroll
                for (int k = 1; k <= FD_PAD; ++k)
                    dpsi += c1[k - 1] * (ay[y + k] * psi_y[off + k * nx] - ay[y - k] * psi_y[off - k * nx]);
                const T tmpy = ((T)1 + by[y]) * d2y + dbydy[y] * dwfcdy + dpsi * rdy;
                w_sum += ((T)1 + by[y]) * tmpy + ay[y] * zeta_y[off];
                psi_y_new[off] = by[y] * dwfcdy + ay[y] * psi_y[off];
                zeta_y_new[off] = by[y] * tmpy + ay[y] * zeta_y[off];
            } else {
                w_sum += diff2_y<T, FD_PAD>(u_cur, off, c2, rdy2, nx);
            }
            if (x < pml_x0 || x >= pml_x1) {
                const T dwfcdx = diff1_x<T, FD_PAD>(u_cur, off, c1, rdx);
                const T d2x = diff2_x<T, FD_PAD>(u_cur, off, c2, rdx2);
                T dpsi = (T)0;
#pragma unroll
                for (int k = 1; k <= FD_PAD; ++k)
                    dpsi += c1[k - 1] * (ax[x + k] * psi_x[off + k] - ax[x - k] * psi_x[off - k]);
                const T tmpx = ((T)1 + bx[x]) * d2x + dbxdx[x] * dwfcdx + dpsi * rdx;
                w_sum += ((T)1 + bx[x]) * tmpx + ax[x] * zeta_x[off];
                psi_x_new[off] = bx[x] * dwfcdx + ax[x] * psi_x[off];
                zeta_x_new[off] = bx[x] * tmpx + ax[x] * zeta_x[off];
            } else {
                w_sum += diff2_x<T, FD_PAD>(u_cur, off, c2, rdx2);
            }
        } else {
            // interior (ex-step_forward_interior_kernel body)
            w_sum = diff2_y<T, FD_PAD>(u_cur, off, c2, rdy2, nx)
                  + diff2_x<T, FD_PAD>(u_cur, off, c2, rdx2);
        }

        u_new[off] = v2dt2 * w_sum + (T)2 * u_cur[off] - u_prev[off];
        if (store && t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            w_store[soff] = (T)2 * v_val * dt2 * w_sum;
        }
    }
}

// ---------------- forward: source injection / receiver recording ----------------
// (flat indices are precomputed on the padded grid, so the kernels only need
//  the per-shot field stride ny*nx and the row-major flat index.)
template <typename T>
__global__ void inject_kernel(
    T* __restrict__ u_new, const T* __restrict__ f, const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        if (idx >= 0)
            u_new[(long)s * ny_nx + idx] += f[(((long)t * n_shots + s) * n_src + k)];
    }
}

template <typename T>
__global__ void record_kernel(
    const T* __restrict__ u_cur, T* __restrict__ r, const long* __restrict__ rec_i,
    int t, int n_shots, int n_rec, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            r[(((long)t * n_shots + s) * n_rec + k)] = u_cur[(long)s * ny_nx + idx];
    }
}

// ---------------- backward: adjoint step (single kernel: interior + PML frame) ----------------
// Exact discrete transpose of the forward kernel: with
//   T1(dy) = dbydy[y+dy]*((1+by[y+dy])*V2DT2_WFC(dy) + by[y+dy]*ZETAY(dy))
//            + by[y+dy]*PSIY(dy)
//   T2(dy) = (1+by[y+dy])*((1+by[y+dy])*V2DT2_WFC(dy) + by[y+dy]*ZETAY(dy))
// the y PML contribution is -DIFFY1(T1) + DIFFY2(T2), and
//   PSIY_TERM(dy) = (1+by[y+dy])*V2DT2_WFC(dy) + by[y+dy]*ZETAY(dy)
//   psiy_new = -ay[y]*DIFFY1(PSIY_TERM) + ay[y]*psiy
//   zetay_new = ay[y]*V2DT2_WFC(0) + ay[y]*zetay
// The interior fast path (no PML memory variables) keeps its own exact
// arithmetic (in particular the `2*lam + wy + wx - lam2` association), so
// results are bitwise identical to the old two-kernel split.
template <typename T, int FD_PAD>
__global__ void step_adjoint_kernel(
    const T* __restrict__ v,
    const T* __restrict__ lam_next, const T* __restrict__ lam_next2,
    T* __restrict__ lam_new,
    const T* __restrict__ psi_y, const T* __restrict__ psi_x,
    const T* __restrict__ zeta_y, const T* __restrict__ zeta_x,
    T* __restrict__ psi_y_new, T* __restrict__ psi_x_new,
    T* __restrict__ zeta_y_new, T* __restrict__ zeta_x_new,
    const T* __restrict__ ay, const T* __restrict__ by, const T* __restrict__ dbydy,
    const T* __restrict__ ax, const T* __restrict__ bx, const T* __restrict__ dbxdx,
    const T* __restrict__ w_store, T* __restrict__ grad_v,
    const T* __restrict__ c1, const T* __restrict__ c2,
    T rdy, T rdx, T rdy2, T rdx2,
    int t, int interval, T scale, T dt2,
    int n_shots, int ny, int nx,
    // backward (widened) PML boundaries, not the forward ones
    int pml_y0_b, int pml_y1_b, int pml_x0_b, int pml_x1_b,
    int v_batched, int64_t snap_off)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= FD_PAD && x >= FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
        const int s_v = v_batched ? s : 0;
        const T* vs = v + (long)s_v * ny * nx;
        const long off = ((long)s * ny + y) * nx + x;
        const T v_val = vs[(long)y * nx + x];
        const T v2dt2 = v_val * v_val * dt2;
        if (y < pml_y0_b || y >= pml_y1_b || x < pml_x0_b || x >= pml_x1_b) {
            // PML frame (ex-step_adjoint_frame_kernel body)
            T w_sum = (T)0;
            // y: transpose of the CPML-modified Laplacian acting on lam_next
            if (y < pml_y0_b || y >= pml_y1_b) {
                T t1_sum = (T)0;
                T t2_sum = c2[0] * (((T)1 + by[y]) * (((T)1 + by[y]) * v2dt2 * lam_next[off] + by[y] * zeta_y[off]));
                T p_sum = (T)0;
#pragma unroll
                for (int k = 1; k <= FD_PAD; ++k) {
                    const T t1_p = dbydy[y + k] * (((T)1 + by[y + k]) * vs[(long)(y + k) * nx + x]
                            * vs[(long)(y + k) * nx + x] * dt2 * lam_next[off + k * nx] + by[y + k] * zeta_y[off + k * nx])
                        + by[y + k] * psi_y[off + k * nx];
                    const T t1_m = dbydy[y - k] * (((T)1 + by[y - k]) * vs[(long)(y - k) * nx + x]
                            * vs[(long)(y - k) * nx + x] * dt2 * lam_next[off - k * nx] + by[y - k] * zeta_y[off - k * nx])
                        + by[y - k] * psi_y[off - k * nx];
                    t1_sum += c1[k - 1] * (t1_p - t1_m);
                    const T t2_p = ((T)1 + by[y + k]) * (((T)1 + by[y + k]) * vs[(long)(y + k) * nx + x]
                            * vs[(long)(y + k) * nx + x] * dt2 * lam_next[off + k * nx] + by[y + k] * zeta_y[off + k * nx]);
                    const T t2_m = ((T)1 + by[y - k]) * (((T)1 + by[y - k]) * vs[(long)(y - k) * nx + x]
                            * vs[(long)(y - k) * nx + x] * dt2 * lam_next[off - k * nx] + by[y - k] * zeta_y[off - k * nx]);
                    t2_sum += c2[k] * (t2_p + t2_m);
                    const T p_p = ((T)1 + by[y + k]) * vs[(long)(y + k) * nx + x] * vs[(long)(y + k) * nx + x]
                            * dt2 * lam_next[off + k * nx] + by[y + k] * zeta_y[off + k * nx];
                    const T p_m = ((T)1 + by[y - k]) * vs[(long)(y - k) * nx + x] * vs[(long)(y - k) * nx + x]
                            * dt2 * lam_next[off - k * nx] + by[y - k] * zeta_y[off - k * nx];
                    p_sum += c1[k - 1] * (p_p - p_m);
                }
                w_sum += -t1_sum * rdy + t2_sum * rdy2;
                psi_y_new[off] = -ay[y] * p_sum * rdy + ay[y] * psi_y[off];
                zeta_y_new[off] = ay[y] * v2dt2 * lam_next[off] + ay[y] * zeta_y[off];
            } else {
                T wy = c2[0] * (v2dt2 * lam_next[off]);
#pragma unroll
                for (int k = 1; k <= FD_PAD; ++k)
                    wy += c2[k] * (vs[(long)(y + k) * nx + x] * vs[(long)(y + k) * nx + x] * dt2 * lam_next[off + k * nx]
                                 + vs[(long)(y - k) * nx + x] * vs[(long)(y - k) * nx + x] * dt2 * lam_next[off - k * nx]);
                w_sum += wy * rdy2;
            }
            // x: same structure
            if (x < pml_x0_b || x >= pml_x1_b) {
                T t1_sum = (T)0;
                T t2_sum = c2[0] * (((T)1 + bx[x]) * (((T)1 + bx[x]) * v2dt2 * lam_next[off] + bx[x] * zeta_x[off]));
                T p_sum = (T)0;
#pragma unroll
                for (int k = 1; k <= FD_PAD; ++k) {
                    const T t1_p = dbxdx[x + k] * (((T)1 + bx[x + k]) * vs[(long)y * nx + x + k]
                            * vs[(long)y * nx + x + k] * dt2 * lam_next[off + k] + bx[x + k] * zeta_x[off + k])
                        + bx[x + k] * psi_x[off + k];
                    const T t1_m = dbxdx[x - k] * (((T)1 + bx[x - k]) * vs[(long)y * nx + x - k]
                            * vs[(long)y * nx + x - k] * dt2 * lam_next[off - k] + bx[x - k] * zeta_x[off - k])
                        + bx[x - k] * psi_x[off - k];
                    t1_sum += c1[k - 1] * (t1_p - t1_m);
                    const T t2_p = ((T)1 + bx[x + k]) * (((T)1 + bx[x + k]) * vs[(long)y * nx + x + k]
                            * vs[(long)y * nx + x + k] * dt2 * lam_next[off + k] + bx[x + k] * zeta_x[off + k]);
                    const T t2_m = ((T)1 + bx[x - k]) * (((T)1 + bx[x - k]) * vs[(long)y * nx + x - k]
                            * vs[(long)y * nx + x - k] * dt2 * lam_next[off - k] + bx[x - k] * zeta_x[off - k]);
                    t2_sum += c2[k] * (t2_p + t2_m);
                    const T p_p = ((T)1 + bx[x + k]) * vs[(long)y * nx + x + k] * vs[(long)y * nx + x + k]
                            * dt2 * lam_next[off + k] + bx[x + k] * zeta_x[off + k];
                    const T p_m = ((T)1 + bx[x - k]) * vs[(long)y * nx + x - k] * vs[(long)y * nx + x - k]
                            * dt2 * lam_next[off - k] + bx[x - k] * zeta_x[off - k];
                    p_sum += c1[k - 1] * (p_p - p_m);
                }
                w_sum += -t1_sum * rdx + t2_sum * rdx2;
                psi_x_new[off] = -ax[x] * p_sum * rdx + ax[x] * psi_x[off];
                zeta_x_new[off] = ax[x] * v2dt2 * lam_next[off] + ax[x] * zeta_x[off];
            } else {
                T wx = c2[0] * (v2dt2 * lam_next[off]);
#pragma unroll
                for (int k = 1; k <= FD_PAD; ++k)
                    wx += c2[k] * (vs[(long)y * nx + x + k] * vs[(long)y * nx + x + k] * dt2 * lam_next[off + k]
                                 + vs[(long)y * nx + x - k] * vs[(long)y * nx + x - k] * dt2 * lam_next[off - k]);
                w_sum += wx * rdx2;
            }
            lam_new[off] = (T)2 * lam_next[off] + w_sum - lam_next2[off];
        } else {
            // interior (ex-step_adjoint_interior_kernel body)
            T wy = c2[0] * (v2dt2 * lam_next[off]);
#pragma unroll
            for (int k = 1; k <= FD_PAD; ++k)
                wy += c2[k] * (vs[(long)(y + k) * nx + x] * vs[(long)(y + k) * nx + x] * dt2 * lam_next[off + k * nx]
                             + vs[(long)(y - k) * nx + x] * vs[(long)(y - k) * nx + x] * dt2 * lam_next[off - k * nx]);
            wy *= rdy2;
            T wx = c2[0] * (v2dt2 * lam_next[off]);
#pragma unroll
            for (int k = 1; k <= FD_PAD; ++k)
                wx += c2[k] * (vs[(long)y * nx + x + k] * vs[(long)y * nx + x + k] * dt2 * lam_next[off + k]
                             + vs[(long)y * nx + x - k] * vs[(long)y * nx + x - k] * dt2 * lam_next[off - k]);
            wx *= rdx2;
            lam_new[off] = (T)2 * lam_next[off] + wy + wx - lam_next2[off];
        }
        if (t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            grad_v[off] += lam_next[off] * w_store[soff] * scale;
        }
    }
}

template <typename T>
__global__ void record_grad_f_kernel(
    const T* __restrict__ lam_next, T* __restrict__ grad_f, const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        grad_f[(((long)t * n_shots + s) * n_src + k)] = (idx >= 0) ? lam_next[(long)s * ny_nx + idx] : (T)0;
    }
}

template <typename T>
__global__ void record_grad_r_kernel(
    T* __restrict__ lam_cur, const T* __restrict__ grad_r, const long* __restrict__ rec_i,
    int t, int n_shots, int n_rec, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            lam_cur[(long)s * ny_nx + idx] += grad_r[(((long)t * n_shots + s) * n_rec + k)];
    }
}

// ---------------- launchers ----------------
#define LAUNCH_STEP(KERN, T, FP, ...)                                         \
    {                                                                          \
        dim3 block(16, 16);                                                    \
        dim3 grid((nx + 15) / 16, (ny + 15) / 16, n_shots);                    \
        KERN<T, FP><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami scalar2d kernel failed"); \
    }

// Dispatch the compile-time-FD_PAD kernel variants for a runtime fd_pad
// (accuracy/2 = 1..4).  The kernel templates unroll the stencil loops with
// constant offsets; each accuracy gets its own compiled instance.
#define LAUNCH_FORWARD(T, ...)                                                   \
    {                                                                            \
        switch ((int)fd_pad) {                                                   \
            case 1: LAUNCH_STEP(step_forward_kernel, T, 1, __VA_ARGS__); break; \
            case 2: LAUNCH_STEP(step_forward_kernel, T, 2, __VA_ARGS__); break; \
            case 3: LAUNCH_STEP(step_forward_kernel, T, 3, __VA_ARGS__); break; \
            case 4: LAUNCH_STEP(step_forward_kernel, T, 4, __VA_ARGS__); break; \
            default: TORCH_CHECK(false, "nami scalar2d: unsupported fd_pad");   \
        }                                                                        \
    }

#define LAUNCH_ADJOINT(T, ...)                                                   \
    {                                                                            \
        switch ((int)fd_pad) {                                                   \
            case 1: LAUNCH_STEP(step_adjoint_kernel, T, 1, __VA_ARGS__); break; \
            case 2: LAUNCH_STEP(step_adjoint_kernel, T, 2, __VA_ARGS__); break; \
            case 3: LAUNCH_STEP(step_adjoint_kernel, T, 3, __VA_ARGS__); break; \
            case 4: LAUNCH_STEP(step_adjoint_kernel, T, 4, __VA_ARGS__); break; \
            default: TORCH_CHECK(false, "nami scalar2d: unsupported fd_pad");   \
        }                                                                        \
    }

#define LAUNCH_STEP_INTERIOR(KERN, T, Y0, Y1, X0, X1, ...)                     \
    {                                                                          \
        dim3 block(16, 16);                                                    \
        dim3 grid(((X1) - (X0) + 15) / 16, ((Y1) - (Y0) + 15) / 16, n_shots); \
        if (((Y1) - (Y0)) > 0 && ((X1) - (X0)) > 0)                           \
            KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami scalar2d interior kernel failed"); \
    }

void forward_step(
    torch::Tensor v, torch::Tensor u_cur, torch::Tensor u_prev,
    torch::Tensor psi_y, torch::Tensor psi_x, torch::Tensor zeta_y, torch::Tensor zeta_x,
    torch::Tensor u_new, torch::Tensor psi_y_new, torch::Tensor psi_x_new,
    torch::Tensor zeta_y_new, torch::Tensor zeta_x_new,
    torch::Tensor ay, torch::Tensor by, torch::Tensor dbydy,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor dbxdx,
    torch::Tensor w_store,
    torch::Tensor c1, torch::Tensor c2,
    double rdy, double rdx, double rdy2, double rdx2,
    int64_t t, int64_t interval, double dt2,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t v_batched, int64_t store, int64_t snap_off, int64_t fd_pad)
{
    // n_shots from the wavefield (survey), not v: a shared model is [1, ny, nx]
    // while wavefields are [n_shots, ny, nx] (v_batched selects model slab 0).
    const int n_shots = u_cur.size(0), ny = u_cur.size(1), nx = u_cur.size(2);
    CHECK_CONTIG(c1);
    CHECK_CONTIG(c2);
    if (v.scalar_type() == torch::kFloat32) {
        LAUNCH_FORWARD(float,
            v.data_ptr<float>(), u_cur.data_ptr<float>(), u_prev.data_ptr<float>(),
            psi_y.data_ptr<float>(), psi_x.data_ptr<float>(), zeta_y.data_ptr<float>(), zeta_x.data_ptr<float>(),
            u_new.data_ptr<float>(), psi_y_new.data_ptr<float>(), psi_x_new.data_ptr<float>(),
            zeta_y_new.data_ptr<float>(), zeta_x_new.data_ptr<float>(),
            ay.data_ptr<float>(), by.data_ptr<float>(), dbydy.data_ptr<float>(),
            ax.data_ptr<float>(), bx.data_ptr<float>(), dbxdx.data_ptr<float>(),
            w_store.data_ptr<float>(),
            c1.data_ptr<float>(), c2.data_ptr<float>(),
            (float)rdy, (float)rdx, (float)rdy2, (float)rdx2,
            (int)t, (int)interval, (float)dt2,
            n_shots, ny, nx, (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)v_batched, (int)store, (int64_t)snap_off);
    } else {
        LAUNCH_FORWARD(double,
            v.data_ptr<double>(), u_cur.data_ptr<double>(), u_prev.data_ptr<double>(),
            psi_y.data_ptr<double>(), psi_x.data_ptr<double>(), zeta_y.data_ptr<double>(), zeta_x.data_ptr<double>(),
            u_new.data_ptr<double>(), psi_y_new.data_ptr<double>(), psi_x_new.data_ptr<double>(),
            zeta_y_new.data_ptr<double>(), zeta_x_new.data_ptr<double>(),
            ay.data_ptr<double>(), by.data_ptr<double>(), dbydy.data_ptr<double>(),
            ax.data_ptr<double>(), bx.data_ptr<double>(), dbxdx.data_ptr<double>(),
            w_store.data_ptr<double>(),
            c1.data_ptr<double>(), c2.data_ptr<double>(),
            rdy, rdx, rdy2, rdx2,
            (int)t, (int)interval, dt2,
            n_shots, ny, nx, (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)v_batched, (int)store, (int64_t)snap_off);
    }
}

void adjoint_step(
    torch::Tensor v,
    torch::Tensor lam_next, torch::Tensor lam_next2, torch::Tensor lam_new,
    torch::Tensor psi_y, torch::Tensor psi_x, torch::Tensor zeta_y, torch::Tensor zeta_x,
    torch::Tensor psi_y_new, torch::Tensor psi_x_new,
    torch::Tensor zeta_y_new, torch::Tensor zeta_x_new,
    torch::Tensor ay, torch::Tensor by, torch::Tensor dbydy,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor dbxdx,
    torch::Tensor w_store, torch::Tensor grad_v,
    torch::Tensor c1, torch::Tensor c2,
    double rdy, double rdx, double rdy2, double rdx2,
    int64_t t, int64_t interval, double scale, double dt2,
    // forward PML boundaries (fd_pad + pml_width); unused by the adjoint
    // launches, which run on the widened backward boundaries below — kept
    // only to preserve the positional calling convention of the Python side
    [[maybe_unused]] int64_t pml_y0, [[maybe_unused]] int64_t pml_y1,
    [[maybe_unused]] int64_t pml_x0, [[maybe_unused]] int64_t pml_x1,
    // backward PML boundaries (forward + fd_pad): the adjoint widens
    // the PML region by one fd_pad because the transpose of the
    // forward PML stencil reads one cell further into the interior.
    int64_t pml_y0_b, int64_t pml_y1_b, int64_t pml_x0_b, int64_t pml_x1_b,
    int64_t v_batched, int64_t snap_off, int64_t fd_pad)
{
    // n_shots from the adjoint wavefield (survey), not v (see forward_step).
    const int n_shots = lam_next.size(0), ny = lam_next.size(1), nx = lam_next.size(2);
    CHECK_CONTIG(c1);
    CHECK_CONTIG(c2);
    if (v.scalar_type() == torch::kFloat32) {
        LAUNCH_ADJOINT(float,
            v.data_ptr<float>(), lam_next.data_ptr<float>(), lam_next2.data_ptr<float>(), lam_new.data_ptr<float>(),
            psi_y.data_ptr<float>(), psi_x.data_ptr<float>(), zeta_y.data_ptr<float>(), zeta_x.data_ptr<float>(),
            psi_y_new.data_ptr<float>(), psi_x_new.data_ptr<float>(),
            zeta_y_new.data_ptr<float>(), zeta_x_new.data_ptr<float>(),
            ay.data_ptr<float>(), by.data_ptr<float>(), dbydy.data_ptr<float>(),
            ax.data_ptr<float>(), bx.data_ptr<float>(), dbxdx.data_ptr<float>(),
            w_store.data_ptr<float>(), grad_v.data_ptr<float>(),
            c1.data_ptr<float>(), c2.data_ptr<float>(),
            (float)rdy, (float)rdx, (float)rdy2, (float)rdx2,
            (int)t, (int)interval, (float)scale, (float)dt2,
            n_shots, ny, nx, (int)pml_y0_b, (int)pml_y1_b, (int)pml_x0_b, (int)pml_x1_b,
            (int)v_batched, (int64_t)snap_off);
    } else {
        LAUNCH_ADJOINT(double,
            v.data_ptr<double>(), lam_next.data_ptr<double>(), lam_next2.data_ptr<double>(), lam_new.data_ptr<double>(),
            psi_y.data_ptr<double>(), psi_x.data_ptr<double>(), zeta_y.data_ptr<double>(), zeta_x.data_ptr<double>(),
            psi_y_new.data_ptr<double>(), psi_x_new.data_ptr<double>(),
            zeta_y_new.data_ptr<double>(), zeta_x_new.data_ptr<double>(),
            ay.data_ptr<double>(), by.data_ptr<double>(), dbydy.data_ptr<double>(),
            ax.data_ptr<double>(), bx.data_ptr<double>(), dbxdx.data_ptr<double>(),
            w_store.data_ptr<double>(), grad_v.data_ptr<double>(),
            c1.data_ptr<double>(), c2.data_ptr<double>(),
            rdy, rdx, rdy2, rdx2,
            (int)t, (int)interval, scale, dt2,
            n_shots, ny, nx, (int)pml_y0_b, (int)pml_y1_b, (int)pml_x0_b, (int)pml_x1_b,
            (int)v_batched, (int64_t)snap_off);
    }
}


void inject(torch::Tensor u_new, torch::Tensor f, torch::Tensor src_i,
            int64_t t, int64_t n_shots, int64_t n_src, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_src + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(u_new.scalar_type(), "inject", [&] {
        inject_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            u_new.data_ptr<scalar_t>(), f.data_ptr<scalar_t>(), src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami inject failed");
}

void record(torch::Tensor u_cur, torch::Tensor r, torch::Tensor rec_i,
            int64_t t, int64_t n_shots, int64_t n_rec, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_rec + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(u_cur.scalar_type(), "record", [&] {
        record_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            u_cur.data_ptr<scalar_t>(), r.data_ptr<scalar_t>(), rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami record failed");
}

void record_grad_f(torch::Tensor lam_next, torch::Tensor grad_f, torch::Tensor src_i,
                   int64_t t, int64_t n_shots, int64_t n_src, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_src + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(lam_next.scalar_type(), "record_grad_f", [&] {
        record_grad_f_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            lam_next.data_ptr<scalar_t>(), grad_f.data_ptr<scalar_t>(), src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami record_grad_f failed");
}

void record_grad_r(torch::Tensor lam_cur, torch::Tensor grad_r, torch::Tensor rec_i,
                   int64_t t, int64_t n_shots, int64_t n_rec, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_rec + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(lam_cur.scalar_type(), "record_grad_r", [&] {
        record_grad_r_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            lam_cur.data_ptr<scalar_t>(), grad_r.data_ptr<scalar_t>(), rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami record_grad_r failed");
}

// ---------------- whole-loop drivers ----------------
// One pybind call runs the entire forward/adjoint pass (deepwave-style):
// the per-step functions above are reused unchanged with the same argument
// order the Python loop used, so results are bitwise identical.

using nami_storage::ckpt_restore;
using nami_storage::ckpt_save;
using nami_storage::zero_buffers;

void forward_loop(
    torch::Tensor v,
    std::vector<torch::Tensor> u,        // ring of 3
    std::vector<torch::Tensor> psi_y, std::vector<torch::Tensor> psi_x,      // rings of 2
    std::vector<torch::Tensor> zeta_y, std::vector<torch::Tensor> zeta_x,    // rings of 2
    torch::Tensor ay, torch::Tensor by, torch::Tensor dbydy,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor dbxdx,
    torch::Tensor w_store,
    torch::Tensor f, torch::Tensor src_i, torch::Tensor r, torch::Tensor rec_i,
    torch::Tensor c1, torch::Tensor c2,
    double rdy, double rdx, double rdy2, double rdx2, double dt2,
    int64_t nt, int64_t interval,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t v_batched, int64_t store, int64_t fd_pad,
    int64_t checkpoint_every, c10::optional<torch::Tensor> ckpt_state,
    // Wavefield I/O (deepwave-style): init_state/final_state use the N_STATE
    // layout [u_cur, u_prev, psi_y, psi_x, zeta_y, zeta_x] (same as ckpt).
    c10::optional<torch::Tensor> init_state,
    c10::optional<torch::Tensor> final_state,
    // Per-step forward callback: called every `callback_frequency` steps as
    // callback(t, nt, u[t%3], u[(t-1)%3]) — live padded physics fields;
    // PML memory variables remain internal.
    py::object callback, int64_t callback_frequency)
{
    TORCH_CHECK(u.size() == 3 && psi_y.size() == 2 && psi_x.size() == 2 &&
                zeta_y.size() == 2 && zeta_x.size() == 2,
                "nami scalar2d forward_loop: bad ring sizes");
    const int64_t n_shots = u[0].size(0);
    const int64_t ny = u[0].size(1), nx = u[0].size(2);
    const int64_t n_src = src_i.size(1), n_rec = rec_i.size(1);
    const int64_t ny_nx = ny * nx;
    const int64_t shot_count = n_shots * ny_nx;
    const bool ckpt = ckpt_state.has_value() && checkpoint_every > 0;
    const bool has_cb = !callback.is_none();

    // Optional initial state (continuation runs): u[0] = u(t=0), u[2] =
    // u(t=-1) (ring slot for (t-1)%3 at t=0), psi/zeta slot 0.
    if (init_state.has_value()) {
        auto c = *init_state;
        u[0].copy_(c[0]);
        u[2].copy_(c[1]);
        psi_y[0].copy_(c[2]);
        psi_x[0].copy_(c[3]);
        zeta_y[0].copy_(c[4]);
        zeta_x[0].copy_(c[5]);
    }

    // NB: ring slots use (t + 2) % 3 for t-1 — C++ % keeps the sign of the
    // dividend, unlike Python's modulo.
    for (int64_t t = 0; t < nt; ++t) {
        if (ckpt && t > 0 && t % checkpoint_every == 0) {
            ckpt_save(*ckpt_state, t / checkpoint_every - 1,
                      {&u[t % 3], &u[(t + 2) % 3], &psi_y[t % 2], &psi_x[t % 2],
                       &zeta_y[t % 2], &zeta_x[t % 2]});
        }
        if (has_cb && t % callback_frequency == 0) {
            py::gil_scoped_acquire gil;
            callback(t, nt, u[t % 3], u[(t + 2) % 3]);
        }
        const int64_t snap_off = store ? (t / interval) * shot_count : 0;
        forward_step(
            v, u[t % 3], u[(t + 2) % 3],
            psi_y[t % 2], psi_x[t % 2], zeta_y[t % 2], zeta_x[t % 2],
            u[(t + 1) % 3], psi_y[(t + 1) % 2], psi_x[(t + 1) % 2],
            zeta_y[(t + 1) % 2], zeta_x[(t + 1) % 2],
            ay, by, dbydy, ax, bx, dbxdx, w_store, c1, c2,
            rdy, rdx, rdy2, rdx2, t, interval, dt2,
            pml_y0, pml_y1, pml_x0, pml_x1, v_batched, store, snap_off, fd_pad);
        if (n_src > 0)
            inject(u[(t + 1) % 3], f, src_i, t, n_shots, n_src, ny_nx);
        if (n_rec > 0)
            record(u[t % 3], r, rec_i, t, n_shots, n_rec, ny_nx);
    }

    // Optional final state (next-step fields for continuation): u[nt%3] is the
    // field just computed by the last forward step (t = nt), u[(nt-1)%3] the
    // previous one (t = nt-1); psi/zeta at the latest slot nt%2.  This makes a
    // split run (continuation via init_state) bitwise match a one-shot run,
    // because receivers record the pre-update field u[t%3].
    if (final_state.has_value()) {
        auto c = *final_state;
        c[0].copy_(u[nt % 3]);
        c[1].copy_(u[(nt - 1) % 3]);
        c[2].copy_(psi_y[nt % 2]);
        c[3].copy_(psi_x[nt % 2]);
        c[4].copy_(zeta_y[nt % 2]);
        c[5].copy_(zeta_x[nt % 2]);
    }
}

void adjoint_loop(
    torch::Tensor v,
    std::vector<torch::Tensor> lam,      // ring of 3
    std::vector<torch::Tensor> psi_y, std::vector<torch::Tensor> psi_x,
    std::vector<torch::Tensor> zeta_y, std::vector<torch::Tensor> zeta_x,
    torch::Tensor ay, torch::Tensor by, torch::Tensor dbydy,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor dbxdx,
    torch::Tensor w_store, torch::Tensor grad_v,
    torch::Tensor grad_r, torch::Tensor rec_i,
    torch::Tensor grad_f, torch::Tensor src_i,
    torch::Tensor f,
    std::vector<torch::Tensor> u,        // replay rings (empty = full storage)
    std::vector<torch::Tensor> psi_y_f, std::vector<torch::Tensor> psi_x_f,
    std::vector<torch::Tensor> zeta_y_f, std::vector<torch::Tensor> zeta_x_f,
    torch::Tensor c1, torch::Tensor c2,
    double rdy, double rdx, double rdy2, double rdx2, double scale, double dt2,
    int64_t nt, int64_t interval,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t pml_y0_b, int64_t pml_y1_b, int64_t pml_x0_b, int64_t pml_x1_b,
    int64_t v_batched, int64_t fd_pad,
    torch::Tensor segments,              // int64 [n_seg, 2] on CPU; empty = full storage
    c10::optional<torch::Tensor> ckpt_state)
{
    TORCH_CHECK(lam.size() == 3 && psi_y.size() == 2 && psi_x.size() == 2 &&
                zeta_y.size() == 2 && zeta_x.size() == 2,
                "nami scalar2d adjoint_loop: bad ring sizes");
    const int64_t n_shots = lam[0].size(0);
    const int64_t ny = lam[0].size(1), nx = lam[0].size(2);
    const int64_t n_src = src_i.size(1), n_rec = rec_i.size(1);
    const int64_t ny_nx = ny * nx;
    const int64_t shot_count = n_shots * ny_nx;

    auto adjoint_at = [&](int64_t t, int64_t snap_off) {
        if (n_src > 0)
            record_grad_f(lam[(t + 1) % 3], grad_f, src_i, t, n_shots, n_src, ny_nx);
        adjoint_step(
            v, lam[(t + 1) % 3], lam[(t + 2) % 3], lam[t % 3],
            psi_y[t % 2], psi_x[t % 2], zeta_y[t % 2], zeta_x[t % 2],
            psi_y[(t + 1) % 2], psi_x[(t + 1) % 2],
            zeta_y[(t + 1) % 2], zeta_x[(t + 1) % 2],
            ay, by, dbydy, ax, bx, dbxdx, w_store, grad_v, c1, c2,
            rdy, rdx, rdy2, rdx2, t, interval, scale, dt2,
            pml_y0, pml_y1, pml_x0, pml_x1,
            pml_y0_b, pml_y1_b, pml_x0_b, pml_x1_b,
            v_batched, snap_off, fd_pad);
        if (n_rec > 0)
            record_grad_r(lam[t % 3], grad_r, rec_i, t, n_shots, n_rec, ny_nx);
    };

    if (segments.numel() == 0) {
        for (int64_t t = nt - 1; t >= 0; --t)
            adjoint_at(t, (t / interval) * shot_count);
        return;
    }

    // Checkpointed backward: per segment, restore the forward state, replay
    // the forward steps to regenerate the w snapshots, then run the adjoint.
    TORCH_CHECK(u.size() == 3 && psi_y_f.size() == 2 && psi_x_f.size() == 2 &&
                zeta_y_f.size() == 2 && zeta_x_f.size() == 2,
                "nami scalar2d adjoint_loop: bad replay ring sizes");
    auto seg = segments.accessor<int64_t, 2>();
    const int64_t n_seg = segments.size(0);
    for (int64_t k = n_seg - 1; k >= 0; --k) {
        const int64_t s0 = seg[k][0], s1 = seg[k][1];
        if (s0 > 0) {
            ckpt_restore(*ckpt_state, k - 1,
                         {&u[s0 % 3], &u[(s0 + 2) % 3], &psi_y_f[s0 % 2],
                          &psi_x_f[s0 % 2], &zeta_y_f[s0 % 2], &zeta_x_f[s0 % 2]});
        } else {
            zero_buffers({&u[0], &u[1], &u[2], &psi_y_f[0], &psi_y_f[1],
                          &psi_x_f[0], &psi_x_f[1], &zeta_y_f[0], &zeta_y_f[1],
                          &zeta_x_f[0], &zeta_x_f[1]});
        }
        for (int64_t t = s0; t < s1; ++t) {
            const int64_t snap_off = ((t - s0) / interval) * shot_count;
            forward_step(
                v, u[t % 3], u[(t + 2) % 3],
                psi_y_f[t % 2], psi_x_f[t % 2], zeta_y_f[t % 2], zeta_x_f[t % 2],
                u[(t + 1) % 3], psi_y_f[(t + 1) % 2], psi_x_f[(t + 1) % 2],
                zeta_y_f[(t + 1) % 2], zeta_x_f[(t + 1) % 2],
                ay, by, dbydy, ax, bx, dbxdx, w_store, c1, c2,
                rdy, rdx, rdy2, rdx2, t, interval, dt2,
                pml_y0, pml_y1, pml_x0, pml_x1, v_batched, 1, snap_off, fd_pad);
            if (n_src > 0)
                inject(u[(t + 1) % 3], f, src_i, t, n_shots, n_src, ny_nx);
        }
        for (int64_t t = s1 - 1; t >= s0; --t)
            adjoint_at(t, ((t - s0) / interval) * shot_count);
    }
}

// ---------------- pybind11 bindings ----------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_step", &forward_step);
    m.def("inject", &inject);
    m.def("record", &record);
    m.def("adjoint_step", &adjoint_step);
    m.def("record_grad_f", &record_grad_f);
    m.def("record_grad_r", &record_grad_r);
    m.def("forward_loop", &forward_loop);
    m.def("adjoint_loop", &adjoint_loop);
    NAMI_STORAGE_PYBIND(m);
}
