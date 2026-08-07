// nami scalar3d Born: 3D acoustic Born FDTD with native CUDA adjoint.
//
// 3D scalar Born FDTD backend (bit-for-bit reproducible in float64),
// combining the 3D structure of src/nami/csrc/scalar3d.cu (CPML with three
// split-field memory pairs, exact PML boundaries, per-shot loops inside the
// kernels, SnapshotStorage-backed gradient snapshots) with the Born coupling
// of src/nami/csrc/born.cu (a scattered wavefield driven by
// ``2 * v * scatter * dt^2 * laplacian(u_bg)``).
//
// Forward:
//   u_bg_new  = v^2 dt^2 lap(u_bg)  + 2 u_bg - u_bg_prev
//   u_sc_new  = v^2 dt^2 lap(u_sc)  + 2 u_sc - u_sc_prev
//               + 2 v scatter dt^2 lap(u_bg)
// The Laplacian snapshots (raw, unscaled) of both fields are stored every
// `interval` steps.
//
// Adjoint (exact discrete transpose of the forward operator): the
// background adjoint evolves with the Laplacian of
//   V2DT2_WFC = v^2 dt^2 lam_bg + 2 v dt^2 scatter lam_sc
// and the scattered adjoint with that of v^2 dt^2 lam_sc; the PML
// memory variables are transposed too, and
//   grad_v      += lam_bg*2 v dt^2 w_store
//                  + lam_sc*(2 dt^2 scatter w_store + 2 v dt^2 wsc_store)
//   grad_scatter+= lam_sc*2 v dt^2 w_store
// The backward pass uses wider PML boundaries
// ``pml_*_b = min(pml_width + 3*fd_pad, dim - fd_pad)`` (the transpose of
// the forward PML stencil reads one fd_pad cell further into the interior).
//
// Kernels are intentionally unoptimised (one launch per step, naive stencil)
// — that is the correctness baseline; performance work lands on top later.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "storage.h"

#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------- finite-difference helpers (regular grid) ----------------
// Same coefficient-array-driven stencils as scalar3d.cu: c1[4]
// first-derivative coefficients for offsets 1..4, c2[5] = [center, off1..off4]
// second-derivative coefficients; loops run to the runtime fd_pad.
template <typename T>
__device__ __forceinline__ T diff2_z(const T* u, long off, const T* c2, T rdz2, long ny_nx, int fd_pad)
{
    T d = c2[0] * u[off];
#pragma unroll
    for (int k = 1; k <= fd_pad; ++k)
        d += c2[k] * (u[off + k * ny_nx] + u[off - k * ny_nx]);
    return d * rdz2;
}

template <typename T>
__device__ __forceinline__ T diff2_y(const T* u, long off, const T* c2, T rdy2, long nx, int fd_pad)
{
    T d = c2[0] * u[off];
#pragma unroll
    for (int k = 1; k <= fd_pad; ++k)
        d += c2[k] * (u[off + k * nx] + u[off - k * nx]);
    return d * rdy2;
}

template <typename T>
__device__ __forceinline__ T diff2_x(const T* u, long off, const T* c2, T rdx2, int fd_pad)
{
    T d = c2[0] * u[off];
#pragma unroll
    for (int k = 1; k <= fd_pad; ++k)
        d += c2[k] * (u[off + k] + u[off - k]);
    return d * rdx2;
}

template <typename T>
__device__ __forceinline__ T diff1_z(const T* u, long off, const T* c1, T rdz, long ny_nx, int fd_pad)
{
    T d = (T)0;
#pragma unroll
    for (int k = 1; k <= fd_pad; ++k)
        d += c1[k - 1] * (u[off + k * ny_nx] - u[off - k * ny_nx]);
    return d * rdz;
}

template <typename T>
__device__ __forceinline__ T diff1_y(const T* u, long off, const T* c1, T rdy, long nx, int fd_pad)
{
    T d = (T)0;
#pragma unroll
    for (int k = 1; k <= fd_pad; ++k)
        d += c1[k - 1] * (u[off + k * nx] - u[off - k * nx]);
    return d * rdy;
}

template <typename T>
__device__ __forceinline__ T diff1_x(const T* u, long off, const T* c1, T rdx, int fd_pad)
{
    T d = (T)0;
#pragma unroll
    for (int k = 1; k <= fd_pad; ++k)
        d += c1[k - 1] * (u[off + k] - u[off - k]);
    return d * rdx;
}

// ---------------- forward: one time step (interior: pure Laplacian, no PML) ----------------
// The scattered wavefield is driven by 2*v*scatter*dt^2 * (Laplacian of the
// background wavefield), matching the discrete forward operator.
template <typename T>
__global__ void step_forward_interior_kernel(
    const T* __restrict__ v, const T* __restrict__ scatter,
    const T* __restrict__ u_cur, const T* __restrict__ u_prev,
    const T* __restrict__ u_sc_cur, const T* __restrict__ u_sc_prev,
    T* __restrict__ u_new, T* __restrict__ u_sc_new,
    T* __restrict__ w_store, T* __restrict__ wsc_store,
    const T* __restrict__ c1, const T* __restrict__ c2,
    T rdz, T rdy, T rdx, T rdz2, T rdy2, T rdx2,
    int t, int interval, T dt2,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int v_batched, int scatter_batched, int store, int64_t snap_off, int fd_pad)
{
    const int z = pml_z0 + blockIdx.z;
    const int y = pml_y0 + blockIdx.y * blockDim.y + threadIdx.y;
    const int x = pml_x0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= 1 && y >= 1 && x >= 1 && z < nz - 1 && y < ny - 1 && x < nx - 1 &&
        z < pml_z1 && y < pml_y1 && x < pml_x1) {
        const long ny_nx = (long)ny * nx;
        const long off_base = ((long)z * ny + y) * nx + x;
        for (int s = 0; s < n_shots; ++s) {
            const int s_v = v_batched ? s : 0;
            const int s_sc = scatter_batched ? s : 0;
            const T* vs = v + (long)s_v * nz * ny_nx;
            const T* scs = scatter + (long)s_sc * nz * ny_nx;
            const long off = (long)s * nz * ny_nx + off_base;
            const T v_val = vs[off_base];
            const T v2dt2 = v_val * v_val * dt2;
            T w_sum = diff2_z(u_cur, off, c2, rdz2, ny_nx, fd_pad)
                    + diff2_y(u_cur, off, c2, rdy2, nx, fd_pad)
                    + diff2_x(u_cur, off, c2, rdx2, fd_pad);
            u_new[off] = v2dt2 * w_sum + (T)2 * u_cur[off] - u_prev[off];
            T wsc_sum = diff2_z(u_sc_cur, off, c2, rdz2, ny_nx, fd_pad)
                      + diff2_y(u_sc_cur, off, c2, rdy2, nx, fd_pad)
                      + diff2_x(u_sc_cur, off, c2, rdx2, fd_pad);
            u_sc_new[off] = v2dt2 * wsc_sum + (T)2 * u_sc_cur[off] - u_sc_prev[off]
                          + (T)2 * v_val * scs[off_base] * dt2 * w_sum;
            if (store && t % interval == 0) {
                const long soff = snap_off + off;
                w_store[soff] = w_sum;
                wsc_store[soff] = wsc_sum;
            }
        }
    }
}

// ---------------- forward: one time step (frame: includes PML borders) ----------------
// The per-dimension CPML update, evaluated for every cell outside the
// interior box [pml_z0, pml_z1) x [pml_y0, pml_y1) x [pml_x0, pml_x1).
template <typename T>
__global__ void step_forward_frame_kernel(
    const T* __restrict__ v, const T* __restrict__ scatter,
    const T* __restrict__ u_cur, const T* __restrict__ u_prev,
    const T* __restrict__ u_sc_cur, const T* __restrict__ u_sc_prev,
    const T* __restrict__ psi_z, const T* __restrict__ psi_y, const T* __restrict__ psi_x,
    const T* __restrict__ zeta_z, const T* __restrict__ zeta_y, const T* __restrict__ zeta_x,
    const T* __restrict__ psi_z_sc, const T* __restrict__ psi_y_sc, const T* __restrict__ psi_x_sc,
    const T* __restrict__ zeta_z_sc, const T* __restrict__ zeta_y_sc, const T* __restrict__ zeta_x_sc,
    T* __restrict__ u_new, T* __restrict__ u_sc_new,
    T* __restrict__ psi_z_new, T* __restrict__ psi_y_new, T* __restrict__ psi_x_new,
    T* __restrict__ zeta_z_new, T* __restrict__ zeta_y_new, T* __restrict__ zeta_x_new,
    T* __restrict__ psi_z_sc_new, T* __restrict__ psi_y_sc_new, T* __restrict__ psi_x_sc_new,
    T* __restrict__ zeta_z_sc_new, T* __restrict__ zeta_y_sc_new, T* __restrict__ zeta_x_sc_new,
    const T* __restrict__ az, const T* __restrict__ bz, const T* __restrict__ dbzdz,
    const T* __restrict__ ay, const T* __restrict__ by, const T* __restrict__ dbydy,
    const T* __restrict__ ax, const T* __restrict__ bx, const T* __restrict__ dbxdx,
    T* __restrict__ w_store, T* __restrict__ wsc_store,
    const T* __restrict__ c1, const T* __restrict__ c2,
    T rdz, T rdy, T rdx, T rdz2, T rdy2, T rdx2,
    int t, int interval, T dt2,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int v_batched, int scatter_batched, int store, int64_t snap_off, int fd_pad)
{
    const int z = fd_pad + blockIdx.z;
    const int y = fd_pad + blockIdx.y * blockDim.y + threadIdx.y;
    const int x = fd_pad + blockIdx.x * blockDim.x + threadIdx.x;
    // strictly interior cells are handled by step_forward_interior_kernel
    if (z >= pml_z0 && z < pml_z1 && y >= pml_y0 && y < pml_y1 &&
        x >= pml_x0 && x < pml_x1)
        return;
    if (z < nz - fd_pad && y < ny - fd_pad && x < nx - fd_pad) {
        const long ny_nx = (long)ny * nx;
        const long off_base = ((long)z * ny + y) * nx + x;
        for (int s = 0; s < n_shots; ++s) {
            const int s_v = v_batched ? s : 0;
            const int s_sc = scatter_batched ? s : 0;
            const T* vs = v + (long)s_v * nz * ny_nx;
            const T* scs = scatter + (long)s_sc * nz * ny_nx;
            const long off = (long)s * nz * ny_nx + off_base;
            const T v_val = vs[off_base];
            const T v2dt2 = v_val * v_val * dt2;
            T w_sum = (T)0;
            T wsc_sum = (T)0;

            if (z < pml_z0 || z >= pml_z1) {
                const T dwfcdz = diff1_z(u_cur, off, c1, rdz, ny_nx, fd_pad);
                const T d2z = diff2_z(u_cur, off, c2, rdz2, ny_nx, fd_pad);
                T dpsi = (T)0;
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k)
                    dpsi += c1[k - 1] * (az[z + k] * psi_z[off + k * ny_nx] - az[z - k] * psi_z[off - k * ny_nx]);
                const T tmpz = ((T)1 + bz[z]) * d2z + dbzdz[z] * dwfcdz + dpsi * rdz;
                w_sum += ((T)1 + bz[z]) * tmpz + az[z] * zeta_z[off];
                psi_z_new[off] = bz[z] * dwfcdz + az[z] * psi_z[off];
                zeta_z_new[off] = bz[z] * tmpz + az[z] * zeta_z[off];

                const T dwfcscdz = diff1_z(u_sc_cur, off, c1, rdz, ny_nx, fd_pad);
                const T d2zsc = diff2_z(u_sc_cur, off, c2, rdz2, ny_nx, fd_pad);
                T dpsisc = (T)0;
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k)
                    dpsisc += c1[k - 1] * (az[z + k] * psi_z_sc[off + k * ny_nx] - az[z - k] * psi_z_sc[off - k * ny_nx]);
                const T tmpzsc = ((T)1 + bz[z]) * d2zsc + dbzdz[z] * dwfcscdz + dpsisc * rdz;
                wsc_sum += ((T)1 + bz[z]) * tmpzsc + az[z] * zeta_z_sc[off];
                psi_z_sc_new[off] = bz[z] * dwfcscdz + az[z] * psi_z_sc[off];
                zeta_z_sc_new[off] = bz[z] * tmpzsc + az[z] * zeta_z_sc[off];
            } else {
                w_sum += diff2_z(u_cur, off, c2, rdz2, ny_nx, fd_pad);
                wsc_sum += diff2_z(u_sc_cur, off, c2, rdz2, ny_nx, fd_pad);
            }
            if (y < pml_y0 || y >= pml_y1) {
                const T dwfcdy = diff1_y(u_cur, off, c1, rdy, nx, fd_pad);
                const T d2y = diff2_y(u_cur, off, c2, rdy2, nx, fd_pad);
                T dpsi = (T)0;
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k)
                    dpsi += c1[k - 1] * (ay[y + k] * psi_y[off + k * nx] - ay[y - k] * psi_y[off - k * nx]);
                const T tmpy = ((T)1 + by[y]) * d2y + dbydy[y] * dwfcdy + dpsi * rdy;
                w_sum += ((T)1 + by[y]) * tmpy + ay[y] * zeta_y[off];
                psi_y_new[off] = by[y] * dwfcdy + ay[y] * psi_y[off];
                zeta_y_new[off] = by[y] * tmpy + ay[y] * zeta_y[off];

                const T dwfcscdy = diff1_y(u_sc_cur, off, c1, rdy, nx, fd_pad);
                const T d2ysc = diff2_y(u_sc_cur, off, c2, rdy2, nx, fd_pad);
                T dpsisc = (T)0;
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k)
                    dpsisc += c1[k - 1] * (ay[y + k] * psi_y_sc[off + k * nx] - ay[y - k] * psi_y_sc[off - k * nx]);
                const T tmpysc = ((T)1 + by[y]) * d2ysc + dbydy[y] * dwfcscdy + dpsisc * rdy;
                wsc_sum += ((T)1 + by[y]) * tmpysc + ay[y] * zeta_y_sc[off];
                psi_y_sc_new[off] = by[y] * dwfcscdy + ay[y] * psi_y_sc[off];
                zeta_y_sc_new[off] = by[y] * tmpysc + ay[y] * zeta_y_sc[off];
            } else {
                w_sum += diff2_y(u_cur, off, c2, rdy2, nx, fd_pad);
                wsc_sum += diff2_y(u_sc_cur, off, c2, rdy2, nx, fd_pad);
            }
            if (x < pml_x0 || x >= pml_x1) {
                const T dwfcdx = diff1_x(u_cur, off, c1, rdx, fd_pad);
                const T d2x = diff2_x(u_cur, off, c2, rdx2, fd_pad);
                T dpsi = (T)0;
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k)
                    dpsi += c1[k - 1] * (ax[x + k] * psi_x[off + k] - ax[x - k] * psi_x[off - k]);
                const T tmpx = ((T)1 + bx[x]) * d2x + dbxdx[x] * dwfcdx + dpsi * rdx;
                w_sum += ((T)1 + bx[x]) * tmpx + ax[x] * zeta_x[off];
                psi_x_new[off] = bx[x] * dwfcdx + ax[x] * psi_x[off];
                zeta_x_new[off] = bx[x] * tmpx + ax[x] * zeta_x[off];

                const T dwfcscdx = diff1_x(u_sc_cur, off, c1, rdx, fd_pad);
                const T d2xsc = diff2_x(u_sc_cur, off, c2, rdx2, fd_pad);
                T dpsisc = (T)0;
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k)
                    dpsisc += c1[k - 1] * (ax[x + k] * psi_x_sc[off + k] - ax[x - k] * psi_x_sc[off - k]);
                const T tmpxsc = ((T)1 + bx[x]) * d2xsc + dbxdx[x] * dwfcscdx + dpsisc * rdx;
                wsc_sum += ((T)1 + bx[x]) * tmpxsc + ax[x] * zeta_x_sc[off];
                psi_x_sc_new[off] = bx[x] * dwfcscdx + ax[x] * psi_x_sc[off];
                zeta_x_sc_new[off] = bx[x] * tmpxsc + ax[x] * zeta_x_sc[off];
            } else {
                w_sum += diff2_x(u_cur, off, c2, rdx2, fd_pad);
                wsc_sum += diff2_x(u_sc_cur, off, c2, rdx2, fd_pad);
            }

            u_new[off] = v2dt2 * w_sum + (T)2 * u_cur[off] - u_prev[off];
            u_sc_new[off] = v2dt2 * wsc_sum + (T)2 * u_sc_cur[off] - u_sc_prev[off]
                          + (T)2 * v_val * scs[off_base] * dt2 * w_sum;
            if (store && t % interval == 0) {
                const long soff = snap_off + off;
                w_store[soff] = w_sum;
                wsc_store[soff] = wsc_sum;
            }
        }
    }
}

// ---------------- forward: source injection / receiver recording ----------------
template <typename T>
__global__ void inject_kernel(
    T* __restrict__ u_new, T* __restrict__ u_sc_new,
    const T* __restrict__ f, const T* __restrict__ f_sc, const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long nz_ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        if (idx >= 0) {
            u_new[(long)s * nz_ny_nx + idx] += f[(((long)t * n_shots + s) * n_src + k)];
            u_sc_new[(long)s * nz_ny_nx + idx] += f_sc[(((long)t * n_shots + s) * n_src + k)];
        }
    }
}

template <typename T>
__global__ void record_kernel(
    const T* __restrict__ u_cur, const T* __restrict__ u_sc_cur,
    T* __restrict__ r, T* __restrict__ r_sc,
    const long* __restrict__ rec_i, const long* __restrict__ rec_sc_i,
    int t, int n_shots, int n_rec, int n_rec_sc, long nz_ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            r[(((long)t * n_shots + s) * n_rec + k)] = u_cur[(long)s * nz_ny_nx + idx];
    }
    if (s < n_shots && k < n_rec_sc) {
        const long idx = rec_sc_i[(long)s * n_rec_sc + k];
        if (idx >= 0)
            r_sc[(((long)t * n_shots + s) * n_rec_sc + k)] = u_sc_cur[(long)s * nz_ny_nx + idx];
    }
}

// ---------------- backward: adjoint step (interior: no PML) ----------------
// Exact discrete transpose of the coupled forward step: the background
// adjoint evolves with the Laplacian of
//   V2DT2_WFC = v^2 dt^2 lam_bg + 2 v dt^2 scatter lam_sc
// and the scattered adjoint with that of v^2 dt^2 lam_sc.
template <typename T>
__global__ void step_adjoint_interior_kernel(
    const T* __restrict__ v, const T* __restrict__ scatter,
    const T* __restrict__ lam_bg_next, const T* __restrict__ lam_bg_next2,
    const T* __restrict__ lam_sc_next, const T* __restrict__ lam_sc_next2,
    T* __restrict__ lam_bg_new, T* __restrict__ lam_sc_new,
    const T* __restrict__ w_store, const T* __restrict__ wsc_store,
    T* __restrict__ grad_v, T* __restrict__ grad_scatter,
    const T* __restrict__ c1, const T* __restrict__ c2,
    T rdz, T rdy, T rdx, T rdz2, T rdy2, T rdx2,
    int t, int interval, T scale, T dt2,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int v_batched, int scatter_batched, int64_t snap_off, int fd_pad)
{
    const int z = pml_z0 + blockIdx.z;
    const int y = pml_y0 + blockIdx.y * blockDim.y + threadIdx.y;
    const int x = pml_x0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= 1 && y >= 1 && x >= 1 && z < nz - 1 && y < ny - 1 && x < nx - 1 &&
        z < pml_z1 && y < pml_y1 && x < pml_x1) {
        const long ny_nx = (long)ny * nx;
        const long off_base = ((long)z * ny + y) * nx + x;
        for (int s = 0; s < n_shots; ++s) {
            const int s_v = v_batched ? s : 0;
            const int s_sc = scatter_batched ? s : 0;
            const T* vs = v + (long)s_v * nz * ny_nx;
            const T* scs = scatter + (long)s_sc * nz * ny_nx;
            const long off = (long)s * nz * ny_nx + off_base;
            const T v_val = vs[off_base];
            const T sc_val = scs[off_base];
            const T v2dt2 = v_val * v_val * dt2;
            T wz = c2[0] * (v2dt2 * lam_bg_next[off] + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off]);
#pragma unroll
            for (int k = 1; k <= fd_pad; ++k) {
                const T v_p = vs[off_base + k * ny_nx];
                const T v_m = vs[off_base - k * ny_nx];
                wz += c2[k] * (v_p * v_p * dt2 * lam_bg_next[off + k * ny_nx]
                             + (T)2 * v_p * scs[off_base + k * ny_nx] * dt2 * lam_sc_next[off + k * ny_nx]
                             + v_m * v_m * dt2 * lam_bg_next[off - k * ny_nx]
                             + (T)2 * v_m * scs[off_base - k * ny_nx] * dt2 * lam_sc_next[off - k * ny_nx]);
            }
            wz *= rdz2;
            T wy = c2[0] * (v2dt2 * lam_bg_next[off] + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off]);
#pragma unroll
            for (int k = 1; k <= fd_pad; ++k) {
                const T v_p = vs[off_base + k * nx];
                const T v_m = vs[off_base - k * nx];
                wy += c2[k] * (v_p * v_p * dt2 * lam_bg_next[off + k * nx]
                             + (T)2 * v_p * scs[off_base + k * nx] * dt2 * lam_sc_next[off + k * nx]
                             + v_m * v_m * dt2 * lam_bg_next[off - k * nx]
                             + (T)2 * v_m * scs[off_base - k * nx] * dt2 * lam_sc_next[off - k * nx]);
            }
            wy *= rdy2;
            T wx = c2[0] * (v2dt2 * lam_bg_next[off] + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off]);
#pragma unroll
            for (int k = 1; k <= fd_pad; ++k) {
                const T v_p = vs[off_base + k];
                const T v_m = vs[off_base - k];
                wx += c2[k] * (v_p * v_p * dt2 * lam_bg_next[off + k]
                             + (T)2 * v_p * scs[off_base + k] * dt2 * lam_sc_next[off + k]
                             + v_m * v_m * dt2 * lam_bg_next[off - k]
                             + (T)2 * v_m * scs[off_base - k] * dt2 * lam_sc_next[off - k]);
            }
            wx *= rdx2;
            lam_bg_new[off] = (T)2 * lam_bg_next[off] + wz + wy + wx - lam_bg_next2[off];

            T wzsc = c2[0] * (v2dt2 * lam_sc_next[off]);
#pragma unroll
            for (int k = 1; k <= fd_pad; ++k) {
                const T v_p = vs[off_base + k * ny_nx];
                const T v_m = vs[off_base - k * ny_nx];
                wzsc += c2[k] * (v_p * v_p * dt2 * lam_sc_next[off + k * ny_nx]
                               + v_m * v_m * dt2 * lam_sc_next[off - k * ny_nx]);
            }
            wzsc *= rdz2;
            T wysc = c2[0] * (v2dt2 * lam_sc_next[off]);
#pragma unroll
            for (int k = 1; k <= fd_pad; ++k) {
                const T v_p = vs[off_base + k * nx];
                const T v_m = vs[off_base - k * nx];
                wysc += c2[k] * (v_p * v_p * dt2 * lam_sc_next[off + k * nx]
                               + v_m * v_m * dt2 * lam_sc_next[off - k * nx]);
            }
            wysc *= rdy2;
            T wxsc = c2[0] * (v2dt2 * lam_sc_next[off]);
#pragma unroll
            for (int k = 1; k <= fd_pad; ++k) {
                const T v_p = vs[off_base + k];
                const T v_m = vs[off_base - k];
                wxsc += c2[k] * (v_p * v_p * dt2 * lam_sc_next[off + k]
                               + v_m * v_m * dt2 * lam_sc_next[off - k]);
            }
            wxsc *= rdx2;
            lam_sc_new[off] = (T)2 * lam_sc_next[off] + wzsc + wysc + wxsc - lam_sc_next2[off];

            if (t % interval == 0) {
                const long soff = snap_off + off;
                grad_v[off] += lam_bg_next[off] * ((T)2 * v_val * dt2 * w_store[soff]) * scale
                             + lam_sc_next[off] * ((T)2 * dt2 * sc_val * w_store[soff]
                                                 + (T)2 * v_val * dt2 * wsc_store[soff]) * scale;
                grad_scatter[off] += lam_sc_next[off] * (T)2 * v_val * dt2 * w_store[soff] * scale;
            }
        }
    }
}

// ---------------- backward: adjoint step (frame: includes PML borders) ----------------
// Transpose of the forward frame kernel with the same Born coupling as the
// adjoint interior step: V2DT2_WFC couples the background adjoint to
// the scattered adjoint (see born.cu for the 2D case).
template <typename T>
__global__ void step_adjoint_frame_kernel(
    const T* __restrict__ v, const T* __restrict__ scatter,
    const T* __restrict__ lam_bg_next, const T* __restrict__ lam_bg_next2,
    const T* __restrict__ lam_sc_next, const T* __restrict__ lam_sc_next2,
    T* __restrict__ lam_bg_new, T* __restrict__ lam_sc_new,
    const T* __restrict__ psi_z, const T* __restrict__ psi_y, const T* __restrict__ psi_x,
    const T* __restrict__ zeta_z, const T* __restrict__ zeta_y, const T* __restrict__ zeta_x,
    const T* __restrict__ psi_z_sc, const T* __restrict__ psi_y_sc, const T* __restrict__ psi_x_sc,
    const T* __restrict__ zeta_z_sc, const T* __restrict__ zeta_y_sc, const T* __restrict__ zeta_x_sc,
    T* __restrict__ psi_z_new, T* __restrict__ psi_y_new, T* __restrict__ psi_x_new,
    T* __restrict__ zeta_z_new, T* __restrict__ zeta_y_new, T* __restrict__ zeta_x_new,
    T* __restrict__ psi_z_sc_new, T* __restrict__ psi_y_sc_new, T* __restrict__ psi_x_sc_new,
    T* __restrict__ zeta_z_sc_new, T* __restrict__ zeta_y_sc_new, T* __restrict__ zeta_x_sc_new,
    const T* __restrict__ az, const T* __restrict__ bz, const T* __restrict__ dbzdz,
    const T* __restrict__ ay, const T* __restrict__ by, const T* __restrict__ dbydy,
    const T* __restrict__ ax, const T* __restrict__ bx, const T* __restrict__ dbxdx,
    const T* __restrict__ w_store, const T* __restrict__ wsc_store,
    T* __restrict__ grad_v, T* __restrict__ grad_scatter,
    const T* __restrict__ c1, const T* __restrict__ c2,
    T rdz, T rdy, T rdx, T rdz2, T rdy2, T rdx2,
    int t, int interval, T scale, T dt2,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int v_batched, int scatter_batched, int64_t snap_off, int fd_pad)
{
    const int z = fd_pad + blockIdx.z;
    const int y = fd_pad + blockIdx.y * blockDim.y + threadIdx.y;
    const int x = fd_pad + blockIdx.x * blockDim.x + threadIdx.x;
    // strictly interior cells are handled by step_adjoint_interior_kernel
    if (z >= pml_z0 && z < pml_z1 && y >= pml_y0 && y < pml_y1 &&
        x >= pml_x0 && x < pml_x1)
        return;
    if (z < nz - fd_pad && y < ny - fd_pad && x < nx - fd_pad) {
        const long ny_nx = (long)ny * nx;
        const long off_base = ((long)z * ny + y) * nx + x;
        for (int s = 0; s < n_shots; ++s) {
            const int s_v = v_batched ? s : 0;
            const int s_sc = scatter_batched ? s : 0;
            const T* vs = v + (long)s_v * nz * ny_nx;
            const T* scs = scatter + (long)s_sc * nz * ny_nx;
            const long off = (long)s * nz * ny_nx + off_base;
            const T v_val = vs[off_base];
            const T sc_val = scs[off_base];
            const T v2dt2 = v_val * v_val * dt2;
            T w_sum = (T)0;
            T wsc_sum = (T)0;

            // z: transpose of the CPML-modified Laplacian acting on lam_bg/lam_sc
            if (z < pml_z0 || z >= pml_z1) {
                T t1_sum = (T)0;
                T t2_sum = c2[0] * (((T)1 + bz[z]) * (((T)1 + bz[z]) * (v2dt2 * lam_bg_next[off]
                            + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off]) + bz[z] * zeta_z[off]));
                T p_sum = (T)0;
                T t1_sum_sc = (T)0;
                T t2_sum_sc = c2[0] * (((T)1 + bz[z]) * (((T)1 + bz[z]) * v2dt2 * lam_sc_next[off]
                            + bz[z] * zeta_z_sc[off]));
                T p_sum_sc = (T)0;
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k) {
                    const T v_p = vs[off_base + k * ny_nx];
                    const T v_m = vs[off_base - k * ny_nx];
                    const T wfc_p = v_p * v_p * dt2 * lam_bg_next[off + k * ny_nx]
                                  + (T)2 * v_p * scs[off_base + k * ny_nx] * dt2 * lam_sc_next[off + k * ny_nx];
                    const T wfc_m = v_m * v_m * dt2 * lam_bg_next[off - k * ny_nx]
                                  + (T)2 * v_m * scs[off_base - k * ny_nx] * dt2 * lam_sc_next[off - k * ny_nx];
                    const T wfcsc_p = v_p * v_p * dt2 * lam_sc_next[off + k * ny_nx];
                    const T wfcsc_m = v_m * v_m * dt2 * lam_sc_next[off - k * ny_nx];
                    const T t1_p = dbzdz[z + k] * (((T)1 + bz[z + k]) * wfc_p + bz[z + k] * zeta_z[off + k * ny_nx])
                                 + bz[z + k] * psi_z[off + k * ny_nx];
                    const T t1_m = dbzdz[z - k] * (((T)1 + bz[z - k]) * wfc_m + bz[z - k] * zeta_z[off - k * ny_nx])
                                 + bz[z - k] * psi_z[off - k * ny_nx];
                    t1_sum += c1[k - 1] * (t1_p - t1_m);
                    const T t2_p = ((T)1 + bz[z + k]) * (((T)1 + bz[z + k]) * wfc_p + bz[z + k] * zeta_z[off + k * ny_nx]);
                    const T t2_m = ((T)1 + bz[z - k]) * (((T)1 + bz[z - k]) * wfc_m + bz[z - k] * zeta_z[off - k * ny_nx]);
                    t2_sum += c2[k] * (t2_p + t2_m);
                    const T p_p = ((T)1 + bz[z + k]) * wfc_p + bz[z + k] * zeta_z[off + k * ny_nx];
                    const T p_m = ((T)1 + bz[z - k]) * wfc_m + bz[z - k] * zeta_z[off - k * ny_nx];
                    p_sum += c1[k - 1] * (p_p - p_m);

                    const T t1_p_sc = dbzdz[z + k] * (((T)1 + bz[z + k]) * wfcsc_p + bz[z + k] * zeta_z_sc[off + k * ny_nx])
                                    + bz[z + k] * psi_z_sc[off + k * ny_nx];
                    const T t1_m_sc = dbzdz[z - k] * (((T)1 + bz[z - k]) * wfcsc_m + bz[z - k] * zeta_z_sc[off - k * ny_nx])
                                    + bz[z - k] * psi_z_sc[off - k * ny_nx];
                    t1_sum_sc += c1[k - 1] * (t1_p_sc - t1_m_sc);
                    const T t2_p_sc = ((T)1 + bz[z + k]) * (((T)1 + bz[z + k]) * wfcsc_p + bz[z + k] * zeta_z_sc[off + k * ny_nx]);
                    const T t2_m_sc = ((T)1 + bz[z - k]) * (((T)1 + bz[z - k]) * wfcsc_m + bz[z - k] * zeta_z_sc[off - k * ny_nx]);
                    t2_sum_sc += c2[k] * (t2_p_sc + t2_m_sc);
                    const T p_p_sc = ((T)1 + bz[z + k]) * wfcsc_p + bz[z + k] * zeta_z_sc[off + k * ny_nx];
                    const T p_m_sc = ((T)1 + bz[z - k]) * wfcsc_m + bz[z - k] * zeta_z_sc[off - k * ny_nx];
                    p_sum_sc += c1[k - 1] * (p_p_sc - p_m_sc);
                }
                w_sum += -t1_sum * rdz + t2_sum * rdz2;
                psi_z_new[off] = -az[z] * p_sum * rdz + az[z] * psi_z[off];
                zeta_z_new[off] = az[z] * (v2dt2 * lam_bg_next[off] + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off])
                                + az[z] * zeta_z[off];
                wsc_sum += -t1_sum_sc * rdz + t2_sum_sc * rdz2;
                psi_z_sc_new[off] = -az[z] * p_sum_sc * rdz + az[z] * psi_z_sc[off];
                zeta_z_sc_new[off] = az[z] * v2dt2 * lam_sc_next[off] + az[z] * zeta_z_sc[off];
            } else {
                T wz = c2[0] * (v2dt2 * lam_bg_next[off] + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off]);
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k) {
                    const T v_p = vs[off_base + k * ny_nx];
                    const T v_m = vs[off_base - k * ny_nx];
                    wz += c2[k] * (v_p * v_p * dt2 * lam_bg_next[off + k * ny_nx]
                                 + (T)2 * v_p * scs[off_base + k * ny_nx] * dt2 * lam_sc_next[off + k * ny_nx]
                                 + v_m * v_m * dt2 * lam_bg_next[off - k * ny_nx]
                                 + (T)2 * v_m * scs[off_base - k * ny_nx] * dt2 * lam_sc_next[off - k * ny_nx]);
                }
                w_sum += wz * rdz2;
                T wzsc = c2[0] * (v2dt2 * lam_sc_next[off]);
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k) {
                    const T v_p = vs[off_base + k * ny_nx];
                    const T v_m = vs[off_base - k * ny_nx];
                    wzsc += c2[k] * (v_p * v_p * dt2 * lam_sc_next[off + k * ny_nx]
                                   + v_m * v_m * dt2 * lam_sc_next[off - k * ny_nx]);
                }
                wsc_sum += wzsc * rdz2;
            }
            // y: same structure
            if (y < pml_y0 || y >= pml_y1) {
                T t1_sum = (T)0;
                T t2_sum = c2[0] * (((T)1 + by[y]) * (((T)1 + by[y]) * (v2dt2 * lam_bg_next[off]
                            + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off]) + by[y] * zeta_y[off]));
                T p_sum = (T)0;
                T t1_sum_sc = (T)0;
                T t2_sum_sc = c2[0] * (((T)1 + by[y]) * (((T)1 + by[y]) * v2dt2 * lam_sc_next[off]
                            + by[y] * zeta_y_sc[off]));
                T p_sum_sc = (T)0;
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k) {
                    const T v_p = vs[off_base + k * nx];
                    const T v_m = vs[off_base - k * nx];
                    const T wfc_p = v_p * v_p * dt2 * lam_bg_next[off + k * nx]
                                  + (T)2 * v_p * scs[off_base + k * nx] * dt2 * lam_sc_next[off + k * nx];
                    const T wfc_m = v_m * v_m * dt2 * lam_bg_next[off - k * nx]
                                  + (T)2 * v_m * scs[off_base - k * nx] * dt2 * lam_sc_next[off - k * nx];
                    const T wfcsc_p = v_p * v_p * dt2 * lam_sc_next[off + k * nx];
                    const T wfcsc_m = v_m * v_m * dt2 * lam_sc_next[off - k * nx];
                    const T t1_p = dbydy[y + k] * (((T)1 + by[y + k]) * wfc_p + by[y + k] * zeta_y[off + k * nx])
                                 + by[y + k] * psi_y[off + k * nx];
                    const T t1_m = dbydy[y - k] * (((T)1 + by[y - k]) * wfc_m + by[y - k] * zeta_y[off - k * nx])
                                 + by[y - k] * psi_y[off - k * nx];
                    t1_sum += c1[k - 1] * (t1_p - t1_m);
                    const T t2_p = ((T)1 + by[y + k]) * (((T)1 + by[y + k]) * wfc_p + by[y + k] * zeta_y[off + k * nx]);
                    const T t2_m = ((T)1 + by[y - k]) * (((T)1 + by[y - k]) * wfc_m + by[y - k] * zeta_y[off - k * nx]);
                    t2_sum += c2[k] * (t2_p + t2_m);
                    const T p_p = ((T)1 + by[y + k]) * wfc_p + by[y + k] * zeta_y[off + k * nx];
                    const T p_m = ((T)1 + by[y - k]) * wfc_m + by[y - k] * zeta_y[off - k * nx];
                    p_sum += c1[k - 1] * (p_p - p_m);

                    const T t1_p_sc = dbydy[y + k] * (((T)1 + by[y + k]) * wfcsc_p + by[y + k] * zeta_y_sc[off + k * nx])
                                    + by[y + k] * psi_y_sc[off + k * nx];
                    const T t1_m_sc = dbydy[y - k] * (((T)1 + by[y - k]) * wfcsc_m + by[y - k] * zeta_y_sc[off - k * nx])
                                    + by[y - k] * psi_y_sc[off - k * nx];
                    t1_sum_sc += c1[k - 1] * (t1_p_sc - t1_m_sc);
                    const T t2_p_sc = ((T)1 + by[y + k]) * (((T)1 + by[y + k]) * wfcsc_p + by[y + k] * zeta_y_sc[off + k * nx]);
                    const T t2_m_sc = ((T)1 + by[y - k]) * (((T)1 + by[y - k]) * wfcsc_m + by[y - k] * zeta_y_sc[off - k * nx]);
                    t2_sum_sc += c2[k] * (t2_p_sc + t2_m_sc);
                    const T p_p_sc = ((T)1 + by[y + k]) * wfcsc_p + by[y + k] * zeta_y_sc[off + k * nx];
                    const T p_m_sc = ((T)1 + by[y - k]) * wfcsc_m + by[y - k] * zeta_y_sc[off - k * nx];
                    p_sum_sc += c1[k - 1] * (p_p_sc - p_m_sc);
                }
                w_sum += -t1_sum * rdy + t2_sum * rdy2;
                psi_y_new[off] = -ay[y] * p_sum * rdy + ay[y] * psi_y[off];
                zeta_y_new[off] = ay[y] * (v2dt2 * lam_bg_next[off] + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off])
                                + ay[y] * zeta_y[off];
                wsc_sum += -t1_sum_sc * rdy + t2_sum_sc * rdy2;
                psi_y_sc_new[off] = -ay[y] * p_sum_sc * rdy + ay[y] * psi_y_sc[off];
                zeta_y_sc_new[off] = ay[y] * v2dt2 * lam_sc_next[off] + ay[y] * zeta_y_sc[off];
            } else {
                T wy = c2[0] * (v2dt2 * lam_bg_next[off] + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off]);
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k) {
                    const T v_p = vs[off_base + k * nx];
                    const T v_m = vs[off_base - k * nx];
                    wy += c2[k] * (v_p * v_p * dt2 * lam_bg_next[off + k * nx]
                                 + (T)2 * v_p * scs[off_base + k * nx] * dt2 * lam_sc_next[off + k * nx]
                                 + v_m * v_m * dt2 * lam_bg_next[off - k * nx]
                                 + (T)2 * v_m * scs[off_base - k * nx] * dt2 * lam_sc_next[off - k * nx]);
                }
                w_sum += wy * rdy2;
                T wysc = c2[0] * (v2dt2 * lam_sc_next[off]);
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k) {
                    const T v_p = vs[off_base + k * nx];
                    const T v_m = vs[off_base - k * nx];
                    wysc += c2[k] * (v_p * v_p * dt2 * lam_sc_next[off + k * nx]
                                   + v_m * v_m * dt2 * lam_sc_next[off - k * nx]);
                }
                wsc_sum += wysc * rdy2;
            }
            // x: same structure
            if (x < pml_x0 || x >= pml_x1) {
                T t1_sum = (T)0;
                T t2_sum = c2[0] * (((T)1 + bx[x]) * (((T)1 + bx[x]) * (v2dt2 * lam_bg_next[off]
                            + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off]) + bx[x] * zeta_x[off]));
                T p_sum = (T)0;
                T t1_sum_sc = (T)0;
                T t2_sum_sc = c2[0] * (((T)1 + bx[x]) * (((T)1 + bx[x]) * v2dt2 * lam_sc_next[off]
                            + bx[x] * zeta_x_sc[off]));
                T p_sum_sc = (T)0;
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k) {
                    const T v_p = vs[off_base + k];
                    const T v_m = vs[off_base - k];
                    const T wfc_p = v_p * v_p * dt2 * lam_bg_next[off + k]
                                  + (T)2 * v_p * scs[off_base + k] * dt2 * lam_sc_next[off + k];
                    const T wfc_m = v_m * v_m * dt2 * lam_bg_next[off - k]
                                  + (T)2 * v_m * scs[off_base - k] * dt2 * lam_sc_next[off - k];
                    const T wfcsc_p = v_p * v_p * dt2 * lam_sc_next[off + k];
                    const T wfcsc_m = v_m * v_m * dt2 * lam_sc_next[off - k];
                    const T t1_p = dbxdx[x + k] * (((T)1 + bx[x + k]) * wfc_p + bx[x + k] * zeta_x[off + k])
                                 + bx[x + k] * psi_x[off + k];
                    const T t1_m = dbxdx[x - k] * (((T)1 + bx[x - k]) * wfc_m + bx[x - k] * zeta_x[off - k])
                                 + bx[x - k] * psi_x[off - k];
                    t1_sum += c1[k - 1] * (t1_p - t1_m);
                    const T t2_p = ((T)1 + bx[x + k]) * (((T)1 + bx[x + k]) * wfc_p + bx[x + k] * zeta_x[off + k]);
                    const T t2_m = ((T)1 + bx[x - k]) * (((T)1 + bx[x - k]) * wfc_m + bx[x - k] * zeta_x[off - k]);
                    t2_sum += c2[k] * (t2_p + t2_m);
                    const T p_p = ((T)1 + bx[x + k]) * wfc_p + bx[x + k] * zeta_x[off + k];
                    const T p_m = ((T)1 + bx[x - k]) * wfc_m + bx[x - k] * zeta_x[off - k];
                    p_sum += c1[k - 1] * (p_p - p_m);

                    const T t1_p_sc = dbxdx[x + k] * (((T)1 + bx[x + k]) * wfcsc_p + bx[x + k] * zeta_x_sc[off + k])
                                    + bx[x + k] * psi_x_sc[off + k];
                    const T t1_m_sc = dbxdx[x - k] * (((T)1 + bx[x - k]) * wfcsc_m + bx[x - k] * zeta_x_sc[off - k])
                                    + bx[x - k] * psi_x_sc[off - k];
                    t1_sum_sc += c1[k - 1] * (t1_p_sc - t1_m_sc);
                    const T t2_p_sc = ((T)1 + bx[x + k]) * (((T)1 + bx[x + k]) * wfcsc_p + bx[x + k] * zeta_x_sc[off + k]);
                    const T t2_m_sc = ((T)1 + bx[x - k]) * (((T)1 + bx[x - k]) * wfcsc_m + bx[x - k] * zeta_x_sc[off - k]);
                    t2_sum_sc += c2[k] * (t2_p_sc + t2_m_sc);
                    const T p_p_sc = ((T)1 + bx[x + k]) * wfcsc_p + bx[x + k] * zeta_x_sc[off + k];
                    const T p_m_sc = ((T)1 + bx[x - k]) * wfcsc_m + bx[x - k] * zeta_x_sc[off - k];
                    p_sum_sc += c1[k - 1] * (p_p_sc - p_m_sc);
                }
                w_sum += -t1_sum * rdx + t2_sum * rdx2;
                psi_x_new[off] = -ax[x] * p_sum * rdx + ax[x] * psi_x[off];
                zeta_x_new[off] = ax[x] * (v2dt2 * lam_bg_next[off] + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off])
                                + ax[x] * zeta_x[off];
                wsc_sum += -t1_sum_sc * rdx + t2_sum_sc * rdx2;
                psi_x_sc_new[off] = -ax[x] * p_sum_sc * rdx + ax[x] * psi_x_sc[off];
                zeta_x_sc_new[off] = ax[x] * v2dt2 * lam_sc_next[off] + ax[x] * zeta_x_sc[off];
            } else {
                T wx = c2[0] * (v2dt2 * lam_bg_next[off] + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off]);
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k) {
                    const T v_p = vs[off_base + k];
                    const T v_m = vs[off_base - k];
                    wx += c2[k] * (v_p * v_p * dt2 * lam_bg_next[off + k]
                                 + (T)2 * v_p * scs[off_base + k] * dt2 * lam_sc_next[off + k]
                                 + v_m * v_m * dt2 * lam_bg_next[off - k]
                                 + (T)2 * v_m * scs[off_base - k] * dt2 * lam_sc_next[off - k]);
                }
                w_sum += wx * rdx2;
                T wxsc = c2[0] * (v2dt2 * lam_sc_next[off]);
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k) {
                    const T v_p = vs[off_base + k];
                    const T v_m = vs[off_base - k];
                    wxsc += c2[k] * (v_p * v_p * dt2 * lam_sc_next[off + k]
                                   + v_m * v_m * dt2 * lam_sc_next[off - k]);
                }
                wsc_sum += wxsc * rdx2;
            }

            lam_bg_new[off] = (T)2 * lam_bg_next[off] + w_sum - lam_bg_next2[off];
            lam_sc_new[off] = (T)2 * lam_sc_next[off] + wsc_sum - lam_sc_next2[off];
            if (t % interval == 0) {
                const long soff = snap_off + off;
                grad_v[off] += lam_bg_next[off] * ((T)2 * v_val * dt2 * w_store[soff]) * scale
                             + lam_sc_next[off] * ((T)2 * dt2 * sc_val * w_store[soff]
                                                 + (T)2 * v_val * dt2 * wsc_store[soff]) * scale;
                grad_scatter[off] += lam_sc_next[off] * (T)2 * v_val * dt2 * w_store[soff] * scale;
            }
        }
    }
}

template <typename T>
__global__ void record_grad_f_kernel(
    const T* __restrict__ lam_bg_next, const T* __restrict__ lam_sc_next,
    T* __restrict__ grad_f, T* __restrict__ grad_f_sc, const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long nz_ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        grad_f[(((long)t * n_shots + s) * n_src + k)] = (idx >= 0) ? lam_bg_next[(long)s * nz_ny_nx + idx] : (T)0;
        grad_f_sc[(((long)t * n_shots + s) * n_src + k)] = (idx >= 0) ? lam_sc_next[(long)s * nz_ny_nx + idx] : (T)0;
    }
}

template <typename T>
__global__ void record_grad_r_kernel(
    T* __restrict__ lam_bg_cur, T* __restrict__ lam_sc_cur,
    const T* __restrict__ grad_r, const T* __restrict__ grad_r_sc,
    const long* __restrict__ rec_i, const long* __restrict__ rec_sc_i,
    int t, int n_shots, int n_rec, int n_rec_sc, long nz_ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            lam_bg_cur[(long)s * nz_ny_nx + idx] += grad_r[(((long)t * n_shots + s) * n_rec + k)];
    }
    if (s < n_shots && k < n_rec_sc) {
        const long idx = rec_sc_i[(long)s * n_rec_sc + k];
        if (idx >= 0)
            lam_sc_cur[(long)s * nz_ny_nx + idx] += grad_r_sc[(((long)t * n_shots + s) * n_rec_sc + k)];
    }
}

// ---------------- launchers ----------------
#define LAUNCH_STEP_INTERIOR(KERN, T, ...)                                     \
    {                                                                          \
        dim3 block(16, 16, 1);                                                 \
        dim3 grid((pml_x1 - pml_x0 + 15) / 16, (pml_y1 - pml_y0 + 15) / 16,    \
                  pml_z1 - pml_z0);                                            \
        if ((pml_z1 - pml_z0) > 0 && (pml_y1 - pml_y0) > 0 &&                  \
            (pml_x1 - pml_x0) > 0)                                             \
            KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami scalar3d_born interior kernel failed"); \
    }

#define LAUNCH_STEP_FRAME(KERN, T, ...)                                        \
    {                                                                          \
        dim3 block(16, 16, 1);                                                 \
        dim3 grid((nx - 2 * fd_pad + 15) / 16, (ny - 2 * fd_pad + 15) / 16,    \
                  nz - 2 * fd_pad);                                            \
        if ((nz - 2 * fd_pad) > 0 && (ny - 2 * fd_pad) > 0 &&                  \
            (nx - 2 * fd_pad) > 0)                                             \
            KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami scalar3d_born frame kernel failed"); \
    }

void forward_step(
    torch::Tensor v, torch::Tensor scatter,
    torch::Tensor u_cur, torch::Tensor u_prev, torch::Tensor u_sc_cur, torch::Tensor u_sc_prev,
    torch::Tensor psi_z, torch::Tensor psi_y, torch::Tensor psi_x,
    torch::Tensor zeta_z, torch::Tensor zeta_y, torch::Tensor zeta_x,
    torch::Tensor psi_z_sc, torch::Tensor psi_y_sc, torch::Tensor psi_x_sc,
    torch::Tensor zeta_z_sc, torch::Tensor zeta_y_sc, torch::Tensor zeta_x_sc,
    torch::Tensor u_new, torch::Tensor u_sc_new,
    torch::Tensor psi_z_new, torch::Tensor psi_y_new, torch::Tensor psi_x_new,
    torch::Tensor zeta_z_new, torch::Tensor zeta_y_new, torch::Tensor zeta_x_new,
    torch::Tensor psi_z_sc_new, torch::Tensor psi_y_sc_new, torch::Tensor psi_x_sc_new,
    torch::Tensor zeta_z_sc_new, torch::Tensor zeta_y_sc_new, torch::Tensor zeta_x_sc_new,
    torch::Tensor az, torch::Tensor bz, torch::Tensor dbzdz,
    torch::Tensor ay, torch::Tensor by, torch::Tensor dbydy,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor dbxdx,
    torch::Tensor w_store, torch::Tensor wsc_store,
    torch::Tensor c1, torch::Tensor c2,
    double rdz, double rdy, double rdx, double rdz2, double rdy2, double rdx2,
    int64_t t, int64_t interval, double dt2,
    int64_t pml_z0, int64_t pml_z1, int64_t pml_y0, int64_t pml_y1,
    int64_t pml_x0, int64_t pml_x1,
    int64_t v_batched, int64_t scatter_batched, int64_t store,
    int64_t snap_off, int64_t fd_pad)
{
    const int n_shots = v.size(0), nz = v.size(1), ny = v.size(2), nx = v.size(3);
    CHECK_CONTIG(c1);
    CHECK_CONTIG(c2);
    if (v.scalar_type() == torch::kFloat32) {
        LAUNCH_STEP_INTERIOR(step_forward_interior_kernel, float,
            v.data_ptr<float>(), scatter.data_ptr<float>(),
            u_cur.data_ptr<float>(), u_prev.data_ptr<float>(),
            u_sc_cur.data_ptr<float>(), u_sc_prev.data_ptr<float>(),
            u_new.data_ptr<float>(), u_sc_new.data_ptr<float>(),
            w_store.data_ptr<float>(), wsc_store.data_ptr<float>(),
            c1.data_ptr<float>(), c2.data_ptr<float>(),
            (float)rdz, (float)rdy, (float)rdx, (float)rdz2, (float)rdy2, (float)rdx2,
            (int)t, (int)interval, (float)dt2,
            n_shots, nz, ny, nx, (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1, (int)v_batched, (int)scatter_batched,
            (int)store, (int64_t)snap_off, (int)fd_pad);
        LAUNCH_STEP_FRAME(step_forward_frame_kernel, float,
            v.data_ptr<float>(), scatter.data_ptr<float>(),
            u_cur.data_ptr<float>(), u_prev.data_ptr<float>(),
            u_sc_cur.data_ptr<float>(), u_sc_prev.data_ptr<float>(),
            psi_z.data_ptr<float>(), psi_y.data_ptr<float>(), psi_x.data_ptr<float>(),
            zeta_z.data_ptr<float>(), zeta_y.data_ptr<float>(), zeta_x.data_ptr<float>(),
            psi_z_sc.data_ptr<float>(), psi_y_sc.data_ptr<float>(), psi_x_sc.data_ptr<float>(),
            zeta_z_sc.data_ptr<float>(), zeta_y_sc.data_ptr<float>(), zeta_x_sc.data_ptr<float>(),
            u_new.data_ptr<float>(), u_sc_new.data_ptr<float>(),
            psi_z_new.data_ptr<float>(), psi_y_new.data_ptr<float>(), psi_x_new.data_ptr<float>(),
            zeta_z_new.data_ptr<float>(), zeta_y_new.data_ptr<float>(), zeta_x_new.data_ptr<float>(),
            psi_z_sc_new.data_ptr<float>(), psi_y_sc_new.data_ptr<float>(), psi_x_sc_new.data_ptr<float>(),
            zeta_z_sc_new.data_ptr<float>(), zeta_y_sc_new.data_ptr<float>(), zeta_x_sc_new.data_ptr<float>(),
            az.data_ptr<float>(), bz.data_ptr<float>(), dbzdz.data_ptr<float>(),
            ay.data_ptr<float>(), by.data_ptr<float>(), dbydy.data_ptr<float>(),
            ax.data_ptr<float>(), bx.data_ptr<float>(), dbxdx.data_ptr<float>(),
            w_store.data_ptr<float>(), wsc_store.data_ptr<float>(),
            c1.data_ptr<float>(), c2.data_ptr<float>(),
            (float)rdz, (float)rdy, (float)rdx, (float)rdz2, (float)rdy2, (float)rdx2,
            (int)t, (int)interval, (float)dt2,
            n_shots, nz, ny, nx, (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1, (int)v_batched, (int)scatter_batched,
            (int)store, (int64_t)snap_off, (int)fd_pad);
    } else {
        LAUNCH_STEP_INTERIOR(step_forward_interior_kernel, double,
            v.data_ptr<double>(), scatter.data_ptr<double>(),
            u_cur.data_ptr<double>(), u_prev.data_ptr<double>(),
            u_sc_cur.data_ptr<double>(), u_sc_prev.data_ptr<double>(),
            u_new.data_ptr<double>(), u_sc_new.data_ptr<double>(),
            w_store.data_ptr<double>(), wsc_store.data_ptr<double>(),
            c1.data_ptr<double>(), c2.data_ptr<double>(),
            rdz, rdy, rdx, rdz2, rdy2, rdx2,
            (int)t, (int)interval, dt2,
            n_shots, nz, ny, nx, (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1, (int)v_batched, (int)scatter_batched,
            (int)store, (int64_t)snap_off, (int)fd_pad);
        LAUNCH_STEP_FRAME(step_forward_frame_kernel, double,
            v.data_ptr<double>(), scatter.data_ptr<double>(),
            u_cur.data_ptr<double>(), u_prev.data_ptr<double>(),
            u_sc_cur.data_ptr<double>(), u_sc_prev.data_ptr<double>(),
            psi_z.data_ptr<double>(), psi_y.data_ptr<double>(), psi_x.data_ptr<double>(),
            zeta_z.data_ptr<double>(), zeta_y.data_ptr<double>(), zeta_x.data_ptr<double>(),
            psi_z_sc.data_ptr<double>(), psi_y_sc.data_ptr<double>(), psi_x_sc.data_ptr<double>(),
            zeta_z_sc.data_ptr<double>(), zeta_y_sc.data_ptr<double>(), zeta_x_sc.data_ptr<double>(),
            u_new.data_ptr<double>(), u_sc_new.data_ptr<double>(),
            psi_z_new.data_ptr<double>(), psi_y_new.data_ptr<double>(), psi_x_new.data_ptr<double>(),
            zeta_z_new.data_ptr<double>(), zeta_y_new.data_ptr<double>(), zeta_x_new.data_ptr<double>(),
            psi_z_sc_new.data_ptr<double>(), psi_y_sc_new.data_ptr<double>(), psi_x_sc_new.data_ptr<double>(),
            zeta_z_sc_new.data_ptr<double>(), zeta_y_sc_new.data_ptr<double>(), zeta_x_sc_new.data_ptr<double>(),
            az.data_ptr<double>(), bz.data_ptr<double>(), dbzdz.data_ptr<double>(),
            ay.data_ptr<double>(), by.data_ptr<double>(), dbydy.data_ptr<double>(),
            ax.data_ptr<double>(), bx.data_ptr<double>(), dbxdx.data_ptr<double>(),
            w_store.data_ptr<double>(), wsc_store.data_ptr<double>(),
            c1.data_ptr<double>(), c2.data_ptr<double>(),
            rdz, rdy, rdx, rdz2, rdy2, rdx2,
            (int)t, (int)interval, dt2,
            n_shots, nz, ny, nx, (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1, (int)v_batched, (int)scatter_batched,
            (int)store, (int64_t)snap_off, (int)fd_pad);
    }
}

void adjoint_step(
    torch::Tensor v, torch::Tensor scatter,
    torch::Tensor lam_bg_next, torch::Tensor lam_bg_next2, torch::Tensor lam_sc_next, torch::Tensor lam_sc_next2,
    torch::Tensor lam_bg_new, torch::Tensor lam_sc_new,
    torch::Tensor psi_z, torch::Tensor psi_y, torch::Tensor psi_x,
    torch::Tensor zeta_z, torch::Tensor zeta_y, torch::Tensor zeta_x,
    torch::Tensor psi_z_sc, torch::Tensor psi_y_sc, torch::Tensor psi_x_sc,
    torch::Tensor zeta_z_sc, torch::Tensor zeta_y_sc, torch::Tensor zeta_x_sc,
    torch::Tensor psi_z_new, torch::Tensor psi_y_new, torch::Tensor psi_x_new,
    torch::Tensor zeta_z_new, torch::Tensor zeta_y_new, torch::Tensor zeta_x_new,
    torch::Tensor psi_z_sc_new, torch::Tensor psi_y_sc_new, torch::Tensor psi_x_sc_new,
    torch::Tensor zeta_z_sc_new, torch::Tensor zeta_y_sc_new, torch::Tensor zeta_x_sc_new,
    torch::Tensor az, torch::Tensor bz, torch::Tensor dbzdz,
    torch::Tensor ay, torch::Tensor by, torch::Tensor dbydy,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor dbxdx,
    torch::Tensor w_store, torch::Tensor wsc_store,
    torch::Tensor grad_v, torch::Tensor grad_scatter,
    torch::Tensor c1, torch::Tensor c2,
    double rdz, double rdy, double rdx, double rdz2, double rdy2, double rdx2,
    int64_t t, int64_t interval, double scale, double dt2,
    // forward PML boundaries (pml_width + 2*fd_pad)
    int64_t pml_z0, int64_t pml_z1, int64_t pml_y0, int64_t pml_y1,
    int64_t pml_x0, int64_t pml_x1,
    // backward PML boundaries (pml_width + 3*fd_pad): the adjoint widens
    // the PML region by one fd_pad because the transpose of the
    // forward PML stencil reads one cell further into the interior.
    int64_t pml_z0_b, int64_t pml_z1_b, int64_t pml_y0_b, int64_t pml_y1_b,
    int64_t pml_x0_b, int64_t pml_x1_b,
    int64_t v_batched, int64_t scatter_batched, int64_t snap_off, int64_t fd_pad)
{
    const int n_shots = v.size(0), nz = v.size(1), ny = v.size(2), nx = v.size(3);
    CHECK_CONTIG(c1);
    CHECK_CONTIG(c2);
    if (v.scalar_type() == torch::kFloat32) {
        LAUNCH_STEP_INTERIOR(step_adjoint_interior_kernel, float,
            v.data_ptr<float>(), scatter.data_ptr<float>(),
            lam_bg_next.data_ptr<float>(), lam_bg_next2.data_ptr<float>(),
            lam_sc_next.data_ptr<float>(), lam_sc_next2.data_ptr<float>(),
            lam_bg_new.data_ptr<float>(), lam_sc_new.data_ptr<float>(),
            w_store.data_ptr<float>(), wsc_store.data_ptr<float>(),
            grad_v.data_ptr<float>(), grad_scatter.data_ptr<float>(),
            c1.data_ptr<float>(), c2.data_ptr<float>(),
            (float)rdz, (float)rdy, (float)rdx, (float)rdz2, (float)rdy2, (float)rdx2,
            (int)t, (int)interval, (float)scale, (float)dt2,
            n_shots, nz, ny, nx, (int)pml_z0_b, (int)pml_z1_b, (int)pml_y0_b, (int)pml_y1_b,
            (int)pml_x0_b, (int)pml_x1_b, (int)v_batched, (int)scatter_batched,
            (int64_t)snap_off, (int)fd_pad);
        LAUNCH_STEP_FRAME(step_adjoint_frame_kernel, float,
            v.data_ptr<float>(), scatter.data_ptr<float>(),
            lam_bg_next.data_ptr<float>(), lam_bg_next2.data_ptr<float>(),
            lam_sc_next.data_ptr<float>(), lam_sc_next2.data_ptr<float>(),
            lam_bg_new.data_ptr<float>(), lam_sc_new.data_ptr<float>(),
            psi_z.data_ptr<float>(), psi_y.data_ptr<float>(), psi_x.data_ptr<float>(),
            zeta_z.data_ptr<float>(), zeta_y.data_ptr<float>(), zeta_x.data_ptr<float>(),
            psi_z_sc.data_ptr<float>(), psi_y_sc.data_ptr<float>(), psi_x_sc.data_ptr<float>(),
            zeta_z_sc.data_ptr<float>(), zeta_y_sc.data_ptr<float>(), zeta_x_sc.data_ptr<float>(),
            psi_z_new.data_ptr<float>(), psi_y_new.data_ptr<float>(), psi_x_new.data_ptr<float>(),
            zeta_z_new.data_ptr<float>(), zeta_y_new.data_ptr<float>(), zeta_x_new.data_ptr<float>(),
            psi_z_sc_new.data_ptr<float>(), psi_y_sc_new.data_ptr<float>(), psi_x_sc_new.data_ptr<float>(),
            zeta_z_sc_new.data_ptr<float>(), zeta_y_sc_new.data_ptr<float>(), zeta_x_sc_new.data_ptr<float>(),
            az.data_ptr<float>(), bz.data_ptr<float>(), dbzdz.data_ptr<float>(),
            ay.data_ptr<float>(), by.data_ptr<float>(), dbydy.data_ptr<float>(),
            ax.data_ptr<float>(), bx.data_ptr<float>(), dbxdx.data_ptr<float>(),
            w_store.data_ptr<float>(), wsc_store.data_ptr<float>(),
            grad_v.data_ptr<float>(), grad_scatter.data_ptr<float>(),
            c1.data_ptr<float>(), c2.data_ptr<float>(),
            (float)rdz, (float)rdy, (float)rdx, (float)rdz2, (float)rdy2, (float)rdx2,
            (int)t, (int)interval, (float)scale, (float)dt2,
            n_shots, nz, ny, nx, (int)pml_z0_b, (int)pml_z1_b, (int)pml_y0_b, (int)pml_y1_b,
            (int)pml_x0_b, (int)pml_x1_b, (int)v_batched, (int)scatter_batched,
            (int64_t)snap_off, (int)fd_pad);
    } else {
        LAUNCH_STEP_INTERIOR(step_adjoint_interior_kernel, double,
            v.data_ptr<double>(), scatter.data_ptr<double>(),
            lam_bg_next.data_ptr<double>(), lam_bg_next2.data_ptr<double>(),
            lam_sc_next.data_ptr<double>(), lam_sc_next2.data_ptr<double>(),
            lam_bg_new.data_ptr<double>(), lam_sc_new.data_ptr<double>(),
            w_store.data_ptr<double>(), wsc_store.data_ptr<double>(),
            grad_v.data_ptr<double>(), grad_scatter.data_ptr<double>(),
            c1.data_ptr<double>(), c2.data_ptr<double>(),
            rdz, rdy, rdx, rdz2, rdy2, rdx2,
            (int)t, (int)interval, scale, dt2,
            n_shots, nz, ny, nx, (int)pml_z0_b, (int)pml_z1_b, (int)pml_y0_b, (int)pml_y1_b,
            (int)pml_x0_b, (int)pml_x1_b, (int)v_batched, (int)scatter_batched,
            (int64_t)snap_off, (int)fd_pad);
        LAUNCH_STEP_FRAME(step_adjoint_frame_kernel, double,
            v.data_ptr<double>(), scatter.data_ptr<double>(),
            lam_bg_next.data_ptr<double>(), lam_bg_next2.data_ptr<double>(),
            lam_sc_next.data_ptr<double>(), lam_sc_next2.data_ptr<double>(),
            lam_bg_new.data_ptr<double>(), lam_sc_new.data_ptr<double>(),
            psi_z.data_ptr<double>(), psi_y.data_ptr<double>(), psi_x.data_ptr<double>(),
            zeta_z.data_ptr<double>(), zeta_y.data_ptr<double>(), zeta_x.data_ptr<double>(),
            psi_z_sc.data_ptr<double>(), psi_y_sc.data_ptr<double>(), psi_x_sc.data_ptr<double>(),
            zeta_z_sc.data_ptr<double>(), zeta_y_sc.data_ptr<double>(), zeta_x_sc.data_ptr<double>(),
            psi_z_new.data_ptr<double>(), psi_y_new.data_ptr<double>(), psi_x_new.data_ptr<double>(),
            zeta_z_new.data_ptr<double>(), zeta_y_new.data_ptr<double>(), zeta_x_new.data_ptr<double>(),
            psi_z_sc_new.data_ptr<double>(), psi_y_sc_new.data_ptr<double>(), psi_x_sc_new.data_ptr<double>(),
            zeta_z_sc_new.data_ptr<double>(), zeta_y_sc_new.data_ptr<double>(), zeta_x_sc_new.data_ptr<double>(),
            az.data_ptr<double>(), bz.data_ptr<double>(), dbzdz.data_ptr<double>(),
            ay.data_ptr<double>(), by.data_ptr<double>(), dbydy.data_ptr<double>(),
            ax.data_ptr<double>(), bx.data_ptr<double>(), dbxdx.data_ptr<double>(),
            w_store.data_ptr<double>(), wsc_store.data_ptr<double>(),
            grad_v.data_ptr<double>(), grad_scatter.data_ptr<double>(),
            c1.data_ptr<double>(), c2.data_ptr<double>(),
            rdz, rdy, rdx, rdz2, rdy2, rdx2,
            (int)t, (int)interval, scale, dt2,
            n_shots, nz, ny, nx, (int)pml_z0_b, (int)pml_z1_b, (int)pml_y0_b, (int)pml_y1_b,
            (int)pml_x0_b, (int)pml_x1_b, (int)v_batched, (int)scatter_batched,
            (int64_t)snap_off, (int)fd_pad);
    }
}

void inject(torch::Tensor u_new, torch::Tensor u_sc_new,
            torch::Tensor f, torch::Tensor f_sc, torch::Tensor src_i,
            int64_t t, int64_t n_shots, int64_t n_src, int64_t nz_ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_src + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(u_new.scalar_type(), "inject", [&] {
        inject_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            u_new.data_ptr<scalar_t>(), u_sc_new.data_ptr<scalar_t>(),
            f.data_ptr<scalar_t>(), f_sc.data_ptr<scalar_t>(), src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)nz_ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami scalar3d_born inject failed");
}

void record(torch::Tensor u_cur, torch::Tensor u_sc_cur,
            torch::Tensor r, torch::Tensor r_sc,
            torch::Tensor rec_i, torch::Tensor rec_sc_i,
            int64_t t, int64_t n_shots, int64_t n_rec, int64_t n_rec_sc, int64_t nz_ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, ((n_rec > n_rec_sc ? n_rec : n_rec_sc) + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(u_cur.scalar_type(), "record", [&] {
        record_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            u_cur.data_ptr<scalar_t>(), u_sc_cur.data_ptr<scalar_t>(),
            r.data_ptr<scalar_t>(), r_sc.data_ptr<scalar_t>(),
            rec_i.data_ptr<int64_t>(), rec_sc_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (int)n_rec_sc, (long)nz_ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami scalar3d_born record failed");
}

void record_grad_f(torch::Tensor lam_bg_next, torch::Tensor lam_sc_next,
                   torch::Tensor grad_f, torch::Tensor grad_f_sc, torch::Tensor src_i,
                   int64_t t, int64_t n_shots, int64_t n_src, int64_t nz_ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_src + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(lam_bg_next.scalar_type(), "record_grad_f", [&] {
        record_grad_f_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            lam_bg_next.data_ptr<scalar_t>(), lam_sc_next.data_ptr<scalar_t>(),
            grad_f.data_ptr<scalar_t>(), grad_f_sc.data_ptr<scalar_t>(), src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)nz_ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami scalar3d_born record_grad_f failed");
}

void record_grad_r(torch::Tensor lam_bg_cur, torch::Tensor lam_sc_cur,
                   torch::Tensor grad_r, torch::Tensor grad_r_sc,
                   torch::Tensor rec_i, torch::Tensor rec_sc_i,
                   int64_t t, int64_t n_shots, int64_t n_rec, int64_t n_rec_sc, int64_t nz_ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, ((n_rec > n_rec_sc ? n_rec : n_rec_sc) + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(lam_bg_cur.scalar_type(), "record_grad_r", [&] {
        record_grad_r_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            lam_bg_cur.data_ptr<scalar_t>(), lam_sc_cur.data_ptr<scalar_t>(),
            grad_r.data_ptr<scalar_t>(), grad_r_sc.data_ptr<scalar_t>(),
            rec_i.data_ptr<int64_t>(), rec_sc_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (int)n_rec_sc, (long)nz_ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami scalar3d_born record_grad_r failed");
}

// ---------------- pybind11 bindings ----------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_step", &forward_step);
    m.def("inject", &inject);
    m.def("record", &record);
    m.def("adjoint_step", &adjoint_step);
    m.def("record_grad_f", &record_grad_f);
    m.def("record_grad_r", &record_grad_r);
    NAMI_STORAGE_PYBIND(m);
}
