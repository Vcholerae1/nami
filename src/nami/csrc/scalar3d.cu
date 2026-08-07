// nami scalar3d: 3D acoustic FDTD with native CUDA adjoint.
//
// 3D scalar FDTD backend (bit-for-bit reproducible in float64), mirroring
// src/nami/csrc/scalar2d.cu exactly: CPML (Pasalic & McGarry) with three
// split-field memory pairs (psi_z/zeta_z, psi_y/zeta_y, psi_x/zeta_x),
// pressure-source convention f = -amp * v^2 dt^2, receivers recording the
// pre-update field, and FD accuracy 2/4/6/8 via the shared coefficient
// tables (fixed max radius 4, zero-padded).
//
// Kernels are intentionally unoptimised (one launch per step, naive stencil)
// — that is the correctness baseline; performance work lands on top later.
//
// PML boundary convention: exact boundaries
//   pml_z0 = min(pml_width + 2*fd_pad, nz - fd_pad),
//   pml_z1 = max(pml_z0, nz - pml_width - 2*fd_pad),
// (and likewise for y, x). This differs from scalar2d.cu's
// `fd_pad + pml_width` convention: in the strip [fd_pad+pml_width,
// 2*fd_pad+pml_width) the CPML branch is evaluated with a = b = 0 at the
// cell but non-zero db and DIFF1(az*psiz) terms, which the 2D kernels
// omit (a latent ~1e-6 discrepancy that scalar2d's short-nt tests never
// reach). With these exact boundaries the result stays correct even when
// the wave has entered the PML.
//
// The adjoint pass uses wider backward boundaries
// ``pml_*_b = min(pml_width + 3*fd_pad, dim - fd_pad)``: the transpose of
// the forward PML stencil reads one fd_pad cell further into the interior,
// so the adjoint CPML branch must cover that strip too (without it, grad_v
// in the cells adjacent to the PML is wrong at the ~1e-3 level).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "storage.h"

#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------- finite-difference helpers (regular grid) ----------------
// Coefficient arrays come from Python with the regular-grid convention:
//   c1[4] = symmetric first-derivative coefficients for offsets 1..4
//   c2[5] = [center, offset1..offset4] second-derivative coefficients.
// The loops are #pragma unroll'ed over the fixed maximum radius 4; unused
// coefficients are exactly zero, so lower orders are reproduced exactly by
// the fixed-radius loops.
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
template <typename T>
__global__ void step_forward_interior_kernel(
    const T* __restrict__ v,
    const T* __restrict__ u_cur, const T* __restrict__ u_prev,
    T* __restrict__ u_new,
    T* __restrict__ w_store,
    const T* __restrict__ c1, const T* __restrict__ c2,
    T rdz, T rdy, T rdx, T rdz2, T rdy2, T rdx2,
    int t, int interval, T dt2,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int v_batched, int store, int64_t snap_off, int fd_pad)
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
            const T* vs = v + (long)s_v * nz * ny_nx;
            const long off = (long)s * nz * ny_nx + off_base;
            const T v_val = vs[off_base];
            const T v2dt2 = v_val * v_val * dt2;
            T w_sum = diff2_z(u_cur, off, c2, rdz2, ny_nx, fd_pad)
                    + diff2_y(u_cur, off, c2, rdy2, nx, fd_pad)
                    + diff2_x(u_cur, off, c2, rdx2, fd_pad);
            u_new[off] = v2dt2 * w_sum + (T)2 * u_cur[off] - u_prev[off];
            if (store && t % interval == 0) {
                const long soff = snap_off + off;
                w_store[soff] = (T)2 * v_val * dt2 * w_sum;
            }
        }
    }
}

// ---------------- forward: one time step (frame: includes PML borders) ----------------
// The per-dimension CPML update, evaluated for every cell outside the
// interior box [pml_z0, pml_z1) x [pml_y0, pml_y1) x [pml_x0, pml_x1).
template <typename T>
__global__ void step_forward_frame_kernel(
    const T* __restrict__ v,
    const T* __restrict__ u_cur, const T* __restrict__ u_prev,
    const T* __restrict__ psi_z, const T* __restrict__ psi_y, const T* __restrict__ psi_x,
    const T* __restrict__ zeta_z, const T* __restrict__ zeta_y, const T* __restrict__ zeta_x,
    T* __restrict__ u_new,
    T* __restrict__ psi_z_new, T* __restrict__ psi_y_new, T* __restrict__ psi_x_new,
    T* __restrict__ zeta_z_new, T* __restrict__ zeta_y_new, T* __restrict__ zeta_x_new,
    const T* __restrict__ az, const T* __restrict__ bz, const T* __restrict__ dbzdz,
    const T* __restrict__ ay, const T* __restrict__ by, const T* __restrict__ dbydy,
    const T* __restrict__ ax, const T* __restrict__ bx, const T* __restrict__ dbxdx,
    T* __restrict__ w_store,
    const T* __restrict__ c1, const T* __restrict__ c2,
    T rdz, T rdy, T rdx, T rdz2, T rdy2, T rdx2,
    int t, int interval, T dt2,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int v_batched, int store, int64_t snap_off, int fd_pad)
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
            const T* vs = v + (long)s_v * nz * ny_nx;
            const long off = (long)s * nz * ny_nx + off_base;
            const T v_val = vs[off_base];
            const T v2dt2 = v_val * v_val * dt2;
            T w_sum = (T)0;

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
            } else {
                w_sum += diff2_z(u_cur, off, c2, rdz2, ny_nx, fd_pad);
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
            } else {
                w_sum += diff2_y(u_cur, off, c2, rdy2, nx, fd_pad);
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
            } else {
                w_sum += diff2_x(u_cur, off, c2, rdx2, fd_pad);
            }

            u_new[off] = v2dt2 * w_sum + (T)2 * u_cur[off] - u_prev[off];
            if (store && t % interval == 0) {
                const long soff = snap_off + off;
                w_store[soff] = (T)2 * v_val * dt2 * w_sum;
            }
        }
    }
}

// ---------------- forward: source injection / receiver recording ----------------
// (flat indices are precomputed on the padded grid, so the kernels only need
//  the per-shot field stride nz*ny*nx and the row-major flat index.)
template <typename T>
__global__ void inject_kernel(
    T* __restrict__ u_new, const T* __restrict__ f, const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long nz_ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        if (idx >= 0)
            u_new[(long)s * nz_ny_nx + idx] += f[(((long)t * n_shots + s) * n_src + k)];
    }
}

template <typename T>
__global__ void record_kernel(
    const T* __restrict__ u_cur, T* __restrict__ r, const long* __restrict__ rec_i,
    int t, int n_shots, int n_rec, long nz_ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            r[(((long)t * n_shots + s) * n_rec + k)] = u_cur[(long)s * nz_ny_nx + idx];
    }
}

// ---------------- backward: adjoint step (interior: no PML) ----------------
// The symmetric second-derivative stencil is self-adjoint; it acts here on
// q(dz) = v2dt2[z+dz] * lam_next[z+dz] (transpose of DIFFZ2(V2DT2_WFC)).
template <typename T>
__global__ void step_adjoint_interior_kernel(
    const T* __restrict__ v,
    const T* __restrict__ lam_next, const T* __restrict__ lam_next2,
    T* __restrict__ lam_new,
    const T* __restrict__ w_store, T* __restrict__ grad_v,
    const T* __restrict__ c1, const T* __restrict__ c2,
    T rdz, T rdy, T rdx, T rdz2, T rdy2, T rdx2,
    int t, int interval, T scale, T dt2,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int v_batched, int64_t snap_off, int fd_pad)
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
            const T* vs = v + (long)s_v * nz * ny_nx;
            const long off = (long)s * nz * ny_nx + off_base;
            const T v_val = vs[off_base];
            const T v2dt2 = v_val * v_val * dt2;
            T wz = c2[0] * (v2dt2 * lam_next[off]);
#pragma unroll
            for (int k = 1; k <= fd_pad; ++k)
                wz += c2[k] * (vs[off_base + k * ny_nx] * vs[off_base + k * ny_nx] * dt2 * lam_next[off + k * ny_nx]
                             + vs[off_base - k * ny_nx] * vs[off_base - k * ny_nx] * dt2 * lam_next[off - k * ny_nx]);
            wz *= rdz2;
            T wy = c2[0] * (v2dt2 * lam_next[off]);
#pragma unroll
            for (int k = 1; k <= fd_pad; ++k)
                wy += c2[k] * (vs[off_base + k * nx] * vs[off_base + k * nx] * dt2 * lam_next[off + k * nx]
                             + vs[off_base - k * nx] * vs[off_base - k * nx] * dt2 * lam_next[off - k * nx]);
            wy *= rdy2;
            T wx = c2[0] * (v2dt2 * lam_next[off]);
#pragma unroll
            for (int k = 1; k <= fd_pad; ++k)
                wx += c2[k] * (vs[off_base + k] * vs[off_base + k] * dt2 * lam_next[off + k]
                             + vs[off_base - k] * vs[off_base - k] * dt2 * lam_next[off - k]);
            wx *= rdx2;
            lam_new[off] = (T)2 * lam_next[off] + wz + wy + wx - lam_next2[off];
            if (t % interval == 0) {
                const long soff = snap_off + off;
                grad_v[off] += lam_next[off] * w_store[soff] * scale;
            }
        }
    }
}

// ---------------- backward: adjoint step (frame: includes PML borders) ----------------
// Exact discrete transpose of the forward frame kernel: with (z-dimension,
// analogously for y and x)
//   T1(dz) = dbzdz[z+dz]*((1+bz[z+dz])*V2DT2_WFC(dz) + bz[z+dz]*ZETAZ(dz))
//            + bz[z+dz]*PSIZ(dz)
//   T2(dz) = (1+bz[z+dz])*((1+bz[z+dz])*V2DT2_WFC(dz) + bz[z+dz]*ZETAZ(dz))
// the z PML contribution is -DIFFZ1(T1) + DIFFZ2(T2), and
//   PSIZ_TERM(dz) = (1+bz[z+dz])*V2DT2_WFC(dz) + bz[z+dz]*ZETAZ(dz)
//   psiz_new = -az[z]*DIFFZ1(PSIZ_TERM) + az[z]*psiz
//   zetaz_new = az[z]*V2DT2_WFC(0) + az[z]*zetaz
template <typename T>
__global__ void step_adjoint_frame_kernel(
    const T* __restrict__ v,
    const T* __restrict__ lam_next, const T* __restrict__ lam_next2,
    T* __restrict__ lam_new,
    const T* __restrict__ psi_z, const T* __restrict__ psi_y, const T* __restrict__ psi_x,
    const T* __restrict__ zeta_z, const T* __restrict__ zeta_y, const T* __restrict__ zeta_x,
    T* __restrict__ psi_z_new, T* __restrict__ psi_y_new, T* __restrict__ psi_x_new,
    T* __restrict__ zeta_z_new, T* __restrict__ zeta_y_new, T* __restrict__ zeta_x_new,
    const T* __restrict__ az, const T* __restrict__ bz, const T* __restrict__ dbzdz,
    const T* __restrict__ ay, const T* __restrict__ by, const T* __restrict__ dbydy,
    const T* __restrict__ ax, const T* __restrict__ bx, const T* __restrict__ dbxdx,
    const T* __restrict__ w_store, T* __restrict__ grad_v,
    const T* __restrict__ c1, const T* __restrict__ c2,
    T rdz, T rdy, T rdx, T rdz2, T rdy2, T rdx2,
    int t, int interval, T scale, T dt2,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int v_batched, int64_t snap_off, int fd_pad)
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
            const T* vs = v + (long)s_v * nz * ny_nx;
            const long off = (long)s * nz * ny_nx + off_base;
            const T v_val = vs[off_base];
            const T v2dt2 = v_val * v_val * dt2;
            T w_sum = (T)0;

            // z: transpose of the CPML-modified Laplacian acting on lam_next
            if (z < pml_z0 || z >= pml_z1) {
                T t1_sum = (T)0;
                T t2_sum = c2[0] * (((T)1 + bz[z]) * (((T)1 + bz[z]) * v2dt2 * lam_next[off] + bz[z] * zeta_z[off]));
                T p_sum = (T)0;
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k) {
                    const T t1_p = dbzdz[z + k] * (((T)1 + bz[z + k]) * vs[off_base + k * ny_nx]
                            * vs[off_base + k * ny_nx] * dt2 * lam_next[off + k * ny_nx] + bz[z + k] * zeta_z[off + k * ny_nx])
                        + bz[z + k] * psi_z[off + k * ny_nx];
                    const T t1_m = dbzdz[z - k] * (((T)1 + bz[z - k]) * vs[off_base - k * ny_nx]
                            * vs[off_base - k * ny_nx] * dt2 * lam_next[off - k * ny_nx] + bz[z - k] * zeta_z[off - k * ny_nx])
                        + bz[z - k] * psi_z[off - k * ny_nx];
                    t1_sum += c1[k - 1] * (t1_p - t1_m);
                    const T t2_p = ((T)1 + bz[z + k]) * (((T)1 + bz[z + k]) * vs[off_base + k * ny_nx]
                            * vs[off_base + k * ny_nx] * dt2 * lam_next[off + k * ny_nx] + bz[z + k] * zeta_z[off + k * ny_nx]);
                    const T t2_m = ((T)1 + bz[z - k]) * (((T)1 + bz[z - k]) * vs[off_base - k * ny_nx]
                            * vs[off_base - k * ny_nx] * dt2 * lam_next[off - k * ny_nx] + bz[z - k] * zeta_z[off - k * ny_nx]);
                    t2_sum += c2[k] * (t2_p + t2_m);
                    const T p_p = ((T)1 + bz[z + k]) * vs[off_base + k * ny_nx] * vs[off_base + k * ny_nx]
                            * dt2 * lam_next[off + k * ny_nx] + bz[z + k] * zeta_z[off + k * ny_nx];
                    const T p_m = ((T)1 + bz[z - k]) * vs[off_base - k * ny_nx] * vs[off_base - k * ny_nx]
                            * dt2 * lam_next[off - k * ny_nx] + bz[z - k] * zeta_z[off - k * ny_nx];
                    p_sum += c1[k - 1] * (p_p - p_m);
                }
                w_sum += -t1_sum * rdz + t2_sum * rdz2;
                psi_z_new[off] = -az[z] * p_sum * rdz + az[z] * psi_z[off];
                zeta_z_new[off] = az[z] * v2dt2 * lam_next[off] + az[z] * zeta_z[off];
            } else {
                T wz = c2[0] * (v2dt2 * lam_next[off]);
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k)
                    wz += c2[k] * (vs[off_base + k * ny_nx] * vs[off_base + k * ny_nx] * dt2 * lam_next[off + k * ny_nx]
                                 + vs[off_base - k * ny_nx] * vs[off_base - k * ny_nx] * dt2 * lam_next[off - k * ny_nx]);
                w_sum += wz * rdz2;
            }
            // y: same structure
            if (y < pml_y0 || y >= pml_y1) {
                T t1_sum = (T)0;
                T t2_sum = c2[0] * (((T)1 + by[y]) * (((T)1 + by[y]) * v2dt2 * lam_next[off] + by[y] * zeta_y[off]));
                T p_sum = (T)0;
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k) {
                    const T t1_p = dbydy[y + k] * (((T)1 + by[y + k]) * vs[off_base + k * nx]
                            * vs[off_base + k * nx] * dt2 * lam_next[off + k * nx] + by[y + k] * zeta_y[off + k * nx])
                        + by[y + k] * psi_y[off + k * nx];
                    const T t1_m = dbydy[y - k] * (((T)1 + by[y - k]) * vs[off_base - k * nx]
                            * vs[off_base - k * nx] * dt2 * lam_next[off - k * nx] + by[y - k] * zeta_y[off - k * nx])
                        + by[y - k] * psi_y[off - k * nx];
                    t1_sum += c1[k - 1] * (t1_p - t1_m);
                    const T t2_p = ((T)1 + by[y + k]) * (((T)1 + by[y + k]) * vs[off_base + k * nx]
                            * vs[off_base + k * nx] * dt2 * lam_next[off + k * nx] + by[y + k] * zeta_y[off + k * nx]);
                    const T t2_m = ((T)1 + by[y - k]) * (((T)1 + by[y - k]) * vs[off_base - k * nx]
                            * vs[off_base - k * nx] * dt2 * lam_next[off - k * nx] + by[y - k] * zeta_y[off - k * nx]);
                    t2_sum += c2[k] * (t2_p + t2_m);
                    const T p_p = ((T)1 + by[y + k]) * vs[off_base + k * nx] * vs[off_base + k * nx]
                            * dt2 * lam_next[off + k * nx] + by[y + k] * zeta_y[off + k * nx];
                    const T p_m = ((T)1 + by[y - k]) * vs[off_base - k * nx] * vs[off_base - k * nx]
                            * dt2 * lam_next[off - k * nx] + by[y - k] * zeta_y[off - k * nx];
                    p_sum += c1[k - 1] * (p_p - p_m);
                }
                w_sum += -t1_sum * rdy + t2_sum * rdy2;
                psi_y_new[off] = -ay[y] * p_sum * rdy + ay[y] * psi_y[off];
                zeta_y_new[off] = ay[y] * v2dt2 * lam_next[off] + ay[y] * zeta_y[off];
            } else {
                T wy = c2[0] * (v2dt2 * lam_next[off]);
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k)
                    wy += c2[k] * (vs[off_base + k * nx] * vs[off_base + k * nx] * dt2 * lam_next[off + k * nx]
                                 + vs[off_base - k * nx] * vs[off_base - k * nx] * dt2 * lam_next[off - k * nx]);
                w_sum += wy * rdy2;
            }
            // x: same structure
            if (x < pml_x0 || x >= pml_x1) {
                T t1_sum = (T)0;
                T t2_sum = c2[0] * (((T)1 + bx[x]) * (((T)1 + bx[x]) * v2dt2 * lam_next[off] + bx[x] * zeta_x[off]));
                T p_sum = (T)0;
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k) {
                    const T t1_p = dbxdx[x + k] * (((T)1 + bx[x + k]) * vs[off_base + k]
                            * vs[off_base + k] * dt2 * lam_next[off + k] + bx[x + k] * zeta_x[off + k])
                        + bx[x + k] * psi_x[off + k];
                    const T t1_m = dbxdx[x - k] * (((T)1 + bx[x - k]) * vs[off_base - k]
                            * vs[off_base - k] * dt2 * lam_next[off - k] + bx[x - k] * zeta_x[off - k])
                        + bx[x - k] * psi_x[off - k];
                    t1_sum += c1[k - 1] * (t1_p - t1_m);
                    const T t2_p = ((T)1 + bx[x + k]) * (((T)1 + bx[x + k]) * vs[off_base + k]
                            * vs[off_base + k] * dt2 * lam_next[off + k] + bx[x + k] * zeta_x[off + k]);
                    const T t2_m = ((T)1 + bx[x - k]) * (((T)1 + bx[x - k]) * vs[off_base - k]
                            * vs[off_base - k] * dt2 * lam_next[off - k] + bx[x - k] * zeta_x[off - k]);
                    t2_sum += c2[k] * (t2_p + t2_m);
                    const T p_p = ((T)1 + bx[x + k]) * vs[off_base + k] * vs[off_base + k]
                            * dt2 * lam_next[off + k] + bx[x + k] * zeta_x[off + k];
                    const T p_m = ((T)1 + bx[x - k]) * vs[off_base - k] * vs[off_base - k]
                            * dt2 * lam_next[off - k] + bx[x - k] * zeta_x[off - k];
                    p_sum += c1[k - 1] * (p_p - p_m);
                }
                w_sum += -t1_sum * rdx + t2_sum * rdx2;
                psi_x_new[off] = -ax[x] * p_sum * rdx + ax[x] * psi_x[off];
                zeta_x_new[off] = ax[x] * v2dt2 * lam_next[off] + ax[x] * zeta_x[off];
            } else {
                T wx = c2[0] * (v2dt2 * lam_next[off]);
#pragma unroll
                for (int k = 1; k <= fd_pad; ++k)
                    wx += c2[k] * (vs[off_base + k] * vs[off_base + k] * dt2 * lam_next[off + k]
                                 + vs[off_base - k] * vs[off_base - k] * dt2 * lam_next[off - k]);
                w_sum += wx * rdx2;
            }

            lam_new[off] = (T)2 * lam_next[off] + w_sum - lam_next2[off];
            if (t % interval == 0) {
                const long soff = snap_off + off;
                grad_v[off] += lam_next[off] * w_store[soff] * scale;
            }
        }
    }
}

template <typename T>
__global__ void record_grad_f_kernel(
    const T* __restrict__ lam_next, T* __restrict__ grad_f, const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long nz_ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        grad_f[(((long)t * n_shots + s) * n_src + k)] = (idx >= 0) ? lam_next[(long)s * nz_ny_nx + idx] : (T)0;
    }
}

template <typename T>
__global__ void record_grad_r_kernel(
    T* __restrict__ lam_cur, const T* __restrict__ grad_r, const long* __restrict__ rec_i,
    int t, int n_shots, int n_rec, long nz_ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            lam_cur[(long)s * nz_ny_nx + idx] += grad_r[(((long)t * n_shots + s) * n_rec + k)];
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
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami scalar3d interior kernel failed"); \
    }

#define LAUNCH_STEP_FRAME(KERN, T, ...)                                        \
    {                                                                          \
        dim3 block(16, 16, 1);                                                 \
        dim3 grid((nx - 2 * fd_pad + 15) / 16, (ny - 2 * fd_pad + 15) / 16,    \
                  nz - 2 * fd_pad);                                            \
        if ((nz - 2 * fd_pad) > 0 && (ny - 2 * fd_pad) > 0 &&                  \
            (nx - 2 * fd_pad) > 0)                                             \
            KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami scalar3d frame kernel failed"); \
    }

void forward_step(
    torch::Tensor v,
    torch::Tensor u_cur, torch::Tensor u_prev,
    torch::Tensor psi_z, torch::Tensor psi_y, torch::Tensor psi_x,
    torch::Tensor zeta_z, torch::Tensor zeta_y, torch::Tensor zeta_x,
    torch::Tensor u_new,
    torch::Tensor psi_z_new, torch::Tensor psi_y_new, torch::Tensor psi_x_new,
    torch::Tensor zeta_z_new, torch::Tensor zeta_y_new, torch::Tensor zeta_x_new,
    torch::Tensor az, torch::Tensor bz, torch::Tensor dbzdz,
    torch::Tensor ay, torch::Tensor by, torch::Tensor dbydy,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor dbxdx,
    torch::Tensor w_store,
    torch::Tensor c1, torch::Tensor c2,
    double rdz, double rdy, double rdx, double rdz2, double rdy2, double rdx2,
    int64_t t, int64_t interval, double dt2,
    int64_t pml_z0, int64_t pml_z1, int64_t pml_y0, int64_t pml_y1,
    int64_t pml_x0, int64_t pml_x1,
    int64_t v_batched, int64_t store, int64_t snap_off, int64_t fd_pad)
{
    const int n_shots = v.size(0), nz = v.size(1), ny = v.size(2), nx = v.size(3);
    CHECK_CONTIG(c1);
    CHECK_CONTIG(c2);
    if (v.scalar_type() == torch::kFloat32) {
        LAUNCH_STEP_INTERIOR(step_forward_interior_kernel, float,
            v.data_ptr<float>(), u_cur.data_ptr<float>(), u_prev.data_ptr<float>(),
            u_new.data_ptr<float>(), w_store.data_ptr<float>(),
            c1.data_ptr<float>(), c2.data_ptr<float>(),
            (float)rdz, (float)rdy, (float)rdx, (float)rdz2, (float)rdy2, (float)rdx2,
            (int)t, (int)interval, (float)dt2,
            n_shots, nz, ny, nx, (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1, (int)v_batched, (int)store, (int64_t)snap_off, (int)fd_pad);
        LAUNCH_STEP_FRAME(step_forward_frame_kernel, float,
            v.data_ptr<float>(), u_cur.data_ptr<float>(), u_prev.data_ptr<float>(),
            psi_z.data_ptr<float>(), psi_y.data_ptr<float>(), psi_x.data_ptr<float>(),
            zeta_z.data_ptr<float>(), zeta_y.data_ptr<float>(), zeta_x.data_ptr<float>(),
            u_new.data_ptr<float>(),
            psi_z_new.data_ptr<float>(), psi_y_new.data_ptr<float>(), psi_x_new.data_ptr<float>(),
            zeta_z_new.data_ptr<float>(), zeta_y_new.data_ptr<float>(), zeta_x_new.data_ptr<float>(),
            az.data_ptr<float>(), bz.data_ptr<float>(), dbzdz.data_ptr<float>(),
            ay.data_ptr<float>(), by.data_ptr<float>(), dbydy.data_ptr<float>(),
            ax.data_ptr<float>(), bx.data_ptr<float>(), dbxdx.data_ptr<float>(),
            w_store.data_ptr<float>(),
            c1.data_ptr<float>(), c2.data_ptr<float>(),
            (float)rdz, (float)rdy, (float)rdx, (float)rdz2, (float)rdy2, (float)rdx2,
            (int)t, (int)interval, (float)dt2,
            n_shots, nz, ny, nx, (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1, (int)v_batched, (int)store, (int64_t)snap_off, (int)fd_pad);
    } else {
        LAUNCH_STEP_INTERIOR(step_forward_interior_kernel, double,
            v.data_ptr<double>(), u_cur.data_ptr<double>(), u_prev.data_ptr<double>(),
            u_new.data_ptr<double>(), w_store.data_ptr<double>(),
            c1.data_ptr<double>(), c2.data_ptr<double>(),
            rdz, rdy, rdx, rdz2, rdy2, rdx2,
            (int)t, (int)interval, dt2,
            n_shots, nz, ny, nx, (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1, (int)v_batched, (int)store, (int64_t)snap_off, (int)fd_pad);
        LAUNCH_STEP_FRAME(step_forward_frame_kernel, double,
            v.data_ptr<double>(), u_cur.data_ptr<double>(), u_prev.data_ptr<double>(),
            psi_z.data_ptr<double>(), psi_y.data_ptr<double>(), psi_x.data_ptr<double>(),
            zeta_z.data_ptr<double>(), zeta_y.data_ptr<double>(), zeta_x.data_ptr<double>(),
            u_new.data_ptr<double>(),
            psi_z_new.data_ptr<double>(), psi_y_new.data_ptr<double>(), psi_x_new.data_ptr<double>(),
            zeta_z_new.data_ptr<double>(), zeta_y_new.data_ptr<double>(), zeta_x_new.data_ptr<double>(),
            az.data_ptr<double>(), bz.data_ptr<double>(), dbzdz.data_ptr<double>(),
            ay.data_ptr<double>(), by.data_ptr<double>(), dbydy.data_ptr<double>(),
            ax.data_ptr<double>(), bx.data_ptr<double>(), dbxdx.data_ptr<double>(),
            w_store.data_ptr<double>(),
            c1.data_ptr<double>(), c2.data_ptr<double>(),
            rdz, rdy, rdx, rdz2, rdy2, rdx2,
            (int)t, (int)interval, dt2,
            n_shots, nz, ny, nx, (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1, (int)v_batched, (int)store, (int64_t)snap_off, (int)fd_pad);
    }
}

void adjoint_step(
    torch::Tensor v,
    torch::Tensor lam_next, torch::Tensor lam_next2, torch::Tensor lam_new,
    torch::Tensor psi_z, torch::Tensor psi_y, torch::Tensor psi_x,
    torch::Tensor zeta_z, torch::Tensor zeta_y, torch::Tensor zeta_x,
    torch::Tensor psi_z_new, torch::Tensor psi_y_new, torch::Tensor psi_x_new,
    torch::Tensor zeta_z_new, torch::Tensor zeta_y_new, torch::Tensor zeta_x_new,
    torch::Tensor az, torch::Tensor bz, torch::Tensor dbzdz,
    torch::Tensor ay, torch::Tensor by, torch::Tensor dbydy,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor dbxdx,
    torch::Tensor w_store, torch::Tensor grad_v,
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
    int64_t v_batched, int64_t snap_off, int64_t fd_pad)
{
    const int n_shots = v.size(0), nz = v.size(1), ny = v.size(2), nx = v.size(3);
    CHECK_CONTIG(c1);
    CHECK_CONTIG(c2);
    if (v.scalar_type() == torch::kFloat32) {
        LAUNCH_STEP_INTERIOR(step_adjoint_interior_kernel, float,
            v.data_ptr<float>(), lam_next.data_ptr<float>(), lam_next2.data_ptr<float>(), lam_new.data_ptr<float>(),
            w_store.data_ptr<float>(), grad_v.data_ptr<float>(),
            c1.data_ptr<float>(), c2.data_ptr<float>(),
            (float)rdz, (float)rdy, (float)rdx, (float)rdz2, (float)rdy2, (float)rdx2,
            (int)t, (int)interval, (float)scale, (float)dt2,
            n_shots, nz, ny, nx, (int)pml_z0_b, (int)pml_z1_b, (int)pml_y0_b, (int)pml_y1_b,
            (int)pml_x0_b, (int)pml_x1_b, (int)v_batched, (int64_t)snap_off, (int)fd_pad);
        LAUNCH_STEP_FRAME(step_adjoint_frame_kernel, float,
            v.data_ptr<float>(), lam_next.data_ptr<float>(), lam_next2.data_ptr<float>(), lam_new.data_ptr<float>(),
            psi_z.data_ptr<float>(), psi_y.data_ptr<float>(), psi_x.data_ptr<float>(),
            zeta_z.data_ptr<float>(), zeta_y.data_ptr<float>(), zeta_x.data_ptr<float>(),
            psi_z_new.data_ptr<float>(), psi_y_new.data_ptr<float>(), psi_x_new.data_ptr<float>(),
            zeta_z_new.data_ptr<float>(), zeta_y_new.data_ptr<float>(), zeta_x_new.data_ptr<float>(),
            az.data_ptr<float>(), bz.data_ptr<float>(), dbzdz.data_ptr<float>(),
            ay.data_ptr<float>(), by.data_ptr<float>(), dbydy.data_ptr<float>(),
            ax.data_ptr<float>(), bx.data_ptr<float>(), dbxdx.data_ptr<float>(),
            w_store.data_ptr<float>(), grad_v.data_ptr<float>(),
            c1.data_ptr<float>(), c2.data_ptr<float>(),
            (float)rdz, (float)rdy, (float)rdx, (float)rdz2, (float)rdy2, (float)rdx2,
            (int)t, (int)interval, (float)scale, (float)dt2,
            n_shots, nz, ny, nx, (int)pml_z0_b, (int)pml_z1_b, (int)pml_y0_b, (int)pml_y1_b,
            (int)pml_x0_b, (int)pml_x1_b, (int)v_batched, (int64_t)snap_off, (int)fd_pad);
    } else {
        LAUNCH_STEP_INTERIOR(step_adjoint_interior_kernel, double,
            v.data_ptr<double>(), lam_next.data_ptr<double>(), lam_next2.data_ptr<double>(), lam_new.data_ptr<double>(),
            w_store.data_ptr<double>(), grad_v.data_ptr<double>(),
            c1.data_ptr<double>(), c2.data_ptr<double>(),
            rdz, rdy, rdx, rdz2, rdy2, rdx2,
            (int)t, (int)interval, scale, dt2,
            n_shots, nz, ny, nx, (int)pml_z0_b, (int)pml_z1_b, (int)pml_y0_b, (int)pml_y1_b,
            (int)pml_x0_b, (int)pml_x1_b, (int)v_batched, (int64_t)snap_off, (int)fd_pad);
        LAUNCH_STEP_FRAME(step_adjoint_frame_kernel, double,
            v.data_ptr<double>(), lam_next.data_ptr<double>(), lam_next2.data_ptr<double>(), lam_new.data_ptr<double>(),
            psi_z.data_ptr<double>(), psi_y.data_ptr<double>(), psi_x.data_ptr<double>(),
            zeta_z.data_ptr<double>(), zeta_y.data_ptr<double>(), zeta_x.data_ptr<double>(),
            psi_z_new.data_ptr<double>(), psi_y_new.data_ptr<double>(), psi_x_new.data_ptr<double>(),
            zeta_z_new.data_ptr<double>(), zeta_y_new.data_ptr<double>(), zeta_x_new.data_ptr<double>(),
            az.data_ptr<double>(), bz.data_ptr<double>(), dbzdz.data_ptr<double>(),
            ay.data_ptr<double>(), by.data_ptr<double>(), dbydy.data_ptr<double>(),
            ax.data_ptr<double>(), bx.data_ptr<double>(), dbxdx.data_ptr<double>(),
            w_store.data_ptr<double>(), grad_v.data_ptr<double>(),
            c1.data_ptr<double>(), c2.data_ptr<double>(),
            rdz, rdy, rdx, rdz2, rdy2, rdx2,
            (int)t, (int)interval, scale, dt2,
            n_shots, nz, ny, nx, (int)pml_z0_b, (int)pml_z1_b, (int)pml_y0_b, (int)pml_y1_b,
            (int)pml_x0_b, (int)pml_x1_b, (int)v_batched, (int64_t)snap_off, (int)fd_pad);
    }
}

void inject(torch::Tensor u_new, torch::Tensor f, torch::Tensor src_i,
            int64_t t, int64_t n_shots, int64_t n_src, int64_t nz_ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_src + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(u_new.scalar_type(), "inject", [&] {
        inject_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            u_new.data_ptr<scalar_t>(), f.data_ptr<scalar_t>(), src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)nz_ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami inject failed");
}

void record(torch::Tensor u_cur, torch::Tensor r, torch::Tensor rec_i,
            int64_t t, int64_t n_shots, int64_t n_rec, int64_t nz_ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_rec + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(u_cur.scalar_type(), "record", [&] {
        record_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            u_cur.data_ptr<scalar_t>(), r.data_ptr<scalar_t>(), rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)nz_ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami record failed");
}

void record_grad_f(torch::Tensor lam_next, torch::Tensor grad_f, torch::Tensor src_i,
                   int64_t t, int64_t n_shots, int64_t n_src, int64_t nz_ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_src + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(lam_next.scalar_type(), "record_grad_f", [&] {
        record_grad_f_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            lam_next.data_ptr<scalar_t>(), grad_f.data_ptr<scalar_t>(), src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)nz_ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami record_grad_f failed");
}

void record_grad_r(torch::Tensor lam_cur, torch::Tensor grad_r, torch::Tensor rec_i,
                   int64_t t, int64_t n_shots, int64_t n_rec, int64_t nz_ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_rec + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(lam_cur.scalar_type(), "record_grad_r", [&] {
        record_grad_r_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            lam_cur.data_ptr<scalar_t>(), grad_r.data_ptr<scalar_t>(), rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)nz_ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami record_grad_r failed");
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
