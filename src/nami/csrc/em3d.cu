/*
 * nami em3d: 3D electromagnetic FDTD (staggered Yee grid, C-PML) with a
 * native exact-transpose CUDA adjoint.
 *
 * Forward/adjoint discretisation follows nami's em2d_tm, generalised to the
 * full 3D Maxwell equations with staggered Yee-grid conventions:
 *
 *     per step: H half-step (Hx/Hy/Hz from curls of E with the half-integer
 *     CPML memory), E integer-step (snapshotting the pre-update Ex/Ey/Ez and
 *     the PML-modified curls for the adjoint), source injection (pre-scaled
 *     by ``cb * -1/(dx dy dz)`` into the source component), receiver
 *     recording (post-injection, from the receiver component).
 *
 * The backward pass is the exact discrete transpose: ``record_grad_r`` /
 * ``record_grad_f`` at the start of each adjoint step, ``coeff_grad`` and
 * ``cq_grad`` model-gradient accumulators on the snapshot interval, the
 * two-stage E/H transpose kernels with the time-reversed CPML memory
 * recursions, and the six-field generalisation of em2d_tm's stage-2
 * divergence transposes.
 *
 * Kernels are intentionally unoptimised (one flat launch per step, naive
 * stencil) -- the correctness baseline; performance work lands on top later.
 *
 * The spatial FD order is user-selectable (accuracy 2/4/6/8) with the
 * ``fd.STAGGERED_DIFF1`` coefficient tables; kernels are driven by the
 * zero-padded coefficient array (max radius 4) and per-side FD padding
 * ``fd_pad = [accuracy // 2, accuracy // 2 - 1] * 3``.  FD_PAD is a
 * COMPILE-TIME constant (== accuracy // 2, one radius for every direction:
 * fd_pad_z0 == fd_pad_y0 == fd_pad_x0): the stencil loops fully unroll with
 * constant offsets, which measurably beats a runtime fd_pad loop.
 * Coefficients and arithmetic are unchanged from the runtime-radius kernels,
 * so results are bitwise identical.  Snapshots (six streams: pre-update
 * Ex/Ey/Ez and PML-modified curls) live in C++-owned GPU-resident storage
 * (see storage.h); memory is controlled from the Python front end via
 * checkpointing of the full wavefield state.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <pybind11/stl.h>
#include "storage.h"

#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------- coefficient-driven staggered difference operators ----------------
// Same conventions as em2d_tm.cu: ``diff_int_*`` evaluates a derivative at a
// half-integer grid point of an integer-stored field (H step), ``diff_half_*``
// at an integer grid point of a half-integer-stored field (E step curl); the
// two are exact discrete transposes of each other up to the curl sign.
//
// FD_PAD is a COMPILE-TIME constant (accuracy // 2, one radius for every
// direction: fd_pad_z0 == fd_pad_y0 == fd_pad_x0): the loops fully unroll
// with constant offsets, which measurably beats a runtime fd_pad loop.  ``c``
// is the zero-padded coefficient array (max radius 4); coefficients and
// arithmetic order are unchanged from the runtime-radius kernels, so results
// are bitwise identical.
template <typename T, int FD_PAD>
__device__ __forceinline__ T diff_int_x(const T* __restrict__ u, long off,
                                        const T* __restrict__ c, T rdx)
{
    T d = (T)0;
#pragma unroll
    for (int k = 1; k <= FD_PAD; ++k)
        d += c[k - 1] * (u[off + k] - u[off - k + 1]);
    return d * rdx;
}

template <typename T, int FD_PAD>
__device__ __forceinline__ T diff_int_y(const T* __restrict__ u, long off,
                                        const T* __restrict__ c, T rdy, int nx)
{
    T d = (T)0;
#pragma unroll
    for (int k = 1; k <= FD_PAD; ++k)
        d += c[k - 1] * (u[off + k * nx] - u[off - (k - 1) * nx]);
    return d * rdy;
}

template <typename T, int FD_PAD>
__device__ __forceinline__ T diff_int_z(const T* __restrict__ u, long off,
                                        const T* __restrict__ c, T rdz, long ny_nx)
{
    T d = (T)0;
#pragma unroll
    for (int k = 1; k <= FD_PAD; ++k)
        d += c[k - 1] * (u[off + k * ny_nx] - u[off - (k - 1) * ny_nx]);
    return d * rdz;
}

template <typename T, int FD_PAD>
__device__ __forceinline__ T diff_half_x(const T* __restrict__ u, long off,
                                         const T* __restrict__ c, T rdx)
{
    T d = (T)0;
#pragma unroll
    for (int k = 1; k <= FD_PAD; ++k)
        d += c[k - 1] * (u[off + k - 1] - u[off - k]);
    return d * rdx;
}

template <typename T, int FD_PAD>
__device__ __forceinline__ T diff_half_y(const T* __restrict__ u, long off,
                                         const T* __restrict__ c, T rdy, int nx)
{
    T d = (T)0;
#pragma unroll
    for (int k = 1; k <= FD_PAD; ++k)
        d += c[k - 1] * (u[off + (k - 1) * nx] - u[off - k * nx]);
    return d * rdy;
}

template <typename T, int FD_PAD>
__device__ __forceinline__ T diff_half_z(const T* __restrict__ u, long off,
                                         const T* __restrict__ c, T rdz, long ny_nx)
{
    T d = (T)0;
#pragma unroll
    for (int k = 1; k <= FD_PAD; ++k)
        d += c[k - 1] * (u[off + (k - 1) * ny_nx] - u[off - k * ny_nx]);
    return d * rdz;
}

// ---------------- forward: H half-step ----------------
// Hx/Hy/Hz are updated from curls of Ex/Ey/Ez with integer-point staggered
// differences (E lives at integer grid points, H at half-integer points); the
// six CPML memory variables use the half-integer profiles, indexed by the
// derivative dimension.  Boundary rows/cols get a zero gradient (the diff is
// only evaluated where the stencil fits), mirroring em2d_tm exactly.
template <typename T, int FD_PAD>
__global__ void step_h_kernel(
    const T* __restrict__ cq,
    const T* __restrict__ ex, const T* __restrict__ ey, const T* __restrict__ ez,
    T* __restrict__ hx, T* __restrict__ hy, T* __restrict__ hz,
    T* __restrict__ m_ey_z, T* __restrict__ m_ez_y, T* __restrict__ m_ez_x,
    T* __restrict__ m_ex_z, T* __restrict__ m_ex_y, T* __restrict__ m_ey_x,
    const T* __restrict__ azh, const T* __restrict__ bzh,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ kzh, const T* __restrict__ kyh, const T* __restrict__ kxh,
    const T* __restrict__ c,
    T rdz, T rdy, T rdx,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int cq_batched)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)n_shots * nz * ny * nx;
    if (i >= total)
        return;
    const long ny_nx = (long)ny * nx;
    const int x = (int)(i % nx);
    const int y = (int)((i / nx) % ny);
    const int z = (int)((i / ((long)nx * ny)) % nz);  // (i / nx) / ny % nz, avoids overflow
    const int s = (int)(i / ((long)nx * ny * nz));
    const long j = (long)z * ny_nx + (long)y * nx + x;
    const T cq_val = cq_batched ? cq[i] : cq[j];

    int pml_z1h = pml_z1 - 1;
    if (pml_z1 <= pml_z0)
        pml_z1h = pml_z0;
    int pml_y1h = pml_y1 - 1;
    if (pml_y1 <= pml_y0)
        pml_y1h = pml_y0;
    int pml_x1h = pml_x1 - 1;
    if (pml_x1 <= pml_x0)
        pml_x1h = pml_x0;

    T dEy_dz = (T)0;
    if (z >= FD_PAD && z < nz - FD_PAD)
        dEy_dz = diff_int_z<T, FD_PAD>(ey, i, c, rdz, ny_nx);
    T dEz_dy = (T)0;
    if (y >= FD_PAD && y < ny - FD_PAD)
        dEz_dy = diff_int_y<T, FD_PAD>(ez, i, c, rdy, nx);
    T dEz_dx = (T)0;
    if (x >= FD_PAD && x < nx - FD_PAD)
        dEz_dx = diff_int_x<T, FD_PAD>(ez, i, c, rdx);
    T dEx_dz = (T)0;
    if (z >= FD_PAD && z < nz - FD_PAD)
        dEx_dz = diff_int_z<T, FD_PAD>(ex, i, c, rdz, ny_nx);
    T dEx_dy = (T)0;
    if (y >= FD_PAD && y < ny - FD_PAD)
        dEx_dy = diff_int_y<T, FD_PAD>(ex, i, c, rdy, nx);
    T dEy_dx = (T)0;
    if (x >= FD_PAD && x < nx - FD_PAD)
        dEy_dx = diff_int_x<T, FD_PAD>(ey, i, c, rdx);

    if (z < pml_z0 || z >= pml_z1h) {
        m_ey_z[i] = bzh[z] * m_ey_z[i] + azh[z] * dEy_dz;
        dEy_dz = dEy_dz / kzh[z] + m_ey_z[i];
        m_ex_z[i] = bzh[z] * m_ex_z[i] + azh[z] * dEx_dz;
        dEx_dz = dEx_dz / kzh[z] + m_ex_z[i];
    }
    if (y < pml_y0 || y >= pml_y1h) {
        m_ez_y[i] = byh[y] * m_ez_y[i] + ayh[y] * dEz_dy;
        dEz_dy = dEz_dy / kyh[y] + m_ez_y[i];
        m_ex_y[i] = byh[y] * m_ex_y[i] + ayh[y] * dEx_dy;
        dEx_dy = dEx_dy / kyh[y] + m_ex_y[i];
    }
    if (x < pml_x0 || x >= pml_x1h) {
        m_ez_x[i] = bxh[x] * m_ez_x[i] + axh[x] * dEz_dx;
        dEz_dx = dEz_dx / kxh[x] + m_ez_x[i];
        m_ey_x[i] = bxh[x] * m_ey_x[i] + axh[x] * dEy_dx;
        dEy_dx = dEy_dx / kxh[x] + m_ey_x[i];
    }

    hx[i] -= cq_val * (dEy_dz - dEz_dy);
    hy[i] -= cq_val * (dEz_dx - dEx_dz);
    hz[i] -= cq_val * (dEx_dy - dEy_dx);
}

// ---------------- forward: E integer-step with snapshots ----------------
// Curls of Hx/Hy/Hz (half-integer-stored fields differentiated back to the
// integer E points) with CPML memory (integer profiles); snapshots the
// pre-update Ex/Ey/Ez and the PML-modified curls on the sampling interval,
// then Ex/Ey/Ez = ca*E + cb*curl.  Only the FD interior is updated (E fields
// outside it are zero, mirroring em2d_tm).
template <typename T, int FD_PAD>
__global__ void step_e_storage_kernel(
    const T* __restrict__ ca, const T* __restrict__ cb,
    const T* __restrict__ hx, const T* __restrict__ hy, const T* __restrict__ hz,
    T* __restrict__ ex, T* __restrict__ ey, T* __restrict__ ez,
    T* __restrict__ m_hy_z, T* __restrict__ m_hz_y, T* __restrict__ m_hz_x,
    T* __restrict__ m_hx_z, T* __restrict__ m_hx_y, T* __restrict__ m_hy_x,
    T* __restrict__ ex_store, T* __restrict__ ey_store, T* __restrict__ ez_store,
    T* __restrict__ curl_x_store, T* __restrict__ curl_y_store, T* __restrict__ curl_z_store,
    const T* __restrict__ az, const T* __restrict__ bz,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ kz, const T* __restrict__ ky, const T* __restrict__ kx,
    const T* __restrict__ c,
    T rdz, T rdy, T rdx,
    int t, int interval, int64_t snap_off,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int ca_batched, int cb_batched)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)n_shots * nz * ny * nx;
    if (i >= total)
        return;
    const long ny_nx = (long)ny * nx;
    const int x = (int)(i % nx);
    const int y = (int)((i / nx) % ny);
    const int z = (int)((i / ((long)nx * ny)) % nz);
    const int s = (int)(i / ((long)nx * ny * nz));
    if (z < FD_PAD || z >= nz - (FD_PAD - 1) || y < FD_PAD ||
        y >= ny - (FD_PAD - 1) || x < FD_PAD || x >= nx - (FD_PAD - 1))
        return;
    const long j = (long)z * ny_nx + (long)y * nx + x;
    const T ca_val = ca_batched ? ca[i] : ca[j];
    const T cb_val = cb_batched ? cb[i] : cb[j];

    T dHy_dz = diff_half_z<T, FD_PAD>(hy, i, c, rdz, ny_nx);
    T dHz_dy = diff_half_y<T, FD_PAD>(hz, i, c, rdy, nx);
    T dHz_dx = diff_half_x<T, FD_PAD>(hz, i, c, rdx);
    T dHx_dz = diff_half_z<T, FD_PAD>(hx, i, c, rdz, ny_nx);
    T dHx_dy = diff_half_y<T, FD_PAD>(hx, i, c, rdy, nx);
    T dHy_dx = diff_half_x<T, FD_PAD>(hy, i, c, rdx);

    if (z < pml_z0 || z >= pml_z1) {
        m_hy_z[i] = bz[z] * m_hy_z[i] + az[z] * dHy_dz;
        dHy_dz = dHy_dz / kz[z] + m_hy_z[i];
        m_hx_z[i] = bz[z] * m_hx_z[i] + az[z] * dHx_dz;
        dHx_dz = dHx_dz / kz[z] + m_hx_z[i];
    }
    if (y < pml_y0 || y >= pml_y1) {
        m_hz_y[i] = by[y] * m_hz_y[i] + ay[y] * dHz_dy;
        dHz_dy = dHz_dy / ky[y] + m_hz_y[i];
        m_hx_y[i] = by[y] * m_hx_y[i] + ay[y] * dHx_dy;
        dHx_dy = dHx_dy / ky[y] + m_hx_y[i];
    }
    if (x < pml_x0 || x >= pml_x1) {
        m_hz_x[i] = bx[x] * m_hz_x[i] + ax[x] * dHz_dx;
        dHz_dx = dHz_dx / kx[x] + m_hz_x[i];
        m_hy_x[i] = bx[x] * m_hy_x[i] + ax[x] * dHy_dx;
        dHy_dx = dHy_dx / kx[x] + m_hy_x[i];
    }

    const T curl_x = dHy_dz - dHz_dy;
    const T curl_y = dHz_dx - dHx_dz;
    const T curl_z = dHx_dy - dHy_dx;
    if (t % interval == 0) {
        const int64_t soff = snap_off + i;
        ex_store[soff] = ex[i];
        ey_store[soff] = ey[i];
        ez_store[soff] = ez[i];
        curl_x_store[soff] = curl_x;
        curl_y_store[soff] = curl_y;
        curl_z_store[soff] = curl_z;
    }
    ex[i] = ca_val * ex[i] + cb_val * curl_x;
    ey[i] = ca_val * ey[i] + cb_val * curl_y;
    ez[i] = ca_val * ez[i] + cb_val * curl_z;
}

// ---------------- forward: source injection / receiver recording ----------------
template <typename T>
__global__ void inject_kernel(
    T* __restrict__ field, const T* __restrict__ f, const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long shot_numel)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        if (idx >= 0)
            field[(long)s * shot_numel + idx] += f[(((long)t * n_shots + s) * n_src + k)];
    }
}

template <typename T>
__global__ void record_kernel(
    const T* __restrict__ field, T* __restrict__ r, const long* __restrict__ rec_i,
    int t, int n_shots, int n_rec, long shot_numel)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            r[(((long)t * n_shots + s) * n_rec + k)] = field[(long)s * shot_numel + idx];
    }
}

// ---------------- backward: receiver / source seeding ----------------
template <typename T>
__global__ void record_grad_r_kernel(
    T* __restrict__ lam_field, const T* __restrict__ grad_r, const long* __restrict__ rec_i,
    int t, int n_shots, int n_rec, long shot_numel)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            lam_field[(long)s * shot_numel + idx] +=
                grad_r[(((long)t * n_shots + s) * n_rec + k)];
    }
}

template <typename T>
__global__ void record_grad_f_kernel(
    const T* __restrict__ lam_field, T* __restrict__ grad_f, const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long shot_numel)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        grad_f[(((long)t * n_shots + s) * n_src + k)] =
            (idx >= 0) ? lam_field[(long)s * shot_numel + idx] : (T)0;
    }
}

// ---------------- backward: ca/cb model gradients ----------------
template <typename T, int FD_PAD>
__global__ void coeff_grad_kernel(
    const T* __restrict__ lam_ex, const T* __restrict__ lam_ey, const T* __restrict__ lam_ez,
    const T* __restrict__ ex_store, const T* __restrict__ ey_store,
    const T* __restrict__ ez_store,
    const T* __restrict__ curl_x_store, const T* __restrict__ curl_y_store,
    const T* __restrict__ curl_z_store,
    T* __restrict__ grad_ca, T* __restrict__ grad_cb,
    T scale, int64_t snap_off,
    int n_shots, int nz, int ny, int nx)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)n_shots * nz * ny * nx;
    if (i >= total)
        return;
    const long ny_nx = (long)ny * nx;
    const int x = (int)(i % nx);
    const int y = (int)((i / nx) % ny);
    const int z = (int)((i / ((long)nx * ny)) % nz);
    if (z < FD_PAD || z >= nz - (FD_PAD - 1) || y < FD_PAD ||
        y >= ny - (FD_PAD - 1) || x < FD_PAD || x >= nx - (FD_PAD - 1))
        return;
    const int64_t soff = snap_off + i;
    const T lex = lam_ex[i] * scale;
    const T ley = lam_ey[i] * scale;
    const T lez = lam_ez[i] * scale;
    grad_ca[i] += lex * ex_store[soff] + ley * ey_store[soff] + lez * ez_store[soff];
    grad_cb[i] += lex * curl_x_store[soff] + ley * curl_y_store[soff] +
                  lez * curl_z_store[soff];
}

// ---------------- backward: transpose of the E update, stage 1 ----------------
// lam_ex/ey/ez *= ca and builds the six work arrays (the PML-modified curls
// applied to cb*lam_E) with the time-reversed integer-profile memory
// recursions; outside the FD interior the work is zeroed for the stage-2
// divergence.  The exact transpose of the forward curl + memory step.
template <typename T, int FD_PAD>
__global__ void adjoint_e_stage1_kernel(
    const T* __restrict__ ca, const T* __restrict__ cb,
    T* __restrict__ lam_ex, T* __restrict__ lam_ey, T* __restrict__ lam_ez,
    T* __restrict__ m_lambda_hy_z, T* __restrict__ m_lambda_hz_y,
    T* __restrict__ m_lambda_hz_x, T* __restrict__ m_lambda_hx_z,
    T* __restrict__ m_lambda_hx_y, T* __restrict__ m_lambda_hy_x,
    T* __restrict__ work_hy_z, T* __restrict__ work_hz_y, T* __restrict__ work_hz_x,
    T* __restrict__ work_hx_z, T* __restrict__ work_hx_y, T* __restrict__ work_hy_x,
    const T* __restrict__ az, const T* __restrict__ bz,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ kz, const T* __restrict__ ky, const T* __restrict__ kx,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int ca_batched, int cb_batched)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)n_shots * nz * ny * nx;
    if (i >= total)
        return;
    const long ny_nx = (long)ny * nx;
    const int x = (int)(i % nx);
    const int y = (int)((i / nx) % ny);
    const int z = (int)((i / ((long)nx * ny)) % nz);
    const int s = (int)(i / ((long)nx * ny * nz));
    const long j = (long)z * ny_nx + (long)y * nx + x;
    const bool active = z >= FD_PAD && z < nz - (FD_PAD - 1) && y >= FD_PAD &&
                        y < ny - (FD_PAD - 1) && x >= FD_PAD && x < nx - (FD_PAD - 1);
    if (!active) {
        work_hy_z[i] = (T)0; work_hz_y[i] = (T)0; work_hz_x[i] = (T)0;
        work_hx_z[i] = (T)0; work_hx_y[i] = (T)0; work_hy_x[i] = (T)0;
        return;
    }
    const T ca_val = ca_batched ? ca[i] : ca[j];
    const T cb_val = cb_batched ? cb[i] : cb[j];
    const T gx = cb_val * lam_ex[i];
    const T gy = cb_val * lam_ey[i];
    const T gz = cb_val * lam_ez[i];
    lam_ex[i] = ca_val * lam_ex[i];
    lam_ey[i] = ca_val * lam_ey[i];
    lam_ez[i] = ca_val * lam_ez[i];

    // Ex: +dHy_dz - dHz_dy
    if (z < pml_z0 || z >= pml_z1) {
        const T w = gx + bz[z] * m_lambda_hy_z[i];
        work_hy_z[i] = gx / kz[z] + az[z] * w;
        m_lambda_hy_z[i] = w;
    } else {
        work_hy_z[i] = gx;
    }
    if (y < pml_y0 || y >= pml_y1) {
        const T w = -gx + by[y] * m_lambda_hz_y[i];
        work_hz_y[i] = -gx / ky[y] + ay[y] * w;
        m_lambda_hz_y[i] = w;
    } else {
        work_hz_y[i] = -gx;
    }
    // Ey: +dHz_dx - dHx_dz
    if (x < pml_x0 || x >= pml_x1) {
        const T w = gy + bx[x] * m_lambda_hz_x[i];
        work_hz_x[i] = gy / kx[x] + ax[x] * w;
        m_lambda_hz_x[i] = w;
    } else {
        work_hz_x[i] = gy;
    }
    if (z < pml_z0 || z >= pml_z1) {
        const T w = -gy + bz[z] * m_lambda_hx_z[i];
        work_hx_z[i] = -gy / kz[z] + az[z] * w;
        m_lambda_hx_z[i] = w;
    } else {
        work_hx_z[i] = -gy;
    }
    // Ez: +dHx_dy - dHy_dx
    if (y < pml_y0 || y >= pml_y1) {
        const T w = gz + by[y] * m_lambda_hx_y[i];
        work_hx_y[i] = gz / ky[y] + ay[y] * w;
        m_lambda_hx_y[i] = w;
    } else {
        work_hx_y[i] = gz;
    }
    if (x < pml_x0 || x >= pml_x1) {
        const T w = -gz + bx[x] * m_lambda_hy_x[i];
        work_hy_x[i] = -gz / kx[x] + ax[x] * w;
        m_lambda_hy_x[i] = w;
    } else {
        work_hy_x[i] = -gz;
    }
}

// ---------------- backward: transpose of the E update, stage 2 ----------------
// lam_hy/hz/hx += the transposes of the six half-grid curls (the transpose of
// diff_half applied to each work array, summed per H component).  No interior
// guard: boundary cells get zero contributions from the zeroed work arrays.
template <typename T, int FD_PAD>
__global__ void adjoint_e_stage2_kernel(
    const T* __restrict__ work_hy_z, const T* __restrict__ work_hz_y,
    const T* __restrict__ work_hz_x, const T* __restrict__ work_hx_z,
    const T* __restrict__ work_hx_y, const T* __restrict__ work_hy_x,
    T* __restrict__ lam_hy, T* __restrict__ lam_hz, T* __restrict__ lam_hx,
    const T* __restrict__ c,
    T rdz, T rdy, T rdx,
    int n_shots, int nz, int ny, int nx)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)n_shots * nz * ny * nx;
    if (i >= total)
        return;
    const long ny_nx = (long)ny * nx;
    const int x = (int)(i % nx);
    const int y = (int)((i / nx) % ny);
    const int z = (int)((i / ((long)nx * ny)) % nz);
    T acc_z = (T)0;
    for (int k = 1; k <= FD_PAD; ++k) {
        const T ck = c[k - 1];
        if (z - k + 1 >= 0)
            acc_z += ck * work_hy_z[i - (long)(k - 1) * ny_nx];
        if (z + k < nz)
            acc_z -= ck * work_hy_z[i + (long)k * ny_nx];
    }
    lam_hy[i] += acc_z * rdz;
    T acc_hy_x = (T)0;
    for (int k = 1; k <= FD_PAD; ++k) {
        const T ck = c[k - 1];
        if (x - k + 1 >= 0)
            acc_hy_x += ck * work_hy_x[i - k + 1];
        if (x + k < nx)
            acc_hy_x -= ck * work_hy_x[i + k];
    }
    lam_hy[i] += acc_hy_x * rdx;

    T acc_y = (T)0;
    for (int k = 1; k <= FD_PAD; ++k) {
        const T ck = c[k - 1];
        if (y - k + 1 >= 0)
            acc_y += ck * work_hz_y[i - (long)(k - 1) * nx];
        if (y + k < ny)
            acc_y -= ck * work_hz_y[i + (long)k * nx];
    }
    lam_hz[i] += acc_y * rdy;
    T acc_hz_x = (T)0;
    for (int k = 1; k <= FD_PAD; ++k) {
        const T ck = c[k - 1];
        if (x - k + 1 >= 0)
            acc_hz_x += ck * work_hz_x[i - k + 1];
        if (x + k < nx)
            acc_hz_x -= ck * work_hz_x[i + k];
    }
    lam_hz[i] += acc_hz_x * rdx;

    T acc_hx_z = (T)0;
    for (int k = 1; k <= FD_PAD; ++k) {
        const T ck = c[k - 1];
        if (z - k + 1 >= 0)
            acc_hx_z += ck * work_hx_z[i - (long)(k - 1) * ny_nx];
        if (z + k < nz)
            acc_hx_z -= ck * work_hx_z[i + (long)k * ny_nx];
    }
    lam_hx[i] += acc_hx_z * rdz;
    T acc_hx_y = (T)0;
    for (int k = 1; k <= FD_PAD; ++k) {
        const T ck = c[k - 1];
        if (y - k + 1 >= 0)
            acc_hx_y += ck * work_hx_y[i - (long)(k - 1) * nx];
        if (y + k < ny)
            acc_hx_y -= ck * work_hx_y[i + (long)k * nx];
    }
    lam_hx[i] += acc_hx_y * rdy;
}

// ---------------- backward: transpose of the H half-step, stage 1 ----------------
// Builds the six work2 arrays from lam_hx/lam_hy/lam_hz through cq and the
// time-reversed half-integer PML memory.  The H-update at the outermost
// stencil rows never feeds the E fields, so those work cells are zeroed
// (matching em2d_tm).
template <typename T, int FD_PAD>
__global__ void adjoint_h_stage1_kernel(
    const T* __restrict__ cq,
    const T* __restrict__ lam_hx, const T* __restrict__ lam_hy, const T* __restrict__ lam_hz,
    T* __restrict__ m_lambda_ey_z, T* __restrict__ m_lambda_ez_y,
    T* __restrict__ m_lambda_ez_x, T* __restrict__ m_lambda_ex_z,
    T* __restrict__ m_lambda_ex_y, T* __restrict__ m_lambda_ey_x,
    T* __restrict__ work2_ey_z, T* __restrict__ work2_ez_y, T* __restrict__ work2_ez_x,
    T* __restrict__ work2_ex_z, T* __restrict__ work2_ex_y, T* __restrict__ work2_ey_x,
    const T* __restrict__ azh, const T* __restrict__ bzh,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ kzh, const T* __restrict__ kyh, const T* __restrict__ kxh,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int cq_batched)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)n_shots * nz * ny * nx;
    if (i >= total)
        return;
    const long ny_nx = (long)ny * nx;
    const int x = (int)(i % nx);
    const int y = (int)((i / nx) % ny);
    const int z = (int)((i / ((long)nx * ny)) % nz);
    const int s = (int)(i / ((long)nx * ny * nz));
    const long j = (long)z * ny_nx + (long)y * nx + x;
    const bool active = z >= FD_PAD && z < nz - (FD_PAD - 1) && y >= FD_PAD &&
                        y < ny - (FD_PAD - 1) && x >= FD_PAD && x < nx - (FD_PAD - 1);
    if (!active) {
        work2_ey_z[i] = (T)0; work2_ez_y[i] = (T)0; work2_ez_x[i] = (T)0;
        work2_ex_z[i] = (T)0; work2_ex_y[i] = (T)0; work2_ey_x[i] = (T)0;
        return;
    }
    const T cq_val = cq_batched ? cq[i] : cq[j];

    int pml_z1h = pml_z1 - 1;
    if (pml_z1 <= pml_z0)
        pml_z1h = pml_z0;
    int pml_y1h = pml_y1 - 1;
    if (pml_y1 <= pml_y0)
        pml_y1h = pml_y0;
    int pml_x1h = pml_x1 - 1;
    if (pml_x1 <= pml_x0)
        pml_x1h = pml_x0;

    // dEy_dz feeds Hx with -cq
    if (z < nz - FD_PAD) {
        const T g2 = -cq_val * lam_hx[i];
        if (z < pml_z0 || z >= pml_z1h) {
            const T w = g2 + bzh[z] * m_lambda_ey_z[i];
            work2_ey_z[i] = g2 / kzh[z] + azh[z] * w;
            m_lambda_ey_z[i] = w;
        } else {
            work2_ey_z[i] = g2;
        }
    } else {
        work2_ey_z[i] = (T)0;
    }
    // dEz_dy feeds Hx with +cq
    if (y < ny - FD_PAD) {
        const T g2 = cq_val * lam_hx[i];
        if (y < pml_y0 || y >= pml_y1h) {
            const T w = g2 + byh[y] * m_lambda_ez_y[i];
            work2_ez_y[i] = g2 / kyh[y] + ayh[y] * w;
            m_lambda_ez_y[i] = w;
        } else {
            work2_ez_y[i] = g2;
        }
    } else {
        work2_ez_y[i] = (T)0;
    }
    // dEz_dx feeds Hy with -cq
    if (x < nx - FD_PAD) {
        const T g2 = -cq_val * lam_hy[i];
        if (x < pml_x0 || x >= pml_x1h) {
            const T w = g2 + bxh[x] * m_lambda_ez_x[i];
            work2_ez_x[i] = g2 / kxh[x] + axh[x] * w;
            m_lambda_ez_x[i] = w;
        } else {
            work2_ez_x[i] = g2;
        }
    } else {
        work2_ez_x[i] = (T)0;
    }
    // dEx_dz feeds Hy with +cq
    if (z < nz - FD_PAD) {
        const T g2 = cq_val * lam_hy[i];
        if (z < pml_z0 || z >= pml_z1h) {
            const T w = g2 + bzh[z] * m_lambda_ex_z[i];
            work2_ex_z[i] = g2 / kzh[z] + azh[z] * w;
            m_lambda_ex_z[i] = w;
        } else {
            work2_ex_z[i] = g2;
        }
    } else {
        work2_ex_z[i] = (T)0;
    }
    // dEx_dy feeds Hz with -cq
    if (y < ny - FD_PAD) {
        const T g2 = -cq_val * lam_hz[i];
        if (y < pml_y0 || y >= pml_y1h) {
            const T w = g2 + byh[y] * m_lambda_ex_y[i];
            work2_ex_y[i] = g2 / kyh[y] + ayh[y] * w;
            m_lambda_ex_y[i] = w;
        } else {
            work2_ex_y[i] = g2;
        }
    } else {
        work2_ex_y[i] = (T)0;
    }
    // dEy_dx feeds Hz with +cq
    if (x < nx - FD_PAD) {
        const T g2 = cq_val * lam_hz[i];
        if (x < pml_x0 || x >= pml_x1h) {
            const T w = g2 + bxh[x] * m_lambda_ey_x[i];
            work2_ey_x[i] = g2 / kxh[x] + axh[x] * w;
            m_lambda_ey_x[i] = w;
        } else {
            work2_ey_x[i] = g2;
        }
    } else {
        work2_ey_x[i] = (T)0;
    }
}

// ---------------- backward: cq model gradient ----------------
// Uses the sampled E differences and the stage-1 memory variables; must run
// after adjoint_h_stage1 (m_lambda_* updated for this step), mirroring
// em2d_tm's cq_grad.
template <typename T, int FD_PAD>
__global__ void cq_grad_kernel(
    const T* __restrict__ cq,
    const T* __restrict__ lam_hx, const T* __restrict__ lam_hy, const T* __restrict__ lam_hz,
    const T* __restrict__ ex_store, const T* __restrict__ ey_store,
    const T* __restrict__ ez_store,
    const T* __restrict__ m_lambda_ey_z, const T* __restrict__ m_lambda_ez_y,
    const T* __restrict__ m_lambda_ez_x, const T* __restrict__ m_lambda_ex_z,
    const T* __restrict__ m_lambda_ex_y, const T* __restrict__ m_lambda_ey_x,
    T* __restrict__ grad_cq,
    const T* __restrict__ azh, const T* __restrict__ kzh,
    const T* __restrict__ ayh, const T* __restrict__ kyh,
    const T* __restrict__ axh, const T* __restrict__ kxh,
    const T* __restrict__ c,
    T rdz, T rdy, T rdx,
    T scale, int64_t snap_off,
    int n_shots, int nz, int ny, int nx,
    int cq_batched)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)n_shots * nz * ny * nx;
    if (i >= total)
        return;
    const long ny_nx = (long)ny * nx;
    const int x = (int)(i % nx);
    const int y = (int)((i / nx) % ny);
    const int z = (int)((i / ((long)nx * ny)) % nz);
    const int s = (int)(i / ((long)nx * ny * nz));
    const long j = (long)z * ny_nx + (long)y * nx + x;
    if (z < FD_PAD || z >= nz - (FD_PAD - 1) || y < FD_PAD ||
        y >= ny - (FD_PAD - 1) || x < FD_PAD || x >= nx - (FD_PAD - 1))
        return;
    const int64_t soff = snap_off + i;
    const T cq_val = cq_batched ? cq[i] : cq[j];
    T term = (T)0;
    if (z < nz - FD_PAD) {
        const T dEy_dz = diff_int_z<T, FD_PAD>(ey_store, soff, c, rdz, ny_nx);
        term += -lam_hx[i] * dEy_dz / kzh[z]
              + (azh[z] / cq_val) * dEy_dz * m_lambda_ey_z[i];
        const T dEx_dz = diff_int_z<T, FD_PAD>(ex_store, soff, c, rdz, ny_nx);
        term += lam_hy[i] * dEx_dz / kzh[z]
              + (azh[z] / cq_val) * dEx_dz * m_lambda_ex_z[i];
    }
    if (y < ny - FD_PAD) {
        const T dEz_dy = diff_int_y<T, FD_PAD>(ez_store, soff, c, rdy, nx);
        term += lam_hx[i] * dEz_dy / kyh[y]
              + (ayh[y] / cq_val) * dEz_dy * m_lambda_ez_y[i];
        const T dEx_dy = diff_int_y<T, FD_PAD>(ex_store, soff, c, rdy, nx);
        term += -lam_hz[i] * dEx_dy / kyh[y]
              + (ayh[y] / cq_val) * dEx_dy * m_lambda_ex_y[i];
    }
    if (x < nx - FD_PAD) {
        const T dEz_dx = diff_int_x<T, FD_PAD>(ez_store, soff, c, rdx);
        term += -lam_hy[i] * dEz_dx / kxh[x]
              + (axh[x] / cq_val) * dEz_dx * m_lambda_ez_x[i];
        const T dEy_dx = diff_int_x<T, FD_PAD>(ey_store, soff, c, rdx);
        term += lam_hz[i] * dEy_dx / kxh[x]
              + (axh[x] / cq_val) * dEy_dx * m_lambda_ey_x[i];
    }
    grad_cq[i] += scale * term;
}

// ---------------- backward: transpose of the H half-step, stage 2 ----------------
// lam_ex/ey/ez += the transposes of the six integer-grid H curls (the
// transpose of diff_int applied to each work2 array, summed per E component).
template <typename T, int FD_PAD>
__global__ void adjoint_h_stage2_kernel(
    const T* __restrict__ work2_ey_z, const T* __restrict__ work2_ez_y,
    const T* __restrict__ work2_ez_x, const T* __restrict__ work2_ex_z,
    const T* __restrict__ work2_ex_y, const T* __restrict__ work2_ey_x,
    T* __restrict__ lam_ex, T* __restrict__ lam_ey, T* __restrict__ lam_ez,
    const T* __restrict__ c,
    T rdz, T rdy, T rdx,
    int n_shots, int nz, int ny, int nx)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)n_shots * nz * ny * nx;
    if (i >= total)
        return;
    const long ny_nx = (long)ny * nx;
    const int x = (int)(i % nx);
    const int y = (int)((i / nx) % ny);
    const int z = (int)((i / ((long)nx * ny)) % nz);
    T sum_ey_z = (T)0;
    for (int k = 1; k <= FD_PAD; ++k) {
        const T ck = c[k - 1];
        if (z >= k)
            sum_ey_z += ck * work2_ey_z[i - (long)k * ny_nx];
        if (z + k - 1 < nz)
            sum_ey_z -= ck * work2_ey_z[i + (long)(k - 1) * ny_nx];
    }
    lam_ey[i] += sum_ey_z * rdz;
    T sum_ey_x = (T)0;
    for (int k = 1; k <= FD_PAD; ++k) {
        const T ck = c[k - 1];
        if (x >= k)
            sum_ey_x += ck * work2_ey_x[i - k];
        if (x + k - 1 < nx)
            sum_ey_x -= ck * work2_ey_x[i + k - 1];
    }
    lam_ey[i] += sum_ey_x * rdx;

    T sum_ex_z = (T)0;
    for (int k = 1; k <= FD_PAD; ++k) {
        const T ck = c[k - 1];
        if (z >= k)
            sum_ex_z += ck * work2_ex_z[i - (long)k * ny_nx];
        if (z + k - 1 < nz)
            sum_ex_z -= ck * work2_ex_z[i + (long)(k - 1) * ny_nx];
    }
    lam_ex[i] += sum_ex_z * rdz;
    T sum_ex_y = (T)0;
    for (int k = 1; k <= FD_PAD; ++k) {
        const T ck = c[k - 1];
        if (y >= k)
            sum_ex_y += ck * work2_ex_y[i - (long)k * nx];
        if (y + k - 1 < ny)
            sum_ex_y -= ck * work2_ex_y[i + (long)(k - 1) * nx];
    }
    lam_ex[i] += sum_ex_y * rdy;

    T sum_ez_y = (T)0;
    for (int k = 1; k <= FD_PAD; ++k) {
        const T ck = c[k - 1];
        if (y >= k)
            sum_ez_y += ck * work2_ez_y[i - (long)k * nx];
        if (y + k - 1 < ny)
            sum_ez_y -= ck * work2_ez_y[i + (long)(k - 1) * nx];
    }
    lam_ez[i] += sum_ez_y * rdy;
    T sum_ez_x = (T)0;
    for (int k = 1; k <= FD_PAD; ++k) {
        const T ck = c[k - 1];
        if (x >= k)
            sum_ez_x += ck * work2_ez_x[i - k];
        if (x + k - 1 < nx)
            sum_ez_x -= ck * work2_ez_x[i + k - 1];
    }
    lam_ez[i] += sum_ez_x * rdx;
}

// ---------------- launchers ----------------
// The compile-time-FD_PAD kernel variants are dispatched from the runtime
// fd_pad_z0 (all three direction radii equal accuracy // 2 == FD_PAD; the
// z1/y1/x1 pads, accuracy // 2 - 1, appear only in the compound interior
// guards as FD_PAD - 1), mirroring em2d_tm's LAUNCH_EM2D: switch on the
// runtime pad, launch KERN<T, FD_PAD> with the constant-folded stencils.
// Coefficients and arithmetic order are unchanged, so results are bitwise
// identical to the runtime-radius kernels.
#define LAUNCH_FLAT(KERN, T, FP, ...)                                          \
    {                                                                          \
        const int64_t _n = (total);                                            \
        if (_n > 0) {                                                          \
            const int threads = 256;                                           \
            const int blocks = (int)((_n + threads - 1) / threads);            \
            KERN<T, FP><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>( \
                __VA_ARGS__);                                                  \
            TORCH_CHECK(cudaGetLastError() == cudaSuccess,                     \
                        "nami em3d " #KERN " failed");                         \
        }                                                                      \
    }

#define LAUNCH_EM3D(KERN, T, ...)                                              \
    {                                                                          \
        switch ((int)fd_pad_z0) {                                              \
            case 1: LAUNCH_FLAT(KERN, T, 1, __VA_ARGS__); break;               \
            case 2: LAUNCH_FLAT(KERN, T, 2, __VA_ARGS__); break;               \
            case 3: LAUNCH_FLAT(KERN, T, 3, __VA_ARGS__); break;               \
            case 4: LAUNCH_FLAT(KERN, T, 4, __VA_ARGS__); break;               \
            default: TORCH_CHECK(false, "nami em3d: unsupported fd_pad");      \
        }                                                                      \
    }

#define LAUNCH_SR(KERN, T, NSHOTS, NSR, ...)                                   \
    {                                                                          \
        dim3 block(32, 4);                                                     \
        dim3 grid((int)(((NSHOTS) + 31) / 32), (int)(((NSR) + 3) / 4));        \
        KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(         \
            __VA_ARGS__);                                                      \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,                         \
                    "nami em3d " #KERN " failed");                             \
    }

static inline int64_t field_numel(const torch::Tensor& f)
{
    return f.numel();
}

void step_h(
    torch::Tensor cq, torch::Tensor ex, torch::Tensor ey, torch::Tensor ez,
    torch::Tensor hx, torch::Tensor hy, torch::Tensor hz,
    torch::Tensor m_ey_z, torch::Tensor m_ez_y, torch::Tensor m_ez_x,
    torch::Tensor m_ex_z, torch::Tensor m_ex_y, torch::Tensor m_ey_x,
    torch::Tensor azh, torch::Tensor bzh, torch::Tensor ayh, torch::Tensor byh,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor kzh, torch::Tensor kyh,
    torch::Tensor kxh,
    torch::Tensor c,
    double rdz, double rdy, double rdx,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t pml_z0, int64_t pml_z1, int64_t pml_y0, int64_t pml_y1,
    int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_z0, int64_t fd_pad_y0, int64_t fd_pad_x0,
    int64_t cq_batched)
{
    const int64_t total = field_numel(ex);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(ex.scalar_type(), "em3d_step_h", [&] {
        LAUNCH_EM3D(step_h_kernel, scalar_t,
            cq.data_ptr<scalar_t>(),
            ex.data_ptr<scalar_t>(), ey.data_ptr<scalar_t>(), ez.data_ptr<scalar_t>(),
            hx.data_ptr<scalar_t>(), hy.data_ptr<scalar_t>(), hz.data_ptr<scalar_t>(),
            m_ey_z.data_ptr<scalar_t>(), m_ez_y.data_ptr<scalar_t>(),
            m_ez_x.data_ptr<scalar_t>(), m_ex_z.data_ptr<scalar_t>(),
            m_ex_y.data_ptr<scalar_t>(), m_ey_x.data_ptr<scalar_t>(),
            azh.data_ptr<scalar_t>(), bzh.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), byh.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), bxh.data_ptr<scalar_t>(),
            kzh.data_ptr<scalar_t>(), kyh.data_ptr<scalar_t>(),
            kxh.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdz, (scalar_t)rdy, (scalar_t)rdx,
            (int)n_shots, (int)nz, (int)ny, (int)nx,
            (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1,
            (int)cq_batched);
    });
}

void step_e_storage(
    torch::Tensor ca, torch::Tensor cb,
    torch::Tensor hx, torch::Tensor hy, torch::Tensor hz,
    torch::Tensor ex, torch::Tensor ey, torch::Tensor ez,
    torch::Tensor m_hy_z, torch::Tensor m_hz_y, torch::Tensor m_hz_x,
    torch::Tensor m_hx_z, torch::Tensor m_hx_y, torch::Tensor m_hy_x,
    torch::Tensor ex_store, torch::Tensor ey_store, torch::Tensor ez_store,
    torch::Tensor curl_x_store, torch::Tensor curl_y_store, torch::Tensor curl_z_store,
    torch::Tensor az, torch::Tensor bz, torch::Tensor ay, torch::Tensor by,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor kz, torch::Tensor ky,
    torch::Tensor kx,
    torch::Tensor c,
    double rdz, double rdy, double rdx,
    int64_t t, int64_t interval, int64_t snap_off,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t pml_z0, int64_t pml_z1, int64_t pml_y0, int64_t pml_y1,
    int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_z0, int64_t fd_pad_z1, int64_t fd_pad_y0, int64_t fd_pad_y1,
    int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t ca_batched, int64_t cb_batched)
{
    const int64_t total = field_numel(ex);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(ex.scalar_type(), "em3d_step_e_storage", [&] {
        LAUNCH_EM3D(step_e_storage_kernel, scalar_t,
            ca.data_ptr<scalar_t>(), cb.data_ptr<scalar_t>(),
            hx.data_ptr<scalar_t>(), hy.data_ptr<scalar_t>(), hz.data_ptr<scalar_t>(),
            ex.data_ptr<scalar_t>(), ey.data_ptr<scalar_t>(), ez.data_ptr<scalar_t>(),
            m_hy_z.data_ptr<scalar_t>(), m_hz_y.data_ptr<scalar_t>(),
            m_hz_x.data_ptr<scalar_t>(), m_hx_z.data_ptr<scalar_t>(),
            m_hx_y.data_ptr<scalar_t>(), m_hy_x.data_ptr<scalar_t>(),
            ex_store.data_ptr<scalar_t>(), ey_store.data_ptr<scalar_t>(),
            ez_store.data_ptr<scalar_t>(),
            curl_x_store.data_ptr<scalar_t>(), curl_y_store.data_ptr<scalar_t>(),
            curl_z_store.data_ptr<scalar_t>(),
            az.data_ptr<scalar_t>(), bz.data_ptr<scalar_t>(),
            ay.data_ptr<scalar_t>(), by.data_ptr<scalar_t>(),
            ax.data_ptr<scalar_t>(), bx.data_ptr<scalar_t>(),
            kz.data_ptr<scalar_t>(), ky.data_ptr<scalar_t>(),
            kx.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdz, (scalar_t)rdy, (scalar_t)rdx,
            (int)t, (int)interval, (int64_t)snap_off,
            (int)n_shots, (int)nz, (int)ny, (int)nx,
            (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1,
            (int)ca_batched, (int)cb_batched);
    });
}

void inject(torch::Tensor field, torch::Tensor f, torch::Tensor src_i,
            int64_t t, int64_t n_shots, int64_t n_src, int64_t shot_numel)
{
    AT_DISPATCH_FLOATING_TYPES(field.scalar_type(), "em3d_inject", [&] {
        LAUNCH_SR(inject_kernel, scalar_t, n_shots, n_src,
            field.data_ptr<scalar_t>(), f.data_ptr<scalar_t>(),
            src_i.data_ptr<int64_t>(), (int)t, (int)n_shots, (int)n_src,
            (long)shot_numel);
    });
}

void record(torch::Tensor field, torch::Tensor r, torch::Tensor rec_i,
            int64_t t, int64_t n_shots, int64_t n_rec, int64_t shot_numel)
{
    AT_DISPATCH_FLOATING_TYPES(field.scalar_type(), "em3d_record", [&] {
        LAUNCH_SR(record_kernel, scalar_t, n_shots, n_rec,
            field.data_ptr<scalar_t>(), r.data_ptr<scalar_t>(),
            rec_i.data_ptr<int64_t>(), (int)t, (int)n_shots, (int)n_rec,
            (long)shot_numel);
    });
}

void record_grad_r(torch::Tensor lam_field, torch::Tensor grad_r, torch::Tensor rec_i,
                   int64_t t, int64_t n_shots, int64_t n_rec, int64_t shot_numel)
{
    AT_DISPATCH_FLOATING_TYPES(lam_field.scalar_type(), "em3d_record_grad_r", [&] {
        LAUNCH_SR(record_grad_r_kernel, scalar_t, n_shots, n_rec,
            lam_field.data_ptr<scalar_t>(), grad_r.data_ptr<scalar_t>(),
            rec_i.data_ptr<int64_t>(), (int)t, (int)n_shots, (int)n_rec,
            (long)shot_numel);
    });
}

void record_grad_f(torch::Tensor lam_field, torch::Tensor grad_f, torch::Tensor src_i,
                   int64_t t, int64_t n_shots, int64_t n_src, int64_t shot_numel)
{
    AT_DISPATCH_FLOATING_TYPES(lam_field.scalar_type(), "em3d_record_grad_f", [&] {
        LAUNCH_SR(record_grad_f_kernel, scalar_t, n_shots, n_src,
            lam_field.data_ptr<scalar_t>(), grad_f.data_ptr<scalar_t>(),
            src_i.data_ptr<int64_t>(), (int)t, (int)n_shots, (int)n_src,
            (long)shot_numel);
    });
}

void coeff_grad(
    torch::Tensor lam_ex, torch::Tensor lam_ey, torch::Tensor lam_ez,
    torch::Tensor ex_store, torch::Tensor ey_store, torch::Tensor ez_store,
    torch::Tensor curl_x_store, torch::Tensor curl_y_store, torch::Tensor curl_z_store,
    torch::Tensor grad_ca, torch::Tensor grad_cb,
    double scale, int64_t snap_off,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t fd_pad_z0, int64_t fd_pad_z1, int64_t fd_pad_y0, int64_t fd_pad_y1,
    int64_t fd_pad_x0, int64_t fd_pad_x1)
{
    const int64_t total = field_numel(lam_ex);
    AT_DISPATCH_FLOATING_TYPES(lam_ex.scalar_type(), "em3d_coeff_grad", [&] {
        LAUNCH_EM3D(coeff_grad_kernel, scalar_t,
            lam_ex.data_ptr<scalar_t>(), lam_ey.data_ptr<scalar_t>(),
            lam_ez.data_ptr<scalar_t>(),
            ex_store.data_ptr<scalar_t>(), ey_store.data_ptr<scalar_t>(),
            ez_store.data_ptr<scalar_t>(),
            curl_x_store.data_ptr<scalar_t>(), curl_y_store.data_ptr<scalar_t>(),
            curl_z_store.data_ptr<scalar_t>(),
            grad_ca.data_ptr<scalar_t>(), grad_cb.data_ptr<scalar_t>(),
            (scalar_t)scale, (int64_t)snap_off,
            (int)n_shots, (int)nz, (int)ny, (int)nx);
    });
}

void adjoint_e_stage1(
    torch::Tensor ca, torch::Tensor cb,
    torch::Tensor lam_ex, torch::Tensor lam_ey, torch::Tensor lam_ez,
    torch::Tensor m_lambda_hy_z, torch::Tensor m_lambda_hz_y,
    torch::Tensor m_lambda_hz_x, torch::Tensor m_lambda_hx_z,
    torch::Tensor m_lambda_hx_y, torch::Tensor m_lambda_hy_x,
    torch::Tensor work_hy_z, torch::Tensor work_hz_y, torch::Tensor work_hz_x,
    torch::Tensor work_hx_z, torch::Tensor work_hx_y, torch::Tensor work_hy_x,
    torch::Tensor az, torch::Tensor bz, torch::Tensor ay, torch::Tensor by,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor kz, torch::Tensor ky,
    torch::Tensor kx,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t pml_z0, int64_t pml_z1, int64_t pml_y0, int64_t pml_y1,
    int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_z0, int64_t fd_pad_z1, int64_t fd_pad_y0, int64_t fd_pad_y1,
    int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t ca_batched, int64_t cb_batched)
{
    const int64_t total = field_numel(lam_ex);
    AT_DISPATCH_FLOATING_TYPES(lam_ex.scalar_type(), "em3d_adjoint_e_stage1", [&] {
        LAUNCH_EM3D(adjoint_e_stage1_kernel, scalar_t,
            ca.data_ptr<scalar_t>(), cb.data_ptr<scalar_t>(),
            lam_ex.data_ptr<scalar_t>(), lam_ey.data_ptr<scalar_t>(),
            lam_ez.data_ptr<scalar_t>(),
            m_lambda_hy_z.data_ptr<scalar_t>(), m_lambda_hz_y.data_ptr<scalar_t>(),
            m_lambda_hz_x.data_ptr<scalar_t>(), m_lambda_hx_z.data_ptr<scalar_t>(),
            m_lambda_hx_y.data_ptr<scalar_t>(), m_lambda_hy_x.data_ptr<scalar_t>(),
            work_hy_z.data_ptr<scalar_t>(), work_hz_y.data_ptr<scalar_t>(),
            work_hz_x.data_ptr<scalar_t>(), work_hx_z.data_ptr<scalar_t>(),
            work_hx_y.data_ptr<scalar_t>(), work_hy_x.data_ptr<scalar_t>(),
            az.data_ptr<scalar_t>(), bz.data_ptr<scalar_t>(),
            ay.data_ptr<scalar_t>(), by.data_ptr<scalar_t>(),
            ax.data_ptr<scalar_t>(), bx.data_ptr<scalar_t>(),
            kz.data_ptr<scalar_t>(), ky.data_ptr<scalar_t>(),
            kx.data_ptr<scalar_t>(),
            (int)n_shots, (int)nz, (int)ny, (int)nx,
            (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1,
            (int)ca_batched, (int)cb_batched);
    });
}

void adjoint_e_stage2(
    torch::Tensor work_hy_z, torch::Tensor work_hz_y, torch::Tensor work_hz_x,
    torch::Tensor work_hx_z, torch::Tensor work_hx_y, torch::Tensor work_hy_x,
    torch::Tensor lam_hy, torch::Tensor lam_hz, torch::Tensor lam_hx,
    torch::Tensor c,
    double rdz, double rdy, double rdx,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t fd_pad_z0, int64_t fd_pad_y0, int64_t fd_pad_x0)
{
    const int64_t total = field_numel(lam_hy);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(lam_hy.scalar_type(), "em3d_adjoint_e_stage2", [&] {
        LAUNCH_EM3D(adjoint_e_stage2_kernel, scalar_t,
            work_hy_z.data_ptr<scalar_t>(), work_hz_y.data_ptr<scalar_t>(),
            work_hz_x.data_ptr<scalar_t>(), work_hx_z.data_ptr<scalar_t>(),
            work_hx_y.data_ptr<scalar_t>(), work_hy_x.data_ptr<scalar_t>(),
            lam_hy.data_ptr<scalar_t>(), lam_hz.data_ptr<scalar_t>(),
            lam_hx.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdz, (scalar_t)rdy, (scalar_t)rdx,
            (int)n_shots, (int)nz, (int)ny, (int)nx);
    });
}

void adjoint_h_stage1(
    torch::Tensor cq,
    torch::Tensor lam_hx, torch::Tensor lam_hy, torch::Tensor lam_hz,
    torch::Tensor m_lambda_ey_z, torch::Tensor m_lambda_ez_y,
    torch::Tensor m_lambda_ez_x, torch::Tensor m_lambda_ex_z,
    torch::Tensor m_lambda_ex_y, torch::Tensor m_lambda_ey_x,
    torch::Tensor work2_ey_z, torch::Tensor work2_ez_y, torch::Tensor work2_ez_x,
    torch::Tensor work2_ex_z, torch::Tensor work2_ex_y, torch::Tensor work2_ey_x,
    torch::Tensor azh, torch::Tensor bzh, torch::Tensor ayh, torch::Tensor byh,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor kzh, torch::Tensor kyh,
    torch::Tensor kxh,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t pml_z0, int64_t pml_z1, int64_t pml_y0, int64_t pml_y1,
    int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_z0, int64_t fd_pad_z1, int64_t fd_pad_y0, int64_t fd_pad_y1,
    int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t cq_batched)
{
    const int64_t total = field_numel(lam_hx);
    AT_DISPATCH_FLOATING_TYPES(lam_hx.scalar_type(), "em3d_adjoint_h_stage1", [&] {
        LAUNCH_EM3D(adjoint_h_stage1_kernel, scalar_t,
            cq.data_ptr<scalar_t>(),
            lam_hx.data_ptr<scalar_t>(), lam_hy.data_ptr<scalar_t>(),
            lam_hz.data_ptr<scalar_t>(),
            m_lambda_ey_z.data_ptr<scalar_t>(), m_lambda_ez_y.data_ptr<scalar_t>(),
            m_lambda_ez_x.data_ptr<scalar_t>(), m_lambda_ex_z.data_ptr<scalar_t>(),
            m_lambda_ex_y.data_ptr<scalar_t>(), m_lambda_ey_x.data_ptr<scalar_t>(),
            work2_ey_z.data_ptr<scalar_t>(), work2_ez_y.data_ptr<scalar_t>(),
            work2_ez_x.data_ptr<scalar_t>(), work2_ex_z.data_ptr<scalar_t>(),
            work2_ex_y.data_ptr<scalar_t>(), work2_ey_x.data_ptr<scalar_t>(),
            azh.data_ptr<scalar_t>(), bzh.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), byh.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), bxh.data_ptr<scalar_t>(),
            kzh.data_ptr<scalar_t>(), kyh.data_ptr<scalar_t>(),
            kxh.data_ptr<scalar_t>(),
            (int)n_shots, (int)nz, (int)ny, (int)nx,
            (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1,
            (int)cq_batched);
    });
}

void cq_grad(
    torch::Tensor cq,
    torch::Tensor lam_hx, torch::Tensor lam_hy, torch::Tensor lam_hz,
    torch::Tensor ex_store, torch::Tensor ey_store, torch::Tensor ez_store,
    torch::Tensor m_lambda_ey_z, torch::Tensor m_lambda_ez_y,
    torch::Tensor m_lambda_ez_x, torch::Tensor m_lambda_ex_z,
    torch::Tensor m_lambda_ex_y, torch::Tensor m_lambda_ey_x,
    torch::Tensor grad_cq,
    torch::Tensor azh, torch::Tensor kzh, torch::Tensor ayh, torch::Tensor kyh,
    torch::Tensor axh, torch::Tensor kxh,
    torch::Tensor c,
    double rdz, double rdy, double rdx,
    double scale, int64_t snap_off,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t fd_pad_z0, int64_t fd_pad_z1, int64_t fd_pad_y0, int64_t fd_pad_y1,
    int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t cq_batched)
{
    const int64_t total = field_numel(lam_hx);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(lam_hx.scalar_type(), "em3d_cq_grad", [&] {
        LAUNCH_EM3D(cq_grad_kernel, scalar_t,
            cq.data_ptr<scalar_t>(),
            lam_hx.data_ptr<scalar_t>(), lam_hy.data_ptr<scalar_t>(),
            lam_hz.data_ptr<scalar_t>(),
            ex_store.data_ptr<scalar_t>(), ey_store.data_ptr<scalar_t>(),
            ez_store.data_ptr<scalar_t>(),
            m_lambda_ey_z.data_ptr<scalar_t>(), m_lambda_ez_y.data_ptr<scalar_t>(),
            m_lambda_ez_x.data_ptr<scalar_t>(), m_lambda_ex_z.data_ptr<scalar_t>(),
            m_lambda_ex_y.data_ptr<scalar_t>(), m_lambda_ey_x.data_ptr<scalar_t>(),
            grad_cq.data_ptr<scalar_t>(),
            azh.data_ptr<scalar_t>(), kzh.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), kyh.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), kxh.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdz, (scalar_t)rdy, (scalar_t)rdx,
            (scalar_t)scale, (int64_t)snap_off,
            (int)n_shots, (int)nz, (int)ny, (int)nx,
            (int)cq_batched);
    });
}

void adjoint_h_stage2(
    torch::Tensor work2_ey_z, torch::Tensor work2_ez_y, torch::Tensor work2_ez_x,
    torch::Tensor work2_ex_z, torch::Tensor work2_ex_y, torch::Tensor work2_ey_x,
    torch::Tensor lam_ex, torch::Tensor lam_ey, torch::Tensor lam_ez,
    torch::Tensor c,
    double rdz, double rdy, double rdx,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t fd_pad_z0, int64_t fd_pad_y0, int64_t fd_pad_x0)
{
    const int64_t total = field_numel(lam_ey);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "em3d_adjoint_h_stage2", [&] {
        LAUNCH_EM3D(adjoint_h_stage2_kernel, scalar_t,
            work2_ey_z.data_ptr<scalar_t>(), work2_ez_y.data_ptr<scalar_t>(),
            work2_ez_x.data_ptr<scalar_t>(), work2_ex_z.data_ptr<scalar_t>(),
            work2_ex_y.data_ptr<scalar_t>(), work2_ey_x.data_ptr<scalar_t>(),
            lam_ex.data_ptr<scalar_t>(), lam_ey.data_ptr<scalar_t>(),
            lam_ez.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdz, (scalar_t)rdy, (scalar_t)rdx,
            (int)n_shots, (int)nz, (int)ny, (int)nx);
    });
}

// ---------------- whole-loop drivers ----------------
// One pybind call runs the entire forward/adjoint pass (deepwave-style):
// the per-step functions above are reused unchanged with the same argument
// order the Python loop used, so results are bitwise identical.

using nami_storage::ckpt_restore;
using nami_storage::ckpt_save;
using nami_storage::zero_buffers;

void forward_loop(
    torch::Tensor ca, torch::Tensor cb, torch::Tensor cq,
    std::vector<torch::Tensor> state,    // 18 flat buffers (see N_STATE order)
    torch::Tensor az, torch::Tensor bz, torch::Tensor ay, torch::Tensor by,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor kz, torch::Tensor ky,
    torch::Tensor kx,
    torch::Tensor azh, torch::Tensor bzh, torch::Tensor ayh, torch::Tensor byh,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor kzh, torch::Tensor kyh,
    torch::Tensor kxh,
    torch::Tensor ex_store, torch::Tensor ey_store, torch::Tensor ez_store,
    torch::Tensor curl_x_store, torch::Tensor curl_y_store, torch::Tensor curl_z_store,
    torch::Tensor f, torch::Tensor src_i, torch::Tensor r, torch::Tensor rec_i,
    torch::Tensor c,
    double rdz, double rdy, double rdx,
    int64_t nt, int64_t interval,
    int64_t pml_z0, int64_t pml_z1, int64_t pml_y0, int64_t pml_y1,
    int64_t pml_x0, int64_t pml_x1,
    int64_t ca_batched, int64_t cb_batched, int64_t cq_batched,
    int64_t fd_pad_z0, int64_t fd_pad_z1, int64_t fd_pad_y0, int64_t fd_pad_y1,
    int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t source_component, int64_t receiver_component,
    int64_t store,                       // 0 = checkpointed forward (snap_off 0)
    int64_t checkpoint_every, c10::optional<torch::Tensor> ckpt_state,
    // Wavefield I/O (deepwave-style): init_state/final_state use the N_STATE
    // layout [ex, ey, ez, hx, hy, hz, 6x H-step mem, 6x E-step mem] (same as
    // ckpt_state; see the N_STATE comment in em3d.py).
    c10::optional<torch::Tensor> init_state,
    c10::optional<torch::Tensor> final_state,
    // Per-step forward callback: called every `callback_frequency` steps as
    // callback(t, nt, ex, ey, ez, hx, hy, hz) — the current padded GPU
    // tensors (live views, no copy).  The PML memory variables are not
    // passed.
    py::object callback, int64_t callback_frequency)
{
    TORCH_CHECK(state.size() == 18, "nami em3d forward_loop: bad state size");
    const int64_t n_shots = state[0].size(0);
    const int64_t nz = state[0].size(1), ny = state[0].size(2), nx = state[0].size(3);
    const int64_t n_src = src_i.size(1), n_rec = rec_i.size(1);
    const int64_t numel = nz * ny * nx;
    const int64_t shot_count = n_shots * numel;
    const bool ckpt = ckpt_state.has_value() && checkpoint_every > 0;
    const bool has_cb = !callback.is_none();
    // State slot order (matches the N_STATE comment in em3d.py):
    //   [0:3] ex, ey, ez   [3:6] hx, hy, hz
    //   [6:12]  H-step memory (m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x)
    //   [12:18] E-step memory (m_hy_z, m_hz_y, m_hz_x, m_hx_z, m_hx_y, m_hy_x)
    std::vector<const torch::Tensor*> state_c;
    for (auto& t_ : state)
        state_c.push_back(&t_);

    // Optional initial state (continuation runs): all 18 buffers are flat, so
    // the state the first step reads (E at t = 0, H at t = -1/2, memory vars)
    // is the buffers themselves.
    if (init_state.has_value()) {
        auto c = *init_state;
        for (size_t j = 0; j < state.size(); ++j)
            state[j].copy_(c[j]);
    }

    for (int64_t t = 0; t < nt; ++t) {
        if (ckpt && t > 0 && t % checkpoint_every == 0)
            ckpt_save(*ckpt_state, t / checkpoint_every - 1, state_c);
        if (has_cb && t % callback_frequency == 0) {
            py::gil_scoped_acquire gil;
            callback(t, nt, state[0], state[1], state[2], state[3],
                     state[4], state[5]);
        }
        const int64_t snap_off = store ? (t / interval) * shot_count : 0;
        step_h(
            cq, state[0], state[1], state[2], state[3], state[4], state[5],
            state[6], state[7], state[8], state[9], state[10], state[11],
            azh, bzh, ayh, byh, axh, bxh, kzh, kyh, kxh,
            c,
            rdz, rdy, rdx,
            n_shots, nz, ny, nx,
            pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
            fd_pad_z0, fd_pad_y0, fd_pad_x0, cq_batched);
        step_e_storage(
            ca, cb, state[3], state[4], state[5], state[0], state[1], state[2],
            state[12], state[13], state[14], state[15], state[16], state[17],
            ex_store, ey_store, ez_store,
            curl_x_store, curl_y_store, curl_z_store,
            az, bz, ay, by, ax, bx, kz, ky, kx,
            c,
            rdz, rdy, rdx, t, interval, snap_off,
            n_shots, nz, ny, nx,
            pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
            fd_pad_z0, fd_pad_z1, fd_pad_y0, fd_pad_y1,
            fd_pad_x0, fd_pad_x1,
            ca_batched, cb_batched);
        if (n_src > 0)
            inject(state[source_component], f, src_i, t, n_shots, n_src, numel);
        if (n_rec > 0)
            record(state[receiver_component], r, rec_i, t, n_shots, n_rec, numel);
    }

    // Optional final state (next-step fields for continuation): the buffers
    // are flat, so after nt steps they already hold the state the next
    // forward step would read — copying them makes a split run (continuation
    // via init_state) bitwise match a one-shot run.
    if (final_state.has_value()) {
        auto c = *final_state;
        for (size_t j = 0; j < state.size(); ++j)
            c[j].copy_(state[j]);
    }
}

void adjoint_loop(
    torch::Tensor ca, torch::Tensor cb, torch::Tensor cq,
    torch::Tensor lam_ex, torch::Tensor lam_ey, torch::Tensor lam_ez,
    torch::Tensor lam_hx, torch::Tensor lam_hy, torch::Tensor lam_hz,
    torch::Tensor m_lambda_hy_z, torch::Tensor m_lambda_hz_y,
    torch::Tensor m_lambda_hz_x, torch::Tensor m_lambda_hx_z,
    torch::Tensor m_lambda_hx_y, torch::Tensor m_lambda_hy_x,
    torch::Tensor work_hy_z, torch::Tensor work_hz_y, torch::Tensor work_hz_x,
    torch::Tensor work_hx_z, torch::Tensor work_hx_y, torch::Tensor work_hy_x,
    torch::Tensor m_lambda_ey_z, torch::Tensor m_lambda_ez_y,
    torch::Tensor m_lambda_ez_x, torch::Tensor m_lambda_ex_z,
    torch::Tensor m_lambda_ex_y, torch::Tensor m_lambda_ey_x,
    torch::Tensor work2_ey_z, torch::Tensor work2_ez_y, torch::Tensor work2_ez_x,
    torch::Tensor work2_ex_z, torch::Tensor work2_ex_y, torch::Tensor work2_ey_x,
    torch::Tensor az, torch::Tensor bz, torch::Tensor ay, torch::Tensor by,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor kz, torch::Tensor ky,
    torch::Tensor kx,
    torch::Tensor azh, torch::Tensor bzh, torch::Tensor ayh, torch::Tensor byh,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor kzh, torch::Tensor kyh,
    torch::Tensor kxh,
    torch::Tensor ex_store, torch::Tensor ey_store, torch::Tensor ez_store,
    torch::Tensor curl_x_store, torch::Tensor curl_y_store, torch::Tensor curl_z_store,
    torch::Tensor grad_ca, torch::Tensor grad_cb, torch::Tensor grad_cq,
    torch::Tensor grad_r, torch::Tensor rec_i,
    torch::Tensor grad_f, torch::Tensor src_i, torch::Tensor f,
    std::vector<torch::Tensor> state_f,  // replay state (empty = full storage)
    torch::Tensor c,
    double rdz, double rdy, double rdx, double scale,
    int64_t nt, int64_t interval,
    int64_t pml_z0, int64_t pml_z1, int64_t pml_y0, int64_t pml_y1,
    int64_t pml_x0, int64_t pml_x1,
    int64_t ca_batched, int64_t cb_batched, int64_t cq_batched,
    int64_t fd_pad_z0, int64_t fd_pad_z1, int64_t fd_pad_y0, int64_t fd_pad_y1,
    int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t source_component, int64_t receiver_component,
    torch::Tensor segments,              // int64 [n_seg, 2] on CPU; empty = full storage
    c10::optional<torch::Tensor> ckpt_state)
{
    const int64_t n_shots = lam_ex.size(0);
    const int64_t nz = lam_ex.size(1), ny = lam_ex.size(2), nx = lam_ex.size(3);
    const int64_t n_src = src_i.size(1), n_rec = rec_i.size(1);
    const int64_t numel = nz * ny * nx;
    const int64_t shot_count = n_shots * numel;
    std::vector<torch::Tensor> lam_e{lam_ex, lam_ey, lam_ez};

    auto adjoint_at = [&](int64_t t, int64_t snap_off) {
        if (n_rec > 0)
            record_grad_r(lam_e[receiver_component], grad_r, rec_i,
                          t, n_shots, n_rec, numel);
        if (n_src > 0)
            record_grad_f(lam_e[source_component], grad_f, src_i,
                          t, n_shots, n_src, numel);
        if (t % interval == 0)
            coeff_grad(
                lam_ex, lam_ey, lam_ez,
                ex_store, ey_store, ez_store,
                curl_x_store, curl_y_store, curl_z_store,
                grad_ca, grad_cb,
                scale, snap_off,
                n_shots, nz, ny, nx,
                fd_pad_z0, fd_pad_z1, fd_pad_y0, fd_pad_y1,
                fd_pad_x0, fd_pad_x1);
        adjoint_e_stage1(
            ca, cb, lam_ex, lam_ey, lam_ez,
            m_lambda_hy_z, m_lambda_hz_y, m_lambda_hz_x,
            m_lambda_hx_z, m_lambda_hx_y, m_lambda_hy_x,
            work_hy_z, work_hz_y, work_hz_x,
            work_hx_z, work_hx_y, work_hy_x,
            az, bz, ay, by, ax, bx, kz, ky, kx,
            n_shots, nz, ny, nx,
            pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
            fd_pad_z0, fd_pad_z1, fd_pad_y0, fd_pad_y1,
            fd_pad_x0, fd_pad_x1,
            ca_batched, cb_batched);
        adjoint_e_stage2(
            work_hy_z, work_hz_y, work_hz_x,
            work_hx_z, work_hx_y, work_hy_x,
            lam_hy, lam_hz, lam_hx, c,
            rdz, rdy, rdx,
            n_shots, nz, ny, nx,
            fd_pad_z0, fd_pad_y0, fd_pad_x0);
        adjoint_h_stage1(
            cq, lam_hx, lam_hy, lam_hz,
            m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x,
            m_lambda_ex_z, m_lambda_ex_y, m_lambda_ey_x,
            work2_ey_z, work2_ez_y, work2_ez_x,
            work2_ex_z, work2_ex_y, work2_ey_x,
            azh, bzh, ayh, byh, axh, bxh, kzh, kyh, kxh,
            n_shots, nz, ny, nx,
            pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
            fd_pad_z0, fd_pad_z1, fd_pad_y0, fd_pad_y1,
            fd_pad_x0, fd_pad_x1,
            cq_batched);
        if (t % interval == 0)
            cq_grad(
                cq, lam_hx, lam_hy, lam_hz,
                ex_store, ey_store, ez_store,
                m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x,
                m_lambda_ex_z, m_lambda_ex_y, m_lambda_ey_x,
                grad_cq,
                azh, kzh, ayh, kyh, axh, kxh,
                c,
                rdz, rdy, rdx,
                scale, snap_off,
                n_shots, nz, ny, nx,
                fd_pad_z0, fd_pad_z1, fd_pad_y0, fd_pad_y1,
                fd_pad_x0, fd_pad_x1,
                cq_batched);
        adjoint_h_stage2(
            work2_ey_z, work2_ez_y, work2_ez_x,
            work2_ex_z, work2_ex_y, work2_ey_x,
            lam_ex, lam_ey, lam_ez, c,
            rdz, rdy, rdx,
            n_shots, nz, ny, nx,
            fd_pad_z0, fd_pad_y0, fd_pad_x0);
    };

    if (segments.numel() == 0) {
        for (int64_t t = nt - 1; t >= 0; --t)
            adjoint_at(t, (t / interval) * shot_count);
        return;
    }

    // Checkpointed backward: per segment, restore the forward state, replay
    // the forward steps to regenerate the w snapshots, then run the adjoint.
    TORCH_CHECK(state_f.size() == 18,
                "nami em3d adjoint_loop: bad replay state size");
    std::vector<torch::Tensor*> state_f_p;
    for (auto& t_ : state_f)
        state_f_p.push_back(&t_);
    auto seg = segments.accessor<int64_t, 2>();
    const int64_t n_seg = segments.size(0);
    for (int64_t k = n_seg - 1; k >= 0; --k) {
        const int64_t s0 = seg[k][0], s1 = seg[k][1];
        if (s0 > 0) {
            ckpt_restore(*ckpt_state, k - 1, state_f_p);
        } else {
            zero_buffers(state_f_p);
        }
        for (int64_t t = s0; t < s1; ++t) {
            const int64_t snap_off = ((t - s0) / interval) * shot_count;
            step_h(
                cq, state_f[0], state_f[1], state_f[2],
                state_f[3], state_f[4], state_f[5],
                state_f[6], state_f[7], state_f[8],
                state_f[9], state_f[10], state_f[11],
                azh, bzh, ayh, byh, axh, bxh, kzh, kyh, kxh,
                c,
                rdz, rdy, rdx,
                n_shots, nz, ny, nx,
                pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
                fd_pad_z0, fd_pad_y0, fd_pad_x0, cq_batched);
            step_e_storage(
                ca, cb, state_f[3], state_f[4], state_f[5],
                state_f[0], state_f[1], state_f[2],
                state_f[12], state_f[13], state_f[14],
                state_f[15], state_f[16], state_f[17],
                ex_store, ey_store, ez_store,
                curl_x_store, curl_y_store, curl_z_store,
                az, bz, ay, by, ax, bx, kz, ky, kx,
                c,
                rdz, rdy, rdx, t, interval, snap_off,
                n_shots, nz, ny, nx,
                pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
                fd_pad_z0, fd_pad_z1, fd_pad_y0, fd_pad_y1,
                fd_pad_x0, fd_pad_x1,
                ca_batched, cb_batched);
            if (n_src > 0)
                inject(state_f[source_component], f, src_i,
                       t, n_shots, n_src, numel);
        }
        for (int64_t t = s1 - 1; t >= s0; --t)
            adjoint_at(t, ((t - s0) / interval) * shot_count);
    }
}

// ---------------- pybind11 bindings ----------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("step_h", &step_h);
    m.def("step_e_storage", &step_e_storage);
    m.def("inject", &inject);
    m.def("record", &record);
    m.def("record_grad_r", &record_grad_r);
    m.def("record_grad_f", &record_grad_f);
    m.def("coeff_grad", &coeff_grad);
    m.def("adjoint_e_stage1", &adjoint_e_stage1);
    m.def("adjoint_e_stage2", &adjoint_e_stage2);
    m.def("adjoint_h_stage1", &adjoint_h_stage1);
    m.def("cq_grad", &cq_grad);
    m.def("adjoint_h_stage2", &adjoint_h_stage2);
    m.def("forward_loop", &forward_loop);
    m.def("adjoint_loop", &adjoint_loop);
    NAMI_STORAGE_PYBIND(m);
}
