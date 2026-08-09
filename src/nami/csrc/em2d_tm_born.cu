// nami EM 2D TM Born: first-order scattering and exact discrete-transpose adjoint.
// The background and scattered fields share the em2d_tm staggered grid and CPML.

#include <torch/extension.h>
#include "storage.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <pybind11/stl.h>

#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------- coefficient-driven staggered difference operators ----------------
// Same conventions as em2d_tm.cu / elastic2d.cu: ``c`` is the staggered-grid
// first-derivative coefficient array (max radius 4, zero-padded for lower
// orders).  The stencil radius is a compile-time template parameter FD_PAD
// (== accuracy // 2 == fd_pad_y0 == fd_pad_x0 for these Born grids), so the
// loops fully unroll with constant offsets and no out-of-bounds reads occur.
//
// diff_int_*: derivative at a half-integer grid point of a field stored at
// integer points; diff_half_*: derivative at an integer grid point of a field
// stored at half-integer points.  The two operators are exact discrete
// transposes of each other (up to the sign convention of the curl), which is
// what the adjoint kernels rely on.
//
// Templating the FD order at compile time is numerically bitwise-identical to
// the old runtime-fd_pad loops: same coefficients, same operation order; only
// the loop bound becomes a compile-time constant.
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

// ==================== EM (2D TM) Born ====================

// ---------------- forward: H half-step (background + scattered) ----------------
// hx -= cq*dey_dy  and  dHx -= cq*ddey_dy + dcq*dey_dy  (and the x-analogue
// for hz/dHz).  The PML-modified derivatives are snapshotted on the sampling
// interval for the cq/dcq gradients.
template <typename T, int FD_PAD>
__global__ void born_step_h_kernel(
    const T* __restrict__ cq, const T* __restrict__ dcq,
    const T* __restrict__ ey, const T* __restrict__ dEy,
    T* __restrict__ hx, T* __restrict__ hz,
    T* __restrict__ dHx, T* __restrict__ dHz,
    T* __restrict__ m_ey_z, T* __restrict__ m_ey_x,
    T* __restrict__ dm_ey_z, T* __restrict__ dm_ey_x,
    T* __restrict__ dey_dy_store, T* __restrict__ dey_dx_store,
    T* __restrict__ ddey_dy_store, T* __restrict__ ddey_dx_store,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ kyh, const T* __restrict__ kxh,
    const T* __restrict__ c,
    T rdy, T rdx,
    int t, int interval,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int cq_batched, int dcq_batched, int store, int64_t snap_off)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < ny && x < nx) {
        const int s_cq = cq_batched ? s : 0;
        const int s_dcq = dcq_batched ? s : 0;
        const T cq_val = cq[(long)s_cq * ny * nx + (long)y * nx + x];
        const T dcq_val = dcq[(long)s_dcq * ny * nx + (long)y * nx + x];
        const long off = ((long)s * ny + y) * nx + x;
        T dey_dy = (T)0;
        if (y >= FD_PAD && y < ny - FD_PAD)
            dey_dy = diff_int_y<T, FD_PAD>(ey, off, c, rdy, nx);
        T ddey_dy = (T)0;
        if (y >= FD_PAD && y < ny - FD_PAD)
            ddey_dy = diff_int_y<T, FD_PAD>(dEy, off, c, rdy, nx);
        if (y < pml_y0 || y >= pml_y1 - 1) {
            m_ey_z[off] = byh[y] * m_ey_z[off] + ayh[y] * dey_dy;
            dm_ey_z[off] = byh[y] * dm_ey_z[off] + ayh[y] * ddey_dy;
            dey_dy = dey_dy / kyh[y] + m_ey_z[off];
            ddey_dy = ddey_dy / kyh[y] + dm_ey_z[off];
        }
        if (store && t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            dey_dy_store[soff] = dey_dy;
            ddey_dy_store[soff] = ddey_dy;
        }
        hx[off] -= cq_val * dey_dy;
        dHx[off] -= cq_val * ddey_dy + dcq_val * dey_dy;
        T dey_dx = (T)0;
        if (x >= FD_PAD && x < nx - FD_PAD)
            dey_dx = diff_int_x<T, FD_PAD>(ey, off, c, rdx);
        T ddey_dx = (T)0;
        if (x >= FD_PAD && x < nx - FD_PAD)
            ddey_dx = diff_int_x<T, FD_PAD>(dEy, off, c, rdx);
        if (x < pml_x0 || x >= pml_x1 - 1) {
            m_ey_x[off] = bxh[x] * m_ey_x[off] + axh[x] * dey_dx;
            dm_ey_x[off] = bxh[x] * dm_ey_x[off] + axh[x] * ddey_dx;
            dey_dx = dey_dx / kxh[x] + m_ey_x[off];
            ddey_dx = ddey_dx / kxh[x] + dm_ey_x[off];
        }
        if (store && t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            dey_dx_store[soff] = dey_dx;
            ddey_dx_store[soff] = ddey_dx;
        }
        hz[off] += cq_val * dey_dx;
        dHz[off] += cq_val * ddey_dx + dcq_val * dey_dx;
    }
}

// ---------------- forward: combined bg + scattered E update with snapshots ----------------
// Computes the background curl (from Hx/Hz with the integer CPML memories)
// and the scattered curl (from dHx/dHz), snapshots the pre-update fields
// and both curls on every step, then updates Ey and dEy:
//   Ey  = ca*Ey  + cb*curl
//   dEy = ca*dEy + cb*dcurl + dca*Ey_old + dcb*curl
template <typename T, int FD_PAD>
__global__ void born_step_e_kernel(
    const T* __restrict__ ca, const T* __restrict__ cb,
    const T* __restrict__ dca, const T* __restrict__ dcb,
    const T* __restrict__ hx, const T* __restrict__ hz,
    const T* __restrict__ dHx, const T* __restrict__ dHz,
    T* __restrict__ ey, T* __restrict__ dEy,
    T* __restrict__ m_hx_z, T* __restrict__ m_hz_x,
    T* __restrict__ dm_hx_z, T* __restrict__ dm_hz_x,
    T* __restrict__ ey_store, T* __restrict__ curl_store,
    T* __restrict__ dEy_store, T* __restrict__ dcurl_store,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ ky, const T* __restrict__ kx,
    const T* __restrict__ c,
    T rdy, T rdx,
    int t, int interval,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int ca_batched, int cb_batched, int dca_batched, int dcb_batched,
    int store, int64_t snap_off)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= FD_PAD && x >= FD_PAD && y < ny - (FD_PAD - 1) &&
        x < nx - (FD_PAD - 1)) {
        const int s_ca = ca_batched ? s : 0;
        const int s_cb = cb_batched ? s : 0;
        const int s_dca = dca_batched ? s : 0;
        const int s_dcb = dcb_batched ? s : 0;
        const long off = ((long)s * ny + y) * nx + x;
        T dhz_dx = diff_half_x<T, FD_PAD>(hz, off, c, rdx);
        if (x < pml_x0 || x >= pml_x1) {
            m_hz_x[off] = bx[x] * m_hz_x[off] + ax[x] * dhz_dx;
            dhz_dx = dhz_dx / kx[x] + m_hz_x[off];
        }
        T dhx_dy = diff_half_y<T, FD_PAD>(hx, off, c, rdy, nx);
        if (y < pml_y0 || y >= pml_y1) {
            m_hx_z[off] = by[y] * m_hx_z[off] + ay[y] * dhx_dy;
            dhx_dy = dhx_dy / ky[y] + m_hx_z[off];
        }
        const T curl = dhz_dx - dhx_dy;
        T ddHz_dx = diff_half_x<T, FD_PAD>(dHz, off, c, rdx);
        if (x < pml_x0 || x >= pml_x1) {
            dm_hz_x[off] = bx[x] * dm_hz_x[off] + ax[x] * ddHz_dx;
            ddHz_dx = ddHz_dx / kx[x] + dm_hz_x[off];
        }
        T ddHx_dy = diff_half_y<T, FD_PAD>(dHx, off, c, rdy, nx);
        if (y < pml_y0 || y >= pml_y1) {
            dm_hx_z[off] = by[y] * dm_hx_z[off] + ay[y] * ddHx_dy;
            ddHx_dy = ddHx_dy / ky[y] + dm_hx_z[off];
        }
        const T dcurl = ddHz_dx - ddHx_dy;
        const T ey_old = ey[off];
        const T dEy_old = dEy[off];
        // snap_off + spatial (same layout as full-wave / born_step_h)
        if (store && t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            ey_store[soff] = ey_old;
            curl_store[soff] = curl;
            dEy_store[soff] = dEy_old;
            dcurl_store[soff] = dcurl;
        }
        const T ca_val = ca[(long)s_ca * ny * nx + (long)y * nx + x];
        const T cb_val = cb[(long)s_cb * ny * nx + (long)y * nx + x];
        const T dca_val = dca[(long)s_dca * ny * nx + (long)y * nx + x];
        const T dcb_val = dcb[(long)s_dcb * ny * nx + (long)y * nx + x];
        ey[off] = ca_val * ey_old + cb_val * curl;
        dEy[off] = ca_val * dEy_old + cb_val * dcurl + dca_val * ey_old + dcb_val * curl;
    }
}

// ---------------- forward: source injection / receiver recording ----------------
template <typename T>
__global__ void born_inject_kernel(
    T* __restrict__ ey, T* __restrict__ dEy,
    const T* __restrict__ f_bg, const T* __restrict__ f_sc,
    const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        if (idx >= 0) {
            const long off = (long)s * ny_nx + idx;
            const long fidx = ((long)t * n_shots + s) * n_src + k;
            ey[off] += f_bg[fidx];
            dEy[off] += f_sc[fidx];
        }
    }
}

template <typename T>
__global__ void born_record_kernel(
    const T* __restrict__ dEy, T* __restrict__ r, const long* __restrict__ rec_i,
    int t, int n_shots, int n_rec, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            r[(((long)t * n_shots + s) * n_rec + k)] = dEy[(long)s * ny_nx + idx];
    }
}

// ---------------- backward: source/receiver gradient exchange ----------------
template <typename T>
__global__ void born_record_grad_r_kernel(
    T* __restrict__ lam_dEy, const T* __restrict__ grad_r, const long* __restrict__ rec_i,
    int t, int n_shots, int n_rec, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            lam_dEy[(long)s * ny_nx + idx] += grad_r[(((long)t * n_shots + s) * n_rec + k)];
    }
}

template <typename T>
__global__ void born_record_grad_f_kernel(
    const T* __restrict__ lam_ey, const T* __restrict__ lam_dEy,
    T* __restrict__ grad_f_bg, T* __restrict__ grad_f_sc,
    const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        const long fidx = ((long)t * n_shots + s) * n_src + k;
        if (idx >= 0) {
            const long off = (long)s * ny_nx + idx;
            grad_f_bg[fidx] = lam_ey[off];
            grad_f_sc[fidx] = lam_dEy[off];
        } else {
            grad_f_bg[fidx] = (T)0;
            grad_f_sc[fidx] = (T)0;
        }
    }
}

// ---------------- backward: ca/cb/dca/dcb model gradients ----------------
template <typename T, int FD_PAD>
__global__ void born_coeff_grad_kernel(
    const T* __restrict__ lam_ey, const T* __restrict__ lam_dEy,
    const T* __restrict__ ey_store, const T* __restrict__ curl_store,
    const T* __restrict__ dEy_store, const T* __restrict__ dcurl_store,
    T* __restrict__ grad_ca, T* __restrict__ grad_cb,
    T* __restrict__ grad_dca, T* __restrict__ grad_dcb,
    int t, int interval, T scale, int64_t snap_off,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= FD_PAD && x >= FD_PAD && y < ny - (FD_PAD - 1) &&
        x < nx - (FD_PAD - 1)) {
        const long off = ((long)s * ny + y) * nx + x;
        const long soff = snap_off + ((long)s * ny + y) * nx + x;
        const T lam = lam_ey[off] * scale;
        const T lam_d = lam_dEy[off] * scale;
        const T ey_v = ey_store[soff];
        const T curl_v = curl_store[soff];
        grad_ca[off] += lam * ey_v + lam_d * dEy_store[soff];
        grad_cb[off] += lam * curl_v + lam_d * dcurl_store[soff];
        grad_dca[off] += lam_d * ey_v;
        grad_dcb[off] += lam_d * curl_v;
    }
}

// ---------------- backward: transpose of the combined E update, stage 1 ----------------
// From the post-E-update adjoints:
//   lam_ey_new  = ca*lam_ey  + dca*lam_dEy
//   lam_dEy_new = ca*lam_dEy
//   g_bg = cb*lam_ey + dcb*lam_dEy   (background curl adjoint)
//   g_sc = cb*lam_dEy                (scattered curl adjoint)
// and builds the four work arrays with the time-reversed CPML recursions.
template <typename T, int FD_PAD>
__global__ void born_adjoint_e_stage1_kernel(
    const T* __restrict__ ca, const T* __restrict__ cb,
    const T* __restrict__ dca, const T* __restrict__ dcb,
    T* __restrict__ lam_ey, T* __restrict__ lam_dEy,
    T* __restrict__ m_lambda_hx_z, T* __restrict__ m_lambda_hz_x,
    T* __restrict__ dm_lambda_hx_z, T* __restrict__ dm_lambda_hz_x,
    T* __restrict__ work_x, T* __restrict__ work_y,
    T* __restrict__ work_dx, T* __restrict__ work_dy,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ ky, const T* __restrict__ kx,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int ca_batched, int cb_batched, int dca_batched, int dcb_batched)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < ny && x < nx) {
        const long off = ((long)s * ny + y) * nx + x;
        if (y >= FD_PAD && x >= FD_PAD && y < ny - (FD_PAD - 1) &&
            x < nx - (FD_PAD - 1)) {
            const int s_ca = ca_batched ? s : 0;
            const int s_cb = cb_batched ? s : 0;
            const int s_dca = dca_batched ? s : 0;
            const int s_dcb = dcb_batched ? s : 0;
            const T ca_val = ca[(long)s_ca * ny * nx + (long)y * nx + x];
            const T cb_val = cb[(long)s_cb * ny * nx + (long)y * nx + x];
            const T dca_val = dca[(long)s_dca * ny * nx + (long)y * nx + x];
            const T dcb_val = dcb[(long)s_dcb * ny * nx + (long)y * nx + x];
            const T lam_ey_v = lam_ey[off];
            const T lam_dEy_v = lam_dEy[off];
            const T g_bg = cb_val * lam_ey_v + dcb_val * lam_dEy_v;
            const T g_sc = cb_val * lam_dEy_v;
            lam_ey[off] = ca_val * lam_ey_v + dca_val * lam_dEy_v;
            lam_dEy[off] = ca_val * lam_dEy_v;
            if (x < pml_x0 || x >= pml_x1) {
                const T w = g_bg + bx[x] * m_lambda_hz_x[off];
                work_x[off] = g_bg / kx[x] + ax[x] * w;
                m_lambda_hz_x[off] = w;
                const T wd = g_sc + bx[x] * dm_lambda_hz_x[off];
                work_dx[off] = g_sc / kx[x] + ax[x] * wd;
                dm_lambda_hz_x[off] = wd;
            } else {
                work_x[off] = g_bg;
                work_dx[off] = g_sc;
            }
            if (y < pml_y0 || y >= pml_y1) {
                const T w = -g_bg + by[y] * m_lambda_hx_z[off];
                work_y[off] = -g_bg / ky[y] + ay[y] * w;
                m_lambda_hx_z[off] = w;
                const T wd = -g_sc + by[y] * dm_lambda_hx_z[off];
                work_dy[off] = -g_sc / ky[y] + ay[y] * wd;
                dm_lambda_hx_z[off] = wd;
            } else {
                work_y[off] = -g_bg;
                work_dy[off] = -g_sc;
            }
        } else {
            work_x[off] = (T)0;
            work_y[off] = (T)0;
            work_dx[off] = (T)0;
            work_dy[off] = (T)0;
        }
    }
}

// ---------------- backward: transpose of the combined E update, stage 2 ----------------
template <typename T, int FD_PAD>
__global__ void born_adjoint_e_stage2_kernel(
    const T* __restrict__ work_x, const T* __restrict__ work_y,
    const T* __restrict__ work_dx, const T* __restrict__ work_dy,
    T* __restrict__ lam_hx, T* __restrict__ lam_hz,
    T* __restrict__ lam_dhx, T* __restrict__ lam_dhz,
    const T* __restrict__ c,
    T rdy, T rdx,
    int n_shots, int ny, int nx)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < ny && x < nx) {
        const long off = ((long)s * ny + y) * nx + x;
        T acc_x = (T)0;
        T acc_dx = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const T ck = c[k - 1];
            if (x - k + 1 >= 0) {
                acc_x += ck * work_x[off - k + 1];
                acc_dx += ck * work_dx[off - k + 1];
            }
            if (x + k < nx) {
                acc_x -= ck * work_x[off + k];
                acc_dx -= ck * work_dx[off + k];
            }
        }
        lam_hz[off] += acc_x * rdx;
        lam_dhz[off] += acc_dx * rdx;
        T acc_y = (T)0;
        T acc_dy = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const T ck = c[k - 1];
            if (y - k + 1 >= 0) {
                acc_y += ck * work_y[off - (k - 1) * nx];
                acc_dy += ck * work_dy[off - (k - 1) * nx];
            }
            if (y + k < ny) {
                acc_y -= ck * work_y[off + k * nx];
                acc_dy -= ck * work_dy[off + k * nx];
            }
        }
        lam_hx[off] += acc_y * rdy;
        lam_dhx[off] += acc_dy * rdy;
    }
}

// ---------------- backward: transpose of the combined H half-step, stage 1 ----------------
// From lam_hx/lam_hz (bg) and lam_dhx/lam_dhz (sc):
//   g2_x  = cq*lam_hz  + dcq*lam_dhz     (bg)    g2_dx = cq*lam_dhz   (sc)
//   g2_y  = -cq*lam_hx - dcq*lam_dhx     (bg)    g2_dy = -cq*lam_dhx  (sc)
// with separate time-reversed half-integer PML memories for each field.
template <typename T, int FD_PAD>
__global__ void born_adjoint_h_stage1_kernel(
    const T* __restrict__ cq, const T* __restrict__ dcq,
    const T* __restrict__ lam_hx, const T* __restrict__ lam_hz,
    const T* __restrict__ lam_dhx, const T* __restrict__ lam_dhz,
    T* __restrict__ m_lambda_ey_x, T* __restrict__ m_lambda_ey_z,
    T* __restrict__ dm_lambda_ey_x, T* __restrict__ dm_lambda_ey_z,
    T* __restrict__ work2_x, T* __restrict__ work2_y,
    T* __restrict__ work2_dx, T* __restrict__ work2_dy,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ kyh, const T* __restrict__ kxh,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int cq_batched, int dcq_batched)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    int pml_y1h = pml_y1 - 1;
    if (pml_y1 <= pml_y0)
        pml_y1h = pml_y0;
    int pml_x1h = pml_x1 - 1;
    if (pml_x1 <= pml_x0)
        pml_x1h = pml_x0;
    if (y < ny && x < nx) {
        const long off = ((long)s * ny + y) * nx + x;
        if (y >= FD_PAD && x >= FD_PAD && y < ny - (FD_PAD - 1) &&
            x < nx - (FD_PAD - 1)) {
            const int s_cq = cq_batched ? s : 0;
            const int s_dcq = dcq_batched ? s : 0;
            const T cq_val = cq[(long)s_cq * ny * nx + (long)y * nx + x];
            const T dcq_val = dcq[(long)s_dcq * ny * nx + (long)y * nx + x];
            if (x < nx - FD_PAD) {
                const T g2 = cq_val * lam_hz[off] + dcq_val * lam_dhz[off];
                const T g2d = cq_val * lam_dhz[off];
                if (x < pml_x0 || x >= pml_x1h) {
                    const T w = g2 + bxh[x] * m_lambda_ey_x[off];
                    work2_x[off] = g2 / kxh[x] + axh[x] * w;
                    m_lambda_ey_x[off] = w;
                    const T wd = g2d + bxh[x] * dm_lambda_ey_x[off];
                    work2_dx[off] = g2d / kxh[x] + axh[x] * wd;
                    dm_lambda_ey_x[off] = wd;
                } else {
                    work2_x[off] = g2;
                    work2_dx[off] = g2d;
                }
            } else {
                work2_x[off] = (T)0;
                work2_dx[off] = (T)0;
            }
            if (y < ny - FD_PAD) {
                const T g2 = -cq_val * lam_hx[off] - dcq_val * lam_dhx[off];
                const T g2d = -cq_val * lam_dhx[off];
                if (y < pml_y0 || y >= pml_y1h) {
                    const T w = g2 + byh[y] * m_lambda_ey_z[off];
                    work2_y[off] = g2 / kyh[y] + ayh[y] * w;
                    m_lambda_ey_z[off] = w;
                    const T wd = g2d + byh[y] * dm_lambda_ey_z[off];
                    work2_dy[off] = g2d / kyh[y] + ayh[y] * wd;
                    dm_lambda_ey_z[off] = wd;
                } else {
                    work2_y[off] = g2;
                    work2_dy[off] = g2d;
                }
            } else {
                work2_y[off] = (T)0;
                work2_dy[off] = (T)0;
            }
        } else {
            work2_x[off] = (T)0;
            work2_y[off] = (T)0;
            work2_dx[off] = (T)0;
            work2_dy[off] = (T)0;
        }
    }
}

// ---------------- backward: cq / dcq model gradients ----------------
// Uses the snapshotted PML-modified H derivatives, so no diff recomputation
// or reconstruction is needed: grad_cq gets the bg and sc contributions,
// grad_dcq only the dcq*deriv(Ey_bg) terms.
template <typename T, int FD_PAD>
__global__ void born_cq_grad_kernel(
    const T* __restrict__ lam_hx, const T* __restrict__ lam_hz,
    const T* __restrict__ lam_dhx, const T* __restrict__ lam_dhz,
    const T* __restrict__ dey_dy_store, const T* __restrict__ dey_dx_store,
    const T* __restrict__ ddey_dy_store, const T* __restrict__ ddey_dx_store,
    T* __restrict__ grad_cq, T* __restrict__ grad_dcq,
    int t, int interval, T scale, int64_t snap_off,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= FD_PAD && x >= FD_PAD && y < ny - (FD_PAD - 1) &&
        x < nx - (FD_PAD - 1)) {
        const long off = ((long)s * ny + y) * nx + x;
        const long soff = snap_off + ((long)s * ny + y) * nx + x;
        T term_cq = (T)0;
        T term_dcq = (T)0;
        if (y < ny - FD_PAD) {
            const T dy_bg = dey_dy_store[soff];
            const T dy_sc = ddey_dy_store[soff];
            term_cq += -lam_hx[off] * dy_bg - lam_dhx[off] * dy_sc;
            term_dcq += -lam_dhx[off] * dy_bg;
        }
        if (x < nx - FD_PAD) {
            const T dx_bg = dey_dx_store[soff];
            const T dx_sc = ddey_dx_store[soff];
            term_cq += lam_hz[off] * dx_bg + lam_dhz[off] * dx_sc;
            term_dcq += lam_dhz[off] * dx_bg;
        }
        grad_cq[off] += scale * term_cq;
        grad_dcq[off] += scale * term_dcq;
    }
}

// ---------------- backward: transpose of the combined H half-step, stage 2 ----------------
template <typename T, int FD_PAD>
__global__ void born_adjoint_h_stage2_kernel(
    const T* __restrict__ work2_x, const T* __restrict__ work2_y,
    const T* __restrict__ work2_dx, const T* __restrict__ work2_dy,
    T* __restrict__ lam_ey, T* __restrict__ lam_dEy,
    const T* __restrict__ c,
    T rdy, T rdx,
    int n_shots, int ny, int nx)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < ny && x < nx) {
        const long off = ((long)s * ny + y) * nx + x;
        T sum_x = (T)0;
        T sum_dx = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const T ck = c[k - 1];
            if (x >= k) {
                sum_x += ck * work2_x[off - k];
                sum_dx += ck * work2_dx[off - k];
            }
            if (x + k - 1 < nx) {
                sum_x -= ck * work2_x[off + k - 1];
                sum_dx -= ck * work2_dx[off + k - 1];
            }
        }
        T sum_y = (T)0;
        T sum_dy = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const T ck = c[k - 1];
            if (y >= k) {
                sum_y += ck * work2_y[off - k * nx];
                sum_dy += ck * work2_dy[off - k * nx];
            }
            if (y + k - 1 < ny) {
                sum_y -= ck * work2_y[off + (k - 1) * nx];
                sum_dy -= ck * work2_dy[off + (k - 1) * nx];
            }
        }
        lam_ey[off] += sum_x * rdx + sum_y * rdy;
        lam_dEy[off] += sum_dx * rdx + sum_dy * rdy;
    }
}

// ---------------- launchers ----------------
#define LAUNCH_GRID_FD(KERN, T, FP, ...)                                       \
    {                                                                          \
        dim3 block(16, 16);                                                    \
        dim3 grid((nx + 15) / 16, (ny + 15) / 16, n_shots);                    \
        KERN<T, FP><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami em2d_tm_born " #KERN " failed"); \
    }

// Dispatch the compile-time-FD_PAD kernel variants for the runtime
// (fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1) = (accuracy/2, accuracy/2 - 1,
// accuracy/2, accuracy/2 - 1): every direction has stencil radius FD_PAD and
// the compound guards use FD_PAD - 1 for the upper (y1/x1) bounds.  The
// switch runs on fd_pad_y0 (== fd_pad_x0): combinations 1..4 for accuracy
// 2/4/6/8.  Compile-time templating is bitwise-identical to the old runtime
// loops: same coefficients and operation order, only the loop bound becomes a
// compile-time constant.
#define LAUNCH_BORN_GRID(KERN, T, ...)                                         \
    {                                                                          \
        switch ((int)fd_pad_y0) {                                              \
            case 1: LAUNCH_GRID_FD(KERN, T, 1, __VA_ARGS__); break;            \
            case 2: LAUNCH_GRID_FD(KERN, T, 2, __VA_ARGS__); break;            \
            case 3: LAUNCH_GRID_FD(KERN, T, 3, __VA_ARGS__); break;            \
            case 4: LAUNCH_GRID_FD(KERN, T, 4, __VA_ARGS__); break;            \
            default: TORCH_CHECK(false, "nami em2d_tm_born: unsupported fd_pad"); \
        }                                                                      \
    }

#define LAUNCH_PT(KERN, T, NSH, NLOC, ...)                                     \
    {                                                                          \
        dim3 block(32, 4);                                                     \
        dim3 grid(((NSH) + 31) / 32, ((NLOC) + 3) / 4);                        \
        KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami em2d_tm_born " #KERN " failed"); \
    }

// ---------------- EM host wrappers ----------------
void born_step_h(
    torch::Tensor cq, torch::Tensor dcq,
    torch::Tensor ey, torch::Tensor dEy,
    torch::Tensor hx, torch::Tensor hz,
    torch::Tensor dHx, torch::Tensor dHz,
    torch::Tensor m_ey_z, torch::Tensor m_ey_x,
    torch::Tensor dm_ey_z, torch::Tensor dm_ey_x,
    torch::Tensor dey_dy_store, torch::Tensor dey_dx_store,
    torch::Tensor ddey_dy_store, torch::Tensor ddey_dx_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor axh, torch::Tensor bxh,
    torch::Tensor kyh, torch::Tensor kxh,
    torch::Tensor c,
    double rdy, double rdx,
    int64_t t, int64_t interval,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_x0,
    int64_t cq_batched, int64_t dcq_batched, int64_t store, int64_t snap_off)
{
    const int n_shots = ey.size(0), ny = ey.size(1), nx = ey.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(ey.scalar_type(), "em2d_tm_born_born_step_h", [&] {
        LAUNCH_BORN_GRID(born_step_h_kernel, scalar_t,
            cq.data_ptr<scalar_t>(), dcq.data_ptr<scalar_t>(),
            ey.data_ptr<scalar_t>(), dEy.data_ptr<scalar_t>(),
            hx.data_ptr<scalar_t>(), hz.data_ptr<scalar_t>(),
            dHx.data_ptr<scalar_t>(), dHz.data_ptr<scalar_t>(),
            m_ey_z.data_ptr<scalar_t>(), m_ey_x.data_ptr<scalar_t>(),
            dm_ey_z.data_ptr<scalar_t>(), dm_ey_x.data_ptr<scalar_t>(),
            dey_dy_store.data_ptr<scalar_t>(), dey_dx_store.data_ptr<scalar_t>(),
            ddey_dy_store.data_ptr<scalar_t>(), ddey_dx_store.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), byh.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), bxh.data_ptr<scalar_t>(),
            kyh.data_ptr<scalar_t>(), kxh.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx,
            (int)t, (int)interval, n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)cq_batched, (int)dcq_batched, (int)store, (int64_t)snap_off);
    });
}

void born_step_e(
    torch::Tensor ca, torch::Tensor cb, torch::Tensor dca, torch::Tensor dcb,
    torch::Tensor hx, torch::Tensor hz,
    torch::Tensor dHx, torch::Tensor dHz,
    torch::Tensor ey, torch::Tensor dEy,
    torch::Tensor m_hx_z, torch::Tensor m_hz_x,
    torch::Tensor dm_hx_z, torch::Tensor dm_hz_x,
    torch::Tensor ey_store, torch::Tensor curl_store,
    torch::Tensor dEy_store, torch::Tensor dcurl_store,
    torch::Tensor ay, torch::Tensor by, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor ky, torch::Tensor kx,
    torch::Tensor c,
    double rdy, double rdx,
    int64_t t, int64_t interval,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t ca_batched, int64_t cb_batched,
    int64_t dca_batched, int64_t dcb_batched, int64_t store, int64_t snap_off)
{
    const int n_shots = ey.size(0), ny = ey.size(1), nx = ey.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(ey.scalar_type(), "em2d_tm_born_born_step_e", [&] {
        LAUNCH_BORN_GRID(born_step_e_kernel, scalar_t,
            ca.data_ptr<scalar_t>(), cb.data_ptr<scalar_t>(),
            dca.data_ptr<scalar_t>(), dcb.data_ptr<scalar_t>(),
            hx.data_ptr<scalar_t>(), hz.data_ptr<scalar_t>(),
            dHx.data_ptr<scalar_t>(), dHz.data_ptr<scalar_t>(),
            ey.data_ptr<scalar_t>(), dEy.data_ptr<scalar_t>(),
            m_hx_z.data_ptr<scalar_t>(), m_hz_x.data_ptr<scalar_t>(),
            dm_hx_z.data_ptr<scalar_t>(), dm_hz_x.data_ptr<scalar_t>(),
            ey_store.data_ptr<scalar_t>(), curl_store.data_ptr<scalar_t>(),
            dEy_store.data_ptr<scalar_t>(), dcurl_store.data_ptr<scalar_t>(),
            ay.data_ptr<scalar_t>(), by.data_ptr<scalar_t>(),
            ax.data_ptr<scalar_t>(), bx.data_ptr<scalar_t>(),
            ky.data_ptr<scalar_t>(), kx.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx,
            (int)t, (int)interval, n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)ca_batched, (int)cb_batched,
            (int)dca_batched, (int)dcb_batched, (int)store, (int64_t)snap_off);
    });
}

void born_inject(
    torch::Tensor ey, torch::Tensor dEy,
    torch::Tensor f_bg, torch::Tensor f_sc, torch::Tensor src_i,
    int64_t t, int64_t n_shots, int64_t n_src, int64_t ny_nx)
{
    AT_DISPATCH_FLOATING_TYPES(ey.scalar_type(), "em2d_tm_born_born_inject", [&] {
        LAUNCH_PT(born_inject_kernel, scalar_t, (int)n_shots, (int)n_src,
            ey.data_ptr<scalar_t>(), dEy.data_ptr<scalar_t>(),
            f_bg.data_ptr<scalar_t>(), f_sc.data_ptr<scalar_t>(),
            src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)ny_nx);
    });
}

void born_record(
    torch::Tensor dEy, torch::Tensor r, torch::Tensor rec_i,
    int64_t t, int64_t n_shots, int64_t n_rec, int64_t ny_nx)
{
    AT_DISPATCH_FLOATING_TYPES(dEy.scalar_type(), "em2d_tm_born_born_record", [&] {
        LAUNCH_PT(born_record_kernel, scalar_t, (int)n_shots, (int)n_rec,
            dEy.data_ptr<scalar_t>(), r.data_ptr<scalar_t>(),
            rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)ny_nx);
    });
}

void born_record_grad_r(
    torch::Tensor lam_dEy, torch::Tensor grad_r, torch::Tensor rec_i,
    int64_t t, int64_t n_shots, int64_t n_rec, int64_t ny_nx)
{
    AT_DISPATCH_FLOATING_TYPES(lam_dEy.scalar_type(), "em2d_tm_born_born_record_grad_r", [&] {
        LAUNCH_PT(born_record_grad_r_kernel, scalar_t, (int)n_shots, (int)n_rec,
            lam_dEy.data_ptr<scalar_t>(), grad_r.data_ptr<scalar_t>(),
            rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)ny_nx);
    });
}

void born_record_grad_f(
    torch::Tensor lam_ey, torch::Tensor lam_dEy,
    torch::Tensor grad_f_bg, torch::Tensor grad_f_sc, torch::Tensor src_i,
    int64_t t, int64_t n_shots, int64_t n_src, int64_t ny_nx)
{
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "em2d_tm_born_born_record_grad_f", [&] {
        LAUNCH_PT(born_record_grad_f_kernel, scalar_t, (int)n_shots, (int)n_src,
            lam_ey.data_ptr<scalar_t>(), lam_dEy.data_ptr<scalar_t>(),
            grad_f_bg.data_ptr<scalar_t>(), grad_f_sc.data_ptr<scalar_t>(),
            src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)ny_nx);
    });
}

void born_coeff_grad(
    torch::Tensor lam_ey, torch::Tensor lam_dEy,
    torch::Tensor ey_store, torch::Tensor curl_store,
    torch::Tensor dEy_store, torch::Tensor dcurl_store,
    torch::Tensor grad_ca, torch::Tensor grad_cb,
    torch::Tensor grad_dca, torch::Tensor grad_dcb,
    int64_t t, int64_t interval, double scale, int64_t snap_off,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1)
{
    const int n_shots = lam_ey.size(0), ny = lam_ey.size(1), nx = lam_ey.size(2);
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "em2d_tm_born_born_coeff_grad", [&] {
        LAUNCH_BORN_GRID(born_coeff_grad_kernel, scalar_t,
            lam_ey.data_ptr<scalar_t>(), lam_dEy.data_ptr<scalar_t>(),
            ey_store.data_ptr<scalar_t>(), curl_store.data_ptr<scalar_t>(),
            dEy_store.data_ptr<scalar_t>(), dcurl_store.data_ptr<scalar_t>(),
            grad_ca.data_ptr<scalar_t>(), grad_cb.data_ptr<scalar_t>(),
            grad_dca.data_ptr<scalar_t>(), grad_dcb.data_ptr<scalar_t>(),
            (int)t, (int)interval, (scalar_t)scale, (int64_t)snap_off, n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1);
    });
}

void born_adjoint_e_stage1(
    torch::Tensor ca, torch::Tensor cb, torch::Tensor dca, torch::Tensor dcb,
    torch::Tensor lam_ey, torch::Tensor lam_dEy,
    torch::Tensor m_lambda_hx_z, torch::Tensor m_lambda_hz_x,
    torch::Tensor dm_lambda_hx_z, torch::Tensor dm_lambda_hz_x,
    torch::Tensor work_x, torch::Tensor work_y,
    torch::Tensor work_dx, torch::Tensor work_dy,
    torch::Tensor ay, torch::Tensor by, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor ky, torch::Tensor kx,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t ca_batched, int64_t cb_batched,
    int64_t dca_batched, int64_t dcb_batched)
{
    const int n_shots = lam_ey.size(0), ny = lam_ey.size(1), nx = lam_ey.size(2);
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "em2d_tm_born_born_adjoint_e_stage1", [&] {
        LAUNCH_BORN_GRID(born_adjoint_e_stage1_kernel, scalar_t,
            ca.data_ptr<scalar_t>(), cb.data_ptr<scalar_t>(),
            dca.data_ptr<scalar_t>(), dcb.data_ptr<scalar_t>(),
            lam_ey.data_ptr<scalar_t>(), lam_dEy.data_ptr<scalar_t>(),
            m_lambda_hx_z.data_ptr<scalar_t>(), m_lambda_hz_x.data_ptr<scalar_t>(),
            dm_lambda_hx_z.data_ptr<scalar_t>(), dm_lambda_hz_x.data_ptr<scalar_t>(),
            work_x.data_ptr<scalar_t>(), work_y.data_ptr<scalar_t>(),
            work_dx.data_ptr<scalar_t>(), work_dy.data_ptr<scalar_t>(),
            ay.data_ptr<scalar_t>(), by.data_ptr<scalar_t>(),
            ax.data_ptr<scalar_t>(), bx.data_ptr<scalar_t>(),
            ky.data_ptr<scalar_t>(), kx.data_ptr<scalar_t>(),
            n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)ca_batched, (int)cb_batched,
            (int)dca_batched, (int)dcb_batched);
    });
}

void born_adjoint_e_stage2(
    torch::Tensor work_x, torch::Tensor work_y,
    torch::Tensor work_dx, torch::Tensor work_dy,
    torch::Tensor lam_hx, torch::Tensor lam_hz,
    torch::Tensor lam_dhx, torch::Tensor lam_dhz,
    torch::Tensor c, double rdy, double rdx,
    int64_t fd_pad_y0, int64_t fd_pad_x0)
{
    const int n_shots = lam_hx.size(0), ny = lam_hx.size(1), nx = lam_hx.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(lam_hx.scalar_type(), "em2d_tm_born_born_adjoint_e_stage2", [&] {
        LAUNCH_BORN_GRID(born_adjoint_e_stage2_kernel, scalar_t,
            work_x.data_ptr<scalar_t>(), work_y.data_ptr<scalar_t>(),
            work_dx.data_ptr<scalar_t>(), work_dy.data_ptr<scalar_t>(),
            lam_hx.data_ptr<scalar_t>(), lam_hz.data_ptr<scalar_t>(),
            lam_dhx.data_ptr<scalar_t>(), lam_dhz.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx,
            n_shots, ny, nx);
    });
}

void born_adjoint_h_stage1(
    torch::Tensor cq, torch::Tensor dcq,
    torch::Tensor lam_hx, torch::Tensor lam_hz,
    torch::Tensor lam_dhx, torch::Tensor lam_dhz,
    torch::Tensor m_lambda_ey_x, torch::Tensor m_lambda_ey_z,
    torch::Tensor dm_lambda_ey_x, torch::Tensor dm_lambda_ey_z,
    torch::Tensor work2_x, torch::Tensor work2_y,
    torch::Tensor work2_dx, torch::Tensor work2_dy,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor axh, torch::Tensor bxh,
    torch::Tensor kyh, torch::Tensor kxh,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t cq_batched, int64_t dcq_batched)
{
    const int n_shots = lam_hx.size(0), ny = lam_hx.size(1), nx = lam_hx.size(2);
    AT_DISPATCH_FLOATING_TYPES(lam_hx.scalar_type(), "em2d_tm_born_born_adjoint_h_stage1", [&] {
        LAUNCH_BORN_GRID(born_adjoint_h_stage1_kernel, scalar_t,
            cq.data_ptr<scalar_t>(), dcq.data_ptr<scalar_t>(),
            lam_hx.data_ptr<scalar_t>(), lam_hz.data_ptr<scalar_t>(),
            lam_dhx.data_ptr<scalar_t>(), lam_dhz.data_ptr<scalar_t>(),
            m_lambda_ey_x.data_ptr<scalar_t>(), m_lambda_ey_z.data_ptr<scalar_t>(),
            dm_lambda_ey_x.data_ptr<scalar_t>(), dm_lambda_ey_z.data_ptr<scalar_t>(),
            work2_x.data_ptr<scalar_t>(), work2_y.data_ptr<scalar_t>(),
            work2_dx.data_ptr<scalar_t>(), work2_dy.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), byh.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), bxh.data_ptr<scalar_t>(),
            kyh.data_ptr<scalar_t>(), kxh.data_ptr<scalar_t>(),
            n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)cq_batched, (int)dcq_batched);
    });
}

void born_cq_grad(
    torch::Tensor lam_hx, torch::Tensor lam_hz,
    torch::Tensor lam_dhx, torch::Tensor lam_dhz,
    torch::Tensor dey_dy_store, torch::Tensor dey_dx_store,
    torch::Tensor ddey_dy_store, torch::Tensor ddey_dx_store,
    torch::Tensor grad_cq, torch::Tensor grad_dcq,
    int64_t t, int64_t interval, double scale, int64_t snap_off,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1)
{
    const int n_shots = lam_hx.size(0), ny = lam_hx.size(1), nx = lam_hx.size(2);
    AT_DISPATCH_FLOATING_TYPES(lam_hx.scalar_type(), "em2d_tm_born_born_cq_grad", [&] {
        LAUNCH_BORN_GRID(born_cq_grad_kernel, scalar_t,
            lam_hx.data_ptr<scalar_t>(), lam_hz.data_ptr<scalar_t>(),
            lam_dhx.data_ptr<scalar_t>(), lam_dhz.data_ptr<scalar_t>(),
            dey_dy_store.data_ptr<scalar_t>(), dey_dx_store.data_ptr<scalar_t>(),
            ddey_dy_store.data_ptr<scalar_t>(), ddey_dx_store.data_ptr<scalar_t>(),
            grad_cq.data_ptr<scalar_t>(), grad_dcq.data_ptr<scalar_t>(),
            (int)t, (int)interval, (scalar_t)scale, (int64_t)snap_off, n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1);
    });
}

void born_adjoint_h_stage2(
    torch::Tensor work2_x, torch::Tensor work2_y,
    torch::Tensor work2_dx, torch::Tensor work2_dy,
    torch::Tensor lam_ey, torch::Tensor lam_dEy,
    torch::Tensor c, double rdy, double rdx,
    int64_t fd_pad_y0, int64_t fd_pad_x0)
{
    const int n_shots = lam_ey.size(0), ny = lam_ey.size(1), nx = lam_ey.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "em2d_tm_born_born_adjoint_h_stage2", [&] {
        LAUNCH_BORN_GRID(born_adjoint_h_stage2_kernel, scalar_t,
            work2_x.data_ptr<scalar_t>(), work2_y.data_ptr<scalar_t>(),
            work2_dx.data_ptr<scalar_t>(), work2_dy.data_ptr<scalar_t>(),
            lam_ey.data_ptr<scalar_t>(), lam_dEy.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx,
            n_shots, ny, nx);
    });
}

// ---------------- whole-loop drivers ----------------
// One pybind call runs the entire forward/adjoint pass (deepwave-style):
// the per-step functions above are reused unchanged with the same argument
// order the Python loops used, so results are bitwise identical.

using nami_storage::ckpt_restore;
using nami_storage::ckpt_save;
using nami_storage::zero_buffers;

// EM (2D TM) Born forward.  state: 14 flat buffers, N_STATE layout
//   [0]ey [1]d_ey [2]hx [3]hz [4]d_hx [5]d_hz
//   [6]m_ey_z [7]m_ey_x [8]dm_ey_z [9]dm_ey_x
//   [10]m_hx_z [11]m_hz_x [12]dm_hx_z [13]dm_hz_x
void born_em_forward_loop(
    torch::Tensor ca, torch::Tensor cb, torch::Tensor cq,
    torch::Tensor dca, torch::Tensor dcb, torch::Tensor dcq,
    std::vector<torch::Tensor> state,
    torch::Tensor ey_store, torch::Tensor curl_store,
    torch::Tensor dEy_store, torch::Tensor dcurl_store,
    torch::Tensor dey_dy_store, torch::Tensor dey_dx_store,
    torch::Tensor ddey_dy_store, torch::Tensor ddey_dx_store,
    torch::Tensor ay, torch::Tensor ayh,
    torch::Tensor ax, torch::Tensor axh,
    torch::Tensor by, torch::Tensor byh,
    torch::Tensor bx, torch::Tensor bxh,
    torch::Tensor ky, torch::Tensor kyh,
    torch::Tensor kx, torch::Tensor kxh,
    torch::Tensor c,
    torch::Tensor f_bg, torch::Tensor f_sc,
    torch::Tensor src_i, torch::Tensor r, torch::Tensor rec_i,
    torch::Tensor r_bg, torch::Tensor bg_rec_i,
    double rdy, double rdx,
    int64_t nt, int64_t interval,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t ca_batched, int64_t cb_batched, int64_t cq_batched,
    int64_t dca_batched, int64_t dcb_batched, int64_t dcq_batched,
    int64_t store, int64_t checkpoint_every,
    c10::optional<torch::Tensor> ckpt_state,
    // Wavefield I/O (deepwave-style): init_state/final_state use the N_STATE
    // layout above (same as ckpt).  The Born buffers are flat (updated in
    // place), so restore/copy each slot verbatim — a split run (continuation
    // via init_state) then bitwise matches a one-shot run.
    c10::optional<torch::Tensor> init_state,
    c10::optional<torch::Tensor> final_state,
    // Per-step forward callback: called every `callback_frequency` steps as
    // callback(t, nt, ey, hx, hz, d_ey, d_hx, d_hz) — live padded physics
    // fields; PML memory variables remain internal.
    py::object callback, int64_t callback_frequency)
{
    TORCH_CHECK(state.size() == 14,
                "nami em2d_tm_born forward_loop: bad state size");
    const int64_t n_shots = state[0].size(0);
    const int64_t ny = state[0].size(1), nx = state[0].size(2);
    const int64_t n_src = src_i.size(1), n_rec = rec_i.size(1);
    const int64_t n_bg_rec = bg_rec_i.size(1);
    const int64_t ny_nx = ny * nx;
    const int64_t shot_count = n_shots * ny_nx;
    const bool ckpt = ckpt_state.has_value() && checkpoint_every > 0;
    const bool has_cb = !callback.is_none();
    std::vector<const torch::Tensor*> state_ptrs;
    for (const auto& s : state)
        state_ptrs.push_back(&s);

    // Optional initial state (continuation runs): restore every buffer.
    if (init_state.has_value()) {
        auto c = *init_state;
        for (size_t i = 0; i < state.size(); ++i)
            state[i].copy_(c[i]);
    }

    for (int64_t t = 0; t < nt; ++t) {
        if (ckpt && t > 0 && t % checkpoint_every == 0)
            ckpt_save(*ckpt_state, t / checkpoint_every - 1, state_ptrs);
        if (has_cb && t % callback_frequency == 0) {
            py::gil_scoped_acquire gil;
            callback(t, nt, state[0], state[2], state[3],
                     state[1], state[4], state[5]);
        }
        const int64_t snap_off = store ? (t / interval) * shot_count : 0;
        born_step_h(
            cq, dcq, state[0], state[1], state[2], state[3], state[4], state[5],
            state[6], state[7], state[8], state[9],
            dey_dy_store, dey_dx_store, ddey_dy_store, ddey_dx_store,
            ayh, byh, axh, bxh, kyh, kxh, c,
            rdy, rdx, t, interval,
            pml_y0, pml_y1, pml_x0, pml_x1,
            fd_pad_y0, fd_pad_x0,
            cq_batched, dcq_batched, store, snap_off);
        born_step_e(
            ca, cb, dca, dcb,
            state[2], state[3], state[4], state[5], state[0], state[1],
            state[10], state[11], state[12], state[13],
            ey_store, curl_store, dEy_store, dcurl_store,
            ay, by, ax, bx, ky, kx, c,
            rdy, rdx, t, interval,
            pml_y0, pml_y1, pml_x0, pml_x1,
            fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
            ca_batched, cb_batched, dca_batched, dcb_batched, store, snap_off);
        if (n_src > 0)
            born_inject(state[0], state[1], f_bg, f_sc, src_i,
                        t, n_shots, n_src, ny_nx);
        if (n_rec > 0)
            born_record(state[1], r, rec_i, t, n_shots, n_rec, ny_nx);
        if (n_bg_rec > 0)
            born_record(state[0], r_bg, bg_rec_i,
                        t, n_shots, n_bg_rec, ny_nx);
    }

    // Optional final state: the buffers already hold the state at time nt
    // (fields are updated in place), so a continuation restores them as-is.
    if (final_state.has_value()) {
        auto c = *final_state;
        for (size_t i = 0; i < state.size(); ++i)
            c[i].copy_(state[i]);
    }
}

// EM (2D TM) Born adjoint.  state_f: 14 replay buffers (same layout as the
// forward state; empty when segments is empty = full storage).
void born_em_adjoint_loop(
    torch::Tensor ca, torch::Tensor cb, torch::Tensor cq,
    torch::Tensor dca, torch::Tensor dcb, torch::Tensor dcq,
    torch::Tensor ey_store, torch::Tensor curl_store,
    torch::Tensor dEy_store, torch::Tensor dcurl_store,
    torch::Tensor dey_dy_store, torch::Tensor dey_dx_store,
    torch::Tensor ddey_dy_store, torch::Tensor ddey_dx_store,
    torch::Tensor grad_ca, torch::Tensor grad_cb, torch::Tensor grad_cq,
    torch::Tensor grad_dca, torch::Tensor grad_dcb, torch::Tensor grad_dcq,
    torch::Tensor grad_f_bg, torch::Tensor grad_f_sc,
    torch::Tensor grad_r, torch::Tensor grad_r_bg,
    torch::Tensor src_i, torch::Tensor rec_i, torch::Tensor bg_rec_i,
    torch::Tensor f_bg, torch::Tensor f_sc,
    torch::Tensor lam_ey, torch::Tensor lam_dEy,
    torch::Tensor lam_hx, torch::Tensor lam_hz,
    torch::Tensor lam_dhx, torch::Tensor lam_dhz,
    torch::Tensor m_lambda_hx_z, torch::Tensor m_lambda_hz_x,
    torch::Tensor dm_lambda_hx_z, torch::Tensor dm_lambda_hz_x,
    torch::Tensor m_lambda_ey_x, torch::Tensor m_lambda_ey_z,
    torch::Tensor dm_lambda_ey_x, torch::Tensor dm_lambda_ey_z,
    torch::Tensor work_x, torch::Tensor work_y,
    torch::Tensor work_dx, torch::Tensor work_dy,
    torch::Tensor work2_x, torch::Tensor work2_y,
    torch::Tensor work2_dx, torch::Tensor work2_dy,
    std::vector<torch::Tensor> state_f,
    torch::Tensor ay, torch::Tensor ayh,
    torch::Tensor ax, torch::Tensor axh,
    torch::Tensor by, torch::Tensor byh,
    torch::Tensor bx, torch::Tensor bxh,
    torch::Tensor ky, torch::Tensor kyh,
    torch::Tensor kx, torch::Tensor kxh,
    torch::Tensor c,
    double rdy, double rdx, double scale,
    int64_t nt, int64_t interval,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t ca_batched, int64_t cb_batched, int64_t cq_batched,
    int64_t dca_batched, int64_t dcb_batched, int64_t dcq_batched,
    torch::Tensor segments,  // int64 [n_seg, 2] on CPU; empty = full storage
    c10::optional<torch::Tensor> ckpt_state)
{
    const int64_t n_shots = lam_ey.size(0);
    const int64_t ny = lam_ey.size(1), nx = lam_ey.size(2);
    const int64_t n_src = src_i.size(1), n_rec = rec_i.size(1);
    const int64_t n_bg_rec = bg_rec_i.size(1);
    const int64_t ny_nx = ny * nx;
    const int64_t shot_count = n_shots * ny_nx;

    auto adjoint_at = [&](int64_t t, int64_t snap_off) {
        if (n_rec > 0)
            born_record_grad_r(lam_dEy, grad_r, rec_i, t, n_shots, n_rec, ny_nx);
        if (n_bg_rec > 0)
            born_record_grad_r(
                lam_ey, grad_r_bg, bg_rec_i,
                t, n_shots, n_bg_rec, ny_nx);
        if (n_src > 0)
            born_record_grad_f(lam_ey, lam_dEy, grad_f_bg, grad_f_sc,
                               src_i, t, n_shots, n_src, ny_nx);
        if (t % interval == 0)
            born_coeff_grad(
                lam_ey, lam_dEy, ey_store, curl_store, dEy_store, dcurl_store,
                grad_ca, grad_cb, grad_dca, grad_dcb,
                t, interval, scale, snap_off,
                pml_y0, pml_y1, pml_x0, pml_x1,
                fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1);
        born_adjoint_e_stage1(
            ca, cb, dca, dcb, lam_ey, lam_dEy,
            m_lambda_hx_z, m_lambda_hz_x, dm_lambda_hx_z, dm_lambda_hz_x,
            work_x, work_y, work_dx, work_dy,
            ay, by, ax, bx, ky, kx,
            pml_y0, pml_y1, pml_x0, pml_x1,
            fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
            ca_batched, cb_batched, dca_batched, dcb_batched);
        born_adjoint_e_stage2(
            work_x, work_y, work_dx, work_dy,
            lam_hx, lam_hz, lam_dhx, lam_dhz,
            c, rdy, rdx, fd_pad_y0, fd_pad_x0);
        born_adjoint_h_stage1(
            cq, dcq, lam_hx, lam_hz, lam_dhx, lam_dhz,
            m_lambda_ey_x, m_lambda_ey_z, dm_lambda_ey_x, dm_lambda_ey_z,
            work2_x, work2_y, work2_dx, work2_dy,
            ayh, byh, axh, bxh, kyh, kxh,
            pml_y0, pml_y1, pml_x0, pml_x1,
            fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
            cq_batched, dcq_batched);
        if (t % interval == 0)
            born_cq_grad(
                lam_hx, lam_hz, lam_dhx, lam_dhz,
                dey_dy_store, dey_dx_store, ddey_dy_store, ddey_dx_store,
                grad_cq, grad_dcq,
                t, interval, scale, snap_off,
                pml_y0, pml_y1, pml_x0, pml_x1,
                fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1);
        born_adjoint_h_stage2(
            work2_x, work2_y, work2_dx, work2_dy,
            lam_ey, lam_dEy,
            c, rdy, rdx, fd_pad_y0, fd_pad_x0);
    };

    if (segments.numel() == 0) {
        for (int64_t t = nt - 1; t >= 0; --t)
            adjoint_at(t, (t / interval) * shot_count);
        return;
    }

    // Checkpointed backward: per segment, restore the forward state, replay
    // the forward steps to regenerate the snapshots, then run the adjoint.
    TORCH_CHECK(state_f.size() == 14,
                "nami em2d_tm_born adjoint_loop: bad replay state size");
    auto seg = segments.accessor<int64_t, 2>();
    const int64_t n_seg = segments.size(0);
    std::vector<torch::Tensor*> state_f_ptrs;
    for (auto& s : state_f)
        state_f_ptrs.push_back(&s);
    for (int64_t k = n_seg - 1; k >= 0; --k) {
        const int64_t s0 = seg[k][0], s1 = seg[k][1];
        if (s0 > 0) {
            ckpt_restore(*ckpt_state, k - 1, state_f_ptrs);
        } else {
            zero_buffers(state_f_ptrs);
        }
        for (int64_t t = s0; t < s1; ++t) {
            const int64_t snap_off = ((t - s0) / interval) * shot_count;
            born_step_h(
                cq, dcq, state_f[0], state_f[1], state_f[2], state_f[3],
                state_f[4], state_f[5],
                state_f[6], state_f[7], state_f[8], state_f[9],
                dey_dy_store, dey_dx_store, ddey_dy_store, ddey_dx_store,
                ayh, byh, axh, bxh, kyh, kxh, c,
                rdy, rdx, t, interval,
                pml_y0, pml_y1, pml_x0, pml_x1,
                fd_pad_y0, fd_pad_x0,
                cq_batched, dcq_batched, 1, snap_off);
            born_step_e(
                ca, cb, dca, dcb,
                state_f[2], state_f[3], state_f[4], state_f[5],
                state_f[0], state_f[1],
                state_f[10], state_f[11], state_f[12], state_f[13],
                ey_store, curl_store, dEy_store, dcurl_store,
                ay, by, ax, bx, ky, kx, c,
                rdy, rdx, t, interval,
                pml_y0, pml_y1, pml_x0, pml_x1,
                fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
                ca_batched, cb_batched, dca_batched, dcb_batched, 1, snap_off);
            if (n_src > 0)
                born_inject(state_f[0], state_f[1], f_bg, f_sc, src_i,
                            t, n_shots, n_src, ny_nx);
        }
        for (int64_t t = s1 - 1; t >= s0; --t)
            adjoint_at(t, ((t - s0) / interval) * shot_count);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("born_step_h", &born_step_h);
    m.def("born_step_e", &born_step_e);
    m.def("born_inject", &born_inject);
    m.def("born_record", &born_record);
    m.def("born_record_grad_r", &born_record_grad_r);
    m.def("born_record_grad_f", &born_record_grad_f);
    m.def("born_coeff_grad", &born_coeff_grad);
    m.def("born_adjoint_e_stage1", &born_adjoint_e_stage1);
    m.def("born_adjoint_e_stage2", &born_adjoint_e_stage2);
    m.def("born_adjoint_h_stage1", &born_adjoint_h_stage1);
    m.def("born_cq_grad", &born_cq_grad);
    m.def("born_adjoint_h_stage2", &born_adjoint_h_stage2);
    NAMI_STORAGE_PYBIND(m);
    m.def("born_em_forward_loop", &born_em_forward_loop);
    m.def("born_em_adjoint_loop", &born_em_adjoint_loop);
}
