// nami EM/elastic Born: first-order Born scattering on the em2d_tm and
// elastic2d staggered grids, with exact discrete-transpose adjoints.
//
// Forward: a background field is propagated with the unperturbed operators
// and a scattered field is driven by scattering sources linear in the
// parameter perturbations and the background wavefield (TM2D Born
// convention for EM; self-derived for elastic).  The backward pass is the
// exact transpose of the coupled (background + scattered) system, so
// gradients flow to both the background and the scatter models.
//
// EM Born: per step the background H update, the scattered H
// update driven by dEy (plus dcq*deriv(Ey) when mu is perturbed), the
// combined E update
//     Ey  = ca*Ey  + cb*curl
//     dEy = ca*dEy + cb*dcurl + dca*Ey_old + dcb*curl
// (snapshotting the pre-update fields and both curls), background and
// scattered source injection (f_bg = cb*-1/(dx dy)*amp, f_sc = dcb*...),
// and receivers recording the scattered post-injection dEy.
//
// Elastic Born (self-derived): linearizing the velocity-stress system
//   vy += by*dt*(DIFFYH1(syy)+DIFFX1(sxy))
//   vx += bx*dt*(DIFFXH1(sxx)+DIFFY1(sxy))
//   syy += dt*(lamb*(dvydy+dvxdx) + 2*mu*dvydy)
//   sxx += dt*(lamb*(dvydy+dvxdx) + 2*mu*dvxdx)
//   sxy += dt*mu_yx*(DIFFXH1(vy)+DIFFYH1(vx))
// around the background gives the scattering sources
//   dvy += dby*dt*w_y(bg)          dvx += dbx*dt*w_x(bg)
//   dsyy += dt*(dlamb*ssum + 2*dmu*dvydy)
//   dsxx += dt*(dlamb*ssum + 2*dmu*dvxdx)
//   dsxy += dt*dmu_yx*w_sum(bg)
// with the scattered field propagating through the background operator
// (same CPML memories, same profiles).  dby/dbx/dmu_yx are the first-order
// linearizations of prepare_parameters (computed on the Python side).
//
// Kernels are intentionally unoptimised (one launch per step, naive
// stencil) and unified (the full CPML logic runs everywhere; a=b=0 in the
// interior reproduces the existing interior/frame split bit-for-bit).

#include <torch/extension.h>
#include "storage.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------- coefficient-driven staggered difference operators ----------------
// Same conventions as em2d_tm.cu / elastic2d.cu: ``c`` is the staggered-grid
// first-derivative coefficient array (max radius 4, zero-padded for lower
// orders).  diff_int_*: derivative at a half-integer grid point of a field
// stored at integer points; diff_half_*: derivative at an integer grid point
// of a field stored at half-integer points.  The loop bound is the runtime
// stencil radius, never a hardcoded maximum.
template <typename T>
__device__ __forceinline__ T diff_int_x(const T* __restrict__ u, long off,
                                        const T* __restrict__ c, T rdx, int radius)
{
    T d = (T)0;
    for (int k = 1; k <= radius; ++k)
        d += c[k - 1] * (u[off + k] - u[off - k + 1]);
    return d * rdx;
}

template <typename T>
__device__ __forceinline__ T diff_int_y(const T* __restrict__ u, long off,
                                        const T* __restrict__ c, T rdy, int nx, int radius)
{
    T d = (T)0;
    for (int k = 1; k <= radius; ++k)
        d += c[k - 1] * (u[off + k * nx] - u[off - (k - 1) * nx]);
    return d * rdy;
}

template <typename T>
__device__ __forceinline__ T diff_half_x(const T* __restrict__ u, long off,
                                         const T* __restrict__ c, T rdx, int radius)
{
    T d = (T)0;
    for (int k = 1; k <= radius; ++k)
        d += c[k - 1] * (u[off + k - 1] - u[off - k]);
    return d * rdx;
}

template <typename T>
__device__ __forceinline__ T diff_half_y(const T* __restrict__ u, long off,
                                         const T* __restrict__ c, T rdy, int nx, int radius)
{
    T d = (T)0;
    for (int k = 1; k <= radius; ++k)
        d += c[k - 1] * (u[off + (k - 1) * nx] - u[off - k * nx]);
    return d * rdy;
}

// ==================== EM (2D TM) Born ====================

// ---------------- forward: H half-step (background + scattered) ----------------
// hx -= cq*dey_dy  and  dHx -= cq*ddey_dy + dcq*dey_dy  (and the x-analogue
// for hz/dHz).  The PML-modified derivatives are snapshotted on the sampling
// interval for the cq/dcq gradients.
template <typename T>
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
    int fd_pad_y0, int fd_pad_x0,
    int cq_batched, int dcq_batched, int store)
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
        if (y >= fd_pad_y0 && y < ny - fd_pad_y0)
            dey_dy = diff_int_y(ey, off, c, rdy, nx, fd_pad_y0);
        T ddey_dy = (T)0;
        if (y >= fd_pad_y0 && y < ny - fd_pad_y0)
            ddey_dy = diff_int_y(dEy, off, c, rdy, nx, fd_pad_y0);
        if (y < pml_y0 || y >= pml_y1 - 1) {
            m_ey_z[off] = byh[y] * m_ey_z[off] + ayh[y] * dey_dy;
            dm_ey_z[off] = byh[y] * dm_ey_z[off] + ayh[y] * ddey_dy;
            dey_dy = dey_dy / kyh[y] + m_ey_z[off];
            ddey_dy = ddey_dy / kyh[y] + dm_ey_z[off];
        }
        if (store && t % interval == 0) {
            const long soff = (((long)(t / interval) * n_shots + s) * ny + y) * nx + x;
            dey_dy_store[soff] = dey_dy;
            ddey_dy_store[soff] = ddey_dy;
        }
        hx[off] -= cq_val * dey_dy;
        dHx[off] -= cq_val * ddey_dy + dcq_val * dey_dy;
        T dey_dx = (T)0;
        if (x >= fd_pad_x0 && x < nx - fd_pad_x0)
            dey_dx = diff_int_x(ey, off, c, rdx, fd_pad_x0);
        T ddey_dx = (T)0;
        if (x >= fd_pad_x0 && x < nx - fd_pad_x0)
            ddey_dx = diff_int_x(dEy, off, c, rdx, fd_pad_x0);
        if (x < pml_x0 || x >= pml_x1 - 1) {
            m_ey_x[off] = bxh[x] * m_ey_x[off] + axh[x] * dey_dx;
            dm_ey_x[off] = bxh[x] * dm_ey_x[off] + axh[x] * ddey_dx;
            dey_dx = dey_dx / kxh[x] + m_ey_x[off];
            ddey_dx = ddey_dx / kxh[x] + dm_ey_x[off];
        }
        if (store && t % interval == 0) {
            const long soff = (((long)(t / interval) * n_shots + s) * ny + y) * nx + x;
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
template <typename T>
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
    int fd_pad_y0, int fd_pad_y1, int fd_pad_x0, int fd_pad_x1,
    int ca_batched, int cb_batched, int dca_batched, int dcb_batched)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= fd_pad_y0 && x >= fd_pad_x0 && y < ny - fd_pad_y1 &&
        x < nx - fd_pad_x1) {
        const int s_ca = ca_batched ? s : 0;
        const int s_cb = cb_batched ? s : 0;
        const int s_dca = dca_batched ? s : 0;
        const int s_dcb = dcb_batched ? s : 0;
        const long off = ((long)s * ny + y) * nx + x;
        const long soff = (((long)(t / interval) * n_shots + s) * ny + y) * nx + x;
        T dhz_dx = diff_half_x(hz, off, c, rdx, fd_pad_x0);
        if (x < pml_x0 || x >= pml_x1) {
            m_hz_x[off] = bx[x] * m_hz_x[off] + ax[x] * dhz_dx;
            dhz_dx = dhz_dx / kx[x] + m_hz_x[off];
        }
        T dhx_dy = diff_half_y(hx, off, c, rdy, nx, fd_pad_y0);
        if (y < pml_y0 || y >= pml_y1) {
            m_hx_z[off] = by[y] * m_hx_z[off] + ay[y] * dhx_dy;
            dhx_dy = dhx_dy / ky[y] + m_hx_z[off];
        }
        const T curl = dhz_dx - dhx_dy;
        T ddHz_dx = diff_half_x(dHz, off, c, rdx, fd_pad_x0);
        if (x < pml_x0 || x >= pml_x1) {
            dm_hz_x[off] = bx[x] * dm_hz_x[off] + ax[x] * ddHz_dx;
            ddHz_dx = ddHz_dx / kx[x] + dm_hz_x[off];
        }
        T ddHx_dy = diff_half_y(dHx, off, c, rdy, nx, fd_pad_y0);
        if (y < pml_y0 || y >= pml_y1) {
            dm_hx_z[off] = by[y] * dm_hx_z[off] + ay[y] * ddHx_dy;
            ddHx_dy = ddHx_dy / ky[y] + dm_hx_z[off];
        }
        const T dcurl = ddHz_dx - ddHx_dy;
        const T ey_old = ey[off];
        const T dEy_old = dEy[off];
        if (t % interval == 0) {
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
template <typename T>
__global__ void born_coeff_grad_kernel(
    const T* __restrict__ lam_ey, const T* __restrict__ lam_dEy,
    const T* __restrict__ ey_store, const T* __restrict__ curl_store,
    const T* __restrict__ dEy_store, const T* __restrict__ dcurl_store,
    T* __restrict__ grad_ca, T* __restrict__ grad_cb,
    T* __restrict__ grad_dca, T* __restrict__ grad_dcb,
    int t, int interval, T scale,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int fd_pad_y0, int fd_pad_y1, int fd_pad_x0, int fd_pad_x1)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= fd_pad_y0 && x >= fd_pad_x0 && y < ny - fd_pad_y1 &&
        x < nx - fd_pad_x1) {
        const long off = ((long)s * ny + y) * nx + x;
        const long soff = (((long)(t / interval) * n_shots + s) * ny + y) * nx + x;
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
template <typename T>
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
    int fd_pad_y0, int fd_pad_y1, int fd_pad_x0, int fd_pad_x1,
    int ca_batched, int cb_batched, int dca_batched, int dcb_batched)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < ny && x < nx) {
        const long off = ((long)s * ny + y) * nx + x;
        if (y >= fd_pad_y0 && x >= fd_pad_x0 && y < ny - fd_pad_y1 &&
            x < nx - fd_pad_x1) {
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
template <typename T>
__global__ void born_adjoint_e_stage2_kernel(
    const T* __restrict__ work_x, const T* __restrict__ work_y,
    const T* __restrict__ work_dx, const T* __restrict__ work_dy,
    T* __restrict__ lam_hx, T* __restrict__ lam_hz,
    T* __restrict__ lam_dhx, T* __restrict__ lam_dhz,
    const T* __restrict__ c,
    T rdy, T rdx,
    int n_shots, int ny, int nx,
    int fd_pad_y0, int fd_pad_x0)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < ny && x < nx) {
        const long off = ((long)s * ny + y) * nx + x;
        T acc_x = (T)0;
        T acc_dx = (T)0;
        for (int k = 1; k <= fd_pad_x0; ++k) {
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
        for (int k = 1; k <= fd_pad_y0; ++k) {
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
template <typename T>
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
    int fd_pad_y0, int fd_pad_y1, int fd_pad_x0, int fd_pad_x1,
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
        if (y >= fd_pad_y0 && x >= fd_pad_x0 && y < ny - fd_pad_y1 &&
            x < nx - fd_pad_x1) {
            const int s_cq = cq_batched ? s : 0;
            const int s_dcq = dcq_batched ? s : 0;
            const T cq_val = cq[(long)s_cq * ny * nx + (long)y * nx + x];
            const T dcq_val = dcq[(long)s_dcq * ny * nx + (long)y * nx + x];
            if (x < nx - fd_pad_x0) {
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
            if (y < ny - fd_pad_y0) {
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
template <typename T>
__global__ void born_cq_grad_kernel(
    const T* __restrict__ lam_hx, const T* __restrict__ lam_hz,
    const T* __restrict__ lam_dhx, const T* __restrict__ lam_dhz,
    const T* __restrict__ dey_dy_store, const T* __restrict__ dey_dx_store,
    const T* __restrict__ ddey_dy_store, const T* __restrict__ ddey_dx_store,
    T* __restrict__ grad_cq, T* __restrict__ grad_dcq,
    int t, int interval, T scale,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int fd_pad_y0, int fd_pad_y1, int fd_pad_x0, int fd_pad_x1)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= fd_pad_y0 && x >= fd_pad_x0 && y < ny - fd_pad_y1 &&
        x < nx - fd_pad_x1) {
        const long off = ((long)s * ny + y) * nx + x;
        const long soff = (((long)(t / interval) * n_shots + s) * ny + y) * nx + x;
        T term_cq = (T)0;
        T term_dcq = (T)0;
        if (y < ny - fd_pad_y0) {
            const T dy_bg = dey_dy_store[soff];
            const T dy_sc = ddey_dy_store[soff];
            term_cq += -lam_hx[off] * dy_bg - lam_dhx[off] * dy_sc;
            term_dcq += -lam_dhx[off] * dy_bg;
        }
        if (x < nx - fd_pad_x0) {
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
template <typename T>
__global__ void born_adjoint_h_stage2_kernel(
    const T* __restrict__ work2_x, const T* __restrict__ work2_y,
    const T* __restrict__ work2_dx, const T* __restrict__ work2_dy,
    T* __restrict__ lam_ey, T* __restrict__ lam_dEy,
    const T* __restrict__ c,
    T rdy, T rdx,
    int n_shots, int ny, int nx,
    int fd_pad_y0, int fd_pad_x0)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y < ny && x < nx) {
        const long off = ((long)s * ny + y) * nx + x;
        T sum_x = (T)0;
        T sum_dx = (T)0;
        for (int k = 1; k <= fd_pad_x0; ++k) {
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
        for (int k = 1; k <= fd_pad_y0; ++k) {
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

// ==================== Elastic (2D velocity-stress) Born ====================

// ---------------- forward: velocity update (background + scattered) ----------------
// Region A (vy): y in [fd_pad_y0, ny-fd_pad_y0), x in [fd_pad_x0, nx-fd_pad_x1)
//   vy += by*dtv*w_y      w_y  = DIFFYH1(syy)+DIFFX1(sxy)  (PML memories)
//   dvy += by*dtv*dw_y + dby*dtv*w_y
// Region B (vx): y in [fd_pad_y0, ny-fd_pad_y1), x in [fd_pad_x0, nx-fd_pad_x0)
//   vx += bx*dtv*w_x      w_x  = DIFFXH1(sxx)+DIFFY1(sxy)
//   dvx += bx*dtv*dw_x + dbx*dtv*w_x
// dtv*w_y, dtv*w_x and the scattered analogues are snapshotted for the
// buoyancy imaging conditions.
template <typename T>
__global__ void born_step_velocity_kernel(
    T* __restrict__ vy, T* __restrict__ vx,
    const T* __restrict__ syy, const T* __restrict__ sxx, const T* __restrict__ sxy,
    T* __restrict__ dvy, T* __restrict__ dvx,
    const T* __restrict__ dsyy, const T* __restrict__ dsxx, const T* __restrict__ dsxy,
    T* __restrict__ m_sigmayyy, T* __restrict__ m_sigmaxyx,
    T* __restrict__ m_sigmaxyy, T* __restrict__ m_sigmaxxx,
    T* __restrict__ dm_sigmayyy, T* __restrict__ dm_sigmaxyx,
    T* __restrict__ dm_sigmaxyy, T* __restrict__ dm_sigmaxxx,
    const T* __restrict__ buoyancy_y, const T* __restrict__ buoyancy_x,
    const T* __restrict__ dbuoyancy_y, const T* __restrict__ dbuoyancy_x,
    T* __restrict__ dvydb_store, T* __restrict__ dvxdb_store,
    T* __restrict__ ddvydb_store, T* __restrict__ ddvxdb_store,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ c,
    int fd_pad_y0, int fd_pad_y1, int fd_pad_x0, int fd_pad_x1,
    T rdy, T rdx, T dtv,
    int t, int interval, int n_shots, int ny, int nx,
    int model_batched, int scatter_batched, int store)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const long ny_nx = (long)ny * nx;
    const long off_s = (long)s * ny_nx;
    const int s_m = model_batched ? s : 0;
    const int s_sc = scatter_batched ? s : 0;
    const T* by_s = buoyancy_y + (long)s_m * ny_nx;
    const T* bx_s = buoyancy_x + (long)s_m * ny_nx;
    const T* dby_s = dbuoyancy_y + (long)s_sc * ny_nx;
    const T* dbx_s = dbuoyancy_x + (long)s_sc * ny_nx;

    if (y >= fd_pad_y0 && y < ny - fd_pad_y0 && x >= fd_pad_x0 &&
        x < nx - fd_pad_x1) {
        const long off = off_s + (long)y * nx + x;
        T d2 = diff_int_y(syy, off, c, rdy, nx, fd_pad_y0);
        m_sigmayyy[off] = ayh[y] * m_sigmayyy[off] + byh[y] * d2;
        d2 += m_sigmayyy[off];
        T d1 = diff_half_x(sxy, off, c, rdx, fd_pad_x0);
        m_sigmaxyx[off] = ax[x] * m_sigmaxyx[off] + bx[x] * d1;
        d1 += m_sigmaxyx[off];
        const T w_y = d2 + d1;
        vy[off] += by_s[(long)y * nx + x] * dtv * w_y;
        T dd2 = diff_int_y(dsyy, off, c, rdy, nx, fd_pad_y0);
        dm_sigmayyy[off] = ayh[y] * dm_sigmayyy[off] + byh[y] * dd2;
        dd2 += dm_sigmayyy[off];
        T dd1 = diff_half_x(dsxy, off, c, rdx, fd_pad_x0);
        dm_sigmaxyx[off] = ax[x] * dm_sigmaxyx[off] + bx[x] * dd1;
        dd1 += dm_sigmaxyx[off];
        const T dw_y = dd2 + dd1;
        dvy[off] += by_s[(long)y * nx + x] * dtv * dw_y
                  + dby_s[(long)y * nx + x] * dtv * w_y;
        if (store && t % interval == 0) {
            const long soff = (((long)(t / interval) * n_shots + s) * ny + y) * nx + x;
            dvydb_store[soff] = dtv * w_y;
            ddvydb_store[soff] = dtv * dw_y;
        }
    }
    if (y >= fd_pad_y0 && y < ny - fd_pad_y1 && x >= fd_pad_x0 &&
        x < nx - fd_pad_x0) {
        const long off = off_s + (long)y * nx + x;
        T d2 = diff_half_y(sxy, off, c, rdy, nx, fd_pad_y0);
        m_sigmaxyy[off] = ay[y] * m_sigmaxyy[off] + by[y] * d2;
        d2 += m_sigmaxyy[off];
        T d1 = diff_int_x(sxx, off, c, rdx, fd_pad_x0);
        m_sigmaxxx[off] = axh[x] * m_sigmaxxx[off] + bxh[x] * d1;
        d1 += m_sigmaxxx[off];
        const T w_x = d1 + d2;
        vx[off] += bx_s[(long)y * nx + x] * dtv * w_x;
        T dd2 = diff_half_y(dsxy, off, c, rdy, nx, fd_pad_y0);
        dm_sigmaxyy[off] = ay[y] * dm_sigmaxyy[off] + by[y] * dd2;
        dd2 += dm_sigmaxyy[off];
        T dd1 = diff_int_x(dsxx, off, c, rdx, fd_pad_x0);
        dm_sigmaxxx[off] = axh[x] * dm_sigmaxxx[off] + bxh[x] * dd1;
        dd1 += dm_sigmaxxx[off];
        const T dw_x = dd1 + dd2;
        dvx[off] += bx_s[(long)y * nx + x] * dtv * dw_x
                  + dbx_s[(long)y * nx + x] * dtv * w_x;
        if (store && t % interval == 0) {
            const long soff = (((long)(t / interval) * n_shots + s) * ny + y) * nx + x;
            dvxdb_store[soff] = dtv * w_x;
            ddvxdb_store[soff] = dtv * dw_x;
        }
    }
}

// ---------------- forward: stress update (background + scattered) ----------------
// sigmaii (y in [fd_pad_y0, ny-fd_pad_y1), x in [fd_pad_x0, nx-fd_pad_x1)):
//   syy += dtv*(lamb*ssum + 2*mu*dvydy)        sxx += dtv*(lamb*ssum + 2*mu*dvxdx)
//   dsyy += dtv*(lamb*dssum + 2*mu*ddvydy) + dtv*(dlamb*ssum + 2*dmu*dvydy)
//   dsxx += dtv*(lamb*dssum + 2*mu*ddvxdx) + dtv*(dlamb*ssum + 2*dmu*dvxdx)
// sigmaxy (y in [fd_pad_y0, ny-fd_pad_y0), x in [fd_pad_x0, nx-fd_pad_x0)):
//   sxy += dtv*mu_yx*w_sum    dsxy += dtv*mu_yx*dw_sum + dtv*dmu_yx*w_sum
// The PML-modified derivatives are snapshotted for the imaging conditions.
template <typename T>
__global__ void born_step_stress_kernel(
    const T* __restrict__ vy, const T* __restrict__ vx,
    const T* __restrict__ dvy, const T* __restrict__ dvx,
    T* __restrict__ syy, T* __restrict__ sxx, T* __restrict__ sxy,
    T* __restrict__ dsyy, T* __restrict__ dsxx, T* __restrict__ dsxy,
    T* __restrict__ m_vyy, T* __restrict__ m_vxx,
    T* __restrict__ m_vxy, T* __restrict__ m_vyx,
    T* __restrict__ dm_vyy, T* __restrict__ dm_vxx,
    T* __restrict__ dm_vxy, T* __restrict__ dm_vyx,
    const T* __restrict__ lamb, const T* __restrict__ mu, const T* __restrict__ mu_yx,
    const T* __restrict__ dlamb, const T* __restrict__ dmu, const T* __restrict__ dmu_yx,
    T* __restrict__ dvydy_store, T* __restrict__ dvxdx_store,
    T* __restrict__ dvxy_store,
    T* __restrict__ ddvydy_store, T* __restrict__ ddvxdx_store,
    T* __restrict__ ddvxy_store,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ c,
    int fd_pad_y0, int fd_pad_y1, int fd_pad_x0, int fd_pad_x1,
    T rdy, T rdx, T dtv,
    int t, int interval, int n_shots, int ny, int nx,
    int model_batched, int scatter_batched, int store)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const long ny_nx = (long)ny * nx;
    const long off_s = (long)s * ny_nx;
    const int s_m = model_batched ? s : 0;
    const int s_sc = scatter_batched ? s : 0;
    const T* lamb_s = lamb + (long)s_m * ny_nx;
    const T* mu_s = mu + (long)s_m * ny_nx;
    const T* mu_yx_s = mu_yx + (long)s_m * ny_nx;
    const T* dlamb_s = dlamb + (long)s_sc * ny_nx;
    const T* dmu_s = dmu + (long)s_sc * ny_nx;
    const T* dmu_yx_s = dmu_yx + (long)s_sc * ny_nx;

    if (y >= fd_pad_y0 && x >= fd_pad_x0 && y < ny - fd_pad_y1 &&
        x < nx - fd_pad_x1) {
        const long off = off_s + (long)y * nx + x;
        T dvydy = diff_half_y(vy, off, c, rdy, nx, fd_pad_y0);
        m_vyy[off] = ay[y] * m_vyy[off] + by[y] * dvydy;
        dvydy += m_vyy[off];
        T dvxdx = diff_half_x(vx, off, c, rdx, fd_pad_x0);
        m_vxx[off] = ax[x] * m_vxx[off] + bx[x] * dvxdx;
        dvxdx += m_vxx[off];
        const T ssum = dvydy + dvxdx;
        const T lamb_v = lamb_s[(long)y * nx + x];
        const T mu_v = mu_s[(long)y * nx + x];
        syy[off] += dtv * (lamb_v * ssum + (T)2 * mu_v * dvydy);
        sxx[off] += dtv * (lamb_v * ssum + (T)2 * mu_v * dvxdx);
        T ddvydy = diff_half_y(dvy, off, c, rdy, nx, fd_pad_y0);
        dm_vyy[off] = ay[y] * dm_vyy[off] + by[y] * ddvydy;
        ddvydy += dm_vyy[off];
        T ddvxdx = diff_half_x(dvx, off, c, rdx, fd_pad_x0);
        dm_vxx[off] = ax[x] * dm_vxx[off] + bx[x] * ddvxdx;
        ddvxdx += dm_vxx[off];
        const T dssum = ddvydy + ddvxdx;
        const T dlamb_v = dlamb_s[(long)y * nx + x];
        const T dmu_v = dmu_s[(long)y * nx + x];
        dsyy[off] += dtv * (lamb_v * dssum + (T)2 * mu_v * ddvydy)
                   + dtv * (dlamb_v * ssum + (T)2 * dmu_v * dvydy);
        dsxx[off] += dtv * (lamb_v * dssum + (T)2 * mu_v * ddvxdx)
                   + dtv * (dlamb_v * ssum + (T)2 * dmu_v * dvxdx);
        if (store && t % interval == 0) {
            const long soff = (((long)(t / interval) * n_shots + s) * ny + y) * nx + x;
            dvydy_store[soff] = dtv * dvydy;
            dvxdx_store[soff] = dtv * dvxdx;
            ddvydy_store[soff] = dtv * ddvydy;
            ddvxdx_store[soff] = dtv * ddvxdx;
        }
    }
    if (y >= fd_pad_y0 && x >= fd_pad_x0 && y < ny - fd_pad_y0 &&
        x < nx - fd_pad_x0) {
        const long off = off_s + (long)y * nx + x;
        T dvxdy = diff_int_y(vx, off, c, rdy, nx, fd_pad_y0);
        m_vxy[off] = ayh[y] * m_vxy[off] + byh[y] * dvxdy;
        dvxdy += m_vxy[off];
        T dvydx = diff_int_x(vy, off, c, rdx, fd_pad_x0);
        m_vyx[off] = axh[x] * m_vyx[off] + bxh[x] * dvydx;
        dvydx += m_vyx[off];
        const T w_sum = dvydx + dvxdy;
        sxy[off] += dtv * mu_yx_s[(long)y * nx + x] * w_sum;
        T ddvxdy = diff_int_y(dvx, off, c, rdy, nx, fd_pad_y0);
        dm_vxy[off] = ayh[y] * dm_vxy[off] + byh[y] * ddvxdy;
        ddvxdy += dm_vxy[off];
        T ddvydx = diff_int_x(dvy, off, c, rdx, fd_pad_x0);
        dm_vyx[off] = axh[x] * dm_vyx[off] + bxh[x] * ddvydx;
        ddvydx += dm_vyx[off];
        const T dw_sum = ddvydx + ddvxdy;
        dsxy[off] += dtv * mu_yx_s[(long)y * nx + x] * dw_sum
                   + dtv * dmu_yx_s[(long)y * nx + x] * w_sum;
        if (store && t % interval == 0) {
            const long soff = (((long)(t / interval) * n_shots + s) * ny + y) * nx + x;
            dvxy_store[soff] = dtv * w_sum;
            ddvxy_store[soff] = dtv * dw_sum;
        }
    }
}

// ---------------- forward: background source injection / scattered recording ----------------
template <typename T>
__global__ void born_inject_pressure_kernel(
    T* __restrict__ syy, T* __restrict__ sxx, const T* __restrict__ f,
    const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        if (idx >= 0) {
            const long off = (long)s * ny_nx + idx;
            const T f_val = f[(((long)t * n_shots + s) * n_src + k)];
            syy[off] += f_val;
            sxx[off] += f_val;
        }
    }
}

template <typename T>
__global__ void born_record_pressure_kernel(
    const T* __restrict__ dsyy, const T* __restrict__ dsxx, T* __restrict__ r,
    const long* __restrict__ rec_i,
    int t, int n_shots, int n_rec, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            r[(((long)t * n_shots + s) * n_rec + k)]
                = dsyy[(long)s * ny_nx + idx] + dsxx[(long)s * ny_nx + idx];
    }
}

// ---------------- backward: source gradient / receiver injection ----------------
template <typename T>
__global__ void born_record_grad_f_p_kernel(
    const T* __restrict__ l_syy, const T* __restrict__ l_sxx, T* __restrict__ grad_f,
    const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        grad_f[(((long)t * n_shots + s) * n_src + k)] =
            (idx >= 0) ? (l_syy[(long)s * ny_nx + idx] + l_sxx[(long)s * ny_nx + idx])
                       : (T)0;
    }
}

template <typename T>
__global__ void born_add_grad_r_kernel(
    T* __restrict__ l_dsyy, T* __restrict__ l_dsxx, const T* __restrict__ grad_r,
    const long* __restrict__ rec_i,
    int t, int n_shots, int n_rec, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0) {
            const long off = (long)s * ny_nx + idx;
            const T g = grad_r[(((long)t * n_shots + s) * n_rec + k)];
            l_dsyy[off] += g;
            l_dsxx[off] += g;
        }
    }
}


// ---------------- backward: transpose of the combined stress update ----------------
// From the post-stress adjoint stresses produces the post-velocity adjoint
// velocities, the buoyancy (and scatter-buoyancy) imaging conditions, and
// the alternating m_sigma* adjoint-memory buffers.  This is the transpose
// of the forward STRESS update (the m_v* adjoint memories are read at the
// stencil points, with the lamb/mu parameters position-dependent there)
// together with the transpose of the forward VELOCITY memory recursions
// (the m_sig* ``new`` buffers are written from the updated adjoint
// velocities).  The in-place m_v* updates and the lamb/mu/mu_yx imaging
// conditions live in born_adjoint_stress_kernel.
template <typename T>
__global__ void born_adjoint_velocity_kernel(
    const T* __restrict__ lamb, const T* __restrict__ mu, const T* __restrict__ mu_yx,
    const T* __restrict__ dlamb, const T* __restrict__ dmu, const T* __restrict__ dmu_yx,
    T* __restrict__ l_vy, T* __restrict__ l_vx,
    T* __restrict__ l_dvy, T* __restrict__ l_dvx,
    const T* __restrict__ l_syy, const T* __restrict__ l_sxx, const T* __restrict__ l_sxy,
    const T* __restrict__ l_dsyy, const T* __restrict__ l_dsxx, const T* __restrict__ l_dsxy,
    const T* __restrict__ m_vyy, const T* __restrict__ m_vxx,
    const T* __restrict__ m_vxy, const T* __restrict__ m_vyx,
    const T* __restrict__ dm_vyy, const T* __restrict__ dm_vxx,
    const T* __restrict__ dm_vxy, const T* __restrict__ dm_vyx,
    const T* __restrict__ m_syyy_old, const T* __restrict__ m_syx_old,
    const T* __restrict__ m_syxy_old, const T* __restrict__ m_syxx_old,
    T* __restrict__ m_syyy_new, T* __restrict__ m_syx_new,
    T* __restrict__ m_syxy_new, T* __restrict__ m_syxx_new,
    const T* __restrict__ dm_syyy_old, const T* __restrict__ dm_syx_old,
    const T* __restrict__ dm_syxy_old, const T* __restrict__ dm_syxx_old,
    T* __restrict__ dm_syyy_new, T* __restrict__ dm_syx_new,
    T* __restrict__ dm_syxy_new, T* __restrict__ dm_syxx_new,
    const T* __restrict__ buoyancy_y, const T* __restrict__ buoyancy_x,
    const T* __restrict__ dbuoyancy_y, const T* __restrict__ dbuoyancy_x,
    T* __restrict__ grad_buoyancy_y, T* __restrict__ grad_buoyancy_x,
    T* __restrict__ grad_dbuoyancy_y, T* __restrict__ grad_dbuoyancy_x,
    const T* __restrict__ dvydb_store, const T* __restrict__ dvxdb_store,
    const T* __restrict__ ddvydb_store, const T* __restrict__ ddvxdb_store,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ c,
    int fd_pad_y0, int fd_pad_y1, int fd_pad_x0, int fd_pad_x1,
    T rdy, T rdx, T dtv,
    int t, int interval, int n_shots, int ny, int nx,
    int model_batched, int scatter_batched)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const long ny_nx = (long)ny * nx;
    const long off_s = (long)s * ny_nx;
    const int s_m = model_batched ? s : 0;
    const int s_sc = scatter_batched ? s : 0;
    const T* lamb_s = lamb + (long)s_m * ny_nx;
    const T* mu_s = mu + (long)s_m * ny_nx;
    const T* mu_yx_s = mu_yx + (long)s_m * ny_nx;
    const T* dlamb_s = dlamb + (long)s_sc * ny_nx;
    const T* dmu_s = dmu + (long)s_sc * ny_nx;
    const T* dmu_yx_s = dmu_yx + (long)s_sc * ny_nx;
    const T* by_s = buoyancy_y + (long)s_m * ny_nx;
    const T* bx_s = buoyancy_x + (long)s_m * ny_nx;
    const T* dby_s = dbuoyancy_y + (long)s_sc * ny_nx;
    const T* dbx_s = dbuoyancy_x + (long)s_sc * ny_nx;

    // vy: y in [fd_pad_y0, ny-fd_pad_y0), x in [fd_pad_x0, nx-fd_pad_x1)
    if (y >= fd_pad_y0 && y < ny - fd_pad_y0 && x >= fd_pad_x0 &&
        x < nx - fd_pad_x1) {
        const long off = off_s + (long)y * nx + x;
        // l_vy y-term: -diff_int_y(W_y), W_y = (1+by)*A_y + by*m_vyy
        // (the stencil points are the integer-row syy/sxx cells, where the
        //  forward m_vyy memory lives, so the transpose gathers ``by`` there)
        T dy = (T)0;
        for (int k = 1; k <= fd_pad_y0; ++k) {
            const long off_m = off - (long)(k - 1) * nx;
            const long off_p = off + (long)k * nx;
            const int ym = y - (k - 1), yp = y + k;
            const T lam_m = lamb_s[off_m];
            const T l2m_m = lam_m + (T)2 * mu_s[off_m];
            const T d2m_m = (T)2 * dmu_s[off_m] + dlamb_s[off_m];
            const T A_m = dtv * (l2m_m * l_syy[off_m] + lam_m * l_sxx[off_m]
                              + d2m_m * l_dsyy[off_m] + dlamb_s[off_m] * l_dsxx[off_m]);
            const T lam_p = lamb_s[off_p];
            const T l2m_p = lam_p + (T)2 * mu_s[off_p];
            const T d2m_p = (T)2 * dmu_s[off_p] + dlamb_s[off_p];
            const T A_p = dtv * (l2m_p * l_syy[off_p] + lam_p * l_sxx[off_p]
                              + d2m_p * l_dsyy[off_p] + dlamb_s[off_p] * l_dsxx[off_p]);
            dy += c[k - 1] *
                  (((T)1 + by[ym]) * A_m + by[ym] * m_vyy[off_m] -
                   ((T)1 + by[yp]) * A_p - by[yp] * m_vyy[off_p]);
        }
        // l_dvy y-term: W_dy = (1+by)*A_dy + by*dm_vyy
        T ddy = (T)0;
        for (int k = 1; k <= fd_pad_y0; ++k) {
            const long off_m = off - (long)(k - 1) * nx;
            const long off_p = off + (long)k * nx;
            const int ym = y - (k - 1), yp = y + k;
            const T lam_m = lamb_s[off_m];
            const T l2m_m = lam_m + (T)2 * mu_s[off_m];
            const T A_dm = dtv * (l2m_m * l_dsyy[off_m] + lam_m * l_dsxx[off_m]);
            const T lam_p = lamb_s[off_p];
            const T l2m_p = lam_p + (T)2 * mu_s[off_p];
            const T A_dp = dtv * (l2m_p * l_dsyy[off_p] + lam_p * l_dsxx[off_p]);
            ddy += c[k - 1] *
                   (((T)1 + by[ym]) * A_dm + by[ym] * dm_vyy[off_m] -
                    ((T)1 + by[yp]) * A_dp - by[yp] * dm_vyy[off_p]);
        }
        // l_vy x-term: -diff_half_x(W_x), W_x = (1+bxh)*B_x + bxh*m_vyx
        // (the stencil points are the half-x sxy cells, where m_vyx lives)
        T dx = (T)0;
        for (int k = 1; k <= fd_pad_x0; ++k) {
            const long off_m = off - k;
            const long off_p = off + k - 1;
            const int xm = x - k, xp = x + k - 1;
            const T B_m = dtv * (mu_yx_s[off_m] * l_sxy[off_m]
                               + dmu_yx_s[off_m] * l_dsxy[off_m]);
            const T B_p = dtv * (mu_yx_s[off_p] * l_sxy[off_p]
                               + dmu_yx_s[off_p] * l_dsxy[off_p]);
            dx += c[k - 1] *
                  (((T)1 + bxh[xm]) * B_m + bxh[xm] * m_vyx[off_m] -
                   ((T)1 + bxh[xp]) * B_p - bxh[xp] * m_vyx[off_p]);
        }
        // l_dvy x-term: W_dx = (1+bxh)*B_dx + bxh*dm_vyx
        T ddx = (T)0;
        for (int k = 1; k <= fd_pad_x0; ++k) {
            const long off_m = off - k;
            const long off_p = off + k - 1;
            const int xm = x - k, xp = x + k - 1;
            const T B_dm = dtv * mu_yx_s[off_m] * l_dsxy[off_m];
            const T B_dp = dtv * mu_yx_s[off_p] * l_dsxy[off_p];
            ddx += c[k - 1] *
                   (((T)1 + bxh[xm]) * B_dm + bxh[xm] * dm_vyx[off_m] -
                    ((T)1 + bxh[xp]) * B_dp - bxh[xp] * dm_vyx[off_p]);
        }
        l_vy[off] += dy * rdy + dx * rdx;
        l_dvy[off] += ddy * rdy + ddx * rdx;
        const long yx = (long)y * nx + x;
        const T vy_new = l_vy[off];
        const T dvy_new = l_dvy[off];
        const T b_y = by_s[yx];
        m_syyy_new[off] = b_y * dtv * ayh[y] * vy_new + ayh[y] * m_syyy_old[off];
        m_syx_new[off] = b_y * dtv * ax[x] * vy_new + ax[x] * m_syx_old[off];
        dm_syyy_new[off] = b_y * dtv * ayh[y] * dvy_new + ayh[y] * dm_syyy_old[off];
        dm_syx_new[off] = b_y * dtv * ax[x] * dvy_new + ax[x] * dm_syx_old[off];
        if (t % interval == 0) {
            const long soff = (((long)(t / interval) * n_shots + s) * ny + y) * nx + x;
            grad_buoyancy_y[off] += vy_new * dvydb_store[soff]
                                  + dvy_new * ddvydb_store[soff];
            grad_dbuoyancy_y[off] += dvy_new * dvydb_store[soff];
        }
    }
    // vx: y in [fd_pad_y0, ny-fd_pad_y1), x in [fd_pad_x0, nx-fd_pad_x0)
    if (y >= fd_pad_y0 && y < ny - fd_pad_y1 && x >= fd_pad_x0 &&
        x < nx - fd_pad_x0) {
        const long off = off_s + (long)y * nx + x;
        // l_vx y-term: -diff_half_y(W_y'), W_y' = (1+byh)*C_y + byh*m_vxy
        T dy = (T)0;
        for (int k = 1; k <= fd_pad_y0; ++k) {
            const long off_m = off - (long)k * nx;
            const long off_p = off + (long)(k - 1) * nx;
            const int ym = y - k, yp = y + k - 1;
            const T C_m = dtv * (mu_yx_s[off_m] * l_sxy[off_m]
                               + dmu_yx_s[off_m] * l_dsxy[off_m]);
            const T C_p = dtv * (mu_yx_s[off_p] * l_sxy[off_p]
                               + dmu_yx_s[off_p] * l_dsxy[off_p]);
            dy += c[k - 1] *
                  (((T)1 + byh[ym]) * C_m + byh[ym] * m_vxy[off_m] -
                   ((T)1 + byh[yp]) * C_p - byh[yp] * m_vxy[off_p]);
        }
        // l_dvx y-term: W_dy' = (1+byh)*C_dy + byh*dm_vxy
        T ddy = (T)0;
        for (int k = 1; k <= fd_pad_y0; ++k) {
            const long off_m = off - (long)k * nx;
            const long off_p = off + (long)(k - 1) * nx;
            const int ym = y - k, yp = y + k - 1;
            const T C_dm = dtv * mu_yx_s[off_m] * l_dsxy[off_m];
            const T C_dp = dtv * mu_yx_s[off_p] * l_dsxy[off_p];
            ddy += c[k - 1] *
                   (((T)1 + byh[ym]) * C_dm + byh[ym] * dm_vxy[off_m] -
                    ((T)1 + byh[yp]) * C_dp - byh[yp] * dm_vxy[off_p]);
        }
        // l_vx x-term: -diff_int_x(W_xx), W_xx = (1+bx)*A_x + bx*m_vxx
        // (the stencil points are the integer-x sxx cells, where m_vxx lives)
        T dx = (T)0;
        for (int k = 1; k <= fd_pad_x0; ++k) {
            const long off_m = off - k + 1;
            const long off_p = off + k;
            const int xm = x - k + 1, xp = x + k;
            const T lam_m = lamb_s[off_m];
            const T l2m_m = lam_m + (T)2 * mu_s[off_m];
            const T d2m_m = (T)2 * dmu_s[off_m] + dlamb_s[off_m];
            const T A_m = dtv * (l2m_m * l_sxx[off_m] + lam_m * l_syy[off_m]
                              + d2m_m * l_dsxx[off_m] + dlamb_s[off_m] * l_dsyy[off_m]);
            const T lam_p = lamb_s[off_p];
            const T l2m_p = lam_p + (T)2 * mu_s[off_p];
            const T d2m_p = (T)2 * dmu_s[off_p] + dlamb_s[off_p];
            const T A_p = dtv * (l2m_p * l_sxx[off_p] + lam_p * l_syy[off_p]
                              + d2m_p * l_dsxx[off_p] + dlamb_s[off_p] * l_dsyy[off_p]);
            dx += c[k - 1] *
                  (((T)1 + bx[xm]) * A_m + bx[xm] * m_vxx[off_m] -
                   ((T)1 + bx[xp]) * A_p - bx[xp] * m_vxx[off_p]);
        }
        // l_dvx x-term: W_dxx = (1+bx)*A_dx + bx*dm_vxx
        T ddx = (T)0;
        for (int k = 1; k <= fd_pad_x0; ++k) {
            const long off_m = off - k + 1;
            const long off_p = off + k;
            const int xm = x - k + 1, xp = x + k;
            const T lam_m = lamb_s[off_m];
            const T l2m_m = lam_m + (T)2 * mu_s[off_m];
            const T A_dm = dtv * (l2m_m * l_dsxx[off_m] + lam_m * l_dsyy[off_m]);
            const T lam_p = lamb_s[off_p];
            const T l2m_p = lam_p + (T)2 * mu_s[off_p];
            const T A_dp = dtv * (l2m_p * l_dsxx[off_p] + lam_p * l_dsyy[off_p]);
            ddx += c[k - 1] *
                   (((T)1 + bx[xm]) * A_dm + bx[xm] * dm_vxx[off_m] -
                    ((T)1 + bx[xp]) * A_dp - bx[xp] * dm_vxx[off_p]);
        }
        l_vx[off] += dy * rdy + dx * rdx;
        l_dvx[off] += ddy * rdy + ddx * rdx;
        const long yx = (long)y * nx + x;
        const T vx_new = l_vx[off];
        const T dvx_new = l_dvx[off];
        const T b_x = bx_s[yx];
        m_syxy_new[off] = b_x * dtv * ay[y] * vx_new + ay[y] * m_syxy_old[off];
        m_syxx_new[off] = b_x * dtv * axh[x] * vx_new + axh[x] * m_syxx_old[off];
        dm_syxy_new[off] = b_x * dtv * ay[y] * dvx_new + ay[y] * dm_syxy_old[off];
        dm_syxx_new[off] = b_x * dtv * axh[x] * dvx_new + axh[x] * dm_syxx_old[off];
        if (t % interval == 0) {
            const long soff = (((long)(t / interval) * n_shots + s) * ny + y) * nx + x;
            grad_buoyancy_x[off] += vx_new * dvxdb_store[soff]
                                  + dvx_new * ddvxdb_store[soff];
            grad_dbuoyancy_x[off] += dvx_new * dvxdb_store[soff];
        }
    }
}

// ---------------- backward: transpose of the combined velocity update ----------------
// From the post-velocity adjoint velocities produces the pre-velocity
// adjoint stresses (via the derivative + alternating m_sigma* memory
// transposes), the in-place m_v* adjoint-memory updates, and the
// lamb/mu/mu_yx (and scatter) imaging conditions.
template <typename T>
__global__ void born_adjoint_stress_kernel(
    const T* __restrict__ buoyancy_y, const T* __restrict__ buoyancy_x,
    const T* __restrict__ dbuoyancy_y, const T* __restrict__ dbuoyancy_x,
    const T* __restrict__ l_vy, const T* __restrict__ l_vx,
    const T* __restrict__ l_dvy, const T* __restrict__ l_dvx,
    T* __restrict__ l_syy, T* __restrict__ l_sxx, T* __restrict__ l_sxy,
    T* __restrict__ l_dsyy, T* __restrict__ l_dsxx, T* __restrict__ l_dsxy,
    T* __restrict__ m_vyy, T* __restrict__ m_vxx,
    T* __restrict__ m_vxy, T* __restrict__ m_vyx,
    T* __restrict__ dm_vyy, T* __restrict__ dm_vxx,
    T* __restrict__ dm_vxy, T* __restrict__ dm_vyx,
    const T* __restrict__ lamb, const T* __restrict__ mu, const T* __restrict__ mu_yx,
    const T* __restrict__ dlamb, const T* __restrict__ dmu, const T* __restrict__ dmu_yx,
    const T* __restrict__ m_syyy_old, const T* __restrict__ m_syx_old,
    const T* __restrict__ m_syxy_old, const T* __restrict__ m_syxx_old,
    const T* __restrict__ dm_syyy_old, const T* __restrict__ dm_syx_old,
    const T* __restrict__ dm_syxy_old, const T* __restrict__ dm_syxx_old,
    T* __restrict__ grad_lamb, T* __restrict__ grad_mu, T* __restrict__ grad_mu_yx,
    T* __restrict__ grad_dlamb, T* __restrict__ grad_dmu, T* __restrict__ grad_dmu_yx,
    const T* __restrict__ dvydy_store, const T* __restrict__ dvxdx_store,
    const T* __restrict__ dvxy_store,
    const T* __restrict__ ddvydy_store, const T* __restrict__ ddvxdx_store,
    const T* __restrict__ ddvxy_store,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ c,
    int fd_pad_y0, int fd_pad_y1, int fd_pad_x0, int fd_pad_x1,
    T rdy, T rdx, T dtv,
    int t, int interval, int n_shots, int ny, int nx,
    int model_batched, int scatter_batched)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const long ny_nx = (long)ny * nx;
    const long off_s = (long)s * ny_nx;
    const int s_m = model_batched ? s : 0;
    const int s_sc = scatter_batched ? s : 0;
    const T* lamb_s = lamb + (long)s_m * ny_nx;
    const T* mu_s = mu + (long)s_m * ny_nx;
    const T* mu_yx_s = mu_yx + (long)s_m * ny_nx;
    const T* dlamb_s = dlamb + (long)s_sc * ny_nx;
    const T* dmu_s = dmu + (long)s_sc * ny_nx;
    const T* dmu_yx_s = dmu_yx + (long)s_sc * ny_nx;
    const T* by_s = buoyancy_y + (long)s_m * ny_nx;
    const T* bx_s = buoyancy_x + (long)s_m * ny_nx;
    const T* dby_s = dbuoyancy_y + (long)s_sc * ny_nx;
    const T* dbx_s = dbuoyancy_x + (long)s_sc * ny_nx;

    // sigmaii: y in [fd_pad_y0, ny-fd_pad_y1), x in [fd_pad_x0, nx-fd_pad_x1)
    if (y >= fd_pad_y0 && x >= fd_pad_x0 && y < ny - fd_pad_y1 &&
        x < nx - fd_pad_x1) {
        const long off = off_s + (long)y * nx + x;
        const T syy_v = l_syy[off];
        const T sxx_v = l_sxx[off];
        const T dsyy_v = l_dsyy[off];
        const T dsxx_v = l_dsxx[off];
        if (t % interval == 0) {
            const long soff = (((long)(t / interval) * n_shots + s) * ny + y) * nx + x;
            const T dy_bg = dvydy_store[soff];
            const T dx_bg = dvxdx_store[soff];
            const T dy_sc = ddvydy_store[soff];
            const T dx_sc = ddvxdx_store[soff];
            grad_lamb[off] += (dy_bg + dx_bg) * (syy_v + sxx_v)
                            + (dy_sc + dx_sc) * (dsyy_v + dsxx_v);
            grad_dlamb[off] += (dy_bg + dx_bg) * (dsyy_v + dsxx_v);
            grad_mu[off] += (T)2 * (dy_bg * syy_v + dx_bg * sxx_v
                                  + dy_sc * dsyy_v + dx_sc * dsxx_v);
            grad_dmu[off] += (T)2 * (dy_bg * dsyy_v + dx_bg * dsxx_v);
        }
        const long yx = (long)y * nx + x;
        const T lam_v = lamb_s[yx];
        const T l2m_v = lam_v + (T)2 * mu_s[yx];
        const T dl_v = dlamb_s[yx];
        const T d2m_v = (T)2 * dmu_s[yx] + dl_v;
        m_vyy[off] = l2m_v * dtv * ay[y] * syy_v + lam_v * dtv * ay[y] * sxx_v
                   + d2m_v * dtv * ay[y] * dsyy_v + dl_v * dtv * ay[y] * dsxx_v
                   + ay[y] * m_vyy[off];
        dm_vyy[off] = l2m_v * dtv * ay[y] * dsyy_v + lam_v * dtv * ay[y] * dsxx_v
                    + ay[y] * dm_vyy[off];
        m_vxx[off] = l2m_v * dtv * ax[x] * sxx_v + lam_v * dtv * ax[x] * syy_v
                   + d2m_v * dtv * ax[x] * dsxx_v + dl_v * dtv * ax[x] * dsyy_v
                   + ax[x] * m_vxx[off];
        dm_vxx[off] = l2m_v * dtv * ax[x] * dsxx_v + lam_v * dtv * ax[x] * dsyy_v
                    + ax[x] * dm_vxx[off];
        // l_syy: -diff_half_y(W_yy), W_yy = dtv*(1+byh)*by_s*l_vy
        //                                  + dtv*(1+byh)*dby_s*l_dvy + byh*m_syyy_old
        T dy = (T)0;
        for (int k = 1; k <= fd_pad_y0; ++k) {
            const long off_m = off - (long)k * nx;
            const long off_p = off + (long)(k - 1) * nx;
            const int ym = y - k, yp = y + k - 1;
            dy += c[k - 1] *
                  (dtv * ((T)1 + byh[ym]) * by_s[off_m] * l_vy[off_m] +
                       dtv * ((T)1 + byh[ym]) * dby_s[off_m] * l_dvy[off_m] +
                       byh[ym] * m_syyy_old[off_m] -
                   (dtv * ((T)1 + byh[yp]) * by_s[off_p] * l_vy[off_p] +
                       dtv * ((T)1 + byh[yp]) * dby_s[off_p] * l_dvy[off_p] +
                       byh[yp] * m_syyy_old[off_p]));
        }
        l_syy[off] += dy * rdy;
        // l_dsyy: W_yy_sc = dtv*(1+byh)*by_s*l_dvy + byh*dm_syyy_old
        T ddy = (T)0;
        for (int k = 1; k <= fd_pad_y0; ++k) {
            const long off_m = off - (long)k * nx;
            const long off_p = off + (long)(k - 1) * nx;
            const int ym = y - k, yp = y + k - 1;
            ddy += c[k - 1] *
                   (dtv * ((T)1 + byh[ym]) * by_s[off_m] * l_dvy[off_m] +
                       byh[ym] * dm_syyy_old[off_m] -
                    (dtv * ((T)1 + byh[yp]) * by_s[off_p] * l_dvy[off_p] +
                       byh[yp] * dm_syyy_old[off_p]));
        }
        l_dsyy[off] += ddy * rdy;
        // l_sxx: -diff_half_x(W_xx'), W_xx' = dtv*(1+bxh)*bx_s*l_vx
        //                                     + dtv*(1+bxh)*dbx_s*l_dvx + bxh*m_syxx_old
        T dx = (T)0;
        for (int k = 1; k <= fd_pad_x0; ++k) {
            const long off_m = off - k;
            const long off_p = off + k - 1;
            const int xm = x - k, xp = x + k - 1;
            dx += c[k - 1] *
                  (dtv * ((T)1 + bxh[xm]) * bx_s[off_m] * l_vx[off_m] +
                       dtv * ((T)1 + bxh[xm]) * dbx_s[off_m] * l_dvx[off_m] +
                       bxh[xm] * m_syxx_old[off_m] -
                   (dtv * ((T)1 + bxh[xp]) * bx_s[off_p] * l_vx[off_p] +
                       dtv * ((T)1 + bxh[xp]) * dbx_s[off_p] * l_dvx[off_p] +
                       bxh[xp] * m_syxx_old[off_p]));
        }
        l_sxx[off] += dx * rdx;
        // l_dsxx: W_xx'_sc = dtv*(1+bxh)*bx_s*l_dvx + bxh*dm_syxx_old
        T ddx = (T)0;
        for (int k = 1; k <= fd_pad_x0; ++k) {
            const long off_m = off - k;
            const long off_p = off + k - 1;
            const int xm = x - k, xp = x + k - 1;
            ddx += c[k - 1] *
                   (dtv * ((T)1 + bxh[xm]) * bx_s[off_m] * l_dvx[off_m] +
                       bxh[xm] * dm_syxx_old[off_m] -
                    (dtv * ((T)1 + bxh[xp]) * bx_s[off_p] * l_dvx[off_p] +
                       bxh[xp] * dm_syxx_old[off_p]));
        }
        l_dsxx[off] += ddx * rdx;
    }
    // sigmaxy: y in [fd_pad_y0, ny-fd_pad_y0), x in [fd_pad_x0, nx-fd_pad_x0)
    if (y >= fd_pad_y0 && x >= fd_pad_x0 && y < ny - fd_pad_y0 &&
        x < nx - fd_pad_x0) {
        const long off = off_s + (long)y * nx + x;
        const T sxy_v = l_sxy[off];
        const T dsxy_v = l_dsxy[off];
        if (t % interval == 0) {
            const long soff = (((long)(t / interval) * n_shots + s) * ny + y) * nx + x;
            grad_mu_yx[off] += dtv * sxy_v * dvxy_store[soff]
                             + dtv * dsxy_v * ddvxy_store[soff];
            grad_dmu_yx[off] += dtv * dsxy_v * dvxy_store[soff];
        }
        const long yx = (long)y * nx + x;
        const T mu_yx_v = mu_yx_s[yx];
        const T dmu_yx_v = dmu_yx_s[yx];
        m_vxy[off] = mu_yx_v * dtv * ayh[y] * sxy_v + ayh[y] * m_vxy[off]
                   + dmu_yx_v * dtv * ayh[y] * dsxy_v;
        dm_vxy[off] = mu_yx_v * dtv * ayh[y] * dsxy_v + ayh[y] * dm_vxy[off];
        m_vyx[off] = mu_yx_v * dtv * axh[x] * sxy_v + axh[x] * m_vyx[off]
                   + dmu_yx_v * dtv * axh[x] * dsxy_v;
        dm_vyx[off] = mu_yx_v * dtv * axh[x] * dsxy_v + axh[x] * dm_vyx[off];
        // l_sxy y-term: -diff_int_y(W_sxy_y), W_sxy_y = dtv*(1+by)*bx_s*l_vx
        //                                             + dtv*(1+by)*dbx_s*l_dvx + by*m_syxy_old
        T dy = (T)0;
        for (int k = 1; k <= fd_pad_y0; ++k) {
            const long off_m = off - (long)(k - 1) * nx;
            const long off_p = off + (long)k * nx;
            const int ym = y - (k - 1), yp = y + k;
            dy += c[k - 1] *
                  (dtv * ((T)1 + by[ym]) * bx_s[off_m] * l_vx[off_m] +
                       dtv * ((T)1 + by[ym]) * dbx_s[off_m] * l_dvx[off_m] +
                       by[ym] * m_syxy_old[off_m] -
                   (dtv * ((T)1 + by[yp]) * bx_s[off_p] * l_vx[off_p] +
                       dtv * ((T)1 + by[yp]) * dbx_s[off_p] * l_dvx[off_p] +
                       by[yp] * m_syxy_old[off_p]));
        }
        l_sxy[off] += dy * rdy;
        // l_dsxy y-term: W_sxy_y_sc = dtv*(1+by)*bx_s*l_dvx + by*dm_syxy_old
        T ddy = (T)0;
        for (int k = 1; k <= fd_pad_y0; ++k) {
            const long off_m = off - (long)(k - 1) * nx;
            const long off_p = off + (long)k * nx;
            const int ym = y - (k - 1), yp = y + k;
            ddy += c[k - 1] *
                   (dtv * ((T)1 + by[ym]) * bx_s[off_m] * l_dvx[off_m] +
                       by[ym] * dm_syxy_old[off_m] -
                    (dtv * ((T)1 + by[yp]) * bx_s[off_p] * l_dvx[off_p] +
                       by[yp] * dm_syxy_old[off_p]));
        }
        l_dsxy[off] += ddy * rdy;
        // l_sxy x-term: -diff_int_x(W_sxy_x), W_sxy_x = dtv*(1+bx)*by_s*l_vy
        //                                             + dtv*(1+bx)*dby_s*l_dvy + bx*m_syx_old
        T dx = (T)0;
        for (int k = 1; k <= fd_pad_x0; ++k) {
            const long off_m = off - k + 1;
            const long off_p = off + k;
            const int xm = x - k + 1, xp = x + k;
            dx += c[k - 1] *
                  (dtv * ((T)1 + bx[xm]) * by_s[off_m] * l_vy[off_m] +
                       dtv * ((T)1 + bx[xm]) * dby_s[off_m] * l_dvy[off_m] +
                       bx[xm] * m_syx_old[off_m] -
                   (dtv * ((T)1 + bx[xp]) * by_s[off_p] * l_vy[off_p] +
                       dtv * ((T)1 + bx[xp]) * dby_s[off_p] * l_dvy[off_p] +
                       bx[xp] * m_syx_old[off_p]));
        }
        l_sxy[off] += dx * rdx;
        // l_dsxy x-term: W_sxy_x_sc = dtv*(1+bx)*by_s*l_dvy + bx*dm_syx_old
        T ddx = (T)0;
        for (int k = 1; k <= fd_pad_x0; ++k) {
            const long off_m = off - k + 1;
            const long off_p = off + k;
            const int xm = x - k + 1, xp = x + k;
            ddx += c[k - 1] *
                   (dtv * ((T)1 + bx[xm]) * by_s[off_m] * l_dvy[off_m] +
                       bx[xm] * dm_syx_old[off_m] -
                    (dtv * ((T)1 + bx[xp]) * by_s[off_p] * l_dvy[off_p] +
                       bx[xp] * dm_syx_old[off_p]));
        }
        l_dsxy[off] += ddx * rdx;
    }
}

// ---------------- launchers ----------------
#define LAUNCH_GRID(KERN, T, ...)                                              \
    {                                                                          \
        dim3 block(16, 16);                                                    \
        dim3 grid((nx + 15) / 16, (ny + 15) / 16, n_shots);                    \
        KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami born_em_el " #KERN " failed"); \
    }

#define LAUNCH_PT(KERN, T, NSH, NLOC, ...)                                     \
    {                                                                          \
        dim3 block(32, 4);                                                     \
        dim3 grid(((NSH) + 31) / 32, ((NLOC) + 3) / 4);                        \
        KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami born_em_el " #KERN " failed"); \
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
    int64_t cq_batched, int64_t dcq_batched, int64_t store)
{
    const int n_shots = ey.size(0), ny = ey.size(1), nx = ey.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(ey.scalar_type(), "born_em_el_born_step_h", [&] {
        LAUNCH_GRID(born_step_h_kernel, scalar_t,
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
            (int)fd_pad_y0, (int)fd_pad_x0,
            (int)cq_batched, (int)dcq_batched, (int)store);
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
    int64_t dca_batched, int64_t dcb_batched)
{
    const int n_shots = ey.size(0), ny = ey.size(1), nx = ey.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(ey.scalar_type(), "born_em_el_born_step_e", [&] {
        LAUNCH_GRID(born_step_e_kernel, scalar_t,
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
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1,
            (int)ca_batched, (int)cb_batched,
            (int)dca_batched, (int)dcb_batched);
    });
}

void born_inject(
    torch::Tensor ey, torch::Tensor dEy,
    torch::Tensor f_bg, torch::Tensor f_sc, torch::Tensor src_i,
    int64_t t, int64_t n_shots, int64_t n_src, int64_t ny_nx)
{
    AT_DISPATCH_FLOATING_TYPES(ey.scalar_type(), "born_em_el_born_inject", [&] {
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
    AT_DISPATCH_FLOATING_TYPES(dEy.scalar_type(), "born_em_el_born_record", [&] {
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
    AT_DISPATCH_FLOATING_TYPES(lam_dEy.scalar_type(), "born_em_el_born_record_grad_r", [&] {
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
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "born_em_el_born_record_grad_f", [&] {
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
    int64_t t, int64_t interval, double scale,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1)
{
    const int n_shots = lam_ey.size(0), ny = lam_ey.size(1), nx = lam_ey.size(2);
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "born_em_el_born_coeff_grad", [&] {
        LAUNCH_GRID(born_coeff_grad_kernel, scalar_t,
            lam_ey.data_ptr<scalar_t>(), lam_dEy.data_ptr<scalar_t>(),
            ey_store.data_ptr<scalar_t>(), curl_store.data_ptr<scalar_t>(),
            dEy_store.data_ptr<scalar_t>(), dcurl_store.data_ptr<scalar_t>(),
            grad_ca.data_ptr<scalar_t>(), grad_cb.data_ptr<scalar_t>(),
            grad_dca.data_ptr<scalar_t>(), grad_dcb.data_ptr<scalar_t>(),
            (int)t, (int)interval, (scalar_t)scale, n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1);
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
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "born_em_el_born_adjoint_e_stage1", [&] {
        LAUNCH_GRID(born_adjoint_e_stage1_kernel, scalar_t,
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
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1,
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
    AT_DISPATCH_FLOATING_TYPES(lam_hx.scalar_type(), "born_em_el_born_adjoint_e_stage2", [&] {
        LAUNCH_GRID(born_adjoint_e_stage2_kernel, scalar_t,
            work_x.data_ptr<scalar_t>(), work_y.data_ptr<scalar_t>(),
            work_dx.data_ptr<scalar_t>(), work_dy.data_ptr<scalar_t>(),
            lam_hx.data_ptr<scalar_t>(), lam_hz.data_ptr<scalar_t>(),
            lam_dhx.data_ptr<scalar_t>(), lam_dhz.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx,
            n_shots, ny, nx, (int)fd_pad_y0, (int)fd_pad_x0);
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
    AT_DISPATCH_FLOATING_TYPES(lam_hx.scalar_type(), "born_em_el_born_adjoint_h_stage1", [&] {
        LAUNCH_GRID(born_adjoint_h_stage1_kernel, scalar_t,
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
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1,
            (int)cq_batched, (int)dcq_batched);
    });
}

void born_cq_grad(
    torch::Tensor lam_hx, torch::Tensor lam_hz,
    torch::Tensor lam_dhx, torch::Tensor lam_dhz,
    torch::Tensor dey_dy_store, torch::Tensor dey_dx_store,
    torch::Tensor ddey_dy_store, torch::Tensor ddey_dx_store,
    torch::Tensor grad_cq, torch::Tensor grad_dcq,
    int64_t t, int64_t interval, double scale,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1)
{
    const int n_shots = lam_hx.size(0), ny = lam_hx.size(1), nx = lam_hx.size(2);
    AT_DISPATCH_FLOATING_TYPES(lam_hx.scalar_type(), "born_em_el_born_cq_grad", [&] {
        LAUNCH_GRID(born_cq_grad_kernel, scalar_t,
            lam_hx.data_ptr<scalar_t>(), lam_hz.data_ptr<scalar_t>(),
            lam_dhx.data_ptr<scalar_t>(), lam_dhz.data_ptr<scalar_t>(),
            dey_dy_store.data_ptr<scalar_t>(), dey_dx_store.data_ptr<scalar_t>(),
            ddey_dy_store.data_ptr<scalar_t>(), ddey_dx_store.data_ptr<scalar_t>(),
            grad_cq.data_ptr<scalar_t>(), grad_dcq.data_ptr<scalar_t>(),
            (int)t, (int)interval, (scalar_t)scale, n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1);
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
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "born_em_el_born_adjoint_h_stage2", [&] {
        LAUNCH_GRID(born_adjoint_h_stage2_kernel, scalar_t,
            work2_x.data_ptr<scalar_t>(), work2_y.data_ptr<scalar_t>(),
            work2_dx.data_ptr<scalar_t>(), work2_dy.data_ptr<scalar_t>(),
            lam_ey.data_ptr<scalar_t>(), lam_dEy.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx,
            n_shots, ny, nx, (int)fd_pad_y0, (int)fd_pad_x0);
    });
}

// ---------------- elastic host wrappers ----------------
void born_step_velocity(
    torch::Tensor vy, torch::Tensor vx,
    torch::Tensor syy, torch::Tensor sxx, torch::Tensor sxy,
    torch::Tensor dvy, torch::Tensor dvx,
    torch::Tensor dsyy, torch::Tensor dsxx, torch::Tensor dsxy,
    torch::Tensor m_sigmayyy, torch::Tensor m_sigmaxyx,
    torch::Tensor m_sigmaxyy, torch::Tensor m_sigmaxxx,
    torch::Tensor dm_sigmayyy, torch::Tensor dm_sigmaxyx,
    torch::Tensor dm_sigmaxyy, torch::Tensor dm_sigmaxxx,
    torch::Tensor buoyancy_y, torch::Tensor buoyancy_x,
    torch::Tensor dbuoyancy_y, torch::Tensor dbuoyancy_x,
    torch::Tensor dvydb_store, torch::Tensor dvxdb_store,
    torch::Tensor ddvydb_store, torch::Tensor ddvxdb_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv,
    int64_t t, int64_t interval,
    int64_t model_batched, int64_t scatter_batched, int64_t store)
{
    const int n_shots = vy.size(0), ny = vy.size(1), nx = vy.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(vy.scalar_type(), "born_em_el_born_step_velocity", [&] {
        LAUNCH_GRID(born_step_velocity_kernel, scalar_t,
            vy.data_ptr<scalar_t>(), vx.data_ptr<scalar_t>(),
            syy.data_ptr<scalar_t>(), sxx.data_ptr<scalar_t>(), sxy.data_ptr<scalar_t>(),
            dvy.data_ptr<scalar_t>(), dvx.data_ptr<scalar_t>(),
            dsyy.data_ptr<scalar_t>(), dsxx.data_ptr<scalar_t>(), dsxy.data_ptr<scalar_t>(),
            m_sigmayyy.data_ptr<scalar_t>(), m_sigmaxyx.data_ptr<scalar_t>(),
            m_sigmaxyy.data_ptr<scalar_t>(), m_sigmaxxx.data_ptr<scalar_t>(),
            dm_sigmayyy.data_ptr<scalar_t>(), dm_sigmaxyx.data_ptr<scalar_t>(),
            dm_sigmaxyy.data_ptr<scalar_t>(), dm_sigmaxxx.data_ptr<scalar_t>(),
            buoyancy_y.data_ptr<scalar_t>(), buoyancy_x.data_ptr<scalar_t>(),
            dbuoyancy_y.data_ptr<scalar_t>(), dbuoyancy_x.data_ptr<scalar_t>(),
            dvydb_store.data_ptr<scalar_t>(), dvxdb_store.data_ptr<scalar_t>(),
            ddvydb_store.data_ptr<scalar_t>(), ddvxdb_store.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), byh.data_ptr<scalar_t>(),
            ay.data_ptr<scalar_t>(), by.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), bxh.data_ptr<scalar_t>(),
            ax.data_ptr<scalar_t>(), bx.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1,
            (scalar_t)rdy, (scalar_t)rdx, (scalar_t)dtv,
            (int)t, (int)interval, n_shots, ny, nx,
            (int)model_batched, (int)scatter_batched, (int)store);
    });
}

void born_step_stress(
    torch::Tensor vy, torch::Tensor vx,
    torch::Tensor dvy, torch::Tensor dvx,
    torch::Tensor syy, torch::Tensor sxx, torch::Tensor sxy,
    torch::Tensor dsyy, torch::Tensor dsxx, torch::Tensor dsxy,
    torch::Tensor m_vyy, torch::Tensor m_vxx,
    torch::Tensor m_vxy, torch::Tensor m_vyx,
    torch::Tensor dm_vyy, torch::Tensor dm_vxx,
    torch::Tensor dm_vxy, torch::Tensor dm_vyx,
    torch::Tensor lamb, torch::Tensor mu, torch::Tensor mu_yx,
    torch::Tensor dlamb, torch::Tensor dmu, torch::Tensor dmu_yx,
    torch::Tensor dvydy_store, torch::Tensor dvxdx_store, torch::Tensor dvxy_store,
    torch::Tensor ddvydy_store, torch::Tensor ddvxdx_store, torch::Tensor ddvxy_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv,
    int64_t t, int64_t interval,
    int64_t model_batched, int64_t scatter_batched, int64_t store)
{
    const int n_shots = vy.size(0), ny = vy.size(1), nx = vy.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(vy.scalar_type(), "born_em_el_born_step_stress", [&] {
        LAUNCH_GRID(born_step_stress_kernel, scalar_t,
            vy.data_ptr<scalar_t>(), vx.data_ptr<scalar_t>(),
            dvy.data_ptr<scalar_t>(), dvx.data_ptr<scalar_t>(),
            syy.data_ptr<scalar_t>(), sxx.data_ptr<scalar_t>(), sxy.data_ptr<scalar_t>(),
            dsyy.data_ptr<scalar_t>(), dsxx.data_ptr<scalar_t>(), dsxy.data_ptr<scalar_t>(),
            m_vyy.data_ptr<scalar_t>(), m_vxx.data_ptr<scalar_t>(),
            m_vxy.data_ptr<scalar_t>(), m_vyx.data_ptr<scalar_t>(),
            dm_vyy.data_ptr<scalar_t>(), dm_vxx.data_ptr<scalar_t>(),
            dm_vxy.data_ptr<scalar_t>(), dm_vyx.data_ptr<scalar_t>(),
            lamb.data_ptr<scalar_t>(), mu.data_ptr<scalar_t>(), mu_yx.data_ptr<scalar_t>(),
            dlamb.data_ptr<scalar_t>(), dmu.data_ptr<scalar_t>(), dmu_yx.data_ptr<scalar_t>(),
            dvydy_store.data_ptr<scalar_t>(), dvxdx_store.data_ptr<scalar_t>(),
            dvxy_store.data_ptr<scalar_t>(),
            ddvydy_store.data_ptr<scalar_t>(), ddvxdx_store.data_ptr<scalar_t>(),
            ddvxy_store.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), byh.data_ptr<scalar_t>(),
            ay.data_ptr<scalar_t>(), by.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), bxh.data_ptr<scalar_t>(),
            ax.data_ptr<scalar_t>(), bx.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1,
            (scalar_t)rdy, (scalar_t)rdx, (scalar_t)dtv,
            (int)t, (int)interval, n_shots, ny, nx,
            (int)model_batched, (int)scatter_batched, (int)store);
    });
}

void born_inject_pressure(
    torch::Tensor syy, torch::Tensor sxx, torch::Tensor f, torch::Tensor src_i,
    int64_t t, int64_t n_shots, int64_t n_src, int64_t ny_nx)
{
    AT_DISPATCH_FLOATING_TYPES(syy.scalar_type(), "born_em_el_born_inject_pressure", [&] {
        LAUNCH_PT(born_inject_pressure_kernel, scalar_t, (int)n_shots, (int)n_src,
            syy.data_ptr<scalar_t>(), sxx.data_ptr<scalar_t>(),
            f.data_ptr<scalar_t>(), src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)ny_nx);
    });
}

void born_record_pressure(
    torch::Tensor dsyy, torch::Tensor dsxx, torch::Tensor r, torch::Tensor rec_i,
    int64_t t, int64_t n_shots, int64_t n_rec, int64_t ny_nx)
{
    AT_DISPATCH_FLOATING_TYPES(dsyy.scalar_type(), "born_em_el_born_record_pressure", [&] {
        LAUNCH_PT(born_record_pressure_kernel, scalar_t, (int)n_shots, (int)n_rec,
            dsyy.data_ptr<scalar_t>(), dsxx.data_ptr<scalar_t>(),
            r.data_ptr<scalar_t>(), rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)ny_nx);
    });
}

void born_record_grad_f_p(
    torch::Tensor l_syy, torch::Tensor l_sxx, torch::Tensor grad_f, torch::Tensor src_i,
    int64_t t, int64_t n_shots, int64_t n_src, int64_t ny_nx)
{
    AT_DISPATCH_FLOATING_TYPES(l_syy.scalar_type(), "born_em_el_born_record_grad_f_p", [&] {
        LAUNCH_PT(born_record_grad_f_p_kernel, scalar_t, (int)n_shots, (int)n_src,
            l_syy.data_ptr<scalar_t>(), l_sxx.data_ptr<scalar_t>(),
            grad_f.data_ptr<scalar_t>(), src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)ny_nx);
    });
}

void born_add_grad_r(
    torch::Tensor l_dsyy, torch::Tensor l_dsxx, torch::Tensor grad_r, torch::Tensor rec_i,
    int64_t t, int64_t n_shots, int64_t n_rec, int64_t ny_nx)
{
    AT_DISPATCH_FLOATING_TYPES(l_dsyy.scalar_type(), "born_em_el_born_add_grad_r", [&] {
        LAUNCH_PT(born_add_grad_r_kernel, scalar_t, (int)n_shots, (int)n_rec,
            l_dsyy.data_ptr<scalar_t>(), l_dsxx.data_ptr<scalar_t>(),
            grad_r.data_ptr<scalar_t>(), rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)ny_nx);
    });
}

void born_adjoint_velocity(
    torch::Tensor lamb, torch::Tensor mu, torch::Tensor mu_yx,
    torch::Tensor dlamb, torch::Tensor dmu, torch::Tensor dmu_yx,
    torch::Tensor l_vy, torch::Tensor l_vx,
    torch::Tensor l_dvy, torch::Tensor l_dvx,
    torch::Tensor l_syy, torch::Tensor l_sxx, torch::Tensor l_sxy,
    torch::Tensor l_dsyy, torch::Tensor l_dsxx, torch::Tensor l_dsxy,
    torch::Tensor m_vyy, torch::Tensor m_vxx,
    torch::Tensor m_vxy, torch::Tensor m_vyx,
    torch::Tensor dm_vyy, torch::Tensor dm_vxx,
    torch::Tensor dm_vxy, torch::Tensor dm_vyx,
    torch::Tensor m_syyy_old, torch::Tensor m_syx_old,
    torch::Tensor m_syxy_old, torch::Tensor m_syxx_old,
    torch::Tensor m_syyy_new, torch::Tensor m_syx_new,
    torch::Tensor m_syxy_new, torch::Tensor m_syxx_new,
    torch::Tensor dm_syyy_old, torch::Tensor dm_syx_old,
    torch::Tensor dm_syxy_old, torch::Tensor dm_syxx_old,
    torch::Tensor dm_syyy_new, torch::Tensor dm_syx_new,
    torch::Tensor dm_syxy_new, torch::Tensor dm_syxx_new,
    torch::Tensor buoyancy_y, torch::Tensor buoyancy_x,
    torch::Tensor dbuoyancy_y, torch::Tensor dbuoyancy_x,
    torch::Tensor grad_buoyancy_y, torch::Tensor grad_buoyancy_x,
    torch::Tensor grad_dbuoyancy_y, torch::Tensor grad_dbuoyancy_x,
    torch::Tensor dvydb_store, torch::Tensor dvxdb_store,
    torch::Tensor ddvydb_store, torch::Tensor ddvxdb_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv,
    int64_t t, int64_t interval,
    int64_t model_batched, int64_t scatter_batched)
{
    const int n_shots = l_vy.size(0), ny = l_vy.size(1), nx = l_vy.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(l_vy.scalar_type(), "born_em_el_born_adjoint_velocity", [&] {
        LAUNCH_GRID(born_adjoint_velocity_kernel, scalar_t,
            lamb.data_ptr<scalar_t>(), mu.data_ptr<scalar_t>(), mu_yx.data_ptr<scalar_t>(),
            dlamb.data_ptr<scalar_t>(), dmu.data_ptr<scalar_t>(), dmu_yx.data_ptr<scalar_t>(),
            l_vy.data_ptr<scalar_t>(), l_vx.data_ptr<scalar_t>(),
            l_dvy.data_ptr<scalar_t>(), l_dvx.data_ptr<scalar_t>(),
            l_syy.data_ptr<scalar_t>(), l_sxx.data_ptr<scalar_t>(), l_sxy.data_ptr<scalar_t>(),
            l_dsyy.data_ptr<scalar_t>(), l_dsxx.data_ptr<scalar_t>(), l_dsxy.data_ptr<scalar_t>(),
            m_vyy.data_ptr<scalar_t>(), m_vxx.data_ptr<scalar_t>(),
            m_vxy.data_ptr<scalar_t>(), m_vyx.data_ptr<scalar_t>(),
            dm_vyy.data_ptr<scalar_t>(), dm_vxx.data_ptr<scalar_t>(),
            dm_vxy.data_ptr<scalar_t>(), dm_vyx.data_ptr<scalar_t>(),
            m_syyy_old.data_ptr<scalar_t>(), m_syx_old.data_ptr<scalar_t>(),
            m_syxy_old.data_ptr<scalar_t>(), m_syxx_old.data_ptr<scalar_t>(),
            m_syyy_new.data_ptr<scalar_t>(), m_syx_new.data_ptr<scalar_t>(),
            m_syxy_new.data_ptr<scalar_t>(), m_syxx_new.data_ptr<scalar_t>(),
            dm_syyy_old.data_ptr<scalar_t>(), dm_syx_old.data_ptr<scalar_t>(),
            dm_syxy_old.data_ptr<scalar_t>(), dm_syxx_old.data_ptr<scalar_t>(),
            dm_syyy_new.data_ptr<scalar_t>(), dm_syx_new.data_ptr<scalar_t>(),
            dm_syxy_new.data_ptr<scalar_t>(), dm_syxx_new.data_ptr<scalar_t>(),
            buoyancy_y.data_ptr<scalar_t>(), buoyancy_x.data_ptr<scalar_t>(),
            dbuoyancy_y.data_ptr<scalar_t>(), dbuoyancy_x.data_ptr<scalar_t>(),
            grad_buoyancy_y.data_ptr<scalar_t>(), grad_buoyancy_x.data_ptr<scalar_t>(),
            grad_dbuoyancy_y.data_ptr<scalar_t>(), grad_dbuoyancy_x.data_ptr<scalar_t>(),
            dvydb_store.data_ptr<scalar_t>(), dvxdb_store.data_ptr<scalar_t>(),
            ddvydb_store.data_ptr<scalar_t>(), ddvxdb_store.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), byh.data_ptr<scalar_t>(),
            ay.data_ptr<scalar_t>(), by.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), bxh.data_ptr<scalar_t>(),
            ax.data_ptr<scalar_t>(), bx.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1,
            (scalar_t)rdy, (scalar_t)rdx, (scalar_t)dtv,
            (int)t, (int)interval, n_shots, ny, nx,
            (int)model_batched, (int)scatter_batched);
    });
}

void born_adjoint_stress(
    torch::Tensor buoyancy_y, torch::Tensor buoyancy_x,
    torch::Tensor dbuoyancy_y, torch::Tensor dbuoyancy_x,
    torch::Tensor l_vy, torch::Tensor l_vx,
    torch::Tensor l_dvy, torch::Tensor l_dvx,
    torch::Tensor l_syy, torch::Tensor l_sxx, torch::Tensor l_sxy,
    torch::Tensor l_dsyy, torch::Tensor l_dsxx, torch::Tensor l_dsxy,
    torch::Tensor m_vyy, torch::Tensor m_vxx,
    torch::Tensor m_vxy, torch::Tensor m_vyx,
    torch::Tensor dm_vyy, torch::Tensor dm_vxx,
    torch::Tensor dm_vxy, torch::Tensor dm_vyx,
    torch::Tensor lamb, torch::Tensor mu, torch::Tensor mu_yx,
    torch::Tensor dlamb, torch::Tensor dmu, torch::Tensor dmu_yx,
    torch::Tensor m_syyy_old, torch::Tensor m_syx_old,
    torch::Tensor m_syxy_old, torch::Tensor m_syxx_old,
    torch::Tensor dm_syyy_old, torch::Tensor dm_syx_old,
    torch::Tensor dm_syxy_old, torch::Tensor dm_syxx_old,
    torch::Tensor grad_lamb, torch::Tensor grad_mu, torch::Tensor grad_mu_yx,
    torch::Tensor grad_dlamb, torch::Tensor grad_dmu, torch::Tensor grad_dmu_yx,
    torch::Tensor dvydy_store, torch::Tensor dvxdx_store, torch::Tensor dvxy_store,
    torch::Tensor ddvydy_store, torch::Tensor ddvxdx_store, torch::Tensor ddvxy_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv,
    int64_t t, int64_t interval,
    int64_t model_batched, int64_t scatter_batched)
{
    const int n_shots = l_vy.size(0), ny = l_vy.size(1), nx = l_vy.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(l_vy.scalar_type(), "born_em_el_born_adjoint_stress", [&] {
        LAUNCH_GRID(born_adjoint_stress_kernel, scalar_t,
            buoyancy_y.data_ptr<scalar_t>(), buoyancy_x.data_ptr<scalar_t>(),
            dbuoyancy_y.data_ptr<scalar_t>(), dbuoyancy_x.data_ptr<scalar_t>(),
            l_vy.data_ptr<scalar_t>(), l_vx.data_ptr<scalar_t>(),
            l_dvy.data_ptr<scalar_t>(), l_dvx.data_ptr<scalar_t>(),
            l_syy.data_ptr<scalar_t>(), l_sxx.data_ptr<scalar_t>(), l_sxy.data_ptr<scalar_t>(),
            l_dsyy.data_ptr<scalar_t>(), l_dsxx.data_ptr<scalar_t>(), l_dsxy.data_ptr<scalar_t>(),
            m_vyy.data_ptr<scalar_t>(), m_vxx.data_ptr<scalar_t>(),
            m_vxy.data_ptr<scalar_t>(), m_vyx.data_ptr<scalar_t>(),
            dm_vyy.data_ptr<scalar_t>(), dm_vxx.data_ptr<scalar_t>(),
            dm_vxy.data_ptr<scalar_t>(), dm_vyx.data_ptr<scalar_t>(),
            lamb.data_ptr<scalar_t>(), mu.data_ptr<scalar_t>(), mu_yx.data_ptr<scalar_t>(),
            dlamb.data_ptr<scalar_t>(), dmu.data_ptr<scalar_t>(), dmu_yx.data_ptr<scalar_t>(),
            m_syyy_old.data_ptr<scalar_t>(), m_syx_old.data_ptr<scalar_t>(),
            m_syxy_old.data_ptr<scalar_t>(), m_syxx_old.data_ptr<scalar_t>(),
            dm_syyy_old.data_ptr<scalar_t>(), dm_syx_old.data_ptr<scalar_t>(),
            dm_syxy_old.data_ptr<scalar_t>(), dm_syxx_old.data_ptr<scalar_t>(),
            grad_lamb.data_ptr<scalar_t>(), grad_mu.data_ptr<scalar_t>(),
            grad_mu_yx.data_ptr<scalar_t>(),
            grad_dlamb.data_ptr<scalar_t>(), grad_dmu.data_ptr<scalar_t>(),
            grad_dmu_yx.data_ptr<scalar_t>(),
            dvydy_store.data_ptr<scalar_t>(), dvxdx_store.data_ptr<scalar_t>(),
            dvxy_store.data_ptr<scalar_t>(),
            ddvydy_store.data_ptr<scalar_t>(), ddvxdx_store.data_ptr<scalar_t>(),
            ddvxy_store.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), byh.data_ptr<scalar_t>(),
            ay.data_ptr<scalar_t>(), by.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), bxh.data_ptr<scalar_t>(),
            ax.data_ptr<scalar_t>(), bx.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1,
            (scalar_t)rdy, (scalar_t)rdx, (scalar_t)dtv,
            (int)t, (int)interval, n_shots, ny, nx,
            (int)model_batched, (int)scatter_batched);
    });
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
    m.def("born_step_velocity", &born_step_velocity);
    m.def("born_step_stress", &born_step_stress);
    m.def("born_inject_pressure", &born_inject_pressure);
    m.def("born_record_pressure", &born_record_pressure);
    m.def("born_record_grad_f_p", &born_record_grad_f_p);
    m.def("born_add_grad_r", &born_add_grad_r);
    NAMI_STORAGE_PYBIND(m);
    m.def("born_adjoint_velocity", &born_adjoint_velocity);
    m.def("born_adjoint_stress", &born_adjoint_stress);
}
