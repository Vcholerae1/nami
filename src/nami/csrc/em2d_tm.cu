
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "storage.h"

#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------- coefficient-driven staggered difference operators ----------------
// ``c`` is the staggered-grid first-derivative coefficient array (max radius
// 4, zero-padded for lower orders); the loop bound is the stencil radius
// (== accuracy // 2 == fd_pad_y0), so no out-of-bounds reads occur.
//
// diff_int_*: derivative at a half-integer grid point of a field stored at
// integer grid points (H step): sum_k c[k-1] * (u[i+k] - u[i-k+1]).
// diff_half_*: derivative at an integer grid point of a field stored at
// half-integer grid points (E step curl): sum_k c[k-1] * (u[i+k-1] - u[i-k]).
// The two operators are exact discrete transposes of each other (up to the
// sign convention of the curl), which is what the adjoint kernels rely on.
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

// ---------------- forward: H half-step ----------------
// Hx/Hz are updated from Ey with integer-point staggered differences
// (Ey lives at integer grid points, Hx/Hz at half-integer points); CPML
// memory variables m_ey_z / m_ey_x use the half-integer profiles (ayh, byh,
// kyh, axh, bxh, kxh).  Boundary rows/cols get a zero gradient, the memory
// recursion still runs (it stays zero there).
template <typename T>
__global__ void step_h_kernel(
    const T* __restrict__ cq,
    const T* __restrict__ ey,
    T* __restrict__ hx, T* __restrict__ hz,
    T* __restrict__ m_ey_z, T* __restrict__ m_ey_x,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ kyh, const T* __restrict__ kxh,
    const T* __restrict__ c,
    T rdy, T rdx,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int fd_pad_y0, int fd_pad_x0,
    int cq_batched)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= ny || x >= nx)
        return;
    const int s_cq = cq_batched ? s : 0;
    const T cq_val = cq[(long)s_cq * ny * nx + (long)y * nx + x];
    const long off = ((long)s * ny + y) * nx + x;
    // strictly-interior cells have zero C-PML profiles, so the memory
    // branches never fire: keep them on the identical memory-free path
    const bool interior = (y >= pml_y0 && y < pml_y1 - 1 &&
                           x >= pml_x0 && x < pml_x1 - 1);
    T dey_dy = (T)0;
    if (y >= fd_pad_y0 && y < ny - fd_pad_y0)
        dey_dy = diff_int_y(ey, off, c, rdy, nx, fd_pad_y0);
    if (interior) {
        hx[off] -= cq_val * dey_dy;
    } else {
        if (y < pml_y0 || y >= pml_y1 - 1) {
            m_ey_z[off] = byh[y] * m_ey_z[off] + ayh[y] * dey_dy;
            dey_dy = dey_dy / kyh[y] + m_ey_z[off];
        }
        hx[off] -= cq_val * dey_dy;
    }
    T dey_dx = (T)0;
    if (x >= fd_pad_x0 && x < nx - fd_pad_x0)
        dey_dx = diff_int_x(ey, off, c, rdx, fd_pad_x0);
    if (interior) {
        hz[off] += cq_val * dey_dx;
    } else {
        if (x < pml_x0 || x >= pml_x1 - 1) {
            m_ey_x[off] = bxh[x] * m_ey_x[off] + axh[x] * dey_dx;
            dey_dx = dey_dx / kxh[x] + m_ey_x[off];
        }
        hz[off] += cq_val * dey_dx;
    }
}

// ---------------- forward: E integer-step with snapshots ----------------
// Curls of Hx/Hz (half-integer-stored fields differentiated back to the
// integer Ey points) with CPML memory (m_hx_z / m_hz_x, integer profiles ay,
// by, ky, ax, bx, kx); snapshots the pre-update Ey and the PML-modified curl
// on the sampling interval (used by coeff_grad / cq_grad), then
// Ey = ca*Ey + cb*curl.
template <typename T>
__global__ void step_e_storage_kernel(
    const T* __restrict__ ca, const T* __restrict__ cb,
    const T* __restrict__ hx, const T* __restrict__ hz,
    T* __restrict__ ey,
    T* __restrict__ m_hx_z, T* __restrict__ m_hz_x,
    T* __restrict__ ey_store, T* __restrict__ curl_store,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ ky, const T* __restrict__ kx,
    const T* __restrict__ c,
    T rdy, T rdx,
    int t, int interval, int64_t snap_off,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int fd_pad_y0, int fd_pad_y1, int fd_pad_x0, int fd_pad_x1,
    int ca_batched, int cb_batched)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= ny || x >= nx)
        return;
    const long off = ((long)s * ny + y) * nx + x;
    if (y >= fd_pad_y0 && x >= fd_pad_x0 && y < ny - fd_pad_y1 &&
        x < nx - fd_pad_x1) {
        // strictly-interior cells have zero C-PML profiles, so the memory
        // branches never fire: keep them on the identical memory-free path
        const bool interior = (y >= pml_y0 && y < pml_y1 &&
                               x >= pml_x0 && x < pml_x1);
        T dhz_dx = diff_half_x(hz, off, c, rdx, fd_pad_x0);
        if (!interior && (x < pml_x0 || x >= pml_x1)) {
            m_hz_x[off] = bx[x] * m_hz_x[off] + ax[x] * dhz_dx;
            dhz_dx = dhz_dx / kx[x] + m_hz_x[off];
        }
        T dhx_dy = diff_half_y(hx, off, c, rdy, nx, fd_pad_y0);
        if (!interior && (y < pml_y0 || y >= pml_y1)) {
            m_hx_z[off] = by[y] * m_hx_z[off] + ay[y] * dhx_dy;
            dhx_dy = dhx_dy / ky[y] + m_hx_z[off];
        }
        const T curl = dhz_dx - dhx_dy;
        if (t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            ey_store[soff] = ey[off];
            curl_store[soff] = curl;
        }
        const int s_ca = ca_batched ? s : 0;
        const int s_cb = cb_batched ? s : 0;
        ey[off] = ca[(long)s_ca * ny * nx + (long)y * nx + x] * ey[off]
                + cb[(long)s_cb * ny * nx + (long)y * nx + x] * curl;
    }
}

// ---------------- forward: source injection / receiver recording ----------------
// (flat indices are precomputed on the padded grid, so the kernels only need
//  the per-shot field stride ny*nx and the row-major flat index.)
template <typename T>
__global__ void inject_kernel(
    T* __restrict__ ey, const T* __restrict__ f, const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        if (idx >= 0)
            ey[(long)s * ny_nx + idx] += f[(((long)t * n_shots + s) * n_src + k)];
    }
}

template <typename T>
__global__ void record_kernel(
    const T* __restrict__ ey, T* __restrict__ r, const long* __restrict__ rec_i,
    int t, int n_shots, int n_rec, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            r[(((long)t * n_shots + s) * n_rec + k)] = ey[(long)s * ny_nx + idx];
    }
}

// ---------------- backward: source/receiver gradient exchange ----------------
template <typename T>
__global__ void record_grad_r_kernel(
    T* __restrict__ lam_ey, const T* __restrict__ grad_r, const long* __restrict__ rec_i,
    int t, int n_shots, int n_rec, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            lam_ey[(long)s * ny_nx + idx] += grad_r[(((long)t * n_shots + s) * n_rec + k)];
    }
}

template <typename T>
__global__ void record_grad_f_kernel(
    const T* __restrict__ lam_ey, T* __restrict__ grad_f, const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        grad_f[(((long)t * n_shots + s) * n_src + k)] =
            (idx >= 0) ? lam_ey[(long)s * ny_nx + idx] : (T)0;
    }
}

// ---------------- backward: ca/cb model gradients ----------------
template <typename T>
__global__ void coeff_grad_kernel(
    const T* __restrict__ lam_ey,
    const T* __restrict__ ey_store, const T* __restrict__ curl_store,
    T* __restrict__ grad_ca, T* __restrict__ grad_cb,
    int t, int interval, T scale, int64_t snap_off,
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
        const long soff = snap_off + ((long)s * ny + y) * nx + x;
        const T lam = lam_ey[off] * scale;
        grad_ca[off] += lam * ey_store[soff];
        grad_cb[off] += lam * curl_store[soff];
    }
}

// ---------------- backward: transpose of the E update, stage 1 ----------------
// lambda_ey *= ca and builds work_x / work_y (the PML-modified curl applied
// to cb*lambda_ey) with the time-reversed memory recursions; outside the
// interior work is zeroed for the stage-2 divergence.
template <typename T>
__global__ void adjoint_e_stage1_kernel(
    const T* __restrict__ ca, const T* __restrict__ cb,
    T* __restrict__ lam_ey,
    T* __restrict__ m_lambda_hx_z, T* __restrict__ m_lambda_hz_x,
    T* __restrict__ work_x, T* __restrict__ work_y,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ ky, const T* __restrict__ kx,
    const T* __restrict__ ey_store, const T* __restrict__ curl_store,
    T* __restrict__ grad_ca, T* __restrict__ grad_cb,
    int t, int interval, T scale, int64_t snap_off,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int fd_pad_y0, int fd_pad_y1, int fd_pad_x0, int fd_pad_x1,
    int ca_batched, int cb_batched)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= ny || x >= nx)
        return;
    const long off = ((long)s * ny + y) * nx + x;
    const bool interior = (y >= pml_y0 && y < pml_y1 &&
                           x >= pml_x0 && x < pml_x1);
    if (y >= fd_pad_y0 && x >= fd_pad_x0 && y < ny - fd_pad_y1 &&
        x < nx - fd_pad_x1) {
        const int s_ca = ca_batched ? s : 0;
        const int s_cb = cb_batched ? s : 0;
        const T ca_val = ca[(long)s_ca * ny * nx + (long)y * nx + x];
        const T cb_val = cb[(long)s_cb * ny * nx + (long)y * nx + x];
        // fused model-gradient accumulation (adjoint of the E update): uses
        // the pre-update lambda value, which only this thread modifies, so
        // the accumulation is race-free without a separate kernel launch
        if (t % interval == 0) {
            const T lam = lam_ey[off] * scale;
            grad_ca[off] += lam * ey_store[snap_off + off];
            grad_cb[off] += lam * curl_store[snap_off + off];
        }
        const T g = cb_val * lam_ey[off];
        lam_ey[off] = ca_val * lam_ey[off];
        if (interior) {
            work_x[off] = g;
            const T gy = -g;
            work_y[off] = gy;
        } else {
            if (x < pml_x0 || x >= pml_x1) {
                const T w = g + bx[x] * m_lambda_hz_x[off];
                work_x[off] = g / kx[x] + ax[x] * w;
                m_lambda_hz_x[off] = w;
            } else {
                work_x[off] = g;
            }
            const T gy = -g;
            if (y < pml_y0 || y >= pml_y1) {
                const T w = gy + by[y] * m_lambda_hx_z[off];
                work_y[off] = gy / ky[y] + ay[y] * w;
                m_lambda_hx_z[off] = w;
            } else {
                work_y[off] = gy;
            }
        }
    } else if (!interior) {
        work_x[off] = (T)0;
        work_y[off] = (T)0;
    }
}

// ---------------- backward: transpose of the E update, stage 2 ----------------
// lambda_hz += d(work_x)/dx, lambda_hx += d(work_y)/dy (the transpose of the
// curl divergence in the forward E step).  No interior guard: boundary rows
// get zero contributions from the zeroed work arrays.
template <typename T>
__global__ void adjoint_e_stage2_kernel(
    const T* __restrict__ work_x, const T* __restrict__ work_y,
    T* __restrict__ lam_hx, T* __restrict__ lam_hz,
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
        for (int k = 1; k <= fd_pad_x0; ++k) {
            const T ck = c[k - 1];
            if (x - k + 1 >= 0)
                acc_x += ck * work_x[off - k + 1];
            if (x + k < nx)
                acc_x -= ck * work_x[off + k];
        }
        lam_hz[off] += acc_x * rdx;
        T acc_y = (T)0;
        for (int k = 1; k <= fd_pad_y0; ++k) {
            const T ck = c[k - 1];
            if (y - k + 1 >= 0)
                acc_y += ck * work_y[off - (k - 1) * nx];
            if (y + k < ny)
                acc_y -= ck * work_y[off + k * nx];
        }
        lam_hx[off] += acc_y * rdy;
    }
}

// ---------------- backward: transpose of the H half-step, stage 1 ----------------
// Builds work2_x / work2_y from lambda_hz / lambda_hx through cq and the
// time-reversed half-integer PML memory; the H-update at the outermost row
// (y = ny - 1) / col (x = nx - 1) never feeds the output, so the adjoint
// zeroes those cells.
template <typename T>
__global__ void adjoint_h_stage1_kernel(
    const T* __restrict__ cq,
    const T* __restrict__ lam_hx, const T* __restrict__ lam_hz,
    T* __restrict__ m_lambda_ey_x, T* __restrict__ m_lambda_ey_z,
    T* __restrict__ work2_x, T* __restrict__ work2_y,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ kyh, const T* __restrict__ kxh,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int fd_pad_y0, int fd_pad_y1, int fd_pad_x0, int fd_pad_x1,
    int cq_batched)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= ny || x >= nx)
        return;
    const long off = ((long)s * ny + y) * nx + x;
    const int pml_y1h = (pml_y1 > pml_y0) ? pml_y1 - 1 : pml_y0;
    const int pml_x1h = (pml_x1 > pml_x0) ? pml_x1 - 1 : pml_x0;
    const bool interior = (y >= pml_y0 && y < pml_y1h &&
                           x >= pml_x0 && x < pml_x1h);
    if (y >= fd_pad_y0 && x >= fd_pad_x0 && y < ny - fd_pad_y1 &&
        x < nx - fd_pad_x1) {
        const int s_cq = cq_batched ? s : 0;
        const T cq_val = cq[(long)s_cq * ny * nx + (long)y * nx + x];
        if (interior) {
            if (x < nx - fd_pad_x0) {
                const T g2 = cq_val * lam_hz[off];
                work2_x[off] = g2;
            } else {
                work2_x[off] = (T)0;
            }
            if (y < ny - fd_pad_y0) {
                const T g2 = -cq_val * lam_hx[off];
                work2_y[off] = g2;
            } else {
                work2_y[off] = (T)0;
            }
        } else {
            if (x < nx - fd_pad_x0) {
                const T g2 = cq_val * lam_hz[off];
                if (x < pml_x0 || x >= pml_x1h) {
                    const T w = g2 + bxh[x] * m_lambda_ey_x[off];
                    work2_x[off] = g2 / kxh[x] + axh[x] * w;
                    m_lambda_ey_x[off] = w;
                } else {
                    work2_x[off] = g2;
                }
            } else {
                work2_x[off] = (T)0;
            }
            if (y < ny - fd_pad_y0) {
                const T g2 = -cq_val * lam_hx[off];
                if (y < pml_y0 || y >= pml_y1h) {
                    const T w = g2 + byh[y] * m_lambda_ey_z[off];
                    work2_y[off] = g2 / kyh[y] + ayh[y] * w;
                    m_lambda_ey_z[off] = w;
                } else {
                    work2_y[off] = g2;
                }
            } else {
                work2_y[off] = (T)0;
            }
        }
    } else if (!interior) {
        work2_x[off] = (T)0;
        work2_y[off] = (T)0;
    }
}

// ---------------- backward: cq model gradient ----------------
// Uses the sampled Ey differences and the stage-1 memory variables; must run
// after adjoint_e_stage2 (lambda_hx/hz complete) and adjoint_h_stage1
// (m_lambda_ey_x/z updated for this step).
template <typename T>
__global__ void cq_grad_kernel(
    const T* __restrict__ cq,
    const T* __restrict__ lam_hx, const T* __restrict__ lam_hz,
    const T* __restrict__ ey_store,
    const T* __restrict__ m_lambda_ey_x, const T* __restrict__ m_lambda_ey_z,
    T* __restrict__ grad_cq,
    const T* __restrict__ ayh, const T* __restrict__ kyh,
    const T* __restrict__ axh, const T* __restrict__ kxh,
    const T* __restrict__ c,
    T rdy, T rdx,
    int t, int interval, T scale, int64_t snap_off,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int fd_pad_y0, int fd_pad_y1, int fd_pad_x0, int fd_pad_x1,
    int cq_batched)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= fd_pad_y0 && x >= fd_pad_x0 && y < ny - fd_pad_y1 &&
        x < nx - fd_pad_x1) {
        const int s_cq = cq_batched ? s : 0;
        const long off = ((long)s * ny + y) * nx + x;
        const long soff = snap_off + ((long)s * ny + y) * nx + x;
        const T cq_val = cq[(long)s_cq * ny * nx + (long)y * nx + x];
        T term = (T)0;
        if (y < ny - fd_pad_y0) {
            const T dey_dy = diff_int_y(ey_store, soff, c, rdy, nx, fd_pad_y0);
            term += -lam_hx[off] * dey_dy / kyh[y]
                  + (ayh[y] / cq_val) * dey_dy * m_lambda_ey_z[off];
        }
        if (x < nx - fd_pad_x0) {
            const T dey_dx = diff_int_x(ey_store, soff, c, rdx, fd_pad_x0);
            term += lam_hz[off] * dey_dx / kxh[x]
                  + (axh[x] / cq_val) * dey_dx * m_lambda_ey_x[off];
        }
        grad_cq[off] += scale * term;
    }
}

// ---------------- backward: transpose of the H half-step, stage 2 ----------------
template <typename T>
__global__ void adjoint_h_stage2_kernel(
    const T* __restrict__ work2_x, const T* __restrict__ work2_y,
    T* __restrict__ lam_ey,
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
        for (int k = 1; k <= fd_pad_x0; ++k) {
            const T ck = c[k - 1];
            if (x >= k)
                sum_x += ck * work2_x[off - k];
            if (x + k - 1 < nx)
                sum_x -= ck * work2_x[off + k - 1];
        }
        T sum_y = (T)0;
        for (int k = 1; k <= fd_pad_y0; ++k) {
            const T ck = c[k - 1];
            if (y >= k)
                sum_y += ck * work2_y[off - k * nx];
            if (y + k - 1 < ny)
                sum_y -= ck * work2_y[off + (k - 1) * nx];
        }
        lam_ey[off] += sum_x * rdx + sum_y * rdy;
    }
}

// ---------------- launchers ----------------
#define LAUNCH_FIELD(KERN, T, ...)                                             \
    {                                                                          \
        dim3 block(16, 16);                                                    \
        dim3 grid((nx + 15) / 16, (ny + 15) / 16, n_shots);                    \
        KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami em2d_tm " #KERN " failed"); \
    }

#define LAUNCH_FIELD_RANGE(KERN, T, Y0, Y1, X0, X1, ...)                       \
    {                                                                          \
        const int _iy0 = (Y0), _iy1 = (Y1), _ix0 = (X0), _ix1 = (X1);          \
        if (_iy1 > _iy0 && _ix1 > _ix0) {                                      \
            dim3 block(16, 16);                                                \
            dim3 grid((_ix1 - _ix0 + 15) / 16, (_iy1 - _iy0 + 15) / 16, n_shots); \
            KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(     \
                __VA_ARGS__, _iy0, _iy1, _ix0, _ix1);                          \
            TORCH_CHECK(cudaGetLastError() == cudaSuccess,                     \
                        "nami em2d_tm " #KERN " failed");                      \
        }                                                                      \
    }

void step_h(
    torch::Tensor cq, torch::Tensor ey, torch::Tensor hx, torch::Tensor hz,
    torch::Tensor m_ey_z, torch::Tensor m_ey_x,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor axh, torch::Tensor bxh,
    torch::Tensor kyh, torch::Tensor kxh,
    torch::Tensor c,
    double rdy, double rdx,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t cq_batched)
{
    const int n_shots = ey.size(0), ny = ey.size(1), nx = ey.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(ey.scalar_type(), "em2d_tm_step_h", [&] {
        LAUNCH_FIELD(step_h_kernel, scalar_t,
            cq.data_ptr<scalar_t>(), ey.data_ptr<scalar_t>(),
            hx.data_ptr<scalar_t>(), hz.data_ptr<scalar_t>(),
            m_ey_z.data_ptr<scalar_t>(), m_ey_x.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), byh.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), bxh.data_ptr<scalar_t>(),
            kyh.data_ptr<scalar_t>(), kxh.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx,
            n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)fd_pad_y0, (int)fd_pad_x0,
            (int)cq_batched);
    });
}

void step_e_storage(
    torch::Tensor ca, torch::Tensor cb,
    torch::Tensor hx, torch::Tensor hz, torch::Tensor ey,
    torch::Tensor m_hx_z, torch::Tensor m_hz_x,
    torch::Tensor ey_store, torch::Tensor curl_store,
    torch::Tensor ay, torch::Tensor by, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor ky, torch::Tensor kx,
    torch::Tensor c,
    double rdy, double rdx,
    int64_t t, int64_t interval, int64_t snap_off,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t ca_batched, int64_t cb_batched)
{
    const int n_shots = ey.size(0), ny = ey.size(1), nx = ey.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(ey.scalar_type(), "em2d_tm_step_e_storage", [&] {
        LAUNCH_FIELD(step_e_storage_kernel, scalar_t,
            ca.data_ptr<scalar_t>(), cb.data_ptr<scalar_t>(),
            hx.data_ptr<scalar_t>(), hz.data_ptr<scalar_t>(),
            ey.data_ptr<scalar_t>(),
            m_hx_z.data_ptr<scalar_t>(), m_hz_x.data_ptr<scalar_t>(),
            ey_store.data_ptr<scalar_t>(), curl_store.data_ptr<scalar_t>(),
            ay.data_ptr<scalar_t>(), by.data_ptr<scalar_t>(),
            ax.data_ptr<scalar_t>(), bx.data_ptr<scalar_t>(),
            ky.data_ptr<scalar_t>(), kx.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx,
            (int)t, (int)interval, (int64_t)snap_off,
            n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1,
            (int)ca_batched, (int)cb_batched);
    });
}

void inject(torch::Tensor ey, torch::Tensor f, torch::Tensor src_i,
            int64_t t, int64_t n_shots, int64_t n_src, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_src + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(ey.scalar_type(), "em2d_tm_inject", [&] {
        inject_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            ey.data_ptr<scalar_t>(), f.data_ptr<scalar_t>(), src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami em2d_tm inject failed");
}

void record(torch::Tensor ey, torch::Tensor r, torch::Tensor rec_i,
            int64_t t, int64_t n_shots, int64_t n_rec, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_rec + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(ey.scalar_type(), "em2d_tm_record", [&] {
        record_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            ey.data_ptr<scalar_t>(), r.data_ptr<scalar_t>(), rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami em2d_tm record failed");
}

void record_grad_r(torch::Tensor lam_ey, torch::Tensor grad_r, torch::Tensor rec_i,
                   int64_t t, int64_t n_shots, int64_t n_rec, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_rec + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "em2d_tm_record_grad_r", [&] {
        record_grad_r_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            lam_ey.data_ptr<scalar_t>(), grad_r.data_ptr<scalar_t>(), rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami em2d_tm record_grad_r failed");
}

void record_grad_f(torch::Tensor lam_ey, torch::Tensor grad_f, torch::Tensor src_i,
                   int64_t t, int64_t n_shots, int64_t n_src, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_src + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "em2d_tm_record_grad_f", [&] {
        record_grad_f_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            lam_ey.data_ptr<scalar_t>(), grad_f.data_ptr<scalar_t>(), src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami em2d_tm record_grad_f failed");
}

void coeff_grad(
    torch::Tensor lam_ey, torch::Tensor ey_store, torch::Tensor curl_store,
    torch::Tensor grad_ca, torch::Tensor grad_cb,
    int64_t t, int64_t interval, double scale, int64_t snap_off,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1)
{
    const int n_shots = lam_ey.size(0), ny = lam_ey.size(1), nx = lam_ey.size(2);
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "em2d_tm_coeff_grad", [&] {
        LAUNCH_FIELD(coeff_grad_kernel, scalar_t,
            lam_ey.data_ptr<scalar_t>(),
            ey_store.data_ptr<scalar_t>(), curl_store.data_ptr<scalar_t>(),
            grad_ca.data_ptr<scalar_t>(), grad_cb.data_ptr<scalar_t>(),
            (int)t, (int)interval, (scalar_t)scale, (int64_t)snap_off,
            n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1);
    });
}

void adjoint_e_stage1(
    torch::Tensor ca, torch::Tensor cb, torch::Tensor lam_ey,
    torch::Tensor m_lambda_hx_z, torch::Tensor m_lambda_hz_x,
    torch::Tensor work_x, torch::Tensor work_y,
    torch::Tensor ay, torch::Tensor by, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor ky, torch::Tensor kx,
    torch::Tensor ey_store, torch::Tensor curl_store,
    torch::Tensor grad_ca, torch::Tensor grad_cb,
    int64_t t, int64_t interval, double scale, int64_t snap_off,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t ca_batched, int64_t cb_batched)
{
    const int n_shots = lam_ey.size(0), ny = lam_ey.size(1), nx = lam_ey.size(2);
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "em2d_tm_adjoint_e_stage1", [&] {
        LAUNCH_FIELD(adjoint_e_stage1_kernel, scalar_t,
            ca.data_ptr<scalar_t>(), cb.data_ptr<scalar_t>(),
            lam_ey.data_ptr<scalar_t>(),
            m_lambda_hx_z.data_ptr<scalar_t>(), m_lambda_hz_x.data_ptr<scalar_t>(),
            work_x.data_ptr<scalar_t>(), work_y.data_ptr<scalar_t>(),
            ay.data_ptr<scalar_t>(), by.data_ptr<scalar_t>(),
            ax.data_ptr<scalar_t>(), bx.data_ptr<scalar_t>(),
            ky.data_ptr<scalar_t>(), kx.data_ptr<scalar_t>(),
            ey_store.data_ptr<scalar_t>(), curl_store.data_ptr<scalar_t>(),
            grad_ca.data_ptr<scalar_t>(), grad_cb.data_ptr<scalar_t>(),
            (int)t, (int)interval, (scalar_t)scale, (int64_t)snap_off,
            n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1,
            (int)ca_batched, (int)cb_batched);
    });
}

void adjoint_e_stage2(
    torch::Tensor work_x, torch::Tensor work_y,
    torch::Tensor lam_hx, torch::Tensor lam_hz,
    torch::Tensor c,
    double rdy, double rdx,
    int64_t fd_pad_y0, int64_t fd_pad_x0)
{
    const int n_shots = lam_hx.size(0), ny = lam_hx.size(1), nx = lam_hx.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(lam_hx.scalar_type(), "em2d_tm_adjoint_e_stage2", [&] {
        LAUNCH_FIELD(adjoint_e_stage2_kernel, scalar_t,
            work_x.data_ptr<scalar_t>(), work_y.data_ptr<scalar_t>(),
            lam_hx.data_ptr<scalar_t>(), lam_hz.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx,
            n_shots, ny, nx,
            (int)fd_pad_y0, (int)fd_pad_x0);
    });
}

void adjoint_h_stage1(
    torch::Tensor cq,
    torch::Tensor lam_hx, torch::Tensor lam_hz,
    torch::Tensor m_lambda_ey_x, torch::Tensor m_lambda_ey_z,
    torch::Tensor work2_x, torch::Tensor work2_y,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor axh, torch::Tensor bxh,
    torch::Tensor kyh, torch::Tensor kxh,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t cq_batched)
{
    const int n_shots = lam_hx.size(0), ny = lam_hx.size(1), nx = lam_hx.size(2);
    AT_DISPATCH_FLOATING_TYPES(lam_hx.scalar_type(), "em2d_tm_adjoint_h_stage1", [&] {
        LAUNCH_FIELD(adjoint_h_stage1_kernel, scalar_t,
            cq.data_ptr<scalar_t>(),
            lam_hx.data_ptr<scalar_t>(), lam_hz.data_ptr<scalar_t>(),
            m_lambda_ey_x.data_ptr<scalar_t>(), m_lambda_ey_z.data_ptr<scalar_t>(),
            work2_x.data_ptr<scalar_t>(), work2_y.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), byh.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), bxh.data_ptr<scalar_t>(),
            kyh.data_ptr<scalar_t>(), kxh.data_ptr<scalar_t>(),
            n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1,
            (int)cq_batched);
    });
}

void cq_grad(
    torch::Tensor cq,
    torch::Tensor lam_hx, torch::Tensor lam_hz,
    torch::Tensor ey_store,
    torch::Tensor m_lambda_ey_x, torch::Tensor m_lambda_ey_z,
    torch::Tensor grad_cq,
    torch::Tensor ayh, torch::Tensor kyh, torch::Tensor axh, torch::Tensor kxh,
    torch::Tensor c,
    double rdy, double rdx,
    int64_t t, int64_t interval, double scale, int64_t snap_off,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t cq_batched)
{
    const int n_shots = lam_hx.size(0), ny = lam_hx.size(1), nx = lam_hx.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(lam_hx.scalar_type(), "em2d_tm_cq_grad", [&] {
        LAUNCH_FIELD(cq_grad_kernel, scalar_t,
            cq.data_ptr<scalar_t>(),
            lam_hx.data_ptr<scalar_t>(), lam_hz.data_ptr<scalar_t>(),
            ey_store.data_ptr<scalar_t>(),
            m_lambda_ey_x.data_ptr<scalar_t>(), m_lambda_ey_z.data_ptr<scalar_t>(),
            grad_cq.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), kyh.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), kxh.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx,
            (int)t, (int)interval, (scalar_t)scale, (int64_t)snap_off,
            n_shots, ny, nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)fd_pad_y0, (int)fd_pad_y1, (int)fd_pad_x0, (int)fd_pad_x1,
            (int)cq_batched);
    });
}

void adjoint_h_stage2(
    torch::Tensor work2_x, torch::Tensor work2_y,
    torch::Tensor lam_ey,
    torch::Tensor c,
    double rdy, double rdx,
    int64_t fd_pad_y0, int64_t fd_pad_x0)
{
    const int n_shots = lam_ey.size(0), ny = lam_ey.size(1), nx = lam_ey.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "em2d_tm_adjoint_h_stage2", [&] {
        LAUNCH_FIELD(adjoint_h_stage2_kernel, scalar_t,
            work2_x.data_ptr<scalar_t>(), work2_y.data_ptr<scalar_t>(),
            lam_ey.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx,
            n_shots, ny, nx,
            (int)fd_pad_y0, (int)fd_pad_x0);
    });
}
// ---------------- fused per-step entry points ----------------
// One pybind call per time step instead of 4-8: the per-call interpreter
// overhead is the dominant cost for the short-grid benchmark runs.
void forward_step(
    torch::Tensor cq, torch::Tensor ey, torch::Tensor hx, torch::Tensor hz,
    torch::Tensor m_ey_z, torch::Tensor m_ey_x,
    torch::Tensor m_hx_z, torch::Tensor m_hz_x,
    torch::Tensor ca, torch::Tensor cb,
    torch::Tensor ey_store, torch::Tensor curl_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor axh, torch::Tensor bxh,
    torch::Tensor kyh, torch::Tensor kxh,
    torch::Tensor ay, torch::Tensor by, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor ky, torch::Tensor kx,
    torch::Tensor c,
    torch::Tensor f, torch::Tensor src_i, torch::Tensor r, torch::Tensor rec_i,
    double rdy, double rdx,
    int64_t t, int64_t interval, int64_t snap_off,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t cq_batched, int64_t ca_batched, int64_t cb_batched,
    int64_t n_shots, int64_t ny, int64_t nx, int64_t ny_nx,
    int64_t n_src, int64_t n_rec, int64_t do_record)
{
    step_h(cq, ey, hx, hz, m_ey_z, m_ey_x,
           ayh, byh, axh, bxh, kyh, kxh, c, rdy, rdx,
           pml_y0, pml_y1, pml_x0, pml_x1,
           fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1, cq_batched);
    step_e_storage(ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
                   ey_store, curl_store,
                   ay, by, ax, bx, ky, kx, c, rdy, rdx,
                   t, interval, snap_off,
                   pml_y0, pml_y1, pml_x0, pml_x1,
                   fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
                   ca_batched, cb_batched);
    if (n_src > 0)
        inject(ey, f, src_i, t, n_shots, n_src, ny_nx);
    if (do_record && n_rec > 0)
        record(ey, r, rec_i, t, n_shots, n_rec, ny_nx);
}

void backward_step(
    torch::Tensor ca, torch::Tensor cb, torch::Tensor cq,
    torch::Tensor lam_ey,
    torch::Tensor m_lambda_hx_z, torch::Tensor m_lambda_hz_x,
    torch::Tensor work_x, torch::Tensor work_y,
    torch::Tensor work2_x, torch::Tensor work2_y,
    torch::Tensor m_lambda_ey_x, torch::Tensor m_lambda_ey_z,
    torch::Tensor lam_hx, torch::Tensor lam_hz,
    torch::Tensor grad_r, torch::Tensor rec_i,
    std::optional<torch::Tensor> grad_f, torch::Tensor src_i,
    torch::Tensor ey_store, torch::Tensor curl_store,
    torch::Tensor grad_ca, torch::Tensor grad_cb,
    std::optional<torch::Tensor> grad_cq,
    torch::Tensor ay, torch::Tensor by, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor ky, torch::Tensor kx,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor axh, torch::Tensor bxh,
    torch::Tensor kyh, torch::Tensor kxh,
    torch::Tensor c,
    double rdy, double rdx,
    int64_t t, int64_t interval, double scale, int64_t snap_off,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t cq_batched, int64_t ca_batched, int64_t cb_batched,
    int64_t n_shots, int64_t ny, int64_t nx, int64_t ny_nx,
    int64_t n_src, int64_t n_rec,
    int64_t need_cq, int64_t need_f)
{
    if (n_rec > 0)
        record_grad_r(lam_ey, grad_r, rec_i, t, n_shots, n_rec, ny_nx);
    if (need_f && n_src > 0)
        record_grad_f(lam_ey, grad_f.value(), src_i, t, n_shots, n_src, ny_nx);
    adjoint_e_stage1(ca, cb, lam_ey, m_lambda_hx_z, m_lambda_hz_x,
                     work_x, work_y, ay, by, ax, bx, ky, kx,
                     ey_store, curl_store, grad_ca, grad_cb,
                     t, interval, scale, snap_off,
                     pml_y0, pml_y1, pml_x0, pml_x1,
                     fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
                     ca_batched, cb_batched);
    adjoint_e_stage2(work_x, work_y, lam_hx, lam_hz, c, rdy, rdx,
                     fd_pad_y0, fd_pad_x0);
    adjoint_h_stage1(cq, lam_hx, lam_hz, m_lambda_ey_x, m_lambda_ey_z,
                     work2_x, work2_y, ayh, byh, axh, bxh, kyh, kxh,
                     pml_y0, pml_y1, pml_x0, pml_x1,
                     fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
                     cq_batched);
    if (need_cq && t % interval == 0)
        cq_grad(cq, lam_hx, lam_hz, ey_store, m_lambda_ey_x, m_lambda_ey_z,
                grad_cq.value(), ayh, kyh, axh, kxh, c, rdy, rdx,
                t, interval, scale, snap_off,
                pml_y0, pml_y1, pml_x0, pml_x1,
                fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
                cq_batched);
    adjoint_h_stage2(work2_x, work2_y, lam_ey, c, rdy, rdx,
                     fd_pad_y0, fd_pad_x0);
}

// ---------------- pybind11 bindings ----------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_step", &forward_step);
    m.def("backward_step", &backward_step);
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
    NAMI_STORAGE_PYBIND(m);
}
