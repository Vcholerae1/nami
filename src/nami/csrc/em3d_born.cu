/*
 * nami em3d_born: 3D electromagnetic first-order Born FDTD (staggered Yee
 * grid, C-PML) with a native exact-transpose CUDA adjoint.
 *
 * Forward/adjoint discretisation is the 3D generalisation of nami's
 * em2d_tm_born on top of nami's em3d background operator (staggered
 * Yee-grid conventions): a background field
 * (Ex/Ey/Ez, Hx/Hy/Hz) is propagated with the unperturbed ca/cb/cq
 * coefficients exactly as ``em3d``, and a scattered field (dEx/dEy/dEz,
 * dHx/dHy/dHz) is driven by the first-order Born scattering sources
 *
 *     dEx = ca*dEx + cb*dcurl_x + dca*ex_old + dcb*curl_x      (y, z anal.)
 *     dHx -= cq*(ddEy_dz - ddEz_dy) + dcq*(dEy_dz - dEz_dy)    (y, z anal.)
 *
 * where ``ex_old``/``curl_*`` are the pre-update background field/curl and
 * ``dca/dcb/dcq`` are the exact linearizations of the material coefficients
 * w.r.t. ``(epsilon_scatter, sigma_scatter, mu_scatter)`` around the
 * background (computed on the Python side: the material-coefficient
 * linearization plus ``dcq = -cq/mu*dmu``).  Source injection pre-scales
 * by ``cb * -1/(dx dy dz)`` for the background and the ``dcb`` analogue
 * for the scattered field (the linearized source term), both into
 * ``source_component``; receivers record the post-injection
 * scattered field from ``receiver_component`` (and the background field
 * from the same component for the optional ``bg_receiver_locations``).
 *
 * The backward pass is the exact discrete transpose of the coupled
 * (background + scattered) system: separate adjoint fields for each wave
 * component, ``record_grad_r``/``record_grad_f`` seeding, ``coeff_grad``
 * and ``cq_grad`` model-gradient accumulators on the snapshot interval, and
 * the two-stage E/H transpose kernels with time-reversed CPML memory
 * recursions (the six-field generalisation of em2d_tm_born's adjoint, in
 * the same style as em3d.cu).  The cq/dcq gradients use the PML-modified
 * E-derivatives snapshotted by the H half-step (as em2d_tm_born does), so
 * no derivative reconstruction is needed.
 *
 * Kernels are intentionally unoptimised (one flat launch per step, naive
 * stencil) -- the correctness baseline.
 *
 * Spatial FD order 2/4/6/8 via the ``fd.STAGGERED_DIFF1`` coefficient
 * tables: kernels are driven by the zero-padded coefficient array (max
 * radius 4) and the per-side FD padding ``fd_pad =
 * [accuracy // 2, accuracy // 2 - 1] * 3``.  Snapshots (24 streams:
 * pre-update E and PML-modified curls for both fields, plus the
 * PML-modified H-step E-derivatives for both fields) live in C++-owned
 * storage with device / pinned-cpu / disk offload and
 * optional bf16 compression.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "storage.h"

#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------- coefficient-driven staggered difference operators ----------------
// Same conventions as em3d.cu / born_em_el.cu: ``diff_int_*`` evaluates a
// derivative at a half-integer grid point of an integer-stored field (H
// step), ``diff_half_*`` at an integer grid point of a half-integer-stored
// field (E step curl); the two are exact discrete transposes of each other
// up to the curl sign.  The loop bound is the runtime stencil radius, never
// a hardcoded maximum.
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
__device__ __forceinline__ T diff_int_z(const T* __restrict__ u, long off,
                                        const T* __restrict__ c, T rdz, long ny_nx, int radius)
{
    T d = (T)0;
    for (int k = 1; k <= radius; ++k)
        d += c[k - 1] * (u[off + k * ny_nx] - u[off - (k - 1) * ny_nx]);
    return d * rdz;
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

template <typename T>
__device__ __forceinline__ T diff_half_z(const T* __restrict__ u, long off,
                                         const T* __restrict__ c, T rdz, long ny_nx, int radius)
{
    T d = (T)0;
    for (int k = 1; k <= radius; ++k)
        d += c[k - 1] * (u[off + (k - 1) * ny_nx] - u[off - k * ny_nx]);
    return d * rdz;
}

// ==================== forward: H half-step (background + scattered) ====================
// Per step the background H update (exactly em3d's step_h) and the scattered
// H update driven by the scattered E curls plus the first-order dcq source
// (``dcq*deriv(E_bg)`` when mu is perturbed).  The six PML-modified
// derivatives of both E fields are snapshotted on the sampling interval for
// the cq/dcq model gradients.
template <typename T>
__global__ void born_step_h_kernel(
    const T* __restrict__ cq, const T* __restrict__ dcq,
    const T* __restrict__ ex, const T* __restrict__ ey, const T* __restrict__ ez,
    const T* __restrict__ dEx, const T* __restrict__ dEy, const T* __restrict__ dEz,
    T* __restrict__ hx, T* __restrict__ hy, T* __restrict__ hz,
    T* __restrict__ dHx, T* __restrict__ dHy, T* __restrict__ dHz,
    T* __restrict__ m_ey_z, T* __restrict__ m_ez_y, T* __restrict__ m_ez_x,
    T* __restrict__ m_ex_z, T* __restrict__ m_ex_y, T* __restrict__ m_ey_x,
    T* __restrict__ dm_ey_z, T* __restrict__ dm_ez_y, T* __restrict__ dm_ez_x,
    T* __restrict__ dm_ex_z, T* __restrict__ dm_ex_y, T* __restrict__ dm_ey_x,
    T* __restrict__ dey_dz_store, T* __restrict__ dez_dy_store,
    T* __restrict__ dez_dx_store, T* __restrict__ dex_dz_store,
    T* __restrict__ dex_dy_store, T* __restrict__ dey_dx_store,
    T* __restrict__ ddey_dz_store, T* __restrict__ ddez_dy_store,
    T* __restrict__ ddez_dx_store, T* __restrict__ ddex_dz_store,
    T* __restrict__ ddex_dy_store, T* __restrict__ ddey_dx_store,
    const T* __restrict__ azh, const T* __restrict__ bzh,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ kzh, const T* __restrict__ kyh, const T* __restrict__ kxh,
    const T* __restrict__ c,
    T rdz, T rdy, T rdx,
    int t, int interval, int64_t snap_off,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int fd_pad_z0, int fd_pad_y0, int fd_pad_x0,
    int cq_batched, int dcq_batched, int store)
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
    const T cq_val = cq_batched ? cq[i] : cq[j];
    const T dcq_val = dcq_batched ? dcq[i] : dcq[j];

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
    T dEz_dy = (T)0;
    T dEz_dx = (T)0;
    T dEx_dz = (T)0;
    T dEx_dy = (T)0;
    T dEy_dx = (T)0;
    if (z >= fd_pad_z0 && z < nz - fd_pad_z0)
        dEy_dz = diff_int_z(ey, i, c, rdz, ny_nx, fd_pad_z0);
    if (y >= fd_pad_y0 && y < ny - fd_pad_y0)
        dEz_dy = diff_int_y(ez, i, c, rdy, nx, fd_pad_y0);
    if (x >= fd_pad_x0 && x < nx - fd_pad_x0)
        dEz_dx = diff_int_x(ez, i, c, rdx, fd_pad_x0);
    if (z >= fd_pad_z0 && z < nz - fd_pad_z0)
        dEx_dz = diff_int_z(ex, i, c, rdz, ny_nx, fd_pad_z0);
    if (y >= fd_pad_y0 && y < ny - fd_pad_y0)
        dEx_dy = diff_int_y(ex, i, c, rdy, nx, fd_pad_y0);
    if (x >= fd_pad_x0 && x < nx - fd_pad_x0)
        dEy_dx = diff_int_x(ey, i, c, rdx, fd_pad_x0);

    T ddEy_dz = (T)0;
    T ddEz_dy = (T)0;
    T ddEz_dx = (T)0;
    T ddEx_dz = (T)0;
    T ddEx_dy = (T)0;
    T ddEy_dx = (T)0;
    if (z >= fd_pad_z0 && z < nz - fd_pad_z0)
        ddEy_dz = diff_int_z(dEy, i, c, rdz, ny_nx, fd_pad_z0);
    if (y >= fd_pad_y0 && y < ny - fd_pad_y0)
        ddEz_dy = diff_int_y(dEz, i, c, rdy, nx, fd_pad_y0);
    if (x >= fd_pad_x0 && x < nx - fd_pad_x0)
        ddEz_dx = diff_int_x(dEz, i, c, rdx, fd_pad_x0);
    if (z >= fd_pad_z0 && z < nz - fd_pad_z0)
        ddEx_dz = diff_int_z(dEx, i, c, rdz, ny_nx, fd_pad_z0);
    if (y >= fd_pad_y0 && y < ny - fd_pad_y0)
        ddEx_dy = diff_int_y(dEx, i, c, rdy, nx, fd_pad_y0);
    if (x >= fd_pad_x0 && x < nx - fd_pad_x0)
        ddEy_dx = diff_int_x(dEy, i, c, rdx, fd_pad_x0);

    if (z < pml_z0 || z >= pml_z1h) {
        m_ey_z[i] = bzh[z] * m_ey_z[i] + azh[z] * dEy_dz;
        dEy_dz = dEy_dz / kzh[z] + m_ey_z[i];
        dm_ey_z[i] = bzh[z] * dm_ey_z[i] + azh[z] * ddEy_dz;
        ddEy_dz = ddEy_dz / kzh[z] + dm_ey_z[i];
        m_ex_z[i] = bzh[z] * m_ex_z[i] + azh[z] * dEx_dz;
        dEx_dz = dEx_dz / kzh[z] + m_ex_z[i];
        dm_ex_z[i] = bzh[z] * dm_ex_z[i] + azh[z] * ddEx_dz;
        ddEx_dz = ddEx_dz / kzh[z] + dm_ex_z[i];
    }
    if (y < pml_y0 || y >= pml_y1h) {
        m_ez_y[i] = byh[y] * m_ez_y[i] + ayh[y] * dEz_dy;
        dEz_dy = dEz_dy / kyh[y] + m_ez_y[i];
        dm_ez_y[i] = byh[y] * dm_ez_y[i] + ayh[y] * ddEz_dy;
        ddEz_dy = ddEz_dy / kyh[y] + dm_ez_y[i];
        m_ex_y[i] = byh[y] * m_ex_y[i] + ayh[y] * dEx_dy;
        dEx_dy = dEx_dy / kyh[y] + m_ex_y[i];
        dm_ex_y[i] = byh[y] * dm_ex_y[i] + ayh[y] * ddEx_dy;
        ddEx_dy = ddEx_dy / kyh[y] + dm_ex_y[i];
    }
    if (x < pml_x0 || x >= pml_x1h) {
        m_ez_x[i] = bxh[x] * m_ez_x[i] + axh[x] * dEz_dx;
        dEz_dx = dEz_dx / kxh[x] + m_ez_x[i];
        dm_ez_x[i] = bxh[x] * dm_ez_x[i] + axh[x] * ddEz_dx;
        ddEz_dx = ddEz_dx / kxh[x] + dm_ez_x[i];
        m_ey_x[i] = bxh[x] * m_ey_x[i] + axh[x] * dEy_dx;
        dEy_dx = dEy_dx / kxh[x] + m_ey_x[i];
        dm_ey_x[i] = bxh[x] * dm_ey_x[i] + axh[x] * ddEy_dx;
        ddEy_dx = ddEy_dx / kxh[x] + dm_ey_x[i];
    }

    if (store && t % interval == 0) {
        const int64_t soff = snap_off + i;
        dey_dz_store[soff] = dEy_dz;
        ddey_dz_store[soff] = ddEy_dz;
        dez_dy_store[soff] = dEz_dy;
        ddez_dy_store[soff] = ddEz_dy;
        dez_dx_store[soff] = dEz_dx;
        ddez_dx_store[soff] = ddEz_dx;
        dex_dz_store[soff] = dEx_dz;
        ddex_dz_store[soff] = ddEx_dz;
        dex_dy_store[soff] = dEx_dy;
        ddex_dy_store[soff] = ddEx_dy;
        dey_dx_store[soff] = dEy_dx;
        ddey_dx_store[soff] = ddEy_dx;
    }

    hx[i] -= cq_val * (dEy_dz - dEz_dy);
    dHx[i] -= cq_val * (ddEy_dz - ddEz_dy) + dcq_val * (dEy_dz - dEz_dy);
    hy[i] -= cq_val * (dEz_dx - dEx_dz);
    dHy[i] -= cq_val * (ddEz_dx - ddEx_dz) + dcq_val * (dEz_dx - dEx_dz);
    hz[i] -= cq_val * (dEx_dy - dEy_dx);
    dHz[i] -= cq_val * (ddEx_dy - ddEy_dx) + dcq_val * (dEx_dy - dEy_dx);
}

// ==================== forward: combined bg + scattered E update with snapshots ====================
// Computes the background curls (from Hx/Hy/Hz with the integer CPML
// memories) and the scattered curls (from dHx/dHy/dHz), snapshots the
// pre-update fields and both curl sets on every interval step, then updates
//   Ex  = ca*Ex  + cb*curl_x
//   dEx = ca*dEx + cb*dcurl_x + dca*ex_old + dcb*curl_x
// (and the y/z analogues).  Only the FD interior is updated (E fields
// outside it are zero, mirroring em3d).
template <typename T>
__global__ void born_step_e_kernel(
    const T* __restrict__ ca, const T* __restrict__ cb,
    const T* __restrict__ dca, const T* __restrict__ dcb,
    const T* __restrict__ hx, const T* __restrict__ hy, const T* __restrict__ hz,
    const T* __restrict__ dHx, const T* __restrict__ dHy, const T* __restrict__ dHz,
    T* __restrict__ ex, T* __restrict__ ey, T* __restrict__ ez,
    T* __restrict__ dEx, T* __restrict__ dEy, T* __restrict__ dEz,
    T* __restrict__ m_hy_z, T* __restrict__ m_hz_y, T* __restrict__ m_hz_x,
    T* __restrict__ m_hx_z, T* __restrict__ m_hx_y, T* __restrict__ m_hy_x,
    T* __restrict__ dm_hy_z, T* __restrict__ dm_hz_y, T* __restrict__ dm_hz_x,
    T* __restrict__ dm_hx_z, T* __restrict__ dm_hx_y, T* __restrict__ dm_hy_x,
    T* __restrict__ ex_store, T* __restrict__ ey_store, T* __restrict__ ez_store,
    T* __restrict__ curl_x_store, T* __restrict__ curl_y_store,
    T* __restrict__ curl_z_store,
    T* __restrict__ dex_store, T* __restrict__ dey_store, T* __restrict__ dez_store,
    T* __restrict__ dcurl_x_store, T* __restrict__ dcurl_y_store,
    T* __restrict__ dcurl_z_store,
    const T* __restrict__ az, const T* __restrict__ bz,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ kz, const T* __restrict__ ky, const T* __restrict__ kx,
    const T* __restrict__ c,
    T rdz, T rdy, T rdx,
    int t, int interval, int64_t snap_off,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int fd_pad_z0, int fd_pad_z1, int fd_pad_y0, int fd_pad_y1,
    int fd_pad_x0, int fd_pad_x1,
    int ca_batched, int cb_batched, int dca_batched, int dcb_batched)
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
    if (z < fd_pad_z0 || z >= nz - fd_pad_z1 || y < fd_pad_y0 ||
        y >= ny - fd_pad_y1 || x < fd_pad_x0 || x >= nx - fd_pad_x1)
        return;
    const long j = (long)z * ny_nx + (long)y * nx + x;
    const T ca_val = ca_batched ? ca[i] : ca[j];
    const T cb_val = cb_batched ? cb[i] : cb[j];
    const T dca_val = dca_batched ? dca[i] : dca[j];
    const T dcb_val = dcb_batched ? dcb[i] : dcb[j];

    T dHy_dz = diff_half_z(hy, i, c, rdz, ny_nx, fd_pad_z0);
    T dHz_dy = diff_half_y(hz, i, c, rdy, nx, fd_pad_y0);
    T dHz_dx = diff_half_x(hz, i, c, rdx, fd_pad_x0);
    T dHx_dz = diff_half_z(hx, i, c, rdz, ny_nx, fd_pad_z0);
    T dHx_dy = diff_half_y(hx, i, c, rdy, nx, fd_pad_y0);
    T dHy_dx = diff_half_x(hy, i, c, rdx, fd_pad_x0);

    T ddHy_dz = diff_half_z(dHy, i, c, rdz, ny_nx, fd_pad_z0);
    T ddHz_dy = diff_half_y(dHz, i, c, rdy, nx, fd_pad_y0);
    T ddHz_dx = diff_half_x(dHz, i, c, rdx, fd_pad_x0);
    T ddHx_dz = diff_half_z(dHx, i, c, rdz, ny_nx, fd_pad_z0);
    T ddHx_dy = diff_half_y(dHx, i, c, rdy, nx, fd_pad_y0);
    T ddHy_dx = diff_half_x(dHy, i, c, rdx, fd_pad_x0);

    if (z < pml_z0 || z >= pml_z1) {
        m_hy_z[i] = bz[z] * m_hy_z[i] + az[z] * dHy_dz;
        dHy_dz = dHy_dz / kz[z] + m_hy_z[i];
        m_hx_z[i] = bz[z] * m_hx_z[i] + az[z] * dHx_dz;
        dHx_dz = dHx_dz / kz[z] + m_hx_z[i];
        dm_hy_z[i] = bz[z] * dm_hy_z[i] + az[z] * ddHy_dz;
        ddHy_dz = ddHy_dz / kz[z] + dm_hy_z[i];
        dm_hx_z[i] = bz[z] * dm_hx_z[i] + az[z] * ddHx_dz;
        ddHx_dz = ddHx_dz / kz[z] + dm_hx_z[i];
    }
    if (y < pml_y0 || y >= pml_y1) {
        m_hz_y[i] = by[y] * m_hz_y[i] + ay[y] * dHz_dy;
        dHz_dy = dHz_dy / ky[y] + m_hz_y[i];
        m_hx_y[i] = by[y] * m_hx_y[i] + ay[y] * dHx_dy;
        dHx_dy = dHx_dy / ky[y] + m_hx_y[i];
        dm_hz_y[i] = by[y] * dm_hz_y[i] + ay[y] * ddHz_dy;
        ddHz_dy = ddHz_dy / ky[y] + dm_hz_y[i];
        dm_hx_y[i] = by[y] * dm_hx_y[i] + ay[y] * ddHx_dy;
        ddHx_dy = ddHx_dy / ky[y] + dm_hx_y[i];
    }
    if (x < pml_x0 || x >= pml_x1) {
        m_hz_x[i] = bx[x] * m_hz_x[i] + ax[x] * dHz_dx;
        dHz_dx = dHz_dx / kx[x] + m_hz_x[i];
        m_hy_x[i] = bx[x] * m_hy_x[i] + ax[x] * dHy_dx;
        dHy_dx = dHy_dx / kx[x] + m_hy_x[i];
        dm_hz_x[i] = bx[x] * dm_hz_x[i] + ax[x] * ddHz_dx;
        ddHz_dx = ddHz_dx / kx[x] + dm_hz_x[i];
        dm_hy_x[i] = bx[x] * dm_hy_x[i] + ax[x] * ddHy_dx;
        ddHy_dx = ddHy_dx / kx[x] + dm_hy_x[i];
    }

    const T curl_x = dHy_dz - dHz_dy;
    const T curl_y = dHz_dx - dHx_dz;
    const T curl_z = dHx_dy - dHy_dx;
    const T dcurl_x = ddHy_dz - ddHz_dy;
    const T dcurl_y = ddHz_dx - ddHx_dz;
    const T dcurl_z = ddHx_dy - ddHy_dx;
    if (t % interval == 0) {
        const int64_t soff = snap_off + i;
        ex_store[soff] = ex[i];
        ey_store[soff] = ey[i];
        ez_store[soff] = ez[i];
        curl_x_store[soff] = curl_x;
        curl_y_store[soff] = curl_y;
        curl_z_store[soff] = curl_z;
        dex_store[soff] = dEx[i];
        dey_store[soff] = dEy[i];
        dez_store[soff] = dEz[i];
        dcurl_x_store[soff] = dcurl_x;
        dcurl_y_store[soff] = dcurl_y;
        dcurl_z_store[soff] = dcurl_z;
    }
    const T ex_old = ex[i];
    const T ey_old = ey[i];
    const T ez_old = ez[i];
    ex[i] = ca_val * ex_old + cb_val * curl_x;
    ey[i] = ca_val * ey_old + cb_val * curl_y;
    ez[i] = ca_val * ez_old + cb_val * curl_z;
    dEx[i] = ca_val * dEx[i] + cb_val * dcurl_x + dca_val * ex_old + dcb_val * curl_x;
    dEy[i] = ca_val * dEy[i] + cb_val * dcurl_y + dca_val * ey_old + dcb_val * curl_y;
    dEz[i] = ca_val * dEz[i] + cb_val * dcurl_z + dca_val * ez_old + dcb_val * curl_z;
}

// ==================== forward: source injection / receiver recording ====================
template <typename T>
__global__ void born_inject_kernel(
    T* __restrict__ field, T* __restrict__ dfield,
    const T* __restrict__ f_bg, const T* __restrict__ f_sc,
    const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long shot_numel)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        if (idx >= 0) {
            const long off = (long)s * shot_numel + idx;
            const long fidx = ((long)t * n_shots + s) * n_src + k;
            field[off] += f_bg[fidx];
            dfield[off] += f_sc[fidx];
        }
    }
}

template <typename T>
__global__ void born_record_kernel(
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

// ==================== backward: receiver / source seeding ====================
template <typename T>
__global__ void born_record_grad_r_kernel(
    T* __restrict__ lam_field, const T* __restrict__ grad_r,
    const long* __restrict__ rec_i,
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
__global__ void born_record_grad_f_kernel(
    const T* __restrict__ lam_field, const T* __restrict__ lam_dfield,
    T* __restrict__ grad_f_bg, T* __restrict__ grad_f_sc,
    const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long shot_numel)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        const long fidx = ((long)t * n_shots + s) * n_src + k;
        if (idx >= 0) {
            const long off = (long)s * shot_numel + idx;
            grad_f_bg[fidx] = lam_field[off];
            grad_f_sc[fidx] = lam_dfield[off];
        } else {
            grad_f_bg[fidx] = (T)0;
            grad_f_sc[fidx] = (T)0;
        }
    }
}

// ==================== backward: ca/cb/dca/dcb model gradients ====================
template <typename T>
__global__ void born_coeff_grad_kernel(
    const T* __restrict__ lam_ex, const T* __restrict__ lam_ey,
    const T* __restrict__ lam_ez,
    const T* __restrict__ lam_dEx, const T* __restrict__ lam_dEy,
    const T* __restrict__ lam_dEz,
    const T* __restrict__ ex_store, const T* __restrict__ ey_store,
    const T* __restrict__ ez_store,
    const T* __restrict__ curl_x_store, const T* __restrict__ curl_y_store,
    const T* __restrict__ curl_z_store,
    const T* __restrict__ dex_store, const T* __restrict__ dey_store,
    const T* __restrict__ dez_store,
    const T* __restrict__ dcurl_x_store, const T* __restrict__ dcurl_y_store,
    const T* __restrict__ dcurl_z_store,
    T* __restrict__ grad_ca, T* __restrict__ grad_cb,
    T* __restrict__ grad_dca, T* __restrict__ grad_dcb,
    T scale, int64_t snap_off,
    int n_shots, int nz, int ny, int nx,
    int fd_pad_z0, int fd_pad_z1, int fd_pad_y0, int fd_pad_y1,
    int fd_pad_x0, int fd_pad_x1)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)n_shots * nz * ny * nx;
    if (i >= total)
        return;
    const int x = (int)(i % nx);
    const int y = (int)((i / nx) % ny);
    const int z = (int)((i / ((long)nx * ny)) % nz);
    if (z < fd_pad_z0 || z >= nz - fd_pad_z1 || y < fd_pad_y0 ||
        y >= ny - fd_pad_y1 || x < fd_pad_x0 || x >= nx - fd_pad_x1)
        return;
    const int64_t soff = snap_off + i;
    const T lex = lam_ex[i] * scale;
    const T ley = lam_ey[i] * scale;
    const T lez = lam_ez[i] * scale;
    const T ldx = lam_dEx[i] * scale;
    const T ldy = lam_dEy[i] * scale;
    const T ldz = lam_dEz[i] * scale;
    grad_ca[i] += lex * ex_store[soff] + ley * ey_store[soff] + lez * ez_store[soff] +
                  ldx * dex_store[soff] + ldy * dey_store[soff] + ldz * dez_store[soff];
    grad_cb[i] += lex * curl_x_store[soff] + ley * curl_y_store[soff] +
                  lez * curl_z_store[soff] + ldx * dcurl_x_store[soff] +
                  ldy * dcurl_y_store[soff] + ldz * dcurl_z_store[soff];
    grad_dca[i] += ldx * ex_store[soff] + ldy * ey_store[soff] + ldz * ez_store[soff];
    grad_dcb[i] += ldx * curl_x_store[soff] + ldy * curl_y_store[soff] +
                   ldz * curl_z_store[soff];
}

// ==================== backward: transpose of the combined E update, stage 1 ====================
// From the post-E-update adjoints:
//   g_bg  = cb*lam_E  + dcb*lam_dE      g_sc = cb*lam_dE
//   lam_E  = ca*lam_E  + dca*lam_dE     lam_dE = ca*lam_dE
// and builds the twelve work arrays (the PML-modified curls applied to the
// g values) with the time-reversed integer-profile memory recursions.
template <typename T>
__global__ void born_adjoint_e_stage1_kernel(
    const T* __restrict__ ca, const T* __restrict__ cb,
    const T* __restrict__ dca, const T* __restrict__ dcb,
    T* __restrict__ lam_ex, T* __restrict__ lam_ey, T* __restrict__ lam_ez,
    T* __restrict__ lam_dEx, T* __restrict__ lam_dEy, T* __restrict__ lam_dEz,
    T* __restrict__ m_lambda_hy_z, T* __restrict__ m_lambda_hz_y,
    T* __restrict__ m_lambda_hz_x, T* __restrict__ m_lambda_hx_z,
    T* __restrict__ m_lambda_hx_y, T* __restrict__ m_lambda_hy_x,
    T* __restrict__ dm_lambda_hy_z, T* __restrict__ dm_lambda_hz_y,
    T* __restrict__ dm_lambda_hz_x, T* __restrict__ dm_lambda_hx_z,
    T* __restrict__ dm_lambda_hx_y, T* __restrict__ dm_lambda_hy_x,
    T* __restrict__ work_hy_z, T* __restrict__ work_hz_y,
    T* __restrict__ work_hz_x, T* __restrict__ work_hx_z,
    T* __restrict__ work_hx_y, T* __restrict__ work_hy_x,
    T* __restrict__ work_dhy_z, T* __restrict__ work_dhz_y,
    T* __restrict__ work_dhz_x, T* __restrict__ work_dhx_z,
    T* __restrict__ work_dhx_y, T* __restrict__ work_dhy_x,
    const T* __restrict__ az, const T* __restrict__ bz,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ kz, const T* __restrict__ ky, const T* __restrict__ kx,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int fd_pad_z0, int fd_pad_z1, int fd_pad_y0, int fd_pad_y1,
    int fd_pad_x0, int fd_pad_x1,
    int ca_batched, int cb_batched, int dca_batched, int dcb_batched)
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
    const bool active = z >= fd_pad_z0 && z < nz - fd_pad_z1 && y >= fd_pad_y0 &&
                        y < ny - fd_pad_y1 && x >= fd_pad_x0 && x < nx - fd_pad_x1;
    if (!active) {
        work_hy_z[i] = (T)0; work_hz_y[i] = (T)0; work_hz_x[i] = (T)0;
        work_hx_z[i] = (T)0; work_hx_y[i] = (T)0; work_hy_x[i] = (T)0;
        work_dhy_z[i] = (T)0; work_dhz_y[i] = (T)0; work_dhz_x[i] = (T)0;
        work_dhx_z[i] = (T)0; work_dhx_y[i] = (T)0; work_dhy_x[i] = (T)0;
        return;
    }
    const T ca_val = ca_batched ? ca[i] : ca[j];
    const T cb_val = cb_batched ? cb[i] : cb[j];
    const T dca_val = dca_batched ? dca[i] : dca[j];
    const T dcb_val = dcb_batched ? dcb[i] : dcb[j];
    const T gx = cb_val * lam_ex[i] + dcb_val * lam_dEx[i];
    const T gy = cb_val * lam_ey[i] + dcb_val * lam_dEy[i];
    const T gz = cb_val * lam_ez[i] + dcb_val * lam_dEz[i];
    const T gxd = cb_val * lam_dEx[i];
    const T gyd = cb_val * lam_dEy[i];
    const T gzd = cb_val * lam_dEz[i];
    lam_ex[i] = ca_val * lam_ex[i] + dca_val * lam_dEx[i];
    lam_ey[i] = ca_val * lam_ey[i] + dca_val * lam_dEy[i];
    lam_ez[i] = ca_val * lam_ez[i] + dca_val * lam_dEz[i];
    lam_dEx[i] = ca_val * lam_dEx[i];
    lam_dEy[i] = ca_val * lam_dEy[i];
    lam_dEz[i] = ca_val * lam_dEz[i];

    // Ex: +dHy_dz - dHz_dy
    if (z < pml_z0 || z >= pml_z1) {
        const T w = gx + bz[z] * m_lambda_hy_z[i];
        work_hy_z[i] = gx / kz[z] + az[z] * w;
        m_lambda_hy_z[i] = w;
        const T wd = gxd + bz[z] * dm_lambda_hy_z[i];
        work_dhy_z[i] = gxd / kz[z] + az[z] * wd;
        dm_lambda_hy_z[i] = wd;
    } else {
        work_hy_z[i] = gx;
        work_dhy_z[i] = gxd;
    }
    if (y < pml_y0 || y >= pml_y1) {
        const T w = -gx + by[y] * m_lambda_hz_y[i];
        work_hz_y[i] = -gx / ky[y] + ay[y] * w;
        m_lambda_hz_y[i] = w;
        const T wd = -gxd + by[y] * dm_lambda_hz_y[i];
        work_dhz_y[i] = -gxd / ky[y] + ay[y] * wd;
        dm_lambda_hz_y[i] = wd;
    } else {
        work_hz_y[i] = -gx;
        work_dhz_y[i] = -gxd;
    }
    // Ey: +dHz_dx - dHx_dz
    if (x < pml_x0 || x >= pml_x1) {
        const T w = gy + bx[x] * m_lambda_hz_x[i];
        work_hz_x[i] = gy / kx[x] + ax[x] * w;
        m_lambda_hz_x[i] = w;
        const T wd = gyd + bx[x] * dm_lambda_hz_x[i];
        work_dhz_x[i] = gyd / kx[x] + ax[x] * wd;
        dm_lambda_hz_x[i] = wd;
    } else {
        work_hz_x[i] = gy;
        work_dhz_x[i] = gyd;
    }
    if (z < pml_z0 || z >= pml_z1) {
        const T w = -gy + bz[z] * m_lambda_hx_z[i];
        work_hx_z[i] = -gy / kz[z] + az[z] * w;
        m_lambda_hx_z[i] = w;
        const T wd = -gyd + bz[z] * dm_lambda_hx_z[i];
        work_dhx_z[i] = -gyd / kz[z] + az[z] * wd;
        dm_lambda_hx_z[i] = wd;
    } else {
        work_hx_z[i] = -gy;
        work_dhx_z[i] = -gyd;
    }
    // Ez: +dHx_dy - dHy_dx
    if (y < pml_y0 || y >= pml_y1) {
        const T w = gz + by[y] * m_lambda_hx_y[i];
        work_hx_y[i] = gz / ky[y] + ay[y] * w;
        m_lambda_hx_y[i] = w;
        const T wd = gzd + by[y] * dm_lambda_hx_y[i];
        work_dhx_y[i] = gzd / ky[y] + ay[y] * wd;
        dm_lambda_hx_y[i] = wd;
    } else {
        work_hx_y[i] = gz;
        work_dhx_y[i] = gzd;
    }
    if (x < pml_x0 || x >= pml_x1) {
        const T w = -gz + bx[x] * m_lambda_hy_x[i];
        work_hy_x[i] = -gz / kx[x] + ax[x] * w;
        m_lambda_hy_x[i] = w;
        const T wd = -gzd + bx[x] * dm_lambda_hy_x[i];
        work_dhy_x[i] = -gzd / kx[x] + ax[x] * wd;
        dm_lambda_hy_x[i] = wd;
    } else {
        work_hy_x[i] = -gz;
        work_dhy_x[i] = -gzd;
    }
}

// ==================== backward: transpose of the combined E update, stage 2 ====================
// lam_hy/hz/hx and lam_dHy/dHz/dHx += the transposes of the six half-grid
// curls applied to the work arrays (the transpose of diff_half), summed per
// H component.  No interior guard: boundary cells get zero contributions
// from the zeroed work arrays.
template <typename T>
__global__ void born_adjoint_e_stage2_kernel(
    const T* __restrict__ work_hy_z, const T* __restrict__ work_hz_y,
    const T* __restrict__ work_hz_x, const T* __restrict__ work_hx_z,
    const T* __restrict__ work_hx_y, const T* __restrict__ work_hy_x,
    const T* __restrict__ work_dhy_z, const T* __restrict__ work_dhz_y,
    const T* __restrict__ work_dhz_x, const T* __restrict__ work_dhx_z,
    const T* __restrict__ work_dhx_y, const T* __restrict__ work_dhy_x,
    T* __restrict__ lam_hy, T* __restrict__ lam_hz, T* __restrict__ lam_hx,
    T* __restrict__ lam_dhy, T* __restrict__ lam_dhz, T* __restrict__ lam_dhx,
    const T* __restrict__ c,
    T rdz, T rdy, T rdx,
    int n_shots, int nz, int ny, int nx,
    int fd_pad_z0, int fd_pad_y0, int fd_pad_x0)
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
    T acc_dz = (T)0;
    for (int k = 1; k <= fd_pad_z0; ++k) {
        const T ck = c[k - 1];
        if (z - k + 1 >= 0) {
            acc_z += ck * work_hy_z[i - (long)(k - 1) * ny_nx];
            acc_dz += ck * work_dhy_z[i - (long)(k - 1) * ny_nx];
        }
        if (z + k < nz) {
            acc_z -= ck * work_hy_z[i + (long)k * ny_nx];
            acc_dz -= ck * work_dhy_z[i + (long)k * ny_nx];
        }
    }
    lam_hy[i] += acc_z * rdz;
    lam_dhy[i] += acc_dz * rdz;
    T acc_hy_x = (T)0;
    T acc_dhy_x = (T)0;
    for (int k = 1; k <= fd_pad_x0; ++k) {
        const T ck = c[k - 1];
        if (x - k + 1 >= 0) {
            acc_hy_x += ck * work_hy_x[i - k + 1];
            acc_dhy_x += ck * work_dhy_x[i - k + 1];
        }
        if (x + k < nx) {
            acc_hy_x -= ck * work_hy_x[i + k];
            acc_dhy_x -= ck * work_dhy_x[i + k];
        }
    }
    lam_hy[i] += acc_hy_x * rdx;
    lam_dhy[i] += acc_dhy_x * rdx;

    T acc_y = (T)0;
    T acc_dy = (T)0;
    for (int k = 1; k <= fd_pad_y0; ++k) {
        const T ck = c[k - 1];
        if (y - k + 1 >= 0) {
            acc_y += ck * work_hz_y[i - (long)(k - 1) * nx];
            acc_dy += ck * work_dhz_y[i - (long)(k - 1) * nx];
        }
        if (y + k < ny) {
            acc_y -= ck * work_hz_y[i + (long)k * nx];
            acc_dy -= ck * work_dhz_y[i + (long)k * nx];
        }
    }
    lam_hz[i] += acc_y * rdy;
    lam_dhz[i] += acc_dy * rdy;
    T acc_hz_x = (T)0;
    T acc_dhz_x = (T)0;
    for (int k = 1; k <= fd_pad_x0; ++k) {
        const T ck = c[k - 1];
        if (x - k + 1 >= 0) {
            acc_hz_x += ck * work_hz_x[i - k + 1];
            acc_dhz_x += ck * work_dhz_x[i - k + 1];
        }
        if (x + k < nx) {
            acc_hz_x -= ck * work_hz_x[i + k];
            acc_dhz_x -= ck * work_dhz_x[i + k];
        }
    }
    lam_hz[i] += acc_hz_x * rdx;
    lam_dhz[i] += acc_dhz_x * rdx;

    T acc_hx_z = (T)0;
    T acc_dhx_z = (T)0;
    for (int k = 1; k <= fd_pad_z0; ++k) {
        const T ck = c[k - 1];
        if (z - k + 1 >= 0) {
            acc_hx_z += ck * work_hx_z[i - (long)(k - 1) * ny_nx];
            acc_dhx_z += ck * work_dhx_z[i - (long)(k - 1) * ny_nx];
        }
        if (z + k < nz) {
            acc_hx_z -= ck * work_hx_z[i + (long)k * ny_nx];
            acc_dhx_z -= ck * work_dhx_z[i + (long)k * ny_nx];
        }
    }
    lam_hx[i] += acc_hx_z * rdz;
    lam_dhx[i] += acc_dhx_z * rdz;
    T acc_hx_y = (T)0;
    T acc_dhx_y = (T)0;
    for (int k = 1; k <= fd_pad_y0; ++k) {
        const T ck = c[k - 1];
        if (y - k + 1 >= 0) {
            acc_hx_y += ck * work_hx_y[i - (long)(k - 1) * nx];
            acc_dhx_y += ck * work_dhx_y[i - (long)(k - 1) * nx];
        }
        if (y + k < ny) {
            acc_hx_y -= ck * work_hx_y[i + (long)k * nx];
            acc_dhx_y -= ck * work_dhx_y[i + (long)k * nx];
        }
    }
    lam_hx[i] += acc_hx_y * rdy;
    lam_dhx[i] += acc_dhx_y * rdy;
}

// ==================== backward: transpose of the combined H half-step, stage 1 ====================
// From lam_hx/lam_hy/lam_hz (bg) and lam_dhx/lam_dhy/lam_dhz (sc):
//   g2_bg = cq*lam_H + dcq*lam_dH        g2_sc = cq*lam_dH
// (with the curl signs) because the scattered H update's ``dcq*deriv(E_bg)``
// source is also linear in the background E field; the bg work arrays get
// both contributions through the bg half-integer memory, the sc arrays only
// the cq part through the sc memory.
template <typename T>
__global__ void born_adjoint_h_stage1_kernel(
    const T* __restrict__ cq, const T* __restrict__ dcq,
    const T* __restrict__ lam_hx, const T* __restrict__ lam_hy,
    const T* __restrict__ lam_hz,
    const T* __restrict__ lam_dhx, const T* __restrict__ lam_dhy,
    const T* __restrict__ lam_dhz,
    T* __restrict__ m_lambda_ey_z, T* __restrict__ m_lambda_ez_y,
    T* __restrict__ m_lambda_ez_x, T* __restrict__ m_lambda_ex_z,
    T* __restrict__ m_lambda_ex_y, T* __restrict__ m_lambda_ey_x,
    T* __restrict__ dm_lambda_ey_z, T* __restrict__ dm_lambda_ez_y,
    T* __restrict__ dm_lambda_ez_x, T* __restrict__ dm_lambda_ex_z,
    T* __restrict__ dm_lambda_ex_y, T* __restrict__ dm_lambda_ey_x,
    T* __restrict__ work2_ey_z, T* __restrict__ work2_ez_y,
    T* __restrict__ work2_ez_x, T* __restrict__ work2_ex_z,
    T* __restrict__ work2_ex_y, T* __restrict__ work2_ey_x,
    T* __restrict__ work2_dEy_z, T* __restrict__ work2_dEz_y,
    T* __restrict__ work2_dEz_x, T* __restrict__ work2_dEx_z,
    T* __restrict__ work2_dEx_y, T* __restrict__ work2_dEy_x,
    const T* __restrict__ azh, const T* __restrict__ bzh,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ kzh, const T* __restrict__ kyh, const T* __restrict__ kxh,
    int n_shots, int nz, int ny, int nx,
    int pml_z0, int pml_z1, int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int fd_pad_z0, int fd_pad_z1, int fd_pad_y0, int fd_pad_y1,
    int fd_pad_x0, int fd_pad_x1,
    int cq_batched, int dcq_batched)
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
    const bool active = z >= fd_pad_z0 && z < nz - fd_pad_z1 && y >= fd_pad_y0 &&
                        y < ny - fd_pad_y1 && x >= fd_pad_x0 && x < nx - fd_pad_x1;
    if (!active) {
        work2_ey_z[i] = (T)0; work2_ez_y[i] = (T)0; work2_ez_x[i] = (T)0;
        work2_ex_z[i] = (T)0; work2_ex_y[i] = (T)0; work2_ey_x[i] = (T)0;
        work2_dEy_z[i] = (T)0; work2_dEz_y[i] = (T)0; work2_dEz_x[i] = (T)0;
        work2_dEx_z[i] = (T)0; work2_dEx_y[i] = (T)0; work2_dEy_x[i] = (T)0;
        return;
    }
    const T cq_val = cq_batched ? cq[i] : cq[j];
    const T dcq_val = dcq_batched ? dcq[i] : dcq[j];

    int pml_z1h = pml_z1 - 1;
    if (pml_z1 <= pml_z0)
        pml_z1h = pml_z0;
    int pml_y1h = pml_y1 - 1;
    if (pml_y1 <= pml_y0)
        pml_y1h = pml_y0;
    int pml_x1h = pml_x1 - 1;
    if (pml_x1 <= pml_x0)
        pml_x1h = pml_x0;

    // dEy_dz feeds Hx with -cq (and dcq*lam_dhx via the dcq source)
    if (z < nz - fd_pad_z0) {
        const T g2 = -cq_val * lam_hx[i] - dcq_val * lam_dhx[i];
        const T g2d = -cq_val * lam_dhx[i];
        if (z < pml_z0 || z >= pml_z1h) {
            const T w = g2 + bzh[z] * m_lambda_ey_z[i];
            work2_ey_z[i] = g2 / kzh[z] + azh[z] * w;
            m_lambda_ey_z[i] = w;
            const T wd = g2d + bzh[z] * dm_lambda_ey_z[i];
            work2_dEy_z[i] = g2d / kzh[z] + azh[z] * wd;
            dm_lambda_ey_z[i] = wd;
        } else {
            work2_ey_z[i] = g2;
            work2_dEy_z[i] = g2d;
        }
    } else {
        work2_ey_z[i] = (T)0;
        work2_dEy_z[i] = (T)0;
    }
    // dEz_dy feeds Hx with +cq
    if (y < ny - fd_pad_y0) {
        const T g2 = cq_val * lam_hx[i] + dcq_val * lam_dhx[i];
        const T g2d = cq_val * lam_dhx[i];
        if (y < pml_y0 || y >= pml_y1h) {
            const T w = g2 + byh[y] * m_lambda_ez_y[i];
            work2_ez_y[i] = g2 / kyh[y] + ayh[y] * w;
            m_lambda_ez_y[i] = w;
            const T wd = g2d + byh[y] * dm_lambda_ez_y[i];
            work2_dEz_y[i] = g2d / kyh[y] + ayh[y] * wd;
            dm_lambda_ez_y[i] = wd;
        } else {
            work2_ez_y[i] = g2;
            work2_dEz_y[i] = g2d;
        }
    } else {
        work2_ez_y[i] = (T)0;
        work2_dEz_y[i] = (T)0;
    }
    // dEz_dx feeds Hy with -cq
    if (x < nx - fd_pad_x0) {
        const T g2 = -cq_val * lam_hy[i] - dcq_val * lam_dhy[i];
        const T g2d = -cq_val * lam_dhy[i];
        if (x < pml_x0 || x >= pml_x1h) {
            const T w = g2 + bxh[x] * m_lambda_ez_x[i];
            work2_ez_x[i] = g2 / kxh[x] + axh[x] * w;
            m_lambda_ez_x[i] = w;
            const T wd = g2d + bxh[x] * dm_lambda_ez_x[i];
            work2_dEz_x[i] = g2d / kxh[x] + axh[x] * wd;
            dm_lambda_ez_x[i] = wd;
        } else {
            work2_ez_x[i] = g2;
            work2_dEz_x[i] = g2d;
        }
    } else {
        work2_ez_x[i] = (T)0;
        work2_dEz_x[i] = (T)0;
    }
    // dEx_dz feeds Hy with +cq
    if (z < nz - fd_pad_z0) {
        const T g2 = cq_val * lam_hy[i] + dcq_val * lam_dhy[i];
        const T g2d = cq_val * lam_dhy[i];
        if (z < pml_z0 || z >= pml_z1h) {
            const T w = g2 + bzh[z] * m_lambda_ex_z[i];
            work2_ex_z[i] = g2 / kzh[z] + azh[z] * w;
            m_lambda_ex_z[i] = w;
            const T wd = g2d + bzh[z] * dm_lambda_ex_z[i];
            work2_dEx_z[i] = g2d / kzh[z] + azh[z] * wd;
            dm_lambda_ex_z[i] = wd;
        } else {
            work2_ex_z[i] = g2;
            work2_dEx_z[i] = g2d;
        }
    } else {
        work2_ex_z[i] = (T)0;
        work2_dEx_z[i] = (T)0;
    }
    // dEx_dy feeds Hz with -cq
    if (y < ny - fd_pad_y0) {
        const T g2 = -cq_val * lam_hz[i] - dcq_val * lam_dhz[i];
        const T g2d = -cq_val * lam_dhz[i];
        if (y < pml_y0 || y >= pml_y1h) {
            const T w = g2 + byh[y] * m_lambda_ex_y[i];
            work2_ex_y[i] = g2 / kyh[y] + ayh[y] * w;
            m_lambda_ex_y[i] = w;
            const T wd = g2d + byh[y] * dm_lambda_ex_y[i];
            work2_dEx_y[i] = g2d / kyh[y] + ayh[y] * wd;
            dm_lambda_ex_y[i] = wd;
        } else {
            work2_ex_y[i] = g2;
            work2_dEx_y[i] = g2d;
        }
    } else {
        work2_ex_y[i] = (T)0;
        work2_dEx_y[i] = (T)0;
    }
    // dEy_dx feeds Hz with +cq
    if (x < nx - fd_pad_x0) {
        const T g2 = cq_val * lam_hz[i] + dcq_val * lam_dhz[i];
        const T g2d = cq_val * lam_dhz[i];
        if (x < pml_x0 || x >= pml_x1h) {
            const T w = g2 + bxh[x] * m_lambda_ey_x[i];
            work2_ey_x[i] = g2 / kxh[x] + axh[x] * w;
            m_lambda_ey_x[i] = w;
            const T wd = g2d + bxh[x] * dm_lambda_ey_x[i];
            work2_dEy_x[i] = g2d / kxh[x] + axh[x] * wd;
            dm_lambda_ey_x[i] = wd;
        } else {
            work2_ey_x[i] = g2;
            work2_dEy_x[i] = g2d;
        }
    } else {
        work2_ey_x[i] = (T)0;
        work2_dEy_x[i] = (T)0;
    }
}

// ==================== backward: cq / dcq model gradients ====================
// Uses the snapshotted PML-modified H-step E-derivatives, so no diff
// recomputation is needed: grad_cq gets the bg and sc contributions,
// grad_dcq only the ``dcq*deriv(E_bg)`` terms.
template <typename T>
__global__ void born_cq_grad_kernel(
    const T* __restrict__ lam_hx, const T* __restrict__ lam_hy,
    const T* __restrict__ lam_hz,
    const T* __restrict__ lam_dhx, const T* __restrict__ lam_dhy,
    const T* __restrict__ lam_dhz,
    const T* __restrict__ dey_dz_store, const T* __restrict__ dez_dy_store,
    const T* __restrict__ dez_dx_store, const T* __restrict__ dex_dz_store,
    const T* __restrict__ dex_dy_store, const T* __restrict__ dey_dx_store,
    const T* __restrict__ ddey_dz_store, const T* __restrict__ ddez_dy_store,
    const T* __restrict__ ddez_dx_store, const T* __restrict__ ddex_dz_store,
    const T* __restrict__ ddex_dy_store, const T* __restrict__ ddey_dx_store,
    T* __restrict__ grad_cq, T* __restrict__ grad_dcq,
    int t, int interval, T scale, int64_t snap_off,
    int n_shots, int nz, int ny, int nx,
    int fd_pad_z0, int fd_pad_z1, int fd_pad_y0, int fd_pad_y1,
    int fd_pad_x0, int fd_pad_x1)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)n_shots * nz * ny * nx;
    if (i >= total)
        return;
    const int x = (int)(i % nx);
    const int y = (int)((i / nx) % ny);
    const int z = (int)((i / ((long)nx * ny)) % nz);
    if (z < fd_pad_z0 || z >= nz - fd_pad_z1 || y < fd_pad_y0 ||
        y >= ny - fd_pad_y1 || x < fd_pad_x0 || x >= nx - fd_pad_x1)
        return;
    const int64_t soff = snap_off + i;
    T term_cq = (T)0;
    T term_dcq = (T)0;
    if (z < nz - fd_pad_z0) {
        const T dyz = dey_dz_store[soff];
        const T ddyz = ddey_dz_store[soff];
        term_cq += -lam_hx[i] * dyz - lam_dhx[i] * ddyz;
        term_dcq += -lam_dhx[i] * dyz;
        const T dxz = dex_dz_store[soff];
        const T ddxz = ddex_dz_store[soff];
        term_cq += lam_hy[i] * dxz + lam_dhy[i] * ddxz;
        term_dcq += lam_dhy[i] * dxz;
    }
    if (y < ny - fd_pad_y0) {
        const T dzy = dez_dy_store[soff];
        const T ddzy = ddez_dy_store[soff];
        term_cq += lam_hx[i] * dzy + lam_dhx[i] * ddzy;
        term_dcq += lam_dhx[i] * dzy;
        const T dxy = dex_dy_store[soff];
        const T ddxy = ddex_dy_store[soff];
        term_cq += -lam_hz[i] * dxy - lam_dhz[i] * ddxy;
        term_dcq += -lam_dhz[i] * dxy;
    }
    if (x < nx - fd_pad_x0) {
        const T dzx = dez_dx_store[soff];
        const T ddzx = ddez_dx_store[soff];
        term_cq += -lam_hy[i] * dzx - lam_dhy[i] * ddzx;
        term_dcq += -lam_dhy[i] * dzx;
        const T dyx = dey_dx_store[soff];
        const T ddyx = ddey_dx_store[soff];
        term_cq += lam_hz[i] * dyx + lam_dhz[i] * ddyx;
        term_dcq += lam_dhz[i] * dyx;
    }
    grad_cq[i] += scale * term_cq;
    grad_dcq[i] += scale * term_dcq;
}

// ==================== backward: transpose of the combined H half-step, stage 2 ====================
// lam_ex/ey/ez and lam_dEx/dEy/dEz += the transposes of the six integer-grid
// H curls (the transpose of diff_int applied to each work2 array, summed per
// E component).
template <typename T>
__global__ void born_adjoint_h_stage2_kernel(
    const T* __restrict__ work2_ey_z, const T* __restrict__ work2_ez_y,
    const T* __restrict__ work2_ez_x, const T* __restrict__ work2_ex_z,
    const T* __restrict__ work2_ex_y, const T* __restrict__ work2_ey_x,
    const T* __restrict__ work2_dEy_z, const T* __restrict__ work2_dEz_y,
    const T* __restrict__ work2_dEz_x, const T* __restrict__ work2_dEx_z,
    const T* __restrict__ work2_dEx_y, const T* __restrict__ work2_dEy_x,
    T* __restrict__ lam_ex, T* __restrict__ lam_ey, T* __restrict__ lam_ez,
    T* __restrict__ lam_dEx, T* __restrict__ lam_dEy, T* __restrict__ lam_dEz,
    const T* __restrict__ c,
    T rdz, T rdy, T rdx,
    int n_shots, int nz, int ny, int nx,
    int fd_pad_z0, int fd_pad_y0, int fd_pad_x0)
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
    T sum_dEy_z = (T)0;
    for (int k = 1; k <= fd_pad_z0; ++k) {
        const T ck = c[k - 1];
        if (z >= k) {
            sum_ey_z += ck * work2_ey_z[i - (long)k * ny_nx];
            sum_dEy_z += ck * work2_dEy_z[i - (long)k * ny_nx];
        }
        if (z + k - 1 < nz) {
            sum_ey_z -= ck * work2_ey_z[i + (long)(k - 1) * ny_nx];
            sum_dEy_z -= ck * work2_dEy_z[i + (long)(k - 1) * ny_nx];
        }
    }
    lam_ey[i] += sum_ey_z * rdz;
    lam_dEy[i] += sum_dEy_z * rdz;
    T sum_ey_x = (T)0;
    T sum_dEy_x = (T)0;
    for (int k = 1; k <= fd_pad_x0; ++k) {
        const T ck = c[k - 1];
        if (x >= k) {
            sum_ey_x += ck * work2_ey_x[i - k];
            sum_dEy_x += ck * work2_dEy_x[i - k];
        }
        if (x + k - 1 < nx) {
            sum_ey_x -= ck * work2_ey_x[i + k - 1];
            sum_dEy_x -= ck * work2_dEy_x[i + k - 1];
        }
    }
    lam_ey[i] += sum_ey_x * rdx;
    lam_dEy[i] += sum_dEy_x * rdx;

    T sum_ex_z = (T)0;
    T sum_dEx_z = (T)0;
    for (int k = 1; k <= fd_pad_z0; ++k) {
        const T ck = c[k - 1];
        if (z >= k) {
            sum_ex_z += ck * work2_ex_z[i - (long)k * ny_nx];
            sum_dEx_z += ck * work2_dEx_z[i - (long)k * ny_nx];
        }
        if (z + k - 1 < nz) {
            sum_ex_z -= ck * work2_ex_z[i + (long)(k - 1) * ny_nx];
            sum_dEx_z -= ck * work2_dEx_z[i + (long)(k - 1) * ny_nx];
        }
    }
    lam_ex[i] += sum_ex_z * rdz;
    lam_dEx[i] += sum_dEx_z * rdz;
    T sum_ex_y = (T)0;
    T sum_dEx_y = (T)0;
    for (int k = 1; k <= fd_pad_y0; ++k) {
        const T ck = c[k - 1];
        if (y >= k) {
            sum_ex_y += ck * work2_ex_y[i - (long)k * nx];
            sum_dEx_y += ck * work2_dEx_y[i - (long)k * nx];
        }
        if (y + k - 1 < ny) {
            sum_ex_y -= ck * work2_ex_y[i + (long)(k - 1) * nx];
            sum_dEx_y -= ck * work2_dEx_y[i + (long)(k - 1) * nx];
        }
    }
    lam_ex[i] += sum_ex_y * rdy;
    lam_dEx[i] += sum_dEx_y * rdy;

    T sum_ez_y = (T)0;
    T sum_dEz_y = (T)0;
    for (int k = 1; k <= fd_pad_y0; ++k) {
        const T ck = c[k - 1];
        if (y >= k) {
            sum_ez_y += ck * work2_ez_y[i - (long)k * nx];
            sum_dEz_y += ck * work2_dEz_y[i - (long)k * nx];
        }
        if (y + k - 1 < ny) {
            sum_ez_y -= ck * work2_ez_y[i + (long)(k - 1) * nx];
            sum_dEz_y -= ck * work2_dEz_y[i + (long)(k - 1) * nx];
        }
    }
    lam_ez[i] += sum_ez_y * rdy;
    lam_dEz[i] += sum_dEz_y * rdy;
    T sum_ez_x = (T)0;
    T sum_dEz_x = (T)0;
    for (int k = 1; k <= fd_pad_x0; ++k) {
        const T ck = c[k - 1];
        if (x >= k) {
            sum_ez_x += ck * work2_ez_x[i - k];
            sum_dEz_x += ck * work2_dEz_x[i - k];
        }
        if (x + k - 1 < nx) {
            sum_ez_x -= ck * work2_ez_x[i + k - 1];
            sum_dEz_x -= ck * work2_dEz_x[i + k - 1];
        }
    }
    lam_ez[i] += sum_ez_x * rdx;
    lam_dEz[i] += sum_dEz_x * rdx;
}

// ---------------- launchers ----------------
#define LAUNCH_FLAT(KERN, T, N, ...)                                           \
    {                                                                          \
        const int64_t _n = (N);                                                \
        if (_n > 0) {                                                          \
            const int threads = 256;                                           \
            const int blocks = (int)((_n + threads - 1) / threads);            \
            KERN<T><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>( \
                __VA_ARGS__);                                                  \
            TORCH_CHECK(cudaGetLastError() == cudaSuccess,                     \
                        "nami em3d_born " #KERN " failed");                    \
        }                                                                      \
    }

#define LAUNCH_SR(KERN, T, NSHOTS, NSR, ...)                                   \
    {                                                                          \
        dim3 block(32, 4);                                                     \
        dim3 grid((int)(((NSHOTS) + 31) / 32), (int)(((NSR) + 3) / 4));        \
        KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(         \
            __VA_ARGS__);                                                      \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,                         \
                    "nami em3d_born " #KERN " failed");                        \
    }

static inline int64_t field_numel(const torch::Tensor& f)
{
    return f.numel();
}

void born_step_h(
    torch::Tensor cq, torch::Tensor dcq,
    torch::Tensor ex, torch::Tensor ey, torch::Tensor ez,
    torch::Tensor dEx, torch::Tensor dEy, torch::Tensor dEz,
    torch::Tensor hx, torch::Tensor hy, torch::Tensor hz,
    torch::Tensor dHx, torch::Tensor dHy, torch::Tensor dHz,
    torch::Tensor m_ey_z, torch::Tensor m_ez_y, torch::Tensor m_ez_x,
    torch::Tensor m_ex_z, torch::Tensor m_ex_y, torch::Tensor m_ey_x,
    torch::Tensor dm_ey_z, torch::Tensor dm_ez_y, torch::Tensor dm_ez_x,
    torch::Tensor dm_ex_z, torch::Tensor dm_ex_y, torch::Tensor dm_ey_x,
    torch::Tensor dey_dz_store, torch::Tensor dez_dy_store,
    torch::Tensor dez_dx_store, torch::Tensor dex_dz_store,
    torch::Tensor dex_dy_store, torch::Tensor dey_dx_store,
    torch::Tensor ddey_dz_store, torch::Tensor ddez_dy_store,
    torch::Tensor ddez_dx_store, torch::Tensor ddex_dz_store,
    torch::Tensor ddex_dy_store, torch::Tensor ddey_dx_store,
    torch::Tensor azh, torch::Tensor bzh, torch::Tensor ayh, torch::Tensor byh,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor kzh, torch::Tensor kyh,
    torch::Tensor kxh,
    torch::Tensor c,
    double rdz, double rdy, double rdx,
    int64_t t, int64_t interval, int64_t snap_off,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t pml_z0, int64_t pml_z1, int64_t pml_y0, int64_t pml_y1,
    int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_z0, int64_t fd_pad_y0, int64_t fd_pad_x0,
    int64_t cq_batched, int64_t dcq_batched, int64_t store)
{
    const int64_t total = field_numel(ex);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(ex.scalar_type(), "em3d_born_born_step_h", [&] {
        LAUNCH_FLAT(born_step_h_kernel, scalar_t, total,
            cq.data_ptr<scalar_t>(), dcq.data_ptr<scalar_t>(),
            ex.data_ptr<scalar_t>(), ey.data_ptr<scalar_t>(), ez.data_ptr<scalar_t>(),
            dEx.data_ptr<scalar_t>(), dEy.data_ptr<scalar_t>(), dEz.data_ptr<scalar_t>(),
            hx.data_ptr<scalar_t>(), hy.data_ptr<scalar_t>(), hz.data_ptr<scalar_t>(),
            dHx.data_ptr<scalar_t>(), dHy.data_ptr<scalar_t>(), dHz.data_ptr<scalar_t>(),
            m_ey_z.data_ptr<scalar_t>(), m_ez_y.data_ptr<scalar_t>(),
            m_ez_x.data_ptr<scalar_t>(), m_ex_z.data_ptr<scalar_t>(),
            m_ex_y.data_ptr<scalar_t>(), m_ey_x.data_ptr<scalar_t>(),
            dm_ey_z.data_ptr<scalar_t>(), dm_ez_y.data_ptr<scalar_t>(),
            dm_ez_x.data_ptr<scalar_t>(), dm_ex_z.data_ptr<scalar_t>(),
            dm_ex_y.data_ptr<scalar_t>(), dm_ey_x.data_ptr<scalar_t>(),
            dey_dz_store.data_ptr<scalar_t>(), dez_dy_store.data_ptr<scalar_t>(),
            dez_dx_store.data_ptr<scalar_t>(), dex_dz_store.data_ptr<scalar_t>(),
            dex_dy_store.data_ptr<scalar_t>(), dey_dx_store.data_ptr<scalar_t>(),
            ddey_dz_store.data_ptr<scalar_t>(), ddez_dy_store.data_ptr<scalar_t>(),
            ddez_dx_store.data_ptr<scalar_t>(), ddex_dz_store.data_ptr<scalar_t>(),
            ddex_dy_store.data_ptr<scalar_t>(), ddey_dx_store.data_ptr<scalar_t>(),
            azh.data_ptr<scalar_t>(), bzh.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), byh.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), bxh.data_ptr<scalar_t>(),
            kzh.data_ptr<scalar_t>(), kyh.data_ptr<scalar_t>(),
            kxh.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdz, (scalar_t)rdy, (scalar_t)rdx,
            (int)t, (int)interval, (int64_t)snap_off,
            (int)n_shots, (int)nz, (int)ny, (int)nx,
            (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1,
            (int)fd_pad_z0, (int)fd_pad_y0, (int)fd_pad_x0,
            (int)cq_batched, (int)dcq_batched, (int)store);
    });
}

void born_step_e(
    torch::Tensor ca, torch::Tensor cb, torch::Tensor dca, torch::Tensor dcb,
    torch::Tensor hx, torch::Tensor hy, torch::Tensor hz,
    torch::Tensor dHx, torch::Tensor dHy, torch::Tensor dHz,
    torch::Tensor ex, torch::Tensor ey, torch::Tensor ez,
    torch::Tensor dEx, torch::Tensor dEy, torch::Tensor dEz,
    torch::Tensor m_hy_z, torch::Tensor m_hz_y, torch::Tensor m_hz_x,
    torch::Tensor m_hx_z, torch::Tensor m_hx_y, torch::Tensor m_hy_x,
    torch::Tensor dm_hy_z, torch::Tensor dm_hz_y, torch::Tensor dm_hz_x,
    torch::Tensor dm_hx_z, torch::Tensor dm_hx_y, torch::Tensor dm_hy_x,
    torch::Tensor ex_store, torch::Tensor ey_store, torch::Tensor ez_store,
    torch::Tensor curl_x_store, torch::Tensor curl_y_store,
    torch::Tensor curl_z_store,
    torch::Tensor dex_store, torch::Tensor dey_store, torch::Tensor dez_store,
    torch::Tensor dcurl_x_store, torch::Tensor dcurl_y_store,
    torch::Tensor dcurl_z_store,
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
    int64_t ca_batched, int64_t cb_batched,
    int64_t dca_batched, int64_t dcb_batched)
{
    const int64_t total = field_numel(ex);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(ex.scalar_type(), "em3d_born_born_step_e", [&] {
        LAUNCH_FLAT(born_step_e_kernel, scalar_t, total,
            ca.data_ptr<scalar_t>(), cb.data_ptr<scalar_t>(),
            dca.data_ptr<scalar_t>(), dcb.data_ptr<scalar_t>(),
            hx.data_ptr<scalar_t>(), hy.data_ptr<scalar_t>(), hz.data_ptr<scalar_t>(),
            dHx.data_ptr<scalar_t>(), dHy.data_ptr<scalar_t>(), dHz.data_ptr<scalar_t>(),
            ex.data_ptr<scalar_t>(), ey.data_ptr<scalar_t>(), ez.data_ptr<scalar_t>(),
            dEx.data_ptr<scalar_t>(), dEy.data_ptr<scalar_t>(), dEz.data_ptr<scalar_t>(),
            m_hy_z.data_ptr<scalar_t>(), m_hz_y.data_ptr<scalar_t>(),
            m_hz_x.data_ptr<scalar_t>(), m_hx_z.data_ptr<scalar_t>(),
            m_hx_y.data_ptr<scalar_t>(), m_hy_x.data_ptr<scalar_t>(),
            dm_hy_z.data_ptr<scalar_t>(), dm_hz_y.data_ptr<scalar_t>(),
            dm_hz_x.data_ptr<scalar_t>(), dm_hx_z.data_ptr<scalar_t>(),
            dm_hx_y.data_ptr<scalar_t>(), dm_hy_x.data_ptr<scalar_t>(),
            ex_store.data_ptr<scalar_t>(), ey_store.data_ptr<scalar_t>(),
            ez_store.data_ptr<scalar_t>(),
            curl_x_store.data_ptr<scalar_t>(), curl_y_store.data_ptr<scalar_t>(),
            curl_z_store.data_ptr<scalar_t>(),
            dex_store.data_ptr<scalar_t>(), dey_store.data_ptr<scalar_t>(),
            dez_store.data_ptr<scalar_t>(),
            dcurl_x_store.data_ptr<scalar_t>(), dcurl_y_store.data_ptr<scalar_t>(),
            dcurl_z_store.data_ptr<scalar_t>(),
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
            (int)fd_pad_z0, (int)fd_pad_z1, (int)fd_pad_y0, (int)fd_pad_y1,
            (int)fd_pad_x0, (int)fd_pad_x1,
            (int)ca_batched, (int)cb_batched,
            (int)dca_batched, (int)dcb_batched);
    });
}

void born_inject(
    torch::Tensor field, torch::Tensor dfield,
    torch::Tensor f_bg, torch::Tensor f_sc, torch::Tensor src_i,
    int64_t t, int64_t n_shots, int64_t n_src, int64_t shot_numel)
{
    AT_DISPATCH_FLOATING_TYPES(field.scalar_type(), "em3d_born_born_inject", [&] {
        LAUNCH_SR(born_inject_kernel, scalar_t, (int)n_shots, (int)n_src,
            field.data_ptr<scalar_t>(), dfield.data_ptr<scalar_t>(),
            f_bg.data_ptr<scalar_t>(), f_sc.data_ptr<scalar_t>(),
            src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)shot_numel);
    });
}

void born_record(
    torch::Tensor field, torch::Tensor r, torch::Tensor rec_i,
    int64_t t, int64_t n_shots, int64_t n_rec, int64_t shot_numel)
{
    AT_DISPATCH_FLOATING_TYPES(field.scalar_type(), "em3d_born_born_record", [&] {
        LAUNCH_SR(born_record_kernel, scalar_t, (int)n_shots, (int)n_rec,
            field.data_ptr<scalar_t>(), r.data_ptr<scalar_t>(),
            rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)shot_numel);
    });
}

void born_record_grad_r(
    torch::Tensor lam_field, torch::Tensor grad_r, torch::Tensor rec_i,
    int64_t t, int64_t n_shots, int64_t n_rec, int64_t shot_numel)
{
    AT_DISPATCH_FLOATING_TYPES(lam_field.scalar_type(), "em3d_born_born_record_grad_r", [&] {
        LAUNCH_SR(born_record_grad_r_kernel, scalar_t, (int)n_shots, (int)n_rec,
            lam_field.data_ptr<scalar_t>(), grad_r.data_ptr<scalar_t>(),
            rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)shot_numel);
    });
}

void born_record_grad_f(
    torch::Tensor lam_field, torch::Tensor lam_dfield,
    torch::Tensor grad_f_bg, torch::Tensor grad_f_sc, torch::Tensor src_i,
    int64_t t, int64_t n_shots, int64_t n_src, int64_t shot_numel)
{
    AT_DISPATCH_FLOATING_TYPES(lam_field.scalar_type(), "em3d_born_born_record_grad_f", [&] {
        LAUNCH_SR(born_record_grad_f_kernel, scalar_t, (int)n_shots, (int)n_src,
            lam_field.data_ptr<scalar_t>(), lam_dfield.data_ptr<scalar_t>(),
            grad_f_bg.data_ptr<scalar_t>(), grad_f_sc.data_ptr<scalar_t>(),
            src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)shot_numel);
    });
}

void born_coeff_grad(
    torch::Tensor lam_ex, torch::Tensor lam_ey, torch::Tensor lam_ez,
    torch::Tensor lam_dEx, torch::Tensor lam_dEy, torch::Tensor lam_dEz,
    torch::Tensor ex_store, torch::Tensor ey_store, torch::Tensor ez_store,
    torch::Tensor curl_x_store, torch::Tensor curl_y_store,
    torch::Tensor curl_z_store,
    torch::Tensor dex_store, torch::Tensor dey_store, torch::Tensor dez_store,
    torch::Tensor dcurl_x_store, torch::Tensor dcurl_y_store,
    torch::Tensor dcurl_z_store,
    torch::Tensor grad_ca, torch::Tensor grad_cb,
    torch::Tensor grad_dca, torch::Tensor grad_dcb,
    double scale, int64_t snap_off,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t fd_pad_z0, int64_t fd_pad_z1, int64_t fd_pad_y0, int64_t fd_pad_y1,
    int64_t fd_pad_x0, int64_t fd_pad_x1)
{
    const int64_t total = field_numel(lam_ex);
    AT_DISPATCH_FLOATING_TYPES(lam_ex.scalar_type(), "em3d_born_born_coeff_grad", [&] {
        LAUNCH_FLAT(born_coeff_grad_kernel, scalar_t, total,
            lam_ex.data_ptr<scalar_t>(), lam_ey.data_ptr<scalar_t>(),
            lam_ez.data_ptr<scalar_t>(),
            lam_dEx.data_ptr<scalar_t>(), lam_dEy.data_ptr<scalar_t>(),
            lam_dEz.data_ptr<scalar_t>(),
            ex_store.data_ptr<scalar_t>(), ey_store.data_ptr<scalar_t>(),
            ez_store.data_ptr<scalar_t>(),
            curl_x_store.data_ptr<scalar_t>(), curl_y_store.data_ptr<scalar_t>(),
            curl_z_store.data_ptr<scalar_t>(),
            dex_store.data_ptr<scalar_t>(), dey_store.data_ptr<scalar_t>(),
            dez_store.data_ptr<scalar_t>(),
            dcurl_x_store.data_ptr<scalar_t>(), dcurl_y_store.data_ptr<scalar_t>(),
            dcurl_z_store.data_ptr<scalar_t>(),
            grad_ca.data_ptr<scalar_t>(), grad_cb.data_ptr<scalar_t>(),
            grad_dca.data_ptr<scalar_t>(), grad_dcb.data_ptr<scalar_t>(),
            (scalar_t)scale, (int64_t)snap_off,
            (int)n_shots, (int)nz, (int)ny, (int)nx,
            (int)fd_pad_z0, (int)fd_pad_z1, (int)fd_pad_y0, (int)fd_pad_y1,
            (int)fd_pad_x0, (int)fd_pad_x1);
    });
}

void born_adjoint_e_stage1(
    torch::Tensor ca, torch::Tensor cb, torch::Tensor dca, torch::Tensor dcb,
    torch::Tensor lam_ex, torch::Tensor lam_ey, torch::Tensor lam_ez,
    torch::Tensor lam_dEx, torch::Tensor lam_dEy, torch::Tensor lam_dEz,
    torch::Tensor m_lambda_hy_z, torch::Tensor m_lambda_hz_y,
    torch::Tensor m_lambda_hz_x, torch::Tensor m_lambda_hx_z,
    torch::Tensor m_lambda_hx_y, torch::Tensor m_lambda_hy_x,
    torch::Tensor dm_lambda_hy_z, torch::Tensor dm_lambda_hz_y,
    torch::Tensor dm_lambda_hz_x, torch::Tensor dm_lambda_hx_z,
    torch::Tensor dm_lambda_hx_y, torch::Tensor dm_lambda_hy_x,
    torch::Tensor work_hy_z, torch::Tensor work_hz_y,
    torch::Tensor work_hz_x, torch::Tensor work_hx_z,
    torch::Tensor work_hx_y, torch::Tensor work_hy_x,
    torch::Tensor work_dhy_z, torch::Tensor work_dhz_y,
    torch::Tensor work_dhz_x, torch::Tensor work_dhx_z,
    torch::Tensor work_dhx_y, torch::Tensor work_dhy_x,
    torch::Tensor az, torch::Tensor bz, torch::Tensor ay, torch::Tensor by,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor kz, torch::Tensor ky,
    torch::Tensor kx,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t pml_z0, int64_t pml_z1, int64_t pml_y0, int64_t pml_y1,
    int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_z0, int64_t fd_pad_z1, int64_t fd_pad_y0, int64_t fd_pad_y1,
    int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t ca_batched, int64_t cb_batched,
    int64_t dca_batched, int64_t dcb_batched)
{
    const int64_t total = field_numel(lam_ex);
    AT_DISPATCH_FLOATING_TYPES(lam_ex.scalar_type(), "em3d_born_born_adjoint_e_stage1", [&] {
        LAUNCH_FLAT(born_adjoint_e_stage1_kernel, scalar_t, total,
            ca.data_ptr<scalar_t>(), cb.data_ptr<scalar_t>(),
            dca.data_ptr<scalar_t>(), dcb.data_ptr<scalar_t>(),
            lam_ex.data_ptr<scalar_t>(), lam_ey.data_ptr<scalar_t>(),
            lam_ez.data_ptr<scalar_t>(),
            lam_dEx.data_ptr<scalar_t>(), lam_dEy.data_ptr<scalar_t>(),
            lam_dEz.data_ptr<scalar_t>(),
            m_lambda_hy_z.data_ptr<scalar_t>(), m_lambda_hz_y.data_ptr<scalar_t>(),
            m_lambda_hz_x.data_ptr<scalar_t>(), m_lambda_hx_z.data_ptr<scalar_t>(),
            m_lambda_hx_y.data_ptr<scalar_t>(), m_lambda_hy_x.data_ptr<scalar_t>(),
            dm_lambda_hy_z.data_ptr<scalar_t>(), dm_lambda_hz_y.data_ptr<scalar_t>(),
            dm_lambda_hz_x.data_ptr<scalar_t>(), dm_lambda_hx_z.data_ptr<scalar_t>(),
            dm_lambda_hx_y.data_ptr<scalar_t>(), dm_lambda_hy_x.data_ptr<scalar_t>(),
            work_hy_z.data_ptr<scalar_t>(), work_hz_y.data_ptr<scalar_t>(),
            work_hz_x.data_ptr<scalar_t>(), work_hx_z.data_ptr<scalar_t>(),
            work_hx_y.data_ptr<scalar_t>(), work_hy_x.data_ptr<scalar_t>(),
            work_dhy_z.data_ptr<scalar_t>(), work_dhz_y.data_ptr<scalar_t>(),
            work_dhz_x.data_ptr<scalar_t>(), work_dhx_z.data_ptr<scalar_t>(),
            work_dhx_y.data_ptr<scalar_t>(), work_dhy_x.data_ptr<scalar_t>(),
            az.data_ptr<scalar_t>(), bz.data_ptr<scalar_t>(),
            ay.data_ptr<scalar_t>(), by.data_ptr<scalar_t>(),
            ax.data_ptr<scalar_t>(), bx.data_ptr<scalar_t>(),
            kz.data_ptr<scalar_t>(), ky.data_ptr<scalar_t>(),
            kx.data_ptr<scalar_t>(),
            (int)n_shots, (int)nz, (int)ny, (int)nx,
            (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1,
            (int)fd_pad_z0, (int)fd_pad_z1, (int)fd_pad_y0, (int)fd_pad_y1,
            (int)fd_pad_x0, (int)fd_pad_x1,
            (int)ca_batched, (int)cb_batched,
            (int)dca_batched, (int)dcb_batched);
    });
}

void born_adjoint_e_stage2(
    torch::Tensor work_hy_z, torch::Tensor work_hz_y,
    torch::Tensor work_hz_x, torch::Tensor work_hx_z,
    torch::Tensor work_hx_y, torch::Tensor work_hy_x,
    torch::Tensor work_dhy_z, torch::Tensor work_dhz_y,
    torch::Tensor work_dhz_x, torch::Tensor work_dhx_z,
    torch::Tensor work_dhx_y, torch::Tensor work_dhy_x,
    torch::Tensor lam_hy, torch::Tensor lam_hz, torch::Tensor lam_hx,
    torch::Tensor lam_dhy, torch::Tensor lam_dhz, torch::Tensor lam_dhx,
    torch::Tensor c,
    double rdz, double rdy, double rdx,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t fd_pad_z0, int64_t fd_pad_y0, int64_t fd_pad_x0)
{
    const int64_t total = field_numel(lam_hy);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(lam_hy.scalar_type(), "em3d_born_born_adjoint_e_stage2", [&] {
        LAUNCH_FLAT(born_adjoint_e_stage2_kernel, scalar_t, total,
            work_hy_z.data_ptr<scalar_t>(), work_hz_y.data_ptr<scalar_t>(),
            work_hz_x.data_ptr<scalar_t>(), work_hx_z.data_ptr<scalar_t>(),
            work_hx_y.data_ptr<scalar_t>(), work_hy_x.data_ptr<scalar_t>(),
            work_dhy_z.data_ptr<scalar_t>(), work_dhz_y.data_ptr<scalar_t>(),
            work_dhz_x.data_ptr<scalar_t>(), work_dhx_z.data_ptr<scalar_t>(),
            work_dhx_y.data_ptr<scalar_t>(), work_dhy_x.data_ptr<scalar_t>(),
            lam_hy.data_ptr<scalar_t>(), lam_hz.data_ptr<scalar_t>(),
            lam_hx.data_ptr<scalar_t>(),
            lam_dhy.data_ptr<scalar_t>(), lam_dhz.data_ptr<scalar_t>(),
            lam_dhx.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdz, (scalar_t)rdy, (scalar_t)rdx,
            (int)n_shots, (int)nz, (int)ny, (int)nx,
            (int)fd_pad_z0, (int)fd_pad_y0, (int)fd_pad_x0);
    });
}

void born_adjoint_h_stage1(
    torch::Tensor cq, torch::Tensor dcq,
    torch::Tensor lam_hx, torch::Tensor lam_hy, torch::Tensor lam_hz,
    torch::Tensor lam_dhx, torch::Tensor lam_dhy, torch::Tensor lam_dhz,
    torch::Tensor m_lambda_ey_z, torch::Tensor m_lambda_ez_y,
    torch::Tensor m_lambda_ez_x, torch::Tensor m_lambda_ex_z,
    torch::Tensor m_lambda_ex_y, torch::Tensor m_lambda_ey_x,
    torch::Tensor dm_lambda_ey_z, torch::Tensor dm_lambda_ez_y,
    torch::Tensor dm_lambda_ez_x, torch::Tensor dm_lambda_ex_z,
    torch::Tensor dm_lambda_ex_y, torch::Tensor dm_lambda_ey_x,
    torch::Tensor work2_ey_z, torch::Tensor work2_ez_y,
    torch::Tensor work2_ez_x, torch::Tensor work2_ex_z,
    torch::Tensor work2_ex_y, torch::Tensor work2_ey_x,
    torch::Tensor work2_dEy_z, torch::Tensor work2_dEz_y,
    torch::Tensor work2_dEz_x, torch::Tensor work2_dEx_z,
    torch::Tensor work2_dEx_y, torch::Tensor work2_dEy_x,
    torch::Tensor azh, torch::Tensor bzh, torch::Tensor ayh, torch::Tensor byh,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor kzh, torch::Tensor kyh,
    torch::Tensor kxh,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t pml_z0, int64_t pml_z1, int64_t pml_y0, int64_t pml_y1,
    int64_t pml_x0, int64_t pml_x1,
    int64_t fd_pad_z0, int64_t fd_pad_z1, int64_t fd_pad_y0, int64_t fd_pad_y1,
    int64_t fd_pad_x0, int64_t fd_pad_x1,
    int64_t cq_batched, int64_t dcq_batched)
{
    const int64_t total = field_numel(lam_hx);
    AT_DISPATCH_FLOATING_TYPES(lam_hx.scalar_type(), "em3d_born_born_adjoint_h_stage1", [&] {
        LAUNCH_FLAT(born_adjoint_h_stage1_kernel, scalar_t, total,
            cq.data_ptr<scalar_t>(), dcq.data_ptr<scalar_t>(),
            lam_hx.data_ptr<scalar_t>(), lam_hy.data_ptr<scalar_t>(),
            lam_hz.data_ptr<scalar_t>(),
            lam_dhx.data_ptr<scalar_t>(), lam_dhy.data_ptr<scalar_t>(),
            lam_dhz.data_ptr<scalar_t>(),
            m_lambda_ey_z.data_ptr<scalar_t>(), m_lambda_ez_y.data_ptr<scalar_t>(),
            m_lambda_ez_x.data_ptr<scalar_t>(), m_lambda_ex_z.data_ptr<scalar_t>(),
            m_lambda_ex_y.data_ptr<scalar_t>(), m_lambda_ey_x.data_ptr<scalar_t>(),
            dm_lambda_ey_z.data_ptr<scalar_t>(), dm_lambda_ez_y.data_ptr<scalar_t>(),
            dm_lambda_ez_x.data_ptr<scalar_t>(), dm_lambda_ex_z.data_ptr<scalar_t>(),
            dm_lambda_ex_y.data_ptr<scalar_t>(), dm_lambda_ey_x.data_ptr<scalar_t>(),
            work2_ey_z.data_ptr<scalar_t>(), work2_ez_y.data_ptr<scalar_t>(),
            work2_ez_x.data_ptr<scalar_t>(), work2_ex_z.data_ptr<scalar_t>(),
            work2_ex_y.data_ptr<scalar_t>(), work2_ey_x.data_ptr<scalar_t>(),
            work2_dEy_z.data_ptr<scalar_t>(), work2_dEz_y.data_ptr<scalar_t>(),
            work2_dEz_x.data_ptr<scalar_t>(), work2_dEx_z.data_ptr<scalar_t>(),
            work2_dEx_y.data_ptr<scalar_t>(), work2_dEy_x.data_ptr<scalar_t>(),
            azh.data_ptr<scalar_t>(), bzh.data_ptr<scalar_t>(),
            ayh.data_ptr<scalar_t>(), byh.data_ptr<scalar_t>(),
            axh.data_ptr<scalar_t>(), bxh.data_ptr<scalar_t>(),
            kzh.data_ptr<scalar_t>(), kyh.data_ptr<scalar_t>(),
            kxh.data_ptr<scalar_t>(),
            (int)n_shots, (int)nz, (int)ny, (int)nx,
            (int)pml_z0, (int)pml_z1, (int)pml_y0, (int)pml_y1,
            (int)pml_x0, (int)pml_x1,
            (int)fd_pad_z0, (int)fd_pad_z1, (int)fd_pad_y0, (int)fd_pad_y1,
            (int)fd_pad_x0, (int)fd_pad_x1,
            (int)cq_batched, (int)dcq_batched);
    });
}

void born_cq_grad(
    torch::Tensor lam_hx, torch::Tensor lam_hy, torch::Tensor lam_hz,
    torch::Tensor lam_dhx, torch::Tensor lam_dhy, torch::Tensor lam_dhz,
    torch::Tensor dey_dz_store, torch::Tensor dez_dy_store,
    torch::Tensor dez_dx_store, torch::Tensor dex_dz_store,
    torch::Tensor dex_dy_store, torch::Tensor dey_dx_store,
    torch::Tensor ddey_dz_store, torch::Tensor ddez_dy_store,
    torch::Tensor ddez_dx_store, torch::Tensor ddex_dz_store,
    torch::Tensor ddex_dy_store, torch::Tensor ddey_dx_store,
    torch::Tensor grad_cq, torch::Tensor grad_dcq,
    int64_t t, int64_t interval, double scale, int64_t snap_off,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t fd_pad_z0, int64_t fd_pad_z1, int64_t fd_pad_y0, int64_t fd_pad_y1,
    int64_t fd_pad_x0, int64_t fd_pad_x1)
{
    const int64_t total = field_numel(lam_hx);
    AT_DISPATCH_FLOATING_TYPES(lam_hx.scalar_type(), "em3d_born_born_cq_grad", [&] {
        LAUNCH_FLAT(born_cq_grad_kernel, scalar_t, total,
            lam_hx.data_ptr<scalar_t>(), lam_hy.data_ptr<scalar_t>(),
            lam_hz.data_ptr<scalar_t>(),
            lam_dhx.data_ptr<scalar_t>(), lam_dhy.data_ptr<scalar_t>(),
            lam_dhz.data_ptr<scalar_t>(),
            dey_dz_store.data_ptr<scalar_t>(), dez_dy_store.data_ptr<scalar_t>(),
            dez_dx_store.data_ptr<scalar_t>(), dex_dz_store.data_ptr<scalar_t>(),
            dex_dy_store.data_ptr<scalar_t>(), dey_dx_store.data_ptr<scalar_t>(),
            ddey_dz_store.data_ptr<scalar_t>(), ddez_dy_store.data_ptr<scalar_t>(),
            ddez_dx_store.data_ptr<scalar_t>(), ddex_dz_store.data_ptr<scalar_t>(),
            ddex_dy_store.data_ptr<scalar_t>(), ddey_dx_store.data_ptr<scalar_t>(),
            grad_cq.data_ptr<scalar_t>(), grad_dcq.data_ptr<scalar_t>(),
            (int)t, (int)interval, (scalar_t)scale, (int64_t)snap_off,
            (int)n_shots, (int)nz, (int)ny, (int)nx,
            (int)fd_pad_z0, (int)fd_pad_z1, (int)fd_pad_y0, (int)fd_pad_y1,
            (int)fd_pad_x0, (int)fd_pad_x1);
    });
}

void born_adjoint_h_stage2(
    torch::Tensor work2_ey_z, torch::Tensor work2_ez_y,
    torch::Tensor work2_ez_x, torch::Tensor work2_ex_z,
    torch::Tensor work2_ex_y, torch::Tensor work2_ey_x,
    torch::Tensor work2_dEy_z, torch::Tensor work2_dEz_y,
    torch::Tensor work2_dEz_x, torch::Tensor work2_dEx_z,
    torch::Tensor work2_dEx_y, torch::Tensor work2_dEy_x,
    torch::Tensor lam_ex, torch::Tensor lam_ey, torch::Tensor lam_ez,
    torch::Tensor lam_dEx, torch::Tensor lam_dEy, torch::Tensor lam_dEz,
    torch::Tensor c,
    double rdz, double rdy, double rdx,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t fd_pad_z0, int64_t fd_pad_y0, int64_t fd_pad_x0)
{
    const int64_t total = field_numel(lam_ey);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(lam_ey.scalar_type(), "em3d_born_born_adjoint_h_stage2", [&] {
        LAUNCH_FLAT(born_adjoint_h_stage2_kernel, scalar_t, total,
            work2_ey_z.data_ptr<scalar_t>(), work2_ez_y.data_ptr<scalar_t>(),
            work2_ez_x.data_ptr<scalar_t>(), work2_ex_z.data_ptr<scalar_t>(),
            work2_ex_y.data_ptr<scalar_t>(), work2_ey_x.data_ptr<scalar_t>(),
            work2_dEy_z.data_ptr<scalar_t>(), work2_dEz_y.data_ptr<scalar_t>(),
            work2_dEz_x.data_ptr<scalar_t>(), work2_dEx_z.data_ptr<scalar_t>(),
            work2_dEx_y.data_ptr<scalar_t>(), work2_dEy_x.data_ptr<scalar_t>(),
            lam_ex.data_ptr<scalar_t>(), lam_ey.data_ptr<scalar_t>(),
            lam_ez.data_ptr<scalar_t>(),
            lam_dEx.data_ptr<scalar_t>(), lam_dEy.data_ptr<scalar_t>(),
            lam_dEz.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            (scalar_t)rdz, (scalar_t)rdy, (scalar_t)rdx,
            (int)n_shots, (int)nz, (int)ny, (int)nx,
            (int)fd_pad_z0, (int)fd_pad_y0, (int)fd_pad_x0);
    });
}

// ---------------- pybind11 bindings ----------------
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
}
