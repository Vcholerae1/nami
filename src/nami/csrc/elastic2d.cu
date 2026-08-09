#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <pybind11/stl.h>
#include "storage.h"

#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------- coefficient-driven staggered difference operators ----------------
// ``c`` is the staggered-grid first-derivative coefficient array (max radius
// 4, zero-padded for lower orders; staggered-grid convention).  The stencil
// radius is a compile-time template parameter FD_PAD (== accuracy // 2 ==
// fd_pad_y0/fd_pad_x0), so the loops fully unroll with constant offsets; the
// coefficients and the summation order are unchanged, so results are bitwise
// identical to the old runtime-radius loops (only the bound is constant).
// diff_int_*: derivative at a half-integer grid point of a field stored at
// integer grid points: sum_k c[k-1]*(u[i+k]-u[i-k+1]).
// diff_half_*: derivative at an integer grid point of a field stored at
// half-integer grid points: sum_k c[k-1]*(u[i+k-1]-u[i-k]).
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
__device__ __forceinline__ T diff_half_x(const T* __restrict__ u, long off,
                                         const T* __restrict__ c, T rdx)
{
    T d = (T)0;
#pragma unroll
    for (int k = 1; k <= FD_PAD; ++k)
        d += c[k - 1] * (u[off + k - 1] - u[off - k]);
    return d * rdx;
}

// ---------------- backward: adjoint stress ----------------
// Transpose of the forward velocity update: propagates the adjoint
// velocities back into the adjoint stresses and the m_v* memory variables,
// and evaluates the imaging condition for lamb/mu/mu_yx.
template <typename T, int FD_PAD>
__global__ void step_adjoint_stress_kernel(
    const T* __restrict__ lamb, const T* __restrict__ mu, const T* __restrict__ mu_yx,
    const T* __restrict__ buoyancy_y, const T* __restrict__ buoyancy_x,
    const T* __restrict__ l_vy, const T* __restrict__ l_vx,
    T* __restrict__ l_syy, T* __restrict__ l_sxx, T* __restrict__ l_sxy,
    T* __restrict__ m_vyy, T* __restrict__ m_vxx,
    T* __restrict__ m_vxy, T* __restrict__ m_vyx,
    const T* __restrict__ m_syyy_old, const T* __restrict__ m_syx_old,
    const T* __restrict__ m_syxy_old, const T* __restrict__ m_syxx_old,
    T* __restrict__ grad_lamb, T* __restrict__ grad_mu, T* __restrict__ grad_mu_yx,
    const T* __restrict__ dvydy_store, const T* __restrict__ dvxdx_store,
    const T* __restrict__ dvydx_plus_dvxdy_store,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ c,
    T rdy, T rdx, T dtv, T scale,
    int t, int interval, int64_t snap_off, int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int model_batched)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= ny || x >= nx)
        return;
    const long ny_nx = (long)ny * nx;
    const long off_s = (long)s * ny_nx;
    const int s_m = model_batched ? s : 0;
    const T* lamb_s = lamb + (long)s_m * ny_nx;
    const T* mu_s = mu + (long)s_m * ny_nx;
    const T* mu_yx_s = mu_yx + (long)s_m * ny_nx;
    const T* by_s = buoyancy_y + (long)s_m * ny_nx;
    const T* bx_s = buoyancy_x + (long)s_m * ny_nx;
    const bool interior = (y >= pml_y0 && y < pml_y1 && x >= pml_x0 && x < pml_x1);

    // sigmaii: y in [FD_PAD, ny-(FD_PAD-1)), x in [FD_PAD, nx-(FD_PAD-1))
    if (y >= FD_PAD && x >= FD_PAD && y < ny - (FD_PAD - 1) &&
        x < nx - (FD_PAD - 1)) {
        const long off = off_s + (long)y * nx + x;
        const T sxx_v = l_sxx[off];
        const T syy_v = l_syy[off];
        if (t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            const T dx_store = dvxdx_store[soff];
            const T dy_store = dvydy_store[soff];
            grad_lamb[off] += scale * (sxx_v + syy_v)
                              * (dx_store + dy_store);
            grad_mu[off] += scale * (T)2
                            * (sxx_v * dx_store + syy_v * dy_store);
        }
        const T lam = lamb_s[(long)y * nx + x];
        const T l2m = lam + (T)2 * mu_s[(long)y * nx + x];
        if (!interior) {
            m_vyy[off] = l2m * dtv * ay[y] * syy_v + lam * dtv * ay[y] * sxx_v
                       + ay[y] * m_vyy[off];
            m_vxx[off] = l2m * dtv * ax[x] * sxx_v + lam * dtv * ax[x] * syy_v
                       + ax[x] * m_vxx[off];
        }
        // l_syy: -diff_half_y(W_yy), W_yy = dtv*(1+byh)*by_s*l_vy + byh*m_syyy_old
        T dy = (T)0;
        if (interior) {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - (long)k * nx;
                const long off_p = off + (long)(k - 1) * nx;
                dy += c[k - 1] * (dtv * by_s[off_m - off_s] * l_vy[off_m] -
                                  dtv * by_s[off_p - off_s] * l_vy[off_p]);
            }
        } else {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - (long)k * nx;
                const long off_p = off + (long)(k - 1) * nx;
                const int ym = y - k, yp = y + k - 1;
                dy += c[k - 1] *
                      (dtv * ((T)1 + byh[ym]) * by_s[off_m - off_s] * l_vy[off_m] + byh[ym] * m_syyy_old[off_m] -
                       (dtv * ((T)1 + byh[yp]) * by_s[off_p - off_s] * l_vy[off_p] + byh[yp] * m_syyy_old[off_p]));
            }
        }
        l_syy[off] += dy * rdy;
        // l_sxx: -diff_half_x(W_xx'), W_xx' = dtv*(1+bxh)*bx_s*l_vx + bxh*m_syxx_old
        T dx = (T)0;
        if (interior) {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - k;
                const long off_p = off + k - 1;
                dx += c[k - 1] * (dtv * bx_s[off_m - off_s] * l_vx[off_m] -
                                  dtv * bx_s[off_p - off_s] * l_vx[off_p]);
            }
        } else {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - k;
                const long off_p = off + k - 1;
                const int xm = x - k, xp = x + k - 1;
                dx += c[k - 1] *
                      (dtv * ((T)1 + bxh[xm]) * bx_s[off_m - off_s] * l_vx[off_m] + bxh[xm] * m_syxx_old[off_m] -
                       (dtv * ((T)1 + bxh[xp]) * bx_s[off_p - off_s] * l_vx[off_p] + bxh[xp] * m_syxx_old[off_p]));
            }
        }
        l_sxx[off] += dx * rdx;
    }
    // sigmaxy: y in [FD_PAD, ny-FD_PAD), x in [FD_PAD, nx-FD_PAD)
    if (y >= FD_PAD && x >= FD_PAD && y < ny - FD_PAD &&
        x < nx - FD_PAD) {
        const long off = off_s + (long)y * nx + x;
        const T sxy_v = l_sxy[off];
        if (t % interval == 0)
            grad_mu_yx[off] += scale * sxy_v
                * dvydx_plus_dvxdy_store[
                    snap_off + ((long)s * ny + y) * nx + x];
        const T mu_yx_v = mu_yx_s[(long)y * nx + x];
        if (!interior) {
            m_vxy[off] = mu_yx_v * dtv * ayh[y] * sxy_v + ayh[y] * m_vxy[off];
            m_vyx[off] = mu_yx_v * dtv * axh[x] * sxy_v + axh[x] * m_vyx[off];
        }
        // l_sxy y-term: -diff_int_y(W_sxy_y), W_sxy_y = dtv*(1+by)*bx_s*l_vx + by*m_syxy_old
        T dy = (T)0;
        if (interior) {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - (long)(k - 1) * nx;
                const long off_p = off + (long)k * nx;
                dy += c[k - 1] * (dtv * bx_s[off_m - off_s] * l_vx[off_m] -
                                  dtv * bx_s[off_p - off_s] * l_vx[off_p]);
            }
        } else {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - (long)(k - 1) * nx;
                const long off_p = off + (long)k * nx;
                const int ym = y - (k - 1), yp = y + k;
                dy += c[k - 1] *
                      (dtv * ((T)1 + by[ym]) * bx_s[off_m - off_s] * l_vx[off_m] + by[ym] * m_syxy_old[off_m] -
                       (dtv * ((T)1 + by[yp]) * bx_s[off_p - off_s] * l_vx[off_p] + by[yp] * m_syxy_old[off_p]));
            }
        }
        l_sxy[off] += dy * rdy;
        // l_sxy x-term: -diff_int_x(W_sxy_x), W_sxy_x = dtv*(1+bx)*by_s*l_vy + bx*m_syx_old
        T dx = (T)0;
        if (interior) {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - k + 1;
                const long off_p = off + k;
                dx += c[k - 1] * (dtv * by_s[off_m - off_s] * l_vy[off_m] -
                                  dtv * by_s[off_p - off_s] * l_vy[off_p]);
            }
        } else {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - k + 1;
                const long off_p = off + k;
                const int xm = x - k + 1, xp = x + k;
                dx += c[k - 1] *
                      (dtv * ((T)1 + bx[xm]) * by_s[off_m - off_s] * l_vy[off_m] + bx[xm] * m_syx_old[off_m] -
                       (dtv * ((T)1 + bx[xp]) * by_s[off_p - off_s] * l_vy[off_p] + bx[xp] * m_syx_old[off_p]));
            }
        }
        l_sxy[off] += dx * rdx;
    }
}

// ---------------- backward: adjoint velocity ----------------
// Transpose of the forward stress update: propagates the adjoint stresses
// back into the adjoint velocities and the m_sigma* adjoint memories, and
// evaluates the imaging condition for the buoyancy.  The m_sigma* adjoint
// memories alternate between the two buffers (the C code's _t/_n trick);
// interior points skip them (zero profiles).
template <typename T, int FD_PAD>
__global__ void step_adjoint_velocity_kernel(
    const T* __restrict__ lamb, const T* __restrict__ mu, const T* __restrict__ mu_yx,
    const T* __restrict__ buoyancy_y, const T* __restrict__ buoyancy_x,
    T* __restrict__ l_vy, T* __restrict__ l_vx,
    const T* __restrict__ l_syy, const T* __restrict__ l_sxx, const T* __restrict__ l_sxy,
    const T* __restrict__ m_vyy, const T* __restrict__ m_vxx,
    const T* __restrict__ m_vxy, const T* __restrict__ m_vyx,
    const T* __restrict__ m_syyy_old, const T* __restrict__ m_syx_old,
    const T* __restrict__ m_syxy_old, const T* __restrict__ m_syxx_old,
    T* __restrict__ m_syyy_new, T* __restrict__ m_syx_new,
    T* __restrict__ m_syxy_new, T* __restrict__ m_syxx_new,
    T* __restrict__ grad_buoyancy_y, T* __restrict__ grad_buoyancy_x,
    const T* __restrict__ dvydbuoyancy_store, const T* __restrict__ dvxdbuoyancy_store,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ c,
    T rdy, T rdx, T dtv, T scale,
    int t, int interval, int64_t snap_off, int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int model_batched)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= ny || x >= nx)
        return;
    const long ny_nx = (long)ny * nx;
    const long off_s = (long)s * ny_nx;
    const int s_m = model_batched ? s : 0;
    const T* lamb_s = lamb + (long)s_m * ny_nx;
    const T* mu_s = mu + (long)s_m * ny_nx;
    const T* mu_yx_s = mu_yx + (long)s_m * ny_nx;
    const T* by_s = buoyancy_y + (long)s_m * ny_nx;
    const T* bx_s = buoyancy_x + (long)s_m * ny_nx;
    const bool interior = (y >= pml_y0 && y < pml_y1 && x >= pml_x0 && x < pml_x1);

    // vy: y in [FD_PAD, ny-FD_PAD), x in [FD_PAD, nx-(FD_PAD-1))
    if (y >= FD_PAD && y < ny - FD_PAD && x >= FD_PAD &&
        x < nx - (FD_PAD - 1)) {
        const long off = off_s + (long)y * nx + x;
        // y-term: -diff_int_y(W_y), W_y = dtv*(lam*l_sxx + (lam+2mu)*l_syy)
        // (+ by*m_vyy in the frame; by is the integer-point profile, the
        // forward m_vyy memory lives on integer y points)
        T dy = (T)0;
        if (interior) {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - (long)(k - 1) * nx;
                const long off_p = off + (long)k * nx;
                const T lam_m = lamb_s[off_m - off_s];
                const T l2m_m = lam_m + (T)2 * mu_s[off_m - off_s];
                const T lam_p = lamb_s[off_p - off_s];
                const T l2m_p = lam_p + (T)2 * mu_s[off_p - off_s];
                dy += c[k - 1] *
                      (dtv * (lam_m * l_sxx[off_m] + l2m_m * l_syy[off_m]) -
                       dtv * (lam_p * l_sxx[off_p] + l2m_p * l_syy[off_p]));
            }
        } else {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - (long)(k - 1) * nx;
                const long off_p = off + (long)k * nx;
                const int ym = y - (k - 1), yp = y + k;
                const T lam_m = lamb_s[off_m - off_s];
                const T l2m_m = lam_m + (T)2 * mu_s[off_m - off_s];
                const T lam_p = lamb_s[off_p - off_s];
                const T l2m_p = lam_p + (T)2 * mu_s[off_p - off_s];
                dy += c[k - 1] *
                      (dtv * ((T)1 + by[ym]) * (lam_m * l_sxx[off_m] + l2m_m * l_syy[off_m]) +
                           by[ym] * m_vyy[off_m] -
                       (dtv * ((T)1 + by[yp]) * (lam_p * l_sxx[off_p] + l2m_p * l_syy[off_p]) +
                           by[yp] * m_vyy[off_p]));
            }
        }
        // x-term: -diff_half_x(W_x), W_x = dtv*mu_yx*l_sxy (+ bxh*m_vyx)
        T dx = (T)0;
        if (interior) {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - k;
                const long off_p = off + k - 1;
                dx += c[k - 1] * (dtv * mu_yx_s[off_m - off_s] * l_sxy[off_m] -
                                  dtv * mu_yx_s[off_p - off_s] * l_sxy[off_p]);
            }
        } else {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - k;
                const long off_p = off + k - 1;
                const int xm = x - k, xp = x + k - 1;
                dx += c[k - 1] *
                      (dtv * ((T)1 + bxh[xm]) * mu_yx_s[off_m - off_s] * l_sxy[off_m] + bxh[xm] * m_vyx[off_m] -
                       (dtv * ((T)1 + bxh[xp]) * mu_yx_s[off_p - off_s] * l_sxy[off_p] + bxh[xp] * m_vyx[off_p]));
            }
        }
        l_vy[off] += dy * rdy + dx * rdx;
        const T vy_new = l_vy[off];
        const T b_y = by_s[(long)y * nx + x];
        if (!interior) {
            m_syyy_new[off] = b_y * dtv * ayh[y] * vy_new + ayh[y] * m_syyy_old[off];
            m_syx_new[off] = b_y * dtv * ax[x] * vy_new + ax[x] * m_syx_old[off];
        }
        if (t % interval == 0)
            grad_buoyancy_y[off] += scale * vy_new
                * dvydbuoyancy_store[snap_off + ((long)s * ny + y) * nx + x];
    }
    // vx: y in [FD_PAD, ny-(FD_PAD-1)), x in [FD_PAD, nx-FD_PAD)
    if (y >= FD_PAD && y < ny - (FD_PAD - 1) && x >= FD_PAD &&
        x < nx - FD_PAD) {
        const long off = off_s + (long)y * nx + x;
        // y-term: -diff_half_y(W_x'), W_x' = dtv*mu_yx*l_sxy (+ byh*m_vxy)
        T dy = (T)0;
        if (interior) {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - (long)k * nx;
                const long off_p = off + (long)(k - 1) * nx;
                dy += c[k - 1] * (dtv * mu_yx_s[off_m - off_s] * l_sxy[off_m] -
                                  dtv * mu_yx_s[off_p - off_s] * l_sxy[off_p]);
            }
        } else {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - (long)k * nx;
                const long off_p = off + (long)(k - 1) * nx;
                const int ym = y - k, yp = y + k - 1;
                dy += c[k - 1] *
                      (dtv * ((T)1 + byh[ym]) * mu_yx_s[off_m - off_s] * l_sxy[off_m] + byh[ym] * m_vxy[off_m] -
                       (dtv * ((T)1 + byh[yp]) * mu_yx_s[off_p - off_s] * l_sxy[off_p] + byh[yp] * m_vxy[off_p]));
            }
        }
        // x-term: -diff_int_x(W_xx), W_xx = dtv*((lam+2mu)*l_sxx + lam*l_syy) (+ bx*m_vxx)
        T dx = (T)0;
        if (interior) {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - k + 1;
                const long off_p = off + k;
                const T lam_m = lamb_s[off_m - off_s];
                const T l2m_m = lam_m + (T)2 * mu_s[off_m - off_s];
                const T lam_p = lamb_s[off_p - off_s];
                const T l2m_p = lam_p + (T)2 * mu_s[off_p - off_s];
                dx += c[k - 1] *
                      (dtv * (l2m_m * l_sxx[off_m] + lam_m * l_syy[off_m]) -
                       dtv * (l2m_p * l_sxx[off_p] + lam_p * l_syy[off_p]));
            }
        } else {
            for (int k = 1; k <= FD_PAD; ++k) {
                const long off_m = off - k + 1;
                const long off_p = off + k;
                const int xm = x - k + 1, xp = x + k;
                const T lam_m = lamb_s[off_m - off_s];
                const T l2m_m = lam_m + (T)2 * mu_s[off_m - off_s];
                const T lam_p = lamb_s[off_p - off_s];
                const T l2m_p = lam_p + (T)2 * mu_s[off_p - off_s];
                dx += c[k - 1] *
                      (dtv * ((T)1 + bx[xm]) * (l2m_m * l_sxx[off_m] + lam_m * l_syy[off_m]) + bx[xm] * m_vxx[off_m] -
                       (dtv * ((T)1 + bx[xp]) * (l2m_p * l_sxx[off_p] + lam_p * l_syy[off_p]) + bx[xp] * m_vxx[off_p]));
            }
        }
        l_vx[off] += dy * rdy + dx * rdx;
        const T vx_new = l_vx[off];
        const T b_x = bx_s[(long)y * nx + x];
        if (!interior) {
            m_syxy_new[off] = b_x * dtv * ay[y] * vx_new + ay[y] * m_syxy_old[off];
            m_syxx_new[off] = b_x * dtv * axh[x] * vx_new + axh[x] * m_syxx_old[off];
        }
        if (t % interval == 0)
            grad_buoyancy_x[off] += scale * vx_new
                * dvxdbuoyancy_store[snap_off + ((long)s * ny + y) * nx + x];
    }
}

// ---------------- forward: stress update ----------------
// sigmaii: y in [1, ny), x in [1, nx);  sigmaxy: y in [1, ny-1), x in [1, nx-1)
template <typename T, int FD_PAD>
__global__ void step_forward_stress_kernel(
    const T* __restrict__ vy, const T* __restrict__ vx,
    T* __restrict__ syy, T* __restrict__ sxx, T* __restrict__ sxy,
    T* __restrict__ m_vyy, T* __restrict__ m_vxx,
    T* __restrict__ m_vxy, T* __restrict__ m_vyx,
    const T* __restrict__ lamb, const T* __restrict__ mu, const T* __restrict__ mu_yx,
    T* __restrict__ dvydy_store, T* __restrict__ dvxdx_store,
    T* __restrict__ dvydx_plus_dvxdy_store,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ c,
    T rdy, T rdx, T dtv,
    int t, int interval, int64_t snap_off, int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int model_batched, int store)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= ny || x >= nx)
        return;
    const long ny_nx = (long)ny * nx;
    const long off_s = (long)s * ny_nx;
    const int s_m = model_batched ? s : 0;
    const T* lamb_s = lamb + (long)s_m * ny_nx;
    const T* mu_s = mu + (long)s_m * ny_nx;
    const T* mu_yx_s = mu_yx + (long)s_m * ny_nx;
    const bool interior = (y >= pml_y0 && y < pml_y1 && x >= pml_x0 && x < pml_x1);

    // sigmaii: y in [FD_PAD, ny-(FD_PAD-1)), x in [FD_PAD, nx-(FD_PAD-1))
    if (y >= FD_PAD && x >= FD_PAD && y < ny - (FD_PAD - 1) &&
        x < nx - (FD_PAD - 1)) {
        const long off = off_s + (long)y * nx + x;
        T dvydy = diff_half_y<T, FD_PAD>(vy, off, c, rdy, nx);
        if (!interior) {
            m_vyy[off] = ay[y] * m_vyy[off] + by[y] * dvydy;
            dvydy += m_vyy[off];
        }
        T dvxdx = diff_half_x<T, FD_PAD>(vx, off, c, rdx);
        if (!interior) {
            m_vxx[off] = ax[x] * m_vxx[off] + bx[x] * dvxdx;
            dvxdx += m_vxx[off];
        }
        if (store && t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            dvydy_store[soff] = dtv * dvydy;
            dvxdx_store[soff] = dtv * dvxdx;
        }
        const T ssum = dvydy + dvxdx;
        const T lamb_v = lamb_s[(long)y * nx + x];
        const T mu_v = mu_s[(long)y * nx + x];
        syy[off] += dtv * (lamb_v * ssum + (T)2 * mu_v * dvydy);
        sxx[off] += dtv * (lamb_v * ssum + (T)2 * mu_v * dvxdx);
    }
    // sigmaxy: y in [FD_PAD, ny-FD_PAD), x in [FD_PAD, nx-FD_PAD)
    if (y >= FD_PAD && x >= FD_PAD && y < ny - FD_PAD &&
        x < nx - FD_PAD) {
        const long off = off_s + (long)y * nx + x;
        T dvxdy = diff_int_y<T, FD_PAD>(vx, off, c, rdy, nx);
        if (!interior) {
            m_vxy[off] = ayh[y] * m_vxy[off] + byh[y] * dvxdy;
            dvxdy += m_vxy[off];
        }
        T dvydx = diff_int_x<T, FD_PAD>(vy, off, c, rdx);
        if (!interior) {
            m_vyx[off] = axh[x] * m_vyx[off] + bxh[x] * dvydx;
            dvydx += m_vyx[off];
        }
        const T w_sum = dvydx + dvxdy;
        if (store && t % interval == 0)
            dvydx_plus_dvxdy_store[snap_off + ((long)s * ny + y) * nx + x]
                = dtv * w_sum;
        sxy[off] += dtv * mu_yx_s[(long)y * nx + x] * w_sum;
    }
}

// ---------------- forward: velocity update ----------------
// vy: y in [1, ny-1), x in [1, nx);  vx: y in [1, ny), x in [1, nx-1)
// Single full-grid kernel; the C-PML memory updates (a = b = 0) vanish in
// the interior box [pml_y0, pml_y1) x [pml_x0, pml_x1), so they are skipped
// there while the arithmetic stays verbatim (bit-identical to the previous
// interior/frame split).
template <typename T, int FD_PAD>
__global__ void step_forward_velocity_kernel(
    T* __restrict__ vy, T* __restrict__ vx,
    const T* __restrict__ syy, const T* __restrict__ sxx, const T* __restrict__ sxy,
    T* __restrict__ m_sigmayyy, T* __restrict__ m_sigmaxyx,
    T* __restrict__ m_sigmaxyy, T* __restrict__ m_sigmaxxx,
    const T* __restrict__ buoyancy_y, const T* __restrict__ buoyancy_x,
    T* __restrict__ dvydbuoyancy_store, T* __restrict__ dvxdbuoyancy_store,
    const T* __restrict__ ayh, const T* __restrict__ byh,
    const T* __restrict__ ay, const T* __restrict__ by,
    const T* __restrict__ axh, const T* __restrict__ bxh,
    const T* __restrict__ ax, const T* __restrict__ bx,
    const T* __restrict__ c,
    T rdy, T rdx, T dtv,
    int t, int interval, int64_t snap_off, int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int model_batched, int store)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= ny || x >= nx)
        return;
    const long ny_nx = (long)ny * nx;
    const long off_s = (long)s * ny_nx;
    const int s_b = model_batched ? s : 0;
    const T* by_s = buoyancy_y + (long)s_b * ny_nx;
    const T* bx_s = buoyancy_x + (long)s_b * ny_nx;
    const bool interior = (y >= pml_y0 && y < pml_y1 && x >= pml_x0 && x < pml_x1);

    // vy: y in [FD_PAD, ny-FD_PAD), x in [FD_PAD, nx-(FD_PAD-1))
    if (y >= FD_PAD && y < ny - FD_PAD && x >= FD_PAD &&
        x < nx - (FD_PAD - 1)) {
        const long off = off_s + (long)y * nx + x;
        T d2 = diff_int_y<T, FD_PAD>(syy, off, c, rdy, nx);
        if (!interior) {
            m_sigmayyy[off] = ayh[y] * m_sigmayyy[off] + byh[y] * d2;
            d2 += m_sigmayyy[off];
        }
        T d1 = diff_half_x<T, FD_PAD>(sxy, off, c, rdx);
        if (!interior) {
            m_sigmaxyx[off] = ax[x] * m_sigmaxyx[off] + bx[x] * d1;
            d1 += m_sigmaxyx[off];
        }
        const T w_sum = d2 + d1;
        if (store && t % interval == 0)
            dvydbuoyancy_store[snap_off + ((long)s * ny + y) * nx + x]
                = dtv * w_sum;
        vy[off] += by_s[(long)y * nx + x] * dtv * w_sum;
    }
    // vx: y in [FD_PAD, ny-(FD_PAD-1)), x in [FD_PAD, nx-FD_PAD)
    if (y >= FD_PAD && y < ny - (FD_PAD - 1) && x >= FD_PAD &&
        x < nx - FD_PAD) {
        const long off = off_s + (long)y * nx + x;
        T d2 = diff_half_y<T, FD_PAD>(sxy, off, c, rdy, nx);
        if (!interior) {
            m_sigmaxyy[off] = ay[y] * m_sigmaxyy[off] + by[y] * d2;
            d2 += m_sigmaxyy[off];
        }
        T d1 = diff_int_x<T, FD_PAD>(sxx, off, c, rdx);
        if (!interior) {
            m_sigmaxxx[off] = axh[x] * m_sigmaxxx[off] + bxh[x] * d1;
            d1 += m_sigmaxxx[off];
        }
        const T w_sum = d1 + d2;
        if (store && t % interval == 0)
            dvxdbuoyancy_store[snap_off + ((long)s * ny + y) * nx + x]
                = dtv * w_sum;
        vx[off] += bx_s[(long)y * nx + x] * dtv * w_sum;
    }
}


// ---------------- interior/PML split ----------------
// The C-PML profiles (a, b) are exactly zero in the strict interior and
// nonzero in the PML frame, so the interior box can be recovered from the
// profiles themselves: [pml_y0, pml_y1) x [pml_x0, pml_x1) with
// pml_y0/pml_x0 = first zero profile index and pml_y1/pml_x1 = last zero
// profile index.  The launchers therefore need no extra parameters.
template <typename T>
static void pml_axis_interior(const T* p, int64_t n, int& p0, int& p1)
{
    p0 = 0;
    while (p0 < (int)n && p[p0] != (T)0) ++p0;
    p1 = p0;
    while (p1 < (int)n && p[p1] == (T)0) ++p1;
    --p1;  // last zero index; the interior box is [p0, p1)
}

static void get_pml_interior_box(
    const torch::Tensor& by, const torch::Tensor& bx,
    int64_t ny, int64_t nx,
    int& pml_y0, int& pml_y1, int& pml_x0, int& pml_x1)
{
    TORCH_CHECK(by.numel() == ny && bx.numel() == nx,
                "nami elastic2d pml profiles must match grid size");
    auto by_cpu = by.to(torch::kCPU);
    auto bx_cpu = bx.to(torch::kCPU);
    TORCH_CHECK(by_cpu.scalar_type() == torch::kFloat32 ||
                    by_cpu.scalar_type() == torch::kFloat64,
                "nami elastic2d pml profiles must be float32 or float64");
    if (by_cpu.scalar_type() == torch::kFloat32) {
        pml_axis_interior(by_cpu.data_ptr<float>(), ny, pml_y0, pml_y1);
        pml_axis_interior(bx_cpu.data_ptr<float>(), nx, pml_x0, pml_x1);
    } else {
        pml_axis_interior(by_cpu.data_ptr<double>(), ny, pml_y0, pml_y1);
        pml_axis_interior(bx_cpu.data_ptr<double>(), nx, pml_x0, pml_x1);
    }
}

// ---------------- forward: source injection / receiver recording ----------------
template <typename T>
__global__ void inject_pressure_kernel(
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
__global__ void record_pressure_kernel(
    const T* __restrict__ syy, const T* __restrict__ sxx, T* __restrict__ r,
    const long* __restrict__ rec_i,
    int t, int n_shots, int n_rec, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            r[(((long)t * n_shots + s) * n_rec + k)]
                = syy[(long)s * ny_nx + idx] + sxx[(long)s * ny_nx + idx];
    }
}

// ---------------- backward: source gradient / receiver injection ----------------
template <typename T>
__global__ void record_grad_f_p_kernel(
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
__global__ void add_grad_r_kernel(
    T* __restrict__ l_syy, T* __restrict__ l_sxx, const T* __restrict__ grad_r,
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
            l_syy[off] += g;
            l_sxx[off] += g;
        }
    }
}

// ---------------- launchers ----------------
#define LAUNCH_FIELD(KERN, T, FP, ...)                                         \
    {                                                                          \
        dim3 block(16, 16);                                                    \
        dim3 grid((nx + 15) / 16, (ny + 15) / 16, n_shots);                    \
        KERN<T, FP><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami elastic2d " #KERN " failed"); \
    }

// Dispatch the compile-time-FD_PAD kernel variants for the runtime fd_pad_y0
// (= accuracy/2): cases 1..4 for accuracy 2/4/6/8.  Every stencil radius in
// the elastic2d kernels is fd_pad_y0/fd_pad_x0 (both == accuracy/2), so a
// single compile-time FD_PAD drives all loop bounds; guards keep their
// original bounds (FD_PAD-1 where the code used fd_pad_y1/x1, FD_PAD where
// fd_pad_y0/x0), so numerics stay bitwise identical.
#define LAUNCH_ELASTIC2D(KERN, T, ...)                                         \
    {                                                                          \
        switch ((int)fd_pad_y0) {                                              \
            case 1: LAUNCH_FIELD(KERN, T, 1, __VA_ARGS__); break;              \
            case 2: LAUNCH_FIELD(KERN, T, 2, __VA_ARGS__); break;              \
            case 3: LAUNCH_FIELD(KERN, T, 3, __VA_ARGS__); break;              \
            case 4: LAUNCH_FIELD(KERN, T, 4, __VA_ARGS__); break;              \
            default: TORCH_CHECK(false, "nami elastic2d: unsupported fd_pad"); \
        }                                                                      \
    }

void step_forward_velocity(
    torch::Tensor vy, torch::Tensor vx,
    torch::Tensor syy, torch::Tensor sxx, torch::Tensor sxy,
    torch::Tensor m_sigmayyy, torch::Tensor m_sigmaxyx,
    torch::Tensor m_sigmaxyy, torch::Tensor m_sigmaxxx,
    torch::Tensor buoyancy_y, torch::Tensor buoyancy_x,
    torch::Tensor dvydbuoyancy_store, torch::Tensor dvxdbuoyancy_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv,
    int64_t t, int64_t interval, int64_t snap_off, int64_t n_shots, int64_t ny, int64_t nx,
    int64_t model_batched, int64_t store,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1)
{
    if (vy.scalar_type() == torch::kFloat32) {
        LAUNCH_ELASTIC2D(step_forward_velocity_kernel, float,
            vy.data_ptr<float>(), vx.data_ptr<float>(),
            syy.data_ptr<float>(), sxx.data_ptr<float>(), sxy.data_ptr<float>(),
            m_sigmayyy.data_ptr<float>(), m_sigmaxyx.data_ptr<float>(),
            m_sigmaxyy.data_ptr<float>(), m_sigmaxxx.data_ptr<float>(),
            buoyancy_y.data_ptr<float>(), buoyancy_x.data_ptr<float>(),
            dvydbuoyancy_store.data_ptr<float>(), dvxdbuoyancy_store.data_ptr<float>(),
            ayh.data_ptr<float>(), byh.data_ptr<float>(), ay.data_ptr<float>(), by.data_ptr<float>(),
            axh.data_ptr<float>(), bxh.data_ptr<float>(), ax.data_ptr<float>(), bx.data_ptr<float>(),
            c.data_ptr<float>(),
            (float)rdy, (float)rdx, (float)dtv,
            (int)t, (int)interval, (int64_t)snap_off, (int)n_shots, (int)ny, (int)nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)model_batched, (int)store);
    } else {
        LAUNCH_ELASTIC2D(step_forward_velocity_kernel, double,
            vy.data_ptr<double>(), vx.data_ptr<double>(),
            syy.data_ptr<double>(), sxx.data_ptr<double>(), sxy.data_ptr<double>(),
            m_sigmayyy.data_ptr<double>(), m_sigmaxyx.data_ptr<double>(),
            m_sigmaxyy.data_ptr<double>(), m_sigmaxxx.data_ptr<double>(),
            buoyancy_y.data_ptr<double>(), buoyancy_x.data_ptr<double>(),
            dvydbuoyancy_store.data_ptr<double>(), dvxdbuoyancy_store.data_ptr<double>(),
            ayh.data_ptr<double>(), byh.data_ptr<double>(), ay.data_ptr<double>(), by.data_ptr<double>(),
            axh.data_ptr<double>(), bxh.data_ptr<double>(), ax.data_ptr<double>(), bx.data_ptr<double>(),
            c.data_ptr<double>(),
            rdy, rdx, dtv,
            (int)t, (int)interval, (int64_t)snap_off, (int)n_shots, (int)ny, (int)nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)model_batched, (int)store);
    }
}

void step_forward_stress(
    torch::Tensor vy, torch::Tensor vx,
    torch::Tensor syy, torch::Tensor sxx, torch::Tensor sxy,
    torch::Tensor m_vyy, torch::Tensor m_vxx, torch::Tensor m_vxy, torch::Tensor m_vyx,
    torch::Tensor lamb, torch::Tensor mu, torch::Tensor mu_yx,
    torch::Tensor dvydy_store, torch::Tensor dvxdx_store,
    torch::Tensor dvydx_plus_dvxdy_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv,
    int64_t t, int64_t interval, int64_t snap_off, int64_t n_shots, int64_t ny, int64_t nx,
    int64_t model_batched, int64_t store,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1)
{
    if (vy.scalar_type() == torch::kFloat32) {
        LAUNCH_ELASTIC2D(step_forward_stress_kernel, float,
            vy.data_ptr<float>(), vx.data_ptr<float>(),
            syy.data_ptr<float>(), sxx.data_ptr<float>(), sxy.data_ptr<float>(),
            m_vyy.data_ptr<float>(), m_vxx.data_ptr<float>(),
            m_vxy.data_ptr<float>(), m_vyx.data_ptr<float>(),
            lamb.data_ptr<float>(), mu.data_ptr<float>(), mu_yx.data_ptr<float>(),
            dvydy_store.data_ptr<float>(), dvxdx_store.data_ptr<float>(),
            dvydx_plus_dvxdy_store.data_ptr<float>(),
            ayh.data_ptr<float>(), byh.data_ptr<float>(), ay.data_ptr<float>(), by.data_ptr<float>(),
            axh.data_ptr<float>(), bxh.data_ptr<float>(), ax.data_ptr<float>(), bx.data_ptr<float>(),
            c.data_ptr<float>(),
            (float)rdy, (float)rdx, (float)dtv,
            (int)t, (int)interval, (int64_t)snap_off, (int)n_shots, (int)ny, (int)nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)model_batched, (int)store);
    } else {
        LAUNCH_ELASTIC2D(step_forward_stress_kernel, double,
            vy.data_ptr<double>(), vx.data_ptr<double>(),
            syy.data_ptr<double>(), sxx.data_ptr<double>(), sxy.data_ptr<double>(),
            m_vyy.data_ptr<double>(), m_vxx.data_ptr<double>(),
            m_vxy.data_ptr<double>(), m_vyx.data_ptr<double>(),
            lamb.data_ptr<double>(), mu.data_ptr<double>(), mu_yx.data_ptr<double>(),
            dvydy_store.data_ptr<double>(), dvxdx_store.data_ptr<double>(),
            dvydx_plus_dvxdy_store.data_ptr<double>(),
            ayh.data_ptr<double>(), byh.data_ptr<double>(), ay.data_ptr<double>(), by.data_ptr<double>(),
            axh.data_ptr<double>(), bxh.data_ptr<double>(), ax.data_ptr<double>(), bx.data_ptr<double>(),
            c.data_ptr<double>(),
            rdy, rdx, dtv,
            (int)t, (int)interval, (int64_t)snap_off, (int)n_shots, (int)ny, (int)nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)model_batched, (int)store);
    }
}

void step_adjoint_velocity(
    torch::Tensor lamb, torch::Tensor mu, torch::Tensor mu_yx,
    torch::Tensor buoyancy_y, torch::Tensor buoyancy_x,
    torch::Tensor l_vy, torch::Tensor l_vx,
    torch::Tensor l_syy, torch::Tensor l_sxx, torch::Tensor l_sxy,
    torch::Tensor m_vyy, torch::Tensor m_vxx, torch::Tensor m_vxy, torch::Tensor m_vyx,
    torch::Tensor m_syyy_old, torch::Tensor m_syx_old,
    torch::Tensor m_syxy_old, torch::Tensor m_syxx_old,
    torch::Tensor m_syyy_new, torch::Tensor m_syx_new,
    torch::Tensor m_syxy_new, torch::Tensor m_syxx_new,
    torch::Tensor grad_buoyancy_y, torch::Tensor grad_buoyancy_x,
    torch::Tensor dvydbuoyancy_store, torch::Tensor dvxdbuoyancy_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv, double scale,
    int64_t t, int64_t interval, int64_t snap_off, int64_t n_shots, int64_t ny, int64_t nx,
    int64_t model_batched,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1)
{
    // The adjoint stencils gather W at points up to fd_pad cells away from
    // the thread, so the memory-free interior box is shrunk by fd_pad on
    // every side: the shrunk-out ring keeps the full PML logic.
    const int iy0 = pml_y0 + (int)fd_pad_y0, iy1 = pml_y1 - (int)fd_pad_y0;
    const int ix0 = pml_x0 + (int)fd_pad_x0, ix1 = pml_x1 - (int)fd_pad_x0;
    if (lamb.scalar_type() == torch::kFloat32) {
        LAUNCH_ELASTIC2D(step_adjoint_velocity_kernel, float,
            lamb.data_ptr<float>(), mu.data_ptr<float>(), mu_yx.data_ptr<float>(),
            buoyancy_y.data_ptr<float>(), buoyancy_x.data_ptr<float>(),
            l_vy.data_ptr<float>(), l_vx.data_ptr<float>(),
            l_syy.data_ptr<float>(), l_sxx.data_ptr<float>(), l_sxy.data_ptr<float>(),
            m_vyy.data_ptr<float>(), m_vxx.data_ptr<float>(),
            m_vxy.data_ptr<float>(), m_vyx.data_ptr<float>(),
            m_syyy_old.data_ptr<float>(), m_syx_old.data_ptr<float>(),
            m_syxy_old.data_ptr<float>(), m_syxx_old.data_ptr<float>(),
            m_syyy_new.data_ptr<float>(), m_syx_new.data_ptr<float>(),
            m_syxy_new.data_ptr<float>(), m_syxx_new.data_ptr<float>(),
            grad_buoyancy_y.data_ptr<float>(), grad_buoyancy_x.data_ptr<float>(),
            dvydbuoyancy_store.data_ptr<float>(), dvxdbuoyancy_store.data_ptr<float>(),
            ayh.data_ptr<float>(), byh.data_ptr<float>(), ay.data_ptr<float>(), by.data_ptr<float>(),
            axh.data_ptr<float>(), bxh.data_ptr<float>(), ax.data_ptr<float>(), bx.data_ptr<float>(),
            c.data_ptr<float>(),
            (float)rdy, (float)rdx, (float)dtv, (float)scale,
            (int)t, (int)interval, (int64_t)snap_off, (int)n_shots, (int)ny, (int)nx,
            iy0, iy1, ix0, ix1,
            (int)model_batched);
    } else {
        LAUNCH_ELASTIC2D(step_adjoint_velocity_kernel, double,
            lamb.data_ptr<double>(), mu.data_ptr<double>(), mu_yx.data_ptr<double>(),
            buoyancy_y.data_ptr<double>(), buoyancy_x.data_ptr<double>(),
            l_vy.data_ptr<double>(), l_vx.data_ptr<double>(),
            l_syy.data_ptr<double>(), l_sxx.data_ptr<double>(), l_sxy.data_ptr<double>(),
            m_vyy.data_ptr<double>(), m_vxx.data_ptr<double>(),
            m_vxy.data_ptr<double>(), m_vyx.data_ptr<double>(),
            m_syyy_old.data_ptr<double>(), m_syx_old.data_ptr<double>(),
            m_syxy_old.data_ptr<double>(), m_syxx_old.data_ptr<double>(),
            m_syyy_new.data_ptr<double>(), m_syx_new.data_ptr<double>(),
            m_syxy_new.data_ptr<double>(), m_syxx_new.data_ptr<double>(),
            grad_buoyancy_y.data_ptr<double>(), grad_buoyancy_x.data_ptr<double>(),
            dvydbuoyancy_store.data_ptr<double>(), dvxdbuoyancy_store.data_ptr<double>(),
            ayh.data_ptr<double>(), byh.data_ptr<double>(), ay.data_ptr<double>(), by.data_ptr<double>(),
            axh.data_ptr<double>(), bxh.data_ptr<double>(), ax.data_ptr<double>(), bx.data_ptr<double>(),
            c.data_ptr<double>(),
            rdy, rdx, dtv, scale,
            (int)t, (int)interval, (int64_t)snap_off, (int)n_shots, (int)ny, (int)nx,
            iy0, iy1, ix0, ix1,
            (int)model_batched);
    }
}

void step_adjoint_stress(
    torch::Tensor lamb, torch::Tensor mu, torch::Tensor mu_yx,
    torch::Tensor buoyancy_y, torch::Tensor buoyancy_x,
    torch::Tensor l_vy, torch::Tensor l_vx,
    torch::Tensor l_syy, torch::Tensor l_sxx, torch::Tensor l_sxy,
    torch::Tensor m_vyy, torch::Tensor m_vxx, torch::Tensor m_vxy, torch::Tensor m_vyx,
    torch::Tensor m_syyy_old, torch::Tensor m_syx_old,
    torch::Tensor m_syxy_old, torch::Tensor m_syxx_old,
    torch::Tensor grad_lamb, torch::Tensor grad_mu, torch::Tensor grad_mu_yx,
    torch::Tensor dvydy_store, torch::Tensor dvxdx_store,
    torch::Tensor dvydx_plus_dvxdy_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv, double scale,
    int64_t t, int64_t interval, int64_t snap_off, int64_t n_shots, int64_t ny, int64_t nx,
    int64_t model_batched,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1)
{
    // Same one-cell shrink as the adjoint velocity kernel.
    const int iy0 = pml_y0 + (int)fd_pad_y0, iy1 = pml_y1 - (int)fd_pad_y0;
    const int ix0 = pml_x0 + (int)fd_pad_x0, ix1 = pml_x1 - (int)fd_pad_x0;
    if (lamb.scalar_type() == torch::kFloat32) {
        LAUNCH_ELASTIC2D(step_adjoint_stress_kernel, float,
            lamb.data_ptr<float>(), mu.data_ptr<float>(), mu_yx.data_ptr<float>(),
            buoyancy_y.data_ptr<float>(), buoyancy_x.data_ptr<float>(),
            l_vy.data_ptr<float>(), l_vx.data_ptr<float>(),
            l_syy.data_ptr<float>(), l_sxx.data_ptr<float>(), l_sxy.data_ptr<float>(),
            m_vyy.data_ptr<float>(), m_vxx.data_ptr<float>(),
            m_vxy.data_ptr<float>(), m_vyx.data_ptr<float>(),
            m_syyy_old.data_ptr<float>(), m_syx_old.data_ptr<float>(),
            m_syxy_old.data_ptr<float>(), m_syxx_old.data_ptr<float>(),
            grad_lamb.data_ptr<float>(), grad_mu.data_ptr<float>(), grad_mu_yx.data_ptr<float>(),
            dvydy_store.data_ptr<float>(), dvxdx_store.data_ptr<float>(),
            dvydx_plus_dvxdy_store.data_ptr<float>(),
            ayh.data_ptr<float>(), byh.data_ptr<float>(), ay.data_ptr<float>(), by.data_ptr<float>(),
            axh.data_ptr<float>(), bxh.data_ptr<float>(), ax.data_ptr<float>(), bx.data_ptr<float>(),
            c.data_ptr<float>(),
            (float)rdy, (float)rdx, (float)dtv, (float)scale,
            (int)t, (int)interval, (int64_t)snap_off, (int)n_shots, (int)ny, (int)nx,
            iy0, iy1, ix0, ix1,
            (int)model_batched);
    } else {
        LAUNCH_ELASTIC2D(step_adjoint_stress_kernel, double,
            lamb.data_ptr<double>(), mu.data_ptr<double>(), mu_yx.data_ptr<double>(),
            buoyancy_y.data_ptr<double>(), buoyancy_x.data_ptr<double>(),
            l_vy.data_ptr<double>(), l_vx.data_ptr<double>(),
            l_syy.data_ptr<double>(), l_sxx.data_ptr<double>(), l_sxy.data_ptr<double>(),
            m_vyy.data_ptr<double>(), m_vxx.data_ptr<double>(),
            m_vxy.data_ptr<double>(), m_vyx.data_ptr<double>(),
            m_syyy_old.data_ptr<double>(), m_syx_old.data_ptr<double>(),
            m_syxy_old.data_ptr<double>(), m_syxx_old.data_ptr<double>(),
            grad_lamb.data_ptr<double>(), grad_mu.data_ptr<double>(), grad_mu_yx.data_ptr<double>(),
            dvydy_store.data_ptr<double>(), dvxdx_store.data_ptr<double>(),
            dvydx_plus_dvxdy_store.data_ptr<double>(),
            ayh.data_ptr<double>(), byh.data_ptr<double>(), ay.data_ptr<double>(), by.data_ptr<double>(),
            axh.data_ptr<double>(), bxh.data_ptr<double>(), ax.data_ptr<double>(), bx.data_ptr<double>(),
            c.data_ptr<double>(),
            rdy, rdx, dtv, scale,
            (int)t, (int)interval, (int64_t)snap_off, (int)n_shots, (int)ny, (int)nx,
            iy0, iy1, ix0, ix1,
            (int)model_batched);
    }
}

void inject_pressure(torch::Tensor syy, torch::Tensor sxx, torch::Tensor f,
                     torch::Tensor src_i, int64_t t, int64_t n_shots,
                     int64_t n_src, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_src + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(syy.scalar_type(), "inject_pressure", [&] {
        inject_pressure_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            syy.data_ptr<scalar_t>(), sxx.data_ptr<scalar_t>(), f.data_ptr<scalar_t>(),
            src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami inject_pressure failed");
}

void record_pressure(torch::Tensor syy, torch::Tensor sxx, torch::Tensor r,
                     torch::Tensor rec_i, int64_t t, int64_t n_shots,
                     int64_t n_rec, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_rec + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(syy.scalar_type(), "record_pressure", [&] {
        record_pressure_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            syy.data_ptr<scalar_t>(), sxx.data_ptr<scalar_t>(), r.data_ptr<scalar_t>(),
            rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami record_pressure failed");
}

void record_grad_f_p(torch::Tensor l_syy, torch::Tensor l_sxx, torch::Tensor grad_f,
                     torch::Tensor src_i, int64_t t, int64_t n_shots,
                     int64_t n_src, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_src + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(l_syy.scalar_type(), "record_grad_f_p", [&] {
        record_grad_f_p_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            l_syy.data_ptr<scalar_t>(), l_sxx.data_ptr<scalar_t>(), grad_f.data_ptr<scalar_t>(),
            src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami record_grad_f_p failed");
}

void add_grad_r(torch::Tensor l_syy, torch::Tensor l_sxx, torch::Tensor grad_r,
                torch::Tensor rec_i, int64_t t, int64_t n_shots,
                int64_t n_rec, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_rec + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(l_syy.scalar_type(), "add_grad_r", [&] {
        add_grad_r_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            l_syy.data_ptr<scalar_t>(), l_sxx.data_ptr<scalar_t>(), grad_r.data_ptr<scalar_t>(),
            rec_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami add_grad_r failed");
}

// ---------------- fused per-step entry points ----------------
// One pybind call per time step instead of four: the per-call interpreter
// overhead is the dominant cost for short (ny, nx) runs.
void forward_step(
    torch::Tensor vy, torch::Tensor vx,
    torch::Tensor syy, torch::Tensor sxx, torch::Tensor sxy,
    torch::Tensor m_sigmayyy, torch::Tensor m_sigmaxyx,
    torch::Tensor m_sigmaxyy, torch::Tensor m_sigmaxxx,
    torch::Tensor buoyancy_y, torch::Tensor buoyancy_x,
    torch::Tensor dvydbuoyancy_store, torch::Tensor dvxdbuoyancy_store,
    torch::Tensor m_vyy, torch::Tensor m_vxx, torch::Tensor m_vxy, torch::Tensor m_vyx,
    torch::Tensor lamb, torch::Tensor mu, torch::Tensor mu_yx,
    torch::Tensor dvydy_store, torch::Tensor dvxdx_store,
    torch::Tensor dvydx_plus_dvxdy_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    torch::Tensor f, torch::Tensor src_i, torch::Tensor rec_i, torch::Tensor r,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv,
    int64_t t, int64_t interval, int64_t snap_off,
    int64_t n_shots, int64_t ny, int64_t nx, int64_t ny_nx,
    int64_t n_src, int64_t n_rec,
    int64_t model_batched, int64_t store, int64_t record,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1)
{
    if (record && n_rec > 0)
        record_pressure(syy, sxx, r, rec_i, t, n_shots, n_rec, ny_nx);
    step_forward_velocity(
        vy, vx, syy, sxx, sxy,
        m_sigmayyy, m_sigmaxyx, m_sigmaxyy, m_sigmaxxx,
        buoyancy_y, buoyancy_x,
        dvydbuoyancy_store, dvxdbuoyancy_store,
        ayh, byh, ay, by, axh, bxh, ax, bx,
        c, fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
        rdy, rdx, dtv, t, interval, snap_off, n_shots, ny, nx,
        model_batched, store,
        pml_y0, pml_y1, pml_x0, pml_x1);
    step_forward_stress(
        vy, vx, syy, sxx, sxy,
        m_vyy, m_vxx, m_vxy, m_vyx,
        lamb, mu, mu_yx,
        dvydy_store, dvxdx_store, dvydx_plus_dvxdy_store,
        ayh, byh, ay, by, axh, bxh, ax, bx,
        c, fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
        rdy, rdx, dtv, t, interval, snap_off, n_shots, ny, nx,
        model_batched, store,
        pml_y0, pml_y1, pml_x0, pml_x1);
    if (n_src > 0)
        inject_pressure(syy, sxx, f, src_i, t, n_shots, n_src, ny_nx);
}

void backward_step(
    torch::Tensor lamb, torch::Tensor mu, torch::Tensor mu_yx,
    torch::Tensor buoyancy_y, torch::Tensor buoyancy_x,
    torch::Tensor l_vy, torch::Tensor l_vx,
    torch::Tensor l_syy, torch::Tensor l_sxx, torch::Tensor l_sxy,
    torch::Tensor m_vyy, torch::Tensor m_vxx, torch::Tensor m_vxy, torch::Tensor m_vyx,
    torch::Tensor m_syyy_old, torch::Tensor m_syx_old,
    torch::Tensor m_syxy_old, torch::Tensor m_syxx_old,
    torch::Tensor m_syyy_new, torch::Tensor m_syx_new,
    torch::Tensor m_syxy_new, torch::Tensor m_syxx_new,
    torch::Tensor grad_buoyancy_y, torch::Tensor grad_buoyancy_x,
    torch::Tensor grad_lamb, torch::Tensor grad_mu, torch::Tensor grad_mu_yx,
    torch::Tensor dvydbuoyancy_store, torch::Tensor dvxdbuoyancy_store,
    torch::Tensor dvydy_store, torch::Tensor dvxdx_store,
    torch::Tensor dvydx_plus_dvxdy_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    torch::Tensor grad_f, torch::Tensor src_i,
    torch::Tensor grad_r, torch::Tensor rec_i,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv, double scale,
    int64_t t, int64_t interval, int64_t snap_off,
    int64_t n_shots, int64_t ny, int64_t nx, int64_t ny_nx,
    int64_t n_src, int64_t n_rec,
    int64_t model_batched,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1)
{
    if (n_src > 0)
        record_grad_f_p(l_syy, l_sxx, grad_f, src_i, t, n_shots, n_src, ny_nx);
    step_adjoint_velocity(
        lamb, mu, mu_yx, buoyancy_y, buoyancy_x,
        l_vy, l_vx, l_syy, l_sxx, l_sxy,
        m_vyy, m_vxx, m_vxy, m_vyx,
        m_syyy_old, m_syx_old, m_syxy_old, m_syxx_old,
        m_syyy_new, m_syx_new, m_syxy_new, m_syxx_new,
        grad_buoyancy_y, grad_buoyancy_x,
        dvydbuoyancy_store, dvxdbuoyancy_store,
        ayh, byh, ay, by, axh, bxh, ax, bx,
        c, fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
        rdy, rdx, dtv, scale, t, interval, snap_off, n_shots, ny, nx,
        model_batched,
        pml_y0, pml_y1, pml_x0, pml_x1);
    step_adjoint_stress(
        lamb, mu, mu_yx, buoyancy_y, buoyancy_x,
        l_vy, l_vx, l_syy, l_sxx, l_sxy,
        m_vyy, m_vxx, m_vxy, m_vyx,
        m_syyy_old, m_syx_old, m_syxy_old, m_syxx_old,
        grad_lamb, grad_mu, grad_mu_yx,
        dvydy_store, dvxdx_store, dvydx_plus_dvxdy_store,
        ayh, byh, ay, by, axh, bxh, ax, bx,
        c, fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
        rdy, rdx, dtv, scale, t, interval, snap_off, n_shots, ny, nx,
        model_batched,
        pml_y0, pml_y1, pml_x0, pml_x1);
    if (n_rec > 0)
        add_grad_r(l_syy, l_sxx, grad_r, rec_i, t, n_shots, n_rec, ny_nx);
}

std::vector<int64_t> get_pml_box(
    torch::Tensor by, torch::Tensor bx, int64_t ny, int64_t nx)
{
    int pml_y0, pml_y1, pml_x0, pml_x1;
    get_pml_interior_box(by, bx, ny, nx, pml_y0, pml_y1, pml_x0, pml_x1);
    return {pml_y0, pml_y1, pml_x0, pml_x1};
}

// ---------------- whole-loop drivers ----------------
// One pybind call runs the entire forward/adjoint pass (deepwave-style):
// the per-step functions above are reused unchanged with the same argument
// order the Python loop used, so results are bitwise identical.
// The 13 flat state buffers (no rings, updated in place) are passed as one
// vector in the N_STATE order documented in elastic2d.py:
//   [vy, vx, syy, sxx, sxy, m_vyy, m_vxx, m_vxy, m_vyx,
//    m_sigmayyy, m_sigmaxyx, m_sigmaxyy, m_sigmaxxx]

using nami_storage::ckpt_restore;
using nami_storage::ckpt_save;
using nami_storage::zero_buffers;

void forward_loop(
    std::vector<torch::Tensor> state,    // 13 flat buffers (N_STATE order)
    torch::Tensor lamb, torch::Tensor mu, torch::Tensor mu_yx,
    torch::Tensor buoyancy_y, torch::Tensor buoyancy_x,
    torch::Tensor dvydy_store, torch::Tensor dvxdx_store,
    torch::Tensor dvxy_store,
    torch::Tensor dvydb_store, torch::Tensor dvxdb_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    torch::Tensor f, torch::Tensor src_i, torch::Tensor rec_i, torch::Tensor r,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv,
    int64_t nt, int64_t interval, int64_t store, int64_t model_batched,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t checkpoint_every, c10::optional<torch::Tensor> ckpt_state,
    // Wavefield I/O (deepwave-style): init_state/final_state use the N_STATE
    // layout [vy, vx, syy, sxx, sxy, m_vyy, m_vxx, m_vxy, m_vyx,
    //          m_sigmayyy, m_sigmaxyx, m_sigmaxyy, m_sigmaxxx] (same as ckpt).
    c10::optional<torch::Tensor> init_state,
    c10::optional<torch::Tensor> final_state,
    // Per-step forward callback: called every `callback_frequency` steps as
    // callback(t, nt, vy, vx, syy, sxx, sxy) — the physics wavefields are the
    // padded GPU tensors (live views, no copy); the PML memory variables are
    // not passed.
    py::object callback, int64_t callback_frequency)
{
    TORCH_CHECK(state.size() == 13, "nami elastic2d forward_loop: bad state size");
    const int64_t n_shots = state[0].size(0);
    const int64_t ny = state[0].size(1), nx = state[0].size(2);
    const int64_t ny_nx = ny * nx;
    const int64_t n_src = src_i.size(1), n_rec = rec_i.size(1);
    const int64_t shot_count = n_shots * ny_nx;
    const bool ckpt = ckpt_state.has_value() && checkpoint_every > 0;
    const bool has_cb = !callback.is_none();

    // Optional initial state (continuation runs): the 13 flat buffers are
    // updated in place, so they ARE the wavefield state at time t; restore
    // each slot in N_STATE order.
    if (init_state.has_value()) {
        auto c = *init_state;
        for (int64_t j = 0; j < 13; ++j)
            state[j].copy_(c[j]);
    }

    for (int64_t t = 0; t < nt; ++t) {
        if (ckpt && t > 0 && t % checkpoint_every == 0) {
            ckpt_save(*ckpt_state, t / checkpoint_every - 1,
                      {&state[0], &state[1], &state[2], &state[3], &state[4],
                       &state[5], &state[6], &state[7], &state[8], &state[9],
                       &state[10], &state[11], &state[12]});
        }
        if (has_cb && t % callback_frequency == 0) {
            py::gil_scoped_acquire gil;
            callback(t, nt, state[0], state[1], state[2], state[3], state[4]);
        }
        const int64_t snap_off = store ? (t / interval) * shot_count : 0;
        forward_step(
            state[0], state[1], state[2], state[3], state[4],
            state[9], state[10], state[11], state[12],
            buoyancy_y, buoyancy_x, dvydb_store, dvxdb_store,
            state[5], state[6], state[7], state[8],
            lamb, mu, mu_yx, dvydy_store, dvxdx_store, dvxy_store,
            ayh, byh, ay, by, axh, bxh, ax, bx,
            c, f, src_i, rec_i, r,
            fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
            rdy, rdx, dtv, t, interval, snap_off,
            n_shots, ny, nx, ny_nx, n_src, n_rec,
            model_batched, store, 1,
            pml_y0, pml_y1, pml_x0, pml_x1);
    }

    // Optional final state (next-step fields for continuation): the flat
    // buffers hold the field just computed by the last forward step, so copy
    // them directly (N_STATE order).  This makes a split run (continuation
    // via init_state) bitwise match a one-shot run.
    if (final_state.has_value()) {
        auto c = *final_state;
        for (int64_t j = 0; j < 13; ++j)
            c[j].copy_(state[j]);
    }
}

void adjoint_loop(
    torch::Tensor lamb, torch::Tensor mu, torch::Tensor mu_yx,
    torch::Tensor buoyancy_y, torch::Tensor buoyancy_x,
    torch::Tensor l_vy, torch::Tensor l_vx,
    torch::Tensor l_syy, torch::Tensor l_sxx, torch::Tensor l_sxy,
    torch::Tensor m_vyy, torch::Tensor m_vxx, torch::Tensor m_vxy, torch::Tensor m_vyx,
    std::vector<torch::Tensor> m_sig_a,  // 4 alternating m_sigma adjoint memories
    std::vector<torch::Tensor> m_sig_b,
    torch::Tensor grad_by, torch::Tensor grad_bx,
    torch::Tensor grad_lamb, torch::Tensor grad_mu, torch::Tensor grad_mu_yx,
    torch::Tensor dvydy_store, torch::Tensor dvxdx_store,
    torch::Tensor dvxy_store,
    torch::Tensor dvydb_store, torch::Tensor dvxdb_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    torch::Tensor grad_f, torch::Tensor src_i,
    torch::Tensor grad_r, torch::Tensor rec_i,
    torch::Tensor f,
    std::vector<torch::Tensor> fstate,   // 13 replay buffers (empty = full storage)
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv, double scale,
    int64_t nt, int64_t interval, int64_t model_batched,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    torch::Tensor segments,              // int64 [n_seg, 2] on CPU; empty = full storage
    c10::optional<torch::Tensor> ckpt_state)
{
    TORCH_CHECK(m_sig_a.size() == 4 && m_sig_b.size() == 4,
                "nami elastic2d adjoint_loop: bad m_sigma bank sizes");
    const int64_t n_shots = l_vy.size(0);
    const int64_t ny = l_vy.size(1), nx = l_vy.size(2);
    const int64_t ny_nx = ny * nx;
    const int64_t n_src = src_i.size(1), n_rec = rec_i.size(1);
    const int64_t shot_count = n_shots * ny_nx;

    // NB: parity = (nt - 1 - t) % 2 is never negative here (t <= nt - 1),
    // unlike ring indices that need the (t + N - 1) % N guard.
    auto adjoint_at = [&](int64_t t, int64_t snap_off) {
        const int parity = (int)((nt - 1 - t) % 2);
        const std::vector<torch::Tensor>& old = parity ? m_sig_b : m_sig_a;
        const std::vector<torch::Tensor>& nw = parity ? m_sig_a : m_sig_b;
        backward_step(
            lamb, mu, mu_yx, buoyancy_y, buoyancy_x,
            l_vy, l_vx, l_syy, l_sxx, l_sxy,
            m_vyy, m_vxx, m_vxy, m_vyx,
            old[0], old[1], old[2], old[3],
            nw[0], nw[1], nw[2], nw[3],
            grad_by, grad_bx, grad_lamb, grad_mu, grad_mu_yx,
            dvydb_store, dvxdb_store,
            dvydy_store, dvxdx_store, dvxy_store,
            ayh, byh, ay, by, axh, bxh, ax, bx,
            c, grad_f, src_i, grad_r, rec_i,
            fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
            rdy, rdx, dtv, scale, t, interval, snap_off,
            n_shots, ny, nx, ny_nx, n_src, n_rec,
            model_batched,
            pml_y0, pml_y1, pml_x0, pml_x1);
    };

    if (segments.numel() == 0) {
        for (int64_t t = nt - 1; t >= 0; --t)
            adjoint_at(t, (t / interval) * shot_count);
        return;
    }

    // Checkpointed backward: per segment, restore the forward state, replay
    // the forward steps to regenerate the snapshots, then run the adjoint.
    // The adjoint state carries across segments.
    TORCH_CHECK(fstate.size() == 13,
                "nami elastic2d adjoint_loop: bad replay state size");
    auto seg = segments.accessor<int64_t, 2>();
    const int64_t n_seg = segments.size(0);
    for (int64_t k = n_seg - 1; k >= 0; --k) {
        const int64_t s0 = seg[k][0], s1 = seg[k][1];
        if (s0 > 0) {
            ckpt_restore(*ckpt_state, k - 1,
                         {&fstate[0], &fstate[1], &fstate[2], &fstate[3],
                          &fstate[4], &fstate[5], &fstate[6], &fstate[7],
                          &fstate[8], &fstate[9], &fstate[10], &fstate[11],
                          &fstate[12]});
        } else {
            zero_buffers({&fstate[0], &fstate[1], &fstate[2], &fstate[3],
                          &fstate[4], &fstate[5], &fstate[6], &fstate[7],
                          &fstate[8], &fstate[9], &fstate[10], &fstate[11],
                          &fstate[12]});
        }
        for (int64_t t = s0; t < s1; ++t) {
            const int64_t snap_off = ((t - s0) / interval) * shot_count;
            // record=0: replay passes grad_r as a dummy receiver buffer.
            forward_step(
                fstate[0], fstate[1], fstate[2], fstate[3], fstate[4],
                fstate[9], fstate[10], fstate[11], fstate[12],
                buoyancy_y, buoyancy_x, dvydb_store, dvxdb_store,
                fstate[5], fstate[6], fstate[7], fstate[8],
                lamb, mu, mu_yx, dvydy_store, dvxdx_store, dvxy_store,
                ayh, byh, ay, by, axh, bxh, ax, bx,
                c, f, src_i, rec_i, grad_r,
                fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
                rdy, rdx, dtv, t, interval, snap_off,
                n_shots, ny, nx, ny_nx, n_src, n_rec,
                model_batched, 1, 0,
                pml_y0, pml_y1, pml_x0, pml_x1);
        }
        for (int64_t t = s1 - 1; t >= s0; --t)
            adjoint_at(t, ((t - s0) / interval) * shot_count);
    }
}

// ---------------- pybind11 bindings ----------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_pml_box", &get_pml_box);
    m.def("forward_step", &forward_step);
    m.def("backward_step", &backward_step);
    m.def("forward_loop", &forward_loop);
    m.def("adjoint_loop", &adjoint_loop);
    m.def("step_forward_velocity", &step_forward_velocity);
    m.def("step_forward_stress", &step_forward_stress);
    m.def("inject_pressure", &inject_pressure);
    m.def("record_pressure", &record_pressure);
    m.def("record_grad_f_p", &record_grad_f_p);
    m.def("add_grad_r", &add_grad_r);
    m.def("step_adjoint_velocity", &step_adjoint_velocity);
    m.def("step_adjoint_stress", &step_adjoint_stress);
    NAMI_STORAGE_PYBIND(m);
}
