// nami elastic 2D Born: first-order scattering and exact discrete-transpose adjoint.
// The background and scattered fields share the elastic2d velocity-stress grid and CPML.

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

// ==================== Elastic (2D velocity-stress) Born ====================

// ---------------- forward: velocity update (background + scattered) ----------------
// Region A (vy): y in [FD_PAD, ny-FD_PAD), x in [FD_PAD, nx-(FD_PAD-1))
//   vy += by*dtv*w_y      w_y  = DIFFYH1(syy)+DIFFX1(sxy)  (PML memories)
//   dvy += by*dtv*dw_y + dby*dtv*w_y
// Region B (vx): y in [FD_PAD, ny-(FD_PAD-1)), x in [FD_PAD, nx-FD_PAD)
//   vx += bx*dtv*w_x      w_x  = DIFFXH1(sxx)+DIFFY1(sxy)
//   dvx += bx*dtv*dw_x + dbx*dtv*w_x
// dtv*w_y, dtv*w_x and the scattered analogues are snapshotted for the
// buoyancy imaging conditions.
template <typename T, int FD_PAD>
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
    T rdy, T rdx, T dtv,
    int t, int interval, int n_shots, int ny, int nx,
    int model_batched, int scatter_batched, int store, int64_t snap_off)
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

    if (y >= FD_PAD && y < ny - FD_PAD && x >= FD_PAD &&
        x < nx - (FD_PAD - 1)) {
        const long off = off_s + (long)y * nx + x;
        T d2 = diff_int_y<T, FD_PAD>(syy, off, c, rdy, nx);
        m_sigmayyy[off] = ayh[y] * m_sigmayyy[off] + byh[y] * d2;
        d2 += m_sigmayyy[off];
        T d1 = diff_half_x<T, FD_PAD>(sxy, off, c, rdx);
        m_sigmaxyx[off] = ax[x] * m_sigmaxyx[off] + bx[x] * d1;
        d1 += m_sigmaxyx[off];
        const T w_y = d2 + d1;
        vy[off] += by_s[(long)y * nx + x] * dtv * w_y;
        T dd2 = diff_int_y<T, FD_PAD>(dsyy, off, c, rdy, nx);
        dm_sigmayyy[off] = ayh[y] * dm_sigmayyy[off] + byh[y] * dd2;
        dd2 += dm_sigmayyy[off];
        T dd1 = diff_half_x<T, FD_PAD>(dsxy, off, c, rdx);
        dm_sigmaxyx[off] = ax[x] * dm_sigmaxyx[off] + bx[x] * dd1;
        dd1 += dm_sigmaxyx[off];
        const T dw_y = dd2 + dd1;
        dvy[off] += by_s[(long)y * nx + x] * dtv * dw_y
                  + dby_s[(long)y * nx + x] * dtv * w_y;
        if (store && t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            dvydb_store[soff] = dtv * w_y;
            ddvydb_store[soff] = dtv * dw_y;
        }
    }
    if (y >= FD_PAD && y < ny - (FD_PAD - 1) && x >= FD_PAD &&
        x < nx - FD_PAD) {
        const long off = off_s + (long)y * nx + x;
        T d2 = diff_half_y<T, FD_PAD>(sxy, off, c, rdy, nx);
        m_sigmaxyy[off] = ay[y] * m_sigmaxyy[off] + by[y] * d2;
        d2 += m_sigmaxyy[off];
        T d1 = diff_int_x<T, FD_PAD>(sxx, off, c, rdx);
        m_sigmaxxx[off] = axh[x] * m_sigmaxxx[off] + bxh[x] * d1;
        d1 += m_sigmaxxx[off];
        const T w_x = d1 + d2;
        vx[off] += bx_s[(long)y * nx + x] * dtv * w_x;
        T dd2 = diff_half_y<T, FD_PAD>(dsxy, off, c, rdy, nx);
        dm_sigmaxyy[off] = ay[y] * dm_sigmaxyy[off] + by[y] * dd2;
        dd2 += dm_sigmaxyy[off];
        T dd1 = diff_int_x<T, FD_PAD>(dsxx, off, c, rdx);
        dm_sigmaxxx[off] = axh[x] * dm_sigmaxxx[off] + bxh[x] * dd1;
        dd1 += dm_sigmaxxx[off];
        const T dw_x = dd1 + dd2;
        dvx[off] += bx_s[(long)y * nx + x] * dtv * dw_x
                  + dbx_s[(long)y * nx + x] * dtv * w_x;
        if (store && t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            dvxdb_store[soff] = dtv * w_x;
            ddvxdb_store[soff] = dtv * dw_x;
        }
    }
}

// ---------------- forward: stress update (background + scattered) ----------------
// sigmaii (y in [FD_PAD, ny-(FD_PAD-1)), x in [FD_PAD, nx-(FD_PAD-1))):
//   syy += dtv*(lamb*ssum + 2*mu*dvydy)        sxx += dtv*(lamb*ssum + 2*mu*dvxdx)
//   dsyy += dtv*(lamb*dssum + 2*mu*ddvydy) + dtv*(dlamb*ssum + 2*dmu*dvydy)
//   dsxx += dtv*(lamb*dssum + 2*mu*ddvxdx) + dtv*(dlamb*ssum + 2*dmu*dvxdx)
// sigmaxy (y in [FD_PAD, ny-FD_PAD), x in [FD_PAD, nx-FD_PAD)):
//   sxy += dtv*mu_yx*w_sum    dsxy += dtv*mu_yx*dw_sum + dtv*dmu_yx*w_sum
// The PML-modified derivatives are snapshotted for the imaging conditions.
template <typename T, int FD_PAD>
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
    T rdy, T rdx, T dtv,
    int t, int interval, int n_shots, int ny, int nx,
    int model_batched, int scatter_batched, int store, int64_t snap_off)
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

    if (y >= FD_PAD && x >= FD_PAD && y < ny - (FD_PAD - 1) &&
        x < nx - (FD_PAD - 1)) {
        const long off = off_s + (long)y * nx + x;
        T dvydy = diff_half_y<T, FD_PAD>(vy, off, c, rdy, nx);
        m_vyy[off] = ay[y] * m_vyy[off] + by[y] * dvydy;
        dvydy += m_vyy[off];
        T dvxdx = diff_half_x<T, FD_PAD>(vx, off, c, rdx);
        m_vxx[off] = ax[x] * m_vxx[off] + bx[x] * dvxdx;
        dvxdx += m_vxx[off];
        const T ssum = dvydy + dvxdx;
        const T lamb_v = lamb_s[(long)y * nx + x];
        const T mu_v = mu_s[(long)y * nx + x];
        syy[off] += dtv * (lamb_v * ssum + (T)2 * mu_v * dvydy);
        sxx[off] += dtv * (lamb_v * ssum + (T)2 * mu_v * dvxdx);
        T ddvydy = diff_half_y<T, FD_PAD>(dvy, off, c, rdy, nx);
        dm_vyy[off] = ay[y] * dm_vyy[off] + by[y] * ddvydy;
        ddvydy += dm_vyy[off];
        T ddvxdx = diff_half_x<T, FD_PAD>(dvx, off, c, rdx);
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
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            dvydy_store[soff] = dtv * dvydy;
            dvxdx_store[soff] = dtv * dvxdx;
            ddvydy_store[soff] = dtv * ddvydy;
            ddvxdx_store[soff] = dtv * ddvxdx;
        }
    }
    if (y >= FD_PAD && x >= FD_PAD && y < ny - FD_PAD &&
        x < nx - FD_PAD) {
        const long off = off_s + (long)y * nx + x;
        T dvxdy = diff_int_y<T, FD_PAD>(vx, off, c, rdy, nx);
        m_vxy[off] = ayh[y] * m_vxy[off] + byh[y] * dvxdy;
        dvxdy += m_vxy[off];
        T dvydx = diff_int_x<T, FD_PAD>(vy, off, c, rdx);
        m_vyx[off] = axh[x] * m_vyx[off] + bxh[x] * dvydx;
        dvydx += m_vyx[off];
        const T w_sum = dvydx + dvxdy;
        sxy[off] += dtv * mu_yx_s[(long)y * nx + x] * w_sum;
        T ddvxdy = diff_int_y<T, FD_PAD>(dvx, off, c, rdy, nx);
        dm_vxy[off] = ayh[y] * dm_vxy[off] + byh[y] * ddvxdy;
        ddvxdy += dm_vxy[off];
        T ddvydx = diff_int_x<T, FD_PAD>(dvy, off, c, rdx);
        dm_vyx[off] = axh[x] * dm_vyx[off] + bxh[x] * ddvydx;
        ddvydx += dm_vyx[off];
        const T dw_sum = ddvydx + ddvxdy;
        dsxy[off] += dtv * mu_yx_s[(long)y * nx + x] * dw_sum
                   + dtv * dmu_yx_s[(long)y * nx + x] * w_sum;
        if (store && t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
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
template <typename T, int FD_PAD>
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
    T rdy, T rdx, T dtv,
    int t, int interval, T scale, int n_shots, int ny, int nx,
    int model_batched, int scatter_batched, int64_t snap_off)
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

    // vy: y in [FD_PAD, ny-FD_PAD), x in [FD_PAD, nx-(FD_PAD-1))
    if (y >= FD_PAD && y < ny - FD_PAD && x >= FD_PAD &&
        x < nx - (FD_PAD - 1)) {
        const long off = off_s + (long)y * nx + x;
        // l_vy y-term: -diff_int_y(W_y), W_y = (1+by)*A_y + by*m_vyy
        // (the stencil points are the integer-row syy/sxx cells, where the
        //  forward m_vyy memory lives, so the transpose gathers ``by`` there)
        T dy = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - (long)(k - 1) * nx;
            const long off_p = off + (long)k * nx;
            const int ym = y - (k - 1), yp = y + k;
            const T lam_m = lamb_s[off_m - off_s];
            const T l2m_m = lam_m + (T)2 * mu_s[off_m - off_s];
            const T d2m_m = (T)2 * dmu_s[off_m - off_s] + dlamb_s[off_m - off_s];
            const T A_m = dtv * (l2m_m * l_syy[off_m] + lam_m * l_sxx[off_m]
                              + d2m_m * l_dsyy[off_m] + dlamb_s[off_m - off_s] * l_dsxx[off_m]);
            const T lam_p = lamb_s[off_p - off_s];
            const T l2m_p = lam_p + (T)2 * mu_s[off_p - off_s];
            const T d2m_p = (T)2 * dmu_s[off_p - off_s] + dlamb_s[off_p - off_s];
            const T A_p = dtv * (l2m_p * l_syy[off_p] + lam_p * l_sxx[off_p]
                              + d2m_p * l_dsyy[off_p] + dlamb_s[off_p - off_s] * l_dsxx[off_p]);
            dy += c[k - 1] *
                  (((T)1 + by[ym]) * A_m + by[ym] * m_vyy[off_m] -
                   ((T)1 + by[yp]) * A_p - by[yp] * m_vyy[off_p]);
        }
        // l_dvy y-term: W_dy = (1+by)*A_dy + by*dm_vyy
        T ddy = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - (long)(k - 1) * nx;
            const long off_p = off + (long)k * nx;
            const int ym = y - (k - 1), yp = y + k;
            const T lam_m = lamb_s[off_m - off_s];
            const T l2m_m = lam_m + (T)2 * mu_s[off_m - off_s];
            const T A_dm = dtv * (l2m_m * l_dsyy[off_m] + lam_m * l_dsxx[off_m]);
            const T lam_p = lamb_s[off_p - off_s];
            const T l2m_p = lam_p + (T)2 * mu_s[off_p - off_s];
            const T A_dp = dtv * (l2m_p * l_dsyy[off_p] + lam_p * l_dsxx[off_p]);
            ddy += c[k - 1] *
                   (((T)1 + by[ym]) * A_dm + by[ym] * dm_vyy[off_m] -
                    ((T)1 + by[yp]) * A_dp - by[yp] * dm_vyy[off_p]);
        }
        // l_vy x-term: -diff_half_x(W_x), W_x = (1+bxh)*B_x + bxh*m_vyx
        // (the stencil points are the half-x sxy cells, where m_vyx lives)
        T dx = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - k;
            const long off_p = off + k - 1;
            const int xm = x - k, xp = x + k - 1;
            const T B_m = dtv * (mu_yx_s[off_m - off_s] * l_sxy[off_m]
                               + dmu_yx_s[off_m - off_s] * l_dsxy[off_m]);
            const T B_p = dtv * (mu_yx_s[off_p - off_s] * l_sxy[off_p]
                               + dmu_yx_s[off_p - off_s] * l_dsxy[off_p]);
            dx += c[k - 1] *
                  (((T)1 + bxh[xm]) * B_m + bxh[xm] * m_vyx[off_m] -
                   ((T)1 + bxh[xp]) * B_p - bxh[xp] * m_vyx[off_p]);
        }
        // l_dvy x-term: W_dx = (1+bxh)*B_dx + bxh*dm_vyx
        T ddx = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - k;
            const long off_p = off + k - 1;
            const int xm = x - k, xp = x + k - 1;
            const T B_dm = dtv * mu_yx_s[off_m - off_s] * l_dsxy[off_m];
            const T B_dp = dtv * mu_yx_s[off_p - off_s] * l_dsxy[off_p];
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
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            grad_buoyancy_y[off] += scale * (vy_new * dvydb_store[soff]
                                             + dvy_new * ddvydb_store[soff]);
            grad_dbuoyancy_y[off] += scale * (dvy_new * dvydb_store[soff]);
        }
    }
    // vx: y in [FD_PAD, ny-(FD_PAD-1)), x in [FD_PAD, nx-FD_PAD)
    if (y >= FD_PAD && y < ny - (FD_PAD - 1) && x >= FD_PAD &&
        x < nx - FD_PAD) {
        const long off = off_s + (long)y * nx + x;
        // l_vx y-term: -diff_half_y(W_y'), W_y' = (1+byh)*C_y + byh*m_vxy
        T dy = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - (long)k * nx;
            const long off_p = off + (long)(k - 1) * nx;
            const int ym = y - k, yp = y + k - 1;
            const T C_m = dtv * (mu_yx_s[off_m - off_s] * l_sxy[off_m]
                               + dmu_yx_s[off_m - off_s] * l_dsxy[off_m]);
            const T C_p = dtv * (mu_yx_s[off_p - off_s] * l_sxy[off_p]
                               + dmu_yx_s[off_p - off_s] * l_dsxy[off_p]);
            dy += c[k - 1] *
                  (((T)1 + byh[ym]) * C_m + byh[ym] * m_vxy[off_m] -
                   ((T)1 + byh[yp]) * C_p - byh[yp] * m_vxy[off_p]);
        }
        // l_dvx y-term: W_dy' = (1+byh)*C_dy + byh*dm_vxy
        T ddy = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - (long)k * nx;
            const long off_p = off + (long)(k - 1) * nx;
            const int ym = y - k, yp = y + k - 1;
            const T C_dm = dtv * mu_yx_s[off_m - off_s] * l_dsxy[off_m];
            const T C_dp = dtv * mu_yx_s[off_p - off_s] * l_dsxy[off_p];
            ddy += c[k - 1] *
                   (((T)1 + byh[ym]) * C_dm + byh[ym] * dm_vxy[off_m] -
                    ((T)1 + byh[yp]) * C_dp - byh[yp] * dm_vxy[off_p]);
        }
        // l_vx x-term: -diff_int_x(W_xx), W_xx = (1+bx)*A_x + bx*m_vxx
        // (the stencil points are the integer-x sxx cells, where m_vxx lives)
        T dx = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - k + 1;
            const long off_p = off + k;
            const int xm = x - k + 1, xp = x + k;
            const T lam_m = lamb_s[off_m - off_s];
            const T l2m_m = lam_m + (T)2 * mu_s[off_m - off_s];
            const T d2m_m = (T)2 * dmu_s[off_m - off_s] + dlamb_s[off_m - off_s];
            const T A_m = dtv * (l2m_m * l_sxx[off_m] + lam_m * l_syy[off_m]
                              + d2m_m * l_dsxx[off_m] + dlamb_s[off_m - off_s] * l_dsyy[off_m]);
            const T lam_p = lamb_s[off_p - off_s];
            const T l2m_p = lam_p + (T)2 * mu_s[off_p - off_s];
            const T d2m_p = (T)2 * dmu_s[off_p - off_s] + dlamb_s[off_p - off_s];
            const T A_p = dtv * (l2m_p * l_sxx[off_p] + lam_p * l_syy[off_p]
                              + d2m_p * l_dsxx[off_p] + dlamb_s[off_p - off_s] * l_dsyy[off_p]);
            dx += c[k - 1] *
                  (((T)1 + bx[xm]) * A_m + bx[xm] * m_vxx[off_m] -
                   ((T)1 + bx[xp]) * A_p - bx[xp] * m_vxx[off_p]);
        }
        // l_dvx x-term: W_dxx = (1+bx)*A_dx + bx*dm_vxx
        T ddx = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - k + 1;
            const long off_p = off + k;
            const int xm = x - k + 1, xp = x + k;
            const T lam_m = lamb_s[off_m - off_s];
            const T l2m_m = lam_m + (T)2 * mu_s[off_m - off_s];
            const T A_dm = dtv * (l2m_m * l_dsxx[off_m] + lam_m * l_dsyy[off_m]);
            const T lam_p = lamb_s[off_p - off_s];
            const T l2m_p = lam_p + (T)2 * mu_s[off_p - off_s];
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
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            grad_buoyancy_x[off] += scale * (vx_new * dvxdb_store[soff]
                                             + dvx_new * ddvxdb_store[soff]);
            grad_dbuoyancy_x[off] += scale * (dvx_new * dvxdb_store[soff]);
        }
    }
}

// ---------------- backward: transpose of the combined velocity update ----------------
// From the post-velocity adjoint velocities produces the pre-velocity
// adjoint stresses (via the derivative + alternating m_sigma* memory
// transposes), the in-place m_v* adjoint-memory updates, and the
// lamb/mu/mu_yx (and scatter) imaging conditions.
template <typename T, int FD_PAD>
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
    T rdy, T rdx, T dtv,
    int t, int interval, T scale, int n_shots, int ny, int nx,
    int model_batched, int scatter_batched, int64_t snap_off)
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

    // sigmaii: y in [FD_PAD, ny-(FD_PAD-1)), x in [FD_PAD, nx-(FD_PAD-1))
    if (y >= FD_PAD && x >= FD_PAD && y < ny - (FD_PAD - 1) &&
        x < nx - (FD_PAD - 1)) {
        const long off = off_s + (long)y * nx + x;
        const T syy_v = l_syy[off];
        const T sxx_v = l_sxx[off];
        const T dsyy_v = l_dsyy[off];
        const T dsxx_v = l_dsxx[off];
        if (t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            const T dy_bg = dvydy_store[soff];
            const T dx_bg = dvxdx_store[soff];
            const T dy_sc = ddvydy_store[soff];
            const T dx_sc = ddvxdx_store[soff];
            grad_lamb[off] += scale * ((dy_bg + dx_bg) * (syy_v + sxx_v)
                                     + (dy_sc + dx_sc) * (dsyy_v + dsxx_v));
            grad_dlamb[off] += scale * ((dy_bg + dx_bg) * (dsyy_v + dsxx_v));
            grad_mu[off] += scale * (T)2 * (dy_bg * syy_v + dx_bg * sxx_v
                                          + dy_sc * dsyy_v + dx_sc * dsxx_v);
            grad_dmu[off] += scale * (T)2 * (dy_bg * dsyy_v + dx_bg * dsxx_v);
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
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - (long)k * nx;
            const long off_p = off + (long)(k - 1) * nx;
            const int ym = y - k, yp = y + k - 1;
            dy += c[k - 1] *
                  (dtv * ((T)1 + byh[ym]) * by_s[off_m - off_s] * l_vy[off_m] +
                       dtv * ((T)1 + byh[ym]) * dby_s[off_m - off_s] * l_dvy[off_m] +
                       byh[ym] * m_syyy_old[off_m] -
                   (dtv * ((T)1 + byh[yp]) * by_s[off_p - off_s] * l_vy[off_p] +
                       dtv * ((T)1 + byh[yp]) * dby_s[off_p - off_s] * l_dvy[off_p] +
                       byh[yp] * m_syyy_old[off_p]));
        }
        l_syy[off] += dy * rdy;
        // l_dsyy: W_yy_sc = dtv*(1+byh)*by_s*l_dvy + byh*dm_syyy_old
        T ddy = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - (long)k * nx;
            const long off_p = off + (long)(k - 1) * nx;
            const int ym = y - k, yp = y + k - 1;
            ddy += c[k - 1] *
                   (dtv * ((T)1 + byh[ym]) * by_s[off_m - off_s] * l_dvy[off_m] +
                       byh[ym] * dm_syyy_old[off_m] -
                    (dtv * ((T)1 + byh[yp]) * by_s[off_p - off_s] * l_dvy[off_p] +
                       byh[yp] * dm_syyy_old[off_p]));
        }
        l_dsyy[off] += ddy * rdy;
        // l_sxx: -diff_half_x(W_xx'), W_xx' = dtv*(1+bxh)*bx_s*l_vx
        //                                     + dtv*(1+bxh)*dbx_s*l_dvx + bxh*m_syxx_old
        T dx = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - k;
            const long off_p = off + k - 1;
            const int xm = x - k, xp = x + k - 1;
            dx += c[k - 1] *
                  (dtv * ((T)1 + bxh[xm]) * bx_s[off_m - off_s] * l_vx[off_m] +
                       dtv * ((T)1 + bxh[xm]) * dbx_s[off_m - off_s] * l_dvx[off_m] +
                       bxh[xm] * m_syxx_old[off_m] -
                   (dtv * ((T)1 + bxh[xp]) * bx_s[off_p - off_s] * l_vx[off_p] +
                       dtv * ((T)1 + bxh[xp]) * dbx_s[off_p - off_s] * l_dvx[off_p] +
                       bxh[xp] * m_syxx_old[off_p]));
        }
        l_sxx[off] += dx * rdx;
        // l_dsxx: W_xx'_sc = dtv*(1+bxh)*bx_s*l_dvx + bxh*dm_syxx_old
        T ddx = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - k;
            const long off_p = off + k - 1;
            const int xm = x - k, xp = x + k - 1;
            ddx += c[k - 1] *
                   (dtv * ((T)1 + bxh[xm]) * bx_s[off_m - off_s] * l_dvx[off_m] +
                       bxh[xm] * dm_syxx_old[off_m] -
                    (dtv * ((T)1 + bxh[xp]) * bx_s[off_p - off_s] * l_dvx[off_p] +
                       bxh[xp] * dm_syxx_old[off_p]));
        }
        l_dsxx[off] += ddx * rdx;
    }
    // sigmaxy: y in [FD_PAD, ny-FD_PAD), x in [FD_PAD, nx-FD_PAD)
    if (y >= FD_PAD && x >= FD_PAD && y < ny - FD_PAD &&
        x < nx - FD_PAD) {
        const long off = off_s + (long)y * nx + x;
        const T sxy_v = l_sxy[off];
        const T dsxy_v = l_dsxy[off];
        if (t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            grad_mu_yx[off] += scale * (dtv * sxy_v * dvxy_store[soff]
                                      + dtv * dsxy_v * ddvxy_store[soff]);
            grad_dmu_yx[off] += scale * (dtv * dsxy_v * dvxy_store[soff]);
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
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - (long)(k - 1) * nx;
            const long off_p = off + (long)k * nx;
            const int ym = y - (k - 1), yp = y + k;
            dy += c[k - 1] *
                  (dtv * ((T)1 + by[ym]) * bx_s[off_m - off_s] * l_vx[off_m] +
                       dtv * ((T)1 + by[ym]) * dbx_s[off_m - off_s] * l_dvx[off_m] +
                       by[ym] * m_syxy_old[off_m] -
                   (dtv * ((T)1 + by[yp]) * bx_s[off_p - off_s] * l_vx[off_p] +
                       dtv * ((T)1 + by[yp]) * dbx_s[off_p - off_s] * l_dvx[off_p] +
                       by[yp] * m_syxy_old[off_p]));
        }
        l_sxy[off] += dy * rdy;
        // l_dsxy y-term: W_sxy_y_sc = dtv*(1+by)*bx_s*l_dvx + by*dm_syxy_old
        T ddy = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - (long)(k - 1) * nx;
            const long off_p = off + (long)k * nx;
            const int ym = y - (k - 1), yp = y + k;
            ddy += c[k - 1] *
                   (dtv * ((T)1 + by[ym]) * bx_s[off_m - off_s] * l_dvx[off_m] +
                       by[ym] * dm_syxy_old[off_m] -
                    (dtv * ((T)1 + by[yp]) * bx_s[off_p - off_s] * l_dvx[off_p] +
                       by[yp] * dm_syxy_old[off_p]));
        }
        l_dsxy[off] += ddy * rdy;
        // l_sxy x-term: -diff_int_x(W_sxy_x), W_sxy_x = dtv*(1+bx)*by_s*l_vy
        //                                             + dtv*(1+bx)*dby_s*l_dvy + bx*m_syx_old
        T dx = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - k + 1;
            const long off_p = off + k;
            const int xm = x - k + 1, xp = x + k;
            dx += c[k - 1] *
                  (dtv * ((T)1 + bx[xm]) * by_s[off_m - off_s] * l_vy[off_m] +
                       dtv * ((T)1 + bx[xm]) * dby_s[off_m - off_s] * l_dvy[off_m] +
                       bx[xm] * m_syx_old[off_m] -
                   (dtv * ((T)1 + bx[xp]) * by_s[off_p - off_s] * l_vy[off_p] +
                       dtv * ((T)1 + bx[xp]) * dby_s[off_p - off_s] * l_dvy[off_p] +
                       bx[xp] * m_syx_old[off_p]));
        }
        l_sxy[off] += dx * rdx;
        // l_dsxy x-term: W_sxy_x_sc = dtv*(1+bx)*by_s*l_dvy + bx*dm_syx_old
        T ddx = (T)0;
        for (int k = 1; k <= FD_PAD; ++k) {
            const long off_m = off - k + 1;
            const long off_p = off + k;
            const int xm = x - k + 1, xp = x + k;
            ddx += c[k - 1] *
                   (dtv * ((T)1 + bx[xm]) * by_s[off_m - off_s] * l_dvy[off_m] +
                       bx[xm] * dm_syx_old[off_m] -
                    (dtv * ((T)1 + bx[xp]) * by_s[off_p - off_s] * l_dvy[off_p] +
                       bx[xp] * dm_syx_old[off_p]));
        }
        l_dsxy[off] += ddx * rdx;
    }
}

// ---------------- launchers ----------------
#define LAUNCH_GRID_FD(KERN, T, FP, ...)                                       \
    {                                                                          \
        dim3 block(16, 16);                                                    \
        dim3 grid((nx + 15) / 16, (ny + 15) / 16, n_shots);                    \
        KERN<T, FP><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami elastic2d_born " #KERN " failed"); \
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
            default: TORCH_CHECK(false, "nami elastic2d_born: unsupported fd_pad"); \
        }                                                                      \
    }

#define LAUNCH_PT(KERN, T, NSH, NLOC, ...)                                     \
    {                                                                          \
        dim3 block(32, 4);                                                     \
        dim3 grid(((NSH) + 31) / 32, ((NLOC) + 3) / 4);                        \
        KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami elastic2d_born " #KERN " failed"); \
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
    int64_t model_batched, int64_t scatter_batched, int64_t store, int64_t snap_off)
{
    const int n_shots = vy.size(0), ny = vy.size(1), nx = vy.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(vy.scalar_type(), "elastic2d_born_born_step_velocity", [&] {
        LAUNCH_BORN_GRID(born_step_velocity_kernel, scalar_t,
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
            (scalar_t)rdy, (scalar_t)rdx, (scalar_t)dtv,
            (int)t, (int)interval, n_shots, ny, nx,
            (int)model_batched, (int)scatter_batched, (int)store, (int64_t)snap_off);
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
    int64_t model_batched, int64_t scatter_batched, int64_t store, int64_t snap_off)
{
    const int n_shots = vy.size(0), ny = vy.size(1), nx = vy.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(vy.scalar_type(), "elastic2d_born_born_step_stress", [&] {
        LAUNCH_BORN_GRID(born_step_stress_kernel, scalar_t,
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
            (scalar_t)rdy, (scalar_t)rdx, (scalar_t)dtv,
            (int)t, (int)interval, n_shots, ny, nx,
            (int)model_batched, (int)scatter_batched, (int)store, (int64_t)snap_off);
    });
}

void born_inject_pressure(
    torch::Tensor syy, torch::Tensor sxx, torch::Tensor f, torch::Tensor src_i,
    int64_t t, int64_t n_shots, int64_t n_src, int64_t ny_nx)
{
    AT_DISPATCH_FLOATING_TYPES(syy.scalar_type(), "elastic2d_born_born_inject_pressure", [&] {
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
    AT_DISPATCH_FLOATING_TYPES(dsyy.scalar_type(), "elastic2d_born_born_record_pressure", [&] {
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
    AT_DISPATCH_FLOATING_TYPES(l_syy.scalar_type(), "elastic2d_born_born_record_grad_f_p", [&] {
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
    AT_DISPATCH_FLOATING_TYPES(l_dsyy.scalar_type(), "elastic2d_born_born_add_grad_r", [&] {
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
    int64_t t, int64_t interval, double scale,
    int64_t model_batched, int64_t scatter_batched, int64_t snap_off)
{
    const int n_shots = l_vy.size(0), ny = l_vy.size(1), nx = l_vy.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(l_vy.scalar_type(), "elastic2d_born_born_adjoint_velocity", [&] {
        LAUNCH_BORN_GRID(born_adjoint_velocity_kernel, scalar_t,
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
            (scalar_t)rdy, (scalar_t)rdx, (scalar_t)dtv,
            (int)t, (int)interval, (scalar_t)scale, n_shots, ny, nx,
            (int)model_batched, (int)scatter_batched, (int64_t)snap_off);
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
    int64_t t, int64_t interval, double scale,
    int64_t model_batched, int64_t scatter_batched, int64_t snap_off)
{
    const int n_shots = l_vy.size(0), ny = l_vy.size(1), nx = l_vy.size(2);
    CHECK_CONTIG(c);
    AT_DISPATCH_FLOATING_TYPES(l_vy.scalar_type(), "elastic2d_born_born_adjoint_stress", [&] {
        LAUNCH_BORN_GRID(born_adjoint_stress_kernel, scalar_t,
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
            (scalar_t)rdy, (scalar_t)rdx, (scalar_t)dtv,
            (int)t, (int)interval, (scalar_t)scale, n_shots, ny, nx,
            (int)model_batched, (int)scatter_batched, (int64_t)snap_off);
    });
}

// ---------------- whole-loop drivers ----------------
// One pybind call runs the entire forward/adjoint pass (deepwave-style):
// the per-step functions above are reused unchanged with the same argument
// order the Python loops used, so results are bitwise identical.

using nami_storage::ckpt_restore;
using nami_storage::ckpt_save;
using nami_storage::zero_buffers;

// Elastic2d Born forward.  state: 26 flat buffers, N_STATE layout
//   [0]vy [1]vx [2]syy [3]sxx [4]sxy
//   [5]m_vyy [6]m_vxx [7]m_vxy [8]m_vyx
//   [9]m_sigmayyy [10]m_sigmaxyx [11]m_sigmaxyy [12]m_sigmaxxx
//   [13..25] scattered counterparts (dvy, dvx, dsyy, dsxx, dsxy, dm_*)
void born_elastic_forward_loop(
    torch::Tensor lamb, torch::Tensor mu, torch::Tensor mu_yx,
    torch::Tensor buoyancy_y, torch::Tensor buoyancy_x,
    torch::Tensor dlamb, torch::Tensor dmu, torch::Tensor dmu_yx,
    torch::Tensor dbuoyancy_y, torch::Tensor dbuoyancy_x,
    std::vector<torch::Tensor> state,
    torch::Tensor dvydy_store, torch::Tensor dvxdx_store,
    torch::Tensor dvxy_store,
    torch::Tensor dvydb_store, torch::Tensor dvxdb_store,
    torch::Tensor ddvydy_store, torch::Tensor ddvxdx_store,
    torch::Tensor ddvxy_store,
    torch::Tensor ddvydb_store, torch::Tensor ddvxdb_store,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    torch::Tensor f, torch::Tensor src_i, torch::Tensor r, torch::Tensor rec_i,
    torch::Tensor r_bg, torch::Tensor bg_rec_i,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv,
    int64_t nt, int64_t interval,
    int64_t model_batched, int64_t scatter_batched,
    int64_t store, int64_t checkpoint_every,
    c10::optional<torch::Tensor> ckpt_state,
    // Wavefield I/O (deepwave-style): init_state/final_state use the N_STATE
    // layout above (same as ckpt).  The Born buffers are flat (updated in
    // place), so restore/copy each slot verbatim — a split run (continuation
    // via init_state) then bitwise matches a one-shot run.
    c10::optional<torch::Tensor> init_state,
    c10::optional<torch::Tensor> final_state,
    // Per-step forward callback: called every `callback_frequency` steps as
    // callback(t, nt, background physics fields..., scattered physics
    // fields...) — live padded tensors; PML memory variables remain internal.
    py::object callback, int64_t callback_frequency)
{
    TORCH_CHECK(state.size() == 26,
                "nami elastic2d_born elastic forward_loop: bad state size");
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
            callback(t, nt,
                     state[0], state[1], state[2], state[3], state[4],
                     state[13], state[14], state[15], state[16], state[17]);
        }
        // receivers record the PRE-step scattered stresses
        if (n_rec > 0)
            born_record_pressure(state[15], state[16], r, rec_i,
                                 t, n_shots, n_rec, ny_nx);
        if (n_bg_rec > 0)
            born_record_pressure(state[2], state[3], r_bg, bg_rec_i,
                                 t, n_shots, n_bg_rec, ny_nx);
        const int64_t snap_off = store ? (t / interval) * shot_count : 0;
        born_step_velocity(
            state[0], state[1], state[2], state[3], state[4],
            state[13], state[14], state[15], state[16], state[17],
            state[9], state[10], state[11], state[12],
            state[22], state[23], state[24], state[25],
            buoyancy_y, buoyancy_x, dbuoyancy_y, dbuoyancy_x,
            dvydb_store, dvxdb_store, ddvydb_store, ddvxdb_store,
            ayh, byh, ay, by, axh, bxh, ax, bx, c,
            fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
            rdy, rdx, dtv, t, interval,
            model_batched, scatter_batched, store, snap_off);
        born_step_stress(
            state[0], state[1], state[13], state[14],
            state[2], state[3], state[4], state[15], state[16], state[17],
            state[5], state[6], state[7], state[8],
            state[18], state[19], state[20], state[21],
            lamb, mu, mu_yx, dlamb, dmu, dmu_yx,
            dvydy_store, dvxdx_store, dvxy_store,
            ddvydy_store, ddvxdx_store, ddvxy_store,
            ayh, byh, ay, by, axh, bxh, ax, bx, c,
            fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
            rdy, rdx, dtv, t, interval,
            model_batched, scatter_batched, store, snap_off);
        // pressure sources inject into the background stresses only
        if (n_src > 0)
            born_inject_pressure(state[2], state[3], f, src_i,
                                 t, n_shots, n_src, ny_nx);
    }

    // Optional final state: the buffers already hold the state at time nt
    // (fields are updated in place), so a continuation restores them as-is.
    if (final_state.has_value()) {
        auto c = *final_state;
        for (size_t i = 0; i < state.size(); ++i)
            c[i].copy_(state[i]);
    }
}

// Elastic2d Born adjoint.  m_sig_a/m_sig_b (+dm_*): parity double banks of 4
// (old = b if (nt-1-t)%2 else a; new = a if parity else b).  state_f: 26
// replay buffers (forward N_STATE layout; empty = full storage).
void born_elastic_adjoint_loop(
    torch::Tensor lamb, torch::Tensor mu, torch::Tensor mu_yx,
    torch::Tensor buoyancy_y, torch::Tensor buoyancy_x,
    torch::Tensor dlamb, torch::Tensor dmu, torch::Tensor dmu_yx,
    torch::Tensor dbuoyancy_y, torch::Tensor dbuoyancy_x,
    torch::Tensor dvydy_store, torch::Tensor dvxdx_store,
    torch::Tensor dvxy_store,
    torch::Tensor dvydb_store, torch::Tensor dvxdb_store,
    torch::Tensor ddvydy_store, torch::Tensor ddvxdx_store,
    torch::Tensor ddvxy_store,
    torch::Tensor ddvydb_store, torch::Tensor ddvxdb_store,
    torch::Tensor grad_lamb, torch::Tensor grad_mu, torch::Tensor grad_mu_yx,
    torch::Tensor grad_dlamb, torch::Tensor grad_dmu,
    torch::Tensor grad_dmu_yx,
    torch::Tensor grad_by, torch::Tensor grad_bx,
    torch::Tensor grad_dby, torch::Tensor grad_dbx,
    torch::Tensor grad_f, torch::Tensor grad_r, torch::Tensor grad_r_bg,
    torch::Tensor src_i, torch::Tensor rec_i, torch::Tensor bg_rec_i,
    torch::Tensor f,
    torch::Tensor l_vy, torch::Tensor l_vx,
    torch::Tensor l_dvy, torch::Tensor l_dvx,
    torch::Tensor l_syy, torch::Tensor l_sxx, torch::Tensor l_sxy,
    torch::Tensor l_dsyy, torch::Tensor l_dsxx, torch::Tensor l_dsxy,
    torch::Tensor m_vyy, torch::Tensor m_vxx,
    torch::Tensor m_vxy, torch::Tensor m_vyx,
    torch::Tensor dm_vyy, torch::Tensor dm_vxx,
    torch::Tensor dm_vxy, torch::Tensor dm_vyx,
    std::vector<torch::Tensor> m_sig_a, std::vector<torch::Tensor> m_sig_b,
    std::vector<torch::Tensor> dm_sig_a, std::vector<torch::Tensor> dm_sig_b,
    std::vector<torch::Tensor> state_f,
    torch::Tensor ayh, torch::Tensor byh, torch::Tensor ay, torch::Tensor by,
    torch::Tensor axh, torch::Tensor bxh, torch::Tensor ax, torch::Tensor bx,
    torch::Tensor c,
    int64_t fd_pad_y0, int64_t fd_pad_y1, int64_t fd_pad_x0, int64_t fd_pad_x1,
    double rdy, double rdx, double dtv, double scale,
    int64_t nt, int64_t interval,
    int64_t model_batched, int64_t scatter_batched,
    torch::Tensor segments,  // int64 [n_seg, 2] on CPU; empty = full storage
    c10::optional<torch::Tensor> ckpt_state)
{
    TORCH_CHECK(m_sig_a.size() == 4 && m_sig_b.size() == 4 &&
                dm_sig_a.size() == 4 && dm_sig_b.size() == 4,
                "nami elastic2d_born elastic adjoint_loop: bad parity bank sizes");
    const int64_t n_shots = l_vy.size(0);
    const int64_t ny = l_vy.size(1), nx = l_vy.size(2);
    const int64_t n_src = src_i.size(1), n_rec = rec_i.size(1);
    const int64_t n_bg_rec = bg_rec_i.size(1);
    const int64_t ny_nx = ny * nx;
    const int64_t shot_count = n_shots * ny_nx;

    auto adjoint_at = [&](int64_t t, int64_t snap_off) {
        // nt - 1 - t >= 0, so C++ % matches Python's modulo here.
        const int64_t parity = (nt - 1 - t) % 2;
        const auto& old = parity ? m_sig_b : m_sig_a;
        const auto& new_ = parity ? m_sig_a : m_sig_b;
        const auto& dold = parity ? dm_sig_b : dm_sig_a;
        const auto& dnew = parity ? dm_sig_a : dm_sig_b;
        if (n_src > 0)
            born_record_grad_f_p(l_syy, l_sxx, grad_f, src_i,
                                 t, n_shots, n_src, ny_nx);
        born_adjoint_velocity(
            lamb, mu, mu_yx, dlamb, dmu, dmu_yx,
            l_vy, l_vx, l_dvy, l_dvx,
            l_syy, l_sxx, l_sxy, l_dsyy, l_dsxx, l_dsxy,
            m_vyy, m_vxx, m_vxy, m_vyx,
            dm_vyy, dm_vxx, dm_vxy, dm_vyx,
            old[0], old[1], old[2], old[3],
            new_[0], new_[1], new_[2], new_[3],
            dold[0], dold[1], dold[2], dold[3],
            dnew[0], dnew[1], dnew[2], dnew[3],
            buoyancy_y, buoyancy_x, dbuoyancy_y, dbuoyancy_x,
            grad_by, grad_bx, grad_dby, grad_dbx,
            dvydb_store, dvxdb_store, ddvydb_store, ddvxdb_store,
            ayh, byh, ay, by, axh, bxh, ax, bx, c,
            fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
            rdy, rdx, dtv, t, interval, scale,
            model_batched, scatter_batched, snap_off);
        born_adjoint_stress(
            buoyancy_y, buoyancy_x, dbuoyancy_y, dbuoyancy_x,
            l_vy, l_vx, l_dvy, l_dvx,
            l_syy, l_sxx, l_sxy, l_dsyy, l_dsxx, l_dsxy,
            m_vyy, m_vxx, m_vxy, m_vyx,
            dm_vyy, dm_vxx, dm_vxy, dm_vyx,
            lamb, mu, mu_yx, dlamb, dmu, dmu_yx,
            old[0], old[1], old[2], old[3],
            dold[0], dold[1], dold[2], dold[3],
            grad_lamb, grad_mu, grad_mu_yx,
            grad_dlamb, grad_dmu, grad_dmu_yx,
            dvydy_store, dvxdx_store, dvxy_store,
            ddvydy_store, ddvxdx_store, ddvxy_store,
            ayh, byh, ay, by, axh, bxh, ax, bx, c,
            fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
            rdy, rdx, dtv, t, interval, scale,
            model_batched, scatter_batched, snap_off);
        if (n_rec > 0)
            born_add_grad_r(l_dsyy, l_dsxx, grad_r, rec_i,
                            t, n_shots, n_rec, ny_nx);
        if (n_bg_rec > 0)
            born_add_grad_r(l_syy, l_sxx, grad_r_bg, bg_rec_i,
                            t, n_shots, n_bg_rec, ny_nx);
    };

    if (segments.numel() == 0) {
        for (int64_t t = nt - 1; t >= 0; --t)
            adjoint_at(t, (t / interval) * shot_count);
        return;
    }

    // Checkpointed backward: per segment, restore the forward state, replay
    // the forward steps to regenerate the snapshots, then run the adjoint.
    TORCH_CHECK(state_f.size() == 26,
                "nami elastic2d_born elastic adjoint_loop: bad replay state size");
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
            born_step_velocity(
                state_f[0], state_f[1], state_f[2], state_f[3], state_f[4],
                state_f[13], state_f[14], state_f[15], state_f[16], state_f[17],
                state_f[9], state_f[10], state_f[11], state_f[12],
                state_f[22], state_f[23], state_f[24], state_f[25],
                buoyancy_y, buoyancy_x, dbuoyancy_y, dbuoyancy_x,
                dvydb_store, dvxdb_store, ddvydb_store, ddvxdb_store,
                ayh, byh, ay, by, axh, bxh, ax, bx, c,
                fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
                rdy, rdx, dtv, t, interval,
                model_batched, scatter_batched, 1, snap_off);
            born_step_stress(
                state_f[0], state_f[1], state_f[13], state_f[14],
                state_f[2], state_f[3], state_f[4],
                state_f[15], state_f[16], state_f[17],
                state_f[5], state_f[6], state_f[7], state_f[8],
                state_f[18], state_f[19], state_f[20], state_f[21],
                lamb, mu, mu_yx, dlamb, dmu, dmu_yx,
                dvydy_store, dvxdx_store, dvxy_store,
                ddvydy_store, ddvxdx_store, ddvxy_store,
                ayh, byh, ay, by, axh, bxh, ax, bx, c,
                fd_pad_y0, fd_pad_y1, fd_pad_x0, fd_pad_x1,
                rdy, rdx, dtv, t, interval,
                model_batched, scatter_batched, 1, snap_off);
            if (n_src > 0)
                born_inject_pressure(state_f[2], state_f[3], f, src_i,
                                     t, n_shots, n_src, ny_nx);
        }
        for (int64_t t = s1 - 1; t >= s0; --t)
            adjoint_at(t, ((t - s0) / interval) * shot_count);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("born_step_velocity", &born_step_velocity);
    m.def("born_step_stress", &born_step_stress);
    m.def("born_inject_pressure", &born_inject_pressure);
    m.def("born_record_pressure", &born_record_pressure);
    m.def("born_record_grad_f_p", &born_record_grad_f_p);
    m.def("born_add_grad_r", &born_add_grad_r);
    m.def("born_adjoint_velocity", &born_adjoint_velocity);
    m.def("born_adjoint_stress", &born_adjoint_stress);
    NAMI_STORAGE_PYBIND(m);
    m.def("born_elastic_forward_loop", &born_elastic_forward_loop);
    m.def("born_elastic_adjoint_loop", &born_elastic_adjoint_loop);
}
