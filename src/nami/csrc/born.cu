#include <torch/extension.h>
#include "storage.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------- finite-difference helpers (regular grid) ----------------
// Same coefficient-array-driven stencils as scalar2d.cu: c1[4] first-derivative
// coefficients for offsets 1..4, c2[5] = [center, off1..off4] second-derivative
// coefficients; loops run to the runtime fd_pad (never a hardcoded max).
template <typename T>
__device__ __forceinline__ T diff2_y(const T* u, long off, const T* c2, T rdy2, int nx, int fd_pad)
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
__device__ __forceinline__ T diff1_y(const T* u, long off, const T* c1, T rdy, int nx, int fd_pad)
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
    T rdy, T rdx, T rdy2, T rdx2,
    int t, int interval, T dt2,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int v_batched, int scatter_batched, int store, int64_t snap_off, int fd_pad)
{
    const int s = blockIdx.z;
    const int y = pml_y0 + blockIdx.y * blockDim.y + threadIdx.y;
    const int x = pml_x0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= 1 && x >= 1 && y < ny - 1 && x < nx - 1 && y < pml_y1 && x < pml_x1) {
        const int s_v = v_batched ? s : 0;
        const int s_sc = scatter_batched ? s : 0;
        const T* vs = v + (long)s_v * ny * nx;
        const T* scs = scatter + (long)s_sc * ny * nx;
        const long off = ((long)s * ny + y) * nx + x;
        const T v_val = vs[(long)y * nx + x];
        const T v2dt2 = v_val * v_val * dt2;
        T w_sum = diff2_y(u_cur, off, c2, rdy2, nx, fd_pad)
                + diff2_x(u_cur, off, c2, rdx2, fd_pad);
        u_new[off] = v2dt2 * w_sum + (T)2 * u_cur[off] - u_prev[off];
        T wsc_sum = diff2_y(u_sc_cur, off, c2, rdy2, nx, fd_pad)
                  + diff2_x(u_sc_cur, off, c2, rdx2, fd_pad);
        u_sc_new[off] = v2dt2 * wsc_sum + (T)2 * u_sc_cur[off] - u_sc_prev[off]
                      + (T)2 * v_val * scs[(long)y * nx + x] * dt2 * w_sum;
        if (store && t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            w_store[soff] = w_sum;
            wsc_store[soff] = wsc_sum;
        }
    }
}

// ---------------- forward: one time step (frame: includes PML border) ----------------
template <typename T>
__global__ void step_forward_frame_kernel(
    const T* __restrict__ v, const T* __restrict__ scatter,
    const T* __restrict__ u_cur, const T* __restrict__ u_prev,
    const T* __restrict__ u_sc_cur, const T* __restrict__ u_sc_prev,
    const T* __restrict__ psi_y, const T* __restrict__ psi_x,
    const T* __restrict__ zeta_y, const T* __restrict__ zeta_x,
    const T* __restrict__ psi_y_sc, const T* __restrict__ psi_x_sc,
    const T* __restrict__ zeta_y_sc, const T* __restrict__ zeta_x_sc,
    T* __restrict__ u_new, T* __restrict__ u_sc_new,
    T* __restrict__ psi_y_new, T* __restrict__ psi_x_new,
    T* __restrict__ zeta_y_new, T* __restrict__ zeta_x_new,
    T* __restrict__ psi_y_sc_new, T* __restrict__ psi_x_sc_new,
    T* __restrict__ zeta_y_sc_new, T* __restrict__ zeta_x_sc_new,
    const T* __restrict__ ay, const T* __restrict__ by, const T* __restrict__ dbydy,
    const T* __restrict__ ax, const T* __restrict__ bx, const T* __restrict__ dbxdx,
    T* __restrict__ w_store, T* __restrict__ wsc_store,
    const T* __restrict__ c1, const T* __restrict__ c2,
    T rdy, T rdx, T rdy2, T rdx2,
    int t, int interval, T dt2,
    int n_shots, int ny, int nx,
    int pml_y0, int pml_y1, int pml_x0, int pml_x1,
    int v_batched, int scatter_batched, int store, int64_t snap_off, int fd_pad)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    // strictly interior cells are handled by step_forward_interior_kernel
    if (y >= pml_y0 && y < pml_y1 && x >= pml_x0 && x < pml_x1)
        return;
    if (y >= fd_pad && x >= fd_pad && y < ny - fd_pad && x < nx - fd_pad) {
        const int s_v = v_batched ? s : 0;
        const int s_sc = scatter_batched ? s : 0;
        const T* vs = v + (long)s_v * ny * nx;
        const T* scs = scatter + (long)s_sc * ny * nx;
        const long off = ((long)s * ny + y) * nx + x;
        const T v_val = vs[(long)y * nx + x];
        const T v2dt2 = v_val * v_val * dt2;
        T w_sum = (T)0;
        T wsc_sum = (T)0;

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
                      + (T)2 * v_val * scs[(long)y * nx + x] * dt2 * w_sum;
        if (store && t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            w_store[soff] = w_sum;
            wsc_store[soff] = wsc_sum;
        }
    }
}

// ---------------- forward: source injection / receiver recording ----------------
template <typename T>
__global__ void inject_kernel(
    T* __restrict__ u_new, T* __restrict__ u_sc_new,
    const T* __restrict__ f, const T* __restrict__ f_sc, const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        if (idx >= 0) {
            u_new[(long)s * ny_nx + idx] += f[(((long)t * n_shots + s) * n_src + k)];
            u_sc_new[(long)s * ny_nx + idx] += f_sc[(((long)t * n_shots + s) * n_src + k)];
        }
    }
}

template <typename T>
__global__ void record_kernel(
    const T* __restrict__ u_cur, const T* __restrict__ u_sc_cur,
    T* __restrict__ r, T* __restrict__ r_sc,
    const long* __restrict__ rec_i, const long* __restrict__ rec_sc_i,
    int t, int n_shots, int n_rec, int n_rec_sc, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            r[(((long)t * n_shots + s) * n_rec + k)] = u_cur[(long)s * ny_nx + idx];
    }
    if (s < n_shots && k < n_rec_sc) {
        const long idx = rec_sc_i[(long)s * n_rec_sc + k];
        if (idx >= 0)
            r_sc[(((long)t * n_shots + s) * n_rec_sc + k)] = u_sc_cur[(long)s * ny_nx + idx];
    }
}

// ---------------- backward: adjoint step (interior: no PML) ----------------
// Exact discrete transpose of the coupled forward step: the background
// adjoint evolves with Laplacian of
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
    T rdy, T rdx, T rdy2, T rdx2,
    int t, int interval, T scale, T dt2,
    int n_shots, int ny, int nx,
    // backward (widened) PML boundaries, not the forward ones
    int pml_y0_b, int pml_y1_b, int pml_x0_b, int pml_x1_b,
    int v_batched, int scatter_batched, int64_t snap_off, int fd_pad)
{
    const int s = blockIdx.z;
    const int y = pml_y0_b + blockIdx.y * blockDim.y + threadIdx.y;
    const int x = pml_x0_b + blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= 1 && x >= 1 && y < ny - 1 && x < nx - 1 && y < pml_y1_b && x < pml_x1_b) {
        const int s_v = v_batched ? s : 0;
        const int s_sc = scatter_batched ? s : 0;
        const T* vs = v + (long)s_v * ny * nx;
        const T* scs = scatter + (long)s_sc * ny * nx;
        const long off = ((long)s * ny + y) * nx + x;
        const T v_val = vs[(long)y * nx + x];
        const T sc_val = scs[(long)y * nx + x];
        const T v2dt2 = v_val * v_val * dt2;
        T wy = c2[0] * (v2dt2 * lam_bg_next[off] + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off]);
#pragma unroll
        for (int k = 1; k <= fd_pad; ++k) {
            const T v_p = vs[(long)(y + k) * nx + x];
            const T v_m = vs[(long)(y - k) * nx + x];
            wy += c2[k] * (v_p * v_p * dt2 * lam_bg_next[off + k * nx]
                         + (T)2 * v_p * scs[(long)(y + k) * nx + x] * dt2 * lam_sc_next[off + k * nx]
                         + v_m * v_m * dt2 * lam_bg_next[off - k * nx]
                         + (T)2 * v_m * scs[(long)(y - k) * nx + x] * dt2 * lam_sc_next[off - k * nx]);
        }
        wy *= rdy2;
        T wx = c2[0] * (v2dt2 * lam_bg_next[off] + (T)2 * v_val * sc_val * dt2 * lam_sc_next[off]);
#pragma unroll
        for (int k = 1; k <= fd_pad; ++k) {
            const T v_p = vs[(long)y * nx + x + k];
            const T v_m = vs[(long)y * nx + x - k];
            wx += c2[k] * (v_p * v_p * dt2 * lam_bg_next[off + k]
                         + (T)2 * v_p * scs[(long)y * nx + x + k] * dt2 * lam_sc_next[off + k]
                         + v_m * v_m * dt2 * lam_bg_next[off - k]
                         + (T)2 * v_m * scs[(long)y * nx + x - k] * dt2 * lam_sc_next[off - k]);
        }
        wx *= rdx2;
        lam_bg_new[off] = (T)2 * lam_bg_next[off] + wy + wx - lam_bg_next2[off];

        T wysc = c2[0] * (v2dt2 * lam_sc_next[off]);
#pragma unroll
        for (int k = 1; k <= fd_pad; ++k) {
            const T v_p = vs[(long)(y + k) * nx + x];
            const T v_m = vs[(long)(y - k) * nx + x];
            wysc += c2[k] * (v_p * v_p * dt2 * lam_sc_next[off + k * nx]
                           + v_m * v_m * dt2 * lam_sc_next[off - k * nx]);
        }
        wysc *= rdy2;
        T wxsc = c2[0] * (v2dt2 * lam_sc_next[off]);
#pragma unroll
        for (int k = 1; k <= fd_pad; ++k) {
            const T v_p = vs[(long)y * nx + x + k];
            const T v_m = vs[(long)y * nx + x - k];
            wxsc += c2[k] * (v_p * v_p * dt2 * lam_sc_next[off + k]
                           + v_m * v_m * dt2 * lam_sc_next[off - k]);
        }
        wxsc *= rdx2;
        lam_sc_new[off] = (T)2 * lam_sc_next[off] + wysc + wxsc - lam_sc_next2[off];

        if (t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            grad_v[off] += lam_bg_next[off] * ((T)2 * v_val * dt2 * w_store[soff]) * scale
                         + lam_sc_next[off] * ((T)2 * dt2 * sc_val * w_store[soff]
                                             + (T)2 * v_val * dt2 * wsc_store[soff]) * scale;
            grad_scatter[off] += lam_sc_next[off] * (T)2 * v_val * dt2 * w_store[soff] * scale;
        }
    }
}

// ---------------- backward: adjoint step (frame: includes PML border) ----------------
// Transpose of the forward frame kernel with the same Born coupling as the
// adjoint interior step (see scalar2d.cu for the uncoupled case).
template <typename T>
__global__ void step_adjoint_frame_kernel(
    const T* __restrict__ v, const T* __restrict__ scatter,
    const T* __restrict__ lam_bg_next, const T* __restrict__ lam_bg_next2,
    const T* __restrict__ lam_sc_next, const T* __restrict__ lam_sc_next2,
    T* __restrict__ lam_bg_new, T* __restrict__ lam_sc_new,
    const T* __restrict__ psi_y, const T* __restrict__ psi_x,
    const T* __restrict__ zeta_y, const T* __restrict__ zeta_x,
    const T* __restrict__ psi_y_sc, const T* __restrict__ psi_x_sc,
    const T* __restrict__ zeta_y_sc, const T* __restrict__ zeta_x_sc,
    T* __restrict__ psi_y_new, T* __restrict__ psi_x_new,
    T* __restrict__ zeta_y_new, T* __restrict__ zeta_x_new,
    T* __restrict__ psi_y_sc_new, T* __restrict__ psi_x_sc_new,
    T* __restrict__ zeta_y_sc_new, T* __restrict__ zeta_x_sc_new,
    const T* __restrict__ ay, const T* __restrict__ by, const T* __restrict__ dbydy,
    const T* __restrict__ ax, const T* __restrict__ bx, const T* __restrict__ dbxdx,
    const T* __restrict__ w_store, const T* __restrict__ wsc_store,
    T* __restrict__ grad_v, T* __restrict__ grad_scatter,
    const T* __restrict__ c1, const T* __restrict__ c2,
    T rdy, T rdx, T rdy2, T rdx2,
    int t, int interval, T scale, T dt2,
    int n_shots, int ny, int nx,
    // backward (widened) PML boundaries, not the forward ones
    int pml_y0_b, int pml_y1_b, int pml_x0_b, int pml_x1_b,
    int v_batched, int scatter_batched, int64_t snap_off, int fd_pad)
{
    const int s = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    // strictly interior cells are handled by step_adjoint_interior_kernel
    if (y >= pml_y0_b && y < pml_y1_b && x >= pml_x0_b && x < pml_x1_b)
        return;
    if (y >= fd_pad && x >= fd_pad && y < ny - fd_pad && x < nx - fd_pad) {
        const int s_v = v_batched ? s : 0;
        const int s_sc = scatter_batched ? s : 0;
        const T* vs = v + (long)s_v * ny * nx;
        const T* scs = scatter + (long)s_sc * ny * nx;
        const long off = ((long)s * ny + y) * nx + x;
        const T v_val = vs[(long)y * nx + x];
        const T sc_val = scs[(long)y * nx + x];
        const T v2dt2 = v_val * v_val * dt2;
        T w_sum = (T)0;
        T wsc_sum = (T)0;

        // y: transpose of the CPML-modified Laplacian acting on lam_bg/lam_sc
        if (y < pml_y0_b || y >= pml_y1_b) {
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
                const T v_p = vs[(long)(y + k) * nx + x];
                const T v_m = vs[(long)(y - k) * nx + x];
                const T wfc_p = v_p * v_p * dt2 * lam_bg_next[off + k * nx]
                              + (T)2 * v_p * scs[(long)(y + k) * nx + x] * dt2 * lam_sc_next[off + k * nx];
                const T wfc_m = v_m * v_m * dt2 * lam_bg_next[off - k * nx]
                              + (T)2 * v_m * scs[(long)(y - k) * nx + x] * dt2 * lam_sc_next[off - k * nx];
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
                const T v_p = vs[(long)(y + k) * nx + x];
                const T v_m = vs[(long)(y - k) * nx + x];
                wy += c2[k] * (v_p * v_p * dt2 * lam_bg_next[off + k * nx]
                             + (T)2 * v_p * scs[(long)(y + k) * nx + x] * dt2 * lam_sc_next[off + k * nx]
                             + v_m * v_m * dt2 * lam_bg_next[off - k * nx]
                             + (T)2 * v_m * scs[(long)(y - k) * nx + x] * dt2 * lam_sc_next[off - k * nx]);
            }
            w_sum += wy * rdy2;
            T wysc = c2[0] * (v2dt2 * lam_sc_next[off]);
#pragma unroll
            for (int k = 1; k <= fd_pad; ++k) {
                const T v_p = vs[(long)(y + k) * nx + x];
                const T v_m = vs[(long)(y - k) * nx + x];
                wysc += c2[k] * (v_p * v_p * dt2 * lam_sc_next[off + k * nx]
                               + v_m * v_m * dt2 * lam_sc_next[off - k * nx]);
            }
            wsc_sum += wysc * rdy2;
        }
        // x: same structure
        if (x < pml_x0_b || x >= pml_x1_b) {
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
                const T v_p = vs[(long)y * nx + x + k];
                const T v_m = vs[(long)y * nx + x - k];
                const T wfc_p = v_p * v_p * dt2 * lam_bg_next[off + k]
                              + (T)2 * v_p * scs[(long)y * nx + x + k] * dt2 * lam_sc_next[off + k];
                const T wfc_m = v_m * v_m * dt2 * lam_bg_next[off - k]
                              + (T)2 * v_m * scs[(long)y * nx + x - k] * dt2 * lam_sc_next[off - k];
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
                const T v_p = vs[(long)y * nx + x + k];
                const T v_m = vs[(long)y * nx + x - k];
                wx += c2[k] * (v_p * v_p * dt2 * lam_bg_next[off + k]
                             + (T)2 * v_p * scs[(long)y * nx + x + k] * dt2 * lam_sc_next[off + k]
                             + v_m * v_m * dt2 * lam_bg_next[off - k]
                             + (T)2 * v_m * scs[(long)y * nx + x - k] * dt2 * lam_sc_next[off - k]);
            }
            w_sum += wx * rdx2;
            T wxsc = c2[0] * (v2dt2 * lam_sc_next[off]);
#pragma unroll
            for (int k = 1; k <= fd_pad; ++k) {
                const T v_p = vs[(long)y * nx + x + k];
                const T v_m = vs[(long)y * nx + x - k];
                wxsc += c2[k] * (v_p * v_p * dt2 * lam_sc_next[off + k]
                               + v_m * v_m * dt2 * lam_sc_next[off - k]);
            }
            wsc_sum += wxsc * rdx2;
        }

        lam_bg_new[off] = (T)2 * lam_bg_next[off] + w_sum - lam_bg_next2[off];
        lam_sc_new[off] = (T)2 * lam_sc_next[off] + wsc_sum - lam_sc_next2[off];
        if (t % interval == 0) {
            const long soff = snap_off + ((long)s * ny + y) * nx + x;
            grad_v[off] += lam_bg_next[off] * ((T)2 * v_val * dt2 * w_store[soff]) * scale
                         + lam_sc_next[off] * ((T)2 * dt2 * sc_val * w_store[soff]
                                             + (T)2 * v_val * dt2 * wsc_store[soff]) * scale;
            grad_scatter[off] += lam_sc_next[off] * (T)2 * v_val * dt2 * w_store[soff] * scale;
        }
    }
}

template <typename T>
__global__ void record_grad_f_kernel(
    const T* __restrict__ lam_bg_next, const T* __restrict__ lam_sc_next,
    T* __restrict__ grad_f, T* __restrict__ grad_f_sc, const long* __restrict__ src_i,
    int t, int n_shots, int n_src, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_src) {
        const long idx = src_i[(long)s * n_src + k];
        grad_f[(((long)t * n_shots + s) * n_src + k)] = (idx >= 0) ? lam_bg_next[(long)s * ny_nx + idx] : (T)0;
        grad_f_sc[(((long)t * n_shots + s) * n_src + k)] = (idx >= 0) ? lam_sc_next[(long)s * ny_nx + idx] : (T)0;
    }
}

template <typename T>
__global__ void record_grad_r_kernel(
    T* __restrict__ lam_bg_cur, T* __restrict__ lam_sc_cur,
    const T* __restrict__ grad_r, const T* __restrict__ grad_r_sc,
    const long* __restrict__ rec_i, const long* __restrict__ rec_sc_i,
    int t, int n_shots, int n_rec, int n_rec_sc, long ny_nx)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (s < n_shots && k < n_rec) {
        const long idx = rec_i[(long)s * n_rec + k];
        if (idx >= 0)
            lam_bg_cur[(long)s * ny_nx + idx] += grad_r[(((long)t * n_shots + s) * n_rec + k)];
    }
    if (s < n_shots && k < n_rec_sc) {
        const long idx = rec_sc_i[(long)s * n_rec_sc + k];
        if (idx >= 0)
            lam_sc_cur[(long)s * ny_nx + idx] += grad_r_sc[(((long)t * n_shots + s) * n_rec_sc + k)];
    }
}

// ---------------- launchers ----------------
#define LAUNCH_STEP(KERN, T, ...)                                              \
    {                                                                          \
        dim3 block(16, 16);                                                    \
        dim3 grid((nx + 15) / 16, (ny + 15) / 16, n_shots);                    \
        KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami born kernel failed"); \
    }

#define LAUNCH_STEP_INTERIOR(KERN, T, Y0, Y1, X0, X1, ...)                     \
    {                                                                          \
        dim3 block(16, 16);                                                    \
        dim3 grid(((X1) - (X0) + 15) / 16, ((Y1) - (Y0) + 15) / 16, n_shots); \
        if (((Y1) - (Y0)) > 0 && ((X1) - (X0)) > 0)                           \
            KERN<T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami born interior kernel failed"); \
    }

void forward_step(
    torch::Tensor v, torch::Tensor scatter,
    torch::Tensor u_cur, torch::Tensor u_prev, torch::Tensor u_sc_cur, torch::Tensor u_sc_prev,
    torch::Tensor psi_y, torch::Tensor psi_x, torch::Tensor zeta_y, torch::Tensor zeta_x,
    torch::Tensor psi_y_sc, torch::Tensor psi_x_sc, torch::Tensor zeta_y_sc, torch::Tensor zeta_x_sc,
    torch::Tensor u_new, torch::Tensor u_sc_new,
    torch::Tensor psi_y_new, torch::Tensor psi_x_new, torch::Tensor zeta_y_new, torch::Tensor zeta_x_new,
    torch::Tensor psi_y_sc_new, torch::Tensor psi_x_sc_new, torch::Tensor zeta_y_sc_new, torch::Tensor zeta_x_sc_new,
    torch::Tensor ay, torch::Tensor by, torch::Tensor dbydy,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor dbxdx,
    torch::Tensor w_store, torch::Tensor wsc_store,
    torch::Tensor c1, torch::Tensor c2,
    double rdy, double rdx, double rdy2, double rdx2,
    int64_t t, int64_t interval, double dt2,
    int64_t n_shots, int64_t ny, int64_t nx,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t v_batched, int64_t scatter_batched, int64_t store, int64_t snap_off, int64_t fd_pad)
{
    // v/scatter need not be batch-expanded: when v_batched/scatter_batched
    // is 0 the kernels read model slab 0 (spatial indices only).
    CHECK_CONTIG(v);
    CHECK_CONTIG(scatter);
    CHECK_CONTIG(u_cur);
    CHECK_CONTIG(u_prev);
    CHECK_CONTIG(u_sc_cur);
    CHECK_CONTIG(u_sc_prev);
    CHECK_CONTIG(u_new);
    CHECK_CONTIG(u_sc_new);
    CHECK_CONTIG(w_store);
    CHECK_CONTIG(wsc_store);
    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "forward_step", [&] {
        LAUNCH_STEP_INTERIOR(step_forward_interior_kernel, scalar_t,
            pml_y0, pml_y1, pml_x0, pml_x1,
            v.data_ptr<scalar_t>(), scatter.data_ptr<scalar_t>(),
            u_cur.data_ptr<scalar_t>(), u_prev.data_ptr<scalar_t>(),
            u_sc_cur.data_ptr<scalar_t>(), u_sc_prev.data_ptr<scalar_t>(),
            u_new.data_ptr<scalar_t>(), u_sc_new.data_ptr<scalar_t>(),
            w_store.data_ptr<scalar_t>(), wsc_store.data_ptr<scalar_t>(),
            c1.data_ptr<scalar_t>(), c2.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx, (scalar_t)rdy2, (scalar_t)rdx2,
            (int)t, (int)interval, (scalar_t)dt2,
            (int)n_shots, (int)ny, (int)nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)v_batched, (int)scatter_batched, (int)store, (int64_t)snap_off, (int)fd_pad);
        LAUNCH_STEP(step_forward_frame_kernel, scalar_t,
            v.data_ptr<scalar_t>(), scatter.data_ptr<scalar_t>(),
            u_cur.data_ptr<scalar_t>(), u_prev.data_ptr<scalar_t>(),
            u_sc_cur.data_ptr<scalar_t>(), u_sc_prev.data_ptr<scalar_t>(),
            psi_y.data_ptr<scalar_t>(), psi_x.data_ptr<scalar_t>(),
            zeta_y.data_ptr<scalar_t>(), zeta_x.data_ptr<scalar_t>(),
            psi_y_sc.data_ptr<scalar_t>(), psi_x_sc.data_ptr<scalar_t>(),
            zeta_y_sc.data_ptr<scalar_t>(), zeta_x_sc.data_ptr<scalar_t>(),
            u_new.data_ptr<scalar_t>(), u_sc_new.data_ptr<scalar_t>(),
            psi_y_new.data_ptr<scalar_t>(), psi_x_new.data_ptr<scalar_t>(),
            zeta_y_new.data_ptr<scalar_t>(), zeta_x_new.data_ptr<scalar_t>(),
            psi_y_sc_new.data_ptr<scalar_t>(), psi_x_sc_new.data_ptr<scalar_t>(),
            zeta_y_sc_new.data_ptr<scalar_t>(), zeta_x_sc_new.data_ptr<scalar_t>(),
            ay.data_ptr<scalar_t>(), by.data_ptr<scalar_t>(), dbydy.data_ptr<scalar_t>(),
            ax.data_ptr<scalar_t>(), bx.data_ptr<scalar_t>(), dbxdx.data_ptr<scalar_t>(),
            w_store.data_ptr<scalar_t>(), wsc_store.data_ptr<scalar_t>(),
            c1.data_ptr<scalar_t>(), c2.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx, (scalar_t)rdy2, (scalar_t)rdx2,
            (int)t, (int)interval, (scalar_t)dt2,
            (int)n_shots, (int)ny, (int)nx,
            (int)pml_y0, (int)pml_y1, (int)pml_x0, (int)pml_x1,
            (int)v_batched, (int)scatter_batched, (int)store, (int64_t)snap_off, (int)fd_pad);
    });
}

void inject(torch::Tensor u_new, torch::Tensor u_sc_new,
            torch::Tensor f, torch::Tensor f_sc, torch::Tensor src_i,
            int64_t t, int64_t n_shots, int64_t n_src, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_src + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(u_new.scalar_type(), "inject", [&] {
        inject_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            u_new.data_ptr<scalar_t>(), u_sc_new.data_ptr<scalar_t>(),
            f.data_ptr<scalar_t>(), f_sc.data_ptr<scalar_t>(), src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami born inject failed");
}

void record(torch::Tensor u_cur, torch::Tensor u_sc_cur,
            torch::Tensor r, torch::Tensor r_sc,
            torch::Tensor rec_i, torch::Tensor rec_sc_i,
            int64_t t, int64_t n_shots, int64_t n_rec, int64_t n_rec_sc, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, ((n_rec > n_rec_sc ? n_rec : n_rec_sc) + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(u_cur.scalar_type(), "record", [&] {
        record_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            u_cur.data_ptr<scalar_t>(), u_sc_cur.data_ptr<scalar_t>(),
            r.data_ptr<scalar_t>(), r_sc.data_ptr<scalar_t>(),
            rec_i.data_ptr<int64_t>(), rec_sc_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (int)n_rec_sc, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami born record failed");
}

void adjoint_step(
    torch::Tensor v, torch::Tensor scatter,
    torch::Tensor lam_bg_next, torch::Tensor lam_bg_next2, torch::Tensor lam_sc_next, torch::Tensor lam_sc_next2,
    torch::Tensor lam_bg_new, torch::Tensor lam_sc_new,
    torch::Tensor psi_y, torch::Tensor psi_x, torch::Tensor zeta_y, torch::Tensor zeta_x,
    torch::Tensor psi_y_sc, torch::Tensor psi_x_sc, torch::Tensor zeta_y_sc, torch::Tensor zeta_x_sc,
    torch::Tensor psi_y_new, torch::Tensor psi_x_new, torch::Tensor zeta_y_new, torch::Tensor zeta_x_new,
    torch::Tensor psi_y_sc_new, torch::Tensor psi_x_sc_new, torch::Tensor zeta_y_sc_new, torch::Tensor zeta_x_sc_new,
    torch::Tensor ay, torch::Tensor by, torch::Tensor dbydy,
    torch::Tensor ax, torch::Tensor bx, torch::Tensor dbxdx,
    torch::Tensor w_store, torch::Tensor wsc_store,
    torch::Tensor grad_v, torch::Tensor grad_scatter,
    torch::Tensor c1, torch::Tensor c2,
    double rdy, double rdx, double rdy2, double rdx2,
    int64_t t, int64_t interval, double scale, double dt2,
    int64_t n_shots, int64_t ny, int64_t nx,
    // forward PML boundaries (fd_pad + pml_width); unused by the adjoint
    // launches, which run on the widened backward boundaries below — kept
    // only to preserve the positional calling convention of the Python side
    [[maybe_unused]] int64_t pml_y0, [[maybe_unused]] int64_t pml_y1,
    [[maybe_unused]] int64_t pml_x0, [[maybe_unused]] int64_t pml_x1,
    // backward PML boundaries (forward + fd_pad): the adjoint widens
    // the PML region by one fd_pad because the transpose of the
    // forward PML stencil reads one cell further into the interior.
    int64_t pml_y0_b, int64_t pml_y1_b, int64_t pml_x0_b, int64_t pml_x1_b,
    int64_t v_batched, int64_t scatter_batched, int64_t snap_off, int64_t fd_pad)
{
    CHECK_CONTIG(v);
    CHECK_CONTIG(scatter);
    CHECK_CONTIG(lam_bg_next);
    CHECK_CONTIG(lam_bg_next2);
    CHECK_CONTIG(lam_sc_next);
    CHECK_CONTIG(lam_sc_next2);
    CHECK_CONTIG(lam_bg_new);
    CHECK_CONTIG(lam_sc_new);
    CHECK_CONTIG(w_store);
    CHECK_CONTIG(wsc_store);
    CHECK_CONTIG(grad_v);
    CHECK_CONTIG(grad_scatter);
    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "adjoint_step", [&] {
        LAUNCH_STEP_INTERIOR(step_adjoint_interior_kernel, scalar_t,
            pml_y0_b, pml_y1_b, pml_x0_b, pml_x1_b,
            v.data_ptr<scalar_t>(), scatter.data_ptr<scalar_t>(),
            lam_bg_next.data_ptr<scalar_t>(), lam_bg_next2.data_ptr<scalar_t>(),
            lam_sc_next.data_ptr<scalar_t>(), lam_sc_next2.data_ptr<scalar_t>(),
            lam_bg_new.data_ptr<scalar_t>(), lam_sc_new.data_ptr<scalar_t>(),
            w_store.data_ptr<scalar_t>(), wsc_store.data_ptr<scalar_t>(),
            grad_v.data_ptr<scalar_t>(), grad_scatter.data_ptr<scalar_t>(),
            c1.data_ptr<scalar_t>(), c2.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx, (scalar_t)rdy2, (scalar_t)rdx2,
            (int)t, (int)interval, (scalar_t)scale, (scalar_t)dt2,
            (int)n_shots, (int)ny, (int)nx,
            (int)pml_y0_b, (int)pml_y1_b, (int)pml_x0_b, (int)pml_x1_b,
            (int)v_batched, (int)scatter_batched, (int64_t)snap_off, (int)fd_pad);
        LAUNCH_STEP(step_adjoint_frame_kernel, scalar_t,
            v.data_ptr<scalar_t>(), scatter.data_ptr<scalar_t>(),
            lam_bg_next.data_ptr<scalar_t>(), lam_bg_next2.data_ptr<scalar_t>(),
            lam_sc_next.data_ptr<scalar_t>(), lam_sc_next2.data_ptr<scalar_t>(),
            lam_bg_new.data_ptr<scalar_t>(), lam_sc_new.data_ptr<scalar_t>(),
            psi_y.data_ptr<scalar_t>(), psi_x.data_ptr<scalar_t>(),
            zeta_y.data_ptr<scalar_t>(), zeta_x.data_ptr<scalar_t>(),
            psi_y_sc.data_ptr<scalar_t>(), psi_x_sc.data_ptr<scalar_t>(),
            zeta_y_sc.data_ptr<scalar_t>(), zeta_x_sc.data_ptr<scalar_t>(),
            psi_y_new.data_ptr<scalar_t>(), psi_x_new.data_ptr<scalar_t>(),
            zeta_y_new.data_ptr<scalar_t>(), zeta_x_new.data_ptr<scalar_t>(),
            psi_y_sc_new.data_ptr<scalar_t>(), psi_x_sc_new.data_ptr<scalar_t>(),
            zeta_y_sc_new.data_ptr<scalar_t>(), zeta_x_sc_new.data_ptr<scalar_t>(),
            ay.data_ptr<scalar_t>(), by.data_ptr<scalar_t>(), dbydy.data_ptr<scalar_t>(),
            ax.data_ptr<scalar_t>(), bx.data_ptr<scalar_t>(), dbxdx.data_ptr<scalar_t>(),
            w_store.data_ptr<scalar_t>(), wsc_store.data_ptr<scalar_t>(),
            grad_v.data_ptr<scalar_t>(), grad_scatter.data_ptr<scalar_t>(),
            c1.data_ptr<scalar_t>(), c2.data_ptr<scalar_t>(),
            (scalar_t)rdy, (scalar_t)rdx, (scalar_t)rdy2, (scalar_t)rdx2,
            (int)t, (int)interval, (scalar_t)scale, (scalar_t)dt2,
            (int)n_shots, (int)ny, (int)nx,
            (int)pml_y0_b, (int)pml_y1_b, (int)pml_x0_b, (int)pml_x1_b,
            (int)v_batched, (int)scatter_batched, (int64_t)snap_off, (int)fd_pad);
    });
}

void record_grad_f(torch::Tensor lam_bg_next, torch::Tensor lam_sc_next,
                   torch::Tensor grad_f, torch::Tensor grad_f_sc, torch::Tensor src_i,
                   int64_t t, int64_t n_shots, int64_t n_src, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, (n_src + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(lam_bg_next.scalar_type(), "record_grad_f", [&] {
        record_grad_f_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            lam_bg_next.data_ptr<scalar_t>(), lam_sc_next.data_ptr<scalar_t>(),
            grad_f.data_ptr<scalar_t>(), grad_f_sc.data_ptr<scalar_t>(), src_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_src, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami born record_grad_f failed");
}

void record_grad_r(torch::Tensor lam_bg_cur, torch::Tensor lam_sc_cur,
                   torch::Tensor grad_r, torch::Tensor grad_r_sc,
                   torch::Tensor rec_i, torch::Tensor rec_sc_i,
                   int64_t t, int64_t n_shots, int64_t n_rec, int64_t n_rec_sc, int64_t ny_nx)
{
    dim3 block(32, 4);
    dim3 grid((n_shots + 31) / 32, ((n_rec > n_rec_sc ? n_rec : n_rec_sc) + 3) / 4);
    AT_DISPATCH_FLOATING_TYPES(lam_bg_cur.scalar_type(), "record_grad_r", [&] {
        record_grad_r_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            lam_bg_cur.data_ptr<scalar_t>(), lam_sc_cur.data_ptr<scalar_t>(),
            grad_r.data_ptr<scalar_t>(), grad_r_sc.data_ptr<scalar_t>(),
            rec_i.data_ptr<int64_t>(), rec_sc_i.data_ptr<int64_t>(),
            (int)t, (int)n_shots, (int)n_rec, (int)n_rec_sc, (long)ny_nx);
    });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nami born record_grad_r failed");
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
