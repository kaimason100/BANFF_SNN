// snn_classify_time_loop_gpu_mex.cu
//
// Dataset-GPU inner loop for classification with adLIF neurons.
// - Static weights, datasets, labels, and optionally W_in*X input currents stay on GPU.
// - train_epoch_gpu accumulates gradients and applies Adam/AMSGrad inside the MEX.
// - validate_gpu returns scalar metrics only.
//
// Command interface from MATLAB:
//
//   snn_classify_time_loop_gpu_mex('init', ...);
//   snn_classify_time_loop_gpu_mex('init_optim', ...);
//   snn_classify_time_loop_gpu_mex('set_data', 'train', X_T, Y_T, true);
//   snn_classify_time_loop_gpu_mex('set_data', 'val', X_T, Y_T, true);
//   [lossN,lossR,corrN,corrR,count] =
//       snn_classify_time_loop_gpu_mex('train_epoch_gpu', 'train', order, batch_size, lr_bias);
//   [lossN,lossR,corrN,corrR,count] =
//       snn_classify_time_loop_gpu_mex('validate_gpu', 'val', batch_size);
//
// Build:
//   Run Classification/build/compile_classification_gpu_mex.mlx
//
// Fast-mode notes:
// - Keeps static weights, datasets, optimizer state, and buffers persistent.
// - Does not allocate/copy spike rasters or diagnostic trajectories.
//
// Notes:
// - All matrices are MATLAB column-major (Fortran-order).
// - Float32 throughout; logs use NaN-filled buffer for u when requested.
// - Data must be passed as column-major [features/classes x samples].
//

#include "mex.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cstring>
#include <string>
#include <algorithm>
#include <climits>

using std::vector;

#ifndef REALMIN_SINGLE
#define REALMIN_SINGLE 1.17549435e-38f
#endif

enum { RECURRENT_LOW_RANK = 0, RECURRENT_FULL_RANK = 1 };
enum { DECODER_SHARED = 0, DECODER_SIGNED = 1 };
enum { REC_STORAGE_DENSE = 0, REC_STORAGE_SPARSE = 1 };

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        mexErrMsgIdAndTxt("snn_classify_mex:cuda", "CUDA error %d (%s) at %s:%d", \
            (int)err_, cudaGetErrorString(err_), __FILE__, __LINE__); \
    } \
} while(0)

// column-major offset: A(m x n), element (i,j) -> i + m*j  (0-based i,j)
__host__ __device__ inline int idx2(int i, int m, int j) { return i + m*j; }

static inline void assertSingle(const mxArray* a, const char* name)
{
    if (mxGetClassID(a) != mxSINGLE_CLASS || mxIsComplex(a))
        mexErrMsgIdAndTxt("snn_classify_mex:type", "%s must be real(single)", name);
}
static inline void assertVecLen(const mxArray* a, int len, const char* name)
{
    if (mxGetNumberOfElements(a) != (mwSize)len)
        mexErrMsgIdAndTxt("snn_classify_mex:shape", "%s length mismatch", name);
}
static inline bool isVectorMx(const mxArray* a)
{
    if (!a) return false;
    if (mxGetNumberOfDimensions(a) != 2) return false;
    mwSize m = mxGetM(a), n = mxGetN(a);
    return (m==1) || (n==1);
}

// ===================== kernels =====================

// y[N] = scale * (A(NxD) * x[D])  (column-major)
__global__ void kDotCol(const float* __restrict__ A,
                        const float* __restrict__ x,
                        float* __restrict__ y,
                        int N, int D, float scale)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.f;
#pragma unroll 4
    for (int d=0; d<D; ++d) acc += A[n + d*N] * x[d];
    y[n] = scale * acc;
}

__global__ void kSparseRecBatch(const int* __restrict__ post,
                                const int* __restrict__ pre,
                                const float* __restrict__ w,
                                int nnz,
                                const float* __restrict__ r,
                                float* __restrict__ Irec,
                                int N,
                                int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nnz * B;
    if (idx >= total) return;
    int e = idx % nnz;
    int b = idx / nnz;
    atomicAdd(Irec + post[e] + b*N, w[e] * r[pre[e] + b*N]);
}

// y_dec[Nrec] = W_out_base_rec(Nrec x N) * r[N]
__global__ void kYDecReduce(const float* __restrict__ Wobr,
                            const float* __restrict__ r,
                            float* __restrict__ y_dec,
                            int Nrec, int N)
{
    int rid = blockIdx.x;
    if (rid >= Nrec) return;
    extern __shared__ float sh[];
    float sum = 0.f;
    for (int h = threadIdx.x; h < N; h += blockDim.x)
        sum += Wobr[rid + h * Nrec] * r[h];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x>>1; s>0; s>>=1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x==0) y_dec[rid] = sh[0];
}

// I_rec = SCALE_rec * (Eta*y_dec) - dself .* r;
// I_tot = I_in + I_rec + B;
// optional logging of I_in/I_rec columns for this step.
__global__ void kBuildItot(const float* __restrict__ I_in,
                           const float* __restrict__ Irec_scaled,
                           const float* __restrict__ r,
                           const float* __restrict__ dself,
                           const float* __restrict__ B,
                           float* __restrict__ I_rec,
                           float* __restrict__ I_tot,
                           float* __restrict__ I_in_store_col,   // nullable
                           float* __restrict__ I_rec_store_col,  // nullable
                           int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float irec = Irec_scaled[i] - dself[i]*r[i];
    float itot = I_in[i] + irec + B[i];
    I_rec[i] = irec; I_tot[i] = itot;
    if (I_in_store_col)  I_in_store_col[i]  = I_in[i];
    if (I_rec_store_col) I_rec_store_col[i] = irec;
}

// Advance u,w one step with fractional spike timing (rho), export spike & surrogate
__global__ void kAdvanceUW(const float* __restrict__ u_prev,
                           const float* __restrict__ w_prev,
                           const float* __restrict__ I_tot,
                           float alpha, float oneMinusAlpha,
                           float beta,  float oneMinusBeta,
                           float log_alpha, float log_beta,
                           float V_th, float V_reset, float E_L,
                           float a_eff, float b_param,
                           float phi_u, float delta_u,
                           float* __restrict__ u_out,
                           float* __restrict__ w_out,
                           float* __restrict__ rho_out,
                           bool*  __restrict__ spike_out,
                           float* __restrict__ surr_out,
                           int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float u0 = u_prev[i];
    float w0 = w_prev[i];
    float It = I_tot[i];

    float u_hat = E_L + alpha*(u0 - E_L) + oneMinusAlpha*(It - w0);
    bool sp = (u_hat >= V_th);

    float u_diff = u_hat - u0;
    float rho = 0.f;
    if (sp && (u_diff > 0.f)) {
        float den = fmaxf(u_diff, REALMIN_SINGLE);
        rho = (V_th - u0) / den;
    }
    if (!isfinite(rho)) rho = 0.f;
    if (rho < 0.f) rho = 0.f; else if (rho > 1.f) rho = 1.f;

    float alpha_pre  = expf(rho        * log_alpha);
    float beta_pre   = expf(rho        * log_beta);
    float alpha_post = expf((1.f-rho)  * log_alpha);
    float beta_post  = expf((1.f-rho)  * log_beta);

    float omApre  = 1.f - alpha_pre;
    float omBpre  = 1.f - beta_pre;
    float omApost = 1.f - alpha_post;
    float omBpost = 1.f - beta_post;

    float u_star = E_L + alpha_pre * (u0 - E_L) + omApre * (It - w0);
    float w1     = beta_pre * w0 + omBpre * (a_eff * (u_star - E_L));

    // surrogate at linear crossing
    float den_pd = fmaxf(delta_u, REALMIN_SINGLE);
    float u_lin  = u0 + rho * (u_hat - u0);
    float x      = (u_lin - V_th) / den_pd;
    float surr   = phi_u * fmaxf(0.f, 1.f - fabsf(x));

    if (sp) { u_star = V_reset; w1 = w1 + b_param; }

    float u2 = E_L + alpha_post * (u_star - E_L) + omApost * (It - w1);
    float w2 = beta_post  * w1 + omBpost  * (a_eff * (u2 - E_L));

    u_out[i]     = u2;
    w_out[i]     = w2;
    rho_out[i]   = rho;
    spike_out[i] = sp;
    surr_out[i]  = surr;
}

// Two-stage synaptic cascades with fractional durations
__global__ void kCascadeAdvancePre(const float* __restrict__ rho,
                                   float* __restrict__ x_syn,
                                   float* __restrict__ r,
                                   float log_gamma_sr, float log_gamma_sd,
                                   int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float f = rho[i];
    float gsr_f = expf(f * log_gamma_sr);
    float gsd_f = expf(f * log_gamma_sd);
    float x1 = gsr_f * x_syn[i];     // z_const = 0 for classification
    x_syn[i] = x1;
    float r1 = gsd_f * r[i] + (1.f - gsd_f) * x1;
    r[i] = r1;
}
__global__ void kCascadeAdvancePost(const float* __restrict__ rho,
                                    float* __restrict__ x_syn,
                                    float* __restrict__ r,
                                    float log_gamma_sr, float log_gamma_sd,
                                    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float f = 1.f - rho[i];
    float gsr_f = expf(f * log_gamma_sr);
    float gsd_f = expf(f * log_gamma_sd);
    float x1 = gsr_f * x_syn[i];
    x_syn[i] = x1;
    float r1 = gsd_f * r[i] + (1.f - gsd_f) * x1;
    r[i] = r1;
}
__global__ void kAddSpikeJumps(const bool* __restrict__ spike,
                               float* __restrict__ x_syn,
                               float jump, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (spike[i]) x_syn[i] += jump;
}

// Per-step outputs: Z(:,k) and Y_read(:,k) via reductions
__global__ void kOutputsColReduce(const float* __restrict__ Wout,
                                  const float* __restrict__ Wread,
                                  const float* __restrict__ r,
                                  const float* __restrict__ r_read,
                                  float* __restrict__ Z_col_k,
                                  float* __restrict__ Y_col_k,
                                  int Dout, int N)
{
    int d = blockIdx.x;
    if (d >= Dout) return;
    extern __shared__ float sh[];
    // Z
    float sZ = 0.f;
    for (int h=threadIdx.x; h<N; h+=blockDim.x)
        sZ += Wout[d + h*Dout] * r[h];
    sh[threadIdx.x] = sZ;
    __syncthreads();
    for (int s = blockDim.x>>1; s>0; s>>=1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x==0) Z_col_k[d] = sh[0];

    // reuse shared mem for Y
    float sY = 0.f;
    for (int h=threadIdx.x; h<N; h+=blockDim.x)
        sY += Wread[d + h*Dout] * r_read[h];
    sh[threadIdx.x] = sY;
    __syncthreads();
    for (int s = blockDim.x>>1; s>0; s>>=1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x==0) Y_col_k[d] = sh[0];
}


// Accumulate averaging-window outputs without storing full trajectories.
__global__ void kOutputsSumReduce(const float* __restrict__ Wout,
                                  const float* __restrict__ Wread,
                                  const float* __restrict__ r,
                                  const float* __restrict__ r_read,
                                  float* __restrict__ Z_sum,
                                  float* __restrict__ Y_sum,
                                  int Dout, int N)
{
    int d = blockIdx.x;
    if (d >= Dout) return;
    extern __shared__ float sh[];
    float sZ = 0.f;
    for (int h=threadIdx.x; h<N; h+=blockDim.x)
        sZ += Wout[d + h*Dout] * r[h];
    sh[threadIdx.x] = sZ;
    __syncthreads();
    for (int s = blockDim.x>>1; s>0; s>>=1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x==0) Z_sum[d] += sh[0];

    float sY = 0.f;
    for (int h=threadIdx.x; h<N; h+=blockDim.x)
        sY += Wread[d + h*Dout] * r_read[h];
    sh[threadIdx.x] = sY;
    __syncthreads();
    for (int s = blockDim.x>>1; s>0; s>>=1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x==0) Y_sum[d] += sh[0];
}

// Eligibility update (no gradient accumulation inside; we just update Ebar_f)
__global__ void kEligUpdate(const float* __restrict__ rho,
                            const bool*  __restrict__ spike,
                            const float* __restrict__ surr,
                            float log_gamma_sr, float log_gamma_sd,
                            float log_alpha, float log_beta,
                            float a_eff, float b_param,
                            float spike_jump_sr,
                            float* __restrict__ eps_v_noa,
                            float* __restrict__ eps_a,
                            float* __restrict__ Ebar_x,
                            float* __restrict__ Ebar_f,
                            int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float rhi = rho[i];
    float ev = eps_v_noa[i];
    float ea = eps_a[i];
    float Ex = Ebar_x[i];
    float Ef = Ebar_f[i];

    float a_pre   = expf(rhi        * log_alpha);
    float b_pre   = expf(rhi        * log_beta );
    float omApre  = 1.f - a_pre;
    float omBpre  = 1.f - b_pre;
    float a_post  = expf((1.f-rhi)  * log_alpha);
    float b_post  = expf((1.f-rhi)  * log_beta );
    float omApost = 1.f - a_post;
    float omBpost = 1.f - b_post;

    // membrane eligibilities to t*
    ev = a_pre * ev + omApre;
    float ev_full_pre = ev - omApre * ea;
    float e_raw = surr[i] * ev_full_pre;

    // adaptation eligibility to t*
    ea = b_pre * ea + omBpre * (a_eff * ev_full_pre);

    // reset at t*
    if (spike[i]) {
        ev = 0.f;
        ea = ea + b_param * e_raw;
    }

    // post to t_{k+1}
    ev = a_post * ev + omApost;
    float ev_full = ev - omApost * ea;
    ea = b_post * ea + omBpost * (a_eff * ev_full);

    // filter eligibility through synapses (pre/post)
    float gsr_pre  = expf(rhi        * log_gamma_sr);
    float gsd_pre  = expf(rhi        * log_gamma_sd);
    float gsr_post = expf((1.f-rhi)  * log_gamma_sr);
    float gsd_post = expf((1.f-rhi)  * log_gamma_sd);

    float Ex_pre = gsr_pre * Ex;
    Ef = gsd_pre * Ef + (1.f - gsd_pre) * Ex_pre;
    if (spike[i]) Ex_pre = Ex_pre + spike_jump_sr * e_raw;
    Ex = gsr_post * Ex_pre;
    Ef = gsd_post * Ef + (1.f - gsd_post) * Ex;

    eps_v_noa[i] = ev;
    eps_a[i]     = ea;
    Ebar_x[i]    = Ex;
    Ebar_f[i]    = Ef;
}

// sum_vec += src_vec (length N)
__global__ void kAccumulateVec(const float* __restrict__ src,
                               float* __restrict__ sum_vec,
                               int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    sum_vec[i] += src[i];
}

// Log u-buffer for selected neurons (0-based indices)
__global__ void kLogUbuffer(const float* __restrict__ u,
                            const int* __restrict__ neurons_idx0,
                            int num_sel,
                            float* __restrict__ u_buf,   // [num_sel x t2s]
                            int t_index, int t2s)
{
    if (t_index >= t2s) return;
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_sel) return;
    int nidx = neurons_idx0[r];
    u_buf[r + num_sel * t_index] = u[nidx];
}

// Log spikes+ w into column; S stored as uint8 (converted to logical at the end)
__global__ void kLogSpikeAndW(const bool* __restrict__ spike_k,
                              const float* __restrict__ w_curr,
                              unsigned char* __restrict__ S_col_k,
                              float* __restrict__ w_store_col,
                              int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (S_col_k)      S_col_k[i]   = spike_k[i] ? 1u : 0u;
    if (w_store_col)  w_store_col[i]= w_curr[i];
}

// ===================== mex entry =====================

extern "C"

__global__ void kFillConst(float* __restrict__ y, float v, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = v;
}

__global__ void kScaleVec(float* __restrict__ y, float scale, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] *= scale;
}

__global__ void kBuildItotBatch(const float* __restrict__ I_in,
                                const float* __restrict__ Irec_scaled,
                                const float* __restrict__ r,
                                const float* __restrict__ dself,
                                const float* __restrict__ Bbias,
                                float* __restrict__ I_rec,
                                float* __restrict__ I_tot,
                                int N_hidden, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int h = idx % N_hidden;
    float irec = Irec_scaled[idx] - dself[h] * r[idx];
    I_rec[idx] = irec;
    I_tot[idx] = I_in[idx] + irec + Bbias[h];
}

__global__ void kBuildItotFullRankBatch(const float* __restrict__ I_in,
                                        const float* __restrict__ Irec_full,
                                        const float* __restrict__ Bbias,
                                        float* __restrict__ I_rec,
                                        float* __restrict__ I_tot,
                                        int N_hidden, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int h = idx % N_hidden;
    float irec = Irec_full[idx];
    I_rec[idx] = irec;
    I_tot[idx] = I_in[idx] + irec + Bbias[h];
}

__global__ void kAccumulateBiasGradBatch(const float* __restrict__ Lstd,
                                         const float* __restrict__ EbarSum,
                                         float* __restrict__ gB,
                                         int N_hidden, int B, float invSteps)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= N_hidden) return;
    float acc = 0.f;
    for (int b=0; b<B; ++b) acc += Lstd[h + N_hidden*b] * EbarSum[h + N_hidden*b] * invSteps;
    gB[h] += acc;
}

__global__ void kClassLossGrad(const float* __restrict__ Z,
                               const float* __restrict__ Yread,
                               const float* __restrict__ Ytrue,
                               float* __restrict__ gZnet,
                               float* __restrict__ gZread,
                               float* __restrict__ lossNet,
                               float* __restrict__ lossRead,
                               int* __restrict__ correctNet,
                               int* __restrict__ correctRead,
                               int C, int B)
{
    int b = blockIdx.x;
    if (b >= B || threadIdx.x != 0) return;
    const float eps = 1.17549435e-38f;
    float maxN = -FLT_MAX, maxR = -FLT_MAX;
    int trueIdx = 0;
    for (int c=0; c<C; ++c) {
        float zn = Z[c + C*b];
        float zr = Yread[c + C*b];
        if (zn > maxN) maxN = zn;
        if (zr > maxR) maxR = zr;
        if (Ytrue[c + C*b] > 0.5f) trueIdx = c;
    }
    float sumN = 0.f, sumR = 0.f;
    for (int c=0; c<C; ++c) {
        sumN += expf(Z[c + C*b] - maxN);
        sumR += expf(Yread[c + C*b] - maxR);
    }
    int predN = 0, predR = 0;
    float bestN = -FLT_MAX, bestR = -FLT_MAX;
    float ln = 0.f, lr = 0.f;
    for (int c=0; c<C; ++c) {
        float yn = Ytrue[c + C*b];
        float pn = expf(Z[c + C*b] - maxN) / fmaxf(sumN, eps);
        float pr = expf(Yread[c + C*b] - maxR) / fmaxf(sumR, eps);
        gZnet[c + C*b] = pn - yn;
        gZread[c + C*b] = pr - yn;
        if (yn > 0.f) {
            ln -= yn * logf(fmaxf(pn, eps));
            lr -= yn * logf(fmaxf(pr, eps));
        }
        if (pn > bestN) { bestN = pn; predN = c; }
        if (pr > bestR) { bestR = pr; predR = c; }
    }
    lossNet[b] = ln;
    lossRead[b] = lr;
    correctNet[b] = (predN == trueIdx) ? 1 : 0;
    correctRead[b] = (predR == trueIdx) ? 1 : 0;
}

__global__ void kGatherColumns(const float* __restrict__ src,
                               float* __restrict__ dst,
                               const int* __restrict__ order,
                               int offset, int rows, int n_src, int B, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int row = i % rows;
    int b = i / rows;
    int src_col = order ? (order[offset + b] - 1) : (offset + b);
    if (src_col < 0 || src_col >= n_src) return;
    dst[row + rows*b] = src[row + rows*src_col];
}

__global__ void kReduceClassMetrics(const float* __restrict__ lossNet,
                                    const float* __restrict__ lossRead,
                                    const int* __restrict__ correctNet,
                                    const int* __restrict__ correctRead,
                                    float* __restrict__ metricSums,
                                    int B)
{
    __shared__ float sLossN[256], sLossR[256];
    __shared__ int sCorrN[256], sCorrR[256];
    int tid = threadIdx.x;
    float ln = 0.f, lr = 0.f;
    int cn = 0, cr = 0;
    for (int i = tid; i < B; i += blockDim.x) {
        ln += lossNet[i];
        lr += lossRead[i];
        cn += correctNet[i];
        cr += correctRead[i];
    }
    sLossN[tid] = ln; sLossR[tid] = lr; sCorrN[tid] = cn; sCorrR[tid] = cr;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sLossN[tid] += sLossN[tid+s];
            sLossR[tid] += sLossR[tid+s];
            sCorrN[tid] += sCorrN[tid+s];
            sCorrR[tid] += sCorrR[tid+s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(metricSums + 0, sLossN[0]);
        atomicAdd(metricSums + 1, sLossR[0]);
        atomicAdd(metricSums + 2, (float)sCorrN[0]);
        atomicAdd(metricSums + 3, (float)sCorrR[0]);
    }
}

__global__ void kAdamBiasUpdate(float* __restrict__ B,
                                const float* __restrict__ gB,
                                float* __restrict__ m,
                                float* __restrict__ v,
                                float* __restrict__ vhatMax,
                                float lr, float beta1, float beta2,
                                float eps, float invN,
                                float bc1, float bc2, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = gB[i] * invN;
    float mi = beta1 * m[i] + (1.f - beta1) * g;
    float vi = beta2 * v[i] + (1.f - beta2) * g * g;
    float mhat = mi / bc1;
    float vhat = vi / bc2;
    float vmax = fmaxf(vhatMax[i], vhat);
    B[i] -= lr * (mhat / (sqrtf(vmax) + eps));
    m[i] = mi; v[i] = vi; vhatMax[i] = vmax;
}

__global__ void kAdamReadUpdate(float* __restrict__ W,
                                const float* __restrict__ Grad,
                                float* __restrict__ m,
                                float* __restrict__ v,
                                float* __restrict__ vhatMax,
                                float lr, float beta1, float beta2,
                                float eps, float weightDecay, float invN,
                                float bc1, float bc2, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float g = Grad[i] * invN + weightDecay * W[i];
    float mi = beta1 * m[i] + (1.f - beta1) * g;
    float vi = beta2 * v[i] + (1.f - beta2) * g * g;
    float mhat = mi / bc1;
    float vhat = vi / bc2;
    float vmax = fmaxf(vhatMax[i], vhat);
    W[i] -= lr * (mhat / (sqrtf(vmax) + eps));
    m[i] = mi; v[i] = vi; vhatMax[i] = vmax;
}

struct GpuData
{
    bool has = false;
    bool precomputed = false;
    int n = 0;
    float* d_X = nullptr;
    float* d_Y = nullptr;
    float* d_I = nullptr;
};

struct EpochContext
{
    bool initialized = false;
    int steps = 0, N_hidden = 0, N_rec = 0, N_in = 0, N_out = 0, max_batch = 0;
    int recurrent_mode = RECURRENT_LOW_RANK, decoder_mode = DECODER_SHARED, recurrent_storage = REC_STORAGE_DENSE, rec_nnz = 0;
    int k_avg_start = 0, steps_avg = 0;
    float alpha = 0.f, oneMinusAlpha = 0.f, beta = 0.f, oneMinusBeta = 0.f;
    float gamma_sr = 0.f, gamma_sd = 0.f, READ_gamma_sr = 0.f, READ_gamma_sd = 0.f;
    float E_L = 0.f, V_th = 0.f, V_reset = 0.f, a_eff = 0.f, b_param = 0.f;
    float phi_u = 0.f, delta_u = 0.f, INPUT_SCALE = 0.f, SCALE_rec = 0.f;
    float spike_jump_sr = 0.f, spike_jump_sr_R = 0.f;
    float log_alpha = 0.f, log_beta = 0.f, log_gsr = 0.f, log_gsd = 0.f, log_Rgsr = 0.f, log_Rgsd = 0.f;
    cublasHandle_t handle = nullptr;

    float *d_W_in=nullptr, *d_W_out_b=nullptr, *d_W_out=nullptr, *d_Eta=nullptr, *d_dself=nullptr, *d_B=nullptr, *d_W_read=nullptr, *d_W_rec=nullptr, *d_rec_w=nullptr;
    int *d_rec_post=nullptr, *d_rec_pre=nullptr;
    float *d_X=nullptr, *d_Y=nullptr;
    float *d_u=nullptr, *d_w=nullptr, *d_xsyn=nullptr, *d_r=nullptr, *d_xsynR=nullptr, *d_rR=nullptr;
    float *d_eps_v_noa=nullptr, *d_eps_a=nullptr, *d_Ebar_x=nullptr, *d_Ebar_f=nullptr, *d_Ebar_f_sum=nullptr, *d_phi_sum=nullptr;
    float *d_I_in=nullptr, *d_y_dec=nullptr, *d_Irec_scaled=nullptr, *d_I_rec=nullptr, *d_I_tot=nullptr;
    bool *d_spike=nullptr;
    float *d_rho=nullptr, *d_surr=nullptr;
    float *d_Z_sum=nullptr, *d_Yread_sum=nullptr, *d_gZnet=nullptr, *d_gZread=nullptr, *d_Lstd=nullptr;
    float *d_gB_epoch=nullptr, *d_Grad_epoch=nullptr, *d_lossNet=nullptr, *d_lossRead=nullptr;
    int *d_correctNet=nullptr, *d_correctRead=nullptr;
    float *d_statsNet=nullptr, *d_statsRead=nullptr;
    float *d_metricSums=nullptr;
    int *d_order=nullptr;
    int order_capacity = 0;

    GpuData train, val, test;
    bool optim_initialized = false;
    float *d_m_b=nullptr, *d_v_b=nullptr, *d_vhat_b=nullptr;
    float *d_M_read=nullptr, *d_V_read=nullptr, *d_Vhat_read=nullptr;
    int t_adam = 0, t_read = 0;
    float bias_b1 = 0.9f, bias_b2 = 0.999f, bias_eps = 1e-8f;
    float read_lr = 1e-4f, read_b1 = 0.9f, read_b2 = 0.999f, read_eps = 1e-8f, read_weight_decay = 0.f;
};

static EpochContext G;
static void freeF(float*& p) { if (p) { cudaFree(p); p=nullptr; } }
static void freeI(int*& p) { if (p) { cudaFree(p); p=nullptr; } }
static void freeB(bool*& p) { if (p) { cudaFree(p); p=nullptr; } }
static void freeData(GpuData& D)
{
    freeF(D.d_X); freeF(D.d_Y); freeF(D.d_I);
    D = GpuData();
}
static void clearContext()
{
    freeF(G.d_W_in); freeF(G.d_W_out_b); freeF(G.d_W_out); freeF(G.d_Eta); freeF(G.d_dself); freeF(G.d_B); freeF(G.d_W_read); freeF(G.d_W_rec); freeF(G.d_rec_w); freeI(G.d_rec_post); freeI(G.d_rec_pre);
    freeF(G.d_X); freeF(G.d_Y);
    freeF(G.d_u); freeF(G.d_w); freeF(G.d_xsyn); freeF(G.d_r); freeF(G.d_xsynR); freeF(G.d_rR);
    freeF(G.d_eps_v_noa); freeF(G.d_eps_a); freeF(G.d_Ebar_x); freeF(G.d_Ebar_f); freeF(G.d_Ebar_f_sum); freeF(G.d_phi_sum);
    freeF(G.d_I_in); freeF(G.d_y_dec); freeF(G.d_Irec_scaled); freeF(G.d_I_rec); freeF(G.d_I_tot);
    freeB(G.d_spike); freeF(G.d_rho); freeF(G.d_surr);
    freeF(G.d_Z_sum); freeF(G.d_Yread_sum); freeF(G.d_gZnet); freeF(G.d_gZread); freeF(G.d_Lstd);
    freeF(G.d_gB_epoch); freeF(G.d_Grad_epoch); freeF(G.d_lossNet); freeF(G.d_lossRead);
    freeI(G.d_correctNet); freeI(G.d_correctRead); freeF(G.d_statsNet); freeF(G.d_statsRead);
    freeF(G.d_metricSums); freeI(G.d_order);
    freeData(G.train); freeData(G.val); freeData(G.test);
    freeF(G.d_m_b); freeF(G.d_v_b); freeF(G.d_vhat_b);
    freeF(G.d_M_read); freeF(G.d_V_read); freeF(G.d_Vhat_read);
    if (G.handle) { cublasDestroy(G.handle); G.handle=nullptr; }
    G = EpochContext();
}
static float getScalarSingle(const mxArray* a, const char* nm)
{
    assertSingle(a, nm);
    if (mxGetNumberOfElements(a) != 1) mexErrMsgIdAndTxt("snn_classify_gpu_mex:scalar", "%s must be scalar", nm);
    return *(float*)mxGetData(a);
}
static int getIntScalar(const mxArray* a, const char* nm)
{
    if (mxGetNumberOfElements(a) != 1) mexErrMsgIdAndTxt("snn_classify_gpu_mex:scalar", "%s must be scalar", nm);
    return (int)mxGetScalar(a);
}
static void requireInitialized()
{
    if (!G.initialized) mexErrMsgIdAndTxt("snn_classify_gpu_mex:state", "Call snn_classify_time_loop_gpu_mex('init', ...) first.");
}
static void cublasCheck(cublasStatus_t st, const char* what)
{
    if (st != CUBLAS_STATUS_SUCCESS) mexErrMsgIdAndTxt("snn_classify_gpu_mex:cublas", "cuBLAS error %d in %s", (int)st, what);
}
static void allocContext()
{
    int N=G.N_hidden, B=G.max_batch, C=G.N_out, R=G.N_rec, D=G.N_in, NB=N*B;
    CUDA_CHECK(cudaMalloc(&G.d_W_in,sizeof(float)*N*D)); CUDA_CHECK(cudaMalloc(&G.d_W_out_b,sizeof(float)*R*N));
    CUDA_CHECK(cudaMalloc(&G.d_W_out,sizeof(float)*C*N)); CUDA_CHECK(cudaMalloc(&G.d_Eta,sizeof(float)*N*R));
    CUDA_CHECK(cudaMalloc(&G.d_dself,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_B,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_W_read,sizeof(float)*C*N));
    if (G.recurrent_mode == RECURRENT_FULL_RANK && G.recurrent_storage == REC_STORAGE_DENSE) CUDA_CHECK(cudaMalloc(&G.d_W_rec,sizeof(float)*N*N));
    if (G.recurrent_mode == RECURRENT_FULL_RANK && G.recurrent_storage == REC_STORAGE_SPARSE && G.rec_nnz > 0) {
        CUDA_CHECK(cudaMalloc(&G.d_rec_post,sizeof(int)*G.rec_nnz));
        CUDA_CHECK(cudaMalloc(&G.d_rec_pre,sizeof(int)*G.rec_nnz));
        CUDA_CHECK(cudaMalloc(&G.d_rec_w,sizeof(float)*G.rec_nnz));
    }
    CUDA_CHECK(cudaMalloc(&G.d_X,sizeof(float)*D*B)); CUDA_CHECK(cudaMalloc(&G.d_Y,sizeof(float)*C*B));
    CUDA_CHECK(cudaMalloc(&G.d_u,sizeof(float)*NB)); CUDA_CHECK(cudaMalloc(&G.d_w,sizeof(float)*NB)); CUDA_CHECK(cudaMalloc(&G.d_xsyn,sizeof(float)*NB)); CUDA_CHECK(cudaMalloc(&G.d_r,sizeof(float)*NB));
    CUDA_CHECK(cudaMalloc(&G.d_xsynR,sizeof(float)*NB)); CUDA_CHECK(cudaMalloc(&G.d_rR,sizeof(float)*NB));
    CUDA_CHECK(cudaMalloc(&G.d_eps_v_noa,sizeof(float)*NB)); CUDA_CHECK(cudaMalloc(&G.d_eps_a,sizeof(float)*NB)); CUDA_CHECK(cudaMalloc(&G.d_Ebar_x,sizeof(float)*NB)); CUDA_CHECK(cudaMalloc(&G.d_Ebar_f,sizeof(float)*NB));
    CUDA_CHECK(cudaMalloc(&G.d_Ebar_f_sum,sizeof(float)*NB)); CUDA_CHECK(cudaMalloc(&G.d_phi_sum,sizeof(float)*NB));
    CUDA_CHECK(cudaMalloc(&G.d_I_in,sizeof(float)*NB)); CUDA_CHECK(cudaMalloc(&G.d_y_dec,sizeof(float)*R*B)); CUDA_CHECK(cudaMalloc(&G.d_Irec_scaled,sizeof(float)*NB)); CUDA_CHECK(cudaMalloc(&G.d_I_rec,sizeof(float)*NB)); CUDA_CHECK(cudaMalloc(&G.d_I_tot,sizeof(float)*NB));
    CUDA_CHECK(cudaMalloc(&G.d_spike,sizeof(bool)*NB)); CUDA_CHECK(cudaMalloc(&G.d_rho,sizeof(float)*NB)); CUDA_CHECK(cudaMalloc(&G.d_surr,sizeof(float)*NB));
    CUDA_CHECK(cudaMalloc(&G.d_Z_sum,sizeof(float)*C*B)); CUDA_CHECK(cudaMalloc(&G.d_Yread_sum,sizeof(float)*C*B)); CUDA_CHECK(cudaMalloc(&G.d_gZnet,sizeof(float)*C*B)); CUDA_CHECK(cudaMalloc(&G.d_gZread,sizeof(float)*C*B)); CUDA_CHECK(cudaMalloc(&G.d_Lstd,sizeof(float)*N*B));
    CUDA_CHECK(cudaMalloc(&G.d_gB_epoch,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_Grad_epoch,sizeof(float)*C*N)); CUDA_CHECK(cudaMalloc(&G.d_lossNet,sizeof(float)*B)); CUDA_CHECK(cudaMalloc(&G.d_lossRead,sizeof(float)*B));
    CUDA_CHECK(cudaMalloc(&G.d_correctNet,sizeof(int)*B)); CUDA_CHECK(cudaMalloc(&G.d_correctRead,sizeof(int)*B)); CUDA_CHECK(cudaMalloc(&G.d_statsNet,sizeof(float)*5*B)); CUDA_CHECK(cudaMalloc(&G.d_statsRead,sizeof(float)*5*B));
    CUDA_CHECK(cudaMalloc(&G.d_metricSums,sizeof(float)*4));
    cublasCheck(cublasCreate(&G.handle), "cublasCreate");
}
static void zeroBatchState(int B)
{
    int total=G.N_hidden*B; dim3 t(256), g((total+t.x-1)/t.x);
    kFillConst<<<g,t>>>(G.d_u,G.E_L,total); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(G.d_w,0,sizeof(float)*total)); CUDA_CHECK(cudaMemset(G.d_xsyn,0,sizeof(float)*total)); CUDA_CHECK(cudaMemset(G.d_r,0,sizeof(float)*total));
    CUDA_CHECK(cudaMemset(G.d_xsynR,0,sizeof(float)*total)); CUDA_CHECK(cudaMemset(G.d_rR,0,sizeof(float)*total)); CUDA_CHECK(cudaMemset(G.d_eps_v_noa,0,sizeof(float)*total)); CUDA_CHECK(cudaMemset(G.d_eps_a,0,sizeof(float)*total));
    CUDA_CHECK(cudaMemset(G.d_Ebar_x,0,sizeof(float)*total)); CUDA_CHECK(cudaMemset(G.d_Ebar_f,0,sizeof(float)*total)); CUDA_CHECK(cudaMemset(G.d_Ebar_f_sum,0,sizeof(float)*total)); CUDA_CHECK(cudaMemset(G.d_phi_sum,0,sizeof(float)*total));
    CUDA_CHECK(cudaMemset(G.d_Z_sum,0,sizeof(float)*G.N_out*B)); CUDA_CHECK(cudaMemset(G.d_Yread_sum,0,sizeof(float)*G.N_out*B));
}
static void runBatchDynamics(int B, bool inputAlreadyComputed=false)
{
    int N=G.N_hidden, C=G.N_out, R=G.N_rec, D=G.N_in, total=N*B;
    dim3 t(256), gN((total+t.x-1)/t.x);
    float one=1.f, zero=0.f, recScale=G.SCALE_rec;
    zeroBatchState(B);
    if (!inputAlreadyComputed) {
        cublasCheck(cublasSgemm(G.handle,CUBLAS_OP_N,CUBLAS_OP_N,N,B,D,&G.INPUT_SCALE,G.d_W_in,N,G.d_X,D,&zero,G.d_I_in,N),"W_in*X");
    }
    for (int k=0;k<G.steps;++k) {
        if (G.recurrent_mode == RECURRENT_LOW_RANK) {
            cublasCheck(cublasSgemm(G.handle,CUBLAS_OP_N,CUBLAS_OP_N,R,B,N,&one,G.d_W_out_b,R,G.d_r,N,&zero,G.d_y_dec,R),"W_out_base*r");
            cublasCheck(cublasSgemm(G.handle,CUBLAS_OP_N,CUBLAS_OP_N,N,B,R,&recScale,G.d_Eta,N,G.d_y_dec,R,&zero,G.d_Irec_scaled,N),"Eta*y_dec");
            kBuildItotBatch<<<gN,t>>>(G.d_I_in,G.d_Irec_scaled,G.d_r,G.d_dself,G.d_B,G.d_I_rec,G.d_I_tot,N,total); CUDA_CHECK(cudaGetLastError());
        } else if (G.recurrent_mode == RECURRENT_FULL_RANK) {
            if (G.recurrent_storage == REC_STORAGE_DENSE) {
                cublasCheck(cublasSgemm(G.handle,CUBLAS_OP_N,CUBLAS_OP_N,N,B,N,&one,G.d_W_rec,N,G.d_r,N,&zero,G.d_Irec_scaled,N),"W_rec*r");
            } else if (G.recurrent_storage == REC_STORAGE_SPARSE) {
                CUDA_CHECK(cudaMemset(G.d_Irec_scaled,0,sizeof(float)*total));
                if (G.rec_nnz > 0) {
                    long long edgeTotal64 = (long long)G.rec_nnz * (long long)B;
                    if (edgeTotal64 > (long long)INT_MAX)
                        mexErrMsgIdAndTxt("snn_classify_gpu_mex:sparseOverflow", "Sparse recurrent edge batch exceeds int32 CUDA indexing; reduce p_rec, N_hidden or batch_size.");
                    int edgeTotal = (int)edgeTotal64;
                    kSparseRecBatch<<<(edgeTotal+255)/256,256>>>(G.d_rec_post,G.d_rec_pre,G.d_rec_w,G.rec_nnz,G.d_r,G.d_Irec_scaled,N,B);
                    CUDA_CHECK(cudaGetLastError());
                }
            } else {
                mexErrMsgIdAndTxt("snn_classify_gpu_mex:arch", "Unknown recurrent_storage_id %d.", G.recurrent_storage);
            }
            kBuildItotFullRankBatch<<<gN,t>>>(G.d_I_in,G.d_Irec_scaled,G.d_B,G.d_I_rec,G.d_I_tot,N,total); CUDA_CHECK(cudaGetLastError());
        } else {
            mexErrMsgIdAndTxt("snn_classify_gpu_mex:arch", "Unknown recurrent_mode_id %d.", G.recurrent_mode);
        }
        kAdvanceUW<<<gN,t>>>(G.d_u,G.d_w,G.d_I_tot,G.alpha,G.oneMinusAlpha,G.beta,G.oneMinusBeta,G.log_alpha,G.log_beta,G.V_th,G.V_reset,G.E_L,G.a_eff,G.b_param,G.phi_u,G.delta_u,G.d_u,G.d_w,G.d_rho,G.d_spike,G.d_surr,total); CUDA_CHECK(cudaGetLastError());
        kCascadeAdvancePre<<<gN,t>>>(G.d_rho,G.d_xsyn,G.d_r,G.log_gsr,G.log_gsd,total); kCascadeAdvancePre<<<gN,t>>>(G.d_rho,G.d_xsynR,G.d_rR,G.log_Rgsr,G.log_Rgsd,total); CUDA_CHECK(cudaGetLastError());
        kAddSpikeJumps<<<gN,t>>>(G.d_spike,G.d_xsyn,G.spike_jump_sr,total); kAddSpikeJumps<<<gN,t>>>(G.d_spike,G.d_xsynR,G.spike_jump_sr_R,total); CUDA_CHECK(cudaGetLastError());
        kCascadeAdvancePost<<<gN,t>>>(G.d_rho,G.d_xsyn,G.d_r,G.log_gsr,G.log_gsd,total); kCascadeAdvancePost<<<gN,t>>>(G.d_rho,G.d_xsynR,G.d_rR,G.log_Rgsr,G.log_Rgsd,total); CUDA_CHECK(cudaGetLastError());
        kEligUpdate<<<gN,t>>>(G.d_rho,G.d_spike,G.d_surr,G.log_gsr,G.log_gsd,G.log_alpha,G.log_beta,G.a_eff,G.b_param,G.spike_jump_sr,G.d_eps_v_noa,G.d_eps_a,G.d_Ebar_x,G.d_Ebar_f,total); CUDA_CHECK(cudaGetLastError());
        if (k >= G.k_avg_start-1) {
            cublasCheck(cublasSgemm(G.handle,CUBLAS_OP_N,CUBLAS_OP_N,C,B,N,&one,G.d_W_out,C,G.d_r,N,&one,G.d_Z_sum,C),"W_out*r");
            cublasCheck(cublasSgemm(G.handle,CUBLAS_OP_N,CUBLAS_OP_N,C,B,N,&one,G.d_W_read,C,G.d_rR,N,&one,G.d_Yread_sum,C),"W_read*rR");
            kAccumulateVec<<<gN,t>>>(G.d_Ebar_f,G.d_Ebar_f_sum,total); CUDA_CHECK(cudaGetLastError());
            kAccumulateVec<<<gN,t>>>(G.d_rR,G.d_phi_sum,total); CUDA_CHECK(cudaGetLastError());
        }
    }
    float inv = 1.f / fmaxf((float)G.steps_avg,1.f);
    int outTotal=C*B;
    kScaleVec<<<(outTotal+255)/256,256>>>(G.d_Z_sum,inv,outTotal); CUDA_CHECK(cudaGetLastError());
    kScaleVec<<<(outTotal+255)/256,256>>>(G.d_Yread_sum,inv,outTotal); CUDA_CHECK(cudaGetLastError());
}
static void gatherBatch(const float* X, int nX, const float* Y, int nY, const int* order, int offset, int B, std::vector<float>& hx, std::vector<float>& hy)
{
    hx.resize((size_t)G.N_in*B); hy.resize((size_t)G.N_out*B);
    for (int j=0;j<B;++j) {
        int idx = order ? ((int)order[offset+j]-1) : (offset+j);
        if (idx < 0 || idx >= nX || idx >= nY) mexErrMsgIdAndTxt("snn_classify_gpu_mex:order","Order index out of range");
        memcpy(&hx[(size_t)j*G.N_in], X + (size_t)idx*G.N_in, sizeof(float)*G.N_in);
        memcpy(&hy[(size_t)j*G.N_out], Y + (size_t)idx*G.N_out, sizeof(float)*G.N_out);
    }
}
static void accumulateGradients(int B)
{
    int N=G.N_hidden, C=G.N_out;
    float one=1.f, zero=0.f, inv=1.f/fmaxf((float)G.steps_avg,1.f);
    cublasCheck(cublasSgemm(G.handle,CUBLAS_OP_T,CUBLAS_OP_N,N,B,C,&one,G.d_W_out,C,G.d_gZnet,C,&zero,G.d_Lstd,N),"W_out'*gZ");
    kAccumulateBiasGradBatch<<<(N+255)/256,256>>>(G.d_Lstd,G.d_Ebar_f_sum,G.d_gB_epoch,N,B,inv); CUDA_CHECK(cudaGetLastError());
    cublasCheck(cublasSgemm(G.handle,CUBLAS_OP_N,CUBLAS_OP_T,C,N,B,&inv,G.d_gZread,C,G.d_phi_sum,N,&one,G.d_Grad_epoch,C),"Grad_read");
}

static void computeLossGrad(int B)
{
    kClassLossGrad<<<B,1>>>(G.d_Z_sum,G.d_Yread_sum,G.d_Y,G.d_gZnet,G.d_gZread,G.d_lossNet,G.d_lossRead,G.d_correctNet,G.d_correctRead,G.N_out,B);
    CUDA_CHECK(cudaGetLastError());
}

static bool getBoolScalar(const mxArray* a, const char* nm)
{
    if (mxGetNumberOfElements(a) != 1) mexErrMsgIdAndTxt("snn_classify_gpu_mex:scalar", "%s must be scalar", nm);
    if (mxIsLogical(a)) return mxIsLogicalScalarTrue(a);
    return mxGetScalar(a) != 0.0;
}

static std::string getString(const mxArray* a, const char* nm)
{
    if (!mxIsChar(a)) mexErrMsgIdAndTxt("snn_classify_gpu_mex:type", "%s must be a char/string", nm);
    char* c = mxArrayToString(a);
    if (!c) mexErrMsgIdAndTxt("snn_classify_gpu_mex:type", "Could not parse %s", nm);
    std::string s(c);
    mxFree(c);
    return s;
}

static GpuData& selectData(const std::string& split)
{
    if (split == "train") return G.train;
    if (split == "val" || split == "validation") return G.val;
    if (split == "test") return G.test;
    mexErrMsgIdAndTxt("snn_classify_gpu_mex:split", "Unknown split '%s'", split.c_str());
    return G.train;
}

static void ensureOrderCapacity(int n)
{
    if (n <= G.order_capacity) return;
    freeI(G.d_order);
    CUDA_CHECK(cudaMalloc(&G.d_order, sizeof(int)*n));
    G.order_capacity = n;
}

static void validateOrderVector(const mxArray* order_m, int n)
{
    if (mxGetClassID(order_m) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("snn_classify_gpu_mex:type", "order must be int32");
    if ((int)mxGetNumberOfElements(order_m) != n)
        mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape", "order length mismatch");
    const int* order = (const int*)mxGetData(order_m);
    for (int i = 0; i < n; ++i) {
        if (order[i] < 1 || order[i] > n)
            mexErrMsgIdAndTxt("snn_classify_gpu_mex:order", "Order index out of range");
    }
}

static void cmdSetData(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)nlhs; (void)plhs;
    if (nrhs != 5) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "set_data expects split, X, Y, precompute_inputs");
    requireInitialized();
    std::string split = getString(prhs[1], "split");
    const mxArray* X_m = prhs[2];
    const mxArray* Y_m = prhs[3];
    bool precompute = getBoolScalar(prhs[4], "precompute_inputs");
    assertSingle(X_m, "X");
    assertSingle(Y_m, "Y");
    if ((int)mxGetM(X_m) != G.N_in) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape", "X must be [N_in x samples]");
    if ((int)mxGetM(Y_m) != G.N_out) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape", "Y must be [N_out x samples]");
    int n = (int)mxGetN(X_m);
    if ((int)mxGetN(Y_m) != n) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape", "X/Y sample count mismatch");

    GpuData& D = selectData(split);
    freeData(D);
    D.n = n;
    D.has = true;
    D.precomputed = precompute;
    CUDA_CHECK(cudaMalloc(&D.d_X, sizeof(float)*G.N_in*n));
    CUDA_CHECK(cudaMalloc(&D.d_Y, sizeof(float)*G.N_out*n));
    CUDA_CHECK(cudaMemcpy(D.d_X, mxGetData(X_m), sizeof(float)*G.N_in*n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D.d_Y, mxGetData(Y_m), sizeof(float)*G.N_out*n, cudaMemcpyHostToDevice));
    if (precompute) {
        CUDA_CHECK(cudaMalloc(&D.d_I, sizeof(float)*G.N_hidden*n));
        float zero = 0.f;
        cublasCheck(cublasSgemm(G.handle, CUBLAS_OP_N, CUBLAS_OP_N,
            G.N_hidden, n, G.N_in, &G.INPUT_SCALE, G.d_W_in, G.N_hidden,
            D.d_X, G.N_in, &zero, D.d_I, G.N_hidden), "precompute W_in*X");
    }
}

static void cmdClearData(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)nlhs; (void)plhs;
    requireInitialized();
    if (nrhs == 1) {
        freeData(G.train); freeData(G.val); freeData(G.test);
        return;
    }
    if (nrhs != 2) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "clear_data expects zero or one split name");
    GpuData& D = selectData(getString(prhs[1], "split"));
    freeData(D);
}

static void cmdInitOptim(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)nlhs; (void)plhs;
    if (nrhs != 9) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "init_optim expects bias_b1,bias_b2,bias_eps,read_lr,read_b1,read_b2,read_eps,read_weight_decay");
    requireInitialized();
    G.bias_b1 = getScalarSingle(prhs[1], "bias_b1");
    G.bias_b2 = getScalarSingle(prhs[2], "bias_b2");
    G.bias_eps = getScalarSingle(prhs[3], "bias_eps");
    G.read_lr = getScalarSingle(prhs[4], "read_lr");
    G.read_b1 = getScalarSingle(prhs[5], "read_b1");
    G.read_b2 = getScalarSingle(prhs[6], "read_b2");
    G.read_eps = getScalarSingle(prhs[7], "read_eps");
    G.read_weight_decay = getScalarSingle(prhs[8], "read_weight_decay");
    int N = G.N_hidden, CN = G.N_out * G.N_hidden;
    freeF(G.d_m_b); freeF(G.d_v_b); freeF(G.d_vhat_b);
    freeF(G.d_M_read); freeF(G.d_V_read); freeF(G.d_Vhat_read);
    CUDA_CHECK(cudaMalloc(&G.d_m_b, sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_v_b, sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_vhat_b, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&G.d_M_read, sizeof(float)*CN)); CUDA_CHECK(cudaMalloc(&G.d_V_read, sizeof(float)*CN)); CUDA_CHECK(cudaMalloc(&G.d_Vhat_read, sizeof(float)*CN));
    CUDA_CHECK(cudaMemset(G.d_m_b,0,sizeof(float)*N)); CUDA_CHECK(cudaMemset(G.d_v_b,0,sizeof(float)*N)); CUDA_CHECK(cudaMemset(G.d_vhat_b,0,sizeof(float)*N));
    CUDA_CHECK(cudaMemset(G.d_M_read,0,sizeof(float)*CN)); CUDA_CHECK(cudaMemset(G.d_V_read,0,sizeof(float)*CN)); CUDA_CHECK(cudaMemset(G.d_Vhat_read,0,sizeof(float)*CN));
    G.t_adam = 0; G.t_read = 0; G.optim_initialized = true;
}

static void applyAdamOnGpu(float lrBias, int nSamples, bool updateBias, bool updateRead)
{
    if (!G.optim_initialized) mexErrMsgIdAndTxt("snn_classify_gpu_mex:optim", "Call init_optim before train_epoch_gpu.");
    float invN = 1.f / fmaxf((float)nSamples, 1.f);
    if (updateBias) {
        G.t_adam += 1;
        float bc1 = 1.f - powf(G.bias_b1, (float)G.t_adam);
        float bc2 = 1.f - powf(G.bias_b2, (float)G.t_adam);
        kAdamBiasUpdate<<<(G.N_hidden+255)/256,256>>>(G.d_B, G.d_gB_epoch, G.d_m_b, G.d_v_b, G.d_vhat_b,
            lrBias, G.bias_b1, G.bias_b2, G.bias_eps, invN, bc1, bc2, G.N_hidden);
        CUDA_CHECK(cudaGetLastError());
    }
    if (updateRead) {
        G.t_read += 1;
        int CN = G.N_out * G.N_hidden;
        float rbc1 = 1.f - powf(G.read_b1, (float)G.t_read);
        float rbc2 = 1.f - powf(G.read_b2, (float)G.t_read);
        kAdamReadUpdate<<<(CN+255)/256,256>>>(G.d_W_read, G.d_Grad_epoch, G.d_M_read, G.d_V_read, G.d_Vhat_read,
            G.read_lr, G.read_b1, G.read_b2, G.read_eps, G.read_weight_decay, invN, rbc1, rbc2, CN);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void runGpuDataset(GpuData& D, const mxArray* order_m, int batchSize, bool doGrad, bool updateWeights,
                               float lrBias, bool updateBias, bool updateRead,
                               double& lossN, double& lossR, double& corrN, double& corrR, int& count)
{
    if (!D.has) mexErrMsgIdAndTxt("snn_classify_gpu_mex:data", "Requested dataset split has not been loaded with set_data.");
    if (batchSize < 1 || batchSize > G.max_batch) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape", "batch_size exceeds max_batch from init");
    const int* d_order = nullptr;
    if (order_m && !mxIsEmpty(order_m)) {
        validateOrderVector(order_m, D.n);
        ensureOrderCapacity(D.n);
        CUDA_CHECK(cudaMemcpy(G.d_order, mxGetData(order_m), sizeof(int)*D.n, cudaMemcpyHostToDevice));
        d_order = G.d_order;
    }
    CUDA_CHECK(cudaMemset(G.d_gB_epoch,0,sizeof(float)*G.N_hidden));
    CUDA_CHECK(cudaMemset(G.d_Grad_epoch,0,sizeof(float)*G.N_out*G.N_hidden));
    CUDA_CHECK(cudaMemset(G.d_metricSums,0,sizeof(float)*4));
    for (int off=0; off<D.n; off+=batchSize) {
        int B = std::min(batchSize, D.n-off);
        int xRows = D.precomputed ? G.N_hidden : G.N_in;
        int xTotal = xRows * B;
        int yTotal = G.N_out * B;
        kGatherColumns<<<(xTotal+255)/256,256>>>(D.precomputed ? D.d_I : D.d_X, D.precomputed ? G.d_I_in : G.d_X, d_order, off, xRows, D.n, B, xTotal);
        CUDA_CHECK(cudaGetLastError());
        kGatherColumns<<<(yTotal+255)/256,256>>>(D.d_Y, G.d_Y, d_order, off, G.N_out, D.n, B, yTotal);
        CUDA_CHECK(cudaGetLastError());
        runBatchDynamics(B, D.precomputed);
        computeLossGrad(B);
        if (doGrad) accumulateGradients(B);
        kReduceClassMetrics<<<1,256>>>(G.d_lossNet,G.d_lossRead,G.d_correctNet,G.d_correctRead,G.d_metricSums,B);
        CUDA_CHECK(cudaGetLastError());
    }
    float h[4] = {0,0,0,0};
    CUDA_CHECK(cudaMemcpy(h, G.d_metricSums, sizeof(float)*4, cudaMemcpyDeviceToHost));
    lossN = h[0]; lossR = h[1]; corrN = h[2]; corrR = h[3]; count = D.n;
    if (updateWeights) applyAdamOnGpu(lrBias, D.n, updateBias, updateRead);
}

static void cmdInit(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 43) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args","init expects 42 arguments after command string");
    (void)nlhs; (void)plhs; clearContext();
    const mxArray *W_in_m=prhs[1], *W_ob_m=prhs[2], *W_out_m=prhs[3], *Eta_m=prhs[4], *dself_m=prhs[5], *B_m=prhs[6], *W_read_m=prhs[7];
    G.alpha=getScalarSingle(prhs[8],"alpha"); G.oneMinusAlpha=getScalarSingle(prhs[9],"oneMinusAlpha"); G.beta=getScalarSingle(prhs[10],"beta"); G.oneMinusBeta=getScalarSingle(prhs[11],"oneMinusBeta");
    G.gamma_sr=getScalarSingle(prhs[12],"gamma_sr"); G.gamma_sd=getScalarSingle(prhs[13],"gamma_sd"); G.READ_gamma_sr=getScalarSingle(prhs[14],"READ_gamma_sr"); G.READ_gamma_sd=getScalarSingle(prhs[15],"READ_gamma_sd");
    G.E_L=getScalarSingle(prhs[16],"E_L"); G.V_th=getScalarSingle(prhs[17],"V_th"); G.V_reset=getScalarSingle(prhs[18],"V_reset"); G.a_eff=getScalarSingle(prhs[19],"a_eff"); G.b_param=getScalarSingle(prhs[20],"b_param");
    G.phi_u=getScalarSingle(prhs[21],"phi_u"); G.delta_u=getScalarSingle(prhs[22],"delta_u"); G.INPUT_SCALE=getScalarSingle(prhs[23],"INPUT_SCALE"); G.SCALE_rec=getScalarSingle(prhs[24],"SCALE_rec");
    G.spike_jump_sr=getScalarSingle(prhs[25],"spike_jump_sr"); G.spike_jump_sr_R=getScalarSingle(prhs[26],"spike_jump_sr_R");
    G.steps=getIntScalar(prhs[27],"steps_present"); G.N_hidden=getIntScalar(prhs[28],"N_hidden"); G.N_rec=getIntScalar(prhs[29],"N_rec"); G.N_in=getIntScalar(prhs[30],"N_in"); G.N_out=getIntScalar(prhs[31],"N_out"); G.k_avg_start=getIntScalar(prhs[32],"k_avg_start"); G.steps_avg=getIntScalar(prhs[33],"steps_avg"); G.max_batch=getIntScalar(prhs[34],"max_batch");
    const mxArray *W_rec_m=prhs[35];
    G.recurrent_mode=getIntScalar(prhs[36],"recurrent_mode_id");
    G.decoder_mode=getIntScalar(prhs[37],"decoder_mode_id");
    G.recurrent_storage=getIntScalar(prhs[38],"recurrent_storage_id");
    const mxArray *rec_post_m=prhs[39], *rec_pre_m=prhs[40], *rec_w_m=prhs[41];
    G.rec_nnz=getIntScalar(prhs[42],"rec_nnz");
    if (G.recurrent_mode != RECURRENT_LOW_RANK && G.recurrent_mode != RECURRENT_FULL_RANK)
        mexErrMsgIdAndTxt("snn_classify_gpu_mex:arch","recurrent_mode_id must be 0 (low_rank) or 1 (full_rank)");
    if (G.decoder_mode != DECODER_SHARED && G.decoder_mode != DECODER_SIGNED)
        mexErrMsgIdAndTxt("snn_classify_gpu_mex:arch","decoder_mode_id must be 0 (shared) or 1 (signed)");
    if (G.recurrent_storage != REC_STORAGE_DENSE && G.recurrent_storage != REC_STORAGE_SPARSE)
        mexErrMsgIdAndTxt("snn_classify_gpu_mex:arch","recurrent_storage_id must be 0 (dense) or 1 (sparse)");
    assertSingle(W_in_m,"W_in"); assertSingle(W_ob_m,"W_out_base_rec"); assertSingle(W_out_m,"W_out"); assertSingle(Eta_m,"Eta_rec"); assertSingle(dself_m,"dself"); assertSingle(B_m,"B"); assertSingle(W_read_m,"W_read");
    assertSingle(W_rec_m,"W_rec"); assertSingle(rec_w_m,"rec_w");
    if (mxGetClassID(rec_post_m) != mxINT32_CLASS || mxGetClassID(rec_pre_m) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("snn_classify_gpu_mex:type","rec_post_idx and rec_pre_idx must be int32");
    if ((int)mxGetM(W_in_m)!=G.N_hidden || (int)mxGetN(W_in_m)!=G.N_in) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","W_in shape mismatch");
    if ((int)mxGetM(W_ob_m)!=G.N_rec || (int)mxGetN(W_ob_m)!=G.N_hidden) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","W_out_base_rec shape mismatch");
    if ((int)mxGetM(W_out_m)!=G.N_out || (int)mxGetN(W_out_m)!=G.N_hidden) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","W_out shape mismatch");
    if ((int)mxGetM(Eta_m)!=G.N_hidden || (int)mxGetN(Eta_m)!=G.N_rec) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","Eta_rec shape mismatch");
    if ((int)mxGetNumberOfElements(dself_m)!=G.N_hidden || (int)mxGetNumberOfElements(B_m)!=G.N_hidden) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","bias vector size mismatch");
    if ((int)mxGetM(W_read_m)!=G.N_out || (int)mxGetN(W_read_m)!=G.N_hidden) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","W_read shape mismatch");
    if (G.rec_nnz < 0 || (int)mxGetNumberOfElements(rec_post_m) != G.rec_nnz || (int)mxGetNumberOfElements(rec_pre_m) != G.rec_nnz || (int)mxGetNumberOfElements(rec_w_m) != G.rec_nnz)
        mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","sparse recurrent edge-list length mismatch");
    if (G.recurrent_mode == RECURRENT_FULL_RANK && G.recurrent_storage == REC_STORAGE_DENSE && ((int)mxGetM(W_rec_m)!=G.N_hidden || (int)mxGetN(W_rec_m)!=G.N_hidden))
        mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","W_rec shape mismatch for full_rank recurrence");
    if (G.max_batch < 1) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","max_batch must be positive");
    allocContext();
    CUDA_CHECK(cudaMemcpy(G.d_W_in,mxGetData(W_in_m),sizeof(float)*G.N_hidden*G.N_in,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(G.d_W_out_b,mxGetData(W_ob_m),sizeof(float)*G.N_rec*G.N_hidden,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.d_W_out,mxGetData(W_out_m),sizeof(float)*G.N_out*G.N_hidden,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(G.d_Eta,mxGetData(Eta_m),sizeof(float)*G.N_hidden*G.N_rec,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.d_dself,mxGetData(dself_m),sizeof(float)*G.N_hidden,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(G.d_B,mxGetData(B_m),sizeof(float)*G.N_hidden,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(G.d_W_read,mxGetData(W_read_m),sizeof(float)*G.N_out*G.N_hidden,cudaMemcpyHostToDevice));
    if (G.recurrent_mode == RECURRENT_FULL_RANK && G.recurrent_storage == REC_STORAGE_DENSE)
        CUDA_CHECK(cudaMemcpy(G.d_W_rec,mxGetData(W_rec_m),sizeof(float)*G.N_hidden*G.N_hidden,cudaMemcpyHostToDevice));
    if (G.recurrent_mode == RECURRENT_FULL_RANK && G.recurrent_storage == REC_STORAGE_SPARSE && G.rec_nnz > 0) {
        CUDA_CHECK(cudaMemcpy(G.d_rec_post,mxGetData(rec_post_m),sizeof(int)*G.rec_nnz,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(G.d_rec_pre,mxGetData(rec_pre_m),sizeof(int)*G.rec_nnz,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(G.d_rec_w,mxGetData(rec_w_m),sizeof(float)*G.rec_nnz,cudaMemcpyHostToDevice));
    }
    G.log_alpha=logf(fmaxf(G.alpha,REALMIN_SINGLE)); G.log_beta=logf(fmaxf(G.beta,REALMIN_SINGLE)); G.log_gsr=logf(fmaxf(G.gamma_sr,REALMIN_SINGLE)); G.log_gsd=logf(fmaxf(G.gamma_sd,REALMIN_SINGLE)); G.log_Rgsr=logf(fmaxf(G.READ_gamma_sr,REALMIN_SINGLE)); G.log_Rgsd=logf(fmaxf(G.READ_gamma_sd,REALMIN_SINGLE));
    G.initialized=true; mexAtExit(clearContext);
}
static void cmdUpdateParams(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)nlhs; (void)plhs; if (nrhs!=3) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args","update_params expects B and W_read"); requireInitialized(); assertSingle(prhs[1],"B"); assertSingle(prhs[2],"W_read");
    if ((int)mxGetNumberOfElements(prhs[1]) != G.N_hidden || (int)mxGetM(prhs[2]) != G.N_out || (int)mxGetN(prhs[2]) != G.N_hidden)
        mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","B or W_read shape mismatch");
    CUDA_CHECK(cudaMemcpy(G.d_B,mxGetData(prhs[1]),sizeof(float)*G.N_hidden,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(G.d_W_read,mxGetData(prhs[2]),sizeof(float)*G.N_out*G.N_hidden,cudaMemcpyHostToDevice));
}

static void runDataset(const mxArray* X_m, const mxArray* Y_m, const mxArray* order_m, int batchSize, bool doGrad,
                       double& lossN, double& lossR, double& corrN, double& corrR, int& count)
{
    assertSingle(X_m,"X"); assertSingle(Y_m,"Y"); if ((int)mxGetM(X_m)!=G.N_in || (int)mxGetM(Y_m)!=G.N_out) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","X/Y must be transposed [features/classes x samples]");
    int n=(int)mxGetN(X_m); if ((int)mxGetN(Y_m)!=n) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","X/Y sample count mismatch");
    if (batchSize<1 || batchSize>G.max_batch) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","batch_size exceeds max_batch from init");
    const float* X=(const float*)mxGetData(X_m); const float* Y=(const float*)mxGetData(Y_m); const int* order=nullptr;
    if (order_m && !mxIsEmpty(order_m)) { if (mxGetClassID(order_m)!=mxINT32_CLASS) mexErrMsgIdAndTxt("snn_classify_gpu_mex:type","order must be int32"); if ((int)mxGetNumberOfElements(order_m)!=n) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape","order length mismatch"); order=(const int*)mxGetData(order_m); }
    CUDA_CHECK(cudaMemset(G.d_gB_epoch,0,sizeof(float)*G.N_hidden)); CUDA_CHECK(cudaMemset(G.d_Grad_epoch,0,sizeof(float)*G.N_out*G.N_hidden));
    std::vector<float> hx, hy, hLossN(G.max_batch), hLossR(G.max_batch); std::vector<int> hCorrN(G.max_batch), hCorrR(G.max_batch);
    lossN=0; lossR=0; corrN=0; corrR=0; count=n;
    for (int off=0; off<n; off+=batchSize) { int B=std::min(batchSize,n-off); gatherBatch(X,n,Y,n,order,off,B,hx,hy); CUDA_CHECK(cudaMemcpy(G.d_X,hx.data(),sizeof(float)*G.N_in*B,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(G.d_Y,hy.data(),sizeof(float)*G.N_out*B,cudaMemcpyHostToDevice)); runBatchDynamics(B); computeLossGrad(B); if (doGrad) accumulateGradients(B); CUDA_CHECK(cudaMemcpy(hLossN.data(),G.d_lossNet,sizeof(float)*B,cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hLossR.data(),G.d_lossRead,sizeof(float)*B,cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hCorrN.data(),G.d_correctNet,sizeof(int)*B,cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hCorrR.data(),G.d_correctRead,sizeof(int)*B,cudaMemcpyDeviceToHost)); for(int j=0;j<B;++j){lossN+=hLossN[j]; lossR+=hLossR[j]; corrN+=hCorrN[j]; corrR+=hCorrR[j];} }
}
static void cmdTrainEpoch(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs!=5 || nlhs!=7) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args","train_epoch returns [lossN,lossR,corrN,corrR,count,gB,Grad]"); requireInitialized(); int batchSize=getIntScalar(prhs[4],"batch_size"); double ln,lr,cn,cr; int count; runDataset(prhs[1],prhs[2],prhs[3],batchSize,true,ln,lr,cn,cr,count);
    plhs[0]=mxCreateDoubleScalar(ln); plhs[1]=mxCreateDoubleScalar(lr); plhs[2]=mxCreateDoubleScalar(cn); plhs[3]=mxCreateDoubleScalar(cr); plhs[4]=mxCreateDoubleScalar((double)count); plhs[5]=mxCreateNumericMatrix(G.N_hidden,1,mxSINGLE_CLASS,mxREAL); plhs[6]=mxCreateNumericMatrix(G.N_out,G.N_hidden,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[5]),G.d_gB_epoch,sizeof(float)*G.N_hidden,cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[6]),G.d_Grad_epoch,sizeof(float)*G.N_out*G.N_hidden,cudaMemcpyDeviceToHost));
}
static void cmdValidate(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs!=4 || nlhs!=5) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args","validate returns [lossN,lossR,corrN,corrR,count]"); requireInitialized(); int batchSize=getIntScalar(prhs[3],"batch_size"); double ln,lr,cn,cr; int count; runDataset(prhs[1],prhs[2],nullptr,batchSize,false,ln,lr,cn,cr,count); plhs[0]=mxCreateDoubleScalar(ln); plhs[1]=mxCreateDoubleScalar(lr); plhs[2]=mxCreateDoubleScalar(cn); plhs[3]=mxCreateDoubleScalar(cr); plhs[4]=mxCreateDoubleScalar((double)count);
}

static void cmdTrainEpochGpu(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 5 || nlhs > 5) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "train_epoch_gpu returns up to [lossN,lossR,corrN,corrR,count]");
    requireInitialized();
    GpuData& D = selectData(getString(prhs[1], "split"));
    int batchSize = getIntScalar(prhs[3], "batch_size");
    float lrBias = getScalarSingle(prhs[4], "lr_bias");
    double ln, lr, cn, cr; int count;
    runGpuDataset(D, prhs[2], batchSize, true, true, lrBias, true, true, ln, lr, cn, cr, count);
    if (nlhs > 0) plhs[0]=mxCreateDoubleScalar(ln);
    if (nlhs > 1) plhs[1]=mxCreateDoubleScalar(lr);
    if (nlhs > 2) plhs[2]=mxCreateDoubleScalar(cn);
    if (nlhs > 3) plhs[3]=mxCreateDoubleScalar(cr);
    if (nlhs > 4) plhs[4]=mxCreateDoubleScalar((double)count);
}

static void cmdTrainPrimaryEpochGpu(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 5 || nlhs > 3) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "train_primary_epoch returns up to [loss,correct,count]");
    requireInitialized();
    GpuData& D = selectData(getString(prhs[1], "split"));
    int batchSize = getIntScalar(prhs[3], "batch_size");
    float lrBias = getScalarSingle(prhs[4], "lr_bias");
    double lossN, lossIgnored, correctN, correctIgnored; int count;
    runGpuDataset(D, prhs[2], batchSize, true, true, lrBias, true, false, lossN, lossIgnored, correctN, correctIgnored, count);
    if (nlhs > 0) plhs[0]=mxCreateDoubleScalar(lossN);
    if (nlhs > 1) plhs[1]=mxCreateDoubleScalar(correctN);
    if (nlhs > 2) plhs[2]=mxCreateDoubleScalar((double)count);
}

static void cmdValidateGpu(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 3 || nlhs > 5) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "validate_gpu returns up to [lossN,lossR,corrN,corrR,count]");
    requireInitialized();
    GpuData& D = selectData(getString(prhs[1], "split"));
    int batchSize = getIntScalar(prhs[2], "batch_size");
    double ln, lr, cn, cr; int count;
    runGpuDataset(D, nullptr, batchSize, false, false, 0.f, false, false, ln, lr, cn, cr, count);
    if (nlhs > 0) plhs[0]=mxCreateDoubleScalar(ln);
    if (nlhs > 1) plhs[1]=mxCreateDoubleScalar(lr);
    if (nlhs > 2) plhs[2]=mxCreateDoubleScalar(cn);
    if (nlhs > 3) plhs[3]=mxCreateDoubleScalar(cr);
    if (nlhs > 4) plhs[4]=mxCreateDoubleScalar((double)count);
}

static void cmdValidatePrimaryGpu(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 3 || nlhs > 3) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "validate_primary returns up to [loss,correct,count]");
    requireInitialized();
    GpuData& D = selectData(getString(prhs[1], "split"));
    int batchSize = getIntScalar(prhs[2], "batch_size");
    double lossN, lossIgnored, correctN, correctIgnored; int count;
    runGpuDataset(D, nullptr, batchSize, false, false, 0.f, false, false, lossN, lossIgnored, correctN, correctIgnored, count);
    if (nlhs > 0) plhs[0]=mxCreateDoubleScalar(lossN);
    if (nlhs > 1) plhs[1]=mxCreateDoubleScalar(correctN);
    if (nlhs > 2) plhs[2]=mxCreateDoubleScalar((double)count);
}

static void cmdPredictPrimaryGpu(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 3 || nlhs != 4) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "predict_primary returns [loss,correct,count,Z]");
    requireInitialized();
    GpuData& D = selectData(getString(prhs[1], "split"));
    if (!D.has) mexErrMsgIdAndTxt("snn_classify_gpu_mex:data", "Requested dataset split has not been loaded with set_data.");
    int batchSize = getIntScalar(prhs[2], "batch_size");
    if (batchSize < 1 || batchSize > G.max_batch) mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape", "batch_size exceeds max_batch from init");
    mxArray* Zall = mxCreateNumericMatrix(G.N_out, D.n, mxSINGLE_CLASS, mxREAL);
    float* hZ = (float*)mxGetData(Zall);
    CUDA_CHECK(cudaMemset(G.d_metricSums,0,sizeof(float)*4));
    for (int off=0; off<D.n; off+=batchSize) {
        int B = std::min(batchSize, D.n-off);
        int xRows = D.precomputed ? G.N_hidden : G.N_in;
        int xTotal = xRows * B;
        int yTotal = G.N_out * B;
        kGatherColumns<<<(xTotal+255)/256,256>>>(D.precomputed ? D.d_I : D.d_X, D.precomputed ? G.d_I_in : G.d_X, nullptr, off, xRows, D.n, B, xTotal);
        CUDA_CHECK(cudaGetLastError());
        kGatherColumns<<<(yTotal+255)/256,256>>>(D.d_Y, G.d_Y, nullptr, off, G.N_out, D.n, B, yTotal);
        CUDA_CHECK(cudaGetLastError());
        runBatchDynamics(B, D.precomputed);
        computeLossGrad(B);
        kReduceClassMetrics<<<1,256>>>(G.d_lossNet,G.d_lossRead,G.d_correctNet,G.d_correctRead,G.d_metricSums,B);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hZ + (size_t)G.N_out*off, G.d_Z_sum, sizeof(float)*G.N_out*B, cudaMemcpyDeviceToHost));
    }
    float h[4] = {0,0,0,0};
    CUDA_CHECK(cudaMemcpy(h, G.d_metricSums, sizeof(float)*4, cudaMemcpyDeviceToHost));
    plhs[0]=mxCreateDoubleScalar(h[0]);
    plhs[1]=mxCreateDoubleScalar(h[2]);
    plhs[2]=mxCreateDoubleScalar((double)D.n);
    plhs[3]=Zall;
}

static void cmdGetParams(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)prhs;
    if (nrhs != 1 || nlhs != 2) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "get_params returns [B,W_read]");
    requireInitialized();
    plhs[0]=mxCreateNumericMatrix(G.N_hidden,1,mxSINGLE_CLASS,mxREAL);
    plhs[1]=mxCreateNumericMatrix(G.N_out,G.N_hidden,mxSINGLE_CLASS,mxREAL);
    CUDA_CHECK(cudaMemcpy(mxGetData(plhs[0]), G.d_B, sizeof(float)*G.N_hidden, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mxGetData(plhs[1]), G.d_W_read, sizeof(float)*G.N_out*G.N_hidden, cudaMemcpyDeviceToHost));
}

static void cmdGetBias(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)prhs;
    if (nrhs != 1 || nlhs != 1) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "get_bias returns B");
    requireInitialized();
    plhs[0]=mxCreateNumericMatrix(G.N_hidden,1,mxSINGLE_CLASS,mxREAL);
    CUDA_CHECK(cudaMemcpy(mxGetData(plhs[0]), G.d_B, sizeof(float)*G.N_hidden, cudaMemcpyDeviceToHost));
}

static void cmdGetOptimState(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)prhs;
    if (nrhs != 1 || nlhs != 4) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "get_optim_state returns [m_b,v_b,vhat_b,t_adam]");
    requireInitialized();
    if (!G.optim_initialized) mexErrMsgIdAndTxt("snn_classify_gpu_mex:optim", "Call init_optim before get_optim_state.");
    plhs[0]=mxCreateNumericMatrix(G.N_hidden,1,mxSINGLE_CLASS,mxREAL);
    plhs[1]=mxCreateNumericMatrix(G.N_hidden,1,mxSINGLE_CLASS,mxREAL);
    plhs[2]=mxCreateNumericMatrix(G.N_hidden,1,mxSINGLE_CLASS,mxREAL);
    plhs[3]=mxCreateDoubleScalar((double)G.t_adam);
    CUDA_CHECK(cudaMemcpy(mxGetData(plhs[0]), G.d_m_b, sizeof(float)*G.N_hidden, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mxGetData(plhs[1]), G.d_v_b, sizeof(float)*G.N_hidden, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mxGetData(plhs[2]), G.d_vhat_b, sizeof(float)*G.N_hidden, cudaMemcpyDeviceToHost));
}

static void cmdSetOptimState(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)nlhs; (void)plhs;
    if (nrhs != 5) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "set_optim_state expects m_b,v_b,vhat_b,t_adam");
    requireInitialized();
    if (!G.optim_initialized) mexErrMsgIdAndTxt("snn_classify_gpu_mex:optim", "Call init_optim before set_optim_state.");
    assertSingle(prhs[1], "m_b");
    assertSingle(prhs[2], "v_b");
    assertSingle(prhs[3], "vhat_b");
    if ((int)mxGetNumberOfElements(prhs[1]) != G.N_hidden ||
        (int)mxGetNumberOfElements(prhs[2]) != G.N_hidden ||
        (int)mxGetNumberOfElements(prhs[3]) != G.N_hidden)
        mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape", "Optimizer state vector length mismatch");
    CUDA_CHECK(cudaMemcpy(G.d_m_b, mxGetData(prhs[1]), sizeof(float)*G.N_hidden, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.d_v_b, mxGetData(prhs[2]), sizeof(float)*G.N_hidden, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.d_vhat_b, mxGetData(prhs[3]), sizeof(float)*G.N_hidden, cudaMemcpyHostToDevice));
    G.t_adam = getIntScalar(prhs[4], "t_adam");
}

static void cmdUpdateBias(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)nlhs; (void)plhs;
    if (nrhs != 2) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "update_bias expects B");
    requireInitialized();
    assertSingle(prhs[1], "B");
    if ((int)mxGetNumberOfElements(prhs[1]) != G.N_hidden)
        mexErrMsgIdAndTxt("snn_classify_gpu_mex:shape", "B shape mismatch");
    CUDA_CHECK(cudaMemcpy(G.d_B, mxGetData(prhs[1]), sizeof(float)*G.N_hidden, cudaMemcpyHostToDevice));
}

static void cmdTrainReadEpochGpu(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 4 || nlhs > 5) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args", "train_read_epoch_gpu returns up to [lossN,lossR,corrN,corrR,count]");
    requireInitialized();
    GpuData& D = selectData(getString(prhs[1], "split"));
    int batchSize = getIntScalar(prhs[3], "batch_size");
    double ln, lr, cn, cr; int count;
    runGpuDataset(D, prhs[2], batchSize, true, true, 0.f, false, true, ln, lr, cn, cr, count);
    if (nlhs > 0) plhs[0]=mxCreateDoubleScalar(ln);
    if (nlhs > 1) plhs[1]=mxCreateDoubleScalar(lr);
    if (nlhs > 2) plhs[2]=mxCreateDoubleScalar(cn);
    if (nlhs > 3) plhs[3]=mxCreateDoubleScalar(cr);
    if (nlhs > 4) plhs[4]=mxCreateDoubleScalar((double)count);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if(nrhs<1 || !mxIsChar(prhs[0])) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args","First input must be command string");
    char* cmd_c=mxArrayToString(prhs[0]); if(!cmd_c) mexErrMsgIdAndTxt("snn_classify_gpu_mex:args","Could not parse command"); std::string cmd(cmd_c); mxFree(cmd_c);
    if(cmd=="init"){cmdInit(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="update_params"){cmdUpdateParams(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="train_epoch"){cmdTrainEpoch(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="validate"){cmdValidate(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="init_optim"){cmdInitOptim(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="set_data"){cmdSetData(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="clear_data"){cmdClearData(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="train_epoch_gpu"){cmdTrainEpochGpu(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="train_primary_epoch"){cmdTrainPrimaryEpochGpu(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="train_read_epoch_gpu"){cmdTrainReadEpochGpu(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="validate_gpu"){cmdValidateGpu(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="validate_primary"){cmdValidatePrimaryGpu(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="predict_primary"){cmdPredictPrimaryGpu(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="get_params"){cmdGetParams(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="get_bias"){cmdGetBias(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="get_optim_state"){cmdGetOptimState(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="set_optim_state"){cmdSetOptimState(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="update_bias"){cmdUpdateBias(nlhs,plhs,nrhs,prhs); return;}
    if(cmd=="clear"){clearContext(); return;}
    mexErrMsgIdAndTxt("snn_classify_gpu_mex:args","Unknown command '%s'",cmd.c_str());
}
