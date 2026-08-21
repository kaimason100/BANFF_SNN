// Package orientation: CUDA/MEX backend. MATLAB wrappers set options and data layout; this file implements GPU kernels, resident state, and command dispatch. Keep argument order aligned with the matching init/train/eval helpers.

// snn_time_loop_gpu_mex.cu  (col-major GPU implementation)
// Build (Windows):
//   mexcuda -R2018a -output snn_time_loop_gpu_mex snn_time_loop_gpu_mex.cu -O ...
//            NVCCFLAGS="--fmad=false --prec-div=true --prec-sqrt=true"
// Build (Linux/macOS):
//   mexcuda -R2018a -output snn_time_loop_gpu_mex snn_time_loop_gpu_mex.cu -O ...
//            NVCCFLAGS="--fmad=false --prec-div=true --prec-sqrt=true"

#include "mex.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <vector>
using std::vector;

#ifndef REALMIN_SINGLE
#define REALMIN_SINGLE 1.17549435e-38f
#endif

enum { RECURRENT_LOW_RANK = 0, RECURRENT_FULL_RANK = 1 };
enum { DECODER_SHARED = 0, DECODER_SIGNED = 1 };
enum { REC_STORAGE_DENSE = 0, REC_STORAGE_SPARSE = 1 };

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    mexErrMsgIdAndTxt("snn_mex:cuda", "CUDA error %d (%s) at %s:%d", \
                      (int)err, cudaGetErrorString(err), __FILE__, __LINE__); } } while(0)

static inline bool isVectorMx(const mxArray* a)
{
    if (mxGetNumberOfDimensions(a) != 2) return false;
    mwSize m = mxGetM(a), n = mxGetN(a);
    return (m==1) || (n==1);
}
static inline void assertSingle(const mxArray* a, const char* name)
{
    if (mxGetClassID(a) != mxSINGLE_CLASS)
        mexErrMsgIdAndTxt("snn_mex:type", "%s must be single", name);
    if (mxIsComplex(a))
        mexErrMsgIdAndTxt("snn_mex:type", "%s must be real", name);
}
static inline void assertLogicalOrNumeric(const mxArray* a, const char* name)
{
    mxClassID cls = mxGetClassID(a);
    if (!(cls==mxLOGICAL_CLASS || cls==mxSINGLE_CLASS || cls==mxDOUBLE_CLASS || cls==mxUINT8_CLASS))
        mexErrMsgIdAndTxt("snn_mex:type", "%s must be logical or numeric", name);
}

__host__ __device__ inline int idx2(int i, int m, int j) { return i + m*j; }

// ====== new: col-major dot (single row per thread) ======
// y[N] = scale * (A(NxD, col-major) * x[D])
__global__ void kDotCol(const float* __restrict__ A, const float* __restrict__ x,
                        float* __restrict__ y, int N, int D, float scale)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.f;
#pragma unroll 4
    for (int d=0; d<D; ++d) acc += A[n + d*N] * x[d];
    y[n] = scale * acc;
}

__global__ void kSparseRec(const int* __restrict__ post,
                           const int* __restrict__ pre,
                           const float* __restrict__ w,
                           int nnz,
                           const float* __restrict__ r,
                           float* __restrict__ Irec)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nnz) return;
    atomicAdd(Irec + post[e], w[e] * r[pre[e]]);
}

// ====== new: y_dec = W_out_base_rec * r  (shared-memory reduction) ======
__global__ void kYDecReduce(const float* __restrict__ Wobr, const float* __restrict__ r,
                            float* __restrict__ y_dec, int Nrec, int N)
{
    int rid = blockIdx.x;              // 0..Nrec-1
    if (rid >= Nrec) return;
    extern __shared__ float sh[];
    float sum = 0.f;
    for (int h = threadIdx.x; h < N; h += blockDim.x) {
        sum += Wobr[rid + h * Nrec] * r[h]; // col-major [Nrec x N]
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x>>1; s>0; s>>=1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x==0) y_dec[rid] = sh[0];
}

// ====== new: outputs Z(:,k) and Y(:,k) with reductions over N ======
__global__ void kOutputsColReduce(const float* __restrict__ Wout,
                                  const float* __restrict__ Wread,
                                  const float* __restrict__ r,
                                  const float* __restrict__ r_read,
                                  float* __restrict__ Z_col_k,
                                  float* __restrict__ Y_col_k,
                                  int D, int N)
{
    int d = blockIdx.x; // one block per output dim
    if (d >= D) return;
    extern __shared__ float sh[];
    // Z
    float sZ = 0.f;
    for (int h = threadIdx.x; h < N; h += blockDim.x)
        sZ += Wout [d + h*D] * r[h]; // col-major [D x N]
    sh[threadIdx.x] = sZ;
    __syncthreads();
    for (int s = blockDim.x>>1; s>0; s>>=1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x==0) Z_col_k[d] = sh[0];

    // reuse sh for Y
    float sY = 0.f;
    for (int h = threadIdx.x; h < N; h += blockDim.x)
        sY += Wread[d + h*D] * r_read[h]; // col-major [D x N]
    sh[threadIdx.x] = sY;
    __syncthreads();
    for (int s = blockDim.x>>1; s>0; s>>=1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x==0) Y_col_k[d] = sh[0];
}


// ===== keep: cheap combiners / unchanged maths =====
__global__ void kBuildItot(const float* __restrict__ I_in,
                           const float* __restrict__ Irec_scaled, // SCALE_rec*Eta*y_dec
                           const float* __restrict__ r,
                           const float* __restrict__ dself,
                           const float* __restrict__ B,
                           float* __restrict__ I_rec,  // out (for logging)
                           float* __restrict__ I_tot,  // out (for update)
                           float* __restrict__ I_in_store_col,  // optional log
                           float* __restrict__ I_rec_store_col, // optional log
                           int N_hidden)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_hidden) return;
    float irec = Irec_scaled[i] - dself[i] * r[i];
    float itot = I_in[i] + irec + B[i];
    I_rec[i] = irec; I_tot[i] = itot;
    if (I_in_store_col)  I_in_store_col[i]  = I_in[i];
    if (I_rec_store_col) I_rec_store_col[i] = irec;
}

__global__ void kBuildItotFullRank(const float* __restrict__ I_in,
                                   const float* __restrict__ Irec_full,
                                   const float* __restrict__ B,
                                   float* __restrict__ I_rec,
                                   float* __restrict__ I_tot,
                                   float* __restrict__ I_in_store_col,
                                   float* __restrict__ I_rec_store_col,
                                   int N_hidden)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_hidden) return;
    float irec = Irec_full[i];
    I_rec[i] = irec;
    I_tot[i] = I_in[i] + irec + B[i];
    if (I_in_store_col)  I_in_store_col[i]  = I_in[i];
    if (I_rec_store_col) I_rec_store_col[i] = irec;
}

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
                           bool*  __restrict__ spike_bool,
                           float* __restrict__ surr_out,
                           int N_hidden)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_hidden) return;

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

    float oneMinusAlpha_pre  = 1.f - alpha_pre;
    float oneMinusBeta_pre   = 1.f - beta_pre;
    float oneMinusAlpha_post = 1.f - alpha_post;
    float oneMinusBeta_post  = 1.f - beta_post;

    float u_star = E_L + alpha_pre * (u0 - E_L) + oneMinusAlpha_pre * (It - w0);
    float w1     = beta_pre * w0 + oneMinusBeta_pre * (a_eff * (u_star - E_L));


        // --- LSTI-exact surrogate: evaluate at the linear-crossing state
    // u_lin = u_pre + rho * (u_hat - u_pre)
    float u_lin  = u0 + rho * (u_hat - u0);
    float den_pd = fmaxf(delta_u, REALMIN_SINGLE);
    float x      = (u_lin - V_th) / den_pd;
    float surr   = phi_u * fmaxf(0.f, 1.f - fabsf(x));


    if (sp) { u_star = V_reset; w1 = w1 + b_param; }

    float u2 = E_L + alpha_post * (u_star - E_L) + oneMinusAlpha_post * (It - w1);
    float w2 = beta_post  * w1 + oneMinusBeta_post  * (a_eff * (u2 - E_L));

    u_out[i]     = u2;
    w_out[i]     = w2;
    rho_out[i]   = rho;
    spike_bool[i]= sp;
    surr_out[i]  = surr;
}

__global__ void kCascadeAdvancePre(const float* __restrict__ rho,
                                   float* __restrict__ x_syn,
                                   float* __restrict__ r,
                                   float log_gamma_sr, float log_gamma_sd,
                                   int N_hidden)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_hidden) return;
    float f = rho[i];
    float gsr_f = expf(f * log_gamma_sr);
    float gsd_f = expf(f * log_gamma_sd);
    float x1 = gsr_f * x_syn[i];
    x_syn[i] = x1;
    float r1 = gsd_f * r[i] + (1.f - gsd_f) * x1;
    r[i] = r1;
}
__global__ void kCascadeAdvancePost(const float* __restrict__ rho,
                                    float* __restrict__ x_syn,
                                    float* __restrict__ r,
                                    float log_gamma_sr, float log_gamma_sd,
                                    int N_hidden)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_hidden) return;
    float f = 1.f - rho[i];
    float gsr_f = expf(f * log_gamma_sr);
    float gsd_f = expf(f * log_gamma_sd);
    float x1 = gsr_f * x_syn[i];
    x_syn[i] = x1;
    float r1 = gsd_f * r[i] + (1.f - gsd_f) * x1;
    r[i] = r1;
}
__global__ void kAddSpikeJumps(const bool* __restrict__ spike, float* __restrict__ x_syn, float jump, int N_hidden)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_hidden) return;
    if (spike[i]) x_syn[i] = x_syn[i] + jump;
}

// ===== modified: bias e-prop uses Wout col-major =====
__global__ void kBiasEpropAndGrad_COL(const float* __restrict__ Wout_col,
                                      const float* __restrict__ gZ,
                                      float* __restrict__ eps_v_noa,
                                      float* __restrict__ eps_a,
                                      float* __restrict__ Ebar_x,
                                      float* __restrict__ Ebar_f,
                                      const bool*  __restrict__ spike,
                                      const float* __restrict__ surr,
                                      float spike_jump_sr,
                                      float log_gamma_sr, float log_gamma_sd,
                                      const float* __restrict__ rho,
                                      float a_eff, float b_param,
                                      float log_alpha, float log_beta,
                                      float* __restrict__ g_b_traj,
                                      int D, int N_hidden)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_hidden) return;

    // Lstd = sum_d W_out(d,i) * gZ(d) ; Wout_col is MATLAB col-major (D x N_hidden)
    float Lstd = 0.f;
#pragma unroll 4
    for (int d=0; d<D; ++d) Lstd += Wout_col[d + i*D] * gZ[d];

    float ev = eps_v_noa[i];
    float ea = eps_a[i];
    float Ex = Ebar_x[i];
    float Ef = Ebar_f[i];

    float rhi = rho[i];
    float a_pre  = expf(rhi        * log_alpha);
    float b_pre  = expf(rhi        * log_beta );
    float omApre = 1.f - a_pre;
    float omBpre = 1.f - b_pre;
    float a_post  = expf((1.f-rhi) * log_alpha);
    float b_post  = expf((1.f-rhi) * log_beta );
    float omApost = 1.f - a_post;
    float omBpost = 1.f - b_post;

    float surr_i = surr[i];

    ev = a_pre * ev + omApre;
    float ev_full_pre = ev - omApre * ea;
    float e_raw = surr_i * ev_full_pre;
    ea = b_pre * ea + omBpre * (a_eff * ev_full_pre);

    if (spike[i]) { ev = 0.f; ea = ea + b_param * e_raw; }

    ev = a_post * ev + omApost;
    float ev_full = ev - omApost * ea;
    ea = b_post * ea + omBpost * (a_eff * ev_full);

    float gsr_pre  = expf(rhi        * log_gamma_sr);
    float gsd_pre  = expf(rhi        * log_gamma_sd);
    float gsr_post = expf((1.f-rhi)  * log_gamma_sr);
    float gsd_post = expf((1.f-rhi)  * log_gamma_sd);

    float Ex_pre = gsr_pre * Ex;
    Ef = gsd_pre * Ef + (1.f - gsd_pre) * Ex_pre;
    if (spike[i]) Ex_pre = Ex_pre + spike_jump_sr * e_raw;
    Ex = gsr_post * Ex_pre;
    Ef = gsd_post * Ef + (1.f - gsd_post) * Ex;

    g_b_traj[i] += Lstd * Ef;

    eps_v_noa[i] = ev;
    eps_a[i]     = ea;
    Ebar_x[i]    = Ex;
    Ebar_f[i]    = Ef;
}

// ===== keep: read grad outer product on (E .* phi) =====
__global__ void kReadGradAccumulate(const float* __restrict__ E,
                                    const float* __restrict__ phi,
                                    float* __restrict__ Grad_epoch,
                                    int D, int N_hidden)
{
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D || i >= N_hidden) return;
    float add = 2.f * E[d] * phi[i];
    // MATLAB wants [D x N_hidden] col-major
    Grad_epoch[d + i*D] += add;
}

// ===== misc kernels unchanged =====
__global__ void kComputeErrors(const float* __restrict__ Zk,
                               const float* __restrict__ yk,
                               const float* __restrict__ t_sup,
                               int D,
                               float* __restrict__ gZ,
                               float* __restrict__ E,
                               float* __restrict__ traj_loss_accum,
                               float* __restrict__ read_loss_accum,
                               bool doReadGrad)
{
    if (threadIdx.x==0 && blockIdx.x==0) {
        float ls = 0.f, lread = 0.f;
        for (int d=0; d<D; ++d) {
            float err = Zk[d] - t_sup[d];
            gZ[d] = 2.f * err;
            ls += err*err;
            float e = yk[d] - t_sup[d];
            E[d] = e;
            if (doReadGrad) lread += e*e;
        }
        *traj_loss_accum += ls;
        if (doReadGrad) *read_loss_accum += lread;
    }
}
__global__ void kIncNacc(unsigned int* __restrict__ Nacc)
{ if (threadIdx.x==0 && blockIdx.x==0) (*Nacc)++; }

__global__ void kLogUbuffer(const float* __restrict__ u,
                            const int* __restrict__ neurons_idx0,
                            int num_u_plot,
                            float* __restrict__ u_buf,
                            int t_index, int t_2s_max_steps)
{
    if (t_index >= t_2s_max_steps) return;
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_u_plot) return;
    int nidx = neurons_idx0[r];
    u_buf[r + num_u_plot * t_index] = u[nidx];
}
__global__ void kCopyZprev(const float* __restrict__ Z_col_k, float* __restrict__ Zprev, int D)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    Zprev[d] = Z_col_k[d];
}
__global__ void kLogSpikeAndW(const bool* __restrict__ spike_k1_bool,
                              const float* __restrict__ w_curr,
                              unsigned char* __restrict__ S_col_k1,
                              float* __restrict__ w_store_col,
                              int N_hidden)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_hidden) return;
    if (S_col_k1)      S_col_k1[i]   = spike_k1_bool[i] ? 1u : 0u;
    if (w_store_col)   w_store_col[i]= w_curr[i];
}


// ======================= gpu command MEX =======================
// Persistent fast dynamical-systems trainer.
// Commands:
//   init(... same static args as the original MEX, excluding x/lambda/log args ...)
//   init_optim(b1,b2,eps,read_lr,read_b1,read_b2,read_eps,read_weight_decay)
//   train_epoch(x, lambda_seq, lr_bias, do_read, update_bias, update_read)
//   train_read_epoch(x, lambda_seq, do_read)
//   run_diagnostic(x, lambda_seq, do_read, t_2s_max_steps, neurons_for_u)
//   get_params()
//   update_params(B, W_read)
//   clear

#include <string>
#include <algorithm>

__global__ void kFillConst(float* __restrict__ y, float v, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = v;
}

__global__ void kZeroUchar(unsigned char* __restrict__ y, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = 0u;
}

__global__ void kAdamBiasUpdateDyn(float* __restrict__ B,
                                   const float* __restrict__ g,
                                   float* __restrict__ m,
                                   float* __restrict__ v,
                                   float* __restrict__ vhat,
                                   float lr,
                                   float b1,
                                   float b2,
                                   float eps,
                                   float invN,
                                   float bc1,
                                   float bc2,
                                   int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float gi = g[i] * invN;
    float mi = b1 * m[i] + (1.f - b1) * gi;
    float vi = b2 * v[i] + (1.f - b2) * gi * gi;
    float vh = vi / fmaxf(bc2, REALMIN_SINGLE);
    vh = fmaxf(vhat[i], vh);
    float mh = mi / fmaxf(bc1, REALMIN_SINGLE);
    B[i] -= lr * mh / (sqrtf(vh) + eps);
    m[i] = mi;
    v[i] = vi;
    vhat[i] = vh;
}

__global__ void kAdamReadUpdateDyn(float* __restrict__ W,
                                   const float* __restrict__ g,
                                   float* __restrict__ m,
                                   float* __restrict__ v,
                                   float* __restrict__ vhat,
                                   float lr,
                                   float b1,
                                   float b2,
                                   float eps,
                                   float weight_decay,
                                   float invN,
                                   float bc1,
                                   float bc2,
                                   int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float gi = g[i] * invN + weight_decay * W[i];
    float mi = b1 * m[i] + (1.f - b1) * gi;
    float vi = b2 * v[i] + (1.f - b2) * gi * gi;
    float vh = vi / fmaxf(bc2, REALMIN_SINGLE);
    vh = fmaxf(vhat[i], vh);
    float mh = mi / fmaxf(bc1, REALMIN_SINGLE);
    W[i] -= lr * mh / (sqrtf(vh) + eps);
    m[i] = mi;
    v[i] = vi;
    vhat[i] = vh;
}

struct GpuDynContext {
    bool initialized = false;
    bool optim_initialized = false;
    int D = 0, N_hidden = 0, N_rec = 0, steps = 0;
    int recurrent_mode = RECURRENT_LOW_RANK, decoder_mode = DECODER_SHARED, recurrent_storage = REC_STORAGE_DENSE, rec_nnz = 0;
    float alpha = 0.f, oneMinusAlpha = 0.f, beta = 0.f, oneMinusBeta = 0.f;
    float gamma_sr = 0.f, gamma_sd = 0.f, READ_gamma_sr = 0.f, READ_gamma_sd = 0.f;
    float E_L = 0.f, V_th = 0.f, V_reset = 0.f, a_eff = 0.f, b_param = 0.f;
    float phi_u = 0.f, delta_u = 0.f, INPUT_SCALE = 0.f, SCALE_rec = 0.f;
    float spike_jump_sr = 0.f, spike_jump_sr_R = 0.f;
    float log_alpha = 0.f, log_beta = 0.f, log_gamma_sr = 0.f, log_gamma_sd = 0.f;
    float log_READ_g_sr = 0.f, log_READ_g_sd = 0.f;

    float bias_b1 = 0.9f, bias_b2 = 0.999f, bias_eps = 1e-8f;
    float read_lr = 1e-4f, read_b1 = 0.9f, read_b2 = 0.999f, read_eps = 1e-8f, read_weight_decay = 0.f;
    int t_adam = 0, t_read = 0;

    float *d_W_in=nullptr, *d_W_out_b=nullptr, *d_W_out=nullptr, *d_Eta=nullptr, *d_dself=nullptr, *d_B=nullptr, *d_W_read=nullptr, *d_W_rec=nullptr, *d_rec_w=nullptr;
    int *d_rec_post=nullptr, *d_rec_pre=nullptr;
    float *d_X=nullptr, *d_T=nullptr, *d_u=nullptr, *d_w=nullptr, *d_xsyn=nullptr, *d_r=nullptr, *d_xsynR=nullptr, *d_rR=nullptr;
    float *d_eps_v_noa=nullptr, *d_eps_a=nullptr, *d_Ebar_x=nullptr, *d_Ebar_f=nullptr;
    float *d_I_in=nullptr, *d_y_dec=nullptr, *d_Irec_scaled=nullptr, *d_I_rec=nullptr, *d_I_tot=nullptr;
    bool *d_spike=nullptr;
    float *d_rho=nullptr, *d_surr=nullptr, *d_Z=nullptr, *d_Yread=nullptr;
    unsigned char *d_S=nullptr;
    float *d_Iin_store=nullptr, *d_Irec_store=nullptr, *d_w_store=nullptr, *d_u_buffer=nullptr;
    int *d_neurons_idx0=nullptr;
    int num_u_plot_capacity = 0, u_buffer_capacity = 0;
    float *d_g_b_traj=nullptr, *d_Grad_epoch=nullptr, *d_traj_loss=nullptr, *d_read_loss=nullptr;
    unsigned int *d_Nacc=nullptr;
    float *d_gZ=nullptr, *d_E=nullptr, *d_Zprev=nullptr;
    float *d_m_b=nullptr, *d_v_b=nullptr, *d_vhat_b=nullptr, *d_M_read=nullptr, *d_V_read=nullptr, *d_Vhat_read=nullptr;
};

static GpuDynContext G;
static void freeF(float*& p){ if(p){ CUDA_CHECK(cudaFree(p)); p=nullptr; } }
static void freeB(bool*& p){ if(p){ CUDA_CHECK(cudaFree(p)); p=nullptr; } }
static void freeU(unsigned char*& p){ if(p){ CUDA_CHECK(cudaFree(p)); p=nullptr; } }
static void freeI(int*& p){ if(p){ CUDA_CHECK(cudaFree(p)); p=nullptr; } }

static void clearContext()
{
    freeF(G.d_W_in); freeF(G.d_W_out_b); freeF(G.d_W_out); freeF(G.d_Eta); freeF(G.d_dself); freeF(G.d_B); freeF(G.d_W_read); freeF(G.d_W_rec); freeF(G.d_rec_w); freeI(G.d_rec_post); freeI(G.d_rec_pre);
    freeF(G.d_X); freeF(G.d_T); freeF(G.d_u); freeF(G.d_w); freeF(G.d_xsyn); freeF(G.d_r); freeF(G.d_xsynR); freeF(G.d_rR);
    freeF(G.d_eps_v_noa); freeF(G.d_eps_a); freeF(G.d_Ebar_x); freeF(G.d_Ebar_f);
    freeF(G.d_I_in); freeF(G.d_y_dec); freeF(G.d_Irec_scaled); freeF(G.d_I_rec); freeF(G.d_I_tot);
    freeB(G.d_spike); freeF(G.d_rho); freeF(G.d_surr); freeF(G.d_Z); freeF(G.d_Yread); freeU(G.d_S);
    freeF(G.d_Iin_store); freeF(G.d_Irec_store); freeF(G.d_w_store); freeF(G.d_u_buffer); freeI(G.d_neurons_idx0);
    freeF(G.d_g_b_traj); freeF(G.d_Grad_epoch); freeF(G.d_traj_loss); freeF(G.d_read_loss); if(G.d_Nacc){ CUDA_CHECK(cudaFree(G.d_Nacc)); G.d_Nacc=nullptr; }
    freeF(G.d_gZ); freeF(G.d_E); freeF(G.d_Zprev);
    freeF(G.d_m_b); freeF(G.d_v_b); freeF(G.d_vhat_b); freeF(G.d_M_read); freeF(G.d_V_read); freeF(G.d_Vhat_read);
    G = GpuDynContext();
}

static inline void requireInitialized()
{
    if (!G.initialized) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:state", "Call init before this command.");
}
static inline float getScalarSingleR(const mxArray* a, const char* nm)
{
    assertSingle(a, nm);
    if (mxGetNumberOfElements(a) != 1) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:scalar", "%s must be scalar", nm);
    return *(float*)mxGetData(a);
}
static inline int getIntScalarR(const mxArray* a, const char* nm)
{
    if (mxGetNumberOfElements(a) != 1) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:scalar", "%s must be scalar", nm);
    return (int)mxGetScalar(a);
}
static inline bool getBoolR(const mxArray* a)
{
    if (mxIsLogical(a) && mxGetNumberOfElements(a) == 1) return mxIsLogicalScalarTrue(a);
    if (mxGetNumberOfElements(a) != 1) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:scalar", "logical scalar expected");
    return mxGetScalar(a) != 0.0;
}
static std::string getStringR(const mxArray* a)
{
    if (!mxIsChar(a)) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:type", "Expected a command string");
    char* c = mxArrayToString(a);
    if (!c) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:type", "Could not parse string");
    std::string s(c); mxFree(c); return s;
}

static std::vector<bool> readLambda(const mxArray* lambda_m)
{
    assertLogicalOrNumeric(lambda_m, "lambda_seq");
    if ((int)mxGetNumberOfElements(lambda_m) != G.steps) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "lambda_seq length mismatch");
    std::vector<bool> lambda(G.steps, true);
    if (mxIsLogical(lambda_m)) {
        mxLogical* p = (mxLogical*)mxGetData(lambda_m);
        for (int k=0;k<G.steps;++k) lambda[k] = (p[k]!=0);
    } else if (mxGetClassID(lambda_m)==mxSINGLE_CLASS) {
        float* p = (float*)mxGetData(lambda_m);
        for (int k=0;k<G.steps;++k) lambda[k] = (p[k]!=0.f);
    } else if (mxGetClassID(lambda_m)==mxDOUBLE_CLASS) {
        double* p = (double*)mxGetData(lambda_m);
        for (int k=0;k<G.steps;++k) lambda[k] = (p[k]!=0.0);
    } else if (mxGetClassID(lambda_m)==mxUINT8_CLASS) {
        unsigned char* p = (unsigned char*)mxGetData(lambda_m);
        for (int k=0;k<G.steps;++k) lambda[k] = (p[k]!=0);
    }
    return lambda;
}

static void ensureLogBuffers(int num_u_plot, int t_2s_max_steps)
{
    int N = G.N_hidden;
    if (!G.d_S) CUDA_CHECK(cudaMalloc(&G.d_S, sizeof(unsigned char)*N*G.steps));
    if (!G.d_Iin_store) CUDA_CHECK(cudaMalloc(&G.d_Iin_store, sizeof(float)*N*(G.steps-1)));
    if (!G.d_Irec_store) CUDA_CHECK(cudaMalloc(&G.d_Irec_store, sizeof(float)*N*(G.steps-1)));
    if (!G.d_w_store) CUDA_CHECK(cudaMalloc(&G.d_w_store, sizeof(float)*N*(G.steps-1)));
    if (num_u_plot > G.num_u_plot_capacity) {
        freeI(G.d_neurons_idx0);
        CUDA_CHECK(cudaMalloc(&G.d_neurons_idx0, sizeof(int)*num_u_plot));
        G.num_u_plot_capacity = num_u_plot;
    }
    int need = num_u_plot * t_2s_max_steps;
    if (need > G.u_buffer_capacity) {
        freeF(G.d_u_buffer);
        CUDA_CHECK(cudaMalloc(&G.d_u_buffer, sizeof(float)*need));
        G.u_buffer_capacity = need;
    }
}

static void resetRunState(bool log_first, int num_u_plot, int t_2s_max_steps)
{
    dim3 gridN((G.N_hidden+255)/256), block(256);
    kFillConst<<<gridN,block>>>(G.d_u, G.E_L, G.N_hidden);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(G.d_w,0,sizeof(float)*G.N_hidden));
    CUDA_CHECK(cudaMemset(G.d_xsyn,0,sizeof(float)*G.N_hidden));
    CUDA_CHECK(cudaMemset(G.d_r,0,sizeof(float)*G.N_hidden));
    CUDA_CHECK(cudaMemset(G.d_xsynR,0,sizeof(float)*G.N_hidden));
    CUDA_CHECK(cudaMemset(G.d_rR,0,sizeof(float)*G.N_hidden));
    CUDA_CHECK(cudaMemset(G.d_eps_v_noa,0,sizeof(float)*G.N_hidden));
    CUDA_CHECK(cudaMemset(G.d_eps_a,0,sizeof(float)*G.N_hidden));
    CUDA_CHECK(cudaMemset(G.d_Ebar_x,0,sizeof(float)*G.N_hidden));
    CUDA_CHECK(cudaMemset(G.d_Ebar_f,0,sizeof(float)*G.N_hidden));
    CUDA_CHECK(cudaMemset(G.d_Z,0,sizeof(float)*G.D*G.steps));
    CUDA_CHECK(cudaMemset(G.d_Yread,0,sizeof(float)*G.D*(G.steps-1)));
    if (G.d_S) {
        kZeroUchar<<<(G.N_hidden*G.steps+255)/256,256>>>(G.d_S, G.N_hidden*G.steps);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemset(G.d_g_b_traj,0,sizeof(float)*G.N_hidden));
    CUDA_CHECK(cudaMemset(G.d_Grad_epoch,0,sizeof(float)*G.D*G.N_hidden));
    CUDA_CHECK(cudaMemset(G.d_traj_loss,0,sizeof(float)));
    CUDA_CHECK(cudaMemset(G.d_read_loss,0,sizeof(float)));
    CUDA_CHECK(cudaMemset(G.d_Nacc,0,sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(G.d_Zprev,0,sizeof(float)*G.D));
    if (log_first) {
        CUDA_CHECK(cudaMemset(G.d_Iin_store,0,sizeof(float)*G.N_hidden*(G.steps-1)));
        CUDA_CHECK(cudaMemset(G.d_Irec_store,0,sizeof(float)*G.N_hidden*(G.steps-1)));
        CUDA_CHECK(cudaMemset(G.d_w_store,0,sizeof(float)*G.N_hidden*(G.steps-1)));
        if (num_u_plot > 0 && t_2s_max_steps > 0) {
            std::vector<float> nanbuf(num_u_plot*t_2s_max_steps, NAN);
            CUDA_CHECK(cudaMemcpy(G.d_u_buffer, nanbuf.data(), sizeof(float)*nanbuf.size(), cudaMemcpyHostToDevice));
        }
    }
}

static void runOne(const std::vector<bool>& lambda_host, bool doReadGrad, bool computeBiasGrad,
                   bool log_first, int num_u_plot, int t_2s_max_steps)
{
    resetRunState(log_first, num_u_plot, t_2s_max_steps);
    dim3 tpbN(256), gridN((G.N_hidden+255)/256);
    dim3 gridR(G.N_rec), tpbR(256); size_t shR = tpbR.x * sizeof(float);
    dim3 gridD(G.D), tpbD(256); size_t shD = tpbD.x * sizeof(float);

    for (int k=0; k<G.steps-1; ++k) {
        const bool useTeacher = (k==0) ? true : lambda_host[k];
        const float* x_ptr = useTeacher ? (G.d_X + idx2(0, G.D, k)) : G.d_Zprev;
        kDotCol<<<gridN,tpbN>>>(G.d_W_in, x_ptr, G.d_I_in, G.N_hidden, G.D, G.INPUT_SCALE);
        CUDA_CHECK(cudaGetLastError());
        float* Iin_col = log_first ? (G.d_Iin_store + k*G.N_hidden) : nullptr;
        float* Irec_col = log_first ? (G.d_Irec_store + k*G.N_hidden) : nullptr;
        if (G.recurrent_mode == RECURRENT_LOW_RANK) {
            kYDecReduce<<<gridR,tpbR,shR>>>(G.d_W_out_b, G.d_r, G.d_y_dec, G.N_rec, G.N_hidden);
            CUDA_CHECK(cudaGetLastError());
            kDotCol<<<gridN,tpbN>>>(G.d_Eta, G.d_y_dec, G.d_Irec_scaled, G.N_hidden, G.N_rec, G.SCALE_rec);
            CUDA_CHECK(cudaGetLastError());
            kBuildItot<<<gridN,tpbN>>>(G.d_I_in, G.d_Irec_scaled, G.d_r, G.d_dself, G.d_B, G.d_I_rec, G.d_I_tot, Iin_col, Irec_col, G.N_hidden);
            CUDA_CHECK(cudaGetLastError());
        } else if (G.recurrent_mode == RECURRENT_FULL_RANK) {
            if (G.recurrent_storage == REC_STORAGE_DENSE) {
                kDotCol<<<gridN,tpbN>>>(G.d_W_rec, G.d_r, G.d_Irec_scaled, G.N_hidden, G.N_hidden, 1.0f);
                CUDA_CHECK(cudaGetLastError());
            } else if (G.recurrent_storage == REC_STORAGE_SPARSE) {
                CUDA_CHECK(cudaMemset(G.d_Irec_scaled,0,sizeof(float)*G.N_hidden));
                if (G.rec_nnz > 0) {
                    kSparseRec<<<(G.rec_nnz+255)/256,256>>>(G.d_rec_post,G.d_rec_pre,G.d_rec_w,G.rec_nnz,G.d_r,G.d_Irec_scaled);
                    CUDA_CHECK(cudaGetLastError());
                }
            } else {
                mexErrMsgIdAndTxt("snn_dyn_gpu_mex:arch", "Unknown recurrent_storage_id %d.", G.recurrent_storage);
            }
            kBuildItotFullRank<<<gridN,tpbN>>>(G.d_I_in, G.d_Irec_scaled, G.d_B, G.d_I_rec, G.d_I_tot, Iin_col, Irec_col, G.N_hidden);
            CUDA_CHECK(cudaGetLastError());
        } else {
            mexErrMsgIdAndTxt("snn_dyn_gpu_mex:arch", "Unknown recurrent_mode_id %d.", G.recurrent_mode);
        }
        kAdvanceUW<<<gridN,tpbN>>>(G.d_u, G.d_w, G.d_I_tot,
            G.alpha, G.oneMinusAlpha, G.beta, G.oneMinusBeta, G.log_alpha, G.log_beta,
            G.V_th, G.V_reset, G.E_L, G.a_eff, G.b_param, G.phi_u, G.delta_u,
            G.d_u, G.d_w, G.d_rho, G.d_spike, G.d_surr, G.N_hidden);
        CUDA_CHECK(cudaGetLastError());
        kCascadeAdvancePre<<<gridN,tpbN>>>(G.d_rho, G.d_xsyn, G.d_r, G.log_gamma_sr, G.log_gamma_sd, G.N_hidden);
        kCascadeAdvancePre<<<gridN,tpbN>>>(G.d_rho, G.d_xsynR, G.d_rR, G.log_READ_g_sr, G.log_READ_g_sd, G.N_hidden);
        CUDA_CHECK(cudaGetLastError());
        kAddSpikeJumps<<<gridN,tpbN>>>(G.d_spike, G.d_xsyn, G.spike_jump_sr, G.N_hidden);
        kAddSpikeJumps<<<gridN,tpbN>>>(G.d_spike, G.d_xsynR, G.spike_jump_sr_R, G.N_hidden);
        CUDA_CHECK(cudaGetLastError());
        kCascadeAdvancePost<<<gridN,tpbN>>>(G.d_rho, G.d_xsyn, G.d_r, G.log_gamma_sr, G.log_gamma_sd, G.N_hidden);
        kCascadeAdvancePost<<<gridN,tpbN>>>(G.d_rho, G.d_xsynR, G.d_rR, G.log_READ_g_sr, G.log_READ_g_sd, G.N_hidden);
        CUDA_CHECK(cudaGetLastError());
        float* Z_col = G.d_Z + k*G.D;
        float* Y_col = G.d_Yread + k*G.D;
        kOutputsColReduce<<<gridD,tpbD,shD>>>(G.d_W_out, G.d_W_read, G.d_r, G.d_rR, Z_col, Y_col, G.D, G.N_hidden);
        CUDA_CHECK(cudaGetLastError());
        const float* t_sup_col = G.d_X + idx2(0, G.D, k+1);
        kComputeErrors<<<1,1>>>(Z_col, Y_col, t_sup_col, G.D, G.d_gZ, G.d_E, G.d_traj_loss, G.d_read_loss, doReadGrad);
        CUDA_CHECK(cudaGetLastError());
        if (doReadGrad) {
            kIncNacc<<<1,1>>>(G.d_Nacc);
            dim3 tB(16,16), bG((G.N_hidden+tB.x-1)/tB.x, (G.D+tB.y-1)/tB.y);
            kReadGradAccumulate<<<bG,tB>>>(G.d_E, G.d_rR, G.d_Grad_epoch, G.D, G.N_hidden);
            CUDA_CHECK(cudaGetLastError());
        }
        if (computeBiasGrad) {
            kBiasEpropAndGrad_COL<<<gridN,tpbN>>>(G.d_W_out, G.d_gZ,
                G.d_eps_v_noa, G.d_eps_a, G.d_Ebar_x, G.d_Ebar_f,
                G.d_spike, G.d_surr, G.spike_jump_sr,
                G.log_gamma_sr, G.log_gamma_sd,
                G.d_rho, G.a_eff, G.b_param,
                G.log_alpha, G.log_beta,
                G.d_g_b_traj, G.D, G.N_hidden);
            CUDA_CHECK(cudaGetLastError());
        }
        unsigned char* S_col = G.d_S ? (G.d_S + (k+1)*G.N_hidden) : nullptr;
        float* w_col = log_first ? (G.d_w_store + k*G.N_hidden) : nullptr;
        kLogSpikeAndW<<<gridN,tpbN>>>(G.d_spike, G.d_w, S_col, w_col, G.N_hidden);
        CUDA_CHECK(cudaGetLastError());
        if (log_first && num_u_plot>0 && k<t_2s_max_steps) {
            kLogUbuffer<<<(num_u_plot+127)/128,128>>>(G.d_u, G.d_neurons_idx0, num_u_plot, G.d_u_buffer, k, t_2s_max_steps);
            CUDA_CHECK(cudaGetLastError());
        }
        kCopyZprev<<<(G.D+255)/256,256>>>(Z_col, G.d_Zprev, G.D);
        CUDA_CHECK(cudaGetLastError());
    }
}


static void applyOptim(bool updateBias, bool updateRead, float lrBias, int biasAverageCount = -1)
{
    if (!G.optim_initialized) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:optim", "Call init_optim before train_epoch.");
    if (updateBias) {
        G.t_adam += 1;
        float bc1 = 1.f - powf(G.bias_b1, (float)G.t_adam);
        float bc2 = 1.f - powf(G.bias_b2, (float)G.t_adam);
        int count = (biasAverageCount > 0) ? biasAverageCount : (G.steps-1);
        float invN = 1.f / fmaxf((float)count, 1.f);
        kAdamBiasUpdateDyn<<<(G.N_hidden+255)/256,256>>>(G.d_B, G.d_g_b_traj, G.d_m_b, G.d_v_b, G.d_vhat_b,
            lrBias, G.bias_b1, G.bias_b2, G.bias_eps, invN, bc1, bc2, G.N_hidden);
        CUDA_CHECK(cudaGetLastError());
    }
    if (updateRead) {
        unsigned int nacc = 0;
        CUDA_CHECK(cudaMemcpy(&nacc, G.d_Nacc, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (nacc > 0) {
            G.t_read += 1;
            float bc1 = 1.f - powf(G.read_b1, (float)G.t_read);
            float bc2 = 1.f - powf(G.read_b2, (float)G.t_read);
            float invN = 1.f / fmaxf((float)nacc, 1.f);
            int CN = G.D * G.N_hidden;
            kAdamReadUpdateDyn<<<(CN+255)/256,256>>>(G.d_W_read, G.d_Grad_epoch, G.d_M_read, G.d_V_read, G.d_Vhat_read,
                G.read_lr, G.read_b1, G.read_b2, G.read_eps, G.read_weight_decay, invN, bc1, bc2, CN);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

static void cmdInit(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)nlhs; (void)plhs;
    if (nrhs != 39) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "init expects 38 arguments after command string");
    clearContext();
    const mxArray *W_in_m=prhs[1], *W_ob_m=prhs[2], *W_out_m=prhs[3], *Eta_m=prhs[4], *dself_m=prhs[5], *B_m=prhs[6], *W_read_m=prhs[7];
    assertSingle(W_in_m,"W_in"); assertSingle(W_ob_m,"W_out_base_rec"); assertSingle(W_out_m,"W_out"); assertSingle(Eta_m,"Eta_rec");
    assertSingle(dself_m,"dself"); assertSingle(B_m,"B"); assertSingle(W_read_m,"W_read");
    G.alpha=getScalarSingleR(prhs[8],"alpha"); G.oneMinusAlpha=getScalarSingleR(prhs[9],"oneMinusAlpha");
    G.beta=getScalarSingleR(prhs[10],"beta"); G.oneMinusBeta=getScalarSingleR(prhs[11],"oneMinusBeta");
    G.gamma_sr=getScalarSingleR(prhs[12],"gamma_sr"); G.gamma_sd=getScalarSingleR(prhs[13],"gamma_sd");
    G.READ_gamma_sr=getScalarSingleR(prhs[14],"READ_gamma_sr"); G.READ_gamma_sd=getScalarSingleR(prhs[15],"READ_gamma_sd");
    G.E_L=getScalarSingleR(prhs[16],"E_L"); G.V_th=getScalarSingleR(prhs[17],"V_th"); G.V_reset=getScalarSingleR(prhs[18],"V_reset");
    G.a_eff=getScalarSingleR(prhs[19],"a_eff"); G.b_param=getScalarSingleR(prhs[20],"b_param");
    G.phi_u=getScalarSingleR(prhs[21],"phi_u"); G.delta_u=getScalarSingleR(prhs[22],"delta_u");
    G.INPUT_SCALE=getScalarSingleR(prhs[23],"INPUT_SCALE"); G.SCALE_rec=getScalarSingleR(prhs[24],"SCALE_rec");
    G.spike_jump_sr=getScalarSingleR(prhs[25],"spike_jump_sr"); G.spike_jump_sr_R=getScalarSingleR(prhs[26],"spike_jump_sr_R");
    G.steps=getIntScalarR(prhs[27],"steps"); G.D=getIntScalarR(prhs[28],"D"); G.N_hidden=getIntScalarR(prhs[29],"N_hidden"); G.N_rec=getIntScalarR(prhs[30],"N_rec");
    const mxArray *W_rec_m=prhs[31];
    G.recurrent_mode=getIntScalarR(prhs[32],"recurrent_mode_id");
    G.decoder_mode=getIntScalarR(prhs[33],"decoder_mode_id");
    G.recurrent_storage=getIntScalarR(prhs[34],"recurrent_storage_id");
    const mxArray *rec_post_m=prhs[35], *rec_pre_m=prhs[36], *rec_w_m=prhs[37];
    G.rec_nnz=getIntScalarR(prhs[38],"rec_nnz");
    if (G.recurrent_mode != RECURRENT_LOW_RANK && G.recurrent_mode != RECURRENT_FULL_RANK)
        mexErrMsgIdAndTxt("snn_dyn_gpu_mex:arch","recurrent_mode_id must be 0 (low_rank) or 1 (full_rank)");
    if (G.decoder_mode != DECODER_SHARED && G.decoder_mode != DECODER_SIGNED)
        mexErrMsgIdAndTxt("snn_dyn_gpu_mex:arch","decoder_mode_id must be 0 (shared) or 1 (signed)");
    if (G.recurrent_storage != REC_STORAGE_DENSE && G.recurrent_storage != REC_STORAGE_SPARSE)
        mexErrMsgIdAndTxt("snn_dyn_gpu_mex:arch","recurrent_storage_id must be 0 (dense) or 1 (sparse)");
    assertSingle(W_rec_m,"W_rec"); assertSingle(rec_w_m,"rec_w");
    if (mxGetClassID(rec_post_m) != mxINT32_CLASS || mxGetClassID(rec_pre_m) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("snn_dyn_gpu_mex:type","rec_post_idx and rec_pre_idx must be int32");
    if ((int)mxGetM(W_in_m)!=G.N_hidden || (int)mxGetN(W_in_m)!=G.D) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "W_in shape mismatch");
    if ((int)mxGetM(W_ob_m)!=G.N_rec || (int)mxGetN(W_ob_m)!=G.N_hidden) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "W_out_base_rec shape mismatch");
    if ((int)mxGetM(W_out_m)!=G.D || (int)mxGetN(W_out_m)!=G.N_hidden) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "W_out shape mismatch");
    if ((int)mxGetM(Eta_m)!=G.N_hidden || (int)mxGetN(Eta_m)!=G.N_rec) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "Eta_rec shape mismatch");
    if ((int)mxGetNumberOfElements(dself_m)!=G.N_hidden || (int)mxGetNumberOfElements(B_m)!=G.N_hidden) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "bias vector shape mismatch");
    if ((int)mxGetM(W_read_m)!=G.D || (int)mxGetN(W_read_m)!=G.N_hidden) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "W_read shape mismatch");
    if (G.rec_nnz < 0 || (int)mxGetNumberOfElements(rec_post_m) != G.rec_nnz || (int)mxGetNumberOfElements(rec_pre_m) != G.rec_nnz || (int)mxGetNumberOfElements(rec_w_m) != G.rec_nnz)
        mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "sparse recurrent edge-list length mismatch");
    if (G.recurrent_mode == RECURRENT_FULL_RANK && G.recurrent_storage == REC_STORAGE_DENSE && ((int)mxGetM(W_rec_m)!=G.N_hidden || (int)mxGetN(W_rec_m)!=G.N_hidden))
        mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "W_rec shape mismatch for full_rank recurrence");

    int N=G.N_hidden, D=G.D, R=G.N_rec;
    CUDA_CHECK(cudaMalloc(&G.d_W_in,sizeof(float)*N*D)); CUDA_CHECK(cudaMalloc(&G.d_W_out_b,sizeof(float)*R*N)); CUDA_CHECK(cudaMalloc(&G.d_W_out,sizeof(float)*D*N));
    CUDA_CHECK(cudaMalloc(&G.d_Eta,sizeof(float)*N*R)); CUDA_CHECK(cudaMalloc(&G.d_dself,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_B,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_W_read,sizeof(float)*D*N));
    if (G.recurrent_mode == RECURRENT_FULL_RANK && G.recurrent_storage == REC_STORAGE_DENSE) CUDA_CHECK(cudaMalloc(&G.d_W_rec,sizeof(float)*N*N));
    if (G.recurrent_mode == RECURRENT_FULL_RANK && G.recurrent_storage == REC_STORAGE_SPARSE && G.rec_nnz > 0) {
        CUDA_CHECK(cudaMalloc(&G.d_rec_post,sizeof(int)*G.rec_nnz));
        CUDA_CHECK(cudaMalloc(&G.d_rec_pre,sizeof(int)*G.rec_nnz));
        CUDA_CHECK(cudaMalloc(&G.d_rec_w,sizeof(float)*G.rec_nnz));
    }
    CUDA_CHECK(cudaMemcpy(G.d_W_in,mxGetData(W_in_m),sizeof(float)*N*D,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(G.d_W_out_b,mxGetData(W_ob_m),sizeof(float)*R*N,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.d_W_out,mxGetData(W_out_m),sizeof(float)*D*N,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(G.d_Eta,mxGetData(Eta_m),sizeof(float)*N*R,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.d_dself,mxGetData(dself_m),sizeof(float)*N,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(G.d_B,mxGetData(B_m),sizeof(float)*N,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(G.d_W_read,mxGetData(W_read_m),sizeof(float)*D*N,cudaMemcpyHostToDevice));
    if (G.recurrent_mode == RECURRENT_FULL_RANK && G.recurrent_storage == REC_STORAGE_DENSE) CUDA_CHECK(cudaMemcpy(G.d_W_rec,mxGetData(W_rec_m),sizeof(float)*N*N,cudaMemcpyHostToDevice));
    if (G.recurrent_mode == RECURRENT_FULL_RANK && G.recurrent_storage == REC_STORAGE_SPARSE && G.rec_nnz > 0) {
        CUDA_CHECK(cudaMemcpy(G.d_rec_post,mxGetData(rec_post_m),sizeof(int)*G.rec_nnz,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(G.d_rec_pre,mxGetData(rec_pre_m),sizeof(int)*G.rec_nnz,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(G.d_rec_w,mxGetData(rec_w_m),sizeof(float)*G.rec_nnz,cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&G.d_X,sizeof(float)*D*G.steps)); CUDA_CHECK(cudaMalloc(&G.d_T,sizeof(float)*D*G.steps)); CUDA_CHECK(cudaMalloc(&G.d_u,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_w,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_xsyn,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_r,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&G.d_xsynR,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_rR,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_eps_v_noa,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_eps_a,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_Ebar_x,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_Ebar_f,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&G.d_I_in,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_y_dec,sizeof(float)*R)); CUDA_CHECK(cudaMalloc(&G.d_Irec_scaled,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_I_rec,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_I_tot,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&G.d_spike,sizeof(bool)*N)); CUDA_CHECK(cudaMalloc(&G.d_rho,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_surr,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_Z,sizeof(float)*D*G.steps)); CUDA_CHECK(cudaMalloc(&G.d_Yread,sizeof(float)*D*(G.steps-1)));
    CUDA_CHECK(cudaMalloc(&G.d_g_b_traj,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_Grad_epoch,sizeof(float)*D*N)); CUDA_CHECK(cudaMalloc(&G.d_traj_loss,sizeof(float))); CUDA_CHECK(cudaMalloc(&G.d_read_loss,sizeof(float))); CUDA_CHECK(cudaMalloc(&G.d_Nacc,sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&G.d_gZ,sizeof(float)*D)); CUDA_CHECK(cudaMalloc(&G.d_E,sizeof(float)*D)); CUDA_CHECK(cudaMalloc(&G.d_Zprev,sizeof(float)*D));
    G.log_alpha=logf(fmaxf(G.alpha,REALMIN_SINGLE)); G.log_beta=logf(fmaxf(G.beta,REALMIN_SINGLE)); G.log_gamma_sr=logf(fmaxf(G.gamma_sr,REALMIN_SINGLE)); G.log_gamma_sd=logf(fmaxf(G.gamma_sd,REALMIN_SINGLE)); G.log_READ_g_sr=logf(fmaxf(G.READ_gamma_sr,REALMIN_SINGLE)); G.log_READ_g_sd=logf(fmaxf(G.READ_gamma_sd,REALMIN_SINGLE));
    G.initialized=true; mexAtExit(clearContext);
}

static void cmdInitOptim(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)nlhs; (void)plhs;
    if (nrhs != 9) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "init_optim expects b1,b2,eps,read_lr,read_b1,read_b2,read_eps,read_weight_decay");
    requireInitialized();
    G.bias_b1=getScalarSingleR(prhs[1],"bias_b1"); G.bias_b2=getScalarSingleR(prhs[2],"bias_b2"); G.bias_eps=getScalarSingleR(prhs[3],"bias_eps");
    G.read_lr=getScalarSingleR(prhs[4],"read_lr"); G.read_b1=getScalarSingleR(prhs[5],"read_b1"); G.read_b2=getScalarSingleR(prhs[6],"read_b2"); G.read_eps=getScalarSingleR(prhs[7],"read_eps"); G.read_weight_decay=getScalarSingleR(prhs[8],"read_weight_decay");
    int N=G.N_hidden, CN=G.D*G.N_hidden;
    CUDA_CHECK(cudaMalloc(&G.d_m_b,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_v_b,sizeof(float)*N)); CUDA_CHECK(cudaMalloc(&G.d_vhat_b,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&G.d_M_read,sizeof(float)*CN)); CUDA_CHECK(cudaMalloc(&G.d_V_read,sizeof(float)*CN)); CUDA_CHECK(cudaMalloc(&G.d_Vhat_read,sizeof(float)*CN));
    CUDA_CHECK(cudaMemset(G.d_m_b,0,sizeof(float)*N)); CUDA_CHECK(cudaMemset(G.d_v_b,0,sizeof(float)*N)); CUDA_CHECK(cudaMemset(G.d_vhat_b,0,sizeof(float)*N));
    CUDA_CHECK(cudaMemset(G.d_M_read,0,sizeof(float)*CN)); CUDA_CHECK(cudaMemset(G.d_V_read,0,sizeof(float)*CN)); CUDA_CHECK(cudaMemset(G.d_Vhat_read,0,sizeof(float)*CN));
    G.t_adam=0; G.t_read=0; G.optim_initialized=true;
}

static void trainCommon(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[], bool readOnlyCommand)
{
    requireInitialized();
    int expected = readOnlyCommand ? 4 : 7;
    if (nrhs != expected || nlhs > 3) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", readOnlyCommand ? "train_read_epoch returns up to [lossN,lossR,count]" : "train_epoch returns up to [lossN,lossR,count]");
    const mxArray* x_m = prhs[1]; const mxArray* lambda_m = prhs[2];
    assertSingle(x_m,"x"); if ((int)mxGetM(x_m)!=G.D || (int)mxGetN(x_m)!=G.steps) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "x must be [D x steps]");
    std::vector<bool> lambda = readLambda(lambda_m);
    CUDA_CHECK(cudaMemcpy(G.d_X, mxGetData(x_m), sizeof(float)*G.D*G.steps, cudaMemcpyHostToDevice));
    float lrBias = readOnlyCommand ? 0.f : getScalarSingleR(prhs[3],"lr_bias");
    bool doRead = readOnlyCommand ? getBoolR(prhs[3]) : getBoolR(prhs[4]);
    bool updateBias = readOnlyCommand ? false : getBoolR(prhs[5]);
    bool updateRead = readOnlyCommand ? true : getBoolR(prhs[6]);
    runOne(lambda, doRead, updateBias, false, 0, 0);
    if (updateBias || updateRead) applyOptim(updateBias, updateRead, lrBias);
    float hLoss[2]={0,0}; unsigned int nacc=0;
    CUDA_CHECK(cudaMemcpy(&hLoss[0], G.d_traj_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hLoss[1], G.d_read_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&nacc, G.d_Nacc, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (nlhs>0) plhs[0]=mxCreateDoubleScalar((double)hLoss[0]);
    if (nlhs>1) plhs[1]=mxCreateDoubleScalar((double)hLoss[1]);
    if (nlhs>2) plhs[2]=mxCreateDoubleScalar((double)((nacc>0)?nacc:(G.steps-1)));
}

static void cmdTrainPrimaryEpoch(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    requireInitialized();
    if (nrhs != 4 || nlhs > 2) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "train_primary_epoch returns up to [loss,count]");
    const mxArray* x_m = prhs[1];
    const mxArray* lambda_m = prhs[2];
    assertSingle(x_m, "x");
    if ((int)mxGetM(x_m) != G.D || (int)mxGetN(x_m) != G.steps)
        mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "x must be [D x steps]");
    std::vector<bool> lambda = readLambda(lambda_m);
    CUDA_CHECK(cudaMemcpy(G.d_X, mxGetData(x_m), sizeof(float)*G.D*G.steps, cudaMemcpyHostToDevice));
    float lrBias = getScalarSingleR(prhs[3], "lr_bias");
    runOne(lambda, false, true, false, 0, 0);
    applyOptim(true, false, lrBias);
    float hLoss = 0.f;
    CUDA_CHECK(cudaMemcpy(&hLoss, G.d_traj_loss, sizeof(float), cudaMemcpyDeviceToHost));
    if (nlhs > 0) plhs[0]=mxCreateDoubleScalar((double)hLoss);
    if (nlhs > 1) plhs[1]=mxCreateDoubleScalar((double)(G.steps-1));
}


static void cmdRunPrimaryEval(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    requireInitialized();
    if (nrhs != 3 || nlhs > 3) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "run_primary_eval returns [Z,loss,count]");
    const mxArray* x_m = prhs[1];
    const mxArray* lambda_m = prhs[2];
    assertSingle(x_m, "x");
    if ((int)mxGetM(x_m) != G.D || (int)mxGetN(x_m) != G.steps)
        mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "x must be [D x steps]");
    std::vector<bool> lambda = readLambda(lambda_m);
    CUDA_CHECK(cudaMemcpy(G.d_X, mxGetData(x_m), sizeof(float)*G.D*G.steps, cudaMemcpyHostToDevice));
    runOne(lambda, false, false, false, 0, 0);
    float hLoss = 0.f;
    CUDA_CHECK(cudaMemcpy(&hLoss, G.d_traj_loss, sizeof(float), cudaMemcpyDeviceToHost));
    if (nlhs > 0) {
        plhs[0]=mxCreateNumericMatrix(G.D,G.steps-1,mxSINGLE_CLASS,mxREAL);
        CUDA_CHECK(cudaMemcpy(mxGetData(plhs[0]),G.d_Z,sizeof(float)*G.D*(G.steps-1),cudaMemcpyDeviceToHost));
    }
    if (nlhs > 1) plhs[1]=mxCreateDoubleScalar((double)hLoss);
    if (nlhs > 2) plhs[2]=mxCreateDoubleScalar((double)(G.steps-1));
}

static void cmdRunPrimaryDiagnostic(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    requireInitialized();
    if (nrhs != 5 || nlhs > 9) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "run_primary_diagnostic returns up to [Z,S,Iin,Irec,w,u,gB,loss,count]");
    const mxArray* x_m = prhs[1];
    const mxArray* lambda_m = prhs[2];
    int t_2s_max_steps = getIntScalarR(prhs[3], "t_2s_max_steps");
    const mxArray* neurons_for_u_m = prhs[4];
    assertSingle(x_m, "x");
    if ((int)mxGetM(x_m) != G.D || (int)mxGetN(x_m) != G.steps)
        mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "x must be [D x steps]");
    if (!isVectorMx(neurons_for_u_m))
        mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "neurons_for_u must be a vector");
    int num_u_plot = (int)mxGetNumberOfElements(neurons_for_u_m);
    std::vector<int> neurons(num_u_plot,0);
    mxClassID cls = mxGetClassID(neurons_for_u_m);
    for (int i=0; i<num_u_plot; ++i) {
        int v=0;
        if (cls == mxINT32_CLASS) v=((int*)mxGetData(neurons_for_u_m))[i]-1;
        else if (cls == mxUINT32_CLASS) v=(int)((unsigned int*)mxGetData(neurons_for_u_m))[i]-1;
        else v=(int)((double*)mxGetData(neurons_for_u_m))[i]-1;
        if (v < 0 || v >= G.N_hidden) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:range", "neurons_for_u out of range");
        neurons[i]=v;
    }
    ensureLogBuffers(num_u_plot, t_2s_max_steps);
    if (num_u_plot > 0) CUDA_CHECK(cudaMemcpy(G.d_neurons_idx0, neurons.data(), sizeof(int)*num_u_plot, cudaMemcpyHostToDevice));
    std::vector<bool> lambda = readLambda(lambda_m);
    CUDA_CHECK(cudaMemcpy(G.d_X, mxGetData(x_m), sizeof(float)*G.D*G.steps, cudaMemcpyHostToDevice));
    runOne(lambda, false, true, true, num_u_plot, t_2s_max_steps);
    if (nlhs>0){ plhs[0]=mxCreateNumericMatrix(G.D,G.steps,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[0]),G.d_Z,sizeof(float)*G.D*G.steps,cudaMemcpyDeviceToHost)); }
    if (nlhs>1){ plhs[1]=mxCreateLogicalMatrix(G.N_hidden,G.steps); mxLogical* Slog=(mxLogical*)mxGetData(plhs[1]); std::vector<unsigned char> tmp(G.N_hidden*G.steps); CUDA_CHECK(cudaMemcpy(tmp.data(),G.d_S,sizeof(unsigned char)*tmp.size(),cudaMemcpyDeviceToHost)); for(size_t i=0;i<tmp.size();++i) Slog[i]=tmp[i]?1:0; }
    if (nlhs>2){ plhs[2]=mxCreateNumericMatrix(G.N_hidden,G.steps-1,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[2]),G.d_Iin_store,sizeof(float)*G.N_hidden*(G.steps-1),cudaMemcpyDeviceToHost)); }
    if (nlhs>3){ plhs[3]=mxCreateNumericMatrix(G.N_hidden,G.steps-1,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[3]),G.d_Irec_store,sizeof(float)*G.N_hidden*(G.steps-1),cudaMemcpyDeviceToHost)); }
    if (nlhs>4){ plhs[4]=mxCreateNumericMatrix(G.N_hidden,G.steps-1,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[4]),G.d_w_store,sizeof(float)*G.N_hidden*(G.steps-1),cudaMemcpyDeviceToHost)); }
    if (nlhs>5){ plhs[5]=mxCreateNumericMatrix(num_u_plot,t_2s_max_steps,mxSINGLE_CLASS,mxREAL); if(num_u_plot>0 && t_2s_max_steps>0) CUDA_CHECK(cudaMemcpy(mxGetData(plhs[5]),G.d_u_buffer,sizeof(float)*num_u_plot*t_2s_max_steps,cudaMemcpyDeviceToHost)); }
    if (nlhs>6){ plhs[6]=mxCreateNumericMatrix(G.N_hidden,1,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[6]),G.d_g_b_traj,sizeof(float)*G.N_hidden,cudaMemcpyDeviceToHost)); }
    if (nlhs>7){ plhs[7]=mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[7]),G.d_traj_loss,sizeof(float),cudaMemcpyDeviceToHost)); }
    if (nlhs>8){ plhs[8]=mxCreateDoubleScalar((double)(G.steps-1)); }
}

static void cmdRunDiagnostic(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    requireInitialized();
    if (nrhs != 6 || nlhs > 12) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "run_diagnostic returns up to the same 12 outputs as the original MEX");
    const mxArray* x_m = prhs[1]; const mxArray* lambda_m = prhs[2];
    bool doRead = getBoolR(prhs[3]); int t_2s_max_steps = getIntScalarR(prhs[4],"t_2s_max_steps"); const mxArray* neurons_for_u_m = prhs[5];
    assertSingle(x_m,"x"); if ((int)mxGetM(x_m)!=G.D || (int)mxGetN(x_m)!=G.steps) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "x must be [D x steps]");
    if (!isVectorMx(neurons_for_u_m)) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "neurons_for_u must be a vector");
    int num_u_plot = (int)mxGetNumberOfElements(neurons_for_u_m);
    std::vector<int> neurons(num_u_plot,0); mxClassID cls=mxGetClassID(neurons_for_u_m);
    for(int i=0;i<num_u_plot;++i){ int v=0; if(cls==mxINT32_CLASS) v=((int*)mxGetData(neurons_for_u_m))[i]-1; else if(cls==mxUINT32_CLASS) v=(int)((unsigned int*)mxGetData(neurons_for_u_m))[i]-1; else v=(int)((double*)mxGetData(neurons_for_u_m))[i]-1; if(v<0||v>=G.N_hidden) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:range", "neurons_for_u out of range"); neurons[i]=v; }
    ensureLogBuffers(num_u_plot,t_2s_max_steps);
    if(num_u_plot>0) CUDA_CHECK(cudaMemcpy(G.d_neurons_idx0, neurons.data(), sizeof(int)*num_u_plot, cudaMemcpyHostToDevice));
    std::vector<bool> lambda = readLambda(lambda_m);
    CUDA_CHECK(cudaMemcpy(G.d_X, mxGetData(x_m), sizeof(float)*G.D*G.steps, cudaMemcpyHostToDevice));
    runOne(lambda, doRead, true, true, num_u_plot, t_2s_max_steps);
    if(nlhs>0){ plhs[0]=mxCreateNumericMatrix(G.D,G.steps,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[0]),G.d_Z,sizeof(float)*G.D*G.steps,cudaMemcpyDeviceToHost)); }
    if(nlhs>1){ plhs[1]=mxCreateNumericMatrix(G.D,G.steps-1,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[1]),G.d_Yread,sizeof(float)*G.D*(G.steps-1),cudaMemcpyDeviceToHost)); }
    if(nlhs>2){ plhs[2]=mxCreateLogicalMatrix(G.N_hidden,G.steps); mxLogical* Slog=(mxLogical*)mxGetData(plhs[2]); std::vector<unsigned char> tmp(G.N_hidden*G.steps); CUDA_CHECK(cudaMemcpy(tmp.data(),G.d_S,sizeof(unsigned char)*tmp.size(),cudaMemcpyDeviceToHost)); for(size_t i=0;i<tmp.size();++i) Slog[i]=tmp[i]?1:0; }
    if(nlhs>3){ plhs[3]=mxCreateNumericMatrix(G.N_hidden,G.steps-1,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[3]),G.d_Iin_store,sizeof(float)*G.N_hidden*(G.steps-1),cudaMemcpyDeviceToHost)); }
    if(nlhs>4){ plhs[4]=mxCreateNumericMatrix(G.N_hidden,G.steps-1,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[4]),G.d_Irec_store,sizeof(float)*G.N_hidden*(G.steps-1),cudaMemcpyDeviceToHost)); }
    if(nlhs>5){ plhs[5]=mxCreateNumericMatrix(G.N_hidden,G.steps-1,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[5]),G.d_w_store,sizeof(float)*G.N_hidden*(G.steps-1),cudaMemcpyDeviceToHost)); }
    if(nlhs>6){ plhs[6]=mxCreateNumericMatrix(num_u_plot,t_2s_max_steps,mxSINGLE_CLASS,mxREAL); if(num_u_plot>0 && t_2s_max_steps>0) CUDA_CHECK(cudaMemcpy(mxGetData(plhs[6]),G.d_u_buffer,sizeof(float)*num_u_plot*t_2s_max_steps,cudaMemcpyDeviceToHost)); }
    if(nlhs>7){ plhs[7]=mxCreateNumericMatrix(G.N_hidden,1,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[7]),G.d_g_b_traj,sizeof(float)*G.N_hidden,cudaMemcpyDeviceToHost)); }
    if(nlhs>8){ plhs[8]=mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[8]),G.d_traj_loss,sizeof(float),cudaMemcpyDeviceToHost)); }
    if(nlhs>9){ plhs[9]=mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[9]),G.d_read_loss,sizeof(float),cudaMemcpyDeviceToHost)); }
    if(nlhs>10){ plhs[10]=mxCreateNumericMatrix(G.D,G.N_hidden,mxSINGLE_CLASS,mxREAL); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[10]),G.d_Grad_epoch,sizeof(float)*G.D*G.N_hidden,cudaMemcpyDeviceToHost)); }
    if(nlhs>11){ plhs[11]=mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL); unsigned int nacc=0; CUDA_CHECK(cudaMemcpy(&nacc,G.d_Nacc,sizeof(unsigned int),cudaMemcpyDeviceToHost)); ((float*)mxGetData(plhs[11]))[0]=(float)nacc; }
}

static void cmdGetParams(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)prhs; if (nrhs != 1 || nlhs != 2) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "get_params returns [B,W_read]");
    requireInitialized(); plhs[0]=mxCreateNumericMatrix(G.N_hidden,1,mxSINGLE_CLASS,mxREAL); plhs[1]=mxCreateNumericMatrix(G.D,G.N_hidden,mxSINGLE_CLASS,mxREAL);
    CUDA_CHECK(cudaMemcpy(mxGetData(plhs[0]),G.d_B,sizeof(float)*G.N_hidden,cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(mxGetData(plhs[1]),G.d_W_read,sizeof(float)*G.D*G.N_hidden,cudaMemcpyDeviceToHost));
}

static void cmdGetBias(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)prhs;
    if (nrhs != 1 || nlhs != 1) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "get_bias returns B");
    requireInitialized();
    plhs[0]=mxCreateNumericMatrix(G.N_hidden,1,mxSINGLE_CLASS,mxREAL);
    CUDA_CHECK(cudaMemcpy(mxGetData(plhs[0]),G.d_B,sizeof(float)*G.N_hidden,cudaMemcpyDeviceToHost));
}

static void cmdGetOptimState(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)prhs;
    if (nrhs != 1 || nlhs != 4) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "get_optim_state returns [m_b,v_b,vhat_b,t_adam]");
    requireInitialized();
    if (!G.optim_initialized) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:optim", "Call init_optim before get_optim_state.");
    plhs[0]=mxCreateNumericMatrix(G.N_hidden,1,mxSINGLE_CLASS,mxREAL);
    plhs[1]=mxCreateNumericMatrix(G.N_hidden,1,mxSINGLE_CLASS,mxREAL);
    plhs[2]=mxCreateNumericMatrix(G.N_hidden,1,mxSINGLE_CLASS,mxREAL);
    plhs[3]=mxCreateDoubleScalar((double)G.t_adam);
    CUDA_CHECK(cudaMemcpy(mxGetData(plhs[0]),G.d_m_b,sizeof(float)*G.N_hidden,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mxGetData(plhs[1]),G.d_v_b,sizeof(float)*G.N_hidden,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mxGetData(plhs[2]),G.d_vhat_b,sizeof(float)*G.N_hidden,cudaMemcpyDeviceToHost));
}

static void cmdSetOptimState(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)nlhs; (void)plhs;
    if (nrhs != 5) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "set_optim_state expects m_b,v_b,vhat_b,t_adam");
    requireInitialized();
    if (!G.optim_initialized) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:optim", "Call init_optim before set_optim_state.");
    assertSingle(prhs[1], "m_b");
    assertSingle(prhs[2], "v_b");
    assertSingle(prhs[3], "vhat_b");
    if ((int)mxGetNumberOfElements(prhs[1]) != G.N_hidden ||
        (int)mxGetNumberOfElements(prhs[2]) != G.N_hidden ||
        (int)mxGetNumberOfElements(prhs[3]) != G.N_hidden)
        mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "Optimizer state vector length mismatch");
    CUDA_CHECK(cudaMemcpy(G.d_m_b,mxGetData(prhs[1]),sizeof(float)*G.N_hidden,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.d_v_b,mxGetData(prhs[2]),sizeof(float)*G.N_hidden,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.d_vhat_b,mxGetData(prhs[3]),sizeof(float)*G.N_hidden,cudaMemcpyHostToDevice));
    G.t_adam = getIntScalarR(prhs[4], "t_adam");
}

static void cmdUpdateParams(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)nlhs; (void)plhs; if (nrhs != 3) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "update_params expects B and W_read"); requireInitialized(); assertSingle(prhs[1],"B"); assertSingle(prhs[2],"W_read");
    if ((int)mxGetNumberOfElements(prhs[1])!=G.N_hidden || (int)mxGetM(prhs[2])!=G.D || (int)mxGetN(prhs[2])!=G.N_hidden) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "B or W_read shape mismatch");
    CUDA_CHECK(cudaMemcpy(G.d_B,mxGetData(prhs[1]),sizeof(float)*G.N_hidden,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(G.d_W_read,mxGetData(prhs[2]),sizeof(float)*G.D*G.N_hidden,cudaMemcpyHostToDevice));
}

static void cmdUpdateBias(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    (void)nlhs; (void)plhs;
    if (nrhs != 2) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "update_bias expects B");
    requireInitialized();
    assertSingle(prhs[1], "B");
    if ((int)mxGetNumberOfElements(prhs[1]) != G.N_hidden)
        mexErrMsgIdAndTxt("snn_dyn_gpu_mex:shape", "B shape mismatch");
    CUDA_CHECK(cudaMemcpy(G.d_B,mxGetData(prhs[1]),sizeof(float)*G.N_hidden,cudaMemcpyHostToDevice));
}

extern "C" void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 1 || !mxIsChar(prhs[0])) mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "First input must be command string");
    std::string cmd = getStringR(prhs[0]);
    if (cmd=="init") { cmdInit(nlhs,plhs,nrhs,prhs); return; }
    if (cmd=="init_optim") { cmdInitOptim(nlhs,plhs,nrhs,prhs); return; }
    if (cmd=="train_epoch") { trainCommon(nlhs,plhs,nrhs,prhs,false); return; }
    if (cmd=="train_primary_epoch") { cmdTrainPrimaryEpoch(nlhs,plhs,nrhs,prhs); return; }
    if (cmd=="train_read_epoch") { trainCommon(nlhs,plhs,nrhs,prhs,true); return; }
    if (cmd=="run_primary_eval") { cmdRunPrimaryEval(nlhs,plhs,nrhs,prhs); return; }
    if (cmd=="run_diagnostic") { cmdRunDiagnostic(nlhs,plhs,nrhs,prhs); return; }
    if (cmd=="run_primary_diagnostic") { cmdRunPrimaryDiagnostic(nlhs,plhs,nrhs,prhs); return; }
    if (cmd=="get_params") { cmdGetParams(nlhs,plhs,nrhs,prhs); return; }
    if (cmd=="get_bias") { cmdGetBias(nlhs,plhs,nrhs,prhs); return; }
    if (cmd=="get_optim_state") { cmdGetOptimState(nlhs,plhs,nrhs,prhs); return; }
    if (cmd=="set_optim_state") { cmdSetOptimState(nlhs,plhs,nrhs,prhs); return; }
    if (cmd=="update_params") { cmdUpdateParams(nlhs,plhs,nrhs,prhs); return; }
    if (cmd=="update_bias") { cmdUpdateBias(nlhs,plhs,nrhs,prhs); return; }
    if (cmd=="clear") { clearContext(); return; }
    mexErrMsgIdAndTxt("snn_dyn_gpu_mex:args", "Unknown command '%s'", cmd.c_str());
}
