// train_qwen3_hip.cpp
// AMD HIP GPU implementation of Qwen3 Training
// Compile: hipcc -O3 train_qwen3_hip.cpp -o train_qwen3_hip

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <random>
#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include <fstream>
#include <sstream>

// ============================================================
// Error checking
// ============================================================
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Hyperparameters (Scaled down to ~250k params for testing)
// ============================================================
const int B = 16;         // batch_size
const int T = 32;         // max_seq_len
const int MAX_ITERS = 8000;
const int EVAL_INTERVAL = 100;
const float LR = 1e-3f;
const int EVAL_ITERS = 200;
const int C = 64;         // d_model
const int NH = 4;         // n_heads
const int NKV = 2;        // n_kv_heads
const int NGRP = NH / NKV;// n_kv_groups
const int DK = C / NH;    // d_k = 16
const int DFF = 256;      // d_ff (4 * C)
const int L = 4;          // n_layers
const float RMS_EPS = 1e-6f;
const int BLOCK_SIZE_KERNEL = 256;

// ============================================================
// Tokenizer
// ============================================================
struct CharTokenizer {
    std::map<char, int> stoi;
    std::map<int, char> itos;
    int vocab_size;

    void init(const std::string& text) {
        std::vector<char> unique_chars;
        for (char ch : text) {
            if (stoi.find(ch) == stoi.end()) {
                stoi[ch] = (int)unique_chars.size();
                itos[(int)unique_chars.size()] = ch;
                unique_chars.push_back(ch);
            }
        }
        std::sort(unique_chars.begin(), unique_chars.end());
        stoi.clear(); itos.clear();
        for (int i = 0; i < (int)unique_chars.size(); i++) {
            stoi[unique_chars[i]] = i;
            itos[i] = unique_chars[i];
        }
        vocab_size = (int)unique_chars.size();
    }

    std::vector<int> encode(const std::string& s) {
        std::vector<int> r;
        for (char ch : s) r.push_back(stoi[ch]);
        return r;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string r;
        for (int id : ids) r += itos[id];
        return r;
    }
};

// ============================================================
// HIP Kernels - Forward Pass
// ============================================================

__global__ void encoder_forward_kernel(float* out, const int* inp,
                                       const float* wte,
                                       int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx >= N) return;
    int c = idx % C;
    int bt = idx / C;
    int ix = inp[bt];
    out[idx] = wte[ix * C + c];
}

__global__ void rmsnorm_forward_kernel(float* out, float* rstd,
                                       const float* inp, const float* weight,
                                       int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx >= N) return;
    int c = idx % C;
    int bt = idx / C;
    const float* x = inp + bt * C;
    float ss = 0.0f;
    for (int i = 0; i < C; i++) ss += x[i] * x[i];
    ss = ss / C + RMS_EPS;
    float s = 1.0f / sqrtf(ss);
    out[idx] = s * x[c] * weight[c];
    if (c == 0) rstd[bt] = s;
}

__global__ void matmul_forward_kernel(float* out, const float* inp,
                                      const float* weight, const float* bias,
                                      int total_bt, int C, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = total_bt * OC;
    if (idx >= N) return;
    int o = idx % OC;
    int bt = idx / OC;
    float val = (bias != NULL) ? bias[o] : 0.0f;
    const float* inp_bt = inp + bt * C;
    const float* wrow = weight + o * C;
    for (int i = 0; i < C; i++) val += inp_bt[i] * wrow[i];
    out[idx] = val;
}

__global__ void rope_forward_kernel(float* out, const float* inp,
                                    const float* cos_table, const float* sin_table,
                                    int B, int T, int n_heads, int d_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * n_heads * d_k;
    if (idx >= N) return;
    
    int i = idx % d_k;
    if (i % 2 != 0) return; // Only process even indices; odd indices are computed alongside them
    
    int h = (idx / d_k) % n_heads;
    int t = (idx / (d_k * n_heads)) % T;
    
    int half_idx = i / 2;          // Map pair index to frequency table index
    int half = d_k / 2;            // Size of the frequency table
    
    float cos_val = cos_table[t * half + half_idx];
    float sin_val = sin_table[t * half + half_idx];
    
    int bth_base = idx - i;        // Base pointer for this specific (b, t, h) vector
    
    float x1 = inp[bth_base + i];     // Even index
    float x2 = inp[bth_base + i + 1]; // Odd index
    
    // Standard 2D rotation matrix application
    out[bth_base + i]     = x1 * cos_val - x2 * sin_val;
    out[bth_base + i + 1] = x1 * sin_val + x2 * cos_val;
}

__global__ void gqa_attention_forward_kernel(
    float* out, float* att, // Removed preatt to save memory
    const float* q, const float* k, const float* v,
    int B, int T, int NH, int NKV, int DK) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * NH;
    if (idx >= total) return;
    
    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (T * NH);
    int kv_h = h / NGRP;
    float scale = 1.0f / sqrtf((float)DK);
    
    const float* q_bth = q + b * T * NH * DK + t * NH * DK + h * DK;
    const float* k_b = k + b * T * NKV * DK;
    const float* v_b = v + b * T * NKV * DK;
    
    float* att_bth = att + b * NH * T * T + h * T * T + t * T;
    
    // Find max for numerical stability (fused, no preatt memory needed)
    float maxval = -1e10f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* key = k_b + t2 * NKV * DK + kv_h * DK;
        float val = 0.0f;
        for (int i = 0; i < DK; i++) val += q_bth[i] * key[i];
        val *= scale;
        if (val > maxval) maxval = val;
    }
    for (int t2 = t + 1; t2 < T; t2++) att_bth[t2] = 0.0f;
    
    // Compute softmax and save att
    float expsum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* key = k_b + t2 * NKV * DK + kv_h * DK;
        float val = 0.0f;
        for (int i = 0; i < DK; i++) val += q_bth[i] * key[i];
        val *= scale;
        float expv = expf(val - maxval);
        expsum += expv;
        att_bth[t2] = expv;
    }
    float inv_sum = (expsum == 0.0f) ? 0.0f : 1.0f / expsum;
    for (int t2 = 0; t2 <= t; t2++) att_bth[t2] *= inv_sum;
    
    // Weighted aggregation of values
    float* out_bth = out + b * T * NH * DK + t * NH * DK + h * DK;
    for (int i = 0; i < DK; i++) out_bth[i] = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* value = v_b + t2 * NKV * DK + kv_h * DK;
        float a = att_bth[t2];
        for (int i = 0; i < DK; i++) out_bth[i] += a * value[i];
    }
}

__global__ void swiglu_forward_kernel(float* out, const float* gate, const float* up, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float g = gate[idx];
    float silu = g / (1.0f + expf(-g));
    out[idx] = silu * up[idx];
}

__global__ void residual_forward_kernel(float* out, const float* inp1, const float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = inp1[idx] + inp2[idx];
}

__global__ void softmax_forward_kernel(float* probs, const float* logits, int B, int T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;
    const float* l_bt = logits + idx * V;
    float* p_bt = probs + idx * V;
    float maxval = -1e10f;
    for (int i = 0; i < V; i++) if (l_bt[i] > maxval) maxval = l_bt[i];
    float sum = 0.0f;
    for (int i = 0; i < V; i++) { p_bt[i] = expf(l_bt[i] - maxval); sum += p_bt[i]; }
    for (int i = 0; i < V; i++) p_bt[i] /= sum;
}

__global__ void crossentropy_forward_kernel(float* losses, const float* probs,
                                            const int* targets, int B, int T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;
    int ix = targets[idx];
    losses[idx] = -logf(probs[idx * V + ix] + 1e-10f);
}

// ============================================================
// HIP Kernels - Backward Pass
// ============================================================

__global__ void encoder_backward_kernel(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx >= N) return;
    int c = idx % C;
    int bt = idx / C;
    int ix = inp[bt];
    float d = dout[idx];
    atomicAdd(&dwte[ix * C + c], d);
}

__global__ void rmsnorm_backward_kernel(float* dinp, float* dweight,
                                        const float* dout, const float* inp,
                                        const float* weight, const float* rstd,
                                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx >= N) return;
    int c = idx % C;
    int bt = idx / C;
    float s_bt = rstd[bt];
    const float* dout_bt = dout + bt * C;
    const float* inp_bt = inp + bt * C;
    
    float dnorm_sum = 0.0f;
    for (int i = 0; i < C; i++) {
        dnorm_sum += weight[i] * dout_bt[i] * inp_bt[i];
    }
    dnorm_sum = dnorm_sum * s_bt * s_bt / C;
    
    atomicAdd(&dweight[c], inp_bt[c] * s_bt * dout_bt[c]);
    dinp[bt * C + c] += s_bt * (weight[c] * dout_bt[c] - dnorm_sum * inp_bt[c]);
}

__global__ void matmul_backward_dinp_kernel(float* dinp, const float* dout,
                                            const float* weight, int total_bt, int C, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = total_bt * C;
    if (idx >= N) return;
    int i = idx % C;
    int bt = idx / C;
    const float* dout_bt = dout + bt * OC;
    float val = 0.0f;
    for (int o = 0; o < OC; o++) val += dout_bt[o] * weight[o * C + i];
    dinp[idx] += val;
}

__global__ void matmul_backward_dweight_dbias_kernel(float* dweight, float* dbias,
                                                     const float* dout, const float* inp,
                                                     int total_bt, int C, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = OC * C;
    if (idx >= N) return;
    int i = idx % C;
    int o = idx / C;
    float dw = 0.0f;
    for (int bt = 0; bt < total_bt; bt++) {
        dw += dout[bt * OC + o] * inp[bt * C + i];
    }
    dweight[idx] += dw;
}

__global__ void rope_backward_kernel(float* dinp, const float* dout,
                                     const float* cos_table, const float* sin_table,
                                     int B, int T, int n_heads, int d_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * n_heads * d_k;
    if (idx >= N) return;
    
    int i = idx % d_k;
    if (i % 2 != 0) return; // Only process even indices
    
    int h = (idx / d_k) % n_heads;
    int t = (idx / (d_k * n_heads)) % T;
    
    int half_idx = i / 2;
    int half = d_k / 2;
    
    float cos_val = cos_table[t * half + half_idx];
    float sin_val = sin_table[t * half + half_idx];
    
    int bth_base = idx - i;
    
    float dy1 = dout[bth_base + i];     // Gradient for even index
    float dy2 = dout[bth_base + i + 1]; // Gradient for odd index
    
    // Backprop through the 2D rotation matrix
    // y1 = x1*cos - x2*sin  =>  dx1 += dy1*cos, dx2 += -dy1*sin
    // y2 = x1*sin + x2*cos  =>  dx1 += dy2*sin, dx2 += dy2*cos
    float dx1 = dy1 * cos_val + dy2 * sin_val;
    float dx2 = -dy1 * sin_val + dy2 * cos_val;
    
    dinp[bth_base + i]     += dx1;
    dinp[bth_base + i + 1] += dx2;
}

__global__ void gqa_attention_backward_kernel(
    float* dq, float* dk, float* dv, // Removed dpreatt, datt
    const float* dout, const float* q, const float* k, const float* v, const float* att,
    int B, int T, int NH, int NKV, int DK) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * NH;
    if (idx >= total) return;
    
    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (T * NH);
    int kv_h = h / NGRP;
    float scale = 1.0f / sqrtf((float)DK);
    
    const float* att_bth = att + b * NH * T * T + h * T * T + t * T;
    const float* dout_bth = dout + b * T * NH * DK + t * NH * DK + h * DK;
    const float* q_bth = q + b * T * NH * DK + t * NH * DK + h * DK;
    float* dq_bth = dq + b * T * NH * DK + t * NH * DK + h * DK;
    
    const float* k_b = k + b * T * NKV * DK;
    const float* v_b = v + b * T * NKV * DK;
    float* dk_b = dk + b * T * NKV * DK;
    float* dv_b = dv + b * T * NKV * DK;
    
    // Pass 1: Compute dot and dV
    float dot = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        float datt_t2 = 0.0f;
        const float* value_t2 = v_b + t2 * NKV * DK + kv_h * DK;
        float* dvalue_t2 = dv_b + t2 * NKV * DK + kv_h * DK;
        for (int i = 0; i < DK; i++) {
            datt_t2 += value_t2[i] * dout_bth[i];
            atomicAdd(&dvalue_t2[i], att_bth[t2] * dout_bth[i]);
        }
        dot += att_bth[t2] * datt_t2;
    }
    
    // Pass 2: Compute dQ and dK (fusing dpreatt)
    for (int t2 = 0; t2 <= t; t2++) {
        float datt_t2 = 0.0f;
        const float* value_t2 = v_b + t2 * NKV * DK + kv_h * DK;
        for (int i = 0; i < DK; i++) {
            datt_t2 += value_t2[i] * dout_bth[i];
        }
        
        float dpreatt_t2 = att_bth[t2] * (datt_t2 - dot) * scale;
        
        const float* key_t2 = k_b + t2 * NKV * DK + kv_h * DK;
        float* dkey_t2 = dk_b + t2 * NKV * DK + kv_h * DK;
        
        for (int i = 0; i < DK; i++) {
            dq_bth[i] += key_t2[i] * dpreatt_t2;
            atomicAdd(&dkey_t2[i], q_bth[i] * dpreatt_t2);
        }
    }
}

__global__ void swiglu_backward_kernel(float* dgate, float* dup,
                                       const float* dout, const float* gate,
                                       const float* up, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float g = gate[idx];
    float sigmoid = 1.0f / (1.0f + expf(-g));
    float silu = g * sigmoid;
    float dsilu = sigmoid + g * sigmoid * (1.0f - sigmoid);
    
    dgate[idx] += dout[idx] * up[idx] * dsilu;
    dup[idx] += dout[idx] * silu;
}

__global__ void residual_backward_kernel(float* dinp1, float* dinp2, const float* dout, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    dinp1[idx] += dout[idx];
    dinp2[idx] += dout[idx];
}

__global__ void crossentropy_softmax_backward_kernel(float* dlogits,
                                                     const float* dlosses, const float* probs, const int* targets,
                                                     int B, int T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;
    float* dlogits_bt = dlogits + idx * V;
    const float* probs_bt = probs + idx * V;
    float dloss = dlosses[idx];
    int ix = targets[idx];
    for (int i = 0; i < V; i++) {
        float indicator = (i == ix) ? 1.0f : 0.0f;
        dlogits_bt[i] += (probs_bt[i] - indicator) * dloss;
    }
}

// ============================================================
// AdamW Update Kernel
// ============================================================
__global__ void adamw_kernel(float* params, float* grads, float* m, float* v,
                             size_t num_parameters, float learning_rate, float beta1, float beta2,
                             float eps, float weight_decay, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) return;
    float param = params[idx];
    float grad = grads[idx];
    float m_val = beta1 * m[idx] + (1.0f - beta1) * grad;
    float v_val = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    m[idx] = m_val;
    v[idx] = v_val;
    float m_hat = m_val / (1.0f - powf(beta1, (float)step));
    float v_hat = v_val / (1.0f - powf(beta2, (float)step));
    params[idx] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
}

// ============================================================
// Model Struct and Memory Management
// ============================================================

struct TensorDesc {
    float* d_ptr;
    size_t size;
};

struct Qwen3 {
    int V; 
    float *d_wte;
    float *d_rms1w, *d_qw, *d_kw, *d_vw, *d_q_norm_w, *d_k_norm_w, *d_attprojw, *d_rms2w;
    float *d_gate_proj_w, *d_up_proj_w, *d_down_proj_w;
    float *d_rmsfw, *d_lm_head;
    
    float *d_dwte;
    float *d_drms1w, *d_dqw, *d_dkw, *d_dvw, *d_dq_norm_w, *d_dk_norm_w, *d_dattprojw, *d_drms2w;
    float *d_dgate_proj_w, *d_dup_proj_w, *d_ddown_proj_w;
    float *d_drmsfw, *d_dlm_head;
    
    float *d_m, *d_v_adam;
    
    float *d_encoded, *d_rms1, *d_rms1_rstd;
    float *d_q, *d_k, *d_v, *d_q_normed, *d_k_normed, *d_q_norm_rstd, *d_k_norm_rstd;
    float *d_q_rope, *d_k_rope;
    float *d_rope_cos, *d_rope_sin;
    float *d_atty, *d_att, *d_attproj, *d_residual2;
    float *d_rms2, *d_rms2_rstd;
    float *d_gate, *d_up, *d_swiglu, *d_down, *d_residual3;
    float *d_rmsf, *d_rmsf_rstd;
    float *d_logits, *d_probs, *d_losses;
    
    float *d_dencoded, *d_drms1;
    float *d_dq, *d_dk, *d_dv, *d_dq_normed, *d_dk_normed;
    float *d_dq_rope, *d_dk_rope;
    float *d_datty, *d_dattproj, *d_dresidual2;
    float *d_drms2, *d_dgate, *d_dup, *d_dswiglu, *d_ddown, *d_dresidual3;
    float *d_drmsf, *d_dlogits;
    int *d_inputs, *d_targets;
    size_t num_parameters;
};

std::vector<TensorDesc> get_param_list(Qwen3* model) {
    return {
        {model->d_wte, (size_t)model->V * C},
        {model->d_rms1w, (size_t)L * C}, {model->d_qw, (size_t)L * NH * DK * C},
        {model->d_kw, (size_t)L * NKV * DK * C}, {model->d_vw, (size_t)L * NKV * DK * C},
        {model->d_q_norm_w, (size_t)L * DK}, {model->d_k_norm_w, (size_t)L * DK},
        {model->d_attprojw, (size_t)L * C * C}, {model->d_rms2w, (size_t)L * C},
        {model->d_gate_proj_w, (size_t)L * DFF * C}, {model->d_up_proj_w, (size_t)L * DFF * C},
        {model->d_down_proj_w, (size_t)L * C * DFF}, {model->d_rmsfw, (size_t)C},
        {model->d_lm_head, (size_t)model->V * C}
    };
}

std::vector<TensorDesc> get_grad_list(Qwen3* model) {
    return {
        {model->d_dwte, (size_t)model->V * C},
        {model->d_drms1w, (size_t)L * C}, {model->d_dqw, (size_t)L * NH * DK * C},
        {model->d_dkw, (size_t)L * NKV * DK * C}, {model->d_dvw, (size_t)L * NKV * DK * C},
        {model->d_dq_norm_w, (size_t)L * DK}, {model->d_dk_norm_w, (size_t)L * DK},
        {model->d_dattprojw, (size_t)L * C * C}, {model->d_drms2w, (size_t)L * C},
        {model->d_dgate_proj_w, (size_t)L * DFF * C}, {model->d_dup_proj_w, (size_t)L * DFF * C},
        {model->d_ddown_proj_w, (size_t)L * C * DFF}, {model->d_drmsfw, (size_t)C},
        {model->d_dlm_head, (size_t)model->V * C}
    };
}

size_t calc_num_parameters(int V) {
    return (size_t)(V * C) + L * (C + NH * DK * C + NKV * DK * C + NKV * DK * C + DK + DK + C * C + C + DFF * C + DFF * C + C * DFF) + C + V * C;
}


void precompute_rope(float* cos_table, float* sin_table, int max_seq_len, int d_k) {
    int half = d_k / 2;
    std::vector<float> freqs(half, 0.0f);
    
    // Standard Qwen/Llama RoPE frequencies
    for (int i = 0; i < half; i++) {
        freqs[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)d_k);
    }
    
    for (int t = 0; t < max_seq_len; t++) {
        for (int j = 0; j < half; j++) {
            float theta = (float)t * freqs[j];
            cos_table[t * half + j] = cosf(theta);
            sin_table[t * half + j] = sinf(theta);
        }
    }
}

void qwen3_alloc(Qwen3 *model, int V) {
    model->V = V;
    model->num_parameters = calc_num_parameters(V);
    
    auto params = get_param_list(model);
    for (auto& p : params) HIP_CHECK(hipMalloc(&p.d_ptr, p.size * sizeof(float)));
    model->d_wte = params[0].d_ptr; model->d_rms1w = params[1].d_ptr; model->d_qw = params[2].d_ptr;
    model->d_kw = params[3].d_ptr; model->d_vw = params[4].d_ptr; model->d_q_norm_w = params[5].d_ptr;
    model->d_k_norm_w = params[6].d_ptr; model->d_attprojw = params[7].d_ptr; model->d_rms2w = params[8].d_ptr;
    model->d_gate_proj_w = params[9].d_ptr; model->d_up_proj_w = params[10].d_ptr; model->d_down_proj_w = params[11].d_ptr;
    model->d_rmsfw = params[12].d_ptr; model->d_lm_head = params[13].d_ptr;

    auto grads = get_grad_list(model);
    for (auto& g : grads) HIP_CHECK(hipMalloc(&g.d_ptr, g.size * sizeof(float)));
    model->d_dwte = grads[0].d_ptr; model->d_drms1w = grads[1].d_ptr; model->d_dqw = grads[2].d_ptr;
    model->d_dkw = grads[3].d_ptr; model->d_dvw = grads[4].d_ptr; model->d_dq_norm_w = grads[5].d_ptr;
    model->d_dk_norm_w = grads[6].d_ptr; model->d_dattprojw = grads[7].d_ptr; model->d_drms2w = grads[8].d_ptr;
    model->d_dgate_proj_w = grads[9].d_ptr; model->d_dup_proj_w = grads[10].d_ptr; model->d_ddown_proj_w = grads[11].d_ptr;
    model->d_drmsfw = grads[12].d_ptr; model->d_dlm_head = grads[13].d_ptr;

    HIP_CHECK(hipMalloc(&model->d_m, model->num_parameters * sizeof(float)));
    HIP_CHECK(hipMalloc(&model->d_v_adam, model->num_parameters * sizeof(float)));
    HIP_CHECK(hipMemset(model->d_m, 0, model->num_parameters * sizeof(float)));
    HIP_CHECK(hipMemset(model->d_v_adam, 0, model->num_parameters * sizeof(float)));

    #define HIP_ALLOC(ptr, size) HIP_CHECK(hipMalloc(&ptr, size * sizeof(float)));
    HIP_ALLOC(model->d_encoded, B * T * C); HIP_ALLOC(model->d_rms1, L * B * T * C);
    HIP_ALLOC(model->d_rms1_rstd, L * B * T);
    HIP_ALLOC(model->d_q, L * B * T * NH * DK); HIP_ALLOC(model->d_k, L * B * T * NKV * DK);
    HIP_ALLOC(model->d_v, L * B * T * NKV * DK); HIP_ALLOC(model->d_q_normed, L * B * T * NH * DK);
    HIP_ALLOC(model->d_k_normed, L * B * T * NKV * DK); HIP_ALLOC(model->d_q_norm_rstd, L * B * T * NH);
    HIP_ALLOC(model->d_k_norm_rstd, L * B * T * NKV); HIP_ALLOC(model->d_q_rope, L * B * T * NH * DK);
    HIP_ALLOC(model->d_k_rope, L * B * T * NKV * DK);
    HIP_ALLOC(model->d_rope_cos, T * (DK / 2)); HIP_ALLOC(model->d_rope_sin, T * (DK / 2));
    HIP_ALLOC(model->d_atty, L * B * T * C); 
    HIP_ALLOC(model->d_att, L * B * NH * T * T); // Only saving att, not preatt
    HIP_ALLOC(model->d_attproj, L * B * T * C);
    HIP_ALLOC(model->d_residual2, L * B * T * C); HIP_ALLOC(model->d_rms2, L * B * T * C);
    HIP_ALLOC(model->d_rms2_rstd, L * B * T); HIP_ALLOC(model->d_gate, L * B * T * DFF);
    HIP_ALLOC(model->d_up, L * B * T * DFF); HIP_ALLOC(model->d_swiglu, L * B * T * DFF);
    HIP_ALLOC(model->d_down, L * B * T * C); HIP_ALLOC(model->d_residual3, L * B * T * C);
    HIP_ALLOC(model->d_rmsf, B * T * C); HIP_ALLOC(model->d_rmsf_rstd, B * T);
    HIP_ALLOC(model->d_logits, B * T * V); HIP_ALLOC(model->d_probs, B * T * V);
    HIP_ALLOC(model->d_losses, B * T);

    HIP_ALLOC(model->d_dencoded, B * T * C); HIP_ALLOC(model->d_drms1, L * B * T * C);
    HIP_ALLOC(model->d_dq, L * B * T * NH * DK); HIP_ALLOC(model->d_dk, L * B * T * NKV * DK);
    HIP_ALLOC(model->d_dv, L * B * T * NKV * DK); HIP_ALLOC(model->d_dq_normed, L * B * T * NH * DK);
    HIP_ALLOC(model->d_dk_normed, L * B * T * NKV * DK); HIP_ALLOC(model->d_dq_rope, L * B * T * NH * DK);
    HIP_ALLOC(model->d_dk_rope, L * B * T * NKV * DK); HIP_ALLOC(model->d_datty, L * B * T * C);
    HIP_ALLOC(model->d_dattproj, L * B * T * C); HIP_ALLOC(model->d_dresidual2, L * B * T * C);
    HIP_ALLOC(model->d_drms2, L * B * T * C); HIP_ALLOC(model->d_dgate, L * B * T * DFF);
    HIP_ALLOC(model->d_dup, L * B * T * DFF); HIP_ALLOC(model->d_dswiglu, L * B * T * DFF);
    HIP_ALLOC(model->d_ddown, L * B * T * C); HIP_ALLOC(model->d_dresidual3, L * B * T * C);
    HIP_ALLOC(model->d_drmsf, B * T * C); HIP_ALLOC(model->d_dlogits, B * T * V);
    #undef HIP_ALLOC

    HIP_CHECK(hipMalloc(&model->d_inputs, B * T * sizeof(int)));
    HIP_CHECK(hipMalloc(&model->d_targets, B * T * sizeof(int)));
    
    std::vector<float> h_rope_cos(T * (DK / 2));
    std::vector<float> h_rope_sin(T * (DK / 2));
    precompute_rope(h_rope_cos.data(), h_rope_sin.data(), T, DK);
    HIP_CHECK(hipMemcpy(model->d_rope_cos, h_rope_cos.data(), T * (DK / 2) * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(model->d_rope_sin, h_rope_sin.data(), T * (DK / 2) * sizeof(float), hipMemcpyHostToDevice));
}

void qwen3_free(Qwen3 *model) {
    auto params = get_param_list(model);
    for (auto& p : params) HIP_CHECK(hipFree(p.d_ptr));
    auto grads = get_grad_list(model);
    for (auto& g : grads) HIP_CHECK(hipFree(g.d_ptr));
    HIP_CHECK(hipFree(model->d_m)); HIP_CHECK(hipFree(model->d_v_adam));
    #define HIP_FREE(ptr) if(ptr) HIP_CHECK(hipFree(ptr));
    HIP_FREE(model->d_encoded); HIP_FREE(model->d_rms1); HIP_FREE(model->d_rms1_rstd);
    HIP_FREE(model->d_q); HIP_FREE(model->d_k); HIP_FREE(model->d_v); HIP_FREE(model->d_q_normed);
    HIP_FREE(model->d_k_normed); HIP_FREE(model->d_q_norm_rstd); HIP_FREE(model->d_k_norm_rstd);
    HIP_FREE(model->d_q_rope); HIP_FREE(model->d_k_rope); HIP_FREE(model->d_rope_cos); HIP_FREE(model->d_rope_sin);
    HIP_FREE(model->d_atty); HIP_FREE(model->d_att); HIP_FREE(model->d_attproj);
    HIP_FREE(model->d_residual2); HIP_FREE(model->d_rms2); HIP_FREE(model->d_rms2_rstd);
    HIP_FREE(model->d_gate); HIP_FREE(model->d_up); HIP_FREE(model->d_swiglu); HIP_FREE(model->d_down);
    HIP_FREE(model->d_residual3); HIP_FREE(model->d_rmsf); HIP_FREE(model->d_rmsf_rstd);
    HIP_FREE(model->d_logits); HIP_FREE(model->d_probs); HIP_FREE(model->d_losses);
    HIP_FREE(model->d_dencoded); HIP_FREE(model->d_drms1); HIP_FREE(model->d_dq); HIP_FREE(model->d_dk);
    HIP_FREE(model->d_dv); HIP_FREE(model->d_dq_normed); HIP_FREE(model->d_dk_normed);
    HIP_FREE(model->d_dq_rope); HIP_FREE(model->d_dk_rope); HIP_FREE(model->d_datty);
    HIP_FREE(model->d_dattproj); HIP_FREE(model->d_dresidual2); HIP_FREE(model->d_drms2); 
    HIP_FREE(model->d_dgate); HIP_FREE(model->d_dup); HIP_FREE(model->d_dswiglu); HIP_FREE(model->d_ddown);
    HIP_FREE(model->d_dresidual3); HIP_FREE(model->d_drmsf); HIP_FREE(model->d_dlogits); 
    HIP_FREE(model->d_inputs); HIP_FREE(model->d_targets);
    #undef HIP_FREE
}

void qwen3_zero_grad(Qwen3 *model) {
    auto grads = get_grad_list(model);
    for (auto& g : grads) HIP_CHECK(hipMemset(g.d_ptr, 0, g.size * sizeof(float)));
    #define HIP_MEMZERO(ptr, size) HIP_CHECK(hipMemset(ptr, 0, size * sizeof(float)))
    HIP_MEMZERO(model->d_dencoded, B * T * C); HIP_MEMZERO(model->d_drms1, L * B * T * C);
    HIP_MEMZERO(model->d_dq, L * B * T * NH * DK); HIP_MEMZERO(model->d_dk, L * B * T * NKV * DK);
    HIP_MEMZERO(model->d_dv, L * B * T * NKV * DK); HIP_MEMZERO(model->d_dq_normed, L * B * T * NH * DK);
    HIP_MEMZERO(model->d_dk_normed, L * B * T * NKV * DK); HIP_MEMZERO(model->d_dq_rope, L * B * T * NH * DK);
    HIP_MEMZERO(model->d_dk_rope, L * B * T * NKV * DK); HIP_MEMZERO(model->d_datty, L * B * T * C);
    HIP_MEMZERO(model->d_dattproj, L * B * T * C); HIP_MEMZERO(model->d_dresidual2, L * B * T * C);
    HIP_MEMZERO(model->d_drms2, L * B * T * C); HIP_MEMZERO(model->d_dgate, L * B * T * DFF);
    HIP_MEMZERO(model->d_dup, L * B * T * DFF); HIP_MEMZERO(model->d_dswiglu, L * B * T * DFF);
    HIP_MEMZERO(model->d_ddown, L * B * T * C); HIP_MEMZERO(model->d_dresidual3, L * B * T * C);
    HIP_MEMZERO(model->d_drmsf, B * T * C); HIP_MEMZERO(model->d_dlogits, B * T * model->V);
    #undef HIP_MEMZERO
}

// ============================================================
// Checkpointing
// ============================================================

void qwen3_save_checkpoint(Qwen3* model, int step, const char* filepath) {
    auto params = get_param_list(model);
    FILE* file = fopen(filepath, "wb");
    if(!file) { printf("Cannot open %s for writing\n", filepath); return; }
    fwrite(&step, sizeof(int), 1, file);
    std::vector<float> cpu_buf(model->num_parameters);
    size_t offset = 0;
    for (auto& p : params) {
        HIP_CHECK(hipMemcpy(cpu_buf.data() + offset, p.d_ptr, p.size * sizeof(float), hipMemcpyDeviceToHost));
        offset += p.size;
    }
    fwrite(cpu_buf.data(), sizeof(float), model->num_parameters, file);
    HIP_CHECK(hipMemcpy(cpu_buf.data(), model->d_m, model->num_parameters * sizeof(float), hipMemcpyDeviceToHost));
    fwrite(cpu_buf.data(), sizeof(float), model->num_parameters, file);
    HIP_CHECK(hipMemcpy(cpu_buf.data(), model->d_v_adam, model->num_parameters * sizeof(float), hipMemcpyDeviceToHost));
    fwrite(cpu_buf.data(), sizeof(float), model->num_parameters, file);
    fclose(file);
    printf("Checkpoint saved to %s at step %d\n", filepath, step);
}

int qwen3_load_checkpoint(Qwen3* model, const char* filepath) {
    FILE* file = fopen(filepath, "rb");
    if(!file) { printf("No checkpoint found at %s, starting fresh.\n", filepath); return 0; }
    int step; fread(&step, sizeof(int), 1, file);
    std::vector<float> cpu_buf(model->num_parameters);
    auto params = get_param_list(model);
    fread(cpu_buf.data(), sizeof(float), model->num_parameters, file);
    size_t offset = 0;
    for (auto& p : params) {
        HIP_CHECK(hipMemcpy(p.d_ptr, cpu_buf.data() + offset, p.size * sizeof(float), hipMemcpyHostToDevice));
        offset += p.size;
    }
    fread(cpu_buf.data(), sizeof(float), model->num_parameters, file);
    HIP_CHECK(hipMemcpy(model->d_m, cpu_buf.data(), model->num_parameters * sizeof(float), hipMemcpyHostToDevice));
    fread(cpu_buf.data(), sizeof(float), model->num_parameters, file);
    HIP_CHECK(hipMemcpy(model->d_v_adam, cpu_buf.data(), model->num_parameters * sizeof(float), hipMemcpyHostToDevice));
    fclose(file);
    printf("Checkpoint loaded from %s, resuming at step %d\n", filepath, step);
    return step;
}

// ============================================================
// GPU Kernel Launch Helpers
// ============================================================
#define LAUNCH(kernel, N, ...) do { \
    int grid = (N + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL; \
    kernel<<<grid, BLOCK_SIZE_KERNEL>>>(__VA_ARGS__); \
    HIP_CHECK(hipGetLastError()); \
} while(0)

// ============================================================
// Forward Pass
// ============================================================
void qwen3_forward(Qwen3 *model, const int* h_inputs, const int* h_targets) {
    int V = model->V;
    HIP_CHECK(hipMemcpy(model->d_inputs, h_inputs, B * T * sizeof(int), hipMemcpyHostToDevice));
    if (h_targets != NULL) HIP_CHECK(hipMemcpy(model->d_targets, h_targets, B * T * sizeof(int), hipMemcpyHostToDevice));

    LAUNCH(encoder_forward_kernel, B * T * C, model->d_encoded, model->d_inputs, model->d_wte, B, T, C);

    for (int l = 0; l < L; l++) {
        float* residual = (l == 0) ? model->d_encoded : model->d_residual3 + (l - 1) * B * T * C;
        
        LAUNCH(rmsnorm_forward_kernel, B * T * C, model->d_rms1 + l*B*T*C, model->d_rms1_rstd + l*B*T, residual, model->d_rms1w + l*C, B, T, C);
        LAUNCH(matmul_forward_kernel, B * T * NH * DK, model->d_q + l*B*T*NH*DK, model->d_rms1 + l*B*T*C, model->d_qw + l*NH*DK*C, NULL, B * T, C, NH * DK);
        LAUNCH(matmul_forward_kernel, B * T * NKV * DK, model->d_k + l*B*T*NKV*DK, model->d_rms1 + l*B*T*C, model->d_kw + l*NKV*DK*C, NULL, B * T, C, NKV * DK);
        LAUNCH(matmul_forward_kernel, B * T * NKV * DK, model->d_v + l*B*T*NKV*DK, model->d_rms1 + l*B*T*C, model->d_vw + l*NKV*DK*C, NULL, B * T, C, NKV * DK);
        
        LAUNCH(rmsnorm_forward_kernel, B * T * NH * DK, model->d_q_normed + l*B*T*NH*DK, model->d_q_norm_rstd + l*B*T*NH, model->d_q + l*B*T*NH*DK, model->d_q_norm_w + l*DK, B*T*NH, 1, DK);
        LAUNCH(rmsnorm_forward_kernel, B * T * NKV * DK, model->d_k_normed + l*B*T*NKV*DK, model->d_k_norm_rstd + l*B*T*NKV, model->d_k + l*B*T*NKV*DK, model->d_k_norm_w + l*DK, B*T*NKV, 1, DK);
        
        LAUNCH(rope_forward_kernel, B * T * NH * DK, model->d_q_rope + l*B*T*NH*DK, model->d_q_normed + l*B*T*NH*DK, model->d_rope_cos, model->d_rope_sin, B, T, NH, DK);
        LAUNCH(rope_forward_kernel, B * T * NKV * DK, model->d_k_rope + l*B*T*NKV*DK, model->d_k_normed + l*B*T*NKV*DK, model->d_rope_cos, model->d_rope_sin, B, T, NKV, DK);
        
        LAUNCH(gqa_attention_forward_kernel, B * T * NH, model->d_atty + l*B*T*NH*DK, model->d_att + l*B*NH*T*T, model->d_q_rope + l*B*T*NH*DK, model->d_k_rope + l*B*T*NKV*DK, model->d_v + l*B*T*NKV*DK, B, T, NH, NKV, DK);
        LAUNCH(matmul_forward_kernel, B * T * C, model->d_attproj + l*B*T*C, model->d_atty + l*B*T*C, model->d_attprojw + l*C*C, NULL, B * T, C, C);
        LAUNCH(residual_forward_kernel, B * T * C, model->d_residual2 + l*B*T*C, residual, model->d_attproj + l*B*T*C, B * T * C);
        
        LAUNCH(rmsnorm_forward_kernel, B * T * C, model->d_rms2 + l*B*T*C, model->d_rms2_rstd + l*B*T, model->d_residual2 + l*B*T*C, model->d_rms2w + l*C, B, T, C);
        LAUNCH(matmul_forward_kernel, B * T * DFF, model->d_gate + l*B*T*DFF, model->d_rms2 + l*B*T*C, model->d_gate_proj_w + l*DFF*C, NULL, B * T, C, DFF);
        LAUNCH(matmul_forward_kernel, B * T * DFF, model->d_up + l*B*T*DFF, model->d_rms2 + l*B*T*C, model->d_up_proj_w + l*DFF*C, NULL, B * T, C, DFF);
        LAUNCH(swiglu_forward_kernel, B * T * DFF, model->d_swiglu + l*B*T*DFF, model->d_gate + l*B*T*DFF, model->d_up + l*B*T*DFF, B * T * DFF);
        LAUNCH(matmul_forward_kernel, B * T * C, model->d_down + l*B*T*C, model->d_swiglu + l*B*T*DFF, model->d_down_proj_w + l*C*DFF, NULL, B * T, DFF, C);
        LAUNCH(residual_forward_kernel, B * T * C, model->d_residual3 + l*B*T*C, model->d_residual2 + l*B*T*C, model->d_down + l*B*T*C, B * T * C);
    }

    float* residual = model->d_residual3 + (L - 1) * B * T * C;
    LAUNCH(rmsnorm_forward_kernel, B * T * C, model->d_rmsf, model->d_rmsf_rstd, residual, model->d_rmsfw, B, T, C);
    LAUNCH(matmul_forward_kernel, B * T * V, model->d_logits, model->d_rmsf, model->d_lm_head, NULL, B * T, C, V);
    LAUNCH(softmax_forward_kernel, B * T, model->d_probs, model->d_logits, B, T, V);
    if (h_targets != NULL) LAUNCH(crossentropy_forward_kernel, B * T, model->d_losses, model->d_probs, model->d_targets, B, T, V);
}

// ============================================================
// Backward Pass
// ============================================================
void qwen3_backward(Qwen3 *model) {
    int V = model->V;
    float dloss_val = 1.0f / (B * T);
    std::vector<float> h_dlosses(B * T, dloss_val);
    HIP_CHECK(hipMemcpy(model->d_losses, h_dlosses.data(), B * T * sizeof(float), hipMemcpyHostToDevice));

    LAUNCH(crossentropy_softmax_backward_kernel, B * T, model->d_dlogits, model->d_losses, model->d_probs, model->d_targets, B, T, V);
    LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_drmsf, model->d_dlogits, model->d_lm_head, B * T, C, V);
    LAUNCH(matmul_backward_dweight_dbias_kernel, V * C, model->d_dlm_head, NULL, model->d_dlogits, model->d_rmsf, B * T, C, V);

    float* residual = model->d_residual3 + (L - 1) * B * T * C;
    float* dresidual = model->d_dresidual3 + (L - 1) * B * T * C;
    LAUNCH(rmsnorm_backward_kernel, B * T * C, dresidual, model->d_drmsfw, model->d_drmsf, residual, model->d_rmsfw, model->d_rmsf_rstd, B, T, C);

    for (int l = L - 1; l >= 0; l--) {
        residual = (l == 0) ? model->d_encoded : model->d_residual3 + (l - 1) * B * T * C;
        dresidual = (l == 0) ? model->d_dencoded : model->d_dresidual3 + (l - 1) * B * T * C;

        LAUNCH(residual_backward_kernel, B * T * C, model->d_dresidual2 + l*B*T*C, model->d_ddown + l*B*T*C, model->d_dresidual3 + l*B*T*C, B * T * C);
        LAUNCH(matmul_backward_dinp_kernel, B * T * DFF, model->d_dswiglu + l*B*T*DFF, model->d_ddown + l*B*T*C, model->d_down_proj_w + l*C*DFF, B * T, DFF, C);
        LAUNCH(matmul_backward_dweight_dbias_kernel, C * DFF, model->d_ddown_proj_w + l*C*DFF, NULL, model->d_ddown + l*B*T*C, model->d_swiglu + l*B*T*DFF, B * T, DFF, C);
        LAUNCH(swiglu_backward_kernel, B * T * DFF, model->d_dgate + l*B*T*DFF, model->d_dup + l*B*T*DFF, model->d_dswiglu + l*B*T*DFF, model->d_gate + l*B*T*DFF, model->d_up + l*B*T*DFF, B * T * DFF);
        
        LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_drms2 + l*B*T*C, model->d_dgate + l*B*T*DFF, model->d_gate_proj_w + l*DFF*C, B * T, C, DFF);
        LAUNCH(matmul_backward_dweight_dbias_kernel, DFF * C, model->d_dgate_proj_w + l*DFF*C, NULL, model->d_dgate + l*B*T*DFF, model->d_rms2 + l*B*T*C, B * T, C, DFF);
        LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_drms2 + l*B*T*C, model->d_dup + l*B*T*DFF, model->d_up_proj_w + l*DFF*C, B * T, C, DFF);
        LAUNCH(matmul_backward_dweight_dbias_kernel, DFF * C, model->d_dup_proj_w + l*DFF*C, NULL, model->d_dup + l*B*T*DFF, model->d_rms2 + l*B*T*C, B * T, C, DFF);
        LAUNCH(rmsnorm_backward_kernel, B * T * C, model->d_dresidual2 + l*B*T*C, model->d_drms2w + l*C, model->d_drms2 + l*B*T*C, model->d_residual2 + l*B*T*C, model->d_rms2w + l*C, model->d_rms2_rstd + l*B*T, B, T, C);
        
        LAUNCH(residual_backward_kernel, B * T * C, dresidual, model->d_dattproj + l*B*T*C, model->d_dresidual2 + l*B*T*C, B * T * C);
        LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_datty + l*B*T*C, model->d_dattproj + l*B*T*C, model->d_attprojw + l*C*C, B * T, C, C);
        LAUNCH(matmul_backward_dweight_dbias_kernel, C * C, model->d_dattprojw + l*C*C, NULL, model->d_dattproj + l*B*T*C, model->d_atty + l*B*T*C, B * T, C, C);
        
        LAUNCH(gqa_attention_backward_kernel, B * T * NH, model->d_dq_rope + l*B*T*NH*DK, model->d_dk_rope + l*B*T*NKV*DK, model->d_dv + l*B*T*NKV*DK, model->d_datty + l*B*T*C, model->d_q_rope + l*B*T*NH*DK, model->d_k_rope + l*B*T*NKV*DK, model->d_v + l*B*T*NKV*DK, model->d_att + l*B*NH*T*T, B, T, NH, NKV, DK);
        LAUNCH(rope_backward_kernel, B * T * NH * DK, model->d_dq_normed + l*B*T*NH*DK, model->d_dq_rope + l*B*T*NH*DK, model->d_rope_cos, model->d_rope_sin, B, T, NH, DK);
        LAUNCH(rope_backward_kernel, B * T * NKV * DK, model->d_dk_normed + l*B*T*NKV*DK, model->d_dk_rope + l*B*T*NKV*DK, model->d_rope_cos, model->d_rope_sin, B, T, NKV, DK);
        
        LAUNCH(rmsnorm_backward_kernel, B * T * NH * DK, model->d_dq + l*B*T*NH*DK, model->d_dq_norm_w + l*DK, model->d_dq_normed + l*B*T*NH*DK, model->d_q + l*B*T*NH*DK, model->d_q_norm_w + l*DK, model->d_q_norm_rstd + l*B*T*NH, B*T*NH, 1, DK);
        LAUNCH(rmsnorm_backward_kernel, B * T * NKV * DK, model->d_dk + l*B*T*NKV*DK, model->d_dk_norm_w + l*DK, model->d_dk_normed + l*B*T*NKV*DK, model->d_k + l*B*T*NKV*DK, model->d_k_norm_w + l*DK, model->d_k_norm_rstd + l*B*T*NKV, B*T*NKV, 1, DK);
        
        LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_drms1 + l*B*T*C, model->d_dq + l*B*T*NH*DK, model->d_qw + l*NH*DK*C, B * T, C, NH * DK);
        LAUNCH(matmul_backward_dweight_dbias_kernel, NH * DK * C, model->d_dqw + l*NH*DK*C, NULL, model->d_dq + l*B*T*NH*DK, model->d_rms1 + l*B*T*C, B * T, C, NH * DK);
        LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_drms1 + l*B*T*C, model->d_dk + l*B*T*NKV*DK, model->d_kw + l*NKV*DK*C, B * T, C, NKV * DK);
        LAUNCH(matmul_backward_dweight_dbias_kernel, NKV * DK * C, model->d_dkw + l*NKV*DK*C, NULL, model->d_dk + l*B*T*NKV*DK, model->d_rms1 + l*B*T*C, B * T, C, NKV * DK);
        LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_drms1 + l*B*T*C, model->d_dv + l*B*T*NKV*DK, model->d_vw + l*NKV*DK*C, B * T, C, NKV * DK);
        LAUNCH(matmul_backward_dweight_dbias_kernel, NKV * DK * C, model->d_dvw + l*NKV*DK*C, NULL, model->d_dv + l*B*T*NKV*DK, model->d_rms1 + l*B*T*C, B * T, C, NKV * DK);
        
        LAUNCH(rmsnorm_backward_kernel, B * T * C, dresidual, model->d_drms1w + l*C, model->d_drms1 + l*B*T*C, residual, model->d_rms1w + l*C, model->d_rms1_rstd + l*B*T, B, T, C);
    } // End of per-layer backward loop

    LAUNCH(encoder_backward_kernel, B * T * C, model->d_dwte, NULL, model->d_dencoded, model->d_inputs, B, T, C);
}

// ============================================================
// Update Parameters
// ============================================================
void qwen3_update(Qwen3 *model, float learning_rate, int step) {
    auto params = get_param_list(model);
    auto grads = get_grad_list(model);
    
    size_t m_offset = 0;
    for(size_t i = 0; i < params.size(); i++) {
        if (params[i].size > 0) {
            int grid = (params[i].size + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL;
            adamw_kernel<<<grid, BLOCK_SIZE_KERNEL>>>(params[i].d_ptr, grads[i].d_ptr, model->d_m + m_offset, model->d_v_adam + m_offset, params[i].size, learning_rate, 0.9f, 0.999f, 1e-8f, 0.1f, step);
            m_offset += params[i].size;
        }
    }
}

// ============================================================
// Main
// ============================================================
int main() {
    std::ifstream file("input.txt");
    if (!file.is_open()) {
        fprintf(stderr, "Could not open input.txt!\n");
        return 1;
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    CharTokenizer tokenizer;
    tokenizer.init(text);
    int V = tokenizer.vocab_size;
    printf("Vocab size: %d\n", V);

    std::vector<int> tokens = tokenizer.encode(text);
    int n = (int)(0.9 * tokens.size());
    std::vector<int> train_tokens(tokens.begin(), tokens.begin() + n);
    std::vector<int> val_tokens(tokens.begin() + n, tokens.end());

    Qwen3 model;
    qwen3_alloc(&model, V);

    std::random_device rd;
    std::mt19937 gen(1337);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    int start_iter = qwen3_load_checkpoint(&model, "qwen3_hip_checkpoint.bin");
    
    if (start_iter == 0) {
        printf("Initializing Qwen3 model weights from scratch...\n");
        
        auto init_random = [&](float* d_ptr, size_t size) {
            std::vector<float> h_buf(size);
            for (size_t i = 0; i < size; i++) h_buf[i] = dist(gen);
            HIP_CHECK(hipMemcpy(d_ptr, h_buf.data(), size * sizeof(float), hipMemcpyHostToDevice));
        };
        auto init_val = [&](float* d_ptr, size_t size, float val) {
            std::vector<float> h_buf(size, val);
            HIP_CHECK(hipMemcpy(d_ptr, h_buf.data(), size * sizeof(float), hipMemcpyHostToDevice));
        };

        init_random(model.d_wte, V * C);
        
        for (int l = 0; l < L; l++) {
            init_val(model.d_rms1w + l*C, C, 1.0f);
            init_random(model.d_qw + l*NH*DK*C, NH*DK*C);
            init_random(model.d_kw + l*NKV*DK*C, NKV*DK*C);
            init_random(model.d_vw + l*NKV*DK*C, NKV*DK*C);
            init_val(model.d_q_norm_w + l*DK, DK, 1.0f);
            init_val(model.d_k_norm_w + l*DK, DK, 1.0f);
            init_random(model.d_attprojw + l*C*C, C*C);
            
            init_val(model.d_rms2w + l*C, C, 1.0f);
            init_random(model.d_gate_proj_w + l*DFF*C, DFF*C);
            init_random(model.d_up_proj_w + l*DFF*C, DFF*C);
            init_random(model.d_down_proj_w + l*C*DFF, C*DFF);
        }
        
        init_val(model.d_rmsfw, C, 1.0f);
        init_random(model.d_lm_head, V * C);
    }

    printf("Qwen3 Model initialized. Parameters: %zu\n", model.num_parameters);

    std::vector<int> h_inputs(B * T);
    std::vector<int> h_targets(B * T);
    std::vector<float> h_losses(B * T);
    
    auto eval_loss = [&](const std::vector<int>& data_tokens) -> float {
        float total_loss = 0.0f;
        int count = 0;
        int data_len = (int)data_tokens.size();
        for (int k = 0; k < EVAL_ITERS; k++) {
            for (int b = 0; b < B; b++) {
                int ix = rand() % (data_len - T);
                for (int t = 0; t < T; t++) {
                    h_inputs[b * T + t] = data_tokens[ix + t];
                    h_targets[b * T + t] = data_tokens[ix + t + 1];
                }
            }
            qwen3_forward(&model, h_inputs.data(), h_targets.data());
            HIP_CHECK(hipMemcpy(h_losses.data(), model.d_losses, B * T * sizeof(float), hipMemcpyDeviceToHost));
            for (int i = 0; i < B * T; i++) total_loss += h_losses[i];
            count += B * T;
        }
        return total_loss / count;
    };

    for (int iter = start_iter; iter < MAX_ITERS; iter++) {
        
        if (iter % EVAL_INTERVAL == 0) {
            float train_loss = eval_loss(train_tokens);
            float val_loss = eval_loss(val_tokens);
            printf("Step %d: train loss %f, val loss %f\n", iter, train_loss, val_loss);
            qwen3_save_checkpoint(&model, iter, "qwen3_hip_checkpoint.bin");
        }

        for (int b = 0; b < B; b++) {
            int ix = rand() % (n - T);
            for (int t = 0; t < T; t++) {
                h_inputs[b * T + t] = train_tokens[ix + t];
                h_targets[b * T + t] = train_tokens[ix + t + 1];
            }
        }

        qwen3_forward(&model, h_inputs.data(), h_targets.data());
        qwen3_zero_grad(&model);
        qwen3_backward(&model);
        qwen3_update(&model, LR, iter + 1);
    }
    
    qwen3_save_checkpoint(&model, MAX_ITERS, "qwen3_hip_checkpoint.bin");

    printf("Generating text...\n");
    std::vector<int> gen_tokens(1, 0); 
    std::vector<float> h_probs(V);

    for (int i = 0; i < 500; i++) {
        int start_idx = std::max(0, (int)gen_tokens.size() - T);
        int context_len = std::min((int)gen_tokens.size(), T);
        
        std::vector<int> h_gen_inputs(B * T, 0);
        for (int t = 0; t < context_len; t++) {
            h_gen_inputs[t] = gen_tokens[start_idx + t];
        }

        qwen3_forward(&model, h_gen_inputs.data(), NULL);
        
        int sample_pos = context_len - 1;
        HIP_CHECK(hipMemcpy(h_probs.data(), model.d_probs + sample_pos * V, V * sizeof(float), hipMemcpyDeviceToHost));
        
        float r = (float)rand() / RAND_MAX;
        float cdf = 0.0f;
        int next_token = V - 1;
        for (int j = 0; j < V; j++) {
            cdf += h_probs[j];
            if (r < cdf) {
                next_token = j;
                break;
            }
        }
        gen_tokens.push_back(next_token);
    }

    std::string generated_text = tokenizer.decode(gen_tokens);
    printf("---\n%s\n---\n", generated_text.c_str());

    qwen3_free(&model);
    return 0;
}