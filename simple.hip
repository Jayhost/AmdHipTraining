// train_gpt_oss_simple.cpp
// Simplified GPT-OSS for testing: Tiny params, char tokenizer, reads input.txt directly.
// Includes all features: Muon, BF16, MoE, GQA, RoPE, Sink Bias, Sliding Window.
// Compile: hipcc -O3 train_gpt_oss_simple.cpp -o train_gpt_oss_simple
// Run:     ./train_gpt_oss_simple

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
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

using bf16 = hip_bfloat16;

__device__ __host__ static inline float bf16_to_float(bf16 val) { return (float)val; }
__device__ __host__ static inline bf16 float_to_bf16(float val) { return (bf16)val; }

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static size_t g_vram_used = 0;
static const size_t VRAM_BUDGET = 8ull * 1024 * 1024 * 1024;
#define TRACKED_MALLOC(ptr, bytes) do { \
    if (g_vram_used + (bytes) > VRAM_BUDGET) { \
        fprintf(stderr, "VRAM BUDGET EXCEEDED! Tried %zu bytes, Used %zu bytes, Limit %zu bytes\n", \
                (size_t)(bytes), g_vram_used, VRAM_BUDGET); \
        exit(1); \
    } \
    HIP_CHECK(hipMalloc(&(ptr), (bytes))); \
    g_vram_used += (bytes); \
} while(0)

// ============================================================
// GPT-OSS Tiny Hyperparameters
// ============================================================
const int B = 8;          // batch_size
const int T = 32;         // max_seq_len
const int MAX_ITERS = 5000;
const int EVAL_INTERVAL = 100;
const int EVAL_ITERS = 50;
const float LR = 1e-3f;   // AdamW LR for 1D params
const int C = 128;        // d_model
const int NH = 4;         // n_heads (Query heads)
const int NKV = 2;        // n_kv_heads (GQA)
const int NGRP = NH / NKV;// n_kv_groups
const int DK = C / NH;    // d_k = 32
const int DFF = 32;       // intermediate_size AFTER SwiGLU
const int L = 4;          // n_layers
const int E = 4;          // num_experts in MoE
const int K = 2;          // experts_per_token (Top-K routing)
const int SLIDING_WINDOW = 8;
const float ROPE_THETA = 10000.0f;
const float ROPE_FACTOR = 4.0f;
const float SINK_INIT = 4.0f;
const float SWIGLU_CLIP = 7.0f;
const float AUX_COEF = 0.02f;
const float RMS_EPS = 1e-6f;
const int BLOCK_SIZE_KERNEL = 256;
const char* CKPT_PATH = "gpt_oss_simple_checkpoint.bin";

// Muon Optimizer Hyperparameters
const float MUON_LR = 1e-3f;   
const float MUON_BETA = 0.95f; 
const int NS_ITERS = 5;        

// ============================================================
// Tokenizer
// ============================================================
struct CharTokenizer {
    std::map<char, int> stoi;
    std::map<int, char> itos;
    int vocab_size = 0;
    void init(const std::string& text) {
        std::vector<char> unique_chars;
        for (char ch : text) if (stoi.find(ch) == stoi.end()) {
            stoi[ch] = (int)unique_chars.size(); itos[(int)unique_chars.size()] = ch; unique_chars.push_back(ch);
        }
        std::sort(unique_chars.begin(), unique_chars.end());
        stoi.clear(); itos.clear();
        for (int i = 0; i < (int)unique_chars.size(); i++) { stoi[unique_chars[i]] = i; itos[i] = unique_chars[i]; }
        vocab_size = (int)unique_chars.size();
    }
    std::vector<int> encode(const std::string& s) { std::vector<int> r; for (char ch : s) r.push_back(stoi[ch]); return r; }
    std::string decode(const std::vector<int>& ids) { std::string r; for (int id : ids) r += itos[id]; return r; }
};

// ============================================================
// HIP Kernels - Forward Pass (BF16)
// ============================================================
__global__ void encoder_forward_kernel(bf16* out, const int* inp, const bf16* wte, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * C) return;
    out[idx] = wte[inp[idx / C] * C + (idx % C)];
}

__global__ void rmsnorm_forward_kernel(bf16* out, float* rstd, const bf16* inp, const bf16* weight, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * C) return;
    int c = idx % C, bt = idx / C; const bf16* x = inp + bt * C;
    float ss = 0.0f; for (int i = 0; i < C; i++) { float xv = bf16_to_float(x[i]); ss += xv * xv; }
    float s = 1.0f / sqrtf(ss / C + RMS_EPS); 
    out[idx] = float_to_bf16(s * bf16_to_float(x[c]) * bf16_to_float(weight[c]));
    if (c == 0) rstd[bt] = s;
}

__global__ void matmul_forward_kernel(bf16* out, const bf16* inp, const bf16* weight, const bf16* bias, int total_bt, int C, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_bt * OC) return;
    int o = idx % OC, bt = idx / OC; float val = (bias != NULL) ? bf16_to_float(bias[o]) : 0.0f;
    const bf16* inp_bt = inp + bt * C; const bf16* wrow = weight + o * C;
    for (int i = 0; i < C; i++) val += bf16_to_float(inp_bt[i]) * bf16_to_float(wrow[i]); 
    out[idx] = float_to_bf16(val);
}

__global__ void rope_qk_forward_kernel(bf16* qkv, int B, int T, int NH, int NKV, int d_k, float rope_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = B * T * NH * d_k;
    int total_k = B * T * NKV * d_k;
    if (idx >= total_q + total_k) return;
    
    bool is_q = (idx < total_q);
    int local_idx = is_q ? idx : idx - total_q;
    int n_heads = is_q ? NH : NKV;
    
    int i = local_idx % d_k; if (i % 2 != 0) return;
    int h = (local_idx / d_k) % n_heads;
    int t = (local_idx / (d_k * n_heads)) % T;
    int b = local_idx / (d_k * n_heads * T);
    
    float theta = 1.0f / powf(ROPE_THETA, (float)(2 * (i / 2)) / (float)d_k);
    float angle = ((float)t / rope_factor) * theta;
    float cos_val = cosf(angle), sin_val = sinf(angle);
    
    int qkv_dim = (NH + 2 * NKV) * d_k;
    int bt_base = (b * T + t) * qkv_dim;
    int offset = is_q ? (h * d_k) : (NH * d_k + h * d_k);
    int target_idx = bt_base + offset + i;
    
    float x1 = bf16_to_float(qkv[target_idx]), x2 = bf16_to_float(qkv[target_idx + 1]);
    qkv[target_idx]     = float_to_bf16(x1 * cos_val - x2 * sin_val);
    qkv[target_idx + 1] = float_to_bf16(x1 * sin_val + x2 * cos_val);
}

__global__ void gqa_sink_sliding_attention_forward_kernel(
    bf16* out, float* att, const bf16* qkv, const bf16* sink_logits,
    int B, int T, int NH, int NKV, int DK, int NGRP, int sliding_window, int is_sliding_layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * NH) return;
    int h = idx % NH, t = (idx / NH) % T, b = idx / (T * NH), kv_h = h / NGRP;
    float scale = 1.0f / sqrtf((float)DK), sink_logit = bf16_to_float(sink_logits[h]);
    
    int qkv_dim = (NH + 2 * NKV) * DK;
    int q_offset = NH * DK;
    int kv_offset = NKV * DK;
    
    const bf16* q_bth = qkv + b * T * qkv_dim + t * qkv_dim + h * DK;
    const bf16* k_base = qkv + b * T * qkv_dim + q_offset;
    const bf16* v_base = qkv + b * T * qkv_dim + q_offset + kv_offset;
    float* att_bth = att + b * NH * T * T + h * T * T + t * T;
    
    float maxval = sink_logit; 
    for (int t2 = 0; t2 <= t; t2++) {
        bool mask_ok = !(is_sliding_layer && (t - t2) >= sliding_window);
        if (mask_ok) {
            const bf16* key = k_base + t2 * qkv_dim + kv_h * DK; float val = 0.0f;
            for (int i = 0; i < DK; i++) val += bf16_to_float(q_bth[i]) * bf16_to_float(key[i]); val *= scale;
            att_bth[t2] = val; if (val > maxval) maxval = val;
        } else att_bth[t2] = -1e9f;
    }
    for (int t2 = t + 1; t2 < T; t2++) att_bth[t2] = 0.0f;

    float expsum = expf(sink_logit - maxval); 
    for (int t2 = 0; t2 <= t; t2++) {
        if (att_bth[t2] != -1e9f) { float expv = expf(att_bth[t2] - maxval); expsum += expv; att_bth[t2] = expv; }
        else att_bth[t2] = 0.0f;
    }
    float inv_sum = (expsum == 0.0f) ? 0.0f : 1.0f / expsum;

    bf16* out_bth = out + b * T * NH * DK + t * NH * DK + h * DK;
    for (int i = 0; i < DK; i++) out_bth[i] = float_to_bf16(0.0f);
    for (int t2 = 0; t2 <= t; t2++) {
        att_bth[t2] *= inv_sum; 
        if (att_bth[t2] > 0.0f) {
            const bf16* value = v_base + t2 * qkv_dim + kv_h * DK; float a = att_bth[t2];
            for (int i = 0; i < DK; i++) out_bth[i] = float_to_bf16(bf16_to_float(out_bth[i]) + a * bf16_to_float(value[i]));
        }
    }
}

__global__ void moe_router_forward_kernel(
    bf16* weights_out, int* indices_out, float* aux_loss_out,
    const bf16* inp, const bf16* router_w, const bf16* router_b, int B, int T, int C, int E, int K, float aux_coef) {
    int s = blockIdx.x * blockDim.x + threadIdx.x; if (s >= B * T) return;
    float logits[16]; 
    for (int e = 0; e < E; e++) {
        float val = bf16_to_float(router_b[e]); 
        for (int c = 0; c < C; c++) val += bf16_to_float(inp[s * C + c]) * bf16_to_float(router_w[c * E + e]); logits[e] = val;
    }
    float topk_vals[8]; int topk_idxs[8]; 
    for (int k = 0; k < K; k++) { topk_vals[k] = -1e9f; topk_idxs[k] = 0; }
    for (int e = 0; e < E; e++) { for (int k = 0; k < K; k++) { if (logits[e] > topk_vals[k]) {
        for (int j = K - 1; j > k; j--) { topk_vals[j] = topk_vals[j-1]; topk_idxs[j] = topk_idxs[j-1]; }
        topk_vals[k] = logits[e]; topk_idxs[k] = e; break;
    }}}
    float max_val = topk_vals[0], sum_exp = 0.0f;
    for (int k = 0; k < K; k++) { topk_vals[k] = expf(topk_vals[k] - max_val); sum_exp += topk_vals[k]; }
    for (int k = 0; k < K; k++) { topk_vals[k] /= sum_exp; weights_out[s * K + k] = float_to_bf16(topk_vals[k]); indices_out[s * K + k] = topk_idxs[k]; }
    atomicAdd(aux_loss_out, topk_vals[0] / (B * T));
}

__global__ void zero_int_array_kernel(int* ptr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx < N) ptr[idx] = 0;
}

__global__ void compute_expert_offsets_kernel(int* expert_offsets, const int* indices, int B, int T, int E, int K) {
    __shared__ int counts[16];
    int tid = threadIdx.x;
    if (tid < E) counts[tid] = 0;
    __syncthreads();
    for (int i = tid; i < B * T * K; i += blockDim.x) {
        int e = indices[i];
        atomicAdd(&counts[e], 1);
    }
    __syncthreads();
    if (tid == 0) {
        expert_offsets[0] = 0;
        for (int e = 0; e < E; e++) expert_offsets[e+1] = expert_offsets[e] + counts[e];
    }
}

__global__ void scatter_expert_info_kernel(
    int* scattered_token_idx, int* expert_pos, bf16* scattered_token_w, int* expert_write_ptr,
    const int* expert_offsets, const int* indices, const bf16* weights, int B, int T, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * K) return;
    int s = idx / K, k = idx % K, e = indices[idx];
    int pos = atomicAdd(&expert_write_ptr[e], 1) + expert_offsets[e];
    scattered_token_idx[pos] = s;
    scattered_token_w[pos] = weights[idx];
    expert_pos[idx] = pos; 
}

__global__ void gather_tokens_kernel(bf16* expert_inp, const bf16* inp, const int* scattered_token_idx, int total_pairs, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_pairs * C) return;
    int c = idx % C, pos = idx / C;
    expert_inp[pos * C + c] = inp[scattered_token_idx[pos] * C + c];
}

__global__ void grouped_matmul_forward_kernel(bf16* out, const bf16* inp, const bf16* weight, const bf16* bias, const int* expert_offsets, int E, int max_pairs, int C, int OC) {
    int e = blockIdx.y;
    int start_pos = expert_offsets[e];
    int end_pos = expert_offsets[e+1];
    int num_tokens = end_pos - start_pos;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * OC) return;
    int o = idx % OC, local_pos = idx / OC, global_pos = start_pos + local_pos;
    float val = bf16_to_float(bias[e * OC + o]);
    const bf16* inp_pos = inp + global_pos * C;
    const bf16* wrow = weight + e * OC * C + o * C;
    for (int i = 0; i < C; i++) val += bf16_to_float(inp_pos[i]) * bf16_to_float(wrow[i]);
    out[global_pos * OC + o] = float_to_bf16(val);
}

__global__ void grouped_swiglu_forward_kernel(bf16* out, bf16* gate_up, int total_pairs, int DFF, float clip) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_pairs * DFF) return;
    int j = idx % DFF, pos = idx / DFF;
    float g = bf16_to_float(gate_up[pos * 2 * DFF + j]);
    float u = bf16_to_float(gate_up[pos * 2 * DFF + DFF + j]);
    g = (clip > 0) ? fmaxf(-clip, fminf(clip, g)) : g;
    u = (clip > 0) ? fmaxf(-clip, fminf(clip, u)) : u;
    float silu = g / (1.0f + expf(-g));
    out[idx] = float_to_bf16(silu * u);
}

__global__ void cache_expert_out_kernel(bf16* expert_out_cache, const bf16* expert_out, const int* expert_pos, int B, int T, int K, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * K * C) return;
    int c = idx % C, k = (idx / C) % K, s = idx / (C * K);
    int pos = expert_pos[s * K + k];
    expert_out_cache[idx] = expert_out[pos * C + c];
}

__global__ void scatter_tokens_kernel(float* out, const bf16* expert_out, const bf16* scattered_token_w, const int* scattered_token_idx, int total_pairs, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_pairs * C) return;
    int c = idx % C, pos = idx / C;
    int s = scattered_token_idx[pos];
    atomicAdd(&out[s * C + c], bf16_to_float(scattered_token_w[pos]) * bf16_to_float(expert_out[pos * C + c]));
}

__global__ void residual_forward_bf16_kernel(bf16* out, const bf16* inp1, const bf16* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= N) return; 
    out[idx] = float_to_bf16(bf16_to_float(inp1[idx]) + bf16_to_float(inp2[idx]));
}

__global__ void residual_forward_f32_kernel(bf16* out, const bf16* inp1, const float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= N) return; 
    out[idx] = float_to_bf16(bf16_to_float(inp1[idx]) + inp2[idx]);
}

__global__ void crossentropy_forward_fused_kernel(float* losses, const bf16* logits, const int* targets, const float* loss_mask, int B, int T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T) return;
    float mask = loss_mask[idx];
    const bf16* l_bt = logits + idx * V;
    float maxval = -1e10f;
    for (int i = 0; i < V; i++) { float v = bf16_to_float(l_bt[i]); if (v > maxval) maxval = v; }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) sum += expf(bf16_to_float(l_bt[i]) - maxval);
    int ix = targets[idx];
    float p_ix = expf(bf16_to_float(l_bt[ix]) - maxval) / sum;
    losses[idx] = mask * (-logf(p_ix + 1e-10f));
}

// ============================================================
// HIP Kernels - Backward Pass (Mostly FP32)
// ============================================================
__global__ void encoder_backward_kernel(float* dwte, float* dwpe, const float* dout, const int* inp, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * C) return;
    atomicAdd(&dwte[inp[idx / C] * C + (idx % C)], dout[idx]);
}

__global__ void rmsnorm_backward_kernel(float* dinp, float* dweight, const float* dout, const bf16* inp, const bf16* weight, const float* rstd, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * C) return;
    int c = idx % C, bt = idx / C; float s_bt = rstd[bt];
    const float* dout_bt = dout + bt * C; const bf16* inp_bt = inp + bt * C;
    float dnorm_sum = 0.0f; 
    for (int i = 0; i < C; i++) dnorm_sum += bf16_to_float(weight[i]) * dout_bt[i] * bf16_to_float(inp_bt[i]);
    dnorm_sum = dnorm_sum * s_bt * s_bt / C;
    atomicAdd(&dweight[c], bf16_to_float(inp_bt[c]) * s_bt * dout_bt[c]);
    dinp[bt * C + c] += s_bt * (bf16_to_float(weight[c]) * dout_bt[c] - dnorm_sum * bf16_to_float(inp_bt[c]));
}

__global__ void matmul_backward_dinp_kernel(float* dinp, const float* dout, const bf16* weight, int total_bt, int C, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_bt * C) return;
    int i = idx % C, bt = idx / C; const float* dout_bt = dout + bt * OC; float val = 0.0f;
    for (int o = 0; o < OC; o++) val += dout_bt[o] * bf16_to_float(weight[o * C + i]); dinp[idx] += val;
}

__global__ void matmul_backward_dweight_dbias_kernel(float* dweight, float* dbias, const float* dout, const bf16* inp, int total_bt, int C, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= OC * C) return;
    int i = idx % C, o = idx / OC; float dw = 0.0f;
    for (int bt = 0; bt < total_bt; bt++) dw += dout[bt * OC + o] * bf16_to_float(inp[bt * C + i]);
    dweight[idx] += dw; 
    if (i == 0 && dbias != NULL) {
        float db = 0.0f;
        for (int bt = 0; bt < total_bt; bt++) db += dout[bt * OC + o];
        atomicAdd(&dbias[o], db); 
    }
}

__global__ void rope_qk_backward_kernel(float* dqkv, int B, int T, int NH, int NKV, int d_k, float rope_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = B * T * NH * d_k;
    int total_k = B * T * NKV * d_k;
    if (idx >= total_q + total_k) return;
    bool is_q = (idx < total_q);
    int local_idx = is_q ? idx : idx - total_q;
    int n_heads = is_q ? NH : NKV;
    int i = local_idx % d_k; if (i % 2 != 0) return;
    int h = (local_idx / d_k) % n_heads;
    int t = (local_idx / (d_k * n_heads)) % T;
    int b = local_idx / (d_k * n_heads * T);
    float theta = 1.0f / powf(ROPE_THETA, (float)(2 * (i / 2)) / (float)d_k);
    float angle = ((float)t / rope_factor) * theta;
    float cos_val = cosf(angle), sin_val = sinf(angle); 
    int qkv_dim = (NH + 2 * NKV) * d_k;
    int bt_base = (b * T + t) * qkv_dim;
    int offset = is_q ? (h * d_k) : (NH * d_k + h * d_k);
    int target_idx = bt_base + offset + i;
    float dy1 = dqkv[target_idx], dy2 = dqkv[target_idx + 1];
    dqkv[target_idx]     = dy1 * cos_val + dy2 * sin_val;
    dqkv[target_idx + 1] = -dy1 * sin_val + dy2 * cos_val;
}

__global__ void gqa_sink_sliding_attention_backward_kernel(
    float* dqkv, float* d_sink_logits,
    const float* dout, const bf16* qkv, const float* att,
    int B, int T, int NH, int NKV, int DK, int NGRP, int sliding_window, int is_sliding_layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * NH) return;
    int h = idx % NH, t = (idx / NH) % T, b = idx / (T * NH), kv_h = h / NGRP;
    float scale = 1.0f / sqrtf((float)DK);
    int qkv_dim = (NH + 2 * NKV) * DK;
    int q_offset = NH * DK, kv_offset = NKV * DK;
    const float* att_bth = att + b * NH * T * T + h * T * T + t * T;
    const float* dout_bth = dout + b * T * NH * DK + t * NH * DK + h * DK;
    const bf16* q_bth = qkv + b * T * qkv_dim + t * qkv_dim + h * DK; 
    float* dq_bth = dqkv + b * T * qkv_dim + t * qkv_dim + h * DK; 
    const bf16* k_base = qkv + b * T * qkv_dim + q_offset;
    const bf16* v_base = qkv + b * T * qkv_dim + q_offset + kv_offset;
    float* dk_base = dqkv + b * T * qkv_dim + q_offset;
    float* dv_base = dqkv + b * T * qkv_dim + q_offset + kv_offset;

    float dot_p_dout = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        float p = att_bth[t2]; if (p <= 0.0f) continue;
        const bf16* value_t2 = v_base + t2 * qkv_dim + kv_h * DK; 
        float* dvalue_t2 = dv_base + t2 * qkv_dim + kv_h * DK;
        float d_att_t2 = 0.0f;
        for (int i = 0; i < DK; i++) { 
            d_att_t2 += bf16_to_float(value_t2[i]) * dout_bth[i]; 
            atomicAdd(&dvalue_t2[i], p * dout_bth[i]); 
        }
        dot_p_dout += p * d_att_t2;
    }

    for (int t2 = 0; t2 <= t; t2++) {
        float p = att_bth[t2]; if (p <= 0.0f) continue;
        const bf16* value_t2 = v_base + t2 * qkv_dim + kv_h * DK; 
        float d_att_t2 = 0.0f;
        for (int i = 0; i < DK; i++) d_att_t2 += bf16_to_float(value_t2[i]) * dout_bth[i];
        float dpreatt_t2 = p * (d_att_t2 - dot_p_dout) * scale;
        const bf16* key_t2 = k_base + t2 * qkv_dim + kv_h * DK; 
        float* dkey_t2 = dk_base + t2 * qkv_dim + kv_h * DK;
        for (int i = 0; i < DK; i++) { 
            dq_bth[i] += bf16_to_float(key_t2[i]) * dpreatt_t2; 
            atomicAdd(&dkey_t2[i], bf16_to_float(q_bth[i]) * dpreatt_t2); 
        }
    }
    float p_sum = 0.0f; for (int t2 = 0; t2 <= t; t2++) p_sum += att_bth[t2];
    float p_sink = 1.0f - p_sum; 
    if (p_sink > 0.0f) atomicAdd(&d_sink_logits[h], -p_sink * dot_p_dout);
}

__global__ void scatter_tokens_backward_kernel(float* d_expert_out, const float* dout, const bf16* scattered_token_w, const int* scattered_token_idx, int total_pairs, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_pairs * C) return;
    int c = idx % C, pos = idx / C;
    int s = scattered_token_idx[pos];
    d_expert_out[pos * C + c] = bf16_to_float(scattered_token_w[pos]) * dout[s * C + c];
}

__global__ void grouped_matmul_backward_dinp_kernel(float* dinp, const float* dout, const bf16* weight, const int* expert_offsets, int E, int max_pairs, int C, int OC) {
    int e = blockIdx.y;
    int start_pos = expert_offsets[e];
    int num_tokens = expert_offsets[e+1] - start_pos;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * C) return;
    int i = idx % C, local_pos = idx / C, global_pos = start_pos + local_pos;
    const float* dout_pos = dout + global_pos * OC;
    const bf16* wrow = weight + e * OC * C;
    float val = 0.0f;
    for (int o = 0; o < OC; o++) val += dout_pos[o] * bf16_to_float(wrow[o * C + i]);
    dinp[global_pos * C + i] += val;
}

__global__ void grouped_matmul_backward_dweight_kernel(float* dweight, float* dbias, const float* dout, const bf16* inp, const int* expert_offsets, int E, int max_pairs, int C, int OC) {
    int e = blockIdx.y;
    int start_pos = expert_offsets[e];
    int num_tokens = expert_offsets[e+1] - start_pos;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= OC * C) return;
    int i = idx % C, o = idx / C;
    float dw = 0.0f;
    for (int t = 0; t < num_tokens; t++) {
        int global_pos = start_pos + t;
        dw += dout[global_pos * OC + o] * bf16_to_float(inp[global_pos * C + i]);
    }
    dweight[e * OC * C + o * C + i] += dw;
    if (i == 0 && dbias != NULL) {
        float db = 0.0f;
        for (int t = 0; t < num_tokens; t++) {
            int global_pos = start_pos + t;
            db += dout[global_pos * OC + o];
        }
        atomicAdd(&dbias[e * OC + o], db); 
    }
}

__global__ void grouped_swiglu_backward_kernel(float* d_gate_up, const bf16* gate_up, const float* d_swiglu, int total_pairs, int DFF, float clip) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_pairs * DFF) return;
    int j = idx % DFF, pos = idx / DFF;
    float g = bf16_to_float(gate_up[pos * 2 * DFF + j]);
    float u = bf16_to_float(gate_up[pos * 2 * DFF + DFF + j]);
    float dg = (clip > 0 && (g <= -clip || g >= clip)) ? 0.0f : 1.0f;
    float du = (clip > 0 && (u <= -clip || u >= clip)) ? 0.0f : 1.0f;
    float sig = 1.0f / (1.0f + expf(-g));
    float dsilu = sig + g * sig * (1.0f - sig);
    d_gate_up[pos * 2 * DFF + j] += d_swiglu[idx] * u * dsilu * dg;
    d_gate_up[pos * 2 * DFF + DFF + j] += d_swiglu[idx] * (g * sig) * du;
}

__global__ void gather_tokens_backward_kernel(float* d_inp, const float* d_expert_inp, const int* scattered_token_idx, int total_pairs, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_pairs * C) return;
    int c = idx % C, pos = idx / C;
    int s = scattered_token_idx[pos];
    atomicAdd(&d_inp[s * C + c], d_expert_inp[pos * C + c]);
}

__global__ void moe_router_backward_kernel(
    float* d_inp, float* d_drw, float* d_drb,
    const bf16* expert_out_cache, const float* dout, const bf16* inp, 
    const bf16* router_w, const bf16* weights, const int* indices,
    int B, int T, int C, int E, int K) {
    int s = blockIdx.x * blockDim.x + threadIdx.x; if (s >= B * T) return;
    float* d_inp_s = d_inp + s * C;
    const float* dout_s = dout + s * C;
    float dps[8];
    for (int k = 0; k < K; k++) {
        const bf16* exp_out_s = expert_out_cache + s * K * C + k * C;
        float dp = 0.0f;
        for (int c = 0; c < C; c++) dp += dout_s[c] * bf16_to_float(exp_out_s[c]);
        dps[k] = dp;
    }
    float sum_dp_w = 0.0f; for(int k=0; k<K; k++) sum_dp_w += bf16_to_float(weights[s * K + k]) * dps[k];
    for(int k=0; k<K; k++) {
        int e = indices[s * K + k]; 
        float d_logit = bf16_to_float(weights[s * K + k]) * (dps[k] - sum_dp_w);
        for(int c=0; c<C; c++) {
            atomicAdd(&d_drw[c * E + e], bf16_to_float(inp[s * C + c]) * d_logit);
            d_inp_s[c] += bf16_to_float(router_w[c * E + e]) * d_logit;
        }
        atomicAdd(&d_drb[e], d_logit);
    }
}

__global__ void residual_backward_kernel(float* dinp1, float* dinp2, const float* dout, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= N) return; dinp1[idx] += dout[idx]; dinp2[idx] += dout[idx];
}

__global__ void crossentropy_softmax_backward_fused_kernel(float* dlogits, const float* dlosses, const bf16* logits, const int* targets, const float* loss_mask, int B, int T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T) return;
    float mask = loss_mask[idx];
    float dloss = dlosses[idx] * mask;
    const bf16* l_bt = logits + idx * V;
    float* dl_bt = dlogits + idx * V;
    float maxval = -1e10f;
    for (int i = 0; i < V; i++) { float v = bf16_to_float(l_bt[i]); if (v > maxval) maxval = v; }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) sum += expf(bf16_to_float(l_bt[i]) - maxval);
    int ix = targets[idx];
    for (int i = 0; i < V; i++) {
        float p = expf(bf16_to_float(l_bt[i]) - maxval) / sum;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        dl_bt[i] += (p - indicator) * dloss;
    }
}

// --- AdamW Kernel (For 1D Params) ---
__global__ void adamw_kernel(bf16* params, float* grads, float* m, float* v, size_t num_parameters, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= num_parameters) return;
    float param = bf16_to_float(params[idx]), grad = grads[idx];
    float m_val = beta1 * m[idx] + (1.0f - beta1) * grad, v_val = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    m[idx] = m_val; v[idx] = v_val;
    params[idx] = float_to_bf16(param - learning_rate * ((m_val / (1.0f - powf(beta1, (float)step))) / (sqrtf(v_val / (1.0f - powf(beta2, (float)step))) + eps) + weight_decay * param));
}

// --- Muon Optimizer Kernels ---
__global__ void muon_momentum_kernel(float* m, const float* grad, size_t N, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= N) return;
    m[idx] = beta * m[idx] + (1.0f - beta) * grad[idx];
}

__global__ void frobenius_norm_sq_kernel(float* out, const float* X, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x)
        sdata[tid] += X[i] * X[i];
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads(); }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

__global__ void scale_by_norm_kernel(float* X, float norm_sq, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= N) return;
    float inv_norm = (norm_sq > 0.0f) ? 1.0f / sqrtf(norm_sq) : 0.0f;
    X[idx] *= inv_norm;
}

__global__ void gram_matrix_kernel(float* G, const float* X, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx / N, j = idx % N;
    float val = 0.0f;
    for (int k = 0; k < M; k++) val += X[k * N + i] * X[k * N + j];
    G[idx] = val;
}

__global__ void ns_step_kernel(float* X_out, const float* X_in, const float* G, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int r = idx / N, c = idx % N;
    float val = 0.0f;
    for (int k = 0; k < N; k++) {
        float scale = (k == c) ? 1.5f : 0.0f;
        val += X_in[r * N + k] * (scale - 0.5f * G[k * N + c]);
    }
    X_out[idx] = val;
}

__global__ void muon_apply_kernel(bf16* params, const float* update, size_t N, float lr, float weight_decay, float scaling) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= N) return;
    float p = bf16_to_float(params[idx]);
    params[idx] = float_to_bf16(p - lr * (update[idx] * scaling + weight_decay * p));
}

// ============================================================
// Model Struct & Mem Management
// ============================================================
struct TensorDesc { bf16* d_ptr; size_t size; };
struct GradDesc { float* d_ptr; size_t size; };
struct GptOSS {
    int V; bf16 *d_wte, *d_rms1w, *d_qkvw, *d_ow, *d_sink_logits;
    bf16 *d_moe_win, *d_moe_bin, *d_moe_wout, *d_moe_bout, *d_moe_rw, *d_moe_rb;
    bf16 *d_rms2w, *d_rmsfw, *d_lm_head;
    float *d_dwte, *d_drms1w, *d_dqkvw, *d_dow, *d_dsink;
    float *d_dmoe_win, *d_dmoe_bin, *d_dmoe_wout, *d_dmoe_bout, *d_dmoe_rw, *d_dmoe_rb;
    float *d_drms2w, *d_drmsfw, *d_dlm_head, *d_m, *d_v_adam;
    
    float *d_muon_ws, *d_muon_ws2; 
    float *d_gram;                   
    float *d_norm_buf;               
    
    bf16 *d_encoded, *d_rms1; float *d_rms1_rstd; bf16 *d_qkv;
    bf16 *d_atty; float *d_att; bf16 *d_attproj, *d_residual2, *d_rms2; float *d_rms2_rstd;
    int *d_expert_offsets, *d_expert_write_ptr, *d_scattered_token_idx, *d_expert_pos;
    bf16 *d_scattered_token_w;
    bf16 *d_expert_inp, *d_expert_gate_up, *d_expert_swiglu, *d_expert_out, *d_expert_out_cache; 
    float *d_moe_out; 
    bf16 *d_router_weights;
    int *d_router_indices; bf16 *d_residual3, *d_rmsf; float *d_rmsf_rstd; 
    bf16 *d_logits; float *d_losses, *d_aux_loss;
    
    float *d_dencoded, *d_drms1, *d_dqkv;
    float *d_datty, *d_dattproj, *d_dresidual2, *d_drms2;
    float *d_dexpert_inp, *d_dexpert_gate_up, *d_dexpert_swiglu, *d_dexpert_out, *d_dmoe_out, *d_dresidual3;
    float *d_drmsf, *d_dlogits; int *d_inputs, *d_targets; size_t num_parameters;
    float *d_loss_mask;
};

std::vector<TensorDesc> get_param_list(GptOSS* m) {
    int qkv_dim = (NH + 2 * NKV) * DK;
    return { {m->d_wte, (size_t)m->V * C}, {m->d_rms1w, (size_t)L * C}, {m->d_qkvw, (size_t)L * C * qkv_dim},
        {m->d_ow, (size_t)L * NH * DK * C}, {m->d_sink_logits, (size_t)L * NH}, 
        {m->d_moe_win, (size_t)L * E * C * 2 * DFF}, {m->d_moe_bin, (size_t)L * E * 2 * DFF},
        {m->d_moe_wout, (size_t)L * E * DFF * C}, {m->d_moe_bout, (size_t)L * E * C}, {m->d_moe_rw, (size_t)L * C * E}, {m->d_moe_rb, (size_t)L * E},
        {m->d_rms2w, (size_t)L * C}, {m->d_rmsfw, (size_t)C}, {m->d_lm_head, (size_t)m->V * C} };
}
std::vector<GradDesc> get_grad_list(GptOSS* m) {
    int qkv_dim = (NH + 2 * NKV) * DK;
    return { {m->d_dwte, (size_t)m->V * C}, {m->d_drms1w, (size_t)L * C}, {m->d_dqkvw, (size_t)L * C * qkv_dim},
        {m->d_dow, (size_t)L * NH * DK * C}, {m->d_dsink, (size_t)L * NH}, 
        {m->d_dmoe_win, (size_t)L * E * C * 2 * DFF}, {m->d_dmoe_bin, (size_t)L * E * 2 * DFF},
        {m->d_dmoe_wout, (size_t)L * E * DFF * C}, {m->d_dmoe_bout, (size_t)L * E * C}, {m->d_dmoe_rw, (size_t)L * C * E}, {m->d_dmoe_rb, (size_t)L * E},
        {m->d_drms2w, (size_t)L * C}, {m->d_drmsfw, (size_t)C}, {m->d_dlm_head, (size_t)m->V * C} };
}
size_t calc_num_params(int V) { 
    int qkv_dim = (NH + 2 * NKV) * DK;
    return (size_t)(V*C) + L*(C + C*qkv_dim + NH*DK*C + NH) + L*(E*C*2*DFF + E*2*DFF + E*DFF*C + E*C + C*E + E) + L*C + C + V*C; 
}

size_t calc_num_1d_params() {
    size_t size1d = 0;
    size1d += L * C; 
    size1d += L * NH; 
    size1d += L * E * 2 * DFF; 
    size1d += L * E * C; 
    size1d += L * E; 
    size1d += L * C; 
    size1d += C; 
    return size1d;
}

void gpt_oss_alloc(GptOSS *model, int V) {
    model->V = V; model->num_parameters = calc_num_params(V);
    
    auto params = get_param_list(model); 
    for (auto& p : params) TRACKED_MALLOC(p.d_ptr, p.size * sizeof(bf16));
    model->d_wte=params[0].d_ptr; model->d_rms1w=params[1].d_ptr; model->d_qkvw=params[2].d_ptr; model->d_ow=params[3].d_ptr;
    model->d_sink_logits=params[4].d_ptr; model->d_moe_win=params[5].d_ptr; model->d_moe_bin=params[6].d_ptr;
    model->d_moe_wout=params[7].d_ptr; model->d_moe_bout=params[8].d_ptr; model->d_moe_rw=params[9].d_ptr;
    model->d_moe_rb=params[10].d_ptr; model->d_rms2w=params[11].d_ptr; model->d_rmsfw=params[12].d_ptr; model->d_lm_head=params[13].d_ptr;
    
    auto grads = get_grad_list(model); 
    for (auto& g : grads) TRACKED_MALLOC(g.d_ptr, g.size * sizeof(float));
    model->d_dwte=grads[0].d_ptr; model->d_drms1w=grads[1].d_ptr; model->d_dqkvw=grads[2].d_ptr; model->d_dow=grads[3].d_ptr;
    model->d_dsink=grads[4].d_ptr; model->d_dmoe_win=grads[5].d_ptr; model->d_dmoe_bin=grads[6].d_ptr;
    model->d_dmoe_wout=grads[7].d_ptr; model->d_dmoe_bout=grads[8].d_ptr; model->d_dmoe_rw=grads[9].d_ptr;
    model->d_dmoe_rb=grads[10].d_ptr; model->d_drms2w=grads[11].d_ptr; model->d_drmsfw=grads[12].d_ptr; model->d_dlm_head=grads[13].d_ptr;
    
    size_t num_1d_params = calc_num_1d_params();
    TRACKED_MALLOC(model->d_m, model->num_parameters * sizeof(float)); 
    TRACKED_MALLOC(model->d_v_adam, num_1d_params * sizeof(float)); 
    HIP_CHECK(hipMemset(model->d_m, 0, model->num_parameters * sizeof(float))); 
    HIP_CHECK(hipMemset(model->d_v_adam, 0, num_1d_params * sizeof(float)));

    size_t max_param_size = 0;
    for (auto& p : params) if (p.size > max_param_size) max_param_size = p.size;
    int max_cols = C; int qkv_dim_local = (NH + 2 * NKV) * DK;
    if (qkv_dim_local > max_cols) max_cols = qkv_dim_local;
    if (2 * DFF > max_cols) max_cols = 2 * DFF;

    TRACKED_MALLOC(model->d_muon_ws, max_param_size * sizeof(float));
    TRACKED_MALLOC(model->d_muon_ws2, max_param_size * sizeof(float));
    TRACKED_MALLOC(model->d_gram, max_cols * max_cols * sizeof(float));
    TRACKED_MALLOC(model->d_norm_buf, sizeof(float));

    int total_pairs = B * T * K;
    #define HIP_ALLOC_BF16(ptr, size) TRACKED_MALLOC(ptr, size * sizeof(bf16));
    #define HIP_ALLOC_F32(ptr, size) TRACKED_MALLOC(ptr, size * sizeof(float));
    HIP_ALLOC_BF16(model->d_encoded, B*T*C); HIP_ALLOC_BF16(model->d_rms1, L*B*T*C); HIP_ALLOC_F32(model->d_rms1_rstd, L*B*T);
    HIP_ALLOC_BF16(model->d_qkv, L*B*T*qkv_dim_local);
    HIP_ALLOC_BF16(model->d_atty, L*B*T*NH*DK); HIP_ALLOC_F32(model->d_att, L*B*NH*T*T); HIP_ALLOC_BF16(model->d_attproj, L*B*T*C);
    HIP_ALLOC_BF16(model->d_residual2, L*B*T*C); HIP_ALLOC_BF16(model->d_rms2, L*B*T*C); HIP_ALLOC_F32(model->d_rms2_rstd, L*B*T);
    
    HIP_ALLOC_BF16(model->d_expert_inp, L*total_pairs*C); 
    HIP_ALLOC_BF16(model->d_expert_gate_up, L*total_pairs*2*DFF); 
    HIP_ALLOC_BF16(model->d_expert_swiglu, L*total_pairs*DFF);
    HIP_ALLOC_BF16(model->d_expert_out, L*total_pairs*C); 
    HIP_ALLOC_BF16(model->d_expert_out_cache, L*total_pairs*C); 
    HIP_ALLOC_F32(model->d_moe_out, L*B*T*C); 
    HIP_ALLOC_BF16(model->d_router_weights, L*B*T*K);
    TRACKED_MALLOC(model->d_router_indices, L*B*T*K*sizeof(int));
    
    HIP_ALLOC_BF16(model->d_residual3, L*B*T*C); HIP_ALLOC_BF16(model->d_rmsf, B*T*C); HIP_ALLOC_F32(model->d_rmsf_rstd, B*T);
    HIP_ALLOC_BF16(model->d_logits, B*T*V); HIP_ALLOC_F32(model->d_losses, B*T); HIP_ALLOC_F32(model->d_aux_loss, 1);
    HIP_ALLOC_F32(model->d_dencoded, B*T*C); HIP_ALLOC_F32(model->d_drms1, L*B*T*C); HIP_ALLOC_F32(model->d_dqkv, L*B*T*qkv_dim_local);
    HIP_ALLOC_F32(model->d_datty, L*B*T*NH*DK); HIP_ALLOC_F32(model->d_dattproj, L*B*T*C); HIP_ALLOC_F32(model->d_dresidual2, L*B*T*C); HIP_ALLOC_F32(model->d_drms2, L*B*T*C);
    HIP_ALLOC_F32(model->d_dexpert_inp, L*total_pairs*C); HIP_ALLOC_F32(model->d_dexpert_gate_up, L*total_pairs*2*DFF); HIP_ALLOC_F32(model->d_dexpert_swiglu, L*total_pairs*DFF);
    HIP_ALLOC_F32(model->d_dexpert_out, L*total_pairs*C); 
    HIP_ALLOC_F32(model->d_dmoe_out, L*B*T*C); 
    HIP_ALLOC_F32(model->d_dresidual3, L*B*T*C);
    HIP_ALLOC_F32(model->d_drmsf, B*T*C); HIP_ALLOC_F32(model->d_dlogits, B*T*V);
    
    HIP_ALLOC_BF16(model->d_scattered_token_w, L*total_pairs);
    TRACKED_MALLOC(model->d_expert_offsets, L*(E+1)*sizeof(int));
    TRACKED_MALLOC(model->d_expert_write_ptr, L*E*sizeof(int));
    TRACKED_MALLOC(model->d_scattered_token_idx, L*total_pairs*sizeof(int));
    TRACKED_MALLOC(model->d_expert_pos, L*total_pairs*sizeof(int));
    #undef HIP_ALLOC_BF16
    #undef HIP_ALLOC_F32
    TRACKED_MALLOC(model->d_inputs, B * T * sizeof(int)); 
    TRACKED_MALLOC(model->d_targets, B * T * sizeof(int));
    TRACKED_MALLOC(model->d_loss_mask, B * T * sizeof(float));
}

void gpt_oss_free(GptOSS *model) {
    auto params = get_param_list(model); for (auto& p : params) HIP_CHECK(hipFree(p.d_ptr));
    auto grads = get_grad_list(model); for (auto& g : grads) HIP_CHECK(hipFree(g.d_ptr));
    HIP_CHECK(hipFree(model->d_m)); HIP_CHECK(hipFree(model->d_v_adam));
    HIP_CHECK(hipFree(model->d_muon_ws)); HIP_CHECK(hipFree(model->d_muon_ws2));
    HIP_CHECK(hipFree(model->d_gram)); HIP_CHECK(hipFree(model->d_norm_buf));
    #define HIP_FREE(ptr) if(ptr) HIP_CHECK(hipFree(ptr));
    HIP_FREE(model->d_encoded); HIP_FREE(model->d_rms1); HIP_FREE(model->d_rms1_rstd); HIP_FREE(model->d_qkv);
    HIP_FREE(model->d_atty); HIP_FREE(model->d_att); HIP_FREE(model->d_attproj); HIP_FREE(model->d_residual2);
    HIP_FREE(model->d_rms2); HIP_FREE(model->d_rms2_rstd); HIP_FREE(model->d_expert_inp); HIP_FREE(model->d_expert_gate_up);
    HIP_FREE(model->d_expert_swiglu); HIP_FREE(model->d_expert_out); HIP_FREE(model->d_expert_out_cache); HIP_FREE(model->d_moe_out); HIP_FREE(model->d_router_weights);
    HIP_FREE(model->d_router_indices); HIP_FREE(model->d_residual3); HIP_FREE(model->d_rmsf); HIP_FREE(model->d_rmsf_rstd);
    HIP_FREE(model->d_logits); HIP_FREE(model->d_losses); HIP_FREE(model->d_aux_loss);
    HIP_FREE(model->d_dencoded); HIP_FREE(model->d_drms1); HIP_FREE(model->d_dqkv);
    HIP_FREE(model->d_datty); HIP_FREE(model->d_dattproj); HIP_FREE(model->d_dresidual2); HIP_FREE(model->d_drms2);
    HIP_FREE(model->d_dexpert_inp); HIP_FREE(model->d_dexpert_gate_up); HIP_FREE(model->d_dexpert_swiglu); HIP_FREE(model->d_dexpert_out);
    HIP_FREE(model->d_dmoe_out); HIP_FREE(model->d_dresidual3); HIP_FREE(model->d_dlogits);
    HIP_FREE(model->d_scattered_token_w); HIP_FREE(model->d_expert_offsets); HIP_FREE(model->d_expert_write_ptr);
    HIP_FREE(model->d_scattered_token_idx); HIP_FREE(model->d_expert_pos);
    HIP_FREE(model->d_inputs); HIP_FREE(model->d_targets); HIP_FREE(model->d_loss_mask);
    #undef HIP_FREE
}

void gpt_oss_zero_grad(GptOSS *model) {
    auto grads = get_grad_list(model); for (auto& g : grads) HIP_CHECK(hipMemset(g.d_ptr, 0, g.size * sizeof(float)));
    #define HIP_MEMZERO_F32(ptr, size) HIP_CHECK(hipMemset(ptr, 0, size * sizeof(float)))
    int qkv_dim = (NH + 2 * NKV) * DK; int total_pairs = B * T * K;
    HIP_MEMZERO_F32(model->d_dencoded, B*T*C); HIP_MEMZERO_F32(model->d_drms1, L*B*T*C); 
    HIP_MEMZERO_F32(model->d_dqkv, L*B*T*qkv_dim); // <-- THE FIX
    HIP_MEMZERO_F32(model->d_datty, L*B*T*NH*DK); HIP_MEMZERO_F32(model->d_dattproj, L*B*T*C); HIP_MEMZERO_F32(model->d_dresidual2, L*B*T*C); HIP_MEMZERO_F32(model->d_drms2, L*B*T*C);
    HIP_MEMZERO_F32(model->d_dexpert_inp, L*total_pairs*C); HIP_MEMZERO_F32(model->d_dexpert_gate_up, L*total_pairs*2*DFF); HIP_MEMZERO_F32(model->d_dexpert_swiglu, L*total_pairs*DFF);
    HIP_MEMZERO_F32(model->d_dexpert_out, L*total_pairs*C); HIP_MEMZERO_F32(model->d_dmoe_out, L*B*T*C); HIP_MEMZERO_F32(model->d_dresidual3, L*B*T*C);
    HIP_MEMZERO_F32(model->d_drmsf, B*T*C); HIP_MEMZERO_F32(model->d_dsink, L*NH);
    HIP_MEMZERO_F32(model->d_dlogits, B*T*model->V);
    #undef HIP_MEMZERO_F32
}

void gpt_oss_save_checkpoint(GptOSS* model, int step, const char* filepath) {
    auto params = get_param_list(model);
    FILE* file = fopen(filepath, "wb");
    if(!file) { printf("Cannot open %s for writing\n", filepath); return; }
    fwrite(&step, sizeof(int), 1, file);
    
    std::vector<bf16> cpu_params(model->num_parameters);
    size_t offset = 0;
    for (auto& p : params) {
        HIP_CHECK(hipMemcpy(cpu_params.data() + offset, p.d_ptr, p.size * sizeof(bf16), hipMemcpyDeviceToHost));
        offset += p.size;
    }
    fwrite(cpu_params.data(), sizeof(bf16), model->num_parameters, file);
    
    std::vector<float> cpu_f32(model->num_parameters);
    HIP_CHECK(hipMemcpy(cpu_f32.data(), model->d_m, model->num_parameters * sizeof(float), hipMemcpyDeviceToHost));
    fwrite(cpu_f32.data(), sizeof(float), model->num_parameters, file);
    
    size_t num_1d = calc_num_1d_params();
    cpu_f32.resize(num_1d);
    HIP_CHECK(hipMemcpy(cpu_f32.data(), model->d_v_adam, num_1d * sizeof(float), hipMemcpyDeviceToHost));
    fwrite(cpu_f32.data(), sizeof(float), num_1d, file);
    
    fclose(file);
    printf("Checkpoint saved to %s at step %d\n", filepath, step);
}

int gpt_oss_load_checkpoint(GptOSS* model, const char* filepath) {
    FILE* file = fopen(filepath, "rb");
    if(!file) { printf("No checkpoint found at %s, starting fresh.\n", filepath); return 0; }
    int step; fread(&step, sizeof(int), 1, file);
    
    std::vector<bf16> cpu_params(model->num_parameters);
    auto params = get_param_list(model);
    fread(cpu_params.data(), sizeof(bf16), model->num_parameters, file);
    size_t offset = 0;
    for (auto& p : params) {
        HIP_CHECK(hipMemcpy(p.d_ptr, cpu_params.data() + offset, p.size * sizeof(bf16), hipMemcpyHostToDevice));
        offset += p.size;
    }
    
    std::vector<float> cpu_f32(model->num_parameters);
    fread(cpu_f32.data(), sizeof(float), model->num_parameters, file);
    HIP_CHECK(hipMemcpy(model->d_m, cpu_f32.data(), model->num_parameters * sizeof(float), hipMemcpyHostToDevice));
    
    size_t num_1d = calc_num_1d_params();
    cpu_f32.resize(num_1d);
    fread(cpu_f32.data(), sizeof(float), num_1d, file);
    HIP_CHECK(hipMemcpy(model->d_v_adam, cpu_f32.data(), num_1d * sizeof(float), hipMemcpyHostToDevice));
    fclose(file);
    printf("Checkpoint loaded from %s, resuming at step %d\n", filepath, step);
    return step;
}

#define LAUNCH(kernel, N, ...) do { int grid = (N + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL; kernel<<<grid, BLOCK_SIZE_KERNEL>>>(__VA_ARGS__); HIP_CHECK(hipGetLastError()); } while(0)
#define LAUNCH_2D(kernel, N, Y, ...) do { int grid = (N + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL; dim3 gd(grid, Y); kernel<<<gd, BLOCK_SIZE_KERNEL>>>(__VA_ARGS__); HIP_CHECK(hipGetLastError()); } while(0)

void gpt_oss_forward(GptOSS *model, const int* h_inputs, const int* h_targets, const float* h_loss_mask) {
    int V = model->V, qkv_dim = (NH + 2 * NKV) * DK, total_pairs = B * T * K;
    HIP_CHECK(hipMemcpy(model->d_inputs, h_inputs, B * T * sizeof(int), hipMemcpyHostToDevice));
    if (h_targets) HIP_CHECK(hipMemcpy(model->d_targets, h_targets, B * T * sizeof(int), hipMemcpyHostToDevice));
    if (h_loss_mask) HIP_CHECK(hipMemcpy(model->d_loss_mask, h_loss_mask, B * T * sizeof(float), hipMemcpyHostToDevice));
    LAUNCH(encoder_forward_kernel, B * T * C, model->d_encoded, model->d_inputs, model->d_wte, B, T, C);
    for (int l = 0; l < L; l++) {
        bf16* residual = (l == 0) ? model->d_encoded : model->d_residual3 + (l - 1) * B * T * C;
        int is_sliding = (l % 2 == 1);
        LAUNCH(rmsnorm_forward_kernel, B * T * C, model->d_rms1 + l*B*T*C, model->d_rms1_rstd + l*B*T, residual, model->d_rms1w + l*C, B, T, C);
        LAUNCH(matmul_forward_kernel, B * T * qkv_dim, model->d_qkv + l*B*T*qkv_dim, model->d_rms1 + l*B*T*C, model->d_qkvw + l*C*qkv_dim, NULL, B * T, C, qkv_dim);
        LAUNCH(rope_qk_forward_kernel, B * T * (NH + NKV) * DK, model->d_qkv + l*B*T*qkv_dim, B, T, NH, NKV, DK, ROPE_FACTOR);
        LAUNCH(gqa_sink_sliding_attention_forward_kernel, B * T * NH, model->d_atty + l*B*T*NH*DK, model->d_att + l*B*NH*T*T, model->d_qkv + l*B*T*qkv_dim, model->d_sink_logits + l*NH, B, T, NH, NKV, DK, NGRP, SLIDING_WINDOW, is_sliding);
        LAUNCH(matmul_forward_kernel, B * T * C, model->d_attproj + l*B*T*C, model->d_atty + l*B*T*C, model->d_ow + l*NH*DK*C, NULL, B * T, NH * DK, C);
        LAUNCH(residual_forward_bf16_kernel, B * T * C, model->d_residual2 + l*B*T*C, residual, model->d_attproj + l*B*T*C, B * T * C);
        LAUNCH(rmsnorm_forward_kernel, B * T * C, model->d_rms2 + l*B*T*C, model->d_rms2_rstd + l*B*T, model->d_residual2 + l*B*T*C, model->d_rms2w + l*C, B, T, C);
        HIP_CHECK(hipMemset(model->d_aux_loss, 0, sizeof(float)));
        LAUNCH(moe_router_forward_kernel, B * T, model->d_router_weights + l*B*T*K, model->d_router_indices + l*B*T*K, model->d_aux_loss, model->d_rms2 + l*B*T*C, model->d_moe_rw + l*C*E, model->d_moe_rb + l*E, B, T, C, E, K, AUX_COEF);
        LAUNCH(compute_expert_offsets_kernel, 1, model->d_expert_offsets + l*(E+1), model->d_router_indices + l*B*T*K, B, T, E, K);
        LAUNCH(zero_int_array_kernel, E, model->d_expert_write_ptr + l*E, E);
        LAUNCH(scatter_expert_info_kernel, total_pairs, model->d_scattered_token_idx + l*total_pairs, model->d_expert_pos + l*total_pairs, model->d_scattered_token_w + l*total_pairs, model->d_expert_write_ptr + l*E, model->d_expert_offsets + l*(E+1), model->d_router_indices + l*B*T*K, model->d_router_weights + l*B*T*K, B, T, K);
        LAUNCH(gather_tokens_kernel, total_pairs * C, model->d_expert_inp + l*total_pairs*C, model->d_rms2 + l*B*T*C, model->d_scattered_token_idx + l*total_pairs, total_pairs, C);
        LAUNCH_2D(grouped_matmul_forward_kernel, total_pairs * 2 * DFF, E, model->d_expert_gate_up + l*total_pairs*2*DFF, model->d_expert_inp + l*total_pairs*C, model->d_moe_win + l*E*C*2*DFF, model->d_moe_bin + l*E*2*DFF, model->d_expert_offsets + l*(E+1), E, total_pairs, C, 2 * DFF);
        LAUNCH(grouped_swiglu_forward_kernel, total_pairs * DFF, model->d_expert_swiglu + l*total_pairs*DFF, model->d_expert_gate_up + l*total_pairs*2*DFF, total_pairs, DFF, SWIGLU_CLIP);
        LAUNCH_2D(grouped_matmul_forward_kernel, total_pairs * C, E, model->d_expert_out + l*total_pairs*C, model->d_expert_swiglu + l*total_pairs*DFF, model->d_moe_wout + l*E*DFF*C, model->d_moe_bout + l*E*C, model->d_expert_offsets + l*(E+1), E, total_pairs, DFF, C);
        LAUNCH(cache_expert_out_kernel, B * T * K * C, model->d_expert_out_cache + l*total_pairs*C, model->d_expert_out + l*total_pairs*C, model->d_expert_pos + l*total_pairs, B, T, K, C);
        HIP_CHECK(hipMemset(model->d_moe_out + l*B*T*C, 0, B * T * C * sizeof(float)));
        LAUNCH(scatter_tokens_kernel, total_pairs * C, model->d_moe_out + l*B*T*C, model->d_expert_out + l*total_pairs*C, model->d_scattered_token_w + l*total_pairs, model->d_scattered_token_idx + l*total_pairs, total_pairs, C);
        LAUNCH(residual_forward_f32_kernel, B * T * C, model->d_residual3 + l*B*T*C, model->d_residual2 + l*B*T*C, model->d_moe_out + l*B*T*C, B * T * C);
    }
    bf16* residual = model->d_residual3 + (L - 1) * B * T * C;
    LAUNCH(rmsnorm_forward_kernel, B * T * C, model->d_rmsf, model->d_rmsf_rstd, residual, model->d_rmsfw, B, T, C);
    LAUNCH(matmul_forward_kernel, B * T * V, model->d_logits, model->d_rmsf, model->d_lm_head, NULL, B * T, C, V);
    if (h_targets) LAUNCH(crossentropy_forward_fused_kernel, B * T, model->d_losses, model->d_logits, model->d_targets, model->d_loss_mask, B, T, V);
}

void gpt_oss_backward(GptOSS *model) {
    int V = model->V; float dloss_val = 1.0f / (B * T);
    int qkv_dim = (NH + 2 * NKV) * DK, total_pairs = B * T * K;
    std::vector<float> h_dlosses(B * T, dloss_val); HIP_CHECK(hipMemcpy(model->d_losses, h_dlosses.data(), B * T * sizeof(float), hipMemcpyHostToDevice));
    LAUNCH(crossentropy_softmax_backward_fused_kernel, B * T, model->d_dlogits, model->d_losses, model->d_logits, model->d_targets, model->d_loss_mask, B, T, V);
    LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_drmsf, model->d_dlogits, model->d_lm_head, B * T, C, V);
    LAUNCH(matmul_backward_dweight_dbias_kernel, V * C, model->d_dlm_head, NULL, model->d_dlogits, model->d_rmsf, B * T, C, V);
    bf16* residual = model->d_residual3 + (L - 1) * B * T * C; float* dresidual = model->d_dresidual3 + (L - 1) * B * T * C;
    LAUNCH(rmsnorm_backward_kernel, B * T * C, dresidual, model->d_drmsfw, model->d_drmsf, residual, model->d_rmsfw, model->d_rmsf_rstd, B, T, C);
    for (int l = L - 1; l >= 0; l--) {
        residual = (l == 0) ? model->d_encoded : model->d_residual3 + (l - 1) * B * T * C;
        dresidual = (l == 0) ? model->d_dencoded : model->d_dresidual3 + (l - 1) * B * T * C;
        int is_sliding = (l % 2 == 1);
        LAUNCH(residual_backward_kernel, B * T * C, model->d_dresidual2 + l*B*T*C, model->d_dmoe_out + l*B*T*C, model->d_dresidual3 + l*B*T*C, B * T * C);
        LAUNCH(scatter_tokens_backward_kernel, total_pairs * C, model->d_dexpert_out + l*total_pairs*C, model->d_dmoe_out + l*B*T*C, model->d_scattered_token_w + l*total_pairs, model->d_scattered_token_idx + l*total_pairs, total_pairs, C);
        LAUNCH_2D(grouped_matmul_backward_dinp_kernel, total_pairs * DFF, E, model->d_dexpert_swiglu + l*total_pairs*DFF, model->d_dexpert_out + l*total_pairs*C, model->d_moe_wout + l*E*DFF*C, model->d_expert_offsets + l*(E+1), E, total_pairs, DFF, C);
        LAUNCH_2D(grouped_matmul_backward_dweight_kernel, 2 * DFF * C, E, model->d_dmoe_wout + l*E*DFF*C, model->d_dmoe_bout + l*E*C, model->d_dexpert_out + l*total_pairs*C, model->d_expert_swiglu + l*total_pairs*DFF, model->d_expert_offsets + l*(E+1), E, total_pairs, DFF, C);
        LAUNCH(grouped_swiglu_backward_kernel, total_pairs * DFF, model->d_dexpert_gate_up + l*total_pairs*2*DFF, model->d_expert_gate_up + l*total_pairs*2*DFF, model->d_dexpert_swiglu + l*total_pairs*DFF, total_pairs, DFF, SWIGLU_CLIP);
        LAUNCH_2D(grouped_matmul_backward_dinp_kernel, total_pairs * C, E, model->d_dexpert_inp + l*total_pairs*C, model->d_dexpert_gate_up + l*total_pairs*2*DFF, model->d_moe_win + l*E*C*2*DFF, model->d_expert_offsets + l*(E+1), E, total_pairs, C, 2 * DFF);
        LAUNCH_2D(grouped_matmul_backward_dweight_kernel, C * 2 * DFF, E, model->d_dmoe_win + l*E*C*2*DFF, model->d_dmoe_bin + l*E*2*DFF, model->d_dexpert_gate_up + l*total_pairs*2*DFF, model->d_expert_inp + l*total_pairs*C, model->d_expert_offsets + l*(E+1), E, total_pairs, C, 2 * DFF);
        LAUNCH(gather_tokens_backward_kernel, total_pairs * C, model->d_drms2 + l*B*T*C, model->d_dexpert_inp + l*total_pairs*C, model->d_scattered_token_idx + l*total_pairs, total_pairs, C);
        LAUNCH(moe_router_backward_kernel, B * T, model->d_drms2 + l*B*T*C, model->d_dmoe_rw + l*C*E, model->d_dmoe_rb + l*E, model->d_expert_out_cache + l*total_pairs*C, model->d_dmoe_out + l*B*T*C, model->d_rms2 + l*B*T*C, model->d_moe_rw + l*C*E, model->d_router_weights + l*B*T*K, model->d_router_indices + l*B*T*K, B, T, C, E, K);
        LAUNCH(rmsnorm_backward_kernel, B * T * C, model->d_dresidual2 + l*B*T*C, model->d_drms2w + l*C, model->d_drms2 + l*B*T*C, model->d_residual2 + l*B*T*C, model->d_rms2w + l*C, model->d_rms2_rstd + l*B*T, B, T, C);
        LAUNCH(residual_backward_kernel, B * T * C, dresidual, model->d_dattproj + l*B*T*C, model->d_dresidual2 + l*B*T*C, B * T * C);
        LAUNCH(matmul_backward_dinp_kernel, B * T * NH * DK, model->d_datty + l*B*T*NH*DK, model->d_dattproj + l*B*T*C, model->d_ow + l*NH*DK*C, B * T, NH * DK, C);
        LAUNCH(matmul_backward_dweight_dbias_kernel, NH * DK * C, model->d_dow + l*NH*DK*C, NULL, model->d_dattproj + l*B*T*C, model->d_atty + l*B*T*C, B * T, NH * DK, C);
        LAUNCH(gqa_sink_sliding_attention_backward_kernel, B * T * NH, model->d_dqkv + l*B*T*qkv_dim, model->d_dsink + l*NH, model->d_datty + l*B*T*C, model->d_qkv + l*B*T*qkv_dim, model->d_att + l*B*NH*T*T, B, T, NH, NKV, DK, NGRP, SLIDING_WINDOW, is_sliding);
        LAUNCH(rope_qk_backward_kernel, B * T * (NH + NKV) * DK, model->d_dqkv + l*B*T*qkv_dim, B, T, NH, NKV, DK, ROPE_FACTOR);
        LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_drms1 + l*B*T*C, model->d_dqkv + l*B*T*qkv_dim, model->d_qkvw + l*C*qkv_dim, B * T, C, qkv_dim);
        LAUNCH(matmul_backward_dweight_dbias_kernel, qkv_dim * C, model->d_dqkvw + l*C*qkv_dim, NULL, model->d_dqkv + l*B*T*qkv_dim, model->d_rms1 + l*B*T*C, B * T, C, qkv_dim);
        LAUNCH(rmsnorm_backward_kernel, B * T * C, dresidual, model->d_drms1w + l*C, model->d_drms1 + l*B*T*C, residual, model->d_rms1w + l*C, model->d_rms1_rstd + l*B*T, B, T, C);
    }
    LAUNCH(encoder_backward_kernel, B * T * C, model->d_dwte, NULL, model->d_dencoded, model->d_inputs, B, T, C);
}

void gpt_oss_update(GptOSS *model, float learning_rate, int step) {
    int V = model->V, qkv_dim = (NH + 2 * NKV) * DK;
    struct P { bf16* ptr; float* grad; size_t size; int rows; int cols; bool is2d; };
    std::vector<P> params = {
        {model->d_wte, model->d_dwte, (size_t)V*C, V, C, true}, {model->d_rms1w, model->d_drms1w, (size_t)L*C, 0, 0, false},
        {model->d_qkvw, model->d_dqkvw, (size_t)L*C*qkv_dim, L*C, qkv_dim, true}, {model->d_ow, model->d_dow, (size_t)L*NH*DK*C, L*NH*DK, C, true},
        {model->d_sink_logits, model->d_dsink, (size_t)L*NH, 0, 0, false}, {model->d_moe_win, model->d_dmoe_win, (size_t)L*E*C*2*DFF, L*E*C, 2*DFF, true},
        {model->d_moe_bin, model->d_dmoe_bin, (size_t)L*E*2*DFF, 0, 0, false}, {model->d_moe_wout, model->d_dmoe_wout, (size_t)L*E*DFF*C, L*E*DFF, C, true},
        {model->d_moe_bout, model->d_dmoe_bout, (size_t)L*E*C, 0, 0, false}, {model->d_moe_rw, model->d_dmoe_rw, (size_t)L*C*E, L*C, E, true},
        {model->d_moe_rb, model->d_dmoe_rb, (size_t)L*E, 0, 0, false}, {model->d_rms2w, model->d_drms2w, (size_t)L*C, 0, 0, false},
        {model->d_rmsfw, model->d_drmsfw, (size_t)C, 0, 0, false}, {model->d_lm_head, model->d_dlm_head, (size_t)V*C, V, C, true},
    };
    size_t m_offset = 0, v_offset = 0;
    for (auto& p : params) {
        if (p.size == 0) { m_offset += p.size; continue; }
        int grid = (p.size + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL;
        if (p.is2d) {
            int M = p.rows, N = p.cols;
            muon_momentum_kernel<<<grid, BLOCK_SIZE_KERNEL>>>(model->d_m + m_offset, p.grad, p.size, MUON_BETA);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipMemcpy(model->d_muon_ws, model->d_m + m_offset, p.size * sizeof(float), hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemset(model->d_norm_buf, 0, sizeof(float)));
            size_t shared_mem = BLOCK_SIZE_KERNEL * sizeof(float);
            frobenius_norm_sq_kernel<<<grid, BLOCK_SIZE_KERNEL, shared_mem>>>(model->d_norm_buf, model->d_muon_ws, p.size);
            HIP_CHECK(hipGetLastError());
            float h_norm_sq; HIP_CHECK(hipMemcpy(&h_norm_sq, model->d_norm_buf, sizeof(float), hipMemcpyDeviceToHost));
            scale_by_norm_kernel<<<grid, BLOCK_SIZE_KERNEL>>>(model->d_muon_ws, h_norm_sq, p.size);
            HIP_CHECK(hipGetLastError());
            float* ws[2] = {model->d_muon_ws, model->d_muon_ws2};
            int cur = 0, gram_grid = (N * N + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL;
            for (int iter = 0; iter < NS_ITERS; iter++) {
                int next = 1 - cur;
                gram_matrix_kernel<<<gram_grid, BLOCK_SIZE_KERNEL>>>(model->d_gram, ws[cur], M, N);
                HIP_CHECK(hipGetLastError());
                ns_step_kernel<<<grid, BLOCK_SIZE_KERNEL>>>(ws[next], ws[cur], model->d_gram, M, N);
                HIP_CHECK(hipGetLastError());
                cur = next;
            }
            float scaling = sqrtf(fmaxf(1.0f, (float)M / (float)N));
            muon_apply_kernel<<<grid, BLOCK_SIZE_KERNEL>>>(p.ptr, ws[cur], p.size, MUON_LR, 0.1f, scaling);
            HIP_CHECK(hipGetLastError());
        } else {
            adamw_kernel<<<grid, BLOCK_SIZE_KERNEL>>>(p.ptr, p.grad, model->d_m + m_offset, model->d_v_adam + v_offset, p.size, learning_rate, 0.9f, 0.999f, 1e-8f, 0.1f, step);
            HIP_CHECK(hipGetLastError());
            v_offset += p.size;
        }
        m_offset += p.size;
    }
}

void gpt_oss_generate(GptOSS *model, CharTokenizer& tokenizer, const std::string& prompt, int max_new_tokens) {
    std::vector<int> tokens = tokenizer.encode(prompt);
    if (tokens.size() > T) tokens = std::vector<int>(tokens.end() - T, tokens.end());
    
    std::vector<int> h_inputs(B * T, 0);
    for (int i = 0; i < tokens.size(); i++) h_inputs[i] = tokens[i];

    printf("%s", prompt.c_str());
    for (int i = 0; i < max_new_tokens; i++) {
        for (int j = 0; j < tokens.size(); j++) h_inputs[j] = tokens[j];

        gpt_oss_forward(model, h_inputs.data(), nullptr, nullptr);
        
        std::vector<bf16> h_logits(B * T * model->V);
        HIP_CHECK(hipMemcpy(h_logits.data(), model->d_logits, B * T * model->V * sizeof(bf16), hipMemcpyDeviceToHost));
        
        int last_token_idx = tokens.size() - 1;
        bf16* l_bt = h_logits.data() + last_token_idx * model->V;
        
        int max_idx = 0;
        float max_val = -1e10f;
        for (int k = 0; k < model->V; k++) {
            float v = bf16_to_float(l_bt[k]);
            if (v > max_val) { max_val = v; max_idx = k; }
        }
        tokens.push_back(max_idx);
        printf("%c", tokenizer.itos[max_idx]);
    }
    printf("\n");
}

// ============================================================
// Main
// ============================================================
int main() {
    std::ifstream f("input.txt");
    if (!f) { printf("input.txt not found!\n"); return 1; }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    
    CharTokenizer tokenizer;
    tokenizer.init(text);
    int V = tokenizer.vocab_size;
    printf("Vocab size: %d\n", V);

    GptOSS model;
    printf("Allocating model with ~%zuM parameters...\n", calc_num_params(V) / 1000000);
    gpt_oss_alloc(&model, V);
    printf("Model allocated. VRAM used: %.2f GB\n", (float)g_vram_used / 1e9);

    std::mt19937 rng(42); std::normal_distribution<float> dist(0.0f, 0.02f);
    auto init_tensor = [&](bf16* d_ptr, size_t size) {
        std::vector<bf16> buf(size);
        for (auto& v : buf) v = float_to_bf16(dist(rng));
        HIP_CHECK(hipMemcpy(d_ptr, buf.data(), size * sizeof(bf16), hipMemcpyHostToDevice));
    };
    auto params = get_param_list(&model);
    for (auto& p : params) if (p.size > 0) init_tensor(p.d_ptr, p.size);
    HIP_CHECK(hipMemset(model.d_moe_bin, 0, L * E * 2 * DFF * sizeof(bf16)));
    HIP_CHECK(hipMemset(model.d_moe_bout, 0, L * E * C * sizeof(bf16)));
    HIP_CHECK(hipMemset(model.d_moe_rb, 0, L * E * sizeof(bf16)));
    HIP_CHECK(hipMemset(model.d_sink_logits, 0, L * NH * sizeof(bf16)));
    {
        std::vector<bf16> h_sink(L * NH);
        for(auto& v : h_sink) v = float_to_bf16(SINK_INIT);
        HIP_CHECK(hipMemcpy(model.d_sink_logits, h_sink.data(), L * NH * sizeof(bf16), hipMemcpyHostToDevice));
    }

    int start_step = gpt_oss_load_checkpoint(&model, CKPT_PATH);
    std::vector<int> h_inputs(B * T); std::vector<int> h_targets(B * T); std::vector<float> h_loss_mask(B * T, 1.0f); 
    std::vector<int> enc = tokenizer.encode(text);

    for (int step = start_step; step < MAX_ITERS; step++) {
        for (int b = 0; b < B; b++) {
            int start_idx = rand() % (enc.size() - T - 1);
            for (int t = 0; t < T; t++) {
                h_inputs[b * T + t] = enc[start_idx + t];
                h_targets[b * T + t] = enc[start_idx + t + 1];
            }
        }
        
        gpt_oss_zero_grad(&model);
        gpt_oss_forward(&model, h_inputs.data(), h_targets.data(), h_loss_mask.data());
        
        if (step % 10 == 0) {
            std::vector<float> losses(B * T);
            HIP_CHECK(hipMemcpy(losses.data(), model.d_losses, B * T * sizeof(float), hipMemcpyDeviceToHost));
            float avg_loss = 0.0f;
            for (float l : losses) avg_loss += l;
            avg_loss /= (B * T);
            printf("Step %d | Loss: %.4f\n", step, avg_loss);
        }
        
        gpt_oss_backward(&model);
        gpt_oss_update(&model, LR, step + 1);

        if (step > 0 && step % EVAL_INTERVAL == 0) {
            gpt_oss_save_checkpoint(&model, step, CKPT_PATH);
            gpt_oss_generate(&model, tokenizer, "Once upon a time", 50);
        }
    }
    gpt_oss_free(&model);
    return 0;
}