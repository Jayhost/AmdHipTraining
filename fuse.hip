// train_gpt_oss_hip.cpp
// AMD HIP GPU implementation of GPT-OSS (Sparse Grouped MoE, Fused QKV, Fused RoPE, GQA, Sink-Bias)
// Compile: hipcc -O3 train_gpt_oss_hip.cpp -o train_gpt_oss_hip

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

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// GPT-OSS Small Test Hyperparameters
// ============================================================
const int B = 16;         // batch_size
const int T = 32;         // max_seq_len
const int MAX_ITERS = 1000;
const int EVAL_INTERVAL = 100;
const float LR = 1e-3f;
const int EVAL_ITERS = 200;
const int C = 128;        // d_model (hidden_dim)
const int NH = 4;         // n_heads (Query heads)
const int NKV = 2;        // n_kv_heads (GQA: 2 KV heads shared by 4 Q heads)
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
const char* CKPT_PATH = "gpt_oss_hip_checkpoint.bin";

// ============================================================
// Tokenizer
// ============================================================
struct CharTokenizer {
    std::map<char, int> stoi;
    std::map<int, char> itos;
    int vocab_size;
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
// HIP Kernels - Forward Pass
// ============================================================
__global__ void encoder_forward_kernel(float* out, const int* inp, const float* wte, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * C) return;
    out[idx] = wte[inp[idx / C] * C + (idx % C)];
}

__global__ void rmsnorm_forward_kernel(float* out, float* rstd, const float* inp, const float* weight, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * C) return;
    int c = idx % C, bt = idx / C; const float* x = inp + bt * C;
    float ss = 0.0f; for (int i = 0; i < C; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / C + RMS_EPS); out[idx] = s * x[c] * weight[c];
    if (c == 0) rstd[bt] = s;
}

__global__ void matmul_forward_kernel(float* out, const float* inp, const float* weight, const float* bias, int total_bt, int C, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_bt * OC) return;
    int o = idx % OC, bt = idx / OC; float val = (bias != NULL) ? bias[o] : 0.0f;
    const float* inp_bt = inp + bt * C; const float* wrow = weight + o * C;
    for (int i = 0; i < C; i++) val += inp_bt[i] * wrow[i]; out[idx] = val;
}

// Fused RoPE applied in-place on the QKV buffer
__global__ void rope_qk_forward_kernel(float* qkv, int B, int T, int NH, int NKV, int d_k, float rope_factor) {
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
    
    // Calculate offset in the interleaved [B, T, QKV] buffer
    int qkv_dim = (NH + 2 * NKV) * d_k;
    int bt_base = (b * T + t) * qkv_dim;
    int offset = is_q ? (h * d_k) : (NH * d_k + h * d_k);
    int target_idx = bt_base + offset + i;
    
    float x1 = qkv[target_idx], x2 = qkv[target_idx + 1];
    qkv[target_idx]     = x1 * cos_val - x2 * sin_val;
    qkv[target_idx + 1] = x1 * sin_val + x2 * cos_val;
}

__global__ void gqa_sink_sliding_attention_forward_kernel(
    float* out, float* att, const float* qkv, const float* sink_logits,
    int B, int T, int NH, int NKV, int DK, int NGRP, int sliding_window, int is_sliding_layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * NH) return;
    int h = idx % NH, t = (idx / NH) % T, b = idx / (T * NH), kv_h = h / NGRP;
    float scale = 1.0f / sqrtf((float)DK), sink_logit = sink_logits[h];
    
    int qkv_dim = (NH + 2 * NKV) * DK;
    int q_offset = NH * DK;
    int kv_offset = NKV * DK;
    
    const float* q_bth = qkv + b * T * qkv_dim + t * qkv_dim + h * DK;
    const float* k_base = qkv + b * T * qkv_dim + q_offset;
    const float* v_base = qkv + b * T * qkv_dim + q_offset + kv_offset;
    float* att_bth = att + b * NH * T * T + h * T * T + t * T;
    
    float maxval = sink_logit; 
    for (int t2 = 0; t2 <= t; t2++) {
        bool mask_ok = !(is_sliding_layer && (t - t2) >= sliding_window);
        if (mask_ok) {
            const float* key = k_base + t2 * qkv_dim + kv_h * DK; float val = 0.0f;
            for (int i = 0; i < DK; i++) val += q_bth[i] * key[i]; val *= scale;
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

    float* out_bth = out + b * T * NH * DK + t * NH * DK + h * DK;
    for (int i = 0; i < DK; i++) out_bth[i] = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        att_bth[t2] *= inv_sum; 
        if (att_bth[t2] > 0.0f) {
            const float* value = v_base + t2 * qkv_dim + kv_h * DK; float a = att_bth[t2];
            for (int i = 0; i < DK; i++) out_bth[i] += a * value[i];
        }
    }
}

__global__ void moe_router_forward_kernel(
    float* weights_out, int* indices_out, float* aux_loss_out,
    const float* inp, const float* router_w, const float* router_b, int B, int T, int C, int E, int K, float aux_coef) {
    int s = blockIdx.x * blockDim.x + threadIdx.x; if (s >= B * T) return;
    float logits[16]; 
    for (int e = 0; e < E; e++) {
        float val = router_b[e]; for (int c = 0; c < C; c++) val += inp[s * C + c] * router_w[c * E + e]; logits[e] = val;
    }
    float topk_vals[8]; int topk_idxs[8]; 
    for (int k = 0; k < K; k++) { topk_vals[k] = -1e9f; topk_idxs[k] = 0; }
    for (int e = 0; e < E; e++) { for (int k = 0; k < K; k++) { if (logits[e] > topk_vals[k]) {
        for (int j = K - 1; j > k; j--) { topk_vals[j] = topk_vals[j-1]; topk_idxs[j] = topk_idxs[j-1]; }
        topk_vals[k] = logits[e]; topk_idxs[k] = e; break;
    }}}
    float max_val = topk_vals[0], sum_exp = 0.0f;
    for (int k = 0; k < K; k++) { topk_vals[k] = expf(topk_vals[k] - max_val); sum_exp += topk_vals[k]; }
    for (int k = 0; k < K; k++) { topk_vals[k] /= sum_exp; weights_out[s * K + k] = topk_vals[k]; indices_out[s * K + k] = topk_idxs[k]; }
    atomicAdd(aux_loss_out, topk_vals[0] / (B * T));
}

// --- SPARSE MOE PIPELINE ---
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
    int* scattered_token_idx, int* expert_pos, float* scattered_token_w, int* expert_write_ptr,
    const int* expert_offsets, const int* indices, const float* weights, int B, int T, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * K) return;
    int s = idx / K, k = idx % K, e = indices[idx];
    int pos = atomicAdd(&expert_write_ptr[e], 1) + expert_offsets[e];
    scattered_token_idx[pos] = s;
    scattered_token_w[pos] = weights[idx];
    expert_pos[idx] = pos; 
}

__global__ void gather_tokens_kernel(float* expert_inp, const float* inp, const int* scattered_token_idx, int total_pairs, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_pairs * C) return;
    int c = idx % C, pos = idx / C;
    expert_inp[pos * C + c] = inp[scattered_token_idx[pos] * C + c];
}

__global__ void grouped_matmul_forward_kernel(float* out, const float* inp, const float* weight, const float* bias, const int* expert_offsets, int E, int max_pairs, int C, int OC) {
    int e = blockIdx.y;
    int start_pos = expert_offsets[e];
    int end_pos = expert_offsets[e+1];
    int num_tokens = end_pos - start_pos;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * OC) return;
    
    int o = idx % OC;
    int local_pos = idx / OC;
    int global_pos = start_pos + local_pos;
    
    float val = bias[e * OC + o];
    const float* inp_pos = inp + global_pos * C;
    const float* wrow = weight + e * OC * C + o * C;
    for (int i = 0; i < C; i++) val += inp_pos[i] * wrow[i];
    out[global_pos * OC + o] = val;
}

__global__ void grouped_swiglu_forward_kernel(float* out, float* gate_up, int total_pairs, int DFF, float clip) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_pairs * DFF) return;
    int j = idx % DFF, pos = idx / DFF;
    float g = gate_up[pos * 2 * DFF + j];
    float u = gate_up[pos * 2 * DFF + DFF + j];
    g = (clip > 0) ? fmaxf(-clip, fminf(clip, g)) : g;
    u = (clip > 0) ? fmaxf(-clip, fminf(clip, u)) : u;
    float silu = g / (1.0f + expf(-g));
    out[idx] = silu * u;
}

__global__ void cache_expert_out_kernel(float* expert_out_cache, const float* expert_out, const int* expert_pos, int B, int T, int K, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * K * C) return;
    int c = idx % C, k = (idx / C) % K, s = idx / (C * K);
    int pos = expert_pos[s * K + k];
    expert_out_cache[idx] = expert_out[pos * C + c];
}

__global__ void scatter_tokens_kernel(float* out, const float* expert_out, const float* scattered_token_w, const int* scattered_token_idx, int total_pairs, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_pairs * C) return;
    int c = idx % C, pos = idx / C;
    int s = scattered_token_idx[pos];
    atomicAdd(&out[s * C + c], scattered_token_w[pos] * expert_out[pos * C + c]);
}

__global__ void residual_forward_kernel(float* out, const float* inp1, const float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= N) return; out[idx] = inp1[idx] + inp2[idx];
}
__global__ void softmax_forward_kernel(float* probs, const float* logits, int B, int T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T) return;
    const float* l_bt = logits + idx * V; float* p_bt = probs + idx * V; float maxval = -1e10f;
    for (int i = 0; i < V; i++) if (l_bt[i] > maxval) maxval = l_bt[i];
    float sum = 0.0f; for (int i = 0; i < V; i++) { p_bt[i] = expf(l_bt[i] - maxval); sum += p_bt[i]; }
    for (int i = 0; i < V; i++) p_bt[i] /= sum;
}
__global__ void crossentropy_forward_kernel(float* losses, const float* probs, const int* targets, int B, int T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T) return;
    losses[idx] = -logf(probs[idx * V + targets[idx]] + 1e-10f);
}

// ============================================================
// HIP Kernels - Backward Pass
// ============================================================
__global__ void encoder_backward_kernel(float* dwte, float* dwpe, const float* dout, const int* inp, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * C) return;
    atomicAdd(&dwte[inp[idx / C] * C + (idx % C)], dout[idx]);
}

__global__ void rmsnorm_backward_kernel(float* dinp, float* dweight, const float* dout, const float* inp, const float* weight, const float* rstd, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * C) return;
    int c = idx % C, bt = idx / C; float s_bt = rstd[bt];
    const float* dout_bt = dout + bt * C, * inp_bt = inp + bt * C;
    float dnorm_sum = 0.0f; for (int i = 0; i < C; i++) dnorm_sum += weight[i] * dout_bt[i] * inp_bt[i];
    dnorm_sum = dnorm_sum * s_bt * s_bt / C;
    atomicAdd(&dweight[c], inp_bt[c] * s_bt * dout_bt[c]);
    dinp[bt * C + c] += s_bt * (weight[c] * dout_bt[c] - dnorm_sum * inp_bt[c]);
}

__global__ void matmul_backward_dinp_kernel(float* dinp, const float* dout, const float* weight, int total_bt, int C, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_bt * C) return;
    int i = idx % C, bt = idx / C; const float* dout_bt = dout + bt * OC; float val = 0.0f;
    for (int o = 0; o < OC; o++) val += dout_bt[o] * weight[o * C + i]; dinp[idx] += val;
}

__global__ void matmul_backward_dweight_dbias_kernel(float* dweight, float* dbias, const float* dout, const float* inp, int total_bt, int C, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= OC * C) return;
    int i = idx % C, o = idx / C; float dw = 0.0f;
    for (int bt = 0; bt < total_bt; bt++) dw += dout[bt * OC + o] * inp[bt * C + i];
    dweight[idx] += dw; 
    if (i == 0 && dbias != NULL) {
        float db = 0.0f;
        for (int bt = 0; bt < total_bt; bt++) db += dout[bt * OC + o];
        atomicAdd(&dbias[o], db); 
    }
}

// Fused RoPE Backward applied in-place on the dQKV buffer
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
    // Use assignment `=` instead of `+=` because the attention backward already accumulated 
    // the post-RoPE gradients here. We are overwriting them with the pre-RoPE gradients.
    dqkv[target_idx]     = dy1 * cos_val + dy2 * sin_val;
    dqkv[target_idx + 1] = -dy1 * sin_val + dy2 * cos_val;
}

__global__ void gqa_sink_sliding_attention_backward_kernel(
    float* dqkv, float* d_sink_logits,
    const float* dout, const float* qkv, const float* att,
    int B, int T, int NH, int NKV, int DK, int NGRP, int sliding_window, int is_sliding_layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T * NH) return;
    int h = idx % NH, t = (idx / NH) % T, b = idx / (T * NH), kv_h = h / NGRP;
    float scale = 1.0f / sqrtf((float)DK);
    
    int qkv_dim = (NH + 2 * NKV) * DK;
    int q_offset = NH * DK;
    int kv_offset = NKV * DK;
    
    const float* att_bth = att + b * NH * T * T + h * T * T + t * T;
    const float* dout_bth = dout + b * T * NH * DK + t * NH * DK + h * DK;
    
    const float* q_bth = qkv + b * T * qkv_dim + t * qkv_dim + h * DK; 
    float* dq_bth = dqkv + b * T * qkv_dim + t * qkv_dim + h * DK; 
    
    const float* k_base = qkv + b * T * qkv_dim + q_offset;
    const float* v_base = qkv + b * T * qkv_dim + q_offset + kv_offset;
    float* dk_base = dqkv + b * T * qkv_dim + q_offset;
    float* dv_base = dqkv + b * T * qkv_dim + q_offset + kv_offset;

    float d_att[32]; 
    float dot_p_dout = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        float p = att_bth[t2]; 
        if (p <= 0.0f) { d_att[t2] = 0.0f; continue; }
        const float* value_t2 = v_base + t2 * qkv_dim + kv_h * DK; 
        float* dvalue_t2 = dv_base + t2 * qkv_dim + kv_h * DK;
        float d_att_t2 = 0.0f;
        for (int i = 0; i < DK; i++) { 
            d_att_t2 += value_t2[i] * dout_bth[i]; 
            atomicAdd(&dvalue_t2[i], p * dout_bth[i]); 
        }
        d_att[t2] = d_att_t2;
        dot_p_dout += p * d_att_t2;
    }

    for (int t2 = 0; t2 <= t; t2++) {
        float p = att_bth[t2]; 
        if (p <= 0.0f) continue;
        float d_att_t2 = d_att[t2];
        float dpreatt_t2 = p * (d_att_t2 - dot_p_dout) * scale;
        const float* key_t2 = k_base + t2 * qkv_dim + kv_h * DK; 
        float* dkey_t2 = dk_base + t2 * qkv_dim + kv_h * DK;
        for (int i = 0; i < DK; i++) { 
            dq_bth[i] += key_t2[i] * dpreatt_t2; 
            atomicAdd(&dkey_t2[i], q_bth[i] * dpreatt_t2); 
        }
    }
    float p_sum = 0.0f; for (int t2 = 0; t2 <= t; t2++) p_sum += att_bth[t2];
    float p_sink = 1.0f - p_sum; 
    if (p_sink > 0.0f) atomicAdd(&d_sink_logits[h], -p_sink * dot_p_dout);
}

// --- SPARSE MOE BACKWARD PIPELINE ---
__global__ void scatter_tokens_backward_kernel(float* d_expert_out, const float* dout, const float* scattered_token_w, const int* scattered_token_idx, int total_pairs, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_pairs * C) return;
    int c = idx % C, pos = idx / C;
    int s = scattered_token_idx[pos];
    d_expert_out[pos * C + c] = scattered_token_w[pos] * dout[s * C + c];
}

__global__ void grouped_matmul_backward_dinp_kernel(float* dinp, const float* dout, const float* weight, const int* expert_offsets, int E, int max_pairs, int C, int OC) {
    int e = blockIdx.y;
    int start_pos = expert_offsets[e];
    int num_tokens = expert_offsets[e+1] - start_pos;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * C) return;
    
    int i = idx % C;
    int local_pos = idx / C;
    int global_pos = start_pos + local_pos;
    
    const float* dout_pos = dout + global_pos * OC;
    const float* wrow = weight + e * OC * C;
    float val = 0.0f;
    for (int o = 0; o < OC; o++) val += dout_pos[o] * wrow[o * C + i];
    dinp[global_pos * C + i] += val;
}

__global__ void grouped_matmul_backward_dweight_kernel(float* dweight, float* dbias, const float* dout, const float* inp, const int* expert_offsets, int E, int max_pairs, int C, int OC) {
    int e = blockIdx.y;
    int start_pos = expert_offsets[e];
    int num_tokens = expert_offsets[e+1] - start_pos;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= OC * C) return;
    
    int i = idx % C;
    int o = idx / C;
    
    float dw = 0.0f;
    for (int t = 0; t < num_tokens; t++) {
        int global_pos = start_pos + t;
        dw += dout[global_pos * OC + o] * inp[global_pos * C + i];
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

__global__ void grouped_swiglu_backward_kernel(float* d_gate_up, const float* gate_up, const float* d_swiglu, int total_pairs, int DFF, float clip) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total_pairs * DFF) return;
    int j = idx % DFF, pos = idx / DFF;
    float g = gate_up[pos * 2 * DFF + j];
    float u = gate_up[pos * 2 * DFF + DFF + j];
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
    const float* expert_out_cache, const float* dout, const float* inp, 
    const float* router_w, const float* weights, const int* indices,
    int B, int T, int C, int E, int K) {
    int s = blockIdx.x * blockDim.x + threadIdx.x; if (s >= B * T) return;
    float* d_inp_s = d_inp + s * C;
    const float* dout_s = dout + s * C;
    
    float dps[8];
    for (int k = 0; k < K; k++) {
        const float* exp_out_s = expert_out_cache + s * K * C + k * C;
        float dp = 0.0f;
        for (int c = 0; c < C; c++) dp += dout_s[c] * exp_out_s[c];
        dps[k] = dp;
    }
    
    float sum_dp_w = 0.0f; for(int k=0; k<K; k++) sum_dp_w += weights[s * K + k] * dps[k];
    for(int k=0; k<K; k++) {
        int e = indices[s * K + k]; 
        float d_logit = weights[s * K + k] * (dps[k] - sum_dp_w);
        for(int c=0; c<C; c++) {
            atomicAdd(&d_drw[c * E + e], inp[s * C + c] * d_logit);
            d_inp_s[c] += router_w[c * E + e] * d_logit;
        }
        atomicAdd(&d_drb[e], d_logit);
    }
}

__global__ void residual_backward_kernel(float* dinp1, float* dinp2, const float* dout, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= N) return; dinp1[idx] += dout[idx]; dinp2[idx] += dout[idx];
}
__global__ void crossentropy_softmax_backward_kernel(float* dlogits, const float* dlosses, const float* probs, const int* targets, int B, int T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= B * T) return;
    float* dlogits_bt = dlogits + idx * V; const float* probs_bt = probs + idx * V; float dloss = dlosses[idx]; int ix = targets[idx];
    for (int i = 0; i < V; i++) { float indicator = (i == ix) ? 1.0f : 0.0f; dlogits_bt[i] += (probs_bt[i] - indicator) * dloss; }
}

__global__ void adamw_kernel(float* params, float* grads, float* m, float* v, size_t num_parameters, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= num_parameters) return;
    float param = params[idx], grad = grads[idx];
    float m_val = beta1 * m[idx] + (1.0f - beta1) * grad, v_val = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    m[idx] = m_val; v[idx] = v_val;
    params[idx] -= learning_rate * ((m_val / (1.0f - powf(beta1, (float)step))) / (sqrtf(v_val / (1.0f - powf(beta2, (float)step))) + eps) + weight_decay * param);
}

// ============================================================
// Model Struct & Mem Management
// ============================================================
struct TensorDesc { float* d_ptr; size_t size; };
struct GptOSS {
    int V; float *d_wte, *d_rms1w, *d_qkvw, *d_ow, *d_sink_logits;
    float *d_moe_win, *d_moe_bin, *d_moe_wout, *d_moe_bout, *d_moe_rw, *d_moe_rb;
    float *d_rms2w, *d_rmsfw, *d_lm_head;
    float *d_dwte, *d_drms1w, *d_dqkvw, *d_dow, *d_dsink;
    float *d_dmoe_win, *d_dmoe_bin, *d_dmoe_wout, *d_dmoe_bout, *d_dmoe_rw, *d_dmoe_rb;
    float *d_drms2w, *d_drmsfw, *d_dlm_head, *d_m, *d_v_adam;
    float *d_encoded, *d_rms1, *d_rms1_rstd, *d_qkv;
    float *d_atty, *d_att, *d_attproj, *d_residual2, *d_rms2, *d_rms2_rstd;
    
    // Sparse MoE Buffers
    int *d_expert_offsets, *d_expert_write_ptr, *d_scattered_token_idx, *d_expert_pos;
    float *d_scattered_token_w;
    float *d_expert_inp, *d_expert_gate_up, *d_expert_swiglu, *d_expert_out, *d_expert_out_cache, *d_moe_out, *d_router_weights;
    int *d_router_indices; float *d_residual3, *d_rmsf, *d_rmsf_rstd, *d_logits, *d_probs, *d_losses, *d_aux_loss;
    
    float *d_dencoded, *d_drms1, *d_dqkv;
    float *d_datty, *d_dattproj, *d_dresidual2, *d_drms2;
    float *d_dexpert_inp, *d_dexpert_gate_up, *d_dexpert_swiglu, *d_dexpert_out, *d_dmoe_out, *d_dresidual3;
    float *d_drmsf, *d_dlogits; int *d_inputs, *d_targets; size_t num_parameters;
};

std::vector<TensorDesc> get_param_list(GptOSS* m) {
    int qkv_dim = (NH + 2 * NKV) * DK;
    return { {m->d_wte, (size_t)m->V * C}, {m->d_rms1w, (size_t)L * C}, {m->d_qkvw, (size_t)L * C * qkv_dim},
        {m->d_ow, (size_t)L * NH * DK * C}, {m->d_sink_logits, (size_t)L * NH}, 
        {m->d_moe_win, (size_t)L * E * C * 2 * DFF}, {m->d_moe_bin, (size_t)L * E * 2 * DFF},
        {m->d_moe_wout, (size_t)L * E * DFF * C}, {m->d_moe_bout, (size_t)L * E * C}, {m->d_moe_rw, (size_t)L * C * E}, {m->d_moe_rb, (size_t)L * E},
        {m->d_rms2w, (size_t)L * C}, {m->d_rmsfw, (size_t)C}, {m->d_lm_head, (size_t)m->V * C} };
}
std::vector<TensorDesc> get_grad_list(GptOSS* m) {
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

void gpt_oss_alloc(GptOSS *model, int V) {
    model->V = V; model->num_parameters = calc_num_params(V);
    auto params = get_param_list(model); for (auto& p : params) HIP_CHECK(hipMalloc(&p.d_ptr, p.size * sizeof(float)));
    model->d_wte=params[0].d_ptr; model->d_rms1w=params[1].d_ptr; model->d_qkvw=params[2].d_ptr; model->d_ow=params[3].d_ptr;
    model->d_sink_logits=params[4].d_ptr; model->d_moe_win=params[5].d_ptr; model->d_moe_bin=params[6].d_ptr;
    model->d_moe_wout=params[7].d_ptr; model->d_moe_bout=params[8].d_ptr; model->d_moe_rw=params[9].d_ptr;
    model->d_moe_rb=params[10].d_ptr; model->d_rms2w=params[11].d_ptr; model->d_rmsfw=params[12].d_ptr; model->d_lm_head=params[13].d_ptr;
    auto grads = get_grad_list(model); for (auto& g : grads) HIP_CHECK(hipMalloc(&g.d_ptr, g.size * sizeof(float)));
    model->d_dwte=grads[0].d_ptr; model->d_drms1w=grads[1].d_ptr; model->d_dqkvw=grads[2].d_ptr; model->d_dow=grads[3].d_ptr;
    model->d_dsink=grads[4].d_ptr; model->d_dmoe_win=grads[5].d_ptr; model->d_dmoe_bin=grads[6].d_ptr;
    model->d_dmoe_wout=grads[7].d_ptr; model->d_dmoe_bout=grads[8].d_ptr; model->d_dmoe_rw=grads[9].d_ptr;
    model->d_dmoe_rb=grads[10].d_ptr; model->d_drms2w=grads[11].d_ptr; model->d_drmsfw=grads[12].d_ptr; model->d_dlm_head=grads[13].d_ptr;
    HIP_CHECK(hipMalloc(&model->d_m, model->num_parameters * sizeof(float))); HIP_CHECK(hipMalloc(&model->d_v_adam, model->num_parameters * sizeof(float)));
    HIP_CHECK(hipMemset(model->d_m, 0, model->num_parameters * sizeof(float))); HIP_CHECK(hipMemset(model->d_v_adam, 0, model->num_parameters * sizeof(float)));

    int qkv_dim = (NH + 2 * NKV) * DK;
    int total_pairs = B * T * K;
    #define HIP_ALLOC(ptr, size) HIP_CHECK(hipMalloc(&ptr, size * sizeof(float)));
    HIP_ALLOC(model->d_encoded, B*T*C); HIP_ALLOC(model->d_rms1, L*B*T*C); HIP_ALLOC(model->d_rms1_rstd, L*B*T);
    HIP_ALLOC(model->d_qkv, L*B*T*qkv_dim);
    HIP_ALLOC(model->d_atty, L*B*T*NH*DK); HIP_ALLOC(model->d_att, L*B*NH*T*T); HIP_ALLOC(model->d_attproj, L*B*T*C);
    HIP_ALLOC(model->d_residual2, L*B*T*C); HIP_ALLOC(model->d_rms2, L*B*T*C); HIP_ALLOC(model->d_rms2_rstd, L*B*T);
    
    HIP_ALLOC(model->d_expert_inp, L*total_pairs*C); 
    HIP_ALLOC(model->d_expert_gate_up, L*total_pairs*2*DFF); 
    HIP_ALLOC(model->d_expert_swiglu, L*total_pairs*DFF);
    HIP_ALLOC(model->d_expert_out, L*total_pairs*C); 
    HIP_ALLOC(model->d_expert_out_cache, L*total_pairs*C); 
    HIP_ALLOC(model->d_moe_out, L*B*T*C); 
    HIP_ALLOC(model->d_router_weights, L*B*T*K);
    HIP_CHECK(hipMalloc(&model->d_router_indices, L*B*T*K*sizeof(int)));
    
    HIP_ALLOC(model->d_residual3, L*B*T*C); HIP_ALLOC(model->d_rmsf, B*T*C); HIP_ALLOC(model->d_rmsf_rstd, B*T);
    HIP_ALLOC(model->d_logits, B*T*V); HIP_ALLOC(model->d_probs, B*T*V); HIP_ALLOC(model->d_losses, B*T); HIP_ALLOC(model->d_aux_loss, 1);
    HIP_ALLOC(model->d_dencoded, B*T*C); HIP_ALLOC(model->d_drms1, L*B*T*C); HIP_ALLOC(model->d_dqkv, L*B*T*qkv_dim);
    HIP_ALLOC(model->d_datty, L*B*T*NH*DK); HIP_ALLOC(model->d_dattproj, L*B*T*C); HIP_ALLOC(model->d_dresidual2, L*B*T*C); HIP_ALLOC(model->d_drms2, L*B*T*C);
    HIP_ALLOC(model->d_dexpert_inp, L*total_pairs*C); HIP_ALLOC(model->d_dexpert_gate_up, L*total_pairs*2*DFF); HIP_ALLOC(model->d_dexpert_swiglu, L*total_pairs*DFF);
    HIP_ALLOC(model->d_dexpert_out, L*total_pairs*C); HIP_ALLOC(model->d_dmoe_out, L*B*T*C); HIP_ALLOC(model->d_dresidual3, L*B*T*C);
    HIP_ALLOC(model->d_drmsf, B*T*C); HIP_ALLOC(model->d_dlogits, B*T*V);
    
    HIP_ALLOC(model->d_scattered_token_w, L*total_pairs);
    HIP_CHECK(hipMalloc(&model->d_expert_offsets, L*(E+1)*sizeof(int)));
    HIP_CHECK(hipMalloc(&model->d_expert_write_ptr, L*E*sizeof(int)));
    HIP_CHECK(hipMalloc(&model->d_scattered_token_idx, L*total_pairs*sizeof(int)));
    HIP_CHECK(hipMalloc(&model->d_expert_pos, L*total_pairs*sizeof(int)));
    #undef HIP_ALLOC
    HIP_CHECK(hipMalloc(&model->d_inputs, B * T * sizeof(int))); HIP_CHECK(hipMalloc(&model->d_targets, B * T * sizeof(int)));
}

void gpt_oss_free(GptOSS *model) {
    auto params = get_param_list(model); for (auto& p : params) HIP_CHECK(hipFree(p.d_ptr));
    auto grads = get_grad_list(model); for (auto& g : grads) HIP_CHECK(hipFree(g.d_ptr));
    HIP_CHECK(hipFree(model->d_m)); HIP_CHECK(hipFree(model->d_v_adam));
    #define HIP_FREE(ptr) if(ptr) HIP_CHECK(hipFree(ptr));
    HIP_FREE(model->d_encoded); HIP_FREE(model->d_rms1); HIP_FREE(model->d_rms1_rstd); HIP_FREE(model->d_qkv);
    HIP_FREE(model->d_atty); HIP_FREE(model->d_att); HIP_FREE(model->d_attproj); HIP_FREE(model->d_residual2);
    HIP_FREE(model->d_rms2); HIP_FREE(model->d_rms2_rstd); HIP_FREE(model->d_expert_inp); HIP_FREE(model->d_expert_gate_up);
    HIP_FREE(model->d_expert_swiglu); HIP_FREE(model->d_expert_out); HIP_FREE(model->d_expert_out_cache); HIP_FREE(model->d_moe_out); HIP_FREE(model->d_router_weights);
    HIP_FREE(model->d_router_indices); HIP_FREE(model->d_residual3); HIP_FREE(model->d_rmsf); HIP_FREE(model->d_rmsf_rstd);
    HIP_FREE(model->d_logits); HIP_FREE(model->d_probs); HIP_FREE(model->d_losses); HIP_FREE(model->d_aux_loss);
    HIP_FREE(model->d_dencoded); HIP_FREE(model->d_drms1); HIP_FREE(model->d_dqkv);
    HIP_FREE(model->d_datty); HIP_FREE(model->d_dattproj); HIP_FREE(model->d_dresidual2); HIP_FREE(model->d_drms2);
    HIP_FREE(model->d_dexpert_inp); HIP_FREE(model->d_dexpert_gate_up); HIP_FREE(model->d_dexpert_swiglu); HIP_FREE(model->d_dexpert_out);
    HIP_FREE(model->d_dmoe_out); HIP_FREE(model->d_dresidual3); HIP_FREE(model->d_drmsf); HIP_FREE(model->d_dlogits);
    HIP_FREE(model->d_scattered_token_w); HIP_FREE(model->d_expert_offsets); HIP_FREE(model->d_expert_write_ptr);
    HIP_FREE(model->d_scattered_token_idx); HIP_FREE(model->d_expert_pos);
    HIP_FREE(model->d_inputs); HIP_FREE(model->d_targets);
    #undef HIP_FREE
}

void gpt_oss_zero_grad(GptOSS *model) {
    auto grads = get_grad_list(model); for (auto& g : grads) HIP_CHECK(hipMemset(g.d_ptr, 0, g.size * sizeof(float)));
    #define HIP_MEMZERO(ptr, size) HIP_CHECK(hipMemset(ptr, 0, size * sizeof(float)))
    int qkv_dim = (NH + 2 * NKV) * DK;
    int total_pairs = B * T * K;
    HIP_MEMZERO(model->d_dencoded, B*T*C); HIP_MEMZERO(model->d_drms1, L*B*T*C); HIP_MEMZERO(model->d_dqkv, L*B*T*qkv_dim);
    HIP_MEMZERO(model->d_datty, L*B*T*NH*DK); HIP_MEMZERO(model->d_dattproj, L*B*T*C); HIP_MEMZERO(model->d_dresidual2, L*B*T*C); HIP_MEMZERO(model->d_drms2, L*B*T*C);
    HIP_MEMZERO(model->d_dexpert_inp, L*total_pairs*C); HIP_MEMZERO(model->d_dexpert_gate_up, L*total_pairs*2*DFF); HIP_MEMZERO(model->d_dexpert_swiglu, L*total_pairs*DFF);
    HIP_MEMZERO(model->d_dexpert_out, L*total_pairs*C); HIP_MEMZERO(model->d_dmoe_out, L*B*T*C); HIP_MEMZERO(model->d_dresidual3, L*B*T*C);
    HIP_MEMZERO(model->d_drmsf, B*T*C); HIP_MEMZERO(model->d_dlogits, B*T*model->V); HIP_MEMZERO(model->d_dsink, L*NH);
    #undef HIP_MEMZERO
}

// ============================================================
// Checkpointing
// ============================================================
void gpt_oss_save_checkpoint(GptOSS* model, int step, const char* filepath) {
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

int gpt_oss_load_checkpoint(GptOSS* model, const char* filepath) {
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

#define LAUNCH(kernel, N, ...) do { int grid = (N + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL; kernel<<<grid, BLOCK_SIZE_KERNEL>>>(__VA_ARGS__); HIP_CHECK(hipGetLastError()); } while(0)
#define LAUNCH_2D(kernel, N, Y, ...) do { int grid = (N + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL; dim3 gd(grid, Y); kernel<<<gd, BLOCK_SIZE_KERNEL>>>(__VA_ARGS__); HIP_CHECK(hipGetLastError()); } while(0)

// ============================================================
// Forward Pass
// ============================================================
void gpt_oss_forward(GptOSS *model, const int* h_inputs, const int* h_targets) {
    int V = model->V;
    int qkv_dim = (NH + 2 * NKV) * DK;
    int total_pairs = B * T * K;
    HIP_CHECK(hipMemcpy(model->d_inputs, h_inputs, B * T * sizeof(int), hipMemcpyHostToDevice));
    if (h_targets) HIP_CHECK(hipMemcpy(model->d_targets, h_targets, B * T * sizeof(int), hipMemcpyHostToDevice));
    LAUNCH(encoder_forward_kernel, B * T * C, model->d_encoded, model->d_inputs, model->d_wte, B, T, C);
    for (int l = 0; l < L; l++) {
        float* residual = (l == 0) ? model->d_encoded : model->d_residual3 + (l - 1) * B * T * C;
        int is_sliding = (l % 2 == 1);
        LAUNCH(rmsnorm_forward_kernel, B * T * C, model->d_rms1 + l*B*T*C, model->d_rms1_rstd + l*B*T, residual, model->d_rms1w + l*C, B, T, C);
        
        // Fused QKV Projection
        LAUNCH(matmul_forward_kernel, B * T * qkv_dim, model->d_qkv + l*B*T*qkv_dim, model->d_rms1 + l*B*T*C, model->d_qkvw + l*C*qkv_dim, NULL, B * T, C, qkv_dim);
        
        // Fused RoPE (in-place)
        LAUNCH(rope_qk_forward_kernel, B * T * (NH + NKV) * DK, model->d_qkv + l*B*T*qkv_dim, B, T, NH, NKV, DK, ROPE_FACTOR);
        
        // Fused Attention (reads interleaved QKV directly)
        LAUNCH(gqa_sink_sliding_attention_forward_kernel, B * T * NH, model->d_atty + l*B*T*NH*DK, model->d_att + l*B*NH*T*T, model->d_qkv + l*B*T*qkv_dim, model->d_sink_logits + l*NH, B, T, NH, NKV, DK, NGRP, SLIDING_WINDOW, is_sliding);
        
        LAUNCH(matmul_forward_kernel, B * T * C, model->d_attproj + l*B*T*C, model->d_atty + l*B*T*C, model->d_ow + l*NH*DK*C, NULL, B * T, NH * DK, C);
        LAUNCH(residual_forward_kernel, B * T * C, model->d_residual2 + l*B*T*C, residual, model->d_attproj + l*B*T*C, B * T * C);
        LAUNCH(rmsnorm_forward_kernel, B * T * C, model->d_rms2 + l*B*T*C, model->d_rms2_rstd + l*B*T, model->d_residual2 + l*B*T*C, model->d_rms2w + l*C, B, T, C);
        
        HIP_CHECK(hipMemset(model->d_aux_loss, 0, sizeof(float)));
        LAUNCH(moe_router_forward_kernel, B * T, model->d_router_weights + l*B*T*K, model->d_router_indices + l*B*T*K, model->d_aux_loss, model->d_rms2 + l*B*T*C, model->d_moe_rw + l*C*E, model->d_moe_rb + l*E, B, T, C, E, K, AUX_COEF);
        
        // --- SPARSE MoE PIPELINE ---
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
        
        LAUNCH(residual_forward_kernel, B * T * C, model->d_residual3 + l*B*T*C, model->d_residual2 + l*B*T*C, model->d_moe_out + l*B*T*C, B * T * C);
    }
    float* residual = model->d_residual3 + (L - 1) * B * T * C;
    LAUNCH(rmsnorm_forward_kernel, B * T * C, model->d_rmsf, model->d_rmsf_rstd, residual, model->d_rmsfw, B, T, C);
    LAUNCH(matmul_forward_kernel, B * T * V, model->d_logits, model->d_rmsf, model->d_lm_head, NULL, B * T, C, V);
    LAUNCH(softmax_forward_kernel, B * T, model->d_probs, model->d_logits, B, T, V);
    if (h_targets) LAUNCH(crossentropy_forward_kernel, B * T, model->d_losses, model->d_probs, model->d_targets, B, T, V);
}

// ============================================================
// Backward Pass
// ============================================================
void gpt_oss_backward(GptOSS *model) {
    int V = model->V; float dloss_val = 1.0f / (B * T);
    int qkv_dim = (NH + 2 * NKV) * DK;
    int total_pairs = B * T * K;
    std::vector<float> h_dlosses(B * T, dloss_val); HIP_CHECK(hipMemcpy(model->d_losses, h_dlosses.data(), B * T * sizeof(float), hipMemcpyHostToDevice));
    LAUNCH(crossentropy_softmax_backward_kernel, B * T, model->d_dlogits, model->d_losses, model->d_probs, model->d_targets, B, T, V);
    LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_drmsf, model->d_dlogits, model->d_lm_head, B * T, C, V);
    LAUNCH(matmul_backward_dweight_dbias_kernel, V * C, model->d_dlm_head, NULL, model->d_dlogits, model->d_rmsf, B * T, C, V);
    float* residual = model->d_residual3 + (L - 1) * B * T * C, * dresidual = model->d_dresidual3 + (L - 1) * B * T * C;
    LAUNCH(rmsnorm_backward_kernel, B * T * C, dresidual, model->d_drmsfw, model->d_drmsf, residual, model->d_rmsfw, model->d_rmsf_rstd, B, T, C);
    for (int l = L - 1; l >= 0; l--) {
        residual = (l == 0) ? model->d_encoded : model->d_residual3 + (l - 1) * B * T * C;
        dresidual = (l == 0) ? model->d_dencoded : model->d_dresidual3 + (l - 1) * B * T * C;
        int is_sliding = (l % 2 == 1);
        LAUNCH(residual_backward_kernel, B * T * C, model->d_dresidual2 + l*B*T*C, model->d_dmoe_out + l*B*T*C, model->d_dresidual3 + l*B*T*C, B * T * C);
        
        // --- SPARSE MoE BACKWARD ---
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
        
        // Attention Backward (writes to interleaved dQKV)
        LAUNCH(gqa_sink_sliding_attention_backward_kernel, B * T * NH, model->d_dqkv + l*B*T*qkv_dim, model->d_dsink + l*NH, model->d_datty + l*B*T*C, model->d_qkv + l*B*T*qkv_dim, model->d_att + l*B*NH*T*T, B, T, NH, NKV, DK, NGRP, SLIDING_WINDOW, is_sliding);
        
        // Fused RoPE Backward (in-place)
        LAUNCH(rope_qk_backward_kernel, B * T * (NH + NKV) * DK, model->d_dqkv + l*B*T*qkv_dim, B, T, NH, NKV, DK, ROPE_FACTOR);
        
        // Fused QKV Backward
        LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_drms1 + l*B*T*C, model->d_dqkv + l*B*T*qkv_dim, model->d_qkvw + l*C*qkv_dim, B * T, C, qkv_dim);
        LAUNCH(matmul_backward_dweight_dbias_kernel, qkv_dim * C, model->d_dqkvw + l*C*qkv_dim, NULL, model->d_dqkv + l*B*T*qkv_dim, model->d_rms1 + l*B*T*C, B * T, C, qkv_dim);
        
        LAUNCH(rmsnorm_backward_kernel, B * T * C, dresidual, model->d_drms1w + l*C, model->d_drms1 + l*B*T*C, residual, model->d_rms1w + l*C, model->d_rms1_rstd + l*B*T, B, T, C);
    }
    LAUNCH(encoder_backward_kernel, B * T * C, model->d_dwte, NULL, model->d_dencoded, model->d_inputs, B, T, C);
}

void gpt_oss_update(GptOSS *model, float learning_rate, int step) {
    auto params = get_param_list(model); auto grads = get_grad_list(model); size_t m_offset = 0;
    for(size_t i = 0; i < params.size(); i++) { if (params[i].size > 0) {
        int grid = (params[i].size + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL;
        adamw_kernel<<<grid, BLOCK_SIZE_KERNEL>>>(params[i].d_ptr, grads[i].d_ptr, model->d_m + m_offset, model->d_v_adam + m_offset, params[i].size, learning_rate, 0.9f, 0.999f, 1e-8f, 0.1f, step); m_offset += params[i].size;
    }}
}

// ============================================================
// Main
// ============================================================
int main() {
    std::ifstream file("input.txt"); if (!file.is_open()) { fprintf(stderr, "Could not open input.txt!\n"); return 1; }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()); file.close();
    CharTokenizer tokenizer; tokenizer.init(text); int V = tokenizer.vocab_size; printf("Vocab size: %d\n", V);
    std::vector<int> tokens = tokenizer.encode(text); int n = (int)(0.9 * tokens.size());
    std::vector<int> train_tokens(tokens.begin(), tokens.begin() + n), val_tokens(tokens.begin() + n, tokens.end());
    GptOSS model; gpt_oss_alloc(&model, V); std::random_device rd; std::mt19937 gen(1337); std::normal_distribution<float> dist(0.0f, 0.02f);
    
    int start_iter = gpt_oss_load_checkpoint(&model, CKPT_PATH);
    int qkv_dim = (NH + 2 * NKV) * DK;
    if (start_iter == 0) {
        printf("Initializing GPT-OSS model weights from scratch...\n");
        auto init_random = [&](float* d_ptr, size_t size) { std::vector<float> h_buf(size); for (size_t i = 0; i < size; i++) h_buf[i] = dist(gen); HIP_CHECK(hipMemcpy(d_ptr, h_buf.data(), size * sizeof(float), hipMemcpyHostToDevice)); };
        auto init_val = [&](float* d_ptr, size_t size, float val) { std::vector<float> h_buf(size, val); HIP_CHECK(hipMemcpy(d_ptr, h_buf.data(), size * sizeof(float), hipMemcpyHostToDevice)); };
        init_random(model.d_wte, V * C);
        for (int l = 0; l < L; l++) {
            init_val(model.d_rms1w + l*C, C, 1.0f); 
            init_random(model.d_qkvw + l*C*qkv_dim, C*qkv_dim); 
            init_random(model.d_ow + l*NH*DK*C, NH*DK*C); init_val(model.d_sink_logits + l*NH, NH, SINK_INIT);
            init_random(model.d_moe_win + l*E*C*2*DFF, E*C*2*DFF); init_val(model.d_moe_bin + l*E*2*DFF, E*2*DFF, 0.0f);
            init_random(model.d_moe_wout + l*E*DFF*C, E*DFF*C); init_val(model.d_moe_bout + l*E*C, E*C, 0.0f);
            init_random(model.d_moe_rw + l*C*E, C*E); init_val(model.d_moe_rb + l*E, E, 0.0f); init_val(model.d_rms2w + l*C, C, 1.0f);
        }
        init_val(model.d_rmsfw, C, 1.0f); init_random(model.d_lm_head, V * C);
    }
    
    printf("GPT-OSS Model initialized. Parameters: %zu\n", model.num_parameters);
    std::vector<int> h_inputs(B * T), h_targets(B * T); std::vector<float> h_losses(B * T);
    auto eval_loss = [&](const std::vector<int>& data_tokens) -> float {
        float total_loss = 0.0f; int count = 0; int data_len = (int)data_tokens.size();
        for (int k = 0; k < EVAL_ITERS; k++) { for (int b = 0; b < B; b++) { int ix = rand() % (data_len - T);
            for (int t = 0; t < T; t++) { h_inputs[b * T + t] = data_tokens[ix + t]; h_targets[b * T + t] = data_tokens[ix + t + 1]; }
        } gpt_oss_forward(&model, h_inputs.data(), h_targets.data()); HIP_CHECK(hipMemcpy(h_losses.data(), model.d_losses, B * T * sizeof(float), hipMemcpyDeviceToHost));
        for (int i = 0; i < B * T; i++) total_loss += h_losses[i]; count += B * T; } return total_loss / count;
    };
    
    for (int iter = start_iter; iter < MAX_ITERS; iter++) {
        if (iter % EVAL_INTERVAL == 0) {
            printf("Step %d: train loss %f, val loss %f\n", iter, eval_loss(train_tokens), eval_loss(val_tokens));
            gpt_oss_save_checkpoint(&model, iter, CKPT_PATH);
        }
        for (int b = 0; b < B; b++) { int ix = rand() % (n - T); for (int t = 0; t < T; t++) { h_inputs[b * T + t] = train_tokens[ix + t]; h_targets[b * T + t] = train_tokens[ix + t + 1]; } }
        gpt_oss_forward(&model, h_inputs.data(), h_targets.data()); gpt_oss_zero_grad(&model); gpt_oss_backward(&model); gpt_oss_update(&model, LR, iter + 1);
    }
    
    gpt_oss_save_checkpoint(&model, MAX_ITERS, CKPT_PATH);
    printf("Generating text...\n"); std::vector<int> gen_tokens(1, 0); std::vector<float> h_probs(V);
    for (int i = 0; i < 500; i++) { int start_idx = std::max(0, (int)gen_tokens.size() - T), context_len = std::min((int)gen_tokens.size(), T);
        std::vector<int> h_gen_inputs(B * T, 0); for (int t = 0; t < context_len; t++) h_gen_inputs[t] = gen_tokens[start_idx + t];
        gpt_oss_forward(&model, h_gen_inputs.data(), NULL); HIP_CHECK(hipMemcpy(h_probs.data(), model.d_probs + (context_len - 1) * V, V * sizeof(float), hipMemcpyDeviceToHost));
        float r = (float)rand() / RAND_MAX, cdf = 0.0f; int next_token = V - 1;
        for (int j = 0; j < V; j++) { cdf += h_probs[j]; if (r < cdf) { next_token = j; break; } } gen_tokens.push_back(next_token);
    }
    std::string generated_text = tokenizer.decode(gen_tokens); printf("---\n%s\n---\n", generated_text.c_str());
    gpt_oss_free(&model); return 0;
}