// eagle_tree_gen_qwen.cpp
// AMD HIP implementation of EAGLE tree verification for Qwen3 architecture
// Compile: hipcc -O3 eagle_tree_gen_qwen.cpp -lhipblas -o eagle_tree_gen_qwen
// Usage: ./eagle_tree_gen_qwen [target_ckpt] [draft_ckpt] [prompt] [max_tokens] [temperature] [top_k] [draft_depth]

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <random>
#include <chrono>
#include <iostream>

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define HIPBLAS_CHECK(call) do { \
    hipblasStatus_t status = call; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLAS error at %s:%d: status %d\n", __FILE__, __LINE__, (int)status); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Hyperparameters (must match training)
// ============================================================
const int T = 32;               // max sequence length
const int C = 64;               // d_model
const int NH = 4;               // query heads
const int NKV = 2;              // key/value heads
const int NGRP = NH / NKV;      // query heads per kv head
const int DK = C / NH;          // head dimension (16)
const int DFF = 256;            // feed-forward intermediate size
const int L = 4;                // target layers
const float RMS_EPS = 1e-6f;
const int BLOCK_SIZE_KERNEL = 256;
const int MAX_TREE_NODES = 16;  // max nodes in draft tree
const int ATTN_STRIDE = T + MAX_TREE_NODES; // for tree attention arrays

// ============================================================
// Tokenizer (character‑level)
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
            stoi[unique_chars[i]] = i; itos[i] = unique_chars[i];
        }
        vocab_size = (int)unique_chars.size();
    }
    std::vector<int> encode(const std::string& s) {
        std::vector<int> r; for (char ch : s) if(stoi.count(ch)) r.push_back(stoi[ch]); return r;
    }
    std::string decode(const std::vector<int>& ids) {
        std::string r; for (int id : ids) if(itos.count(id)) r += itos[id]; return r;
    }
};

// ============================================================
// HIP kernels – Qwen3 specific
// ============================================================

// Embedding (token only, no position embedding)
__global__ void encoder_forward_kernel(float* out, const int* inp,
                                       const float* wte, int B, int S, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * S * C;
    if (idx >= N) return;
    int c = idx % C;
    int bs = idx / C;
    int ix = inp[bs];
    out[idx] = wte[ix * C + c];
}

// RMSNorm forward
__global__ void rmsnorm_forward_kernel(float* out, float* rstd,
                                       const float* inp, const float* weight,
                                       int B, int S, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * S * C;
    if (idx >= N) return;
    int c = idx % C;
    int bs = idx / C;
    const float* x = inp + bs * C;
    float ss = 0.0f;
    for (int i = 0; i < C; i++) ss += x[i] * x[i];
    ss = ss / C + RMS_EPS;
    float s = 1.0f / sqrtf(ss);
    out[idx] = s * x[c] * weight[c];
    if (c == 0) rstd[bs] = s;
}

// RoPE with explicit positions array (for prefill and incremental)
__global__ void rope_forward_with_pos_kernel(float* out, const float* inp,
                                             const float* cos_table, const float* sin_table,
                                             const int* positions, int total_tokens, int n_heads, int d_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = total_tokens * n_heads * d_k;
    if (idx >= N) return;
    int i = idx % d_k;
    if (i % 2 != 0) return;
    int token_head = idx / d_k;
    int h = token_head % n_heads;
    int token_idx = token_head / n_heads;
    int pos = positions[token_idx];
    int half_idx = i / 2;
    int half = d_k / 2;
    float cos_val = cos_table[pos * half + half_idx];
    float sin_val = sin_table[pos * half + half_idx];
    int base = token_idx * n_heads * d_k + h * d_k;
    float x1 = inp[base + i];
    float x2 = inp[base + i + 1];
    out[base + i]     = x1 * cos_val - x2 * sin_val;
    out[base + i + 1] = x1 * sin_val + x2 * cos_val;
}

// GQA attention (causal) for prefill
__global__ void gqa_attention_forward_kernel(float* out, float* att,
                                             const float* q, const float* k, const float* v,
                                             int B, int S, int NH, int NKV, int DK) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * NH;
    if (idx >= total) return;
    int h = idx % NH;
    int t = (idx / NH) % S;
    int b = idx / (S * NH);
    int kv_h = h / NGRP;
    float scale = 1.0f / sqrtf((float)DK);

    const float* q_bth = q + b * S * NH * DK + t * NH * DK + h * DK;
    const float* k_b = k + b * S * NKV * DK;
    const float* v_b = v + b * S * NKV * DK;
    float* att_bth = att + b * NH * S * S + h * S * S + t * S;

    float maxval = -1e10f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* key = k_b + t2 * NKV * DK + kv_h * DK;
        float val = 0.0f;
        for (int i = 0; i < DK; i++) val += q_bth[i] * key[i];
        val *= scale;
        if (val > maxval) maxval = val;
    }
    for (int t2 = t + 1; t2 < S; t2++) att_bth[t2] = 0.0f;

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

    float* out_bth = out + b * S * NH * DK + t * NH * DK + h * DK;
    for (int i = 0; i < DK; i++) out_bth[i] = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* value = v_b + t2 * NKV * DK + kv_h * DK;
        float a = att_bth[t2];
        for (int i = 0; i < DK; i++) out_bth[i] += a * value[i];
    }
}

// SwiGLU forward
__global__ void swiglu_forward_kernel(float* out, const float* gate, const float* up, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float g = gate[idx];
    float silu = g / (1.0f + expf(-g));
    out[idx] = silu * up[idx];
}

// Residual add
__global__ void residual_forward_kernel(float* out, const float* inp1, const float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = inp1[idx] + inp2[idx];
}

// Softmax
__global__ void softmax_forward_kernel(float* probs, const float* logits, int B, int S, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * S) return;
    const float* l_bt = logits + idx * V;
    float* p_bt = probs + idx * V;
    float maxval = -1e10f;
    for (int i = 0; i < V; i++) if (l_bt[i] > maxval) maxval = l_bt[i];
    float sum = 0.0f;
    for (int i = 0; i < V; i++) { p_bt[i] = expf(l_bt[i] - maxval); sum += p_bt[i]; }
    for (int i = 0; i < V; i++) p_bt[i] /= sum;
}

// Scale logits by temperature
__global__ void scale_kernel(float* data, int N, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] *= scale;
}

// Token embedding (for draft)
__global__ void token_embed_kernel(float* out, const int* inp, const float* wte, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int c = idx % C;
    int bt = idx / C;
    int ix = inp[bt];
    out[idx] = wte[ix * C + c];
}

// Concatenate two (B*S, C) tensors along last dim → (B*S, 2C)
__global__ void cat_last_dim_kernel(float* out, const float* in1, const float* in2,
                                     int total_bs, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = total_bs * 2 * C;
    if (idx >= N) return;
    int c = idx % (2 * C);
    int bs = idx / (2 * C);
    if (c < C) out[idx] = in1[bs * C + c];
    else       out[idx] = in2[bs * C + (c - C)];
}

// Bias add (for matmul via hipBLAS)
__global__ void bias_add_kernel(float* out, const float* bias, int total_bs, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = total_bs * OC;
    if (idx >= N) return;
    int o = idx % OC;
    out[idx] += bias[o];
}

// Append K and V to persistent cache
__global__ void append_kv_to_cache_kernel(float* k_cache, float* v_cache,
                                          const float* k_new, const float* v_new,
                                          int S_new, int cache_offset, int NKVDK) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = S_new * NKVDK;
    if (idx >= N) return;
    int d = idx % NKVDK;
    int s = idx / NKVDK;
    int dst_idx = (cache_offset + s) * NKVDK + d;
    k_cache[dst_idx] = k_new[s * NKVDK + d];
    v_cache[dst_idx] = v_new[s * NKVDK + d];
}

// ============================================================
// GQA attention with KV-cache and tree mask
// ============================================================
__global__ void gqa_attention_tree_cached_kernel(
    float* out, float* att,
    const float* q_rope,
    const float* k_rope_tree,
    const float* v_tree,
    const float* k_cache,
    const float* v_cache,
    const int* tree_mask,
    int B, int S_tree, int cache_len, int attn_stride,
    int NH, int NKV, int DK)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S_tree * NH;
    if (idx >= total) return;

    int h = idx % NH;
    int t = (idx / NH) % S_tree;
    int b = idx / (S_tree * NH);
    int kv_h = h / NGRP;
    float scale = 1.0f / sqrtf((float)DK);

    const float* query = q_rope + b * S_tree * NH * DK + t * NH * DK + h * DK;
    float* att_row = att + b * NH * S_tree * attn_stride
                     + h * S_tree * attn_stride + t * attn_stride;

    float maxval = -1e10f;

    // Attend to cached keys
    for (int c = 0; c < cache_len; c++) {
        const float* key_c = k_cache + c * NKV * DK + kv_h * DK;
        float val = 0.0f;
        for (int i = 0; i < DK; i++) val += query[i] * key_c[i];
        val *= scale;
        att_row[c] = val;
        if (val > maxval) maxval = val;
    }

    // Attend to tree keys (only those allowed by tree_mask)
    for (int t2 = 0; t2 < S_tree; t2++) {
        int pos = cache_len + t2;
        if (tree_mask[t * S_tree + t2]) {
            const float* key_t2 = k_rope_tree + b * S_tree * NKV * DK + t2 * NKV * DK + kv_h * DK;
            float val = 0.0f;
            for (int i = 0; i < DK; i++) val += query[i] * key_t2[i];
            val *= scale;
            att_row[pos] = val;
            if (val > maxval) maxval = val;
        } else {
            att_row[pos] = -1e10f;
        }
    }

    // Softmax
    float expsum = 0.0f;
    int total_len = cache_len + S_tree;
    for (int i = 0; i < total_len; i++) {
        float expv = expf(att_row[i] - maxval);
        expsum += expv;
        att_row[i] = expv;
    }
    float inv_sum = (expsum == 0.0f) ? 0.0f : 1.0f / expsum;
    for (int i = 0; i < total_len; i++) att_row[i] *= inv_sum;

    // Weighted sum of values
    float* out_ptr = out + b * S_tree * NH * DK + t * NH * DK + h * DK;
    for (int i = 0; i < DK; i++) out_ptr[i] = 0.0f;

    // From cache values
    for (int c = 0; c < cache_len; c++) {
        const float* value_c = v_cache + c * NKV * DK + kv_h * DK;
        float a = att_row[c];
        for (int i = 0; i < DK; i++) out_ptr[i] += a * value_c[i];
    }
    // From tree values
    for (int t2 = 0; t2 < S_tree; t2++) {
        if (tree_mask[t * S_tree + t2]) {
            const float* value_t2 = v_tree + b * S_tree * NKV * DK + t2 * NKV * DK + kv_h * DK;
            float a = att_row[cache_len + t2];
            for (int i = 0; i < DK; i++) out_ptr[i] += a * value_t2[i];
        }
    }
}

// ============================================================
// Kernel launch macro
// ============================================================
#define LAUNCH(kernel, N, ...) do { \
    int grid = (N + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL; \
    kernel<<<grid, BLOCK_SIZE_KERNEL>>>(__VA_ARGS__); \
    HIP_CHECK(hipGetLastError()); \
} while(0)

// ============================================================
// Matmul using hipBLAS (helper)
// ============================================================
void matmul_forward_hipblas(hipblasHandle_t handle,
                            float* out, const float* inp,
                            const float* weight, const float* bias,
                            int total_bs, int C_dim, int OC) {
    const float alpha = 1.0f, beta_val = 0.0f;
    HIPBLAS_CHECK(hipblasSgemm(handle,
        HIPBLAS_OP_T, HIPBLAS_OP_N,
        OC, total_bs, C_dim,
        &alpha,
        weight, C_dim,
        inp, C_dim,
        &beta_val,
        out, OC
    ));
    if (bias != NULL) {
        LAUNCH(bias_add_kernel, total_bs * OC, out, bias, total_bs, OC);
    }
}

// ============================================================
// Qwen3 Target Model (inference only)
// ============================================================
struct TensorDesc { float* d_ptr; size_t size; };

struct Qwen3 {
    int V;
    // Parameters (no bias)
    float *d_wte;
    float *d_rms1w, *d_qw, *d_kw, *d_vw, *d_q_norm_w, *d_k_norm_w, *d_attprojw, *d_rms2w;
    float *d_fc_upw, *d_fc_gatew, *d_fc_downw, *d_rmsfw, *d_lm_head;
    // RoPE tables
    float *d_rope_cos, *d_rope_sin;

    // Activations for prefill (size T)
    float *d_encoded;
    float *d_rms1, *d_rms1_rstd;
    float *d_q, *d_k, *d_v;
    float *d_q_normed, *d_k_normed, *d_q_norm_rstd, *d_k_norm_rstd;
    float *d_q_rope, *d_k_rope;
    float *d_atty, *d_att, *d_attproj, *d_residual2;
    float *d_rms2, *d_rms2_rstd;
    float *d_fc_up, *d_fc_gate, *d_fc_activated, *d_fc_down, *d_residual3;
    float *d_rmsf, *d_rmsf_rstd, *d_logits, *d_probs;

    // KV caches (per layer, size T * NKV*DK)
    float *d_k_cache;
    float *d_v_cache;

    // Tree-specific activations (max nodes)
    float *d_tree_encoded;
    float *d_tree_rms1, *d_tree_rms1_rstd;
    float *d_tree_q, *d_tree_k, *d_tree_v;
    float *d_tree_q_normed, *d_tree_k_normed, *d_tree_q_norm_rstd, *d_tree_k_norm_rstd;
    float *d_tree_q_rope, *d_tree_k_rope;
    float *d_tree_atty, *d_tree_att, *d_tree_attproj, *d_tree_residual2;
    float *d_tree_rms2, *d_tree_rms2_rstd;
    float *d_tree_fc_up, *d_tree_fc_gate, *d_tree_fc_activated, *d_tree_fc_down, *d_tree_residual3;
    float *d_tree_rmsf, *d_tree_rmsf_rstd, *d_tree_logits, *d_tree_probs;

    int *d_inputs, *d_positions, *d_tree_mask;
    size_t num_parameters;
};

std::vector<TensorDesc> get_param_list(Qwen3* m) {
    return {
        {m->d_wte, (size_t)m->V * C},
        {m->d_rms1w, (size_t)L * C}, {m->d_qw, (size_t)L * C * C},
        {m->d_kw, (size_t)L * C * (NKV*DK)}, {m->d_vw, (size_t)L * C * (NKV*DK)},
        {m->d_q_norm_w, (size_t)L * DK}, {m->d_k_norm_w, (size_t)L * DK},
        {m->d_attprojw, (size_t)L * C * C}, {m->d_rms2w, (size_t)L * C},
        {m->d_fc_upw, (size_t)L * C * DFF}, {m->d_fc_gatew, (size_t)L * C * DFF},
        {m->d_fc_downw, (size_t)L * DFF * C}, {m->d_rmsfw, (size_t)C},
        {m->d_lm_head, (size_t)m->V * C}
    };
}

void qwen3_alloc(Qwen3 *m, int V) {
    m->V = V;
    auto params = get_param_list(m);
    m->num_parameters = 0;
    for (auto& p : params) m->num_parameters += p.size;
    for (auto& p : params) HIP_CHECK(hipMalloc(&p.d_ptr, p.size * sizeof(float)));
    m->d_wte = params[0].d_ptr; m->d_rms1w = params[1].d_ptr; m->d_qw = params[2].d_ptr;
    m->d_kw = params[3].d_ptr; m->d_vw = params[4].d_ptr; m->d_q_norm_w = params[5].d_ptr;
    m->d_k_norm_w = params[6].d_ptr; m->d_attprojw = params[7].d_ptr; m->d_rms2w = params[8].d_ptr;
    m->d_fc_upw = params[9].d_ptr; m->d_fc_gatew = params[10].d_ptr; m->d_fc_downw = params[11].d_ptr;
    m->d_rmsfw = params[12].d_ptr; m->d_lm_head = params[13].d_ptr;

    // RoPE tables
    HIP_CHECK(hipMalloc(&m->d_rope_cos, T * (DK / 2) * sizeof(float)));
    HIP_CHECK(hipMalloc(&m->d_rope_sin, T * (DK / 2) * sizeof(float)));
    std::vector<float> h_cos(T * (DK/2)), h_sin(T * (DK/2));
    for (int t = 0; t < T; t++) for (int i = 0; i < DK/2; i++) {
        float freq = 1.0f / powf(10000.0f, (float)(2*i)/(float)DK);
        float val = t * freq;
        h_cos[t*(DK/2)+i] = cosf(val); h_sin[t*(DK/2)+i] = sinf(val);
    }
    HIP_CHECK(hipMemcpy(m->d_rope_cos, h_cos.data(), T*(DK/2)*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(m->d_rope_sin, h_sin.data(), T*(DK/2)*sizeof(float), hipMemcpyHostToDevice));

    // Allocate prefill activations (size T)
    #define A(ptr, size) HIP_CHECK(hipMalloc(&ptr, size * sizeof(float)));
    A(m->d_encoded, T*C);
    A(m->d_rms1, L*T*C); A(m->d_rms1_rstd, L*T);
    A(m->d_q, L*T*C); A(m->d_k, L*T*(NKV*DK)); A(m->d_v, L*T*(NKV*DK));
    A(m->d_q_normed, L*T*C); A(m->d_k_normed, L*T*(NKV*DK));
    A(m->d_q_norm_rstd, L*T*NH); A(m->d_k_norm_rstd, L*T*NKV);
    A(m->d_q_rope, L*T*C); A(m->d_k_rope, L*T*(NKV*DK));
    A(m->d_atty, L*T*C); A(m->d_att, L*NH*T*T);
    A(m->d_attproj, L*T*C); A(m->d_residual2, L*T*C);
    A(m->d_rms2, L*T*C); A(m->d_rms2_rstd, L*T);
    A(m->d_fc_up, L*T*DFF); A(m->d_fc_gate, L*T*DFF);
    A(m->d_fc_activated, L*T*DFF); A(m->d_fc_down, L*T*C);
    A(m->d_residual3, L*T*C);
    A(m->d_rmsf, T*C); A(m->d_rmsf_rstd, T);
    A(m->d_logits, T*m->V); A(m->d_probs, T*m->V);
    #undef A

    // KV caches (per layer)
    HIP_CHECK(hipMalloc(&m->d_k_cache, L * T * (NKV*DK) * sizeof(float)));
    HIP_CHECK(hipMalloc(&m->d_v_cache, L * T * (NKV*DK) * sizeof(float)));

    // Tree buffers
    #define A(ptr, size) HIP_CHECK(hipMalloc(&ptr, size * sizeof(float)));
    A(m->d_tree_encoded,    MAX_TREE_NODES * C);
    A(m->d_tree_rms1,       MAX_TREE_NODES * C);
    A(m->d_tree_rms1_rstd,  MAX_TREE_NODES);
    A(m->d_tree_q,          MAX_TREE_NODES * C);
    A(m->d_tree_k,          MAX_TREE_NODES * (NKV*DK));
    A(m->d_tree_v,          MAX_TREE_NODES * (NKV*DK));
    A(m->d_tree_q_normed,   MAX_TREE_NODES * C);
    A(m->d_tree_k_normed,   MAX_TREE_NODES * (NKV*DK));
    A(m->d_tree_q_norm_rstd,MAX_TREE_NODES * NH);
    A(m->d_tree_k_norm_rstd,MAX_TREE_NODES * NKV);
    A(m->d_tree_q_rope,     MAX_TREE_NODES * C);
    A(m->d_tree_k_rope,     MAX_TREE_NODES * (NKV*DK));
    A(m->d_tree_atty,       MAX_TREE_NODES * C);
    A(m->d_tree_att,        NH * MAX_TREE_NODES * ATTN_STRIDE);
    A(m->d_tree_attproj,    MAX_TREE_NODES * C);
    A(m->d_tree_residual2,  MAX_TREE_NODES * C);
    A(m->d_tree_rms2,       MAX_TREE_NODES * C);
    A(m->d_tree_rms2_rstd,  MAX_TREE_NODES);
    A(m->d_tree_fc_up,      MAX_TREE_NODES * DFF);
    A(m->d_tree_fc_gate,    MAX_TREE_NODES * DFF);
    A(m->d_tree_fc_activated,MAX_TREE_NODES * DFF);
    A(m->d_tree_fc_down,    MAX_TREE_NODES * C);
    A(m->d_tree_residual3,  L * MAX_TREE_NODES * C);
    A(m->d_tree_rmsf,       MAX_TREE_NODES * C);
    A(m->d_tree_rmsf_rstd,  MAX_TREE_NODES);
    A(m->d_tree_logits,     MAX_TREE_NODES * m->V);
    A(m->d_tree_probs,      MAX_TREE_NODES * m->V);
    #undef A

    HIP_CHECK(hipMalloc(&m->d_inputs, T * sizeof(int)));
    HIP_CHECK(hipMalloc(&m->d_positions, T * sizeof(int)));
    HIP_CHECK(hipMalloc(&m->d_tree_mask, MAX_TREE_NODES * MAX_TREE_NODES * sizeof(int)));
}

void qwen3_load_params(Qwen3* m, const char* fp) {
    FILE* f = fopen(fp, "rb");
    if (!f) { fprintf(stderr, "No target checkpoint at %s\n", fp); exit(1); }
    int step; fread(&step, sizeof(int), 1, f);
    auto params = get_param_list(m);
    std::vector<float> buf(m->num_parameters);
    fread(buf.data(), sizeof(float), m->num_parameters, f);
    size_t off = 0;
    for (auto& p : params) {
        HIP_CHECK(hipMemcpy(p.d_ptr, buf.data()+off, p.size*sizeof(float), hipMemcpyHostToDevice));
        off += p.size;
    }
    fclose(f);
    printf("Target loaded from %s (step %d)\n", fp, step);
}

// ============================================================
// Qwen3 forward functions
// ============================================================

void qwen3_forward_prefill(Qwen3 *m, hipblasHandle_t handle,
                           const int* h_inputs, int ctx_len, float temperature) {
    int V = m->V;
    HIP_CHECK(hipMemcpy(m->d_inputs, h_inputs, ctx_len * sizeof(int), hipMemcpyHostToDevice));

    LAUNCH(encoder_forward_kernel, ctx_len * C, m->d_encoded, m->d_inputs, m->d_wte, 1, ctx_len, C);

    std::vector<int> h_pos(ctx_len);
    for (int i = 0; i < ctx_len; i++) h_pos[i] = i;
    HIP_CHECK(hipMemcpy(m->d_positions, h_pos.data(), ctx_len * sizeof(int), hipMemcpyHostToDevice));

    for (int l = 0; l < L; l++) {
        float* res = (l == 0) ? m->d_encoded : m->d_residual3 + (l-1)*T*C;

        LAUNCH(rmsnorm_forward_kernel, ctx_len*C, m->d_rms1 + l*T*C, m->d_rms1_rstd + l*T,
               res, m->d_rms1w + l*C, 1, ctx_len, C);

        matmul_forward_hipblas(handle, m->d_q + l*T*C, m->d_rms1 + l*T*C, m->d_qw + l*C*C, NULL, ctx_len, C, C);
        matmul_forward_hipblas(handle, m->d_k + l*T*(NKV*DK), m->d_rms1 + l*T*C, m->d_kw + l*C*(NKV*DK), NULL, ctx_len, C, NKV*DK);
        matmul_forward_hipblas(handle, m->d_v + l*T*(NKV*DK), m->d_rms1 + l*T*C, m->d_vw + l*C*(NKV*DK), NULL, ctx_len, C, NKV*DK);

        // QK-Norm
        LAUNCH(rmsnorm_forward_kernel, ctx_len*NH*DK, m->d_q_normed + l*T*C, m->d_q_norm_rstd + l*T*NH,
               m->d_q + l*T*C, m->d_q_norm_w + l*DK, ctx_len*NH, 1, DK);
        LAUNCH(rmsnorm_forward_kernel, ctx_len*NKV*DK, m->d_k_normed + l*T*(NKV*DK), m->d_k_norm_rstd + l*T*NKV,
               m->d_k + l*T*(NKV*DK), m->d_k_norm_w + l*DK, ctx_len*NKV, 1, DK);

        // RoPE
        LAUNCH(rope_forward_with_pos_kernel, ctx_len*NH*DK,
               m->d_q_rope + l*T*C, m->d_q_normed + l*T*C,
               m->d_rope_cos, m->d_rope_sin, m->d_positions,
               ctx_len, NH, DK);
        LAUNCH(rope_forward_with_pos_kernel, ctx_len*NKV*DK,
               m->d_k_rope + l*T*(NKV*DK), m->d_k_normed + l*T*(NKV*DK),
               m->d_rope_cos, m->d_rope_sin, m->d_positions,
               ctx_len, NKV, DK);

        // GQA attention (causal)
        LAUNCH(gqa_attention_forward_kernel, ctx_len*NH, m->d_atty + l*T*C, m->d_att + l*NH*T*T,
               m->d_q_rope + l*T*C, m->d_k_rope + l*T*(NKV*DK), m->d_v + l*T*(NKV*DK),
               1, ctx_len, NH, NKV, DK);

        // Cache K and V
        HIP_CHECK(hipMemcpy(m->d_k_cache + l * T * (NKV*DK),
                            m->d_k_rope + l*T*(NKV*DK),
                            ctx_len * (NKV*DK) * sizeof(float), hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(m->d_v_cache + l * T * (NKV*DK),
                            m->d_v + l*T*(NKV*DK),
                            ctx_len * (NKV*DK) * sizeof(float), hipMemcpyDeviceToDevice));

        // Attention output projection
        matmul_forward_hipblas(handle, m->d_attproj + l*T*C, m->d_atty + l*T*C,
                               m->d_attprojw + l*C*C, NULL, ctx_len, C, C);
        LAUNCH(residual_forward_kernel, ctx_len*C, m->d_residual2 + l*T*C, res, m->d_attproj + l*T*C, ctx_len*C);

        // Second RMSNorm
        LAUNCH(rmsnorm_forward_kernel, ctx_len*C, m->d_rms2 + l*T*C, m->d_rms2_rstd + l*T,
               m->d_residual2 + l*T*C, m->d_rms2w + l*C, 1, ctx_len, C);

        // SwiGLU
        matmul_forward_hipblas(handle, m->d_fc_up + l*T*DFF, m->d_rms2 + l*T*C,
                               m->d_fc_upw + l*C*DFF, NULL, ctx_len, C, DFF);
        matmul_forward_hipblas(handle, m->d_fc_gate + l*T*DFF, m->d_rms2 + l*T*C,
                               m->d_fc_gatew + l*C*DFF, NULL, ctx_len, C, DFF);
        LAUNCH(swiglu_forward_kernel, ctx_len*DFF, m->d_fc_activated + l*T*DFF,
               m->d_fc_gate + l*T*DFF, m->d_fc_up + l*T*DFF, ctx_len*DFF);
        matmul_forward_hipblas(handle, m->d_fc_down + l*T*C, m->d_fc_activated + l*T*DFF,
                               m->d_fc_downw + l*DFF*C, NULL, ctx_len, DFF, C);
        LAUNCH(residual_forward_kernel, ctx_len*C, m->d_residual3 + l*T*C,
               m->d_residual2 + l*T*C, m->d_fc_down + l*T*C, ctx_len*C);
    }

    float* res = m->d_residual3 + (L-1)*T*C;
    LAUNCH(rmsnorm_forward_kernel, ctx_len*C, m->d_rmsf, m->d_rmsf_rstd,
           res, m->d_rmsfw, 1, ctx_len, C);
    matmul_forward_hipblas(handle, m->d_logits, m->d_rmsf, m->d_lm_head, NULL, ctx_len, C, V);
    if (temperature != 1.0f) LAUNCH(scale_kernel, ctx_len*V, m->d_logits, ctx_len*V, 1.0f/temperature);
    LAUNCH(softmax_forward_kernel, ctx_len, m->d_probs, m->d_logits, 1, ctx_len, V);
}

// Incremental forward with KV-cache
void qwen3_forward_incremental(Qwen3 *m, hipblasHandle_t handle,
                               const int* h_new_tokens, int S_new, int cache_len,
                               float temperature,
                               std::vector<float>& out_feats, std::vector<float>& out_probs) {
    int V = m->V;
    std::vector<int> h_positions(S_new);
    for (int i = 0; i < S_new; i++) h_positions[i] = std::min(T - 1, cache_len + i);

    std::vector<int> h_causal_mask(S_new * S_new, 0);
    for (int i = 0; i < S_new; i++)
        for (int j = 0; j <= i; j++)
            h_causal_mask[i * S_new + j] = 1;

    HIP_CHECK(hipMemcpy(m->d_inputs, h_new_tokens, S_new * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(m->d_positions, h_positions.data(), S_new * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(m->d_tree_mask, h_causal_mask.data(), S_new * S_new * sizeof(int), hipMemcpyHostToDevice));

    LAUNCH(encoder_forward_kernel, S_new*C, m->d_tree_encoded, m->d_inputs, m->d_wte, 1, S_new, C);

    for (int l = 0; l < L; l++) {
        float* res = (l == 0) ? m->d_tree_encoded : m->d_tree_residual3 + (l-1)*MAX_TREE_NODES*C;

        LAUNCH(rmsnorm_forward_kernel, S_new*C, m->d_tree_rms1, m->d_tree_rms1_rstd,
               res, m->d_rms1w + l*C, 1, S_new, C);

        matmul_forward_hipblas(handle, m->d_tree_q, m->d_tree_rms1, m->d_qw + l*C*C, NULL, S_new, C, C);
        matmul_forward_hipblas(handle, m->d_tree_k, m->d_tree_rms1, m->d_kw + l*C*(NKV*DK), NULL, S_new, C, NKV*DK);
        matmul_forward_hipblas(handle, m->d_tree_v, m->d_tree_rms1, m->d_vw + l*C*(NKV*DK), NULL, S_new, C, NKV*DK);

        LAUNCH(rmsnorm_forward_kernel, S_new*NH*DK, m->d_tree_q_normed, m->d_tree_q_norm_rstd,
               m->d_tree_q, m->d_q_norm_w + l*DK, S_new*NH, 1, DK);
        LAUNCH(rmsnorm_forward_kernel, S_new*NKV*DK, m->d_tree_k_normed, m->d_tree_k_norm_rstd,
               m->d_tree_k, m->d_k_norm_w + l*DK, S_new*NKV, 1, DK);

        LAUNCH(rope_forward_with_pos_kernel, S_new*NH*DK,
               m->d_tree_q_rope, m->d_tree_q_normed,
               m->d_rope_cos, m->d_rope_sin, m->d_positions,
               S_new, NH, DK);
        LAUNCH(rope_forward_with_pos_kernel, S_new*NKV*DK,
               m->d_tree_k_rope, m->d_tree_k_normed,
               m->d_rope_cos, m->d_rope_sin, m->d_positions,
               S_new, NKV, DK);

        LAUNCH(gqa_attention_tree_cached_kernel, S_new*NH,
               m->d_tree_atty, m->d_tree_att,
               m->d_tree_q_rope, m->d_tree_k_rope, m->d_tree_v,
               m->d_k_cache + l * T * (NKV*DK),
               m->d_v_cache + l * T * (NKV*DK),
               m->d_tree_mask,
               1, S_new, cache_len, ATTN_STRIDE, NH, NKV, DK);

        LAUNCH(append_kv_to_cache_kernel, S_new*(NKV*DK),
               m->d_k_cache + l * T * (NKV*DK),
               m->d_v_cache + l * T * (NKV*DK),
               m->d_tree_k_rope, m->d_tree_v,
               S_new, cache_len, NKV*DK);

        matmul_forward_hipblas(handle, m->d_tree_attproj, m->d_tree_atty,
                               m->d_attprojw + l*C*C, NULL, S_new, C, C);
        LAUNCH(residual_forward_kernel, S_new*C, m->d_tree_residual2, res, m->d_tree_attproj, S_new*C);

        LAUNCH(rmsnorm_forward_kernel, S_new*C, m->d_tree_rms2, m->d_tree_rms2_rstd,
               m->d_tree_residual2, m->d_rms2w + l*C, 1, S_new, C);

        matmul_forward_hipblas(handle, m->d_tree_fc_up, m->d_tree_rms2,
                               m->d_fc_upw + l*C*DFF, NULL, S_new, C, DFF);
        matmul_forward_hipblas(handle, m->d_tree_fc_gate, m->d_tree_rms2,
                               m->d_fc_gatew + l*C*DFF, NULL, S_new, C, DFF);
        LAUNCH(swiglu_forward_kernel, S_new*DFF, m->d_tree_fc_activated, m->d_tree_fc_gate, m->d_tree_fc_up, S_new*DFF);
        matmul_forward_hipblas(handle, m->d_tree_fc_down, m->d_tree_fc_activated,
                               m->d_fc_downw + l*DFF*C, NULL, S_new, DFF, C);
        LAUNCH(residual_forward_kernel, S_new*C, m->d_tree_residual3 + l*MAX_TREE_NODES*C,
               m->d_tree_residual2, m->d_tree_fc_down, S_new*C);
    }

    float* res = m->d_tree_residual3 + (L-1)*MAX_TREE_NODES*C;
    LAUNCH(rmsnorm_forward_kernel, S_new*C, m->d_tree_rmsf, m->d_tree_rmsf_rstd,
           res, m->d_rmsfw, 1, S_new, C);
    matmul_forward_hipblas(handle, m->d_tree_logits, m->d_tree_rmsf, m->d_lm_head, NULL, S_new, C, V);
    if (temperature != 1.0f) LAUNCH(scale_kernel, S_new*V, m->d_tree_logits, S_new*V, 1.0f/temperature);
    LAUNCH(softmax_forward_kernel, S_new, m->d_tree_probs, m->d_tree_logits, 1, S_new, V);

    out_feats.resize(S_new * C);
    out_probs.resize(S_new * V);
    HIP_CHECK(hipMemcpy(out_feats.data(), m->d_tree_rmsf, S_new*C*sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(out_probs.data(), m->d_tree_probs, S_new*V*sizeof(float), hipMemcpyDeviceToHost));
}

// Tree verification forward (no cache update)
void qwen3_forward_tree_cached(Qwen3 *m, hipblasHandle_t handle,
                               const int* h_tree_tokens, const int* h_tree_positions,
                               const int* h_tree_mask, int S_tree, int cache_len,
                               float temperature) {
    int V = m->V;
    HIP_CHECK(hipMemcpy(m->d_inputs, h_tree_tokens, S_tree * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(m->d_positions, h_tree_positions, S_tree * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(m->d_tree_mask, h_tree_mask, S_tree * S_tree * sizeof(int), hipMemcpyHostToDevice));

    LAUNCH(encoder_forward_kernel, S_tree*C, m->d_tree_encoded, m->d_inputs, m->d_wte, 1, S_tree, C);

    for (int l = 0; l < L; l++) {
        float* res = (l == 0) ? m->d_tree_encoded : m->d_tree_residual3 + (l-1)*MAX_TREE_NODES*C;

        LAUNCH(rmsnorm_forward_kernel, S_tree*C, m->d_tree_rms1, m->d_tree_rms1_rstd,
               res, m->d_rms1w + l*C, 1, S_tree, C);

        matmul_forward_hipblas(handle, m->d_tree_q, m->d_tree_rms1, m->d_qw + l*C*C, NULL, S_tree, C, C);
        matmul_forward_hipblas(handle, m->d_tree_k, m->d_tree_rms1, m->d_kw + l*C*(NKV*DK), NULL, S_tree, C, NKV*DK);
        matmul_forward_hipblas(handle, m->d_tree_v, m->d_tree_rms1, m->d_vw + l*C*(NKV*DK), NULL, S_tree, C, NKV*DK);

        LAUNCH(rmsnorm_forward_kernel, S_tree*NH*DK, m->d_tree_q_normed, m->d_tree_q_norm_rstd,
               m->d_tree_q, m->d_q_norm_w + l*DK, S_tree*NH, 1, DK);
        LAUNCH(rmsnorm_forward_kernel, S_tree*NKV*DK, m->d_tree_k_normed, m->d_tree_k_norm_rstd,
               m->d_tree_k, m->d_k_norm_w + l*DK, S_tree*NKV, 1, DK);

        LAUNCH(rope_forward_with_pos_kernel, S_tree*NH*DK,
               m->d_tree_q_rope, m->d_tree_q_normed,
               m->d_rope_cos, m->d_rope_sin, m->d_positions,
               S_tree, NH, DK);
        LAUNCH(rope_forward_with_pos_kernel, S_tree*NKV*DK,
               m->d_tree_k_rope, m->d_tree_k_normed,
               m->d_rope_cos, m->d_rope_sin, m->d_positions,
               S_tree, NKV, DK);

        LAUNCH(gqa_attention_tree_cached_kernel, S_tree*NH,
               m->d_tree_atty, m->d_tree_att,
               m->d_tree_q_rope, m->d_tree_k_rope, m->d_tree_v,
               m->d_k_cache + l * T * (NKV*DK),
               m->d_v_cache + l * T * (NKV*DK),
               m->d_tree_mask,
               1, S_tree, cache_len, ATTN_STRIDE, NH, NKV, DK);

        matmul_forward_hipblas(handle, m->d_tree_attproj, m->d_tree_atty,
                               m->d_attprojw + l*C*C, NULL, S_tree, C, C);
        LAUNCH(residual_forward_kernel, S_tree*C, m->d_tree_residual2, res, m->d_tree_attproj, S_tree*C);

        LAUNCH(rmsnorm_forward_kernel, S_tree*C, m->d_tree_rms2, m->d_tree_rms2_rstd,
               m->d_tree_residual2, m->d_rms2w + l*C, 1, S_tree, C);

        matmul_forward_hipblas(handle, m->d_tree_fc_up, m->d_tree_rms2, m->d_fc_upw + l*C*DFF, NULL, S_tree, C, DFF);
        matmul_forward_hipblas(handle, m->d_tree_fc_gate, m->d_tree_rms2, m->d_fc_gatew + l*C*DFF, NULL, S_tree, C, DFF);
        LAUNCH(swiglu_forward_kernel, S_tree*DFF, m->d_tree_fc_activated, m->d_tree_fc_gate, m->d_tree_fc_up, S_tree*DFF);
        matmul_forward_hipblas(handle, m->d_tree_fc_down, m->d_tree_fc_activated, m->d_fc_downw + l*DFF*C, NULL, S_tree, DFF, C);
        LAUNCH(residual_forward_kernel, S_tree*C, m->d_tree_residual3 + l*MAX_TREE_NODES*C,
               m->d_tree_residual2, m->d_tree_fc_down, S_tree*C);
    }

    float* res = m->d_tree_residual3 + (L-1)*MAX_TREE_NODES*C;
    LAUNCH(rmsnorm_forward_kernel, S_tree*C, m->d_tree_rmsf, m->d_tree_rmsf_rstd,
           res, m->d_rmsfw, 1, S_tree, C);
    matmul_forward_hipblas(handle, m->d_tree_logits, m->d_tree_rmsf, m->d_lm_head, NULL, S_tree, C, V);
    if (temperature != 1.0f) LAUNCH(scale_kernel, S_tree*V, m->d_tree_logits, S_tree*V, 1.0f/temperature);
    LAUNCH(softmax_forward_kernel, S_tree, m->d_tree_probs, m->d_tree_logits, 1, S_tree, V);
}

// ============================================================
// EAGLE Draft Model for Qwen3
// ============================================================
struct EAGLEDraft {
    int V;
    float* d_wte;
    float* d_lm_head;
    float *d_fcw;
    float *d_ln1w, *d_qw, *d_kw, *d_vw, *d_q_norm_w, *d_k_norm_w, *d_attprojw, *d_ln2w;
    float *d_fc_upw, *d_fc_gatew, *d_fc_downw;
    float *d_lnfw;
    float *d_rope_cos, *d_rope_sin;

    float *d_shifted_tok_emb, *d_fused, *d_fc_out;
    float *d_ln1, *d_ln1_rstd;
    float *d_q, *d_k, *d_v;
    float *d_q_normed, *d_k_normed, *d_q_norm_rstd, *d_k_norm_rstd;
    float *d_q_rope, *d_k_rope;
    float *d_atty, *d_att, *d_attproj, *d_residual2;
    float *d_ln2, *d_ln2_rstd;
    float *d_fc_up, *d_fc_gate, *d_fc_activated, *d_fc_down, *d_residual3;
    float *d_lnf, *d_lnf_rstd;
    float *d_logits, *d_probs;

    size_t num_trainable_parameters;
};

std::vector<TensorDesc> get_draft_param_list(EAGLEDraft* m) {
    return {
        {m->d_fcw, (size_t)2*C*C},
        {m->d_ln1w, (size_t)C}, {m->d_qw, (size_t)C*C},
        {m->d_kw, (size_t)C*(NKV*DK)}, {m->d_vw, (size_t)C*(NKV*DK)},
        {m->d_q_norm_w, (size_t)DK}, {m->d_k_norm_w, (size_t)DK},
        {m->d_attprojw, (size_t)C*C}, {m->d_ln2w, (size_t)C},
        {m->d_fc_upw, (size_t)C*DFF}, {m->d_fc_gatew, (size_t)C*DFF},
        {m->d_fc_downw, (size_t)DFF*C}, {m->d_lnfw, (size_t)C}
    };
}

void eagle_alloc(EAGLEDraft *m, Qwen3* t) {
    m->V = t->V;
    m->d_wte = t->d_wte;
    m->d_lm_head = t->d_lm_head;
    m->d_rope_cos = t->d_rope_cos;
    m->d_rope_sin = t->d_rope_sin;

    auto params = get_draft_param_list(m);
    m->num_trainable_parameters = 0;
    for (auto& p : params) m->num_trainable_parameters += p.size;
    for (auto& p : params) HIP_CHECK(hipMalloc(&p.d_ptr, p.size * sizeof(float)));
    m->d_fcw = params[0].d_ptr; m->d_ln1w = params[1].d_ptr; m->d_qw = params[2].d_ptr;
    m->d_kw = params[3].d_ptr; m->d_vw = params[4].d_ptr; m->d_q_norm_w = params[5].d_ptr;
    m->d_k_norm_w = params[6].d_ptr; m->d_attprojw = params[7].d_ptr; m->d_ln2w = params[8].d_ptr;
    m->d_fc_upw = params[9].d_ptr; m->d_fc_gatew = params[10].d_ptr; m->d_fc_downw = params[11].d_ptr;
    m->d_lnfw = params[12].d_ptr;

    #define A(ptr, size) HIP_CHECK(hipMalloc(&ptr, size * sizeof(float)));
    A(m->d_shifted_tok_emb, T*C);
    A(m->d_fused, T*2*C);
    A(m->d_fc_out, T*C);
    A(m->d_ln1, T*C); A(m->d_ln1_rstd, T);
    A(m->d_q, T*C); A(m->d_k, T*(NKV*DK)); A(m->d_v, T*(NKV*DK));
    A(m->d_q_normed, T*C); A(m->d_k_normed, T*(NKV*DK));
    A(m->d_q_norm_rstd, T*NH); A(m->d_k_norm_rstd, T*NKV);
    A(m->d_q_rope, T*C); A(m->d_k_rope, T*(NKV*DK));
    A(m->d_atty, T*C); A(m->d_att, NH*T*T);
    A(m->d_attproj, T*C); A(m->d_residual2, T*C);
    A(m->d_ln2, T*C); A(m->d_ln2_rstd, T);
    A(m->d_fc_up, T*DFF); A(m->d_fc_gate, T*DFF);
    A(m->d_fc_activated, T*DFF); A(m->d_fc_down, T*C);
    A(m->d_residual3, T*C);
    A(m->d_lnf, T*C); A(m->d_lnf_rstd, T);
    A(m->d_logits, T*m->V); A(m->d_probs, T*m->V);
    #undef A
}

void eagle_load_params(EAGLEDraft* m, const char* fp) {
    FILE* f = fopen(fp, "rb");
    if (!f) { fprintf(stderr, "No draft checkpoint at %s\n", fp); exit(1); }
    int step; fread(&step, sizeof(int), 1, f);
    std::vector<float> buf(m->num_trainable_parameters);
    auto params = get_draft_param_list(m);
    fread(buf.data(), sizeof(float), m->num_trainable_parameters, f);
    size_t off = 0;
    for (auto& p : params) {
        HIP_CHECK(hipMemcpy(p.d_ptr, buf.data()+off, p.size*sizeof(float), hipMemcpyHostToDevice));
        off += p.size;
    }
    fclose(f);
    printf("Draft loaded from %s (step %d)\n", fp, step);
}

// Draft forward: B batches, each of length S (must be ≤ T)
// The caller can then copy the results from m->d_lnf and m->d_probs.
void eagle_draft_forward(EAGLEDraft* m, hipblasHandle_t handle,
                         const float* d_feat_seq, const int* d_tok_seq, const int* d_positions,
                         int B, int S, float temperature) {
    int V = m->V;
    int total_bs = B * S;

    LAUNCH(token_embed_kernel, total_bs*C, m->d_shifted_tok_emb, d_tok_seq, m->d_wte, total_bs*C);
    LAUNCH(cat_last_dim_kernel, total_bs*2*C, m->d_fused, d_feat_seq, m->d_shifted_tok_emb, total_bs, C);
    matmul_forward_hipblas(handle, m->d_fc_out, m->d_fused, m->d_fcw, NULL, total_bs, 2*C, C);

    float* res = m->d_fc_out;
    LAUNCH(rmsnorm_forward_kernel, total_bs*C, m->d_ln1, m->d_ln1_rstd, res, m->d_ln1w, B, S, C);
    matmul_forward_hipblas(handle, m->d_q, m->d_ln1, m->d_qw, NULL, total_bs, C, C);
    matmul_forward_hipblas(handle, m->d_k, m->d_ln1, m->d_kw, NULL, total_bs, C, NKV*DK);
    matmul_forward_hipblas(handle, m->d_v, m->d_ln1, m->d_vw, NULL, total_bs, C, NKV*DK);

    LAUNCH(rmsnorm_forward_kernel, total_bs*NH*DK, m->d_q_normed, m->d_q_norm_rstd,
           m->d_q, m->d_q_norm_w, total_bs*NH, 1, DK);
    LAUNCH(rmsnorm_forward_kernel, total_bs*NKV*DK, m->d_k_normed, m->d_k_norm_rstd,
           m->d_k, m->d_k_norm_w, total_bs*NKV, 1, DK);

    LAUNCH(rope_forward_with_pos_kernel, total_bs*NH*DK,
           m->d_q_rope, m->d_q_normed, m->d_rope_cos, m->d_rope_sin,
           d_positions, total_bs, NH, DK);
    LAUNCH(rope_forward_with_pos_kernel, total_bs*NKV*DK,
           m->d_k_rope, m->d_k_normed, m->d_rope_cos, m->d_rope_sin,
           d_positions, total_bs, NKV, DK);

    LAUNCH(gqa_attention_forward_kernel, total_bs*NH, m->d_atty, m->d_att,
           m->d_q_rope, m->d_k_rope, m->d_v, B, S, NH, NKV, DK);
    matmul_forward_hipblas(handle, m->d_attproj, m->d_atty, m->d_attprojw, NULL, total_bs, C, C);
    LAUNCH(residual_forward_kernel, total_bs*C, m->d_residual2, res, m->d_attproj, total_bs*C);

    LAUNCH(rmsnorm_forward_kernel, total_bs*C, m->d_ln2, m->d_ln2_rstd, m->d_residual2, m->d_ln2w, B, S, C);
    matmul_forward_hipblas(handle, m->d_fc_up, m->d_ln2, m->d_fc_upw, NULL, total_bs, C, DFF);
    matmul_forward_hipblas(handle, m->d_fc_gate, m->d_ln2, m->d_fc_gatew, NULL, total_bs, C, DFF);
    LAUNCH(swiglu_forward_kernel, total_bs*DFF, m->d_fc_activated, m->d_fc_gate, m->d_fc_up, total_bs*DFF);
    matmul_forward_hipblas(handle, m->d_fc_down, m->d_fc_activated, m->d_fc_downw, NULL, total_bs, DFF, C);
    LAUNCH(residual_forward_kernel, total_bs*C, m->d_residual3, m->d_residual2, m->d_fc_down, total_bs*C);

    LAUNCH(rmsnorm_forward_kernel, total_bs*C, m->d_lnf, m->d_lnf_rstd, m->d_residual3, m->d_lnfw, B, S, C);
    matmul_forward_hipblas(handle, m->d_logits, m->d_lnf, m->d_lm_head, NULL, total_bs, C, V);
    if (temperature != 1.0f) LAUNCH(scale_kernel, total_bs*V, m->d_logits, total_bs*V, 1.0f/temperature);
    LAUNCH(softmax_forward_kernel, total_bs, m->d_probs, m->d_logits, B, S, V);
}

// ============================================================
// Sampling utilities
// ============================================================
int sample_multinomial(const std::vector<float>& probs, int V) {
    float r = (float)rand() / RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < V; i++) { cdf += probs[i]; if (r < cdf) return i; }
    return V-1;
}
int sample_greedy(const std::vector<float>& probs, int V) {
    int mi = 0; float mv = probs[0];
    for (int i = 1; i < V; i++) if (probs[i] > mv) { mv = probs[i]; mi = i; }
    return mi;
}
std::vector<std::pair<float,int>> top_k_tokens(const std::vector<float>& probs, int k, int V) {
    std::vector<std::pair<float,int>> pi(V);
    for (int i = 0; i < V; i++) pi[i] = {probs[i], i};
    std::partial_sort(pi.begin(), pi.begin() + std::min(k, V), pi.end(),
                      [](auto& a, auto& b){return a.first > b.first;});
    return pi;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    const char* target_ckpt = "qwen3_hip_checkpoint.bin";
    const char* draft_ckpt = "eagle_draft_checkpoint.bin";
    const char* data_path = "input.txt";
    std::string prompt = "\n";
    int max_new_tokens = 2000;
    float temperature = 0.4f;
    int top_k = 2, draft_depth = 3;
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    if (argc >= 2) target_ckpt = argv[1];
    if (argc >= 3) draft_ckpt = argv[2];
    if (argc >= 4) prompt = argv[3];
    if (argc >= 5) max_new_tokens = atoi(argv[4]);
    if (argc >= 6) temperature = atof(argv[5]);
    if (argc >= 7) top_k = atoi(argv[6]);
    if (argc >= 8) draft_depth = atoi(argv[7]);
    if (argc >= 9) seed = atoi(argv[8]);

    printf("=== EAGLE Tree Verification (Qwen3) ===\n");
    printf("top_k=%d depth=%d temp=%.2f\n", top_k, draft_depth, temperature);
    srand(seed);

    hipblasHandle_t hipblas_handle;
    HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));

    std::ifstream file(data_path);
    if (!file.is_open()) { fprintf(stderr, "No %s\n", data_path); return 1; }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    CharTokenizer tok; tok.init(text);
    int V = tok.vocab_size;
    printf("Vocab size: %d\n", V);

    Qwen3 target;
    qwen3_alloc(&target, V);
    qwen3_load_params(&target, target_ckpt);

    EAGLEDraft draft;
    eagle_alloc(&draft, &target);
    eagle_load_params(&draft, draft_ckpt);

    float *d_dfeat;
    int *d_dtok, *d_dpos;
    HIP_CHECK(hipMalloc(&d_dfeat, T * C * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dtok, T * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_dpos, T * sizeof(int)));

    std::vector<int> gen_tokens = tok.encode(prompt);
    if (gen_tokens.empty()) gen_tokens.push_back(0);

    std::vector<float> h_feat(C), h_prob(V);

    // ─── Prefill on prompt ───
    int max_cache_capacity = T - draft_depth - 1;
    if (max_cache_capacity < 1) max_cache_capacity = 1;
    if ((int)gen_tokens.size() > max_cache_capacity) {
        gen_tokens = std::vector<int>(gen_tokens.end() - max_cache_capacity, gen_tokens.end());
    }
    int cache_len = gen_tokens.size();
    std::vector<int> h_init_inputs(cache_len);
    for (int i = 0; i < cache_len; i++) h_init_inputs[i] = gen_tokens[i];
    qwen3_forward_prefill(&target, hipblas_handle, h_init_inputs.data(), cache_len, temperature);

    std::vector<float> h_lnf(cache_len * C);
    std::vector<float> h_probs_ctx(cache_len * V);
    HIP_CHECK(hipMemcpy(h_lnf.data(), target.d_rmsf, cache_len*C*sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_probs_ctx.data(), target.d_probs, cache_len*V*sizeof(float), hipMemcpyDeviceToHost));
    memcpy(h_feat.data(), h_lnf.data() + (cache_len-1)*C, C*sizeof(float));
    memcpy(h_prob.data(), h_probs_ctx.data() + (cache_len-1)*V, V*sizeof(float));

    // ─── Generation loop ───
    while ((int)gen_tokens.size() < max_new_tokens + (int)tok.encode(prompt).size()) {

        int root_token = (temperature > 0) ? sample_multinomial(h_prob, V) : sample_greedy(h_prob, V);

        struct DraftTree {
            int num_nodes;
            std::vector<int> tokens, parents, depths;
            std::vector<float> next_feats, draft_probs;
            std::vector<int> max_children;
        } tree;

        tree.tokens.push_back(root_token);
        tree.parents.push_back(-1);
        tree.depths.push_back(0);
        tree.next_feats.resize(C);
        memcpy(tree.next_feats.data(), h_feat.data(), C*sizeof(float));
        tree.max_children.push_back(top_k);
        tree.num_nodes = 1;

        int max_tree_nodes = std::min(T / 2, MAX_TREE_NODES);
        if (max_tree_nodes < 1) max_tree_nodes = 1;

        for (int d = 0; d < draft_depth && tree.num_nodes < max_tree_nodes; d++) {
            int n_before = tree.num_nodes;
            std::vector<int> expand_nodes;
            for (int i = 0; i < n_before; i++) {
                if (tree.depths[i] == d) {
                    int existing_children = 0;
                    for (int j = 0; j < tree.num_nodes; j++)
                        if (tree.parents[j] == i) existing_children++;
                    if (existing_children < tree.max_children[i])
                        expand_nodes.push_back(i);
                }
            }
            int L_seq = d + 1;
            int batch_start = 0;
            while (batch_start < (int)expand_nodes.size() && tree.num_nodes < max_tree_nodes) {
                int B_draft = std::min((int)expand_nodes.size() - batch_start, T / L_seq);
                if (B_draft == 0) break;

                std::vector<float> feat_batch(B_draft * L_seq * C);
                std::vector<int> tok_batch(B_draft * L_seq);
                std::vector<int> pos_batch(B_draft * L_seq);

                for (int b = 0; b < B_draft; b++) {
                    int i = expand_nodes[batch_start + b];
                    std::vector<int> path;
                    int cur = i;
                    while (cur != -1) { path.push_back(cur); cur = tree.parents[cur]; }
                    std::reverse(path.begin(), path.end());
                    for (int p = 0; p < L_seq; p++) {
                        int ni = path[p];
                        tok_batch[b * L_seq + p] = tree.tokens[ni];
                        memcpy(feat_batch.data() + (b * L_seq + p) * C,
                               tree.next_feats.data() + ni * C, C * sizeof(float));
                        pos_batch[b * L_seq + p] = std::min(T - 1, cache_len + p);
                    }
                }

                HIP_CHECK(hipMemcpy(d_dfeat, feat_batch.data(), B_draft * L_seq * C * sizeof(float), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_dtok, tok_batch.data(), B_draft * L_seq * sizeof(int), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_dpos, pos_batch.data(), B_draft * L_seq * sizeof(int), hipMemcpyHostToDevice));

                eagle_draft_forward(&draft, hipblas_handle, d_dfeat, d_dtok, d_dpos,
                                    B_draft, L_seq, temperature);

                // Copy results back
                std::vector<float> batch_lnf(B_draft * L_seq * C);
                std::vector<float> batch_probs(B_draft * L_seq * V);
                HIP_CHECK(hipMemcpy(batch_lnf.data(), draft.d_lnf, B_draft * L_seq * C * sizeof(float), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(batch_probs.data(), draft.d_probs, B_draft * L_seq * V * sizeof(float), hipMemcpyDeviceToHost));

                for (int b = 0; b < B_draft; b++) {
                    int i = expand_nodes[batch_start + b];
                    std::vector<float> child_feat(C), child_prob(V);
                    int offset = (b * L_seq + L_seq - 1) * C;
                    memcpy(child_feat.data(), batch_lnf.data() + offset, C * sizeof(float));
                    int prob_offset = (b * L_seq + L_seq - 1) * V;
                    memcpy(child_prob.data(), batch_probs.data() + prob_offset, V * sizeof(float));

                    std::vector<int> children_tokens;
                    if (temperature > 0) {
                        std::set<int> seen;
                        for (int s = 0; s < top_k; s++) {
                            int tok = sample_multinomial(child_prob, V);
                            if (seen.find(tok) == seen.end()) { children_tokens.push_back(tok); seen.insert(tok); }
                        }
                        std::sort(children_tokens.begin(), children_tokens.end(), [&](int a, int b) {
                            return child_prob[a] > child_prob[b];
                        });
                    } else {
                        auto topk = top_k_tokens(child_prob, top_k, V);
                        for (auto& p : topk) children_tokens.push_back(p.second);
                    }

                    for (int k_idx = 0; k_idx < (int)children_tokens.size() && tree.num_nodes < max_tree_nodes; k_idx++) {
                        tree.tokens.push_back(children_tokens[k_idx]);
                        tree.parents.push_back(i);
                        tree.depths.push_back(d + 1);
                        tree.draft_probs.insert(tree.draft_probs.end(), child_prob.begin(), child_prob.end());
                        tree.next_feats.insert(tree.next_feats.end(), child_feat.begin(), child_feat.end());
                        tree.max_children.push_back(std::max(1, top_k - k_idx));
                        tree.num_nodes++;
                    }
                }
                batch_start += B_draft;
            }
        }

        int S_tree = tree.num_nodes;

        // Tree verification
        std::vector<int> h_tree_tokens(S_tree), h_tree_positions(S_tree);
        std::vector<int> h_tree_mask(S_tree * S_tree, 0);
        for (int i = 0; i < S_tree; i++) {
            h_tree_tokens[i] = tree.tokens[i];
            h_tree_positions[i] = std::min(T - 1, cache_len + tree.depths[i]);
        }
        for (int i = 0; i < S_tree; i++) {
            h_tree_mask[i * S_tree + i] = 1;
            int ancestor = tree.parents[i];
            while (ancestor != -1) {
                h_tree_mask[i * S_tree + ancestor] = 1;
                ancestor = tree.parents[ancestor];
            }
        }
        qwen3_forward_tree_cached(&target, hipblas_handle,
                                  h_tree_tokens.data(), h_tree_positions.data(),
                                  h_tree_mask.data(), S_tree, cache_len, temperature);

        std::vector<float> h_tree_probs(S_tree * V);
        HIP_CHECK(hipMemcpy(h_tree_probs.data(), target.d_tree_probs, S_tree * V * sizeof(float), hipMemcpyDeviceToHost));

        // Acceptance sampling (same as original)
        std::vector<bool> accepted(S_tree, false);
        accepted[0] = true;
        for (int i = 1; i < S_tree; i++) {
            int parent = tree.parents[i];
            if (!accepted[parent]) continue;
            float p_target = h_tree_probs[parent * V + tree.tokens[i]];
            float p_draft  = tree.draft_probs[i * V + tree.tokens[i]];
            float ratio = p_target / (p_draft + 1e-10f);
            float r = (float)rand() / RAND_MAX;
            accepted[i] = (r < std::min(1.0f, ratio));
        }

        std::vector<int> accepted_tokens;
        accepted_tokens.push_back(tree.tokens[0]);
        int current_node = 0;
        bool stopped = false;
        while (!stopped) {
            std::vector<int> children;
            for (int j = current_node + 1; j < S_tree; j++)
                if (tree.parents[j] == current_node) children.push_back(j);
            if (children.empty()) {
                std::vector<float> target_dist(V);
                memcpy(target_dist.data(), h_tree_probs.data() + current_node * V, V * sizeof(float));
                int bonus = (temperature > 0) ? sample_multinomial(target_dist, V) : sample_greedy(target_dist, V);
                accepted_tokens.push_back(bonus);
                break;
            }
            std::vector<float> weights;
            std::vector<int> weight_to_child;
            float sum_min_pq = 0.0f;
            for (int child : children) {
                int t = tree.tokens[child];
                float p_target = h_tree_probs[current_node * V + t];
                float p_draft  = tree.draft_probs[child * V + t];
                float w = std::min(p_target, p_draft);
                weights.push_back(w); weight_to_child.push_back(child);
                sum_min_pq += w;
            }
            float W_resample = 1.0f - sum_min_pq;
            if (W_resample < 0.0f) W_resample = 0.0f;
            weights.push_back(W_resample); weight_to_child.push_back(-1);

            float total_weight = 0.0f;
            for (float w : weights) total_weight += w;
            int chosen_outcome = -1;
            if (total_weight > 0) {
                float r = (float)rand() / RAND_MAX * total_weight;
                float cdf = 0.0f;
                for (int i = 0; i < (int)weights.size(); i++) { cdf += weights[i]; if (r < cdf) { chosen_outcome = i; break; } }
                if (chosen_outcome == -1) chosen_outcome = (int)weights.size() - 1;
            } else chosen_outcome = (int)weights.size() - 1;
            int chosen_child = weight_to_child[chosen_outcome];
            if (chosen_child != -1) {
                accepted_tokens.push_back(tree.tokens[chosen_child]);
                current_node = chosen_child;
            } else {
                std::vector<float> adjusted_dist(V, 0.0f);
                float sum_adjusted = 0.0f;
                for (int v = 0; v < V; v++) {
                    float p_target = h_tree_probs[current_node * V + v];
                    float p_draft  = tree.draft_probs[children[0] * V + v];
                    float diff = p_target - p_draft;
                    if (diff > 0) { adjusted_dist[v] = diff; sum_adjusted += diff; }
                }
                if (sum_adjusted > 0) {
                    for (int v = 0; v < V; v++) adjusted_dist[v] /= sum_adjusted;
                    int resampled = (temperature > 0) ? sample_multinomial(adjusted_dist, V) : sample_greedy(adjusted_dist, V);
                    accepted_tokens.push_back(resampled);
                } else {
                    std::vector<float> target_dist(V);
                    memcpy(target_dist.data(), h_tree_probs.data() + current_node * V, V * sizeof(float));
                    int resampled = (temperature > 0) ? sample_multinomial(target_dist, V) : sample_greedy(target_dist, V);
                    accepted_tokens.push_back(resampled);
                }
                stopped = true;
            }
        }

        gen_tokens.insert(gen_tokens.end(), accepted_tokens.begin(), accepted_tokens.end());

        // Update KV-cache and features
        max_cache_capacity = T - draft_depth - 1;
        if (max_cache_capacity < 1) max_cache_capacity = 1;
        if (cache_len + (int)accepted_tokens.size() > max_cache_capacity) {
            int start = std::max(0, (int)gen_tokens.size() - max_cache_capacity);
            int ctx_len = std::min(max_cache_capacity, (int)gen_tokens.size() - start);
            std::vector<int> h_inputs(ctx_len);
            for (int i = 0; i < ctx_len; i++) h_inputs[i] = gen_tokens[start + i];
            qwen3_forward_prefill(&target, hipblas_handle, h_inputs.data(), ctx_len, temperature);
            cache_len = ctx_len;
            std::vector<float> new_lnf(ctx_len*C), new_probs(ctx_len*V);
            HIP_CHECK(hipMemcpy(new_lnf.data(), target.d_rmsf, ctx_len*C*sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(new_probs.data(), target.d_probs, ctx_len*V*sizeof(float), hipMemcpyDeviceToHost));
            memcpy(h_feat.data(), new_lnf.data() + (ctx_len-1)*C, C*sizeof(float));
            memcpy(h_prob.data(), new_probs.data() + (ctx_len-1)*V, V*sizeof(float));
        } else {
            std::vector<float> inc_feats, inc_probs;
            qwen3_forward_incremental(&target, hipblas_handle, accepted_tokens.data(),
                                      (int)accepted_tokens.size(), cache_len, temperature,
                                      inc_feats, inc_probs);
            cache_len += (int)accepted_tokens.size();
            int last_idx = (int)accepted_tokens.size() - 1;
            memcpy(h_feat.data(), inc_feats.data() + last_idx*C, C*sizeof(float));
            memcpy(h_prob.data(), inc_probs.data() + last_idx*V, V*sizeof(float));
        }

        if ((int)gen_tokens.size() % 50 < draft_depth + 2)
            printf("[%zu tokens generated]\n", gen_tokens.size());
    }

    int prompt_len = tok.encode(prompt).size();
    if ((int)gen_tokens.size() > max_new_tokens + prompt_len)
        gen_tokens.resize(max_new_tokens + prompt_len);

    printf("\n=== Generated Text ===\n%s\n=== End ===\n", tok.decode(gen_tokens).c_str());

    HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));
    HIP_CHECK(hipFree(d_dfeat));
    HIP_CHECK(hipFree(d_dtok));
    HIP_CHECK(hipFree(d_dpos));

    return 0;
}