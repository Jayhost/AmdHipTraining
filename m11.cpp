// train_qwen3_vulkan_multi.cpp
// Vulkan Compute Multi-GPU Qwen3 Training
//
// Compile: g++ -O3 -std=c++17 train_qwen3_vulkan_multi.cpp -o train_qwen3_vulkan_multi -lvulkan -lshaderc_shared
//
// Trains a small Qwen3-style transformer using Vulkan compute shaders.
// Supports multi-GPU data-parallel training with gradient averaging.

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
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
#include <array>
#include <chrono>

// ============================================================
// Debug & Logging Utilities
// ============================================================

// Toggle for verbose debug output
#define DEBUG_VERBOSE 1

// Toggle for Vulkan validation layers (useful during development)
#define ENABLE_VALIDATION 0

#define LOG(fmt, ...)    fprintf(stdout, fmt "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) fprintf(stderr, "ERROR: " fmt "\n", ##__VA_ARGS__)

#if DEBUG_VERBOSE
  #define LOG_DBG(fmt, ...) fprintf(stdout, "[DBG] " fmt "\n", ##__VA_ARGS__)
#else
  #define LOG_DBG(fmt, ...) do {} while(0)
#endif

// Convert VkResult to human-readable string
static const char* vkResultString(VkResult err) {
    switch (err) {
        case VK_SUCCESS:                       return "VK_SUCCESS";
        case VK_NOT_READY:                     return "VK_NOT_READY";
        case VK_TIMEOUT:                       return "VK_TIMEOUT";
        case VK_INCOMPLETE:                    return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY:      return "OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:    return "OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:   return "INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:             return "DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED:       return "MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT:       return "LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:   return "EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:     return "FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER:     return "INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS:        return "TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED:    return "FORMAT_NOT_SUPPORTED";
        default:                               return "UNKNOWN_VK_ERROR";
    }
}

// Vulkan error check with file, line, and error name
#define VK_CHECK(call) do { \
    VkResult _vk_err = (call); \
    if (_vk_err != VK_SUCCESS) { \
        LOG_ERR("Vulkan call failed at %s:%d — %s (%d)", \
                __FILE__, __LINE__, vkResultString(_vk_err), _vk_err); \
        exit(1); \
    } \
} while(0)

// Reinterpret float bits as int (for packing floats into int push constants)
static inline int floatToIntBits(float v) {
    int r;
    memcpy(&r, &v, sizeof(float));
    return r;
}

// ============================================================
// Hyperparameters  (~1.5M parameters)
// ============================================================

const int B_TOTAL       = 256;          // Total batch size across all GPUs
const int T             = 64;           // Sequence length
const int MAX_ITERS     = 550;          // Training iterations
const int EVAL_INTERVAL = 100;          // Evaluate every N steps
const int EVAL_ITERS    = 50;           // Batches per evaluation
const float LR          = 1e-2f;        // Learning rate (scaled up for B=256)

// Model dimensions
const int C             = 128;          // Embedding / residual stream dim
const int NH            = 8;            // Number of attention heads
const int NKV           = 4;            // Number of KV heads (GQA)
const int NGRP          = NH / NKV;     // Query heads per KV head = 2
const int DK            = C / NH;       // Head dimension = 16
const int DFF           = 512;          // FFN hidden dimension
const int L             = 6;            // Number of transformer layers

const float RMS_EPS     = 1e-6f;

// Vulkan compute
const int WORKGROUP_SIZE = 256;
const int MAX_GPUS       = 8;
const int NUM_GPUS       = 3;           // Set to 2+ for multi-GPU

// ============================================================
// Parameter Group Indices
// ============================================================
enum ParamIdx {
    PARAM_TOKEN_EMB  = 0,   // [V, C]        — token embedding
    PARAM_ATTN_NORM  = 1,   // [L, C]        — pre-attention RMSNorm weight
    PARAM_WQ         = 2,   // [L, NH*DK, C] — query projection
    PARAM_WK         = 3,   // [L, NKV*DK, C]— key projection
    PARAM_WV         = 4,   // [L, NKV*DK, C]— value projection
    PARAM_Q_NORM     = 5,   // [L, DK]       — query RMSNorm weight (QK-norm)
    PARAM_K_NORM     = 6,   // [L, DK]       — key RMSNorm weight (QK-norm)
    PARAM_ATTN_PROJ  = 7,   // [L, C, C]     — attention output projection
    PARAM_FFN_NORM   = 8,   // [L, C]        — pre-FFN RMSNorm weight
    PARAM_FFN_GATE   = 9,   // [L, DFF, C]   — FFN gate projection
    PARAM_FFN_UP     = 10,  // [L, DFF, C]   — FFN up projection
    PARAM_FFN_DOWN   = 11,  // [L, C, DFF]   — FFN down projection
    PARAM_FINAL_NORM = 12,  // [C]           — final RMSNorm weight
    PARAM_LM_HEAD    = 13,  // [V, C]        — language model head
    NUM_PARAM_GROUPS = 14,
};

// ============================================================
// Character-Level Tokenizer
// ============================================================
struct CharTokenizer {
    std::map<char, int> stoi;
    std::map<int, char> itos;
    int vocab_size = 0;

    void init(const std::string& text) {
        std::vector<char> unique_chars;
        for (char c : text) {
            if (stoi.find(c) == stoi.end()) {
                stoi[c] = (int)unique_chars.size();
                itos[(int)unique_chars.size()] = c;
                unique_chars.push_back(c);
            }
        }
        std::sort(unique_chars.begin(), unique_chars.end());
        stoi.clear();
        itos.clear();
        for (int i = 0; i < (int)unique_chars.size(); i++) {
            stoi[unique_chars[i]] = i;
            itos[i] = unique_chars[i];
        }
        vocab_size = (int)unique_chars.size();
    }

    std::vector<int> encode(const std::string& s) {
        std::vector<int> ids;
        ids.reserve(s.size());
        for (char c : s) ids.push_back(stoi[c]);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string result;
        for (int id : ids) result += itos[id];
        return result;
    }
};

// ============================================================
// GLSL Shader Sources
// ============================================================

static const char* glsl_helpers = R"glsl(
float loadF(int idx) { return uintBitsToFloat(fdata[idx]); }
void  storeF(int idx, float val) { fdata[idx] = floatBitsToUint(val); }
void atomicAddF(int idx, float val) {
    uint expected = fdata[idx];
    while (true) {
        uint new_val = floatBitsToUint(uintBitsToFloat(expected) + val);
        uint old_val = atomicCompSwap(fdata[idx], expected, new_val);
        if (old_val == expected) break;
        expected = old_val;
    }
}
)glsl";

static const char* sh_embed = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(set = 0, binding = 1) buffer IntBuf   { int  idata[]; };
layout(std430, push_constant) uniform PC {
    int o, io, w, B, T, C, p1, p2, p3, p4, p5, p6, p7, p8, p9;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.B * p.T * p.C;
    if (i >= N) return;
    int c  = i % p.C;
    int bt = i / p.C;
    int token_id = idata[p.io + bt];
    storeF(p.o + i, loadF(p.w + token_id * p.C + c));
}
)glsl";

static const char* sh_rmsnorm = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int o, rs, inp, w, B, T, C, p1, p2, p3, p4, p5, p6, p7, p8, p9;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.B * p.T * p.C;
    if (i >= N) return;
    int c  = i % p.C;
    int bt = i / p.C;
    float ss = 0.0;
    for (int j = 0; j < p.C; j++) {
        float x = loadF(p.inp + bt * p.C + j);
        ss += x * x;
    }
    ss = ss / p.C + 1e-6;
    float rstd = 1.0 / sqrt(ss);
    storeF(p.o + i, rstd * loadF(p.inp + bt * p.C + c) * loadF(p.w + c));
    if (c == 0) storeF(p.rs + bt, rstd);
}
)glsl";

static const char* sh_matmul = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int o, inp, w, b, tb, C, OC, hb, p1, p2, p3, p4, p5, p6, p7;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.tb * p.OC;
    if (i >= N) return;
    int oo = i % p.OC;
    int bt = i / p.OC;
    float v = (p.hb != 0) ? loadF(p.b + oo) : 0.0;
    for (int j = 0; j < p.C; j++)
        v += loadF(p.inp + bt * p.C + j) * loadF(p.w + oo * p.C + j);
    storeF(p.o + i, v);
}
)glsl";

// --- Fused Down Projection + Residual (Forward) ---
// Computes down(swiglu) + res_prev + attproj. Bypasses res2 entirely.
// PC: [o, sg, wd, res_prev, attproj, B, T, C, DFF, ...]
static const char* sh_down_res = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int o, sg, wd, res_prev, attproj, B, T, C, DFF, p1, p2, p3, p4, p5, p6;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.B * p.T * p.C;
    if (i >= N) return;
    int c = i % p.C;
    int bt = i / p.C;
    
    float val = 0.0;
    for (int j = 0; j < p.DFF; j++)
        val += loadF(p.sg + bt * p.DFF + j) * loadF(p.wd + c * p.DFF + j);
        
    val += loadF(p.res_prev + i) + loadF(p.attproj + i);
    storeF(p.o + i, val);
}
)glsl";

static const char* sh_qkv = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int oq, ok, ov, inp, wq, wk, wv, B, T, C, NH, NKV, DK, p1, p2, p3;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int BT = p.B * p.T;
    int total_q   = BT * p.NH * p.DK;
    int total_qk  = total_q + BT * p.NKV * p.DK;
    int total_qkv = total_qk + BT * p.NKV * p.DK;
    if (i >= total_qkv) return;
    int bt, out_d;
    float val = 0.0;
    if (i < total_q) {
        bt = i / (p.NH * p.DK);
        out_d = i % (p.NH * p.DK);
        for (int j = 0; j < p.C; j++)
            val += loadF(p.inp + bt * p.C + j) * loadF(p.wq + out_d * p.C + j);
        storeF(p.oq + i, val);
    } else if (i < total_qk) {
        int ii = i - total_q;
        bt = ii / (p.NKV * p.DK);
        out_d = ii % (p.NKV * p.DK);
        for (int j = 0; j < p.C; j++)
            val += loadF(p.inp + bt * p.C + j) * loadF(p.wk + out_d * p.C + j);
        storeF(p.ok + ii, val);
    } else {
        int ii = i - total_qk;
        bt = ii / (p.NKV * p.DK);
        out_d = ii % (p.NKV * p.DK);
        for (int j = 0; j < p.C; j++)
            val += loadF(p.inp + bt * p.C + j) * loadF(p.wv + out_d * p.C + j);
        storeF(p.ov + ii, val);
    }
}
)glsl";

static const char* sh_rope = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int o, inp, co, si, B, T, nh, dk, p1, p2, p3, p4, p5, p6, p7;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.B * p.T * p.nh * p.dk;
    if (i >= N) return;
    int d = i % p.dk;
    if (d % 2 != 0) return;
    int h  = (i / p.dk) % p.nh;
    int t  = (i / (p.dk * p.nh)) % p.T;
    int hi = d / 2;
    int hd = p.dk / 2;
    float cos_val = loadF(p.co + t * hd + hi);
    float sin_val = loadF(p.si + t * hd + hi);
    int base = i - d;
    float x1 = loadF(p.inp + base + d);
    float x2 = loadF(p.inp + base + d + 1);
    storeF(p.o + base + d,     x1 * cos_val - x2 * sin_val);
    storeF(p.o + base + d + 1, x1 * sin_val + x2 * cos_val);
}
)glsl";

static const char* sh_gqa = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int o, att, qo, ko, vo, B, T, NH, NKV, DK, NGRP, p1, p2, p3, p4, p5;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int tot = p.B * p.T * p.NH;
    if (i >= tot) return;
    int h  = i % p.NH;
    int t  = (i / p.NH) % p.T;
    int b  = i / (p.T * p.NH);
    int kv = h / p.NGRP;
    float scale = 1.0 / sqrt(float(p.DK));
    int q_bt   = p.qo  + b*p.T*p.NH*p.DK + t*p.NH*p.DK + h*p.DK;
    int k_base = p.ko  + b*p.T*p.NKV*p.DK;
    int v_base = p.vo  + b*p.T*p.NKV*p.DK;
    int att_bt = p.att + b*p.NH*p.T*p.T + h*p.T*p.T + t*p.T;
    float max_val = -1e10;
    for (int t2 = 0; t2 <= t; t2++) {
        int k_t2 = k_base + t2 * p.NKV * p.DK + kv * p.DK;
        float val = 0.0;
        for (int j = 0; j < p.DK; j++) val += loadF(q_bt + j) * loadF(k_t2 + j);
        val *= scale;
        if (val > max_val) max_val = val;
    }
    for (int t2 = t + 1; t2 < p.T; t2++) storeF(att_bt + t2, 0.0);
    float exp_sum = 0.0;
    for (int t2 = 0; t2 <= t; t2++) {
        int k_t2 = k_base + t2 * p.NKV * p.DK + kv * p.DK;
        float val = 0.0;
        for (int j = 0; j < p.DK; j++) val += loadF(q_bt + j) * loadF(k_t2 + j);
        val *= scale;
        float ev = exp(val - max_val);
        exp_sum += ev;
        storeF(att_bt + t2, ev);
    }
    float inv_sum = (exp_sum == 0.0) ? 0.0 : 1.0 / exp_sum;
    for (int t2 = 0; t2 <= t; t2++) storeF(att_bt + t2, loadF(att_bt + t2) * inv_sum);
    int o_bt = p.o + b*p.T*p.NH*p.DK + t*p.NH*p.DK + h*p.DK;
    for (int j = 0; j < p.DK; j++) storeF(o_bt + j, 0.0);
    for (int t2 = 0; t2 <= t; t2++) {
        int v_t2 = v_base + t2 * p.NKV * p.DK + kv * p.DK;
        float a = loadF(att_bt + t2);
        for (int j = 0; j < p.DK; j++) storeF(o_bt + j, loadF(o_bt + j) + a * loadF(v_t2 + j));
    }
}
)glsl";

static const char* sh_swiglu = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int o, go, uo, N, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.N) return;
    float g = loadF(p.go + i);
    float sig = 1.0 / (1.0 + exp(-g));
    storeF(p.o + i, g * sig * loadF(p.uo + i));
}
)glsl";

// --- Fused Gate/Up/SwiGLU (Forward) ---
// Computes Gate, Up, and SwiGLU in one pass. Does NOT save Gate/Up to VRAM.
// PC: [o, inp, wg, wu, B, T, C, DFF, ...]
static const char* sh_gate_up_swiglu = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int o, inp, wg, wu, B, T, C, DFF, p1, p2, p3, p4, p5, p6, p7;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.B * p.T * p.DFF;
    if (i >= N) return;
    int d = i % p.DFF;
    int bt = i / p.DFF;
    
    float g = 0.0;
    float u = 0.0;
    for (int j = 0; j < p.C; j++) {
        float v = loadF(p.inp + bt * p.C + j);
        g += v * loadF(p.wg + d * p.C + j);
        u += v * loadF(p.wu + d * p.C + j);
    }
    float sig = 1.0 / (1.0 + exp(-g));
    storeF(p.o + i, g * sig * u); // Write only the final SwiGLU output
}
)glsl";

// --- Fused SwiGLU Backward (with Recomputation) ---
// Recomputes Gate and Up from rms2_out, then calculates dGate and dUp.
// PC: [dg, du, do2, inp, wg, wu, B, T, C, DFF, ...]
static const char* sh_swiglu_bwd_recomp = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int dg, du, do2, inp, wg, wu, B, T, C, DFF, p1, p2, p3, p4, p5;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.B * p.T * p.DFF;
    if (i >= N) return;
    int d = i % p.DFF;
    int bt = i / p.DFF;
    
    // 1. Recompute gate and up
    float g = 0.0;
    float u = 0.0;
    for (int j = 0; j < p.C; j++) {
        float v = loadF(p.inp + bt * p.C + j);
        g += v * loadF(p.wg + d * p.C + j);
        u += v * loadF(p.wu + d * p.C + j);
    }
    
    // 2. Compute gradients
    float sig = 1.0 / (1.0 + exp(-g));
    float dsilu = sig + g * sig * (1.0 - sig);
    
    float dswiglu = loadF(p.do2 + i);
    storeF(p.dg + i, dswiglu * u * dsilu);
    storeF(p.du + i, dswiglu * g * sig);
}
)glsl";

static const char* sh_residual = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int o, i1, i2, N, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.N) return;
    storeF(p.o + i, loadF(p.i1 + i) + loadF(p.i2 + i));
}
)glsl";

static const char* sh_softmax = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int po, lo, B, T, V, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.B * p.T) return;
    int logits_base = p.lo + i * p.V;
    int probs_base  = p.po + i * p.V;
    float max_val = -1e10;
    for (int j = 0; j < p.V; j++)
        if (loadF(logits_base + j) > max_val) max_val = loadF(logits_base + j);
    float s = 0.0;
    for (int j = 0; j < p.V; j++) {
        storeF(probs_base + j, exp(loadF(logits_base + j) - max_val));
        s += loadF(probs_base + j);
    }
    for (int j = 0; j < p.V; j++) storeF(probs_base + j, loadF(probs_base + j) / s);
}
)glsl";

static const char* sh_cross_entropy = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(set = 0, binding = 1) buffer IntBuf   { int idata[]; };
layout(std430, push_constant) uniform PC {
    int lo, po, to, B, T, V, p1, p2, p3, p4, p5, p6, p7, p8, p9;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.B * p.T) return;
    int target_id = idata[p.to + i];
    storeF(p.lo + i, -log(loadF(p.po + i * p.V + target_id) + 1e-10));
}
)glsl";

static const char* sh_fill_zero = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int o, N, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.N) return;
    storeF(p.o + i, 0.0);
}
)glsl";

static const char* sh_ce_softmax_bwd = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(set = 0, binding = 1) buffer IntBuf   { int idata[]; };
layout(std430, push_constant) uniform PC {
    int dlo, po, to, B, T, V, p1, p2, p3, dlv, p4, p5;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.B * p.T) return;
    int dlogits_base = p.dlo + i * p.V;
    int probs_base   = p.po  + i * p.V;
    int target_id    = idata[p.to + i];
    float dLoss = uintBitsToFloat(p.dlv);
    for (int j = 0; j < p.V; j++) {
        float indicator = (j == target_id) ? 1.0 : 0.0;
        storeF(dlogits_base + j, loadF(dlogits_base + j) + (loadF(probs_base + j) - indicator) * dLoss);
    }
}
)glsl";

static const char* sh_embed_bwd = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(set = 0, binding = 1) buffer IntBuf   { int idata[]; };
layout(std430, push_constant) uniform PC {
    int dw, do2, io, B, T, C, p1, p2, p3, p4, p5, p6, p7, p8, p9;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.B * p.T * p.C;
    if (i >= N) return;
    int c  = i % p.C;
    int bt = i / p.C;
    int token_id = idata[p.io + bt];
    atomicAddF(p.dw + token_id * p.C + c, loadF(p.do2 + i));
}
)glsl";

static const char* sh_rmsnorm_bwd = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int di, dw, do2, inp, w, rs, B, T, C, p1, p2, p3, p4, p5;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.B * p.T * p.C;
    if (i >= N) return;
    int c  = i % p.C;
    int bt = i / p.C;
    float rstd = loadF(p.rs + bt);
    float ds = 0.0;
    for (int j = 0; j < p.C; j++)
        ds += loadF(p.w + j) * loadF(p.do2 + bt * p.C + j) * loadF(p.inp + bt * p.C + j);
    ds = ds * rstd * rstd / p.C;
    atomicAddF(p.dw + c, loadF(p.inp + bt * p.C + c) * rstd * loadF(p.do2 + bt * p.C + c));
    storeF(p.di + bt * p.C + c, loadF(p.di + bt * p.C + c) + rstd * (loadF(p.w + c) * loadF(p.do2 + bt * p.C + c) - ds * loadF(p.inp + bt * p.C + c)));
}
)glsl";

// --- Fused Residual + RMSNorm (Forward) ---
// Adds in1 + in2, then RMSNorms the result. Does NOT save the intermediate sum.
// PC: [o, rstd, i1, i2, w, B, T, C, ...]
static const char* sh_res_rms = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int o, rs, i1, i2, w, B, T, C, p1, p2, p3, p4, p5, p6, p7;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.B * p.T * p.C;
    if (i >= N) return;
    int c  = i % p.C;
    int bt = i / p.C;
    
    float ss = 0.0;
    for (int j = 0; j < p.C; j++) {
        float x = loadF(p.i1 + bt * p.C + j) + loadF(p.i2 + bt * p.C + j);
        ss += x * x;
    }
    float rstd = 1.0 / sqrt(ss / p.C + 1e-6);
    float val = loadF(p.i1 + i) + loadF(p.i2 + i);
    storeF(p.o + i, rstd * val * loadF(p.w + c));
    if (c == 0) storeF(p.rs + bt, rstd);
}
)glsl";

// --- Fused Residual + RMSNorm (Backward with Recomputation) ---
// Recomputes the sum, calculates dW, and splits the gradient back to d1 and d2.
// PC: [d1, d2, dw, do2, i1, i2, w, rstd, B, T, C, ...]
static const char* sh_res_rms_bwd = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int d1, d2, dw, do2, i1, i2, w, rs, B, T, C, p1, p2, p3, p4, p5;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.B * p.T * p.C;
    if (i >= N) return;
    int c  = i % p.C;
    int bt = i / p.C;
    
    float rstd = loadF(p.rs + bt);
    float val = loadF(p.i1 + i) + loadF(p.i2 + i);
    
    // ds = sum_j (w_j * dOut_j * val_j) * rstd^2 / C
    float ds = 0.0;
    for (int j = 0; j < p.C; j++) {
        float x = loadF(p.i1 + bt * p.C + j) + loadF(p.i2 + bt * p.C + j);
        ds += loadF(p.w + j) * loadF(p.do2 + bt * p.C + j) * x;
    }
    ds = ds * rstd * rstd / p.C;
    
    atomicAddF(p.dw + c, val * rstd * loadF(p.do2 + i));
    
    float dres = rstd * (loadF(p.w + c) * loadF(p.do2 + i) - ds * val);
    storeF(p.d1 + i, loadF(p.d1 + i) + dres);
    storeF(p.d2 + i, loadF(p.d2 + i) + dres);
}
)glsl";

static const char* sh_matmul_din = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int di, do2, w, tb, C, OC, p1, p2, p3, p4, p5, p6, p7, p8, p9;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.tb * p.C;
    if (i >= N) return;
    int j  = i % p.C;
    int bt = i / p.C;
    float v = 0.0;
    for (int o = 0; o < p.OC; o++) v += loadF(p.do2 + bt * p.OC + o) * loadF(p.w + o * p.C + j);
    storeF(p.di + i, loadF(p.di + i) + v);
}
)glsl";

static const char* sh_matmul_dweight = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int dw, db, do2, inp, tb, C, OC, hb, p1, p2, p3, p4, p5, p6, p7;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.OC * p.C;
    if (i >= N) return;
    int j = i % p.C;
    int o = i / p.C;
    float dw_val = 0.0;
    for (int bt = 0; bt < p.tb; bt++) dw_val += loadF(p.do2 + bt * p.OC + o) * loadF(p.inp + bt * p.C + j);
    storeF(p.dw + i, loadF(p.dw + i) + dw_val);
}
)glsl";

static const char* sh_rope_bwd = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int di, do2, co, si, B, T, nh, dk, p1, p2, p3, p4, p5, p6, p7;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int N = p.B * p.T * p.nh * p.dk;
    if (i >= N) return;
    int d = i % p.dk;
    if (d % 2 != 0) return;
    int h  = (i / p.dk) % p.nh;
    int t  = (i / (p.dk * p.nh)) % p.T;
    int hi = d / 2;
    int hd = p.dk / 2;
    float cos_val = loadF(p.co + t * hd + hi);
    float sin_val = loadF(p.si + t * hd + hi);
    int base = i - d;
    float dy1 = loadF(p.do2 + base + d);
    float dy2 = loadF(p.do2 + base + d + 1);
    storeF(p.di + base + d,     loadF(p.di + base + d)     + dy1 * cos_val + dy2 * sin_val);
    storeF(p.di + base + d + 1, loadF(p.di + base + d + 1) + (-dy1 * sin_val + dy2 * cos_val));
}
)glsl";

// --- Fused QK-Norm + RoPE (Forward) ---
// Normalizes Q (or K) and applies RoPE in one pass. Does NOT save the intermediate norm.
// PC: [o, rstd, inp, w, co, si, B, T, nh, dk, ...]
static const char* sh_qk_norm_rope = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int o, rs, inp, w, co, si, B, T, nh, dk, p1, p2, p3, p4, p5, p6;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int hd = p.dk / 2;
    int N = p.B * p.T * p.nh * hd; // 1 thread per pair
    if (i >= N) return;

    int pair = i % hd;
    int h = (i / hd) % p.nh;
    int t = (i / (hd * p.nh)) % p.T;
    int b = i / (hd * p.nh * p.T);
    int base = b * p.T * p.nh * p.dk + t * p.nh * p.dk + h * p.dk;

    // 1. Compute RMSNorm for this head
    float ss = 0.0;
    for (int j = 0; j < p.dk; j++) {
        float x = loadF(p.inp + base + j);
        ss += x * x;
    }
    float rstd = 1.0 / sqrt(ss / p.dk + 1e-6);
    if (pair == 0) storeF(p.rs + b * p.T * p.nh + t * p.nh + h, rstd);

    // 2. Apply RoPE
    float cos_val = loadF(p.co + t * hd + pair);
    float sin_val = loadF(p.si + t * hd + pair);

    float x1 = loadF(p.inp + base + pair * 2);
    float x2 = loadF(p.inp + base + pair * 2 + 1);

    float n1 = rstd * x1 * loadF(p.w + pair * 2);
    float n2 = rstd * x2 * loadF(p.w + pair * 2 + 1);

    storeF(p.o + base + pair * 2,     n1 * cos_val - n2 * sin_val);
    storeF(p.o + base + pair * 2 + 1, n1 * sin_val + n2 * cos_val);
}
)glsl";

// --- Fused QK-Norm + RoPE (Backward with Recomputation) ---
// Reads dQ_rot, un-rotates to get dQ_norm, recomputes Q_norm from Q, and calculates dQ and dW.
// PC: [di, dw, do2, inp, w, rs, co, si, B, T, nh, dk, ...]
static const char* sh_qk_norm_rope_bwd = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int di, dw, do2, inp, w, rs, co, si, B, T, nh, dk, p1, p2, p3, p4;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int hd = p.dk / 2;
    int N = p.B * p.T * p.nh * hd;
    if (i >= N) return;

    int pair = i % hd;
    int h = (i / hd) % p.nh;
    int t = (i / (hd * p.nh)) % p.T;
    int b = i / (hd * p.nh * p.T);
    int base = b * p.T * p.nh * p.dk + t * p.nh * p.dk + h * p.dk;
    int rs_idx = b * p.T * p.nh + t * p.nh + h;
    float rstd = loadF(p.rs + rs_idx);

    // 1. Compute ds = sum_j (w_j * dNorm_j * norm_j) * rstd^2 / dk
    float ds = 0.0;
    for (int j = 0; j < hd; j++) {
        float cos_val = loadF(p.co + t * hd + j);
        float sin_val = loadF(p.si + t * hd + j);
        
        // Unrotate dOut to get dNorm
        float do1 = loadF(p.do2 + base + j * 2);
        float do2_val = loadF(p.do2 + base + j * 2 + 1);
        float d_norm1 = do1 * cos_val + do2_val * sin_val;
        float d_norm2 = -do1 * sin_val + do2_val * cos_val;

        // Recompute norm
        float x1 = loadF(p.inp + base + j * 2);
        float x2 = loadF(p.inp + base + j * 2 + 1);
        float norm1 = rstd * x1 * loadF(p.w + j * 2);
        float norm2 = rstd * x2 * loadF(p.w + j * 2 + 1);

        ds += loadF(p.w + j * 2) * d_norm1 * norm1;
        ds += loadF(p.w + j * 2 + 1) * d_norm2 * norm2;
    }
    ds = ds * rstd * rstd / p.dk;

    // 2. Compute gradients for this thread's pair
    float cos_val = loadF(p.co + t * hd + pair);
    float sin_val = loadF(p.si + t * hd + pair);
    
    float do1 = loadF(p.do2 + base + pair * 2);
    float do2_val = loadF(p.do2 + base + pair * 2 + 1);
    float d_norm1 = do1 * cos_val + do2_val * sin_val;
    float d_norm2 = -do1 * sin_val + do2_val * cos_val;

    float x1 = loadF(p.inp + base + pair * 2);
    float x2 = loadF(p.inp + base + pair * 2 + 1);
    float w1 = loadF(p.w + pair * 2);
    float w2 = loadF(p.w + pair * 2 + 1);

    atomicAddF(p.dw + pair * 2, x1 * rstd * d_norm1);
    atomicAddF(p.dw + pair * 2 + 1, x2 * rstd * d_norm2);

    float din1 = rstd * (w1 * d_norm1 - ds * x1);
    float din2 = rstd * (w2 * d_norm2 - ds * x2);
    storeF(p.di + base + pair * 2, loadF(p.di + base + pair * 2) + din1);
    storeF(p.di + base + pair * 2 + 1, loadF(p.di + base + pair * 2 + 1) + din2);
}
)glsl";

static const char* sh_gqa_bwd = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int dq, dk, dv, do2, q, k, v, att, B, T, NH, NKV, DK, NGRP, p1, p2, p3, p4, p5;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int tot = p.B * p.T * p.NH;
    if (i >= tot) return;
    int h  = i % p.NH;
    int t  = (i / p.NH) % p.T;
    int b  = i / (p.T * p.NH);
    int kv = h / p.NGRP;
    float scale = 1.0 / sqrt(float(p.DK));
    int att_bt = p.att + b*p.NH*p.T*p.T + h*p.T*p.T + t*p.T;
    int do_bt  = p.do2 + b*p.T*p.NH*p.DK + t*p.NH*p.DK + h*p.DK;
    int q_bt   = p.q   + b*p.T*p.NH*p.DK + t*p.NH*p.DK + h*p.DK;
    int dq_bt  = p.dq  + b*p.T*p.NH*p.DK + t*p.NH*p.DK + h*p.DK;
    int k_base = p.k  + b*p.T*p.NKV*p.DK;
    int v_base = p.v  + b*p.T*p.NKV*p.DK;
    int dk_base = p.dk + b*p.T*p.NKV*p.DK;
    int dv_base = p.dv + b*p.T*p.NKV*p.DK;
    float dot = 0.0;
    for (int t2 = 0; t2 <= t; t2++) {
        float da = 0.0;
        int v_t2  = v_base  + t2 * p.NKV * p.DK + kv * p.DK;
        int dv_t2 = dv_base + t2 * p.NKV * p.DK + kv * p.DK;
        for (int j = 0; j < p.DK; j++) {
            da += loadF(v_t2 + j) * loadF(do_bt + j);
            atomicAddF(dv_t2 + j, loadF(att_bt + t2) * loadF(do_bt + j));
        }
        dot += loadF(att_bt + t2) * da;
    }
    for (int t2 = 0; t2 <= t; t2++) {
        float da = 0.0;
        int v_t2 = v_base + t2 * p.NKV * p.DK + kv * p.DK;
        for (int j = 0; j < p.DK; j++) da += loadF(v_t2 + j) * loadF(do_bt + j);
        float d_score = loadF(att_bt + t2) * (da - dot) * scale;
        int k_t2  = k_base  + t2 * p.NKV * p.DK + kv * p.DK;
        int dk_t2 = dk_base + t2 * p.NKV * p.DK + kv * p.DK;
        for (int j = 0; j < p.DK; j++) {
            storeF(dq_bt + j, loadF(dq_bt + j) + loadF(k_t2 + j) * d_score);
            atomicAddF(dk_t2 + j, loadF(q_bt + j) * d_score);
        }
    }
}
)glsl";

static const char* sh_swiglu_bwd = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int dg, du, do2, go, uo, N, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.N) return;
    float g   = loadF(p.go + i);
    float sig = 1.0 / (1.0 + exp(-g));
    float dsilu = sig + g * sig * (1.0 - sig);
    storeF(p.dg + i, loadF(p.dg + i) + loadF(p.do2 + i) * loadF(p.uo + i) * dsilu);
    storeF(p.du + i, loadF(p.du + i) + loadF(p.do2 + i) * g * sig);
}
)glsl";

static const char* sh_residual_bwd = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int d1, d2, do2, N, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.N) return;
    storeF(p.d1 + i, loadF(p.d1 + i) + loadF(p.do2 + i));
    storeF(p.d2 + i, loadF(p.d2 + i) + loadF(p.do2 + i));
}
)glsl";

// --- Fused 3-Way Residual Backward ---
// Splits d_res3 into d_layer_input, d_attn_proj, and d_down.
// PC: [d1, d2, d3, do2, N, ...]
static const char* sh_resid_bwd_3way = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int d1, d2, d3, do2, N, p1, p2, p3, p4, p4_2, p5, p6, p7, p8, p9, p10, p11;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.N) return;
    float val = loadF(p.do2 + i);
    storeF(p.d1 + i, loadF(p.d1 + i) + val);
    storeF(p.d2 + i, loadF(p.d2 + i) + val);
    storeF(p.d3 + i, loadF(p.d3 + i) + val);
}
)glsl";

static const char* sh_adamw = R"glsl(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer FloatBuf { uint fdata[]; };
layout(std430, push_constant) uniform PC {
    int pa, gr, m, v, N, step, p1, p2, p3, p4, p5, p6, p7, p8, p9;
} p;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.N) return;
    float lr  = 1e-2;
    float b1  = 0.9;
    float b2  = 0.999;
    float eps = 1e-8;
    float wd  = 0.01;
    float param = loadF(p.pa + i);
    float grad  = loadF(p.gr + i);
    float m_new = b1 * loadF(p.m + i) + (1.0 - b1) * grad;
    float v_new = b2 * loadF(p.v + i) + (1.0 - b2) * grad * grad;
    storeF(p.m + i, m_new);
    storeF(p.v + i, v_new);
    float m_hat = m_new / (1.0 - pow(b1, float(p.step)));
    float v_hat = v_new / (1.0 - pow(b2, float(p.step)));
    storeF(p.pa + i, param - lr * (m_hat / (sqrt(v_hat) + eps) + wd * param));
}
)glsl";

// ============================================================
// Vulkan Types & Helpers
// ============================================================

struct VulkanContext {
    VkInstance         instance;
    VkPhysicalDevice   physicalDevice;
    VkDevice           device;
    VkQueue            queue;
    VkCommandPool      cmdPool;
    VkCommandBuffer    cmdBuffer;
    VkDescriptorSetLayout descSetLayout;
    VkDescriptorPool   descPool;
    VkDescriptorSet    descSet;
    VkFence            fence;
    uint32_t           queueFamily;
    char               deviceName[256];
};

struct VulkanBuffer {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    void*          mappedPtr;    // nullptr if not host-visible
    VkDeviceSize   size;
};

struct VulkanPipeline {
    VkPipeline       pipeline;
    VkPipelineLayout layout;
};

std::vector<uint32_t> compileGlslToSpv(const char* source, const char* name) {
    std::string src(source);
    size_t pos = src.find("void main()");
    if (pos != std::string::npos) src.insert(pos, "\n" + std::string(glsl_helpers) + "\n");
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    auto result = compiler.CompileGlslToSpv(src.c_str(), src.length(), shaderc_compute_shader, name, options);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        LOG_ERR("Shader compilation failed for '%s':\n%s", name, result.GetErrorMessage().c_str());
        exit(1);
    }
    return {result.cbegin(), result.cend()};
}

VkShaderModule createShaderModule(const VulkanContext& ctx, const std::vector<uint32_t>& spirv) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = spirv.size() * sizeof(uint32_t);
    ci.pCode    = spirv.data();
    VkShaderModule module;
    VK_CHECK(vkCreateShaderModule(ctx.device, &ci, nullptr, &module));
    return module;
}

uint32_t findMemoryType(const VulkanContext& ctx, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    LOG_ERR("Failed to find suitable memory type");
    exit(1);
}

void createBuffer(const VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VulkanBuffer& out) {
    VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size = size;
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(ctx.device, &bufInfo, nullptr, &out.buffer));
    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(ctx.device, out.buffer, &memReqs);
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx, memReqs.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &out.memory));
    VK_CHECK(vkBindBufferMemory(ctx.device, out.buffer, out.memory, 0));
    out.mappedPtr = nullptr;
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        VK_CHECK(vkMapMemory(ctx.device, out.memory, 0, size, 0, &out.mappedPtr));
    out.size = size;
}

VulkanPipeline createComputePipeline(const VulkanContext& ctx, VkShaderModule shaderModule) {
    VulkanPipeline pipe;
    VkPipelineShaderStageCreateInfo shaderStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    shaderStage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = shaderModule;
    shaderStage.pName  = "main";
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset     = 0;
    pcRange.size       = 16 * sizeof(int);
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &ctx.descSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &pipe.layout));
    VkComputePipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeInfo.stage  = shaderStage;
    pipeInfo.layout = pipe.layout;
    VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe.pipeline));
    vkDestroyShaderModule(ctx.device, shaderModule, nullptr);
    return pipe;
}

void initVulkanContext(VulkanContext& ctx, VkInstance instance, VkPhysicalDevice physDev) {
    ctx.instance       = instance;
    ctx.physicalDevice = physDev;
    VkPhysicalDeviceProperties devProps;
    vkGetPhysicalDeviceProperties(physDev, &devProps);
    strncpy(ctx.deviceName, devProps.deviceName, 255);
    ctx.deviceName[255] = '\0';
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &queueFamilyCount, queueFamilies.data());
    ctx.queueFamily = 0;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx.queueFamily = i; break;
        }
    }
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = ctx.queueFamily;
    queueInfo.queueCount       = 1;
    queueInfo.pQueuePriorities = &queuePriority;
    VkDeviceCreateInfo devInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos    = &queueInfo;
    VK_CHECK(vkCreateDevice(physDev, &devInfo, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.queue);
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags              = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex   = ctx.queueFamily;
    VK_CHECK(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.cmdPool));
    VkCommandBufferAllocateInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdInfo.commandPool        = ctx.cmdPool;
    cmdInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cmdInfo, &ctx.cmdBuffer));
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.fence));
    VkDescriptorSetLayoutBinding floatBinding{};
    floatBinding.binding         = 0;
    floatBinding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    floatBinding.descriptorCount = 1;
    floatBinding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding intBinding{};
    intBinding.binding         = 1;
    intBinding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    intBinding.descriptorCount = 1;
    intBinding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {floatBinding, intBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings    = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &ctx.descSetLayout));
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCreateInfo.maxSets       = 1;
    poolCreateInfo.poolSizeCount = 1;
    poolCreateInfo.pPoolSizes    = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(ctx.device, &poolCreateInfo, nullptr, &ctx.descPool));
    VkDescriptorSetAllocateInfo setAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setAllocInfo.descriptorPool     = ctx.descPool;
    setAllocInfo.descriptorSetCount = 1;
    setAllocInfo.pSetLayouts        = &ctx.descSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(ctx.device, &setAllocInfo, &ctx.descSet));
}

void bindBuffers(const VulkanContext& ctx, const VulkanBuffer& floatBuf, const VulkanBuffer& intBuf) {
    VkDescriptorBufferInfo floatInfo{};
    floatInfo.buffer = floatBuf.buffer;
    floatInfo.offset = 0;
    floatInfo.range  = floatBuf.size;
    VkDescriptorBufferInfo intInfo{};
    intInfo.buffer = intBuf.buffer;
    intInfo.offset = 0;
    intInfo.range  = intBuf.size;
    VkWriteDescriptorSet floatWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    floatWrite.dstSet          = ctx.descSet;
    floatWrite.dstBinding      = 0;
    floatWrite.descriptorCount = 1;
    floatWrite.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    floatWrite.pBufferInfo     = &floatInfo;
    VkWriteDescriptorSet intWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    intWrite.dstSet          = ctx.descSet;
    intWrite.dstBinding      = 1;
    intWrite.descriptorCount = 1;
    intWrite.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    intWrite.pBufferInfo     = &intInfo;
    std::array<VkWriteDescriptorSet, 2> writes = {floatWrite, intWrite};
    vkUpdateDescriptorSets(ctx.device, 2, writes.data(), 0, nullptr);
}

void beginCommandBuffer(const VulkanContext& ctx) {
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkResetCommandBuffer(ctx.cmdBuffer, 0));
    VK_CHECK(vkBeginCommandBuffer(ctx.cmdBuffer, &beginInfo));
}

void recordCompute(const VulkanContext& ctx, VulkanPipeline pipe, int workCount, const std::array<int, 16>& pushConstants) {
    vkCmdBindPipeline(ctx.cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, 1, &ctx.descSet, 0, nullptr);
    vkCmdPushConstants(ctx.cmdBuffer, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 * sizeof(int), pushConstants.data());
    int numWorkgroups = (workCount + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    vkCmdDispatch(ctx.cmdBuffer, numWorkgroups, 1, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx.cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void endCommandBuffer(const VulkanContext& ctx) { VK_CHECK(vkEndCommandBuffer(ctx.cmdBuffer)); }
void submitCommandBuffer(const VulkanContext& ctx) {
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &ctx.cmdBuffer;
    VK_CHECK(vkQueueSubmit(ctx.queue, 1, &submitInfo, ctx.fence));
}
void waitForFence(const VulkanContext& ctx) {
    VK_CHECK(vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(ctx.device, 1, &ctx.fence));
}
void submitAndWaitAll(VulkanContext* ctxs, int numGpus) {
    for (int g = 0; g < numGpus; g++) submitCommandBuffer(ctxs[g]);
    for (int g = 0; g < numGpus; g++) waitForFence(ctxs[g]);
}

// ============================================================
// Push-Constant Builders
// ============================================================
static std::array<int,16> pcEmbed(int out, int inIds, int weight, int B, int T, int C) { return {out, inIds, weight, B, T, C, 0,0,0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcQKV(int oq, int ok, int ov, int inP, int wq, int wk, int wv, int B, int T, int C, int NH, int NKV, int DK) { return {oq, ok, ov, inP, wq, wk, wv, B, T, C, NH, NKV, DK, 0, 0, 0}; }
static std::array<int,16> pcRmsNorm(int out, int rstd, int in, int weight, int B, int T, int C) { return {out, rstd, in, weight, B, T, C, 0,0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcMatMul(int out, int in, int weight, int bias, int BT, int inDim, int outDim, int hasBias) { return {out, in, weight, bias, BT, inDim, outDim, hasBias, 0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcRoPE(int out, int in, int cos, int sin, int B, int T, int numHeads, int headDim) { return {out, in, cos, sin, B, T, numHeads, headDim, 0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcGQA(int out, int attn, int q, int k, int v, int B, int T, int NH, int NKV, int DK, int NGRP) { return {out, attn, q, k, v, B, T, NH, NKV, DK, NGRP, 0,0,0,0,0}; }
static std::array<int,16> pcSwiGLU(int out, int gate, int up, int N) { return {out, gate, up, N, 0,0,0,0,0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcResidual(int out, int in1, int in2, int N) { return {out, in1, in2, N, 0,0,0,0,0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcSoftmax(int probsOut, int logitsIn, int B, int T, int V) { return {probsOut, logitsIn, B, T, V, 0,0,0,0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcCrossEntropy(int lossOut, int probsIn, int targetIn, int B, int T, int V) { return {lossOut, probsIn, targetIn, B, T, V, 0,0,0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcFillZero(int out, int N) { return {out, N, 0,0,0,0,0,0,0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcCESoftmaxBwd(int dLogits, int probs, int targets, int B, int T, int V, int dLossBits) { return {dLogits, probs, targets, B, T, V, 0, 0, 0, dLossBits, 0,0,0,0,0,0}; }
static std::array<int,16> pcEmbedBwd(int dWeight, int dOut, int inIds, int B, int T, int C) { return {dWeight, dOut, inIds, B, T, C, 0,0,0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcRmsNormBwd(int dIn, int dWeight, int dOut, int in, int weight, int rstd, int B, int T, int C) { return {dIn, dWeight, dOut, in, weight, rstd, B, T, C, 0,0,0,0,0,0,0}; }
static std::array<int,16> pcMatMulDIn(int dIn, int dOut, int weight, int BT, int inDim, int outDim) { return {dIn, dOut, weight, BT, inDim, outDim, 0,0,0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcMatMulDWeight(int dWeight, int dBias, int dOut, int in, int BT, int inDim, int outDim, int hasBias) { return {dWeight, dBias, dOut, in, BT, inDim, outDim, hasBias, 0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcRoPEBwd(int dIn, int dOut, int cos, int sin, int B, int T, int numHeads, int headDim) { return {dIn, dOut, cos, sin, B, T, numHeads, headDim, 0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcGQABwd(int dQ, int dK, int dV, int dOut, int q, int k, int v, int attn, int B, int T, int NH, int NKV, int DK, int NGRP) { return {dQ, dK, dV, dOut, q, k, v, attn, B, T, NH, NKV, DK, NGRP, 0, 0}; }
static std::array<int,16> pcSwiGLUBwd(int dGate, int dUp, int dOut, int gate, int up, int N) { return {dGate, dUp, dOut, gate, up, N, 0,0,0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcResidualBwd(int d1, int d2, int dOut, int N) { return {d1, d2, dOut, N, 0,0,0,0,0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcAdamW(int params, int grads, int m, int v, int N, int step) { return {params, grads, m, v, N, step, 0,0,0,0,0,0,0,0,0,0}; }
static std::array<int,16> pcGateUpSwiGLU(int o, int inp, int wg, int wu, int B, int T, int C, int DFF) {
    return {o, inp, wg, wu, B, T, C, DFF, 0, 0, 0, 0, 0, 0, 0, 0};
}

static std::array<int,16> pcSwiGLUBwdRecomp(int dg, int du, int do2, int inp, int wg, int wu, int B, int T, int C, int DFF) {
    return {dg, du, do2, inp, wg, wu, B, T, C, DFF, 0, 0, 0, 0, 0, 0};
}

static std::array<int,16> pcResRms(int o, int rstd, int i1, int i2, int w, int B, int T, int C) {
    return {o, rstd, i1, i2, w, B, T, C, 0,0,0,0,0,0,0,0};
}

static std::array<int,16> pcResRmsBwd(int d1, int d2, int dw, int do2, int i1, int i2, int w, int rstd, int B, int T, int C) {
    return {d1, d2, dw, do2, i1, i2, w, rstd, B, T, C, 0,0,0,0,0};
}

static std::array<int,16> pcDownRes(int o, int sg, int wd, int res_prev, int attproj, int B, int T, int C, int DFF) {
    return {o, sg, wd, res_prev, attproj, B, T, C, DFF, 0, 0, 0, 0, 0, 0, 0};
}

static std::array<int,16> pcResidBwd3Way(int d1, int d2, int d3, int do2, int N) {
    return {d1, d2, d3, do2, N, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}

static std::array<int,16> pcQKNormRoPE(int o, int rs, int inp, int w, int co, int si, int B, int T, int nh, int dk) {
    return {o, rs, inp, w, co, si, B, T, nh, dk, 0, 0, 0, 0, 0, 0};
}

static std::array<int,16> pcQKNormRoPEBwd(int di, int dw, int do2, int inp, int w, int rs, int co, int si, int B, int T, int nh, int dk) {
    return {di, dw, do2, inp, w, rs, co, si, B, T, nh, dk, 0, 0, 0, 0};
}

// ============================================================
// Model Definition
// ============================================================

struct Qwen3Model {
    int vocab_size, batch_size;
    size_t num_params;
    std::vector<size_t> param_off, grad_off;
    size_t m_off, v_off;

    size_t off_embed, off_rms1_out, off_rms1_rstd;
    size_t off_q, off_k, off_v, off_q_norm, off_k_norm, off_q_norm_rstd, off_k_norm_rstd;
    size_t off_q_rot, off_k_rot, off_rope_cos, off_rope_sin;
    size_t off_attn_out, off_attn_scores, off_attn_proj, off_res2;
    size_t off_rms2_out, off_rms2_rstd, off_gate, off_up_proj, off_swiglu, off_down, off_res3;
    size_t off_final_norm, off_final_rstd, off_logits, off_probs, off_losses;

    size_t off_d_embed, off_d_rms1, off_dq, off_dk, off_dv;
    size_t off_d_q_norm, off_d_k_norm, off_d_q_rot, off_d_k_rot;
    size_t off_d_attn_out, off_d_attn_proj, off_d_res2, off_d_rms2;
    size_t off_d_gate, off_d_up, off_d_swiglu, off_d_down, off_d_res3, off_d_final_norm, off_d_logits;

    size_t off_input_ids, off_target_ids;
};

std::vector<size_t> getParamSizes(int vocab_size) {
    return {
        (size_t)vocab_size * C, (size_t)L * C, (size_t)L * NH * DK * C, (size_t)L * NKV * DK * C,
        (size_t)L * NKV * DK * C, (size_t)L * DK, (size_t)L * DK, (size_t)L * C * C, (size_t)L * C,
        (size_t)L * DFF * C, (size_t)L * DFF * C, (size_t)L * C * DFF, (size_t)C, (size_t)vocab_size * C,
    };
}

size_t allocModelBuffers(Qwen3Model* m, int vocab_size, int batch_size) {
    m->vocab_size  = vocab_size; m->batch_size  = batch_size;
    auto sizes = getParamSizes(vocab_size);
    m->num_params = 0;
    for (auto s : sizes) m->num_params += s;
    m->param_off.resize(sizes.size()); m->grad_off.resize(sizes.size());
    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); i++) { m->param_off[i] = offset; offset += sizes[i]; }
    m->m_off = offset; offset += m->num_params;
    m->v_off = offset; offset += m->num_params;
    for (size_t i = 0; i < sizes.size(); i++) { m->grad_off[i] = offset + m->param_off[i]; }
    offset += m->num_params;

    auto alloc = [&](size_t sz) -> size_t { size_t r = offset; offset += sz; return r; };
    int BT = batch_size * T, BTC = batch_size * T * C;
    int BT_NH_DK = batch_size * T * NH * DK, BT_NKV_DK = batch_size * T * NKV * DK;
    int BT_DFF = batch_size * T * DFF, B_NH_TT = batch_size * NH * T * T, T_DH = T * (DK / 2);

    m->off_embed = alloc(BTC); m->off_rms1_out = alloc(L*BTC); m->off_rms1_rstd = alloc(L*BT);
    m->off_q = alloc(L*BT_NH_DK); m->off_k = alloc(L*BT_NKV_DK); m->off_v = alloc(L*BT_NKV_DK);
    m->off_q_norm = alloc(L*BT_NH_DK); m->off_k_norm = alloc(L*BT_NKV_DK);
    m->off_q_norm_rstd = alloc(L*batch_size*T*NH); m->off_k_norm_rstd = alloc(L*batch_size*T*NKV);
    m->off_q_rot = alloc(L*BT_NH_DK); m->off_k_rot = alloc(L*BT_NKV_DK);
    m->off_rope_cos = alloc(T_DH); m->off_rope_sin = alloc(T_DH);
    m->off_attn_out = alloc(L*BTC); m->off_attn_scores = alloc(L*B_NH_TT); m->off_attn_proj = alloc(L*BTC); m->off_res2 = alloc(L*BTC);
    m->off_rms2_out = alloc(L*BTC); m->off_rms2_rstd = alloc(L*BT);
    m->off_gate = alloc(L*BT_DFF); m->off_up_proj = alloc(L*BT_DFF); m->off_swiglu = alloc(L*BT_DFF);
    m->off_down = alloc(L*BTC); m->off_res3 = alloc(L*BTC);
    m->off_final_norm = alloc(BTC); m->off_final_rstd = alloc(BT);
    m->off_logits = alloc(batch_size*T*vocab_size); m->off_probs = alloc(batch_size*T*vocab_size); m->off_losses = alloc(BT);

    m->off_d_embed = alloc(BTC); m->off_d_rms1 = alloc(L*BTC);
    m->off_dq = alloc(L*BT_NH_DK); m->off_dk = alloc(L*BT_NKV_DK); m->off_dv = alloc(L*BT_NKV_DK);
    m->off_d_q_norm = alloc(L*BT_NH_DK); m->off_d_k_norm = alloc(L*BT_NKV_DK);
    m->off_d_q_rot = alloc(L*BT_NH_DK); m->off_d_k_rot = alloc(L*BT_NKV_DK);
    m->off_d_attn_out = alloc(L*BTC); m->off_d_attn_proj = alloc(L*BTC); m->off_d_res2 = alloc(L*BTC); m->off_d_rms2 = alloc(L*BTC);
    m->off_d_gate = alloc(L*BT_DFF); m->off_d_up = alloc(L*BT_DFF); m->off_d_swiglu = alloc(L*BT_DFF);
    m->off_d_down = alloc(L*BTC); m->off_d_res3 = alloc(L*BTC);
    m->off_d_final_norm = alloc(BTC); m->off_d_logits = alloc(batch_size*T*vocab_size);

    m->off_input_ids = 0; m->off_target_ids = BT;
    return offset;
}

void saveCheckpoint(const float* data, const Qwen3Model& m, int step, const char* filepath) {
    FILE* file = fopen(filepath, "wb");
    if (!file) return;
    fwrite(&step, 4, 1, file);
    fwrite(data, 4, m.num_params, file); 
    fwrite(data + m.m_off, 4, m.num_params, file); 
    fwrite(data + m.v_off, 4, m.num_params, file);
    fclose(file);
    LOG("Saved checkpoint at step %d", step);
}

int loadCheckpoint(float* data, const Qwen3Model& m, const char* filepath) {
    FILE* file = fopen(filepath, "rb");
    if (!file) return 0;
    int step; fread(&step, 4, 1, file);
    fread(data, 4, m.num_params, file);
    fread(data + m.m_off, 4, m.num_params, file);
    fread(data + m.v_off, 4, m.num_params, file);
    fclose(file);
    LOG("Loaded checkpoint from step %d", step);
    return step;
}

void precomputeRopeTables(float* cos_table, float* sin_table, int max_len, int head_dim) {
    int half = head_dim / 2;
    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; i++) inv_freq[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)head_dim);
    for (int t = 0; t < max_len; t++) {
        for (int j = 0; j < half; j++) {
            float theta = (float)t * inv_freq[j];
            cos_table[t * half + j] = cosf(theta);
            sin_table[t * half + j] = sinf(theta);
        }
    }
}

using Clock = std::chrono::high_resolution_clock;

int main() {
    srand(1337);
    std::ifstream file("input.txt");
    if (!file.is_open()) { LOG_ERR("Cannot open input.txt"); return 1; }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    CharTokenizer tokenizer; tokenizer.init(text);
    int V = tokenizer.vocab_size;
    LOG("Vocabulary size: %d", V);

    std::vector<int> tokens = tokenizer.encode(text);
    int split = (int)(0.9 * tokens.size());
    std::vector<int> train_data(tokens.begin(), tokens.begin() + split);
    std::vector<int> val_data(tokens.begin() + split, tokens.end());

    int num_gpus = std::min(NUM_GPUS, MAX_GPUS);
    VulkanContext contexts[MAX_GPUS];
    VkInstance instance;
    {
        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = "Qwen3_Vulkan_Training"; appInfo.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instInfo.pApplicationInfo = &appInfo;
        VK_CHECK(vkCreateInstance(&instInfo, nullptr, &instance));
    }
    {
        uint32_t deviceCount = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
        std::vector<VkPhysicalDevice> physDevices(deviceCount);
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, physDevices.data()));
        num_gpus = std::min((int)deviceCount, num_gpus);
        LOG("Found %d physical device(s), using %d", deviceCount, num_gpus);
        for (int g = 0; g < num_gpus; g++) {
            initVulkanContext(contexts[g], instance, physDevices[g]);
            VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(physDevices[g], &props);
            LOG("  GPU %d: %s", g, props.deviceName);
        }
    }

    int batch_per_gpu[MAX_GPUS];
    if (num_gpus == 3) {
        float weights[] = {0.488f, 0.256f, 0.256f};
        int allocated = 0;
        for (int g = 0; g < 3; g++) {
            batch_per_gpu[g] = (int)std::round(B_TOTAL * weights[g]);
            if (batch_per_gpu[g] < 1) batch_per_gpu[g] = 1;
            allocated += batch_per_gpu[g];
        }
        batch_per_gpu[0] += (B_TOTAL - allocated);
    } else if (num_gpus == 2) {
        batch_per_gpu[0] = (int)std::round(B_TOTAL * 0.6f);
        batch_per_gpu[1] = B_TOTAL - batch_per_gpu[0];
    } else {
        for (int g = 0; g < num_gpus; g++) batch_per_gpu[g] = B_TOTAL / num_gpus;
    }
    int total_batch = 0;
    for (int g = 0; g < num_gpus; g++) total_batch += batch_per_gpu[g];
    LOG("Batch sizes: total=%d", total_batch);

    Qwen3Model models[MAX_GPUS];
    size_t float_buf_size[MAX_GPUS], int_buf_size[MAX_GPUS];
    VulkanBuffer floatBufs[MAX_GPUS], stagingBufs[MAX_GPUS], intBufs[MAX_GPUS];

    for (int g = 0; g < num_gpus; g++) {
        float_buf_size[g] = allocModelBuffers(&models[g], V, batch_per_gpu[g]);
        int_buf_size[g]   = 2 * batch_per_gpu[g] * T;

        createBuffer(contexts[g], float_buf_size[g] * sizeof(float),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, floatBufs[g]);
        createBuffer(contexts[g], float_buf_size[g] * sizeof(float),
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufs[g]);
        createBuffer(contexts[g], int_buf_size[g] * sizeof(int),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, intBufs[g]);
        bindBuffers(contexts[g], floatBufs[g], intBufs[g]);
        memset(stagingBufs[g].mappedPtr, 0, float_buf_size[g] * sizeof(float));
        memset(intBufs[g].mappedPtr, 0, int_buf_size[g] * sizeof(int));
    }

    float* f0 = (float*)stagingBufs[0].mappedPtr;
    int start_iter = loadCheckpoint(f0, models[0], "qwen3_ckpt.bin");
    // if (start_iter > 0) { LOG("Removing old checkpoint to restart from scratch"); remove("qwen3_ckpt.bin"); start_iter = 0; }
    if (start_iter == 0) {
        std::mt19937 rng(1337); std::normal_distribution<float> normal_dist(0.0f, 0.02f);
        auto init_random = [&](size_t off, size_t count) { for (size_t i = 0; i < count; i++) f0[off + i] = normal_dist(rng); };
        auto init_constant = [&](size_t off, size_t count, float val) { for (size_t i = 0; i < count; i++) f0[off + i] = val; };
        auto& po = models[0].param_off; auto sizes = getParamSizes(V);
        float res_scale = 1.0f / sqrtf(2.0f * L);
        init_random(po[PARAM_TOKEN_EMB], sizes[PARAM_TOKEN_EMB]);
        for (int l = 0; l < L; l++) {
            init_constant(po[PARAM_ATTN_NORM] + l * C, C, 1.0f);
            init_random(po[PARAM_WQ] + l * NH * DK * C, sizes[PARAM_WQ] / L);
            init_random(po[PARAM_WK] + l * NKV * DK * C, sizes[PARAM_WK] / L);
            init_random(po[PARAM_WV] + l * NKV * DK * C, sizes[PARAM_WV] / L);
            init_constant(po[PARAM_Q_NORM] + l * DK, DK, 1.0f);
            init_constant(po[PARAM_K_NORM] + l * DK, DK, 1.0f);
            for (size_t i = 0; i < sizes[PARAM_ATTN_PROJ] / L; i++) f0[po[PARAM_ATTN_PROJ] + l * C * C + i] = normal_dist(rng) * res_scale;
            init_constant(po[PARAM_FFN_NORM] + l * C, C, 1.0f);
            init_random(po[PARAM_FFN_GATE] + l * DFF * C, sizes[PARAM_FFN_GATE] / L);
            init_random(po[PARAM_FFN_UP] + l * DFF * C, sizes[PARAM_FFN_UP] / L);
            for (size_t i = 0; i < sizes[PARAM_FFN_DOWN] / L; i++) f0[po[PARAM_FFN_DOWN] + l * C * DFF + i] = normal_dist(rng) * res_scale;
        }
        init_constant(po[PARAM_FINAL_NORM], sizes[PARAM_FINAL_NORM], 1.0f);
        init_random(po[PARAM_LM_HEAD], sizes[PARAM_LM_HEAD]);
        memset(f0 + models[0].m_off, 0, models[0].num_params * sizeof(float));
        memset(f0 + models[0].v_off, 0, models[0].num_params * sizeof(float));
    }
    for (int g = 1; g < num_gpus; g++) {
        float* fg = (float*)stagingBufs[g].mappedPtr;
        memcpy(fg, f0, models[0].num_params * sizeof(float));
        memcpy(fg + models[g].m_off, f0 + models[0].m_off, models[0].num_params * sizeof(float));
        memcpy(fg + models[g].v_off, f0 + models[0].v_off, models[0].num_params * sizeof(float));
    }
    for (int g = 0; g < num_gpus; g++) {
        std::vector<float> rope_cos(T * (DK / 2)), rope_sin(T * (DK / 2));
        precomputeRopeTables(rope_cos.data(), rope_sin.data(), T, DK);
        float* fg = (float*)stagingBufs[g].mappedPtr;
        memcpy(fg + models[g].off_rope_cos, rope_cos.data(), rope_cos.size() * sizeof(float));
        memcpy(fg + models[g].off_rope_sin, rope_sin.data(), rope_sin.size() * sizeof(float));
    }
    for (int g = 0; g < num_gpus; g++) {
        beginCommandBuffer(contexts[g]);
        VkBufferCopy region{0, 0, float_buf_size[g] * sizeof(float)};
        vkCmdCopyBuffer(contexts[g].cmdBuffer, stagingBufs[g].buffer, floatBufs[g].buffer, 1, &region);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(contexts[g].cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        endCommandBuffer(contexts[g]); submitCommandBuffer(contexts[g]);
    }
    for (int g = 0; g < num_gpus; g++) waitForFence(contexts[g]);

    auto makePipelines = [&](const char* src, const char* nm) -> std::array<VulkanPipeline, MAX_GPUS> {
        std::array<VulkanPipeline, MAX_GPUS> pipes; auto spv = compileGlslToSpv(src, nm);
        for (int g = 0; g < num_gpus; g++) pipes[g] = createComputePipeline(contexts[g], createShaderModule(contexts[g], spv));
        return pipes;
    };
    auto pipeEmbed = makePipelines(sh_embed, "embed"); 
    auto pipeRmsNorm = makePipelines(sh_rmsnorm, "rmsnorm");
    auto pipeResRms  = makePipelines(sh_res_rms, "res_rms"); 
    auto pipeMatMul = makePipelines(sh_matmul, "matmul");
    auto pipeQKV = makePipelines(sh_qkv, "qkv");
    auto pipeRoPE = makePipelines(sh_rope, "rope"); 
    auto pipeQKNormRoPE = makePipelines(sh_qk_norm_rope, "qk_norm_rope");
    auto pipeGQA = makePipelines(sh_gqa, "gqa");
    auto pipeSwiGLU = makePipelines(sh_swiglu, "swiglu"); 
    auto pipeResid = makePipelines(sh_residual, "residual");//keeping old to see change
    auto pipeDownRes = makePipelines(sh_down_res, "down_res"); 
    auto pipeGateUpSwiGLU = makePipelines(sh_gate_up_swiglu, "gate_up_swiglu");
    auto pipeSoftmax = makePipelines(sh_softmax, "softmax"); auto pipeCE = makePipelines(sh_cross_entropy, "cross_entropy");
    auto pipeZero = makePipelines(sh_fill_zero, "fill_zero");
    auto pipeCESoftmaxBwd = makePipelines(sh_ce_softmax_bwd, "ce_softmax_bwd"); auto pipeEmbedBwd = makePipelines(sh_embed_bwd, "embed_bwd");
    auto pipeRmsNormBwd = makePipelines(sh_rmsnorm_bwd, "rmsnorm_bwd");
    auto pipeResRmsBwd  = makePipelines(sh_res_rms_bwd, "res_rms_bwd"); 
    auto pipeMatMulDIn = makePipelines(sh_matmul_din, "matmul_din");
    auto pipeMatMulDWeight = makePipelines(sh_matmul_dweight, "matmul_dweight"); 
    auto pipeRoPEBwd = makePipelines(sh_rope_bwd, "rope_bwd");
    auto pipeQKNormRoPEBwd = makePipelines(sh_qk_norm_rope_bwd, "qk_norm_rope_bwd");
    auto pipeGQABwd = makePipelines(sh_gqa_bwd, "gqa_bwd"); 
    auto pipeSwiGLUBwd = makePipelines(sh_swiglu_bwd, "swiglu_bwd");
    auto pipeSwiGLUBwdRecomp = makePipelines(sh_swiglu_bwd_recomp, "swiglu_bwd_recomp");
    auto pipeResidBwd = makePipelines(sh_residual_bwd, "residual_bwd"); 
    auto pipeResidBwd3Way = makePipelines(sh_resid_bwd_3way, "resid_bwd_3way");
    auto pipeAdamW = makePipelines(sh_adamw, "adamw");
    auto param_sizes = getParamSizes(V);

    auto recordForwardPass = [&](int g, bool compute_loss) {
        auto& ctx = contexts[g]; auto& m = models[g]; int Bl = batch_per_gpu[g];
        int BlT = Bl * T, BlTC = Bl * T * C, BlTNHDK = Bl * T * NH * DK, BlTNKVDK = Bl * T * NKV * DK;
        int BlTDFF = Bl * T * DFF, BlNHTT = Bl * NH * T * T; auto po = m.param_off;
        recordCompute(ctx, pipeEmbed[g], BlTC, pcEmbed(m.off_embed, m.off_input_ids, po[PARAM_TOKEN_EMB], Bl, T, C));
        for (int l = 0; l < L; l++) {
            size_t layer_input = (l == 0) ? m.off_embed : m.off_res3 + (l - 1) * BlTC;
            recordCompute(ctx, pipeRmsNorm[g], BlTC, pcRmsNorm(m.off_rms1_out + l * BlTC, m.off_rms1_rstd + l * BlT, layer_input, po[PARAM_ATTN_NORM] + l * C, Bl, T, C));
            int total_qkv_threads = BlT * (NH + 2 * NKV) * DK;
            recordCompute(ctx, pipeQKV[g], total_qkv_threads, pcQKV(m.off_q + l * BlTNHDK, m.off_k + l * BlTNKVDK, m.off_v + l * BlTNKVDK, m.off_rms1_out + l * BlTC, po[PARAM_WQ] + l * NH * DK * C, po[PARAM_WK] + l * NKV * DK * C, po[PARAM_WV] + l * NKV * DK * C, Bl, T, C, NH, NKV, DK));
            
                        // FUSED QK-Norm + RoPE
            int total_q_pairs = BlT * NH * (DK / 2);
            recordCompute(ctx, pipeQKNormRoPE[g], total_q_pairs,
                pcQKNormRoPE(m.off_q_rot + l * BlTNHDK, m.off_q_norm_rstd + l * Bl * T * NH,
                             m.off_q + l * BlTNHDK, po[PARAM_Q_NORM] + l * DK,
                             m.off_rope_cos, m.off_rope_sin, Bl, T, NH, DK));
                             
            int total_k_pairs = BlT * NKV * (DK / 2);
            recordCompute(ctx, pipeQKNormRoPE[g], total_k_pairs,
                pcQKNormRoPE(m.off_k_rot + l * BlTNKVDK, m.off_k_norm_rstd + l * Bl * T * NKV,
                             m.off_k + l * BlTNKVDK, po[PARAM_K_NORM] + l * DK,
                             m.off_rope_cos, m.off_rope_sin, Bl, T, NKV, DK));


            recordCompute(ctx, pipeGQA[g], BlT * NH, pcGQA(m.off_attn_out + l * BlTC, m.off_attn_scores + l * BlNHTT, m.off_q_rot + l * BlTNHDK, m.off_k_rot + l * BlTNKVDK, m.off_v + l * BlTNKVDK, Bl, T, NH, NKV, DK, NGRP));
            recordCompute(ctx, pipeMatMul[g], BlTC, pcMatMul(m.off_attn_proj + l * BlTC, m.off_attn_out + l * BlTC, po[PARAM_ATTN_PROJ] + l * C * C, 0, BlT, C, C, 0));
                       // --- FFN block ---
            // FUSED Residual + RMSNorm before FFN
            recordCompute(ctx, pipeResRms[g], BlTC,
                pcResRms(m.off_rms2_out + l * BlTC, m.off_rms2_rstd + l * BlT,
                         layer_input, m.off_attn_proj + l * BlTC,
                         po[PARAM_FFN_NORM] + l * C, Bl, T, C));
                        // FUSED Gate + Up + SwiGLU
            recordCompute(ctx, pipeGateUpSwiGLU[g], BlTDFF,
                pcGateUpSwiGLU(m.off_swiglu + l * BlTDFF, m.off_rms2_out + l * BlTC,
                               po[PARAM_FFN_GATE] + l * DFF * C, po[PARAM_FFN_UP] + l * DFF * C,
                               Bl, T, C, DFF));
                        // FUSED Down projection + Residual connection
            recordCompute(ctx, pipeDownRes[g], BlTC,
                pcDownRes(m.off_res3 + l * BlTC, m.off_swiglu + l * BlTDFF,
                          po[PARAM_FFN_DOWN] + l * C * DFF,
                          layer_input, m.off_attn_proj + l * BlTC, Bl, T, C, DFF));
        }
        size_t final_input = m.off_res3 + (L - 1) * BlTC;
        recordCompute(ctx, pipeRmsNorm[g], BlTC, pcRmsNorm(m.off_final_norm, m.off_final_rstd, final_input, po[PARAM_FINAL_NORM], Bl, T, C));
        recordCompute(ctx, pipeMatMul[g], BlT * V, pcMatMul(m.off_logits, m.off_final_norm, po[PARAM_LM_HEAD], 0, BlT, C, V, 0));
        recordCompute(ctx, pipeSoftmax[g], BlT, pcSoftmax(m.off_probs, m.off_logits, Bl, T, V));
        if (compute_loss) recordCompute(ctx, pipeCE[g], BlT, pcCrossEntropy(m.off_losses, m.off_probs, m.off_target_ids, Bl, T, V));
    };

    auto recordLayerBackward = [&](int g, int l, size_t layer_input, size_t d_layer_input) {
        auto& ctx = contexts[g]; auto& m = models[g]; int Bl = batch_per_gpu[g];
        int BlT = Bl * T, BlTC = Bl * T * C, BlTNHDK = Bl * T * NH * DK, BlTNKVDK = Bl * T * NKV * DK;
        int BlTDFF = Bl * T * DFF, BlNHTT = Bl * NH * T * T; auto po = m.param_off; auto go = m.grad_off;
        // --- Backward through FFN residual (3-way split) ---
        // Splits d_res3 directly into d_layer_input, d_attn_proj, and d_down
        recordCompute(ctx, pipeResidBwd3Way[g], BlTC,
            pcResidBwd3Way(d_layer_input, m.off_d_attn_proj + l * BlTC, 
                           m.off_d_down + l * BlTC, m.off_d_res3 + l * BlTC, BlTC));
        recordCompute(ctx, pipeMatMulDIn[g], BlTDFF, pcMatMulDIn(m.off_d_swiglu + l * BlTDFF, m.off_d_down + l * BlTC, po[PARAM_FFN_DOWN] + l * C * DFF, BlT, DFF, C));
        recordCompute(ctx, pipeMatMulDWeight[g], C * DFF, pcMatMulDWeight(go[PARAM_FFN_DOWN] + l * C * DFF, 0, m.off_d_down + l * BlTC, m.off_swiglu + l * BlTDFF, BlT, DFF, C, 0));
                // --- Backward through SwiGLU (Recomputes Gate/Up internally) ---
        recordCompute(ctx, pipeSwiGLUBwdRecomp[g], BlTDFF,
            pcSwiGLUBwdRecomp(m.off_d_gate + l * BlTDFF, m.off_d_up + l * BlTDFF,
                              m.off_d_swiglu + l * BlTDFF, m.off_rms2_out + l * BlTC,
                              po[PARAM_FFN_GATE] + l * DFF * C, po[PARAM_FFN_UP] + l * DFF * C,
                              Bl, T, C, DFF));
        recordCompute(ctx, pipeMatMulDIn[g], BlTC, pcMatMulDIn(m.off_d_rms2 + l * BlTC, m.off_d_gate + l * BlTDFF, po[PARAM_FFN_GATE] + l * DFF * C, BlT, C, DFF));
        recordCompute(ctx, pipeMatMulDWeight[g], DFF * C, pcMatMulDWeight(go[PARAM_FFN_GATE] + l * DFF * C, 0, m.off_d_gate + l * BlTDFF, m.off_rms2_out + l * BlTC, BlT, C, DFF, 0));
        recordCompute(ctx, pipeMatMulDIn[g], BlTC, pcMatMulDIn(m.off_d_rms2 + l * BlTC, m.off_d_up + l * BlTDFF, po[PARAM_FFN_UP] + l * DFF * C, BlT, C, DFF));
        recordCompute(ctx, pipeMatMulDWeight[g], DFF * C, pcMatMulDWeight(go[PARAM_FFN_UP] + l * DFF * C, 0, m.off_d_up + l * BlTDFF, m.off_rms2_out + l * BlTC, BlT, C, DFF, 0));
                // --- FUSED Backward through FFN RMSNorm + Attention Residual ---
        recordCompute(ctx, pipeResRmsBwd[g], BlTC,
            pcResRmsBwd(d_layer_input, m.off_d_attn_proj + l * BlTC,
                        go[PARAM_FFN_NORM] + l * C, m.off_d_rms2 + l * BlTC,
                        layer_input, m.off_attn_proj + l * BlTC,
                        po[PARAM_FFN_NORM] + l * C, m.off_rms2_rstd + l * BlT,
                        Bl, T, C));
        recordCompute(ctx, pipeMatMulDIn[g], BlTC, pcMatMulDIn(m.off_d_attn_out + l * BlTC, m.off_d_attn_proj + l * BlTC, po[PARAM_ATTN_PROJ] + l * C * C, BlT, C, C));
        recordCompute(ctx, pipeMatMulDWeight[g], C * C, pcMatMulDWeight(go[PARAM_ATTN_PROJ] + l * C * C, 0, m.off_d_attn_proj + l * BlTC, m.off_attn_out + l * BlTC, BlT, C, C, 0));
        recordCompute(ctx, pipeGQABwd[g], BlT * NH, pcGQABwd(m.off_d_q_rot + l * BlTNHDK, m.off_d_k_rot + l * BlTNKVDK, m.off_dv + l * BlTNKVDK, m.off_d_attn_out + l * BlTC, m.off_q_rot + l * BlTNHDK, m.off_k_rot + l * BlTNKVDK, m.off_v + l * BlTNKVDK, m.off_attn_scores + l * BlNHTT, Bl, T, NH, NKV, DK, NGRP));
                // --- FUSED Backward through QK-Norm + RoPE ---
        recordCompute(ctx, pipeQKNormRoPEBwd[g], BlT * NH * (DK / 2),
            pcQKNormRoPEBwd(m.off_dq + l * BlTNHDK, go[PARAM_Q_NORM] + l * DK,
                            m.off_d_q_rot + l * BlTNHDK, m.off_q + l * BlTNHDK,
                            po[PARAM_Q_NORM] + l * DK, m.off_q_norm_rstd + l * Bl * T * NH,
                            m.off_rope_cos, m.off_rope_sin, Bl, T, NH, DK));
                            
        recordCompute(ctx, pipeQKNormRoPEBwd[g], BlT * NKV * (DK / 2),
            pcQKNormRoPEBwd(m.off_dk + l * BlTNKVDK, go[PARAM_K_NORM] + l * DK,
                            m.off_d_k_rot + l * BlTNKVDK, m.off_k + l * BlTNKVDK,
                            po[PARAM_K_NORM] + l * DK, m.off_k_norm_rstd + l * Bl * T * NKV,
                            m.off_rope_cos, m.off_rope_sin, Bl, T, NKV, DK));
        recordCompute(ctx, pipeMatMulDIn[g], BlTC, pcMatMulDIn(m.off_d_rms1 + l * BlTC, m.off_dq + l * BlTNHDK, po[PARAM_WQ] + l * NH * DK * C, BlT, C, NH * DK));
        recordCompute(ctx, pipeMatMulDWeight[g], NH * DK * C, pcMatMulDWeight(go[PARAM_WQ] + l * NH * DK * C, 0, m.off_dq + l * BlTNHDK, m.off_rms1_out + l * BlTC, BlT, C, NH * DK, 0));
        recordCompute(ctx, pipeMatMulDIn[g], BlTC, pcMatMulDIn(m.off_d_rms1 + l * BlTC, m.off_dk + l * BlTNKVDK, po[PARAM_WK] + l * NKV * DK * C, BlT, C, NKV * DK));
        recordCompute(ctx, pipeMatMulDWeight[g], NKV * DK * C, pcMatMulDWeight(go[PARAM_WK] + l * NKV * DK * C, 0, m.off_dk + l * BlTNKVDK, m.off_rms1_out + l * BlTC, BlT, C, NKV * DK, 0));
        recordCompute(ctx, pipeMatMulDIn[g], BlTC, pcMatMulDIn(m.off_d_rms1 + l * BlTC, m.off_dv + l * BlTNKVDK, po[PARAM_WV] + l * NKV * DK * C, BlT, C, NKV * DK));
        recordCompute(ctx, pipeMatMulDWeight[g], NKV * DK * C, pcMatMulDWeight(go[PARAM_WV] + l * NKV * DK * C, 0, m.off_dv + l * BlTNKVDK, m.off_rms1_out + l * BlTC, BlT, C, NKV * DK, 0));
        recordCompute(ctx, pipeRmsNormBwd[g], BlTC, pcRmsNormBwd(d_layer_input, go[PARAM_ATTN_NORM] + l * C, m.off_d_rms1 + l * BlTC, layer_input, po[PARAM_ATTN_NORM] + l * C, m.off_rms1_rstd + l * BlT, Bl, T, C));
    };

    auto recordZeroGradients = [&](int g) {
        auto& ctx = contexts[g]; auto& m = models[g]; int Bl = batch_per_gpu[g];
        int BlTC = Bl * T * C, BlTNHDK = Bl * T * NH * DK, BlTNKVDK = Bl * T * NKV * DK, BlTDFF = Bl * T * DFF;
        for (size_t i = 0; i < m.grad_off.size(); i++) {
            if (param_sizes[i] > 0) recordCompute(ctx, pipeZero[g], (int)param_sizes[i], pcFillZero((int)m.grad_off[i], (int)param_sizes[i]));
        }
        recordCompute(ctx, pipeZero[g], BlTC, pcFillZero(m.off_d_embed, BlTC));
        recordCompute(ctx, pipeZero[g], L * BlTC, pcFillZero(m.off_d_rms1, L * BlTC));
        recordCompute(ctx, pipeZero[g], L * BlTNHDK, pcFillZero(m.off_dq, L * BlTNHDK));
        recordCompute(ctx, pipeZero[g], L * BlTNKVDK, pcFillZero(m.off_dk, L * BlTNKVDK));
        recordCompute(ctx, pipeZero[g], L * BlTNKVDK, pcFillZero(m.off_dv, L * BlTNKVDK));
        recordCompute(ctx, pipeZero[g], L * BlTNHDK, pcFillZero(m.off_d_q_norm, L * BlTNHDK));
        recordCompute(ctx, pipeZero[g], L * BlTNKVDK, pcFillZero(m.off_d_k_norm, L * BlTNKVDK));
        recordCompute(ctx, pipeZero[g], L * BlTNHDK, pcFillZero(m.off_d_q_rot, L * BlTNHDK));
        recordCompute(ctx, pipeZero[g], L * BlTNKVDK, pcFillZero(m.off_d_k_rot, L * BlTNKVDK));
        recordCompute(ctx, pipeZero[g], L * BlTC, pcFillZero(m.off_d_attn_out, L * BlTC));
        recordCompute(ctx, pipeZero[g], L * BlTC, pcFillZero(m.off_d_attn_proj, L * BlTC));
        recordCompute(ctx, pipeZero[g], L * BlTC, pcFillZero(m.off_d_res2, L * BlTC));
        recordCompute(ctx, pipeZero[g], L * BlTC, pcFillZero(m.off_d_rms2, L * BlTC));
        recordCompute(ctx, pipeZero[g], L * BlTDFF, pcFillZero(m.off_d_gate, L * BlTDFF));
        recordCompute(ctx, pipeZero[g], L * BlTDFF, pcFillZero(m.off_d_up, L * BlTDFF));
        recordCompute(ctx, pipeZero[g], L * BlTDFF, pcFillZero(m.off_d_swiglu, L * BlTDFF));
        recordCompute(ctx, pipeZero[g], L * BlTC, pcFillZero(m.off_d_down, L * BlTC));
        recordCompute(ctx, pipeZero[g], L * BlTC, pcFillZero(m.off_d_res3, L * BlTC));
        recordCompute(ctx, pipeZero[g], BlTC, pcFillZero(m.off_d_final_norm, BlTC));
        recordCompute(ctx, pipeZero[g], Bl * T * V, pcFillZero(m.off_d_logits, Bl * T * V));
    };

    std::vector<float> grad_avg(models[0].num_params);
    float dLoss_val = 1.0f / (float)(total_batch * T);
    int dLoss_bits = floatToIntBits(dLoss_val);

    auto evalLoss = [&](const std::vector<int>& dataset) -> float {
        float total_loss = 0.0f; int total_count = 0; int data_len = (int)dataset.size();
        std::vector<int> input_ids(total_batch * T), target_ids(total_batch * T);
        std::vector<float> losses(total_batch * T);
        for (int k = 0; k < EVAL_ITERS; k++) {
            for (int b = 0; b < total_batch; b++) {
                int start = rand() % (data_len - T);
                for (int t = 0; t < T; t++) { input_ids[b * T + t] = dataset[start + t]; target_ids[b * T + t] = dataset[start + t + 1]; }
            }
            int offset = 0;
            for (int g = 0; g < num_gpus; g++) {
                memcpy((int*)intBufs[g].mappedPtr + models[g].off_input_ids, input_ids.data() + offset * T, batch_per_gpu[g] * T * sizeof(int));
                memcpy((int*)intBufs[g].mappedPtr + models[g].off_target_ids, target_ids.data() + offset * T, batch_per_gpu[g] * T * sizeof(int));
                offset += batch_per_gpu[g];
            }
            for (int g = 0; g < num_gpus; g++) {
                beginCommandBuffer(contexts[g]); recordForwardPass(g, true); endCommandBuffer(contexts[g]);
            }
            submitAndWaitAll(contexts, num_gpus);
            for (int g = 0; g < num_gpus; g++) {
                beginCommandBuffer(contexts[g]);
                VkBufferCopy loss_region{models[g].off_losses * sizeof(float), models[g].off_losses * sizeof(float), batch_per_gpu[g] * T * sizeof(float)};
                vkCmdCopyBuffer(contexts[g].cmdBuffer, floatBufs[g].buffer, stagingBufs[g].buffer, 1, &loss_region);
                endCommandBuffer(contexts[g]); submitCommandBuffer(contexts[g]);
            }
            for (int g = 0; g < num_gpus; g++) waitForFence(contexts[g]);
            offset = 0;
            for (int g = 0; g < num_gpus; g++) {
                memcpy(losses.data() + offset * T, (float*)stagingBufs[g].mappedPtr + models[g].off_losses, batch_per_gpu[g] * T * sizeof(float));
                offset += batch_per_gpu[g];
            }
            for (int i = 0; i < total_batch * T; i++) total_loss += losses[i];
            total_count += total_batch * T;
        }
        return total_loss / total_count;
    };

    double gpu0_ms = 0, gpu1_ms = 0, allreduce_ms = 0, adamw_ms = 0, total_ms = 0;

    for (int iter = start_iter; iter < MAX_ITERS; iter++) {
        if (iter % EVAL_INTERVAL == 0) {
            float train_loss = evalLoss(train_data); float val_loss = evalLoss(val_data);
            LOG("Step %d: train_loss=%.4f  val_loss=%.4f", iter, train_loss, val_loss);
            beginCommandBuffer(contexts[0]);
            VkBufferCopy cp_region{0, 0, models[0].num_params * sizeof(float)};
            vkCmdCopyBuffer(contexts[0].cmdBuffer, floatBufs[0].buffer, stagingBufs[0].buffer, 1, &cp_region);
            endCommandBuffer(contexts[0]); submitCommandBuffer(contexts[0]); waitForFence(contexts[0]);
            saveCheckpoint((float*)stagingBufs[0].mappedPtr, models[0], iter, "qwen3_ckpt.bin");
            if (iter > 0) {
                if (num_gpus > 1) LOG("  GPU0=%.1f  GPU1=%.1f  AllReduce=%.1f  AdamW=%.1f  Total=%.1f ms/it", gpu0_ms / EVAL_INTERVAL, gpu1_ms / EVAL_INTERVAL, allreduce_ms / EVAL_INTERVAL, adamw_ms / EVAL_INTERVAL, total_ms / EVAL_INTERVAL);
                else LOG("  GPU0=%.1f  AllReduce=%.1f  AdamW=%.1f  Total=%.1f ms/it", gpu0_ms / EVAL_INTERVAL, allreduce_ms / EVAL_INTERVAL, adamw_ms / EVAL_INTERVAL, total_ms / EVAL_INTERVAL);
            }
            gpu0_ms = gpu1_ms = allreduce_ms = adamw_ms = total_ms = 0;
        }

        auto iter_start = Clock::now();
        std::vector<int> input_ids(total_batch * T), target_ids(total_batch * T);
        for (int b = 0; b < total_batch; b++) {
            int start = rand() % ((int)train_data.size() - T);
            for (int t = 0; t < T; t++) { input_ids[b * T + t] = train_data[start + t]; target_ids[b * T + t] = train_data[start + t + 1]; }
        }
        int offset = 0;
        for (int g = 0; g < num_gpus; g++) {
            memcpy((int*)intBufs[g].mappedPtr + models[g].off_input_ids, input_ids.data() + offset * T, batch_per_gpu[g] * T * sizeof(int));
            memcpy((int*)intBufs[g].mappedPtr + models[g].off_target_ids, target_ids.data() + offset * T, batch_per_gpu[g] * T * sizeof(int));
            offset += batch_per_gpu[g];
        }

        for (int g = 0; g < num_gpus; g++) {
            auto& ctx = contexts[g];
            beginCommandBuffer(ctx); recordForwardPass(g, true); recordZeroGradients(g); endCommandBuffer(ctx);
        }
        for (int g = 0; g < num_gpus; g++) submitCommandBuffer(contexts[g]);
        for (int g = 0; g < num_gpus; g++) waitForFence(contexts[g]);

        for (int g = 0; g < num_gpus; g++) {
            auto& ctx = contexts[g]; auto& m = models[g]; int Bl = batch_per_gpu[g];
            int BlT = Bl * T, BlTC = Bl * T * C;
            beginCommandBuffer(ctx);
            recordCompute(ctx, pipeCESoftmaxBwd[g], BlT, pcCESoftmaxBwd(m.off_d_logits, m.off_probs, m.off_target_ids, Bl, T, V, dLoss_bits));
            recordCompute(ctx, pipeMatMulDIn[g], BlTC, pcMatMulDIn(m.off_d_final_norm, m.off_d_logits, m.param_off[PARAM_LM_HEAD], BlT, C, V));
            recordCompute(ctx, pipeMatMulDWeight[g], C * V, pcMatMulDWeight(m.grad_off[PARAM_LM_HEAD], 0, m.off_d_logits, m.off_final_norm, BlT, C, V, 0));
            size_t final_input = m.off_res3 + (L - 1) * (size_t)BlTC; size_t d_final_input = m.off_d_res3 + (L - 1) * (size_t)BlTC;
            recordCompute(ctx, pipeRmsNormBwd[g], BlTC, pcRmsNormBwd(d_final_input, m.grad_off[PARAM_FINAL_NORM], m.off_d_final_norm, final_input, m.param_off[PARAM_FINAL_NORM], m.off_final_rstd, Bl, T, C));
            for (int l = L - 1; l >= 0; l--) {
                size_t layer_input = (l == 0) ? m.off_embed : m.off_res3 + (l - 1) * (size_t)BlTC;
                size_t d_layer_input = (l > 0) ? m.off_d_res3 + (l - 1) * (size_t)BlTC : m.off_d_embed;
                recordLayerBackward(g, l, layer_input, d_layer_input);
            }
            recordCompute(ctx, pipeEmbedBwd[g], BlTC, pcEmbedBwd(m.grad_off[PARAM_TOKEN_EMB], m.off_d_embed, m.off_input_ids, Bl, T, C));
            endCommandBuffer(ctx);
        }
        for (int g = 0; g < num_gpus; g++) submitCommandBuffer(contexts[g]);
        auto t0 = Clock::now(); waitForFence(contexts[0]); auto t1 = Clock::now(); gpu0_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (num_gpus > 1) { auto t2 = Clock::now(); waitForFence(contexts[1]); auto t3 = Clock::now(); gpu1_ms += std::chrono::duration<double, std::milli>(t3 - t2).count(); }
        for (int g = 2; g < num_gpus; g++) waitForFence(contexts[g]);

        // --- AllReduce & Parallel AdamW ---
        auto ar_start = Clock::now();
        for (int g = 0; g < num_gpus; g++) {
            beginCommandBuffer(contexts[g]);
            VkBufferCopy region{models[g].grad_off[0] * sizeof(float), models[g].grad_off[0] * sizeof(float), models[0].num_params * sizeof(float)};
            vkCmdCopyBuffer(contexts[g].cmdBuffer, floatBufs[g].buffer, stagingBufs[g].buffer, 1, &region);
            endCommandBuffer(contexts[g]); submitCommandBuffer(contexts[g]);
        }
        for (int g = 0; g < num_gpus; g++) waitForFence(contexts[g]);

        memset(grad_avg.data(), 0, models[0].num_params * sizeof(float));
        for (int g = 0; g < num_gpus; g++) {
            float* grads = (float*)stagingBufs[g].mappedPtr + models[g].grad_off[0];
            for (size_t i = 0; i < models[0].num_params; i++) grad_avg[i] += grads[i];
        }
        for (size_t i = 0; i < models[0].num_params; i++) grad_avg[i] /= num_gpus;

        for (int g = 0; g < num_gpus; g++) {
            memcpy((float*)stagingBufs[g].mappedPtr + models[g].grad_off[0], grad_avg.data(), models[0].num_params * sizeof(float));
        }

        for (int g = 0; g < num_gpus; g++) {
            beginCommandBuffer(contexts[g]);
            VkBufferCopy grad_region{models[g].grad_off[0] * sizeof(float), models[g].grad_off[0] * sizeof(float), models[0].num_params * sizeof(float)};
            vkCmdCopyBuffer(contexts[g].cmdBuffer, stagingBufs[g].buffer, floatBufs[g].buffer, 1, &grad_region);
            VkMemoryBarrier grad_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            grad_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            grad_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(contexts[g].cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &grad_barrier, 0, nullptr, 0, nullptr);
            for (size_t i = 0; i < models[g].param_off.size(); i++) {
                if (param_sizes[i] > 0) {
                    recordCompute(contexts[g], pipeAdamW[g], (int)param_sizes[i],
                        pcAdamW((int)models[g].param_off[i], (int)models[g].grad_off[i], (int)(models[g].m_off + models[g].param_off[i]), (int)(models[g].v_off + models[g].param_off[i]), (int)param_sizes[i], iter + 1));
                }
            }
            endCommandBuffer(contexts[g]); submitCommandBuffer(contexts[g]);
        }
        for (int g = 0; g < num_gpus; g++) waitForFence(contexts[g]);

        auto aw_end = Clock::now();
        adamw_ms += std::chrono::duration<double, std::milli>(aw_end - ar_start).count();
        allreduce_ms += 0.0f; 
        total_ms += std::chrono::duration<double, std::milli>(Clock::now() - iter_start).count();

#if DEBUG_VERBOSE
        if (iter % 10 == 0) {
            float norm = 0.0f;
            const float* grads = (float*)stagingBufs[0].mappedPtr + models[0].grad_off[0];
            for (size_t i = 0; i < models[0].num_params; i++) norm += grads[i] * grads[i];
            LOG_DBG("Step %d: grad_norm = %.6f", iter, sqrtf(norm));
        }
#endif
    }

    beginCommandBuffer(contexts[0]);
    VkBufferCopy cp_region{0, 0, 3 * models[0].num_params * sizeof(float)};
    vkCmdCopyBuffer(contexts[0].cmdBuffer, floatBufs[0].buffer, stagingBufs[0].buffer, 1, &cp_region);
    endCommandBuffer(contexts[0]); submitCommandBuffer(contexts[0]); waitForFence(contexts[0]);
    saveCheckpoint((float*)stagingBufs[0].mappedPtr, models[0], MAX_ITERS, "qwen3_ckpt.bin");

    // ========================================================
    // Text Generation
    // ========================================================
    LOG("Generating sample text...");
    
    // Temporarily set batch size to 1 for fast generation 
    // (the buffer is large enough, so this is perfectly safe)
    batch_per_gpu[0] = 1; 
    
    std::vector<int> generated(1, 0);  // Start with first token
    std::vector<float> head_probs(V);

    for (int i = 0; i < 500; i++) {
        int start_pos = std::max(0, (int)generated.size() - T);
        int ctx_len   = std::min((int)generated.size(), T);

        // Prepare input
        std::vector<int> gen_input(batch_per_gpu[0] * T, 0);
        for (int t = 0; t < ctx_len; t++)
            gen_input[t] = generated[start_pos + t];

        memcpy((int*)intBufs[0].mappedPtr + models[0].off_input_ids,
               gen_input.data(), batch_per_gpu[0] * T * sizeof(int));

        // Forward pass (no loss)
        beginCommandBuffer(contexts[0]);
        recordForwardPass(0, false);
        endCommandBuffer(contexts[0]);
        submitCommandBuffer(contexts[0]);
        waitForFence(contexts[0]);

        // Sample from last position's distribution
        // 1. Calculate the exact offset for the probabilities we need
        VkDeviceSize prob_offset = (models[0].off_probs + (ctx_len - 1) * V) * sizeof(float);
        
        // 2. Copy probs from DEVICE to STAGING
        beginCommandBuffer(contexts[0]);
        VkBufferCopy prob_region{prob_offset, prob_offset, V * sizeof(float)};
        vkCmdCopyBuffer(contexts[0].cmdBuffer, floatBufs[0].buffer, stagingBufs[0].buffer, 1, &prob_region);
        endCommandBuffer(contexts[0]);
        submitCommandBuffer(contexts[0]);
        waitForFence(contexts[0]);

        // 3. Read from STAGING
        memcpy(head_probs.data(),
               (float*)stagingBufs[0].mappedPtr + models[0].off_probs + (ctx_len - 1) * V,
               V * sizeof(float));

        float r = (float)rand() / RAND_MAX;
        float cdf = 0.0f;
        int next_token = V - 1;
        for (int j = 0; j < V; j++) {
            cdf += head_probs[j];
            if (r < cdf) { next_token = j; break; }
        }
        generated.push_back(next_token);
    }

    // YOU WERE MISSING THESE TWO LINES!
    LOG("---");
    LOG("%s", tokenizer.decode(generated).c_str());
    LOG("---");

    for (int g = 0; g < num_gpus; g++) {
        vkDeviceWaitIdle(contexts[g].device);
        vkDestroyBuffer(contexts[g].device, floatBufs[g].buffer, nullptr); vkFreeMemory(contexts[g].device, floatBufs[g].memory, nullptr);
        vkDestroyBuffer(contexts[g].device, stagingBufs[g].buffer, nullptr); vkFreeMemory(contexts[g].device, stagingBufs[g].memory, nullptr);
        vkDestroyBuffer(contexts[g].device, intBufs[g].buffer, nullptr); vkFreeMemory(contexts[g].device, intBufs[g].memory, nullptr);
    }
    return 0;
}