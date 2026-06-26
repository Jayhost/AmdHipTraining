// train_gpt2_hip.cpp
// AMD HIP GPU implementation of nanoGPT training
// Compile: hipcc -O3 train_gpt2_hip.cpp -o train_gpt2_hip

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
// Hyperparameters (matching Python)
// ============================================================
const int B = 16;         // batch_size
const int T = 32;         // block_size
const int MAX_ITERS = 1000;
const int EVAL_INTERVAL = 100;
const float LR = 1e-3f;
const int EVAL_ITERS = 200;
const int C = 64;         // n_embd
const int NH = 4;         // n_head
const int L = 4;          // n_layer
const int HS = C / NH;   // head_size = 16
const int C3 = 3 * C;    // 192
const int C4 = 4 * C;    // 256
const int BLOCK_SIZE_KERNEL = 256; // GPU thread block size

// ============================================================
// Tokenizer (character-level)
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
                                       const float* wte, const float* wpe,
                                       int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx >= N) return;
    int c = idx % C;
    int bt = idx / C;
    int t = bt % T;
    int ix = inp[bt];
    out[idx] = wte[ix * C + c] + wpe[t * C + c];
}

__global__ void layernorm_forward_kernel(float* out, float* mean, float* rstd,
                                         const float* inp, const float* weight, const float* bias,
                                         int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx >= N) return;
    int c = idx % C;
    int bt = idx / C;
    const float* x = inp + bt * C;
    float m = 0.0f;
    for (int i = 0; i < C; i++) m += x[i];
    m /= C;
    float v = 0.0f;
    for (int i = 0; i < C; i++) { float d = x[i] - m; v += d * d; }
    v /= C;
    float s = 1.0f / sqrtf(v + 1e-5f);
    out[idx] = s * (x[c] - m) * weight[c] + bias[c];
    if (c == 0) { mean[bt] = m; rstd[bt] = s; }
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

__global__ void attention_forward_kernel(float* out, float* preatt, float* att,
                                         const float* inp,
                                         int B, int T, int C, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * NH;
    if (idx >= total) return;
    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (T * NH);
    int hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);
    int C3 = 3 * C;
    const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
    float* preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
    float* att_bth = att + b * NH * T * T + h * T * T + t * T;
    float maxval = -1e10f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
        float val = 0.0f;
        for (int i = 0; i < hs; i++) val += query_t[i] * key_t2[i];
        val *= scale;
        preatt_bth[t2] = val;
        if (val > maxval) maxval = val;
    }
    for (int t2 = t + 1; t2 < T; t2++) preatt_bth[t2] = -1e10f;
    float expsum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        float expv = expf(preatt_bth[t2] - maxval);
        expsum += expv;
        att_bth[t2] = expv;
    }
    float inv_sum = (expsum == 0.0f) ? 0.0f : 1.0f / expsum;
    for (int t2 = 0; t2 <= t; t2++) att_bth[t2] *= inv_sum;
    for (int t2 = t + 1; t2 < T; t2++) att_bth[t2] = 0.0f;
    float* out_bth = out + b * T * C + t * C + h * hs;
    for (int i = 0; i < hs; i++) out_bth[i] = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + 2 * C;
        float a = att_bth[t2];
        for (int i = 0; i < hs; i++) out_bth[i] += a * value_t2[i];
    }
}

__global__ void relu_forward_kernel(float* out, const float* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = fmaxf(inp[idx], 0.0f);
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
    int t = bt % T;
    int ix = inp[bt];
    float d = dout[idx];
    atomicAdd(&dwte[ix * C + c], d);
    atomicAdd(&dwpe[t * C + c], d);
}

__global__ void layernorm_backward_kernel(float* dinp, float* dweight, float* dbias,
                                          const float* dout, const float* inp,
                                          const float* weight, const float* mean, const float* rstd,
                                          int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx >= N) return;
    int c = idx % C;
    int bt = idx / C;
    float m_bt = mean[bt], s_bt = rstd[bt];
    const float* dout_bt = dout + bt * C;
    const float* inp_bt = inp + bt * C;
    float dnorm_mean = 0.0f, dnorm_norm_mean = 0.0f;
    for (int i = 0; i < C; i++) {
        float norm_i = (inp_bt[i] - m_bt) * s_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_i;
    }
    dnorm_mean /= C;
    dnorm_norm_mean /= C;
    float norm_c = (inp_bt[c] - m_bt) * s_bt;
    atomicAdd(&dweight[c], norm_c * dout_bt[c]);
    atomicAdd(&dbias[c], dout_bt[c]);
    float dval = (weight[c] * dout_bt[c] - dnorm_mean - norm_c * dnorm_norm_mean) * s_bt;
    dinp[bt * C + c] += dval;
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
    if (i == 0 && dbias != NULL) {
        float db = 0.0f;
        for (int bt = 0; bt < total_bt; bt++) db += dout[bt * OC + o];
        dbias[o] += db;
    }
}

__global__ void attention_backward_kernel(float* dinp, float* dpreatt, float* datt,
                                          const float* dout, const float* inp, const float* att,
                                          int B, int T, int C, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * NH;
    if (idx >= total) return;
    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (T * NH);
    int hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);
    int C3 = 3 * C;
    const float* att_bth = att + b * NH * T * T + h * T * T + t * T;
    float* datt_bth = datt + b * NH * T * T + h * T * T + t * T;
    float* dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
    const float* dout_bth = dout + b * T * C + t * C + h * hs;
    const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
    float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;

    for (int t2 = 0; t2 <= t; t2++) {
        const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + 2 * C;
        float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + 2 * C;
        for (int i = 0; i < hs; i++) {
            datt_bth[t2] += value_t2[i] * dout_bth[i];
            atomicAdd(&dvalue_t2[i], att_bth[t2] * dout_bth[i]);
        }
    }

    float dot = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        dot += att_bth[t2] * datt_bth[t2];
    }
    for (int t2 = 0; t2 <= t; t2++) {
        dpreatt_bth[t2] = att_bth[t2] * (datt_bth[t2] - dot);
    }

    for (int t2 = 0; t2 <= t; t2++) {
        const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
        float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C;
        float dp = dpreatt_bth[t2] * scale;
        for (int i = 0; i < hs; i++) {
            atomicAdd(&dquery_t[i], key_t2[i] * dp);
            atomicAdd(&dkey_t2[i], query_t[i] * dp);
        }
    }
}

__global__ void relu_backward_kernel(float* dinp, const float* inp, const float* dout, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    dinp[idx] += (inp[idx] > 0.0f) ? dout[idx] : 0.0f;
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

struct GPT2 {
    int V; 
    float *d_wte, *d_wpe;
    float *d_ln1w, *d_ln1b, *d_qkvw, *d_qkvb, *d_attprojw, *d_attprojb;
    float *d_ln2w, *d_ln2b, *d_fcw, *d_fcb, *d_fcprojw, *d_fcprojb;
    float *d_lnfw, *d_lnfb;
    float *d_lm_head, *d_lm_bias; // Output layers
    
    float *d_dwte, *d_dwpe;
    float *d_dln1w, *d_dln1b, *d_dqkvw, *d_dqkvb, *d_dattprojw, *d_dattprojb;
    float *d_dln2w, *d_dln2b, *d_dfcw, *d_dfcb, *d_dfcprojw, *d_dfcprojb;
    float *d_dlnfw, *d_dlnfb;
    float *d_dlm_head, *d_dlm_bias; // Output layer grads
    
    float *d_m, *d_v;
    float *d_encoded, *d_ln1, *d_ln1_mean, *d_ln1_rstd;
    float *d_qkv, *d_atty, *d_preatt, *d_att, *d_attproj, *d_residual2;
    float *d_ln2, *d_ln2_mean, *d_ln2_rstd;
    float *d_fch, *d_fch_relu, *d_fcproj, *d_residual3;
    float *d_lnf, *d_lnf_mean, *d_lnf_rstd;
    float *d_logits, *d_probs, *d_losses;
    float *d_dencoded, *d_dln1, *d_dqkv, *d_datty, *d_dpreatt, *d_datt, *d_dattproj, *d_dresidual2;
    float *d_dln2, *d_dfch, *d_dfch_relu, *d_dfcproj, *d_dresidual3;
    float *d_dlnf, *d_dlogits;
    int *d_inputs, *d_targets;
    size_t num_parameters;
};

std::vector<TensorDesc> get_param_list(GPT2* model) {
    return {
        {model->d_wte, (size_t)model->V * C}, {model->d_wpe, (size_t)T * C},
        {model->d_ln1w, (size_t)L * C}, {model->d_ln1b, (size_t)L * C},
        {model->d_qkvw, (size_t)L * 3*C*C}, {model->d_qkvb, (size_t)L * 3*C},
        {model->d_attprojw, (size_t)L * C*C}, {model->d_attprojb, (size_t)L * C},
        {model->d_ln2w, (size_t)L * C}, {model->d_ln2b, (size_t)L * C},
        {model->d_fcw, (size_t)L * 4*C*C}, {model->d_fcb, (size_t)L * 4*C},
        {model->d_fcprojw, (size_t)L * C*4*C}, {model->d_fcprojb, (size_t)L * C},
        {model->d_lnfw, (size_t)C}, {model->d_lnfb, (size_t)C},
        {model->d_lm_head, (size_t)model->V * C}, {model->d_lm_bias, (size_t)model->V}
    };
}

std::vector<TensorDesc> get_grad_list(GPT2* model) {
    return {
        {model->d_dwte, (size_t)model->V * C}, {model->d_dwpe, (size_t)T * C},
        {model->d_dln1w, (size_t)L * C}, {model->d_dln1b, (size_t)L * C},
        {model->d_dqkvw, (size_t)L * 3*C*C}, {model->d_dqkvb, (size_t)L * 3*C},
        {model->d_dattprojw, (size_t)L * C*C}, {model->d_dattprojb, (size_t)L * C},
        {model->d_dln2w, (size_t)L * C}, {model->d_dln2b, (size_t)L * C},
        {model->d_dfcw, (size_t)L * 4*C*C}, {model->d_dfcb, (size_t)L * 4*C},
        {model->d_dfcprojw, (size_t)L * C*4*C}, {model->d_dfcprojb, (size_t)L * C},
        {model->d_dlnfw, (size_t)C}, {model->d_dlnfb, (size_t)C},
        {model->d_dlm_head, (size_t)model->V * C}, {model->d_dlm_bias, (size_t)model->V}
    };
}

size_t calc_num_parameters(int V) {
    return (size_t)(V * C) + (T * C) +
           L * (C + C + 3*C*C + 3*C + C*C + C + C + C + 4*C*C + 4*C + C*4*C + C) + C + C +
           V * C + V; // lm_head and lm_bias
}

void gpt2_alloc(GPT2 *model, int V) {
    model->V = V;
    model->num_parameters = calc_num_parameters(V);
    
    auto params = get_param_list(model);
    for (auto& p : params) HIP_CHECK(hipMalloc(&p.d_ptr, p.size * sizeof(float)));
    model->d_wte = params[0].d_ptr; model->d_wpe = params[1].d_ptr;
    model->d_ln1w = params[2].d_ptr; model->d_ln1b = params[3].d_ptr;
    model->d_qkvw = params[4].d_ptr; model->d_qkvb = params[5].d_ptr;
    model->d_attprojw = params[6].d_ptr; model->d_attprojb = params[7].d_ptr;
    model->d_ln2w = params[8].d_ptr; model->d_ln2b = params[9].d_ptr;
    model->d_fcw = params[10].d_ptr; model->d_fcb = params[11].d_ptr;
    model->d_fcprojw = params[12].d_ptr; model->d_fcprojb = params[13].d_ptr;
    model->d_lnfw = params[14].d_ptr; model->d_lnfb = params[15].d_ptr;
    model->d_lm_head = params[16].d_ptr; model->d_lm_bias = params[17].d_ptr;

    auto grads = get_grad_list(model);
    for (auto& g : grads) HIP_CHECK(hipMalloc(&g.d_ptr, g.size * sizeof(float)));
    model->d_dwte = grads[0].d_ptr; model->d_dwpe = grads[1].d_ptr;
    model->d_dln1w = grads[2].d_ptr; model->d_dln1b = grads[3].d_ptr;
    model->d_dqkvw = grads[4].d_ptr; model->d_dqkvb = grads[5].d_ptr;
    model->d_dattprojw = grads[6].d_ptr; model->d_dattprojb = grads[7].d_ptr;
    model->d_dln2w = grads[8].d_ptr; model->d_dln2b = grads[9].d_ptr;
    model->d_dfcw = grads[10].d_ptr; model->d_dfcb = grads[11].d_ptr;
    model->d_dfcprojw = grads[12].d_ptr; model->d_dfcprojb = grads[13].d_ptr;
    model->d_dlnfw = grads[14].d_ptr; model->d_dlnfb = grads[15].d_ptr;
    model->d_dlm_head = grads[16].d_ptr; model->d_dlm_bias = grads[17].d_ptr;

    HIP_CHECK(hipMalloc(&model->d_m, model->num_parameters * sizeof(float)));
    HIP_CHECK(hipMalloc(&model->d_v, model->num_parameters * sizeof(float)));
    HIP_CHECK(hipMemset(model->d_m, 0, model->num_parameters * sizeof(float)));
    HIP_CHECK(hipMemset(model->d_v, 0, model->num_parameters * sizeof(float)));

    #define HIP_ALLOC(ptr, size) HIP_CHECK(hipMalloc(&ptr, size * sizeof(float)));
    HIP_ALLOC(model->d_encoded, B * T * C); HIP_ALLOC(model->d_ln1, L * B * T * C);
    HIP_ALLOC(model->d_ln1_mean, L * B * T); HIP_ALLOC(model->d_ln1_rstd, L * B * T);
    HIP_ALLOC(model->d_qkv, L * B * T * 3*C); HIP_ALLOC(model->d_atty, L * B * T * C);
    HIP_ALLOC(model->d_preatt, L * B * NH * T * T); HIP_ALLOC(model->d_att, L * B * NH * T * T);
    HIP_ALLOC(model->d_attproj, L * B * T * C); HIP_ALLOC(model->d_residual2, L * B * T * C);
    HIP_ALLOC(model->d_ln2, L * B * T * C); HIP_ALLOC(model->d_ln2_mean, L * B * T);
    HIP_ALLOC(model->d_ln2_rstd, L * B * T); HIP_ALLOC(model->d_fch, L * B * T * 4*C);
    HIP_ALLOC(model->d_fch_relu, L * B * T * 4*C); HIP_ALLOC(model->d_fcproj, L * B * T * C);
    HIP_ALLOC(model->d_residual3, L * B * T * C); HIP_ALLOC(model->d_lnf, B * T * C);
    HIP_ALLOC(model->d_lnf_mean, B * T); HIP_ALLOC(model->d_lnf_rstd, B * T);
    HIP_ALLOC(model->d_logits, B * T * V); HIP_ALLOC(model->d_probs, B * T * V);
    HIP_ALLOC(model->d_losses, B * T);

    HIP_ALLOC(model->d_dencoded, B * T * C); HIP_ALLOC(model->d_dln1, L * B * T * C);
    HIP_ALLOC(model->d_dqkv, L * B * T * 3*C); HIP_ALLOC(model->d_datty, L * B * T * C);
    HIP_ALLOC(model->d_dpreatt, L * B * NH * T * T); HIP_ALLOC(model->d_datt, L * B * NH * T * T);
    HIP_ALLOC(model->d_dattproj, L * B * T * C); HIP_ALLOC(model->d_dresidual2, L * B * T * C);
    HIP_ALLOC(model->d_dln2, L * B * T * C); HIP_ALLOC(model->d_dfch, L * B * T * 4*C);
    HIP_ALLOC(model->d_dfch_relu, L * B * T * 4*C); HIP_ALLOC(model->d_dfcproj, L * B * T * C);
    HIP_ALLOC(model->d_dresidual3, L * B * T * C); HIP_ALLOC(model->d_dlnf, B * T * C);
    HIP_ALLOC(model->d_dlogits, B * T * V);
    #undef HIP_ALLOC

    HIP_CHECK(hipMalloc(&model->d_inputs, B * T * sizeof(int)));
    HIP_CHECK(hipMalloc(&model->d_targets, B * T * sizeof(int)));
}

void gpt2_free(GPT2 *model) {
    auto params = get_param_list(model);
    for (auto& p : params) HIP_CHECK(hipFree(p.d_ptr));
    auto grads = get_grad_list(model);
    for (auto& g : grads) HIP_CHECK(hipFree(g.d_ptr));
    HIP_CHECK(hipFree(model->d_m)); HIP_CHECK(hipFree(model->d_v));
    #define HIP_FREE(ptr) if(ptr) HIP_CHECK(hipFree(ptr));
    HIP_FREE(model->d_encoded); HIP_FREE(model->d_ln1); HIP_FREE(model->d_ln1_mean); HIP_FREE(model->d_ln1_rstd);
    HIP_FREE(model->d_qkv); HIP_FREE(model->d_atty); HIP_FREE(model->d_preatt); HIP_FREE(model->d_att);
    HIP_FREE(model->d_attproj); HIP_FREE(model->d_residual2); HIP_FREE(model->d_ln2); HIP_FREE(model->d_ln2_mean);
    HIP_FREE(model->d_ln2_rstd); HIP_FREE(model->d_fch); HIP_FREE(model->d_fch_relu); HIP_FREE(model->d_fcproj);
    HIP_FREE(model->d_residual3); HIP_FREE(model->d_lnf); HIP_FREE(model->d_lnf_mean); HIP_FREE(model->d_lnf_rstd);
    HIP_FREE(model->d_logits); HIP_FREE(model->d_probs); HIP_FREE(model->d_losses);
    HIP_FREE(model->d_dencoded); HIP_FREE(model->d_dln1); HIP_FREE(model->d_dqkv); HIP_FREE(model->d_datty);
    HIP_FREE(model->d_dpreatt); HIP_FREE(model->d_datt); HIP_FREE(model->d_dattproj); HIP_FREE(model->d_dresidual2);
    HIP_FREE(model->d_dln2); HIP_FREE(model->d_dfch); HIP_FREE(model->d_dfch_relu); HIP_FREE(model->d_dfcproj);
    HIP_FREE(model->d_dresidual3); HIP_FREE(model->d_dlnf); HIP_FREE(model->d_dlogits);
    HIP_FREE(model->d_inputs); HIP_FREE(model->d_targets);
    #undef HIP_FREE
}

void gpt2_zero_grad(GPT2 *model) {
    auto grads = get_grad_list(model);
    for (auto& g : grads) HIP_CHECK(hipMemset(g.d_ptr, 0, g.size * sizeof(float)));
    
    #define HIP_MEMZERO(ptr, size) HIP_CHECK(hipMemset(ptr, 0, size * sizeof(float)))
    HIP_MEMZERO(model->d_dencoded, B * T * C); HIP_MEMZERO(model->d_dln1, L * B * T * C);
    HIP_MEMZERO(model->d_dqkv, L * B * T * 3*C); HIP_MEMZERO(model->d_datty, L * B * T * C);
    HIP_MEMZERO(model->d_dpreatt, L * B * NH * T * T); HIP_MEMZERO(model->d_datt, L * B * NH * T * T);
    HIP_MEMZERO(model->d_dattproj, L * B * T * C); HIP_MEMZERO(model->d_dresidual2, L * B * T * C);
    HIP_MEMZERO(model->d_dln2, L * B * T * C); HIP_MEMZERO(model->d_dfch, L * B * T * 4*C);
    HIP_MEMZERO(model->d_dfch_relu, L * B * T * 4*C); HIP_MEMZERO(model->d_dfcproj, L * B * T * C);
    HIP_MEMZERO(model->d_dresidual3, L * B * T * C); HIP_MEMZERO(model->d_dlnf, B * T * C);
    HIP_MEMZERO(model->d_dlogits, B * T * model->V);
    #undef HIP_MEMZERO
}

// ============================================================
// Checkpointing
// ============================================================

void gpt2_save_checkpoint(GPT2* model, int step, const char* filepath) {
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
    
    HIP_CHECK(hipMemcpy(cpu_buf.data(), model->d_v, model->num_parameters * sizeof(float), hipMemcpyDeviceToHost));
    fwrite(cpu_buf.data(), sizeof(float), model->num_parameters, file);
    
    fclose(file);
    printf("Checkpoint saved to %s at step %d\n", filepath, step);
}

int gpt2_load_checkpoint(GPT2* model, const char* filepath) {
    FILE* file = fopen(filepath, "rb");
    if(!file) {
        printf("No checkpoint found at %s, starting fresh.\n", filepath);
        return 0;
    }
    
    int step;
    fread(&step, sizeof(int), 1, file);
    
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
    HIP_CHECK(hipMemcpy(model->d_v, cpu_buf.data(), model->num_parameters * sizeof(float), hipMemcpyHostToDevice));
    
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
void gpt2_forward(GPT2 *model, const int* h_inputs, const int* h_targets) {
    int V = model->V;
    HIP_CHECK(hipMemcpy(model->d_inputs, h_inputs, B * T * sizeof(int), hipMemcpyHostToDevice));
    if (h_targets != NULL) {
        HIP_CHECK(hipMemcpy(model->d_targets, h_targets, B * T * sizeof(int), hipMemcpyHostToDevice));
    }

    LAUNCH(encoder_forward_kernel, B * T * C, model->d_encoded, model->d_inputs, model->d_wte, model->d_wpe, B, T, C);

    for (int l = 0; l < L; l++) {
        float* residual = (l == 0) ? model->d_encoded : model->d_residual3 + (l - 1) * B * T * C;
        
        LAUNCH(layernorm_forward_kernel, B * T * C, model->d_ln1 + l*B*T*C, model->d_ln1_mean + l*B*T, model->d_ln1_rstd + l*B*T,
               residual, model->d_ln1w + l*C, model->d_ln1b + l*C, B, T, C);
        LAUNCH(matmul_forward_kernel, B * T * 3 * C, model->d_qkv + l*B*T*3*C, model->d_ln1 + l*B*T*C,
               model->d_qkvw + l*3*C*C, model->d_qkvb + l*3*C, B * T, C, 3 * C);
        LAUNCH(attention_forward_kernel, B * T * NH, model->d_atty + l*B*T*C, model->d_preatt + l*B*NH*T*T, model->d_att + l*B*NH*T*T,
               model->d_qkv + l*B*T*3*C, B, T, C, NH);
        LAUNCH(matmul_forward_kernel, B * T * C, model->d_attproj + l*B*T*C, model->d_atty + l*B*T*C,
               model->d_attprojw + l*C*C, model->d_attprojb + l*C, B * T, C, C);
        LAUNCH(residual_forward_kernel, B * T * C, model->d_residual2 + l*B*T*C, residual, model->d_attproj + l*B*T*C, B * T * C);
        LAUNCH(layernorm_forward_kernel, B * T * C, model->d_ln2 + l*B*T*C, model->d_ln2_mean + l*B*T, model->d_ln2_rstd + l*B*T,
               model->d_residual2 + l*B*T*C, model->d_ln2w + l*C, model->d_ln2b + l*C, B, T, C);
        LAUNCH(matmul_forward_kernel, B * T * 4 * C, model->d_fch + l*B*T*4*C, model->d_ln2 + l*B*T*C,
               model->d_fcw + l*4*C*C, model->d_fcb + l*4*C, B * T, C, 4 * C);
        LAUNCH(relu_forward_kernel, B * T * 4 * C, model->d_fch_relu + l*B*T*4*C, model->d_fch + l*B*T*4*C, B * T * 4 * C);
        LAUNCH(matmul_forward_kernel, B * T * C, model->d_fcproj + l*B*T*C, model->d_fch_relu + l*B*T*4*C,
               model->d_fcprojw + l*C*4*C, model->d_fcprojb + l*C, B * T, 4 * C, C);
        LAUNCH(residual_forward_kernel, B * T * C, model->d_residual3 + l*B*T*C, model->d_residual2 + l*B*T*C, model->d_fcproj + l*B*T*C, B * T * C);
    }

    float* residual = model->d_residual3 + (L - 1) * B * T * C;
    LAUNCH(layernorm_forward_kernel, B * T * C, model->d_lnf, model->d_lnf_mean, model->d_lnf_rstd,
           residual, model->d_lnfw, model->d_lnfb, B, T, C);
           
    // Use lm_head and lm_bias instead of wte
    LAUNCH(matmul_forward_kernel, B * T * V, model->d_logits, model->d_lnf, model->d_lm_head, model->d_lm_bias, B * T, C, V);
    LAUNCH(softmax_forward_kernel, B * T, model->d_probs, model->d_logits, B, T, V);
    
    if (h_targets != NULL) {
        LAUNCH(crossentropy_forward_kernel, B * T, model->d_losses, model->d_probs, model->d_targets, B, T, V);
    }
}

// ============================================================
// Backward Pass
// ============================================================
void gpt2_backward(GPT2 *model) {
    int V = model->V;
    
    float dloss_val = 1.0f / (B * T);
    std::vector<float> h_dlosses(B * T, dloss_val);
    HIP_CHECK(hipMemcpy(model->d_losses, h_dlosses.data(), B * T * sizeof(float), hipMemcpyHostToDevice));

    LAUNCH(crossentropy_softmax_backward_kernel, B * T, model->d_dlogits, model->d_losses, model->d_probs, model->d_targets, B, T, V);
    
    // Backprop through lm_head and lm_bias
    LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_dlnf, model->d_dlogits, model->d_lm_head, B * T, C, V);
    LAUNCH(matmul_backward_dweight_dbias_kernel, V * C, model->d_dlm_head, model->d_dlm_bias, model->d_dlogits, model->d_lnf, B * T, C, V);

    float* residual = model->d_residual3 + (L - 1) * B * T * C;
    float* dresidual = model->d_dresidual3 + (L - 1) * B * T * C;
    LAUNCH(layernorm_backward_kernel, B * T * C, dresidual, model->d_dlnfw, model->d_dlnfb, model->d_dlnf, residual, model->d_lnfw, model->d_lnf_mean, model->d_lnf_rstd, B, T, C);

    for (int l = L - 1; l >= 0; l--) {
        residual = (l == 0) ? model->d_encoded : model->d_residual3 + (l - 1) * B * T * C;
        dresidual = (l == 0) ? model->d_dencoded : model->d_dresidual3 + (l - 1) * B * T * C;

        LAUNCH(residual_backward_kernel, B * T * C, model->d_dresidual2 + l*B*T*C, model->d_dfcproj + l*B*T*C, model->d_dresidual3 + l*B*T*C, B * T * C);
        LAUNCH(matmul_backward_dinp_kernel, B * T * 4 * C, model->d_dfch_relu + l*B*T*4*C, model->d_dfcproj + l*B*T*C, model->d_fcprojw + l*C*4*C, B * T, 4 * C, C);
        LAUNCH(matmul_backward_dweight_dbias_kernel, C * 4 * C, model->d_dfcprojw + l*C*4*C, model->d_dfcprojb + l*C, model->d_dfcproj + l*B*T*C, model->d_fch_relu + l*B*T*4*C, B * T, 4 * C, C);
        LAUNCH(relu_backward_kernel, B * T * 4 * C, model->d_dfch + l*B*T*4*C, model->d_fch + l*B*T*4*C, model->d_dfch_relu + l*B*T*4*C, B * T * 4 * C);
        LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_dln2 + l*B*T*C, model->d_dfch + l*B*T*4*C, model->d_fcw + l*4*C*C, B * T, C, 4 * C);
        LAUNCH(matmul_backward_dweight_dbias_kernel, 4 * C * C, model->d_dfcw + l*4*C*C, model->d_dfcb + l*4*C, model->d_dfch + l*B*T*4*C, model->d_ln2 + l*B*T*C, B * T, C, 4 * C);
        LAUNCH(layernorm_backward_kernel, B * T * C, model->d_dresidual2 + l*B*T*C, model->d_dln2w + l*C, model->d_dln2b + l*C, model->d_dln2 + l*B*T*C, model->d_residual2 + l*B*T*C, model->d_ln2w + l*C, model->d_ln2_mean + l*B*T, model->d_ln2_rstd + l*B*T, B, T, C);
        LAUNCH(residual_backward_kernel, B * T * C, dresidual, model->d_dattproj + l*B*T*C, model->d_dresidual2 + l*B*T*C, B * T * C);
        LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_datty + l*B*T*C, model->d_dattproj + l*B*T*C, model->d_attprojw + l*C*C, B * T, C, C);
        LAUNCH(matmul_backward_dweight_dbias_kernel, C * C, model->d_dattprojw + l*C*C, model->d_dattprojb + l*C, model->d_dattproj + l*B*T*C, model->d_atty + l*B*T*C, B * T, C, C);
        LAUNCH(attention_backward_kernel, B * T * NH, model->d_dqkv + l*B*T*3*C, model->d_dpreatt + l*B*NH*T*T, model->d_datt + l*B*NH*T*T, model->d_datty + l*B*T*C, model->d_qkv + l*B*T*3*C, model->d_att + l*B*NH*T*T, B, T, C, NH);
        LAUNCH(matmul_backward_dinp_kernel, B * T * 3 * C, model->d_dln1 + l*B*T*C, model->d_dqkv + l*B*T*3*C, model->d_qkvw + l*3*C*C, B * T, C, 3 * C);
        LAUNCH(matmul_backward_dweight_dbias_kernel, 3 * C * C, model->d_dqkvw + l*3*C*C, model->d_dqkvb + l*3*C, model->d_dqkv + l*B*T*3*C, model->d_ln1 + l*B*T*C, B * T, C, 3 * C);
        LAUNCH(layernorm_backward_kernel, B * T * C, dresidual, model->d_dln1w + l*C, model->d_dln1b + l*C, model->d_dln1 + l*B*T*C, residual, model->d_ln1w + l*C, model->d_ln1_mean + l*B*T, model->d_ln1_rstd + l*B*T, B, T, C);
    }
    LAUNCH(encoder_backward_kernel, B * T * C, model->d_dwte, model->d_dwpe, model->d_dencoded, model->d_inputs, B, T, C);
}

// ============================================================
// Update Parameters
// ============================================================
void gpt2_update(GPT2 *model, float learning_rate, int step) {
    auto params = get_param_list(model);
    auto grads = get_grad_list(model);
    
    size_t m_offset = 0;
    for(size_t i = 0; i < params.size(); i++) {
        if (params[i].size > 0) {
            int grid = (params[i].size + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL;
            // Fixed weight_decay = 0.01f to match PyTorch's AdamW default
            adamw_kernel<<<grid, BLOCK_SIZE_KERNEL>>>(params[i].d_ptr, grads[i].d_ptr, model->d_m + m_offset, model->d_v + m_offset, params[i].size, learning_rate, 0.9f, 0.999f, 1e-8f, 0.01f, step);
            m_offset += params[i].size;
        }
    }
}

// ============================================================
// Main
// ============================================================
int main() {
    std::ifstream file("small.txt");
    if (!file.is_open()) {
        fprintf(stderr, "Could not open small.txt!\n");
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
    std::vector<int> val_tokens(tokens.begin() + n, tokens.end()); // Added validation split

    GPT2 model;
    gpt2_alloc(&model, V);

    std::random_device rd;
    std::mt19937 gen(1337);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    int start_iter = gpt2_load_checkpoint(&model, "gpt_hip_checkpoint.bin");
    
    if (start_iter == 0) {
        printf("Initializing model weights from scratch...\n");
        
        // Explicit, explicit, explicit. No modulo math bugs.
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
        init_random(model.d_wpe, T * C); // Fixed
        
        init_val(model.d_ln1w, L * C, 1.0f); // Fixed LN init
        init_val(model.d_ln1b, L * C, 0.0f);
        init_random(model.d_qkvw, L * 3*C*C);
        init_val(model.d_qkvb, L * 3*C, 0.0f);
        init_random(model.d_attprojw, L * C*C);
        init_val(model.d_attprojb, L * C, 0.0f);
        
        init_val(model.d_ln2w, L * C, 1.0f); // Fixed LN init
        init_val(model.d_ln2b, L * C, 0.0f);
        init_random(model.d_fcw, L * 4*C*C);
        init_val(model.d_fcb, L * 4*C, 0.0f);
        init_random(model.d_fcprojw, L * C*4*C);
        init_val(model.d_fcprojb, L * C, 0.0f);
        
        init_val(model.d_lnfw, C, 1.0f); // Fixed LN init
        init_val(model.d_lnfb, C, 0.0f);
        
        init_random(model.d_lm_head, V * C);
        init_val(model.d_lm_bias, V, 0.0f); // Fixed bias addition
    }

    printf("Model initialized. Parameters: %zu\n", model.num_parameters);

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
            gpt2_forward(&model, h_inputs.data(), h_targets.data());
            HIP_CHECK(hipMemcpy(h_losses.data(), model.d_losses, B * T * sizeof(float), hipMemcpyDeviceToHost));
            for (int i = 0; i < B * T; i++) total_loss += h_losses[i];
            count += B * T;
        }
        return total_loss / count;
    };

    for (int iter = start_iter; iter < MAX_ITERS; iter++) {
        
        if (iter % EVAL_INTERVAL == 0) {
            float train_loss = eval_loss(train_tokens);
            float val_loss = eval_loss(val_tokens); // Evaluate validation set
            printf("Step %d: train loss %f, val loss %f\n", iter, train_loss, val_loss);
            gpt2_save_checkpoint(&model, iter, "gpt_hip_checkpoint.bin");
        }

        for (int b = 0; b < B; b++) {
            int ix = rand() % (n - T);
            for (int t = 0; t < T; t++) {
                h_inputs[b * T + t] = train_tokens[ix + t];
                h_targets[b * T + t] = train_tokens[ix + t + 1];
            }
        }

        gpt2_forward(&model, h_inputs.data(), h_targets.data());
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, LR, iter + 1);
    }
    
    gpt2_save_checkpoint(&model, MAX_ITERS, "gpt_hip_checkpoint.bin");

    printf("Generating text...\n");
    std::vector<int> gen_tokens(1, 0); 
    std::vector<float> h_probs(V);

    for (int i = 0; i < 500; i++) {
        int start_idx = std::max(0, (int)gen_tokens.size() - T);
        int context_len = std::min((int)gen_tokens.size(), T);
        
        // We still populate B=16 rows for simplicity, but only row 0 matters
        std::vector<int> h_gen_inputs(B * T, 0);
        for (int t = 0; t < context_len; t++) {
            h_gen_inputs[t] = gen_tokens[start_idx + t];
        }

        gpt2_forward(&model, h_gen_inputs.data(), NULL);
        
        int sample_pos = context_len - 1;
        // Offset by 0 * V because we only use batch 0
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

    gpt2_free(&model);
    return 0;
}