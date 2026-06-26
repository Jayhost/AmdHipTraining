// gen_gpt2_hip.cpp
// AMD HIP GPU implementation of GPT-2 text generation (inference only)
// Compile: hipcc -O3 gen_gpt2_hip.cpp -o gen_gpt2_hip
// Usage: ./gen_gpt2_hip [checkpoint] [prompt] [num_tokens] [temperature] [top_k]

#include <hip/hip_runtime.h>
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
#include <ctime>
#include <random>
#include <chrono>
#include <iostream>

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
// Hyperparameters (must match training)
// ============================================================
const int B = 1;          // batch_size (single sequence for generation)
const int T = 32;         // block_size
const int C = 64;         // n_embd
const int NH = 4;         // n_head
const int L = 4;          // n_layer
const int HS = C / NH;   // head_size = 16
const int BLOCK_SIZE_KERNEL = 256;

// ============================================================
// Tokenizer (character-level, identical to training)
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
        for (char ch : s) {
            if (stoi.find(ch) == stoi.end()) {
                fprintf(stderr, "Warning: character '%c' not in vocabulary, skipping\n", ch);
                continue;
            }
            r.push_back(stoi[ch]);
        }
        return r;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string r;
        for (int id : ids) {
            if (itos.find(id) != itos.end()) {
                r += itos[id];
            }
        }
        return r;
    }
};

// ============================================================
// HIP Kernels - Forward Pass Only
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

__global__ void scale_kernel(float* data, int N, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] *= scale;
}

// ============================================================
// Model Struct and Memory Management (generation-only)
// ============================================================

struct TensorDesc {
    float* d_ptr;
    size_t size;
};

struct GPT2 {
    int V;
    // Parameters
    float *d_wte, *d_wpe;
    float *d_ln1w, *d_ln1b, *d_qkvw, *d_qkvb, *d_attprojw, *d_attprojb;
    float *d_ln2w, *d_ln2b, *d_fcw, *d_fcb, *d_fcprojw, *d_fcprojb;
    float *d_lnfw, *d_lnfb;
    float *d_lm_head, *d_lm_bias;

    // Forward activations only
    float *d_encoded, *d_ln1, *d_ln1_mean, *d_ln1_rstd;
    float *d_qkv, *d_atty, *d_preatt, *d_att, *d_attproj, *d_residual2;
    float *d_ln2, *d_ln2_mean, *d_ln2_rstd;
    float *d_fch, *d_fch_relu, *d_fcproj, *d_residual3;
    float *d_lnf, *d_lnf_mean, *d_lnf_rstd;
    float *d_logits, *d_probs;
    int *d_inputs;
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

size_t calc_num_parameters(int V) {
    return (size_t)(V * C) + (T * C) +
           L * (C + C + 3*C*C + 3*C + C*C + C + C + C + 4*C*C + 4*C + C*4*C + C) + C + C +
           V * C + V;
}

void gpt2_alloc(GPT2 *model, int V) {
    model->V = V;
    model->num_parameters = calc_num_parameters(V);

    // Allocate parameters
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

    // Allocate forward activations (B=1)
    #define HIP_ALLOC(ptr, size) HIP_CHECK(hipMalloc(&ptr, size * sizeof(float)));
    HIP_ALLOC(model->d_encoded, B * T * C);
    HIP_ALLOC(model->d_ln1, L * B * T * C);
    HIP_ALLOC(model->d_ln1_mean, L * B * T);
    HIP_ALLOC(model->d_ln1_rstd, L * B * T);
    HIP_ALLOC(model->d_qkv, L * B * T * 3*C);
    HIP_ALLOC(model->d_atty, L * B * T * C);
    HIP_ALLOC(model->d_preatt, L * B * NH * T * T);
    HIP_ALLOC(model->d_att, L * B * NH * T * T);
    HIP_ALLOC(model->d_attproj, L * B * T * C);
    HIP_ALLOC(model->d_residual2, L * B * T * C);
    HIP_ALLOC(model->d_ln2, L * B * T * C);
    HIP_ALLOC(model->d_ln2_mean, L * B * T);
    HIP_ALLOC(model->d_ln2_rstd, L * B * T);
    HIP_ALLOC(model->d_fch, L * B * T * 4*C);
    HIP_ALLOC(model->d_fch_relu, L * B * T * 4*C);
    HIP_ALLOC(model->d_fcproj, L * B * T * C);
    HIP_ALLOC(model->d_residual3, L * B * T * C);
    HIP_ALLOC(model->d_lnf, B * T * C);
    HIP_ALLOC(model->d_lnf_mean, B * T);
    HIP_ALLOC(model->d_lnf_rstd, B * T);
    HIP_ALLOC(model->d_logits, B * T * V);
    HIP_ALLOC(model->d_probs, B * T * V);
    #undef HIP_ALLOC

    HIP_CHECK(hipMalloc(&model->d_inputs, B * T * sizeof(int)));
}

void gpt2_free(GPT2 *model) {
    auto params = get_param_list(model);
    for (auto& p : params) HIP_CHECK(hipFree(p.d_ptr));

    #define HIP_FREE(ptr) if(ptr) HIP_CHECK(hipFree(ptr));
    HIP_FREE(model->d_encoded); HIP_FREE(model->d_ln1); HIP_FREE(model->d_ln1_mean); HIP_FREE(model->d_ln1_rstd);
    HIP_FREE(model->d_qkv); HIP_FREE(model->d_atty); HIP_FREE(model->d_preatt); HIP_FREE(model->d_att);
    HIP_FREE(model->d_attproj); HIP_FREE(model->d_residual2); HIP_FREE(model->d_ln2); HIP_FREE(model->d_ln2_mean);
    HIP_FREE(model->d_ln2_rstd); HIP_FREE(model->d_fch); HIP_FREE(model->d_fch_relu); HIP_FREE(model->d_fcproj);
    HIP_FREE(model->d_residual3); HIP_FREE(model->d_lnf); HIP_FREE(model->d_lnf_mean); HIP_FREE(model->d_lnf_rstd);
    HIP_FREE(model->d_logits); HIP_FREE(model->d_probs);
    HIP_FREE(model->d_inputs);
    #undef HIP_FREE
}

// ============================================================
// Checkpoint Loading (parameters only, skip optimizer state)
// ============================================================

int gpt2_load_checkpoint(GPT2* model, const char* filepath) {
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "No checkpoint found at %s! A trained checkpoint is required for generation.\n", filepath);
        exit(1);
    }

    int step;
    fread(&step, sizeof(int), 1, file);

    auto params = get_param_list(model);
    std::vector<float> cpu_buf(model->num_parameters);

    // Read parameters
    fread(cpu_buf.data(), sizeof(float), model->num_parameters, file);
    size_t offset = 0;
    for (auto& p : params) {
        HIP_CHECK(hipMemcpy(p.d_ptr, cpu_buf.data() + offset, p.size * sizeof(float), hipMemcpyHostToDevice));
        offset += p.size;
    }

    // Skip optimizer state (m and v) — not needed for generation
    fclose(file);
    printf("Checkpoint loaded from %s (step %d). Parameters: %zu\n", filepath, step, model->num_parameters);
    return step;
}

// ============================================================
// GPU Kernel Launch Helper
// ============================================================
#define LAUNCH(kernel, N, ...) do { \
    int grid = (N + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL; \
    kernel<<<grid, BLOCK_SIZE_KERNEL>>>(__VA_ARGS__); \
    HIP_CHECK(hipGetLastError()); \
} while(0)

// ============================================================
// Forward Pass (generation mode — no loss computation)
// ============================================================
void gpt2_forward(GPT2 *model, const int* h_inputs, float temperature) {
    int V = model->V;
    HIP_CHECK(hipMemcpy(model->d_inputs, h_inputs, B * T * sizeof(int), hipMemcpyHostToDevice));

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

    LAUNCH(matmul_forward_kernel, B * T * V, model->d_logits, model->d_lnf, model->d_lm_head, model->d_lm_bias, B * T, C, V);

    // Apply temperature scaling to logits before softmax
    if (temperature != 1.0f) {
        LAUNCH(scale_kernel, B * T * V, model->d_logits, B * T * V, 1.0f / temperature);
    }

    LAUNCH(softmax_forward_kernel, B * T, model->d_probs, model->d_logits, B, T, V);
}

// ============================================================
// Sampling (CPU-side)
// ============================================================

int sample_greedy(const float* probs, int V) {
    int max_idx = 0;
    float max_val = probs[0];
    for (int i = 1; i < V; i++) {
        if (probs[i] > max_val) {
            max_val = probs[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int sample_multinomial(const float* probs, int V, int top_k) {
    if (top_k > 0 && top_k < V) {
        // Top-k: restrict to the k most probable tokens, then sample
        std::vector<std::pair<float, int>> prob_idx(V);
        for (int i = 0; i < V; i++) prob_idx[i] = {probs[i], i};
        std::partial_sort(prob_idx.begin(), prob_idx.begin() + top_k, prob_idx.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        // Renormalize top-k probabilities
        float sum = 0.0f;
        for (int i = 0; i < top_k; i++) sum += prob_idx[i].first;

        float r = (float)rand() / RAND_MAX * sum;
        float cdf = 0.0f;
        for (int i = 0; i < top_k; i++) {
            cdf += prob_idx[i].first;
            if (r < cdf) return prob_idx[i].second;
        }
        return prob_idx[top_k - 1].second;
    }

    // Standard multinomial sampling from full distribution
    float r = (float)rand() / RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < V; i++) {
        cdf += probs[i];
        if (r < cdf) return i;
    }
    return V - 1;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    // Defaults
    const char* checkpoint_path = "gpt_hip_checkpoint.bin";
    const char* data_path = "small.txt";
    std::string prompt = "\n";
    int num_tokens = 2000;
    float temperature = 0.8f;
    int top_k = 40;
    //unsigned int seed = 1337;
    unsigned seed = static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(0, 99);
    std::cout << d(rng) << '\n';

    if (argc >= 2) checkpoint_path = argv[1];
    if (argc >= 3) prompt = argv[2];
    if (argc >= 4) num_tokens = atoi(argv[3]);
    if (argc >= 5) temperature = atof(argv[4]);
    if (argc >= 6) top_k = atoi(argv[5]);
    if (argc >= 7) seed = (unsigned int)atoi(argv[6]);

    printf("=== GPT-2 Generation (HIP GPU) ===\n");
    printf("Checkpoint:   %s\n", checkpoint_path);
    printf("Data file:    %s (for tokenizer)\n", data_path);
    printf("Prompt:       \"%s\"\n", prompt.c_str());
    printf("Max tokens:   %d\n", num_tokens);
    printf("Temperature:  %.2f\n", temperature);
    printf("Top-k:        %d\n", top_k);
    printf("Seed:         %u\n", seed);
    printf("\n");

    srand(seed);

    // Load training text to build the identical tokenizer
    std::ifstream file(data_path);
    if (!file.is_open()) {
        fprintf(stderr, "Could not open %s for tokenizer initialization!\n", data_path);
        return 1;
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    CharTokenizer tokenizer;
    tokenizer.init(text);
    int V = tokenizer.vocab_size;
    printf("Vocab size: %d\n", V);

    // Allocate and load model
    GPT2 model;
    gpt2_alloc(&model, V);
    gpt2_load_checkpoint(&model, checkpoint_path);

    // Encode the prompt
    std::vector<int> gen_tokens = tokenizer.encode(prompt);
    if (gen_tokens.empty()) {
        fprintf(stderr, "Prompt encoded to 0 tokens. Using token 0 as start.\n");
        gen_tokens.push_back(0);
    }
    printf("Prompt: %zu tokens\n", gen_tokens.size());

    // Generation loop
    std::vector<int> h_inputs(B * T, 0);
    std::vector<float> h_probs(V);

    for (int i = 0; i < num_tokens; i++) {
        int seq_len = (int)gen_tokens.size();
        int start_idx = std::max(0, seq_len - T);
        int context_len = std::min(seq_len, T);

        // Fill input: most recent context_len tokens, pad with 0
        std::fill(h_inputs.begin(), h_inputs.end(), 0);
        for (int t = 0; t < context_len; t++) {
            h_inputs[t] = gen_tokens[start_idx + t];
        }

        // Forward pass with temperature
        gpt2_forward(&model, h_inputs.data(), temperature);

        // Copy probabilities at the last context position
        int sample_pos = context_len - 1;
        HIP_CHECK(hipMemcpy(h_probs.data(), model.d_probs + sample_pos * V, V * sizeof(float), hipMemcpyDeviceToHost));

        // Sample next token
        int next_token;
        if (temperature <= 0.01f) {
            next_token = sample_greedy(h_probs.data(), V);
        } else {
            next_token = sample_multinomial(h_probs.data(), V, top_k);
        }
        gen_tokens.push_back(next_token);

        // Progress indicator
        if ((i + 1) % 100 == 0) {
            printf("[%d/%d tokens generated]\n", i + 1, num_tokens);
        }
    }

    // Decode and print
    std::string generated_text = tokenizer.decode(gen_tokens);
    printf("\n=== Generated Text ===\n%s\n=== End ===\n", generated_text.c_str());

    gpt2_free(&model);
    return 0;
}