// train_qwen3_hip.cpp
// AMD HIP GPU implementation of Qwen3 Training with Pre-tokenized Data + Loss Masking
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
#include <unordered_map>
#include <utility>
#include <climits>
#include <cctype>

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
// Hyperparameters
// ============================================================
const int B = 4;          // batch_size
const int T = 256;        // max_seq_len (increased for longer stories)
const int MAX_ITERS = 50000;
const int EVAL_INTERVAL = 500;
const float LR = 5e-5f;   // slightly lower for bigger model
const int EVAL_ITERS = 5;
const int C = 512;        // d_model (was 256)
const int NH = 8;         // n_heads (was 4)
const int NKV = 2;        // n_kv_heads (keep small for efficiency)
const int NGRP = NH / NKV;// n_kv_groups = 4
const int DK = C / NH;    // d_k = 64
const int DFF = 1536;     // d_ff = 3 * C
const int L = 6;          // n_layers (was 4)
const float RMS_EPS = 1e-6f;
const int BLOCK_SIZE_KERNEL = 256;
const char* SYSTEM_PROMPT = "You are a helpful assistant.";

// ============================================================
// BPE Tokenizer (kept for generation - training uses pre-tokenized data)
// ============================================================
#define MAX_VOCAB 200000
#define MAX_MERGES 200000
#define MAX_TOKEN_LEN 256

struct MergeRule {
    char left[MAX_TOKEN_LEN];
    char right[MAX_TOKEN_LEN];
    char merged[MAX_TOKEN_LEN * 2];
    int rank;
};

struct Tokenizer {
    int vocab_size;
};

static char* vocab[MAX_VOCAB];
static int num_vocab = 0;
static MergeRule merges[MAX_MERGES];
static int merge_count = 0;
static unsigned int byte_to_unicode_map[256];
static char unicode_bytes[256][5];

const char* special_tokens[] = {
    "<|im_start|>",
    "<|im_end|>",
    "",
};
int num_special_tokens = sizeof(special_tokens) / sizeof(special_tokens[0]);

static std::unordered_map<std::string, int> g_vocab_map;
static std::map<std::pair<std::string, std::string>, int> g_merge_map;
static bool g_use_raw_bytes = false;

// ============================================================
// JSON parsing helpers
// ============================================================
static inline const char* skip_ws(const char* p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static std::string parse_json_string_val(const char** pp) {
    const char* p = *pp;
    p = skip_ws(p);
    if (*p != '"') { *pp = p; return ""; }
    p++;
    std::string result;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            if (!*p) break;
            switch (*p) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case '/':  result += '/'; break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                case 'b':  result += '\b'; break;
                case 'f':  result += '\f'; break;
                case 'u': {
                    p++;
                    unsigned int cp = 0;
                    for (int i = 0; i < 4 && *p; i++, p++) {
                        cp *= 16;
                        if (*p >= '0' && *p <= '9') cp += *p - '0';
                        else if (*p >= 'a' && *p <= 'f') cp += *p - 'a' + 10;
                        else if (*p >= 'A' && *p <= 'F') cp += *p - 'A' + 10;
                    }
                    p--;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        const char* np = p + 1;
                        while (*np && isspace(*np)) np++;
                        if (*np == '\\' && *(np + 1) == 'u') {
                            np += 2;
                            unsigned int cp2 = 0;
                            for (int i = 0; i < 4 && *np; i++, np++) {
                                cp2 *= 16;
                                if (*np >= '0' && *np <= '9') cp2 += *np - '0';
                                else if (*np >= 'a' && *np <= 'f') cp2 += *np - 'a' + 10;
                                else if (*np >= 'A' && *np <= 'F') cp2 += *np - 'A' + 10;
                            }
                            if (cp2 >= 0xDC00 && cp2 <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (cp2 - 0xDC00);
                                p = np - 1;
                            }
                        }
                    }
                    if (cp < 0x80) {
                        result += (char)cp;
                    } else if (cp < 0x800) {
                        result += (char)(0xC0 | (cp >> 6));
                        result += (char)(0x80 | (cp & 0x3F));
                    } else if (cp < 0x10000) {
                        result += (char)(0xE0 | (cp >> 12));
                        result += (char)(0x80 | ((cp >> 6) & 0x3F));
                        result += (char)(0x80 | (cp & 0x3F));
                    } else {
                        result += (char)(0xF0 | (cp >> 18));
                        result += (char)(0x80 | ((cp >> 12) & 0x3F));
                        result += (char)(0x80 | ((cp >> 6) & 0x3F));
                        result += (char)(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: result += *p; break;
            }
        } else {
            result += *p;
        }
        p++;
    }
    if (*p == '"') p++;
    *pp = p;
    return result;
}

static int parse_json_int(const char** pp) {
    const char* p = *pp;
    p = skip_ws(p);
    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    int val = 0;
    while (*p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); p++; }
    *pp = p;
    return sign * val;
}

static void skip_json_value(const char** pp) {
    const char* p = *pp;
    p = skip_ws(p);
    if (*p == '"') { parse_json_string_val(&p); }
    else if (*p == '{') {
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == '"') { parse_json_string_val(&p); continue; }
            if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') depth--;
            p++;
        }
    } else if (*p == '[') {
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == '"') { parse_json_string_val(&p); continue; }
            if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') depth--;
            p++;
        }
    } else {
        while (*p && *p != ',' && *p != '}' && *p != ']') p++;
    }
    *pp = p;
}

// ============================================================
// Byte-to-unicode mapping (GPT-2 style)
// ============================================================
void init_byte_unicode_map() {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            byte_to_unicode_map[b] = b;
        } else {
            byte_to_unicode_map[b] = 256 + n++;
        }
        uint32_t cp = byte_to_unicode_map[b];
        if (cp < 128) {
            unicode_bytes[b][0] = cp; unicode_bytes[b][1] = '\0';
        } else if (cp < 2048) {
            unicode_bytes[b][0] = 0xC0 | (cp >> 6);
            unicode_bytes[b][1] = 0x80 | (cp & 0x3F);
            unicode_bytes[b][2] = '\0';
        } else {
            unicode_bytes[b][0] = 0xE0 | (cp >> 12);
            unicode_bytes[b][1] = 0x80 | ((cp >> 6) & 0x3F);
            unicode_bytes[b][2] = 0x80 | (cp & 0x3F);
            unicode_bytes[b][3] = '\0';
        }
    }
}

int unicode_to_byte(const char* utf8) {
    for (int b = 0; b < 256; b++) {
        if (strcmp(utf8, unicode_bytes[b]) == 0) return b;
    }
    return -1;
}

// ============================================================
// Load vocab
// ============================================================
void load_vocab(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* data = (char*)malloc(fsize + 1);
    fread(data, 1, fsize, f);
    data[fsize] = '\0';
    fclose(f);
    for (int i = 0; i < MAX_VOCAB; i++) vocab[i] = NULL;
    num_vocab = 0;
    const char* p = data;
    while (*p && isspace(*p)) p++;
    if (*p == '{') {
        const char* vocab_start = NULL;
        const char* added_start = NULL;
        const char* s = data;
        while ((s = strstr(s, "\"vocab\"")) != NULL) {
            const char* q = s + 7; while (*q && isspace(*q)) q++;
            if (*q == ':') { q++; while (*q && isspace(*q)) q++; if (*q == '{') { vocab_start = q; break; } }
            s += 7;
        }
        s = data;
        while ((s = strstr(s, "\"added_tokens\"")) != NULL) {
            const char* q = s + 14; while (*q && isspace(*q)) q++;
            if (*q == ':') { q++; while (*q && isspace(*q)) q++; if (*q == '[') { added_start = q; break; } }
            s += 14;
        }
        if (!vocab_start) vocab_start = p;
        p = vocab_start;
        if (*p == '{') p++;
        while (*p && *p != '}') {
            p = skip_ws(p);
            if (*p == '}' || *p == '\0') break;
            if (*p == ',') { p++; continue; }
            std::string key = parse_json_string_val(&p);
            p = skip_ws(p); if (*p == ':') p++; p = skip_ws(p);
            int val = parse_json_int(&p);
            if (val >= 0 && val < MAX_VOCAB) {
                if (vocab[val]) free(vocab[val]);
                vocab[val] = strdup(key.c_str());
                if (val + 1 > num_vocab) num_vocab = val + 1;
            }
        }
        if (added_start) {
            p = added_start; if (*p == '[') p++;
            while (*p && *p != ']') {
                p = skip_ws(p);
                if (*p == ']') break;
                if (*p == ',') { p++; continue; }
                if (*p == '{') {
                    std::string content; int id = -1; p++;
                    while (*p && *p != '}') {
                        p = skip_ws(p); if (*p == '}') break; if (*p == ',') { p++; continue; }
                        std::string obj_key = parse_json_string_val(&p);
                        p = skip_ws(p); if (*p == ':') p++; p = skip_ws(p);
                        if (obj_key == "content") content = parse_json_string_val(&p);
                        else if (obj_key == "id") id = parse_json_int(&p);
                        else skip_json_value(&p);
                    }
                    if (*p == '}') p++;
                    if (id >= 0 && id < MAX_VOCAB && !content.empty()) {
                        if (vocab[id]) free(vocab[id]);
                        vocab[id] = strdup(content.c_str());
                        if (id + 1 > num_vocab) num_vocab = id + 1;
                    }
                } else { p++; }
            }
        }
        printf("Loaded vocab from JSON: %d entries\n", num_vocab);
    } else {
        std::istringstream iss(data);
        std::string line;
        bool has_index_prefix = false;
        while (std::getline(iss, line)) {
            while (!line.empty() && isspace(line.back())) line.pop_back();
            while (!line.empty() && isspace(line.front())) line = line.substr(1);
            if (line.empty()) continue;
            if (!line.empty() && isdigit(line[0])) has_index_prefix = true;
            break;
        }
        iss.clear(); iss.str(data);
        while (std::getline(iss, line)) {
            while (!line.empty() && isspace(line.back())) line.pop_back();
            while (!line.empty() && isspace(line.front())) line = line.substr(1);
            if (line.empty()) continue;
            if (has_index_prefix) {
                size_t sp = line.find(' ');
                if (sp != std::string::npos) {
                    int idx = std::stoi(line.substr(0, sp));
                    std::string tok = line.substr(sp + 1);
                    if (idx >= 0 && idx < MAX_VOCAB) {
                        if (vocab[idx]) free(vocab[idx]);
                        vocab[idx] = strdup(tok.c_str());
                        if (idx + 1 > num_vocab) num_vocab = idx + 1;
                    }
                }
            } else {
                if (num_vocab < MAX_VOCAB) { vocab[num_vocab] = strdup(line.c_str()); num_vocab++; }
            }
        }
        printf("Loaded vocab from text: %d entries\n", num_vocab);
    }
    free(data);
}

void load_merges(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { perror("merges.txt"); exit(1); }
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || isspace(line[0])) continue;
        char* a = strtok(line, " \t\r\n"); char* b = strtok(NULL, " \t\r\n");
        if (!a || !b) continue;
        strncpy(merges[merge_count].left, a, MAX_TOKEN_LEN);
        strncpy(merges[merge_count].right, b, MAX_TOKEN_LEN);
        snprintf(merges[merge_count].merged, sizeof(merges[merge_count].merged), "%s%s", a, b);
        merges[merge_count].rank = merge_count;
        merge_count++;
    }
    fclose(f);
    printf("Loaded %d merge rules from %s\n", merge_count, path);
}

void build_lookup_maps() {
    g_vocab_map.clear(); g_merge_map.clear();
    for (int i = 0; i < num_vocab; i++) if (vocab[i]) g_vocab_map[std::string(vocab[i])] = i;
    for (int i = 0; i < merge_count; i++) g_merge_map[{std::string(merges[i].left), std::string(merges[i].right)}] = merges[i].rank;
}

int match_special_token(const char* str, int* match_len) {
    for (int i = 0; i < num_special_tokens; i++) {
        const char* tok = special_tokens[i]; int len = strlen(tok);
        if (strncmp(str, tok, len) == 0) {
            auto it = g_vocab_map.find(std::string(tok));
            if (it != g_vocab_map.end()) { *match_len = len; return it->second; }
        }
    }
    return -1;
}

char* decode_token_id(int token_id) {
    if (token_id < 0 || token_id >= num_vocab || !vocab[token_id]) {
        char* out = (char*)malloc(8); strcpy(out, "<?>"); return out;
    }
    const char* encoded = vocab[token_id];
    char* out = (char*)malloc(1024); int rlen = 0;
    for (int i = 0; encoded[i]; ) {
        int matched = 0;
        for (int len = 1; len <= 4 && encoded[i + len - 1]; len++) {
            char utf8[6] = {0}; strncpy(utf8, &encoded[i], len);
            int b = unicode_to_byte(utf8);
            if (b >= 0) { out[rlen++] = (char)b; i += len; matched = 1; break; }
        }
        if (!matched) out[rlen++] = encoded[i++];
    }
    out[rlen] = '\0'; return out;
}

void free_vocab() {
    for (int i = 0; i < num_vocab; i++) { if (vocab[i]) { free(vocab[i]); vocab[i] = NULL; } }
    num_vocab = 0;
}

void detect_vocab_encoding() {
    int gpt2_count = 0;
    for (int b = 0; b < 256; b++) { std::string ub(unicode_bytes[b]); if (g_vocab_map.find(ub) != g_vocab_map.end()) gpt2_count++; }
    int raw_count = 0;
    for (int b = 1; b < 256; b++) { char raw[2] = {(char)b, '\0'}; std::string rs(raw, 1); if (g_vocab_map.find(rs) != g_vocab_map.end()) raw_count++; }
    g_use_raw_bytes = (raw_count > gpt2_count);
    printf("Vocab encoding: %s (GPT2=%d, raw=%d)\n", g_use_raw_bytes ? "RAW BYTES" : "GPT-2 UNICODE", gpt2_count, raw_count);
}

std::string byte_to_token(unsigned char b) {
    return g_use_raw_bytes ? std::string(1, (char)b) : std::string(unicode_bytes[b]);
}

std::vector<std::string> pre_tokenize(const std::string& text) {
    std::vector<std::string> chunks; size_t i = 0;
    while (i < text.size()) {
        int match_len = 0; int special_id = match_special_token(text.c_str() + i, &match_len);
        if (special_id >= 0) { chunks.push_back(std::string(vocab[special_id])); i += match_len; continue; }
        size_t start = i; unsigned char c = text[i];
        if (c == ' ') { i++; while (i < text.size() && text[i] != ' ' && text[i] != '\n' && text[i] != '\r') i++; }
        else if (c == '\n' || c == '\r') { i++; }
        else if (std::isalpha(c)) { while (i < text.size() && std::isalpha((unsigned char)text[i])) i++; }
        else if (std::isdigit(c)) { while (i < text.size() && std::isdigit((unsigned char)text[i])) i++; }
        else { i++; }
        if (i > start) chunks.push_back(text.substr(start, i - start));
    }
    return chunks;
}

std::vector<int> bpe_encode_word(const std::string& word) {
    if (word.empty()) return {};
    std::vector<std::string> tokens;
    for (unsigned char c : word) tokens.push_back(byte_to_token(c));
    while (tokens.size() > 1) {
        int best_rank = INT_MAX; int best_pos = -1;
        for (size_t j = 0; j < tokens.size() - 1; j++) {
            auto key = std::make_pair(tokens[j], tokens[j + 1]);
            auto it = g_merge_map.find(key);
            if (it != g_merge_map.end() && it->second < best_rank) {
                std::string merged = tokens[j] + tokens[j + 1];
                if (g_vocab_map.find(merged) != g_vocab_map.end()) { best_rank = it->second; best_pos = (int)j; }
            }
        }
        if (best_pos == -1) break;
        tokens[best_pos] += tokens[best_pos + 1];
        tokens.erase(tokens.begin() + best_pos + 1);
    }
    std::vector<int> ids;
    for (auto& tok : tokens) {
        auto it = g_vocab_map.find(tok);
        if (it != g_vocab_map.end()) ids.push_back(it->second);
    }
    return ids;
}

std::vector<int> bpe_encode_text(const std::string& text) {
    std::vector<int> all_tokens;
    auto chunks = pre_tokenize(text);
    for (auto& chunk : chunks) {
        int match_len = 0; int special_id = match_special_token(chunk.c_str(), &match_len);
        if (special_id >= 0 && (size_t)match_len == chunk.size()) { all_tokens.push_back(special_id); continue; }
        auto ids = bpe_encode_word(chunk);
        all_tokens.insert(all_tokens.end(), ids.begin(), ids.end());
    }
    return all_tokens;
}

std::string decode_ids(const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) {
        if (id < 0 || id >= num_vocab || !vocab[id]) { result += "<?>"; continue; }
        if (g_use_raw_bytes) { result += vocab[id]; }
        else { char* decoded = decode_token_id(id); result += decoded; free(decoded); }
    }
    return result;
}

void build_tokenizer(Tokenizer* t) {
    load_vocab("vocab.txt"); load_merges("merges.txt");
    init_byte_unicode_map(); build_lookup_maps(); detect_vocab_encoding();
    t->vocab_size = num_vocab;
}

// ============================================================
// Pre-tokenized data loader (NEW - for training)
// ============================================================
struct PreTokenizedData {
    int* token_ids;
    int* loss_masks;
    size_t num_tokens;
};

PreTokenizedData load_pretokenized(const char* ids_path, const char* masks_path) {
    PreTokenizedData data;
    FILE* f_ids = fopen(ids_path, "rb");
    if (!f_ids) {
        fprintf(stderr, "Cannot open %s - run prepare_qwen3_data.py first!\n", ids_path);
        data.token_ids = NULL; data.loss_masks = NULL; data.num_tokens = 0;
        return data;
    }
    fseek(f_ids, 0, SEEK_END);
    data.num_tokens = ftell(f_ids) / sizeof(int);
    fseek(f_ids, 0, SEEK_SET);
    data.token_ids = (int*)malloc(data.num_tokens * sizeof(int));
    fread(data.token_ids, sizeof(int), data.num_tokens, f_ids);
    fclose(f_ids);

    FILE* f_masks = fopen(masks_path, "rb");
    if (!f_masks) {
        fprintf(stderr, "Cannot open %s\n", masks_path); exit(1);
    }
    data.loss_masks = (int*)malloc(data.num_tokens * sizeof(int));
    fread(data.loss_masks, sizeof(int), data.num_tokens, f_masks);
    fclose(f_masks);

    int assistant_tokens = 0;
    for (size_t i = 0; i < data.num_tokens; i++) if (data.loss_masks[i]) assistant_tokens++;
    printf("Loaded %zu pre-tokenized tokens from %s\n", data.num_tokens, ids_path);
    printf("  Assistant tokens (loss=1): %d (%.1f%%)\n",
           assistant_tokens, 100.0 * assistant_tokens / data.num_tokens);
    return data;
}

void free_pretokenized(PreTokenizedData* data) {
    if (data->token_ids) free(data->token_ids);
    if (data->loss_masks) free(data->loss_masks);
    data->num_tokens = 0;
}

// ============================================================
// HIP Kernels - Forward Pass
// ============================================================

__global__ void encoder_forward_kernel(float* out, const int* inp,
                                       const float* wte, int B, int T, int C) {
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
    if (i % 2 != 0) return;
    int h = (idx / d_k) % n_heads;
    int t = (idx / (d_k * n_heads)) % T;
    int half_idx = i / 2;
    int half = d_k / 2;
    float cos_val = cos_table[t * half + half_idx];
    float sin_val = sin_table[t * half + half_idx];
    int bth_base = idx - i;
    float x1 = inp[bth_base + i];
    float x2 = inp[bth_base + i + 1];
    out[bth_base + i]     = x1 * cos_val - x2 * sin_val;
    out[bth_base + i + 1] = x1 * sin_val + x2 * cos_val;
}

__global__ void gqa_attention_forward_kernel(
    float* out, float* att,
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
    float maxval = -1e10f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* key = k_b + t2 * NKV * DK + kv_h * DK;
        float val = 0.0f;
        for (int i = 0; i < DK; i++) val += q_bth[i] * key[i];
        val *= scale;
        if (val > maxval) maxval = val;
    }
    for (int t2 = t + 1; t2 < T; t2++) att_bth[t2] = 0.0f;
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

// NEW: Loss-masked cross-entropy forward
__global__ void crossentropy_forward_masked_kernel(float* losses, const float* probs,
                                                   const int* targets, const int* loss_mask,
                                                   int B, int T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;
    if (loss_mask[idx] == 0) { losses[idx] = 0.0f; return; }
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
    for (int i = 0; i < C; i++) dnorm_sum += weight[i] * dout_bt[i] * inp_bt[i];
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
    for (int bt = 0; bt < total_bt; bt++) dw += dout[bt * OC + o] * inp[bt * C + i];
    dweight[idx] += dw;
}

__global__ void rope_backward_kernel(float* dinp, const float* dout,
                                     const float* cos_table, const float* sin_table,
                                     int B, int T, int n_heads, int d_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * n_heads * d_k;
    if (idx >= N) return;
    int i = idx % d_k;
    if (i % 2 != 0) return;
    int h = (idx / d_k) % n_heads;
    int t = (idx / (d_k * n_heads)) % T;
    int half_idx = i / 2;
    int half = d_k / 2;
    float cos_val = cos_table[t * half + half_idx];
    float sin_val = sin_table[t * half + half_idx];
    int bth_base = idx - i;
    float dy1 = dout[bth_base + i];
    float dy2 = dout[bth_base + i + 1];
    float dx1 = dy1 * cos_val + dy2 * sin_val;
    float dx2 = -dy1 * sin_val + dy2 * cos_val;
    dinp[bth_base + i]     += dx1;
    dinp[bth_base + i + 1] += dx2;
}

__global__ void gqa_attention_backward_kernel(
    float* dq, float* dk, float* dv,
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
    for (int t2 = 0; t2 <= t; t2++) {
        float datt_t2 = 0.0f;
        const float* value_t2 = v_b + t2 * NKV * DK + kv_h * DK;
        for (int i = 0; i < DK; i++) datt_t2 += value_t2[i] * dout_bth[i];
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
    float dsilu = sigmoid + g * sigmoid * (1.0f - sigmoid);
    dgate[idx] += dout[idx] * up[idx] * dsilu;
    dup[idx] += dout[idx] * (g * sigmoid);
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

// NEW: Loss-masked backward kernel
__global__ void crossentropy_softmax_backward_masked_kernel(float* dlogits,
                                                            const float* dlosses, const float* probs,
                                                            const int* targets, const int* loss_mask,
                                                            int B, int T, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;
    float* dlogits_bt = dlogits + idx * V;
    // Zero out masked positions completely
    if (loss_mask[idx] == 0) {
        for (int i = 0; i < V; i++) dlogits_bt[i] = 0.0f;
        return;
    }
    const float* probs_bt = probs + idx * V;
    float dloss = dlosses[idx];
    int ix = targets[idx];
    for (int i = 0; i < V; i++) {
        float indicator = (i == ix) ? 1.0f : 0.0f;
        dlogits_bt[i] = (probs_bt[i] - indicator) * dloss;
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
struct TensorDesc { float* d_ptr; size_t size; };

struct Qwen3 {
    int V;
    float *d_wte;
    float *d_rms1w, *d_qw, *d_kw, *d_vw, *d_q_norm_w, *d_k_norm_w, *d_attprojw, *d_rms2w;
    float *d_gate_proj_w, *d_up_proj_w, *d_down_proj_w;
    float *d_rmsfw;  // d_lm_head REMOVED — tied to d_wte
    float *d_dwte;
    float *d_drms1w, *d_dqw, *d_dkw, *d_dvw, *d_dq_norm_w, *d_dk_norm_w, *d_dattprojw, *d_drms2w;
    float *d_dgate_proj_w, *d_dup_proj_w, *d_ddown_proj_w;
    float *d_drmsfw;  // d_dlm_head REMOVED — tied to d_dwte
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
    int *d_loss_mask;  // NEW: loss mask on GPU
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
        {model->d_down_proj_w, (size_t)L * C * DFF}, {model->d_rmsfw, (size_t)C}
        // d_lm_head REMOVED — tied to d_wte
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
        {model->d_ddown_proj_w, (size_t)L * C * DFF}, {model->d_drmsfw, (size_t)C}
        // d_dlm_head REMOVED — tied to d_dwte
    };
}

size_t calc_num_parameters(int V) {
    return (size_t)(V * C) + L * (
        C + NH * DK * C + NKV * DK * C + NKV * DK * C +
        DK + DK + C * C + C +
        DFF * C + DFF * C + C * DFF
    ) + C;
}
void precompute_rope(float* cos_table, float* sin_table, int max_seq_len, int d_k) {
    int half = d_k / 2;
    std::vector<float> freqs(half, 0.0f);
    for (int i = 0; i < half; i++) freqs[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)d_k);
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
    model->d_rmsfw = params[12].d_ptr;

    auto grads = get_grad_list(model);
    for (auto& g : grads) HIP_CHECK(hipMalloc(&g.d_ptr, g.size * sizeof(float)));
    model->d_dwte = grads[0].d_ptr; model->d_drms1w = grads[1].d_ptr; model->d_dqw = grads[2].d_ptr;
    model->d_dkw = grads[3].d_ptr; model->d_dvw = grads[4].d_ptr; model->d_dq_norm_w = grads[5].d_ptr;
    model->d_dk_norm_w = grads[6].d_ptr; model->d_dattprojw = grads[7].d_ptr; model->d_drms2w = grads[8].d_ptr;
    model->d_dgate_proj_w = grads[9].d_ptr; model->d_dup_proj_w = grads[10].d_ptr; model->d_ddown_proj_w = grads[11].d_ptr;
    model->d_drmsfw = grads[12].d_ptr;
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
    HIP_ALLOC(model->d_att, L * B * NH * T * T);
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
    HIP_CHECK(hipMalloc(&model->d_loss_mask, B * T * sizeof(int)));  // NEW

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
    HIP_FREE(model->d_inputs); HIP_FREE(model->d_targets); HIP_FREE(model->d_loss_mask);
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
// Gradient Clipping (MUST be after Qwen3 struct and get_grad_list)
// ============================================================
__global__ void clip_gradients_kernel(float* grads, size_t N, float max_norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float g = grads[idx];
    // NaN check
    if (g != g) { grads[idx] = 0.0f; return; }
    // Inf check
    if (g > 1e10f || g < -1e10f) { grads[idx] = 0.0f; return; }
    // Clip
    if (g > max_norm) grads[idx] = max_norm;
    if (g < -max_norm) grads[idx] = -max_norm;
}

void qwen3_clip_gradients(Qwen3* model, float max_norm) {
    auto grads = get_grad_list(model);
    for (auto& g : grads) {
        if (g.size > 0) {
            int grid = (g.size + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL;
            clip_gradients_kernel<<<grid, BLOCK_SIZE_KERNEL>>>(g.d_ptr, g.size, max_norm);
        }
    }
    // Also clip the logit gradients (the biggest ones)
    int grid_logits = (B * T * model->V + BLOCK_SIZE_KERNEL - 1) / BLOCK_SIZE_KERNEL;
    clip_gradients_kernel<<<grid_logits, BLOCK_SIZE_KERNEL>>>(model->d_dlogits, (size_t)B * T * model->V, max_norm);
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
// Forward Pass (MODIFIED: added loss_mask parameter)
// ============================================================
void qwen3_forward(Qwen3 *model, const int* h_inputs, const int* h_targets,
                   const int* h_loss_mask) {
    int V = model->V;
    HIP_CHECK(hipMemcpy(model->d_inputs, h_inputs, B * T * sizeof(int), hipMemcpyHostToDevice));
    if (h_targets != NULL) {
        HIP_CHECK(hipMemcpy(model->d_targets, h_targets, B * T * sizeof(int), hipMemcpyHostToDevice));
    }
    if (h_loss_mask != NULL) {
        HIP_CHECK(hipMemcpy(model->d_loss_mask, h_loss_mask, B * T * sizeof(int), hipMemcpyHostToDevice));
    }

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
        // TIED WEIGHTS: output projection reuses token embedding weights
    LAUNCH(matmul_forward_kernel, B * T * V, model->d_logits, model->d_rmsf, model->d_wte, NULL, B * T, C, V);
    LAUNCH(softmax_forward_kernel, B * T, model->d_probs, model->d_logits, B, T, V);

    if (h_targets != NULL) {
        if (h_loss_mask != NULL) {
            // NEW: Use masked cross-entropy
            LAUNCH(crossentropy_forward_masked_kernel, B * T, model->d_losses, model->d_probs, model->d_targets, model->d_loss_mask, B, T, V);
        } else {
            // Fallback: no masking (legacy mode)
            LAUNCH(crossentropy_forward_kernel, B * T, model->d_losses, model->d_probs, model->d_targets, B, T, V);
        }
    }
}

// ============================================================
// Backward Pass (MODIFIED: loss_scale parameter, masked backward)
// ============================================================
void qwen3_backward(Qwen3 *model, float loss_scale, bool use_loss_mask) {
    int V = model->V;

    // Set dlosses to loss_scale for all positions
    std::vector<float> h_dlosses(B * T, loss_scale);
    HIP_CHECK(hipMemcpy(model->d_losses, h_dlosses.data(), B * T * sizeof(float), hipMemcpyHostToDevice));

    // Zero d_dlogits before backward (safety)
    HIP_CHECK(hipMemset(model->d_dlogits, 0, B * T * V * sizeof(float)));

    if (use_loss_mask) {
        LAUNCH(crossentropy_softmax_backward_masked_kernel, B * T, model->d_dlogits, model->d_losses, model->d_probs, model->d_targets, model->d_loss_mask, B, T, V);
    } else {
        LAUNCH(crossentropy_softmax_backward_kernel, B * T, model->d_dlogits, model->d_losses, model->d_probs, model->d_targets, B, T, V);
    }

        // TIED WEIGHTS: use d_wte instead of d_lm_head for backward pass
    LAUNCH(matmul_backward_dinp_kernel, B * T * C, model->d_drmsf, model->d_dlogits, model->d_wte, B * T, C, V);
    LAUNCH(matmul_backward_dweight_dbias_kernel, V * C, model->d_dwte, NULL, model->d_dlogits, model->d_rmsf, B * T, C, V);

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
    }
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
            adamw_kernel<<<grid, BLOCK_SIZE_KERNEL>>>(params[i].d_ptr, grads[i].d_ptr, model->d_m + m_offset, model->d_v_adam + m_offset, params[i].size, learning_rate, 0.9f, 0.999f, 1e-8f, 0.01f, step);
            // weight_decay changed from 0.1 to 0.01
            m_offset += params[i].size;
        }
    }
}

// ============================================================
// Main
// ============================================================
int main() {
    srand(42);

    printf("============================================================\n");
    printf("Qwen3 HIP Training with Loss Masking\n");
    printf("B=%d T=%d C=%d NH=%d NKV=%d DK=%d DFF=%d L=%d\n", B, T, C, NH, NKV, DK, DFF, L);
    printf("============================================================\n");

    // ================================================================
    // 1. Load training data
    // ================================================================
    bool use_pretokenized = false;
    PreTokenizedData train_data = {};
    PreTokenizedData val_data = {};
    std::vector<int> legacy_train_tokens;
    std::vector<int> legacy_val_tokens;
    int V = 151643; // Qwen3 vocab size default

    // Try loading pre-tokenized data first
    train_data = load_pretokenized("pretokenized/train_ids.bin", "pretokenized/train_masks.bin");
    if (train_data.num_tokens > 0) {
        val_data = load_pretokenized("pretokenized/val_ids.bin", "pretokenized/val_masks.bin");
        if (val_data.num_tokens > 0) {
            use_pretokenized = true;
            // Auto-detect V from data
            for (size_t i = 0; i < train_data.num_tokens; i++)
                if (train_data.token_ids[i] >= V) V = train_data.token_ids[i] + 1;
            printf("Using PRE-TOKENIZED data with loss masking\n");
            printf("  Train: %zu tokens, Val: %zu tokens\n", train_data.num_tokens, val_data.num_tokens);
        }
    }

    if (!use_pretokenized) {
        // Fallback: load raw text + BPE encode (no loss masking)
        printf("\nWARNING: No pre-tokenized data found!\n");
        printf("Run: python prepare_qwen3_data.py alpaca\n");
        printf("Falling back to input.txt with BPE encoding (NO loss masking)\n\n");

        std::ifstream file("input.txt");
        if (!file.is_open()) {
            fprintf(stderr, "ERROR: No input.txt either! Please provide training data.\n");
            return 1;
        }
        std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        Tokenizer tokenizer;
        build_tokenizer(&tokenizer);
        V = tokenizer.vocab_size;

        printf("Encoding text with BPE...\n");
        std::vector<int> tokens = bpe_encode_text(text);
        printf("Encoded %zu chars -> %zu tokens\n", text.size(), tokens.size());

        int n = (int)(0.9 * tokens.size());
        legacy_train_tokens = std::vector<int>(tokens.begin(), tokens.begin() + n);
        legacy_val_tokens = std::vector<int>(tokens.begin() + n, tokens.end());
    }

    printf("Vocab size V = %d\n", V);
    printf("Parameters: %zu\n", calc_num_parameters(V));

    // ================================================================
    // 2. Allocate model
    // ================================================================
    Qwen3 model;
    qwen3_alloc(&model, V);

    int start_iter = qwen3_load_checkpoint(&model, "qwen3_hip_checkpoint.bin");

        if (start_iter == 0) {
        printf("Initializing Qwen3 model weights from scratch...\n");
        std::mt19937 gen(1337);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        auto init_random = [&](float* d_ptr, size_t size) {
            std::vector<float> h_buf(size);
            for (size_t i = 0; i < size; i++) h_buf[i] = dist(gen);
            HIP_CHECK(hipMemcpy(d_ptr, h_buf.data(), size * sizeof(float), hipMemcpyHostToDevice));
        };
        auto init_val = [&](float* d_ptr, size_t size, float val) {
            std::vector<float> h_buf(size, val);
            HIP_CHECK(hipMemcpy(d_ptr, h_buf.data(), size * sizeof(float), hipMemcpyHostToDevice));
        };

        // Use Xavier/He initialization for better stability with larger model
        float emb_std = 1.0f / sqrtf((float)C);
        std::normal_distribution<float> emb_dist(0.0f, emb_std);
        {
            std::vector<float> h_buf(V * C);
            for (size_t i = 0; i < (size_t)(V * C); i++) h_buf[i] = emb_dist(gen);
            HIP_CHECK(hipMemcpy(model.d_wte, h_buf.data(), V * C * sizeof(float), hipMemcpyHostToDevice));
        }

        for (int l = 0; l < L; l++) {
            init_val(model.d_rms1w + l*C, C, 1.0f);

            // He initialization for attention projections
            float attn_std = 1.0f / sqrtf((float)C);
            std::normal_distribution<float> attn_dist(0.0f, attn_std);
            
            auto init_he = [&](float* d_ptr, size_t size) {
                std::vector<float> h_buf(size);
                for (size_t i = 0; i < size; i++) h_buf[i] = attn_dist(gen);
                HIP_CHECK(hipMemcpy(d_ptr, h_buf.data(), size * sizeof(float), hipMemcpyHostToDevice));
            };
            
            init_he(model.d_qw + l*NH*DK*C, NH*DK*C);
            init_he(model.d_kw + l*NKV*DK*C, NKV*DK*C);
            init_he(model.d_vw + l*NKV*DK*C, NKV*DK*C);
            init_val(model.d_q_norm_w + l*DK, DK, 1.0f);
            init_val(model.d_k_norm_w + l*DK, DK, 1.0f);
            init_he(model.d_attprojw + l*C*C, C*C);

            init_val(model.d_rms2w + l*C, C, 1.0f);

            // Smaller init for FFN (residual scaling)
            float ffn_std = 1.0f / sqrtf((float)(2.0f * L));
            std::normal_distribution<float> ffn_dist(0.0f, ffn_std);
            auto init_ffn = [&](float* d_ptr, size_t size) {
                std::vector<float> h_buf(size);
                for (size_t i = 0; i < size; i++) h_buf[i] = ffn_dist(gen);
                HIP_CHECK(hipMemcpy(d_ptr, h_buf.data(), size * sizeof(float), hipMemcpyHostToDevice));
            };
            init_ffn(model.d_gate_proj_w + l*DFF*C, DFF*C);
            init_ffn(model.d_up_proj_w + l*DFF*C, DFF*C);
            init_ffn(model.d_down_proj_w + l*C*DFF, C*DFF);
        }

        init_val(model.d_rmsfw, C, 1.0f);
    }
    printf("Model ready. Parameters: %zu\n", model.num_parameters);

    // ================================================================
    // 3. Training buffers
    // ================================================================
    std::vector<int> h_inputs(B * T);
    std::vector<int> h_targets(B * T);
    std::vector<int> h_loss_mask(B * T, 1); // default: all 1s (no masking)
    std::vector<float> h_losses(B * T);

    // ================================================================
    // 4. Batch sampling functions
    // ================================================================
    auto get_batch_pretokenized = [&](const PreTokenizedData& data) {
        int data_len = (int)data.num_tokens;
        for (int b = 0; b < B; b++) {
            int ix = rand() % (data_len - T - 1);
            for (int t = 0; t < T; t++) {
                h_inputs[b * T + t] = data.token_ids[ix + t];
                h_targets[b * T + t] = data.token_ids[ix + t + 1];
                // Loss mask: check if TARGET position is an assistant token
                h_loss_mask[b * T + t] = data.loss_masks[ix + t + 1];
            }
        }
    };

    auto get_batch_legacy = [&](const std::vector<int>& tokens) {
        int data_len = (int)tokens.size();
        for (int b = 0; b < B; b++) {
            int ix = rand() % (data_len - T - 1);
            for (int t = 0; t < T; t++) {
                h_inputs[b * T + t] = tokens[ix + t];
                h_targets[b * T + t] = tokens[ix + t + 1];
                h_loss_mask[b * T + t] = 1; // no masking in legacy mode
            }
        }
    };

    // ================================================================
    // 5. Evaluation
    // ================================================================
    auto eval_loss = [&]() -> std::pair<float,float> {
        float train_loss = 0.0f, val_loss = 0.0f;
        int train_masked = 0, val_masked = 0;

        // Train loss
        for (int k = 0; k < EVAL_ITERS; k++) {
            if (use_pretokenized) get_batch_pretokenized(train_data);
            else get_batch_legacy(legacy_train_tokens);

            qwen3_forward(&model, h_inputs.data(), h_targets.data(),
                          use_pretokenized ? h_loss_mask.data() : NULL);
            HIP_CHECK(hipMemcpy(h_losses.data(), model.d_losses, B * T * sizeof(float), hipMemcpyDeviceToHost));

            for (int i = 0; i < B * T; i++) {
                if (!use_pretokenized || h_loss_mask[i]) {
                    train_loss += h_losses[i];
                    train_masked++;
                }
            }
        }

        // Val loss
        for (int k = 0; k < EVAL_ITERS; k++) {
            if (use_pretokenized) get_batch_pretokenized(val_data);
            else get_batch_legacy(legacy_val_tokens);

            qwen3_forward(&model, h_inputs.data(), h_targets.data(),
                          use_pretokenized ? h_loss_mask.data() : NULL);
            HIP_CHECK(hipMemcpy(h_losses.data(), model.d_losses, B * T * sizeof(float), hipMemcpyDeviceToHost));

            for (int i = 0; i < B * T; i++) {
                if (!use_pretokenized || h_loss_mask[i]) {
                    val_loss += h_losses[i];
                    val_masked++;
                }
            }
        }

        float tl = train_masked > 0 ? train_loss / train_masked : 0.0f;
        float vl = val_masked > 0 ? val_loss / val_masked : 0.0f;
        return {tl, vl};
    };

    // ================================================================
    // 6. Training loop
    // ================================================================
    printf("\nStarting training for %d iterations...\n", MAX_ITERS);
    if (use_pretokenized) printf("  Mode: PRE-TOKENIZED with LOSS MASKING\n");
    else printf("  Mode: LEGACY (raw text, no loss masking)\n");

    for (int iter = start_iter; iter < MAX_ITERS; iter++) {

        if (iter % EVAL_INTERVAL == 0) {
            auto [train_loss, val_loss] = eval_loss();
            printf("Step %d: train_loss=%.4f val_loss=%.4f\n", iter, train_loss, val_loss);
            fflush(stdout);
            qwen3_save_checkpoint(&model, iter, "qwen3_hip_checkpoint.bin");
        }

        // Get batch
        if (use_pretokenized) get_batch_pretokenized(train_data);
        else get_batch_legacy(legacy_train_tokens);

        // Count masked positions for proper loss scaling
        int num_masked = 0;
        if (use_pretokenized) {
            for (int i = 0; i < B * T; i++) if (h_loss_mask[i]) num_masked++;
        } else {
            num_masked = B * T;
        }
        float loss_scale = num_masked > 0 ? 1.0f / num_masked : 0.0f;

        // Forward + Backward + Update
                qwen3_forward(&model, h_inputs.data(), h_targets.data(),
                      use_pretokenized ? h_loss_mask.data() : NULL);
        qwen3_zero_grad(&model);
        qwen3_backward(&model, loss_scale, use_pretokenized);
        qwen3_clip_gradients(&model, 1.0f);   // NEW: clip before update
        qwen3_update(&model, LR, iter + 1);

        if (iter % 10 == 0) {
            // Quick per-step loss for monitoring
            float step_loss = 0.0f;
            HIP_CHECK(hipMemcpy(h_losses.data(), model.d_losses, B * T * sizeof(float), hipMemcpyDeviceToHost));
            for (int i = 0; i < B * T; i++) {
                if (!use_pretokenized || h_loss_mask[i]) step_loss += h_losses[i];
            }
            step_loss = num_masked > 0 ? step_loss / num_masked : 0.0f;
            printf("Step %d: loss=%.4f (masked=%d/%d)\n", iter, step_loss, num_masked, B*T);
            fflush(stdout);
        }
    }

    qwen3_save_checkpoint(&model, MAX_ITERS, "qwen3_hip_checkpoint.bin");

    // ================================================================
    // 7. Chat-mode text generation
    // ================================================================
    printf("\n============================================================\n");
    printf("Chat Generation\n");
    printf("============================================================\n");

    // Load BPE tokenizer for encoding/decoding
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer);

    // Find special token IDs
    int im_end_id = -1, im_start_id = -1;
    for (int i = 0; i < num_vocab; i++) {
        if (vocab[i]) {
            if (strcmp(vocab[i], "<|im_end|>") == 0) im_end_id = i;
            if (strcmp(vocab[i], "<|im_start|>") == 0) im_start_id = i;
        }
    }
    printf("Special tokens: <|im_end|>=%d, <|im_start|>=%d\n", im_end_id, im_start_id);

    // Format prompt with Qwen3 chat template
    std::string user_prompt = "What is 2+2? Please explain.";
    std::string full_prompt = std::string("<|im_start|>system\n") +
                              SYSTEM_PROMPT +
                              "<|im_end|>\n" +
                              "<|im_start|>user\n" +
                              user_prompt +
                              "<|im_end|>\n" +
                              "<|im_start|>assistant\n";

    std::vector<int> gen_tokens = bpe_encode_text(full_prompt);
    printf("Prompt: %s\n", user_prompt.c_str());
    printf("Encoded to %zu tokens\n", gen_tokens.size());

    std::vector<float> h_probs(V);
    int max_gen_tokens = 200;

    for (int i = 0; i < max_gen_tokens; i++) {
        int context_len = std::min((int)gen_tokens.size(), T);
        int start_idx = std::max(0, (int)gen_tokens.size() - T);

        std::vector<int> h_gen_inputs(B * T, 0);
        for (int t = 0; t < context_len; t++) {
            h_gen_inputs[t] = gen_tokens[start_idx + t];
        }
        std::vector<int> h_dummy_targets(B * T, 0);
        std::vector<int> h_dummy_mask(B * T, 0);

        qwen3_forward(&model, h_gen_inputs.data(), h_dummy_targets.data(), h_dummy_mask.data());

        int sample_pos = context_len - 1;
        HIP_CHECK(hipMemcpy(h_probs.data(), model.d_probs + sample_pos * V, V * sizeof(float), hipMemcpyDeviceToHost));

        // Temperature + top-p sampling
        float temperature = 0.7f;
        float top_p = 0.9f;

        float max_prob = -1e10f;
        for (int j = 0; j < V; j++) if (h_probs[j] > max_prob) max_prob = h_probs[j];
        float sum = 0.0f;
        for (int j = 0; j < V; j++) {
            h_probs[j] = expf((h_probs[j] - max_prob) / temperature);
            sum += h_probs[j];
        }
        for (int j = 0; j < V; j++) h_probs[j] /= sum;

        // Top-p filtering
        std::vector<std::pair<float, int>> prob_idx(V);
        for (int j = 0; j < V; j++) prob_idx[j] = {h_probs[j], j};
        std::sort(prob_idx.begin(), prob_idx.end(), std::greater<>());

        float cumsum = 0.0f;
        int cutoff = V;
        for (int j = 0; j < V; j++) {
            cumsum += prob_idx[j].first;
            if (cumsum > top_p) { cutoff = j + 1; break; }
        }

        float top_sum = 0.0f;
        for (int j = 0; j < cutoff; j++) top_sum += prob_idx[j].first;

        float r = (float)rand() / RAND_MAX * top_sum;
        float cdf = 0.0f;
        int next_token = prob_idx[cutoff - 1].second;
        for (int j = 0; j < cutoff; j++) {
            cdf += prob_idx[j].first;
            if (r < cdf) { next_token = prob_idx[j].second; break; }
        }

        gen_tokens.push_back(next_token);

        // Stop conditions
        if (next_token == im_end_id) {
            printf("  [hit <|im_end|>]\n");
            break;
        }
        if (next_token == im_start_id) {
            printf("  [hit <|im_start|>]\n");
            break;
        }
    }

    // Decode and print
    std::string generated = decode_ids(gen_tokens);
    printf("\n--- Generated Text ---\n%s\n---\n", generated.c_str());

    // ================================================================
    // 8. Interactive chat loop (optional)
    // ================================================================
    printf("\n============================================================\n");
    printf("Interactive Chat (type 'quit' to exit)\n");
    printf("============================================================\n");

    std::vector<int> chat_history; // accumulates the whole conversation

    // Add system prompt to history
    std::string sys_text = std::string("<|im_start|>system\n") + SYSTEM_PROMPT + "<|im_end|>\n";
    auto sys_tokens = bpe_encode_text(sys_text);
    chat_history.insert(chat_history.end(), sys_tokens.begin(), sys_tokens.end());

    while (true) {
        printf("\nYou: ");
        char input_buf[1024];
        if (!fgets(input_buf, sizeof(input_buf), stdin)) break;
        std::string user_input(input_buf);
        while (!user_input.empty() && (user_input.back() == '\n' || user_input.back() == '\r'))
            user_input.pop_back();
        if (user_input == "quit" || user_input == "exit") break;
        if (user_input.empty()) continue;

        // Add user message to history
        std::string user_text = "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";
        auto user_toks = bpe_encode_text(user_text);
        chat_history.insert(chat_history.end(), user_toks.begin(), user_toks.end());

        // Generate response
        std::vector<int> response_tokens;
        for (int i = 0; i < 200; i++) {
            int context_len = std::min((int)chat_history.size() + (int)response_tokens.size(), T);
            int total_len = (int)chat_history.size() + (int)response_tokens.size();
            int start_idx = std::max(0, total_len - T);

            std::vector<int> all_toks = chat_history;
            all_toks.insert(all_toks.end(), response_tokens.begin(), response_tokens.end());

            std::vector<int> h_gen_inputs(B * T, 0);
            for (int t = 0; t < context_len; t++) {
                if (start_idx + t < (int)all_toks.size())
                    h_gen_inputs[t] = all_toks[start_idx + t];
            }
            std::vector<int> h_dummy_targets(B * T, 0);
            std::vector<int> h_dummy_mask(B * T, 0);

            qwen3_forward(&model, h_gen_inputs.data(), h_dummy_targets.data(), h_dummy_mask.data());

            int sample_pos = context_len - 1;
            HIP_CHECK(hipMemcpy(h_probs.data(), model.d_probs + sample_pos * V, V * sizeof(float), hipMemcpyDeviceToHost));

            // Temperature sampling
            float temperature = 0.7f;
            float max_prob = -1e10f;
            for (int j = 0; j < V; j++) if (h_probs[j] > max_prob) max_prob = h_probs[j];
            float sum = 0.0f;
            for (int j = 0; j < V; j++) { h_probs[j] = expf((h_probs[j] - max_prob) / temperature); sum += h_probs[j]; }
            for (int j = 0; j < V; j++) h_probs[j] /= sum;

            float r = (float)rand() / RAND_MAX;
            float cdf = 0.0f;
            int next_token = 0;
            for (int j = 0; j < V; j++) {
                cdf += h_probs[j];
                if (r < cdf) { next_token = j; break; }
            }

            response_tokens.push_back(next_token);

            if (next_token == im_end_id || next_token == im_start_id) break;
        }

        // Add assistant response to history
        chat_history.insert(chat_history.end(), response_tokens.begin(), response_tokens.end());

        // Decode and print
        std::string response = decode_ids(response_tokens);
        // Remove trailing <|im_end|> if present
        std::string end_tag = "<|im_end|>";
        if (response.size() >= end_tag.size() &&
            response.substr(response.size() - end_tag.size()) == end_tag) {
            response = response.substr(0, response.size() - end_tag.size());
        }
        printf("Assistant: %s\n", response.c_str());
    }

    // ================================================================
    // 9. Cleanup
    // ================================================================
    qwen3_free(&model);
    if (use_pretokenized) {
        free_pretokenized(&train_data);
        free_pretokenized(&val_data);
    }
    free_vocab();

    printf("\nDone!\n");
    return 0;
}