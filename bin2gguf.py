import struct
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

# --- Config (Must match your C++ hyperparameters) ---
V = 32
C = 64
NH = 4
NKV = 2
DK = 16
DFF = 256
L = 4
BIN_FILE = "qwen3_hip_checkpoint.bin"
OUT_FILE = "qwen3_custom_f16.gguf"

def calc_num_parameters(V, C, NH, NKV, DK, DFF, L):
    return (V * C) + L * (
        C + NH * DK * C + NKV * DK * C + NKV * DK * C + 
        DK + DK + C * C + C + 
        DFF * C + DFF * C + C * DFF
    ) + C + V * C

num_params = calc_num_parameters(V, C, NH, NKV, DK, DFF, L)
print(f"Expected parameters: {num_params}")

# --- Read C++ Binary ---
with open(BIN_FILE, "rb") as f:
    # Read step
    step = struct.unpack('i', f.read(4))[0]
    print(f"Checkpoint step: {step}")
    
    # Read weights
    params_bytes = f.read(num_params * 4)
    params = np.frombuffer(params_bytes, dtype=np.float32)
    
    # We ignore the Adam m and v buffers, we only need the weights!

# --- Map to Standard GGUF Tensors ---
writer = GGUFWriter(OUT_FILE, "qwen2") # Use qwen2 architecture base

# Global Metadata
writer.add_context_length(32)
writer.add_embedding_length(C)
writer.add_block_count(L)
writer.add_feed_forward_length(DFF)
writer.add_attention_head_count(NH)
writer.add_attention_head_count_kv(NKV)
writer.add_layer_norm_rms_eps(1e-6)
writer.add_rope_freq_base(10000.0) # Adjust if you change your RoPE
writer.add_file_type(GGMLQuantizationType.F16)

offset = 0

def extract_tensor(name, shape):
    global offset
    size = 1
    for dim in shape: size *= dim
    
    tensor_data = params[offset : offset + size].copy()
    offset += size
    
    # GGUF expects row-major, out-features first. Our C++ matmuls are (OC, C), which matches.
    tensor_data = tensor_data.reshape(shape)
    
    # Convert to float16 to save space (llama.cpp will quantize this later)
    tensor_data = tensor_data.astype(np.float16)
    writer.add_tensor(name, tensor_data)
    print(f"Added {name} | Shape: {shape} | Offset: {offset-size}")

# 1. Token Embedding
extract_tensor("token_embd.weight", (V, C))

# 2. Layers
for l in range(L):
    prefix = f"blk.{l}"
    
    extract_tensor(f"{prefix}.attn_norm.weight", (C,))
    extract_tensor(f"{prefix}.attn_q.weight", (NH * DK, C))
    extract_tensor(f"{prefix}.attn_k.weight", (NKV * DK, C))
    extract_tensor(f"{prefix}.attn_v.weight", (NKV * DK, C))
    
    # QK-Norm (Supported in latest llama.cpp Qwen2/3 implementations)
    extract_tensor(f"{prefix}.attn_q_norm.weight", (DK,))
    extract_tensor(f"{prefix}.attn_k_norm.weight", (DK,))
    
    extract_tensor(f"{prefix}.attn_output.weight", (C, C))
    extract_tensor(f"{prefix}.ffn_norm.weight", (C,))
    
    extract_tensor(f"{prefix}.ffn_gate.weight", (DFF, C)) # gate_proj
    extract_tensor(f"{prefix}.ffn_up.weight", (DFF, C))   # up_proj
    extract_tensor(f"{prefix}.ffn_down.weight", (C, DFF))  # down_proj

# 3. Final Norm and Output
extract_tensor("output_norm.weight", (C,))
extract_tensor("output.weight", (V, C))

if offset != num_params:
    print(f"WARNING: Mismatch! Used {offset} params, expected {num_params}")

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

print(f"\nSuccessfully wrote {OUT_FILE}")