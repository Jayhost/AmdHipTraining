import json
import os
import numpy as np
import tiktoken
from tqdm import tqdm

# Configuration
INPUT_FILE = "temppy.jsonl"       # Path to your JSONL file
TRAIN_FILE = "train.bin"
TRAIN_MASK_FILE = "train_mask.bin"
VAL_FILE = "val.bin"
VAL_MASK_FILE = "val_mask.bin"

T = 1024                             # Sequence length
VAL_SPLIT = 0.1                      # 10% of data for validation

# Initialize tiktoken with the cl100k_base BPE (used by GPT-4/ChatGPT)
enc = tiktoken.get_encoding("cl100k_base")

# ChatML Special Tokens
IM_START_ID = 100264
IM_END_ID = 100265

def encode_role_and_content(role, content):
    """Encodes a ChatML turn: <|im_start|>role\ncontent<|im_end|>\n"""
    tokens = [IM_START_ID]
    tokens += enc.encode(role + "\n")
    tokens += enc.encode(content)
    tokens += [IM_END_ID, enc.encode_single_token("\n")]
    return tokens

def process_conversation(conversations):
    """
    Processes a list of turns into token IDs and a target loss mask.
    """
    tokens = []
    is_assistant_token = [] # Tracks if the token at this index is part of assistant output

    # Add a default system prompt if missing
    if not conversations or conversations[0].get("from") != "system":
        sys_text = "You are a helpful assistant."
        sys_tokens = encode_role_and_content("system", sys_text)
        tokens.extend(sys_tokens)
        is_assistant_token.extend([False] * len(sys_tokens))

    for turn in conversations:
        role = turn.get("from", "").lower()
        content = turn.get("value", "")
        
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
            
        if role not in ["user", "assistant"]:
            continue

        turn_tokens = encode_role_and_content(role, content)
        tokens.extend(turn_tokens)
        
        # Mark if these tokens belong to the assistant
        is_assistant_token.extend([role == "assistant"] * len(turn_tokens))

    # Convert boolean list to float32 (1.0 for True, 0.0 for False)
    # This mask is perfectly aligned with the tokens array.
    loss_mask = [1.0 if is_assist else 0.0 for is_assist in is_assistant_token]

    return tokens, loss_mask

def pack_data(all_tokens, all_masks):
    """Packs token sequences and masks into T-length blocks."""
    packed_tokens = []
    packed_masks = []
    
    # Flatten all tokens and masks into one giant array
    flat_tokens = [tok for seq in all_tokens for tok in seq]
    flat_masks  = [m for seq in all_masks for m in seq]
    
    # Chop into T-sized chunks
    for i in range(0, len(flat_tokens) - T, T):
        chunk_tok = flat_tokens[i : i + T]
        chunk_msk = flat_masks[i : i + T]
        
        # If the last chunk is slightly shorter than T, pad it
        if len(chunk_tok) < T:
            pad_len = T - len(chunk_tok)
            chunk_tok += [0] * pad_len
            chunk_msk += [0.0] * pad_len
            
        packed_tokens.append(chunk_tok)
        packed_masks.append(chunk_msk)
        
    return np.array(packed_tokens, dtype=np.int32), np.array(packed_masks, dtype=np.float32)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    all_tokens = []
    all_masks = []
    
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Processing conversations"):
        try:
            data = json.loads(line)
            # The JSONL format you provided has a "conversations" field
            if "conversations" in data:
                conv = data["conversations"]
            else:
                # Fallback to input/output if conversations is missing
                conv = [
                    {"from": "human", "value": data.get("input", "")},
                    {"from": "gpt", "value": data.get("output", "")}
                ]
                
            toks, msks = process_conversation(conv)
            
            # Ensure sequence is at least T long to avoid packing errors
            if len(toks) > T:
                toks = toks[:T]
                msks = msks[:T]
            
            all_tokens.append(toks)
            all_masks.append(msks)
        except Exception as e:
            print(f"Skipping line due to error: {e}")

    print("Packing sequences to 1024 tokens...")
    tokens_arr, masks_arr = pack_data(all_tokens, all_masks)
    
    # Shuffle the packed sequences
    print("Shuffling data...")
    indices = np.arange(len(tokens_arr))
    np.random.shuffle(indices)
    tokens_arr = tokens_arr[indices]
    masks_arr = masks_arr[indices]
    
    # Train/Val Split
    split_idx = int(len(tokens_arr) * (1 - VAL_SPLIT))
    train_toks, val_toks = tokens_arr[:split_idx], tokens_arr[split_idx:]
    train_msks, val_msks = masks_arr[:split_idx], masks_arr[split_idx:]
    
    # Save to binary files
    # C++ can read these directly into memory. 
    # Shape: [num_sequences, T]
    print(f"Saving {TRAIN_FILE} and {TRAIN_MASK_FILE}...")
    train_toks.tofile(TRAIN_FILE)
    train_msks.tofile(TRAIN_MASK_FILE)
    
    print(f"Saving {VAL_FILE} and {VAL_MASK_FILE}...")
    val_toks.tofile(VAL_FILE)
    val_msks.tofile(VAL_MASK_FILE)
    
    print(f"Done! Train sequences: {len(train_toks)}, Val sequences: {len(val_toks)}")
    print(f"Vocab size: {enc.n_vocab}")

if __name__ == "__main__":
    main()