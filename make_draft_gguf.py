"""Strata-MoEBoost Draft GGUF Generator by Gravitize.IGP.

When a model has no built-in MTP layers or speculative draft model,
this tool extracts the vocabulary, token embeddings, and first transformer block
to generate an ultra-fast draft.gguf model for speculative acceleration.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import gguf
from gguf import GGUFWriter, GGUFValueType

def create_draft_model(src_path: str, out_path: str, max_layers: int = 1):
    print(f"=== Creating Draft GGUF Model for Speculative Decoding ===")
    print(f"Source: {src_path}")
    print(f"Output: {out_path}")
    t0 = time.time()

    reader = gguf.GGUFReader(src_path)
    
    # Extract architecture
    arch_field = reader.get_field("general.architecture")
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8") if arch_field else "llama"
    print(f"[Draft Maker] Source Architecture: {arch}")

    writer = GGUFWriter(out_path, arch=arch)

    # Copy metadata
    essential_prefixes = ("general.", "tokenizer.")
    for field in reader.fields.values():
        name = field.name
        if not any(name.startswith(p) for p in essential_prefixes):
            continue
        if name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"):
            continue

        ftype = field.types[0]
        if ftype == GGUFValueType.STRING:
            val = bytes(field.parts[field.data[0]]).decode("utf-8", errors="ignore")
            if name == "general.name": val += "-draft"
            writer.add_string(name, val)
        elif ftype == GGUFValueType.UINT32:
            writer.add_uint32(name, int(field.parts[field.data[0]][0]))
        elif ftype == GGUFValueType.INT32:
            writer.add_int32(name, int(field.parts[field.data[0]][0]))
        elif ftype == GGUFValueType.FLOAT32:
            writer.add_float32(name, float(field.parts[field.data[0]][0]))
        elif ftype == GGUFValueType.BOOL:
            writer.add_bool(name, bool(field.parts[field.data[0]][0]))
        elif ftype == GGUFValueType.ARRAY:
            arr_type = field.types[1]
            if arr_type == GGUFValueType.STRING:
                str_list = [bytes(field.parts[i]).decode("utf-8", errors="ignore") for i in field.data]
                writer.add_array(name, str_list)
            elif arr_type in (GGUFValueType.INT32, GGUFValueType.UINT32):
                int_list = [int(field.parts[i][0]) for i in field.data]
                writer.add_array(name, int_list)

    # Copy all model architecture-specific parameters (e.g. layer_norm_rms_epsilon, rope.freq_base, etc.)
    for field in reader.fields.values():
        name = field.name
        if not name.startswith(f"{arch}."):
            continue
        if name in (f"{arch}.block_count", f"{arch}.expert_count", f"{arch}.expert_used_count"):
            continue

        ftype = field.types[0]
        if ftype == GGUFValueType.STRING:
            val = bytes(field.parts[field.data[0]]).decode("utf-8", errors="ignore")
            writer.add_string(name, val)
        elif ftype == GGUFValueType.UINT32:
            writer.add_uint32(name, int(field.parts[field.data[0]][0]))
        elif ftype == GGUFValueType.INT32:
            writer.add_int32(name, int(field.parts[field.data[0]][0]))
        elif ftype == GGUFValueType.FLOAT32:
            writer.add_float32(name, float(field.parts[field.data[0]][0]))
        elif ftype == GGUFValueType.BOOL:
            writer.add_bool(name, bool(field.parts[field.data[0]][0]))

    # Set draft model layer count
    writer.add_uint32(f"{arch}.block_count", max_layers)

    # Transfer draft tensors directly using t.data
    copied = 0
    for t in reader.tensors:
        name = t.name
        should_copy = (
            name in ("token_embd.weight", "output_norm.weight", "output.weight") or
            name.startswith("blk.0.attn_") or
            name in ("blk.0.attn_norm.weight", "blk.0.ffn_norm.weight")
        )
        # Skip MoE expert bodies for flyweight draft model
        if "exps" in name or "expert" in name:
            should_copy = False

        if should_copy:
            raw_bytes = bytes(t.data)
            writer.add_tensor(name, np.frombuffer(raw_bytes, dtype=np.uint8), raw_dtype=t.tensor_type)
            copied += 1
            print(f"  + Added draft tensor: {name} ({round(t.n_bytes / (1024*1024), 2)} MB)")

    print(f"[Draft Maker] Writing draft GGUF ({copied} tensors)...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    dt = time.time() - t0
    out_sz = Path(out_path).stat().st_size / (1024**2)
    print(f"=== Draft Model Created Successfully in {dt:.2f}s! ===")
    print(f"File: {out_path} ({out_sz:.2f} MB)")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a compact draft.gguf for speculative decoding")
    parser.add_argument("--model", type=str, default="/media/local-ai/Model/cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf")
    parser.add_argument("--out", type=str, default="/media/local-ai/Model/draft.gguf")
    parser.add_argument("--layers", type=int, default=1)
    args = parser.parse_args()

    create_draft_model(args.model, args.out, max_layers=args.layers)
