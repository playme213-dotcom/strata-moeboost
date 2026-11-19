"""Build per-layer expert byte ranges from GGUF tensor metadata."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from engine.inspector import GGML_BLOCKS


def model_profile_key(details: dict) -> str:
    """Stable model key across folder moves, with sampled content to distinguish revisions."""
    path = Path(details["path"])
    size = int(details.get("file_size") or path.stat().st_size)
    identity = {key: value for key, value in details.items()
                if key in {"architecture", "block_count", "expert_count", "active_experts",
                           "tensor_count", "quantization", "metadata"}}
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True, default=str,
                                     separators=(",", ":")).encode())
    digest.update(str(size).encode())
    # Sample small windows rather than reading a multi-gigabyte GGUF on each load.
    window = 64 * 1024
    with path.open("rb") as stream:
        for offset in sorted({0, max(0, size // 4), max(0, size // 2),
                              max(0, (size * 3) // 4), max(0, size - window)}):
            stream.seek(offset)
            digest.update(stream.read(window))
    return digest.hexdigest()[:12]


def build_expert_manifest(details: dict, model_id: str) -> dict[str, dict]:
    expert_count = details.get("expert_count")
    if not isinstance(expert_count, int) or expert_count < 1:
        return {}
    model_key = model_profile_key(details)
    result: dict[str, dict] = {}
    for layer, tensors in details.get("layers", {}).items():
        for tensor in tensors:
            if not tensor.get("is_expert") or not tensor.get("size_exact"):
                continue
            dims = [int(dim) for dim in tensor["dimensions"]]
            axes = [index for index, dim in enumerate(dims) if dim == expert_count]
            explicit_id = tensor.get("expert_id")
            block = GGML_BLOCKS.get(int(tensor["ggml_type"]))
            if not block:
                continue
            block_elements, block_bytes, _ = block
            tensor_size = int(tensor["byte_size"])
            tensor_path = details["path"]
            if axes:
                # GGML dimensions are ordered fastest to slowest. Prefer the last dimension
                # matching expert_count; this is the common packed expert-axis layout.
                axis = axes[-1]
                inner_elements = math.prod(dims[:axis]) if axis else 1
                if inner_elements % block_elements:
                    continue
                inner_bytes = inner_elements // block_elements * block_bytes
                outer_count = math.prod(dims[axis + 1:]) if axis + 1 < len(dims) else 1
                if inner_bytes * expert_count * outer_count != tensor_size:
                    continue
                for expert_index in range(expert_count):
                    entry = result.setdefault(f"{model_key}.layer{layer}.expert{expert_index}",
                                              {"path": tensor_path, "offset": 0, "size": 0, "segments": []})
                    for outer in range(outer_count):
                        offset = tensor["absolute_offset"] + (outer * expert_count + expert_index) * inner_bytes
                        entry["segments"].append({"path": tensor_path, "offset": offset, "size": inner_bytes})
                        entry["size"] += inner_bytes
            elif explicit_id is not None and 0 <= int(explicit_id) < expert_count:
                key = f"{model_key}.layer{layer}.expert{int(explicit_id)}"
                entry = result.setdefault(key, {"path": tensor_path, "offset": 0, "size": 0, "segments": []})
                entry["segments"].append({"path": tensor_path, "offset": tensor["absolute_offset"],
                                          "size": tensor_size})
                entry["size"] += tensor_size
    for entry in result.values():
        entry["offset"] = entry["segments"][0]["offset"]
    return result
