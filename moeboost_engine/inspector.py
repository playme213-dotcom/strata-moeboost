"""Bounded GGUF v2/v3 metadata and tensor-table reader."""
from __future__ import annotations

import os
import re
import struct
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

MAX_KV_ENTRIES = 2_000_000
MAX_TENSORS = 10_000_000
MAX_STRING_BYTES = 256 * 1024 * 1024
MAX_ARRAY_ITEMS = 100_000_000
GGUF_SCALARS = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
    5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8),
    11: ("q", 8), 12: ("d", 8),
}
GGUF_TYPE_NAMES = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32",
                   6: "f32", 7: "bool", 8: "string", 9: "array", 10: "u64", 11: "i64", 12: "f64"}

# (values per quantization block, bytes per block), indexed by ggml_type.
# Unknown/new types remain discoverable by offset but have no guessed exact size.
GGML_BLOCKS: dict[int, tuple[int, int, str]] = {
    0: (1, 4, "F32"), 1: (1, 2, "F16"), 2: (32, 18, "Q4_0"), 3: (32, 20, "Q4_1"),
    6: (32, 22, "Q5_0"), 7: (32, 24, "Q5_1"), 8: (32, 34, "Q8_0"), 9: (32, 40, "Q8_1"),
    10: (256, 84, "Q2_K"), 11: (256, 110, "Q3_K"), 12: (256, 144, "Q4_K"),
    13: (256, 176, "Q5_K"), 14: (256, 210, "Q6_K"), 15: (256, 292, "Q8_K"),
    16: (256, 66, "IQ2_XXS"), 17: (256, 74, "IQ2_XS"), 18: (256, 98, "IQ3_XXS"),
    19: (256, 50, "IQ1_S"), 20: (32, 18, "IQ4_NL"), 21: (256, 110, "IQ3_S"),
    22: (256, 82, "IQ2_S"), 23: (256, 136, "IQ4_XS"), 24: (1, 1, "I8"),
    25: (1, 2, "I16"), 26: (1, 4, "I32"), 27: (1, 8, "I64"), 28: (1, 8, "F64"),
    29: (256, 56, "IQ1_M"), 30: (1, 2, "BF16"), 34: (256, 54, "TQ1_0"),
    35: (256, 66, "TQ2_0"), 39: (32, 17, "MXFP4"), 40: (64, 36, "NVFP4"),
    41: (128, 18, "Q1_0"), 42: (64, 18, "Q2_0"), 43: (1, 1, "F8_E4M3"),
}
LAYER_RE = re.compile(r"(?:^|\.)blk\.(\d+)(?:\.|$)")
EXPERT_RE = re.compile(r"(?:^|\.)(?:ffn_(?:gate|up|down)_exps|expert|experts)(?:\.|_|$)", re.I)
EXPERT_ID_RE = re.compile(r"(?:expert|experts)[._](\d+)", re.I)


@dataclass
class TensorInfo:
    name: str
    dimensions: list[int]
    ggml_type: int
    quantization: str
    relative_offset: int
    absolute_offset: int
    byte_size: int | None
    size_exact: bool
    layer: int | None
    is_expert: bool
    expert_id: int | None = None


class GGUFReader:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).expanduser().resolve()
        self.metadata: dict[str, object] = {}
        self.tensors: list[TensorInfo] = []
        self.version = 0
        self.byte_order = "<"
        self.data_offset = 0
        self.alignment = 32
        self.file_size = 0

    def _read_exact(self, stream: BinaryIO, size: int) -> bytes:
        if size < 0 or size > self.file_size:
            raise ValueError(f"invalid GGUF read size {size}")
        data = stream.read(size)
        if len(data) != size:
            raise ValueError(f"truncated GGUF file at byte {stream.tell()}")
        return data

    def _unpack(self, stream: BinaryIO, fmt: str):
        return struct.unpack(self.byte_order + fmt, self._read_exact(stream, struct.calcsize(fmt)))[0]

    def _string(self, stream: BinaryIO) -> str:
        size = self._unpack(stream, "Q")
        if size > MAX_STRING_BYTES or size > self.file_size - stream.tell():
            raise ValueError(f"invalid GGUF string length {size}")
        try:
            return self._read_exact(stream, size).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("GGUF contains a non-UTF-8 metadata or tensor name") from exc

    def _scalar(self, stream: BinaryIO, kind: int):
        if kind == 8:
            return self._string(stream)
        if kind not in GGUF_SCALARS:
            raise ValueError(f"unsupported GGUF metadata type {kind}")
        fmt, _ = GGUF_SCALARS[kind]
        return self._unpack(stream, fmt)

    def _skip(self, stream: BinaryIO, count: int) -> None:
        if count < 0 or count > self.file_size - stream.tell():
            raise ValueError("GGUF metadata array exceeds file bounds")
        stream.seek(count, os.SEEK_CUR)

    def _value(self, stream: BinaryIO, kind: int):
        if kind != 9:
            return self._scalar(stream, kind)
        subtype = self._unpack(stream, "I")
        count = self._unpack(stream, "Q")
        if count > MAX_ARRAY_ITEMS:
            raise ValueError(f"GGUF metadata array is too large: {count}")
        if subtype == 8:
            for _ in range(count):
                self._string(stream)
            return {"array_type": "string", "count": count}
        if subtype == 9 or subtype not in GGUF_SCALARS:
            raise ValueError(f"invalid GGUF array element type {subtype}")
        fmt, width = GGUF_SCALARS[subtype]
        if count > (self.file_size - stream.tell()) // width:
            raise ValueError("GGUF metadata array length exceeds file bounds")
        # Retain scalar arrays only when small; token arrays are skipped without allocating RAM.
        if count <= 4096:
            return list(struct.unpack(self.byte_order + fmt * count, self._read_exact(stream, width * count)))
        self._skip(stream, width * count)
        return {"array_type": GGUF_TYPE_NAMES[subtype], "count": count}

    def inspect(self) -> dict:
        try:
            self.file_size = self.path.stat().st_size
            with self.path.open("rb") as stream:
                if self._read_exact(stream, 4) != b"GGUF":
                    raise ValueError("invalid GGUF magic; expected GGUF")
                raw_version = self._read_exact(stream, 4)
                little, big = struct.unpack("<I", raw_version)[0], struct.unpack(">I", raw_version)[0]
                if 2 <= little <= 3:
                    self.byte_order, self.version = "<", little
                elif big == 3:
                    self.byte_order, self.version = ">", big
                else:
                    raise ValueError(f"unsupported GGUF version (little={little}, big={big})")
                tensor_count = self._unpack(stream, "Q")
                kv_count = self._unpack(stream, "Q")
                if tensor_count > MAX_TENSORS or kv_count > MAX_KV_ENTRIES:
                    raise ValueError("GGUF header declares an unreasonable tensor or metadata count")
                for _ in range(kv_count):
                    key = self._string(stream)
                    kind = self._unpack(stream, "I")
                    value = self._value(stream, kind)
                    self.metadata[key] = value
                self.alignment = int(self.metadata.get("general.alignment", 32))
                if self.alignment < 1 or self.alignment > 1_048_576 or self.alignment & (self.alignment - 1):
                    raise ValueError(f"invalid GGUF alignment: {self.alignment}")
                raw_tensors = []
                for _ in range(tensor_count):
                    name = self._string(stream)
                    n_dims = self._unpack(stream, "I")
                    if n_dims > 8:
                        raise ValueError(f"tensor {name} has invalid dimension count {n_dims}")
                    dims = [self._unpack(stream, "Q") for _ in range(n_dims)]
                    if any(dim == 0 for dim in dims):
                        raise ValueError(f"tensor {name} has a zero-sized dimension")
                    ggml_type = self._unpack(stream, "I")
                    relative = self._unpack(stream, "Q")
                    raw_tensors.append((name, dims, ggml_type, relative))
                self.data_offset = (stream.tell() + self.alignment - 1) & ~(self.alignment - 1)
                is_split_header = bool(tensor_count == 0 and "split.count" in self.metadata)
                if not is_split_header and self.data_offset > self.file_size:
                    raise ValueError("GGUF tensor table extends beyond end of file")
                architecture = self.metadata.get("general.architecture", "unknown")
                architecture = architecture if isinstance(architecture, str) else "unknown"
                arch_prefix = architecture + "."
                expert_count = self.metadata.get(arch_prefix + "expert_count")
                active_experts = self.metadata.get(arch_prefix + "expert_used_count")
                if not isinstance(expert_count, int):
                    expert_count = None
                if not isinstance(active_experts, int):
                    active_experts = None
                names_indicate_moe = any(EXPERT_RE.search(row[0]) for row in raw_tensors)
                is_moe = bool((expert_count or 0) > 1 or names_indicate_moe)
                sorted_offsets = sorted({row[3] for row in raw_tensors})
                next_offsets = {value: sorted_offsets[index + 1] if index + 1 < len(sorted_offsets) else None
                                for index, value in enumerate(sorted_offsets)}
                for name, dims, kind, relative in raw_tensors:
                    block = GGML_BLOCKS.get(kind)
                    byte_size = None
                    exact = False
                    if block:
                        elements, bytes_per_block, _ = block
                        n_elements = 1
                        for dim in dims:
                            n_elements *= dim
                        if n_elements % elements == 0:
                            byte_size = (n_elements // elements) * bytes_per_block
                            exact = True
                    next_offset = next_offsets.get(relative)
                    if byte_size is None and next_offset is not None:
                        byte_size = next_offset - relative
                    absolute = self.data_offset + relative
                    if byte_size is None:
                        byte_size = max(0, self.file_size - absolute)
                    if absolute > self.file_size or (byte_size is not None and absolute + byte_size > self.file_size):
                        raise ValueError(f"tensor {name} data range exceeds the GGUF file")
                    layer_match, expert_match = LAYER_RE.search(name), EXPERT_ID_RE.search(name)
                    is_expert = bool(EXPERT_RE.search(name))
                    self.tensors.append(TensorInfo(name, dims, kind,
                        block[2] if block else f"GGML_TYPE_{kind}", relative, absolute,
                        byte_size, exact, int(layer_match.group(1)) if layer_match else None,
                        is_expert, int(expert_match.group(1)) if expert_match else None))
                layer_map: dict[str, list[dict]] = {}
                expert_tensor_count = 0
                for tensor in self.tensors:
                    if tensor.layer is None:
                        continue
                    row = asdict(tensor)
                    layer_map.setdefault(str(tensor.layer), []).append(row)
                    if tensor.is_expert:
                        expert_tensor_count += 1
            quant_bytes: dict[str, int] = defaultdict(int)
            for tensor in self.tensors:
                if tensor.byte_size is not None:
                    quant_bytes[tensor.quantization] += tensor.byte_size
            predominant_quant = max(quant_bytes, key=quant_bytes.get) if quant_bytes else None
            if is_moe and not expert_count:
                expert_count = max((t.dimensions[-1] for t in self.tensors if t.is_expert and t.dimensions),
                                   default=None)
            mtp_re = re.compile(r"(?:^|\.)(?:mtp|nextn|eh_proj|enorm|hnorm|shared_head)(?:\.|$)", re.IGNORECASE)
            mtp_tensor_count = sum(1 for t in self.tensors if mtp_re.search(t.name))
            mtp_layers = int(
                self.metadata.get(arch_prefix + "nextn_predict_layers")
                or self.metadata.get(arch_prefix + "num_nextn_predict_layers")
                or self.metadata.get(arch_prefix + "mtp_layers")
                or 0
            )
            native_mtp_heads = bool(mtp_layers > 0 or mtp_tensor_count > 0)
            return {
                "path": str(self.path), "file_size": self.file_size, "version": self.version,
                "architecture": architecture, "classification": "MoE" if is_moe else "Dense",
                "expert_count": expert_count, "active_experts": active_experts,
                "block_count": self.metadata.get(arch_prefix + "block_count"),
                "quantization": predominant_quant or self.metadata.get("general.file_type",
                                                                          self.metadata.get("general.quantization_version")),
                "file_type_code": self.metadata.get("general.file_type"),
                "alignment": self.alignment, "data_offset": self.data_offset,
                "tensor_count": len(self.tensors), "expert_tensor_count": expert_tensor_count,
                "mtp_support": True,
                "native_mtp_heads": native_mtp_heads,
                "mtp_layers": mtp_layers,
                "mtp_tensor_count": mtp_tensor_count,
                "layers": layer_map,
                "metadata": {k: v for k, v in self.metadata.items() if not isinstance(v, list)},
            }
        except OSError as exc:
            raise ValueError(f"cannot read GGUF file {self.path}: {exc}") from exc


def inspect_gguf(path: str | os.PathLike[str]) -> dict:
    return GGUFReader(path).inspect()
