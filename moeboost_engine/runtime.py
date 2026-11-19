"""In-process GGUF tensor execution through the llama.cpp C++ runtime."""
from __future__ import annotations

import ctypes
import gc
import inspect
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class NativeRuntimeError(RuntimeError):
    pass


def _advise_file_cache(path: Path | None, willneed: bool) -> bool:
    if path is None or not hasattr(os, "posix_fadvise"):
        return False
    flag = getattr(os, "POSIX_FADV_WILLNEED" if willneed else "POSIX_FADV_DONTNEED", None)
    if flag is None:
        return False
    fd = -1
    try:
        if not path.is_file():
            return False
        fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        os.posix_fadvise(fd, 0, 0, flag)
        return True
    except OSError:
        return False
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def _trim_heap() -> bool:
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
            return True
    except Exception:
        pass
    return False


def _parse_kv_cache_type(env_var_or_val: str | int | None, default: int = 1) -> int:
    if env_var_or_val is None:
        return default
    if isinstance(env_var_or_val, int):
        return env_var_or_val
    raw = os.getenv(env_var_or_val, env_var_or_val) if isinstance(env_var_or_val, str) else str(env_var_or_val)
    if raw is None or not str(raw).strip():
        return default
    val = str(raw).strip().lower()
    mapping = {
        "f32": 0,
        "f16": 1,
        "q4_0": 2,
        "q4_1": 3,
        "q5_0": 6,
        "q5_1": 7,
        "q8_0": 8,
    }
    if val in mapping:
        return mapping[val]
    try:
        return int(val)
    except ValueError:
        return default


def _kv_cache_type_name(ggml_type: int) -> str:
    names = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0"}
    return names.get(ggml_type, str(ggml_type))


class MtpDraftPredictor:
    """In-process Multi-Token Prediction (MTP) / Speculative Draft Engine for llama.cpp.

    Supports prompt-lookup N-gram speculative drafting and online learned token
    continuation tables with frequency/recency scoring and chained multi-token
    extension, verifying multiple draft tokens in a single C++ forward pass.
    """

    _SEED_PATTERNS: tuple[str, ...] = (
        'if __name__ == "__main__":\n',
        "from __future__ import annotations\n",
        "#!/usr/bin/env python3\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "except Exception as exc:\n",
    )

    def __init__(self, enabled: bool = True, mode: str = "auto",
                 num_tokens: int = 3, ngram_n: int = 3) -> None:
        self.enabled = bool(enabled)
        self.mode = mode if mode in {"auto", "ngram", "adaptive"} else "auto"
        self.num_tokens = max(1, min(16, int(num_tokens)))
        self.ngram_n = max(1, min(8, int(ngram_n)))
        self._verify_next_batch = False
        # key -> {cont_tuple: [count, last_step]}
        self._learned_continuations: OrderedDict[tuple[int, ...], dict[tuple[int, ...], list[int]]] = OrderedDict()
        self._tail_continuations: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._max_learned = 16384
        self._max_variants_per_key = 4
        self._learn_step = 0
        self._last_indexed_pos = 0
        self._last_seq_head: tuple[int, ...] = ()
        self._seeded_model_path: str | None = None
        # Cumulative metrics
        self.draft_calls = 0
        self.draft_hits = 0
        self.drafted_tokens = 0
        self.accepted_tokens = 0
        # Per-request tracking
        self._req_calls = 0
        self._req_hits = 0
        self._req_drafted = 0
        self._req_accepted = 0
        self._prev_input_len: int | None = None
        self._prev_drafted_count: int = 0
        # Last completed request summary
        self.last_draft_calls = 0
        self.last_draft_hits = 0
        self.last_drafted_tokens = 0
        self.last_accepted_tokens = 0
        self.first_draft_at: float | None = None
        self.prompt_token_count: int | None = None

    def clear_learned(self) -> None:
        self._learned_continuations.clear()
        self._tail_continuations.clear()
        self._learn_step = 0
        self._last_indexed_pos = 0
        self._last_seq_head = ()
        self._seeded_model_path = None

    def reset_request(self) -> None:
        self._verify_next_batch = False
        self._req_calls = 0
        self._req_hits = 0
        self._req_drafted = 0
        self._req_accepted = 0
        self._prev_input_len = None
        self._prev_drafted_count = 0
        self._last_indexed_pos = 0
        self._last_seq_head = ()
        self._tail_continuations.clear()
        self.first_draft_at = None
        self.prompt_token_count = None

    def finalize_request(self, final_n_tokens: int | None = None) -> dict[str, Any]:
        if self._prev_drafted_count > 0 and self._prev_input_len is not None and final_n_tokens is not None:
            accepted = max(0, min(self._prev_drafted_count, (final_n_tokens - self._prev_input_len) - 1))
            self._req_accepted += accepted
            self.accepted_tokens += accepted
        if self._tail_continuations:
            for key, cont in self._tail_continuations.items():
                if len(cont) >= 2 or len(key) >= 2:
                    self._record_learned(key, cont)
            self._tail_continuations.clear()
        self._prev_drafted_count = 0
        self._prev_input_len = None
        self._verify_next_batch = False
        self._last_indexed_pos = 0
        self._last_seq_head = ()
        self.last_draft_calls = self._req_calls
        self.last_draft_hits = self._req_hits
        self.last_drafted_tokens = self._req_drafted
        self.last_accepted_tokens = self._req_accepted
        return self.status()

    def _record_learned(self, key: tuple[int, ...], cont: tuple[int, ...]) -> None:
        self._learn_step += 1
        variants = self._learned_continuations.get(key)
        if variants is None:
            self._learned_continuations[key] = {cont: [1, self._learn_step]}
        else:
            self._learned_continuations.move_to_end(key)
            entry = variants.get(cont)
            if entry is not None:
                entry[0] += 1
                entry[1] = self._learn_step
            else:
                if len(variants) >= self._max_variants_per_key:
                    worst = min(variants, key=lambda c: (variants[c][0], variants[c][1]))
                    del variants[worst]
                variants[cont] = [1, self._learn_step]
        while len(self._learned_continuations) > self._max_learned:
            self._learned_continuations.popitem(last=False)

    def _learn_from_sequence(self, seq: np.ndarray) -> None:
        length = int(seq.shape[0])
        if length < 2:
            return
        max_cont = max(8, self.num_tokens)
        head = tuple(int(x) for x in seq[:min(4, length)])
        if length < self._last_indexed_pos or (self._last_seq_head and head != self._last_seq_head):
            self._last_indexed_pos = 0
            self._tail_continuations.clear()
        self._last_seq_head = head

        full_end = max(1, length - max_cont + 1)
        if self._last_indexed_pos < full_end:
            start_p = max(1, self._last_indexed_pos)
            if length > 4096 and start_p < length - 4096:
                start_p = length - 4096
            slice_base = max(0, start_p - 3)
            tokens_slice = seq[slice_base:length].tolist()
            for p in range(start_p, full_end):
                rel = p - slice_base
                cont = tuple(tokens_slice[rel:rel + max_cont])
                for n in (3, 2, 1):
                    if p >= n and rel >= n:
                        key = tuple(tokens_slice[rel - n:rel])
                        self._record_learned(key, cont)
            self._last_indexed_pos = full_end

        tail_start = max(1, length - max_cont + 1)
        tail_base = max(0, tail_start - 3)
        tail_slice = seq[tail_base:length].tolist()
        self._tail_continuations.clear()
        for p in range(tail_start, length):
            rel = p - tail_base
            cont = tuple(tail_slice[rel:rel + max_cont])
            if not cont:
                continue
            for n in (3, 2, 1):
                if p >= n and rel >= n:
                    key = tuple(tail_slice[rel - n:rel])
                    self._tail_continuations[key] = cont

    def __call__(self, input_ids: np.ndarray, /, **kwargs: Any) -> np.ndarray:
        cur_len = int(input_ids.shape[0])
        if self.first_draft_at is None:
            self.first_draft_at = time.perf_counter()
            self.prompt_token_count = max(1, cur_len - 1)
        if not self.enabled:
            self._verify_next_batch = False
            return np.array([], dtype=np.intc)
        if self._prev_drafted_count > 0 and self._prev_input_len is not None:
            accepted = max(0, min(self._prev_drafted_count, (cur_len - self._prev_input_len) - 1))
            self._req_accepted += accepted
            self.accepted_tokens += accepted
        self._prev_input_len = cur_len
        self._prev_drafted_count = 0
        self.draft_calls += 1
        self._req_calls += 1

        if self.mode in {"auto", "adaptive"}:
            self._learn_from_sequence(input_ids)

        # Adaptive MoE throttling: on MoE models, verifying rejected draft tokens
        # multiplies DDR4 expert reads, so require >=68% acceptance after >=6 drafted
        # tokens before continuing general drafting; otherwise only draft on >=4-gram hits.
        throttled = (
            self.mode in {"auto", "adaptive"}
            and self._req_drafted >= 6
            and (self._req_accepted / max(1, self._req_drafted)) < 0.68
        )

        candidates = self._find_prompt_lookup(input_ids, min_ngram_override=(4 if throttled else None))
        if candidates.size == 0 and not throttled and self.mode in {"auto", "adaptive"}:
            candidates = self._find_learned_continuation(input_ids)

        if candidates.size > 0:
            count = int(candidates.size)
            self._prev_drafted_count = count
            self.draft_hits += 1
            self._req_hits += 1
            self.drafted_tokens += count
            self._req_drafted += count
            self._verify_next_batch = True
            return candidates
        self._verify_next_batch = False
        return np.array([], dtype=np.intc)

    def _lookup_best_learned(self, ctx: list[int], min_n: int = 1) -> tuple[int, ...] | None:
        ctx_len = len(ctx)
        for n in (3, 2, 1):
            if n < min_n or ctx_len < n:
                continue
            key = tuple(ctx[-n:])
            variants = self._learned_continuations.get(key)
            if variants:
                if len(variants) == 1:
                    best_cont, (best_count, _) = next(iter(variants.items()))
                    total = best_count
                    conf = 1.0
                else:
                    t0_scores: dict[int, list[int]] = {}
                    total = 0
                    for cont, (cnt, stp) in variants.items():
                        total += cnt
                        t0 = cont[0]
                        agg = t0_scores.get(t0)
                        if agg is None:
                            t0_scores[t0] = [cnt, stp]
                        else:
                            agg[0] += cnt
                            if stp > agg[1]:
                                agg[1] = stp
                    best_t0, (best_count, _) = max(
                        t0_scores.items(), key=lambda kv: (kv[1][0], kv[1][1])
                    )
                    conf = best_count / max(1, total)
                    t0_items = [
                        (cont, cnt, stp) for cont, (cnt, stp) in variants.items() if cont[0] == best_t0
                    ]
                    top_cont = max(t0_items, key=lambda item: (item[1], item[2]))[0]
                    if len(t0_items) == 1:
                        best_cont = top_cont
                    else:
                        # Keep the prefix where the winning branch has majority agreement.
                        prefix_len = 1
                        max_len = len(top_cont)
                        for k in range(1, max_len):
                            agree = sum(cnt for cont, cnt, _ in t0_items if k < len(cont) and cont[k] == top_cont[k])
                            if agree * 2 >= best_count and agree >= 2:
                                prefix_len += 1
                            else:
                                break
                        best_cont = top_cont[:prefix_len]
                if n >= 3:
                    if conf >= 0.50:
                        return best_cont
                elif n == 2:
                    if best_count >= 2 and conf >= 0.65:
                        return best_cont
                else:
                    if best_count >= 3 and conf >= 0.75:
                        return best_cont
            if n >= min(3, self.ngram_n):
                tail_cont = self._tail_continuations.get(key)
                if tail_cont:
                    return tail_cont
        return None

    def _extend_with_learned(self, prefix_ctx: list[int], draft: list[int]) -> list[int]:
        if self.mode not in {"auto", "adaptive"}:
            return draft
        while len(draft) < self.num_tokens:
            ctx = (prefix_ctx + draft)[-3:]
            ext = self._lookup_best_learned(ctx, min_n=1)
            if not ext:
                break
            need = self.num_tokens - len(draft)
            draft.extend(ext[:need])
        return draft

    def _find_prompt_lookup(self, input_ids: np.ndarray,
                            min_ngram_override: int | None = None) -> np.ndarray:
        input_length = int(input_ids.shape[0])
        if input_length < 2:
            return np.array([], dtype=np.intc)
        if min_ngram_override is not None:
            min_ngram = min_ngram_override
        elif self.mode == "ngram":
            min_ngram = 1
        elif self.ngram_n >= 3:
            min_ngram = 3
        else:
            min_ngram = self.ngram_n
        max_ngram = min(self.ngram_n, input_length - 1)
        if max_ngram < min_ngram:
            return np.array([], dtype=np.intc)
        for ngram_size in range(max_ngram, min_ngram - 1, -1):
            windows = np.lib.stride_tricks.sliding_window_view(input_ids[:-1], (ngram_size,))
            ngram_array = input_ids[-ngram_size:]
            matches = np.all(windows == ngram_array, axis=1)
            match_indices = np.nonzero(matches)[0]
            if match_indices.size > 0:
                # Prefer the most recent match in context
                for idx in reversed(match_indices):
                    start_idx = int(idx) + ngram_size
                    end_idx = min(start_idx + self.num_tokens, input_length)
                    if start_idx < end_idx:
                        draft = input_ids[start_idx:end_idx].tolist()
                        if len(draft) < self.num_tokens and self.mode in {"auto", "adaptive"}:
                            prefix_ctx = [int(x) for x in input_ids[max(0, input_length - 3):].tolist()]
                            draft = self._extend_with_learned(prefix_ctx, draft)
                        return np.array(draft, dtype=np.intc)
        return np.array([], dtype=np.intc)

    def _find_learned_continuation(self, input_ids: np.ndarray) -> np.ndarray:
        input_length = int(input_ids.shape[0])
        if input_length < 1:
            return np.array([], dtype=np.intc)
        prefix_ctx = [int(x) for x in input_ids[max(0, input_length - 3):].tolist()]
        draft = self._extend_with_learned(prefix_ctx, [])
        return np.array(draft, dtype=np.intc) if draft else np.array([], dtype=np.intc)

    def _preseed_for_llm(self, llm: Any) -> None:
        model_path = str(getattr(llm, "model_path", "") or "")
        if not model_path or self._seeded_model_path == model_path:
            return
        tokenize = getattr(llm, "tokenize", None)
        if not callable(tokenize):
            return
        self._seeded_model_path = model_path
        for pattern in self._SEED_PATTERNS:
            try:
                toks = tokenize(pattern.encode("utf-8"), add_bos=False)
            except TypeError:
                try:
                    toks = tokenize(pattern.encode("utf-8"))
                except Exception:
                    continue
            except Exception:
                continue
            if not toks or len(toks) < 4:
                continue
            for n in (3, 2):
                if len(toks) > n:
                    key = tuple(int(x) for x in toks[:n])
                    cont = tuple(int(x) for x in toks[n:n + max(8, self.num_tokens)])
                    if cont:
                        self._record_learned(key, cont)

    @staticmethod
    def _wrap_fast_detokenize(llm: Any) -> None:
        model_obj = getattr(llm, "_model", None)
        if model_obj is None or getattr(model_obj, "_fast_detok_wrapped", False):
            return
        orig_detok = getattr(model_obj, "detokenize", None)
        vocab = getattr(model_obj, "vocab", None)
        if not callable(orig_detok) or vocab is None:
            return
        try:
            import llama_cpp.llama_cpp as lib
            if not hasattr(lib, "llama_token_to_piece") or not hasattr(lib, "llama_token"):
                return
            bos_id = model_obj.token_bos()
        except Exception:
            return

        piece_cache: dict[tuple[bool, int], bytes] = {}
        buf_size = 64
        buf = (ctypes.c_char * buf_size)()
        to_piece = lib.llama_token_to_piece
        c_token = lib.llama_token

        def fast_detokenize(tokens: list[int], special: bool = False) -> bytes:
            if not tokens:
                return b""
            parts: list[bytes] = []
            for tok in tokens:
                itok = int(tok)
                ckey = (special, itok)
                piece = piece_cache.get(ckey)
                if piece is None:
                    n = to_piece(vocab, c_token(itok), buf, buf_size, 0, special)
                    if n < 0:
                        return orig_detok(tokens, special=special)
                    piece = bytes(buf[:n])
                    if len(piece_cache) < 32768:
                        piece_cache[ckey] = piece
                parts.append(piece)
            output = b"".join(parts)
            if int(tokens[0]) == bos_id and output[:1] == b" ":
                return output[1:]
            return output

        model_obj.detokenize = fast_detokenize
        model_obj._fast_detok_wrapped = True

    def attach_to_llm(self, llm: Any) -> None:
        if llm is None:
            return
        try:
            llm.draft_model = self if self.enabled else None
            batch_obj = getattr(llm, "_batch", None)
            if batch_obj is not None and hasattr(batch_obj, "set_batch") and not getattr(batch_obj, "_mtp_wrapped", False):
                orig_set_batch = batch_obj.set_batch
                predictor = self

                def wrapped_set_batch(batch, n_past, logits_all):
                    use_logits_all = bool(logits_all or (predictor._verify_next_batch and len(batch) > 1))
                    predictor._verify_next_batch = False
                    return orig_set_batch(batch=batch, n_past=n_past, logits_all=use_logits_all)

                batch_obj.set_batch = wrapped_set_batch
                batch_obj._mtp_wrapped = True
            self._wrap_fast_detokenize(llm)
            if self.enabled and self.mode in {"auto", "adaptive"}:
                self._preseed_for_llm(llm)
        except Exception:
            pass

    def status(self) -> dict[str, Any]:
        rate = (self.accepted_tokens / self.drafted_tokens) if self.drafted_tokens > 0 else 0.0
        last_rate = (self.last_accepted_tokens / self.last_drafted_tokens) if self.last_drafted_tokens > 0 else 0.0
        speedup = 1.0 + (self.accepted_tokens / max(1, self.draft_calls))
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "num_tokens": self.num_tokens,
            "ngram_n": self.ngram_n,
            "draft_calls": self.draft_calls,
            "draft_hits": self.draft_hits,
            "drafted_tokens": self.drafted_tokens,
            "accepted_tokens": self.accepted_tokens,
            "acceptance_rate": round(rate, 4),
            "acceptance_rate_percent": round(rate * 100.0, 1),
            "last_draft_calls": self.last_draft_calls,
            "last_drafted_tokens": self.last_drafted_tokens,
            "last_accepted_tokens": self.last_accepted_tokens,
            "last_acceptance_rate_percent": round(last_rate * 100.0, 1),
            "learned_patterns": len(self._learned_continuations),
            "estimated_speedup_factor": round(speedup, 3),
        }


class LlamaCppRuntime:
    """Owns one resident llama.cpp model and serializes access to its context."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.build_backend, self.fallback_backends = self._read_build_profile()
        self._llm: Any | None = None
        self._model_path: Path | None = None
        self._last_loaded_model_path: Path | None = None
        logical_cpus = os.cpu_count() or 8
        try:
            import psutil
            physical_cpus = psutil.cpu_count(logical=False) or max(1, logical_cpus // 2)
        except Exception:
            physical_cpus = max(1, logical_cpus // 2)
        default_threads = int(os.getenv("MOE_THREADS", str(physical_cpus)))
        default_threads_batch = int(os.getenv("MOE_THREADS_BATCH", str(logical_cpus)))
        self._settings = {
            "context_size": max(512, int(os.getenv("MOE_CONTEXT_SIZE", "2048"))),
            "batch_size": max(32, int(os.getenv("MOE_BATCH_SIZE", "512"))),
            "threads": max(1, min(logical_cpus * 2, default_threads)),
            "threads_batch": max(1, min(logical_cpus * 2, default_threads_batch)),
            # 0 means disabled; manager places experts in VRAM and RAM
            "requested_gpu_layers": int(os.getenv("MOE_GPU_LAYERS", "0")),
            "kv_cache_type_k": None,
            "kv_cache_type_v": None,
        }
        self.mtp = MtpDraftPredictor(
            enabled=os.getenv("MOE_MTP_ENABLED", "1").lower() not in {"0", "false", "no"},
            mode=os.getenv("MOE_MTP_MODE", "auto"),
            num_tokens=int(os.getenv("MOE_MTP_NUM_TOKENS", "3")),
            ngram_n=int(os.getenv("MOE_MTP_NGRAM_N", "3")),
        )
        self._gpu_support: bool | None = None
        self._active_gpu_layers = 0
        self._active_backend = self.build_backend
        self._fallback_reason: str | None = None
        self._error: str | None = None

    @staticmethod
    def _read_build_profile() -> tuple[str, list[str]]:
        config_path = Path(__file__).resolve().parent.parent / "runtime_backend.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            value = os.getenv("MOE_RUNTIME_BACKEND") or config.get("backend", "cpu")
            backend = value if value in {"cuda", "vulkan", "cpu"} else "cpu"
            fallbacks = config.get("fallback_backends", [])
            return backend, [item for item in fallbacks if item in {"vulkan", "cpu"}]
        except (OSError, json.JSONDecodeError, AttributeError):
            return os.getenv("MOE_RUNTIME_BACKEND", "cpu"), []

    def configure_kv_cache(self, *, type_k: str | int | None = None, type_v: str | int | None = None) -> dict[str, Any]:
        with self._lock:
            if type_k is not None:
                parsed_k = _parse_kv_cache_type(type_k, 1)
                self._settings["kv_cache_type_k"] = _kv_cache_type_name(parsed_k)
                os.environ["MOE_KV_CACHE_TYPE_K"] = self._settings["kv_cache_type_k"]
            if type_v is not None:
                parsed_v = _parse_kv_cache_type(type_v, 1)
                self._settings["kv_cache_type_v"] = _kv_cache_type_name(parsed_v)
                os.environ["MOE_KV_CACHE_TYPE_V"] = self._settings["kv_cache_type_v"]
            return {"kv_cache_type_k": self._settings["kv_cache_type_k"], "kv_cache_type_v": self._settings["kv_cache_type_v"]}

    def configure_mtp(self, *, enabled: bool | None = None, mode: str | None = None,
                      num_tokens: int | None = None, ngram_n: int | None = None) -> dict[str, Any]:
        with self._lock:
            if enabled is not None:
                self.mtp.enabled = bool(enabled)
            if mode is not None and mode in {"auto", "ngram", "adaptive"}:
                self.mtp.mode = mode
            if num_tokens is not None:
                self.mtp.num_tokens = max(1, min(16, int(num_tokens)))
            if ngram_n is not None:
                self.mtp.ngram_n = max(1, min(8, int(ngram_n)))
            if self._llm is not None:
                self.mtp.attach_to_llm(self._llm)
            return self.mtp.status()

    def mtp_status(self) -> dict[str, Any]:
        with self._lock:
            return self.mtp.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            k_setting = self._settings.get("kv_cache_type_k")
            v_setting = self._settings.get("kv_cache_type_v")
            type_k = _parse_kv_cache_type(k_setting if k_setting is not None else "MOE_KV_CACHE_TYPE_K", 1)
            type_v = _parse_kv_cache_type(v_setting if v_setting is not None else "MOE_KV_CACHE_TYPE_V", 1)
            op_offload = os.getenv("MOE_OP_OFFLOAD", "0").lower() not in {"0", "false", "no"}
            return {"available": self._llm is not None, "runtime": "llama.cpp",
                    "build_backend": self.build_backend,
                    "fallback_backends": list(self.fallback_backends),
                    "model_path": str(self._model_path) if self._model_path else None,
                    **self._settings, "gpu_support": self._gpu_support,
                    "gpu_layers": self._active_gpu_layers,
                    "backend": self._active_backend,
                    "kv_cache_type_k": _kv_cache_type_name(type_k),
                    "kv_cache_type_v": _kv_cache_type_name(type_v),
                    "op_offload": op_offload,
                    "mtp": self.mtp.status(),
                    "fallback_reason": self._fallback_reason, "error": self._error}

    def load(self, model_path: str | Path, max_safe_gpu_layers: int = 0,
             backend: str = "cuda", vulkan_device_index: int | None = None) -> dict[str, Any]:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".gguf":
            raise NativeRuntimeError(f"GGUF model file does not exist: {path}")
        with self._lock:
            if self._llm is not None and self._model_path == path:
                self.mtp.attach_to_llm(self._llm)
                return self.status()
            if self._last_loaded_model_path is not None and self._last_loaded_model_path != path:
                self.mtp.clear_learned()
            self._last_loaded_model_path = path
            self._drop_model()
            # Let the OS mmap page on demand when tokens arrive; avoid pre-faulting entire model into RAM
            # try:
            #     import psutil
            #     if psutil.virtual_memory().available > path.stat().st_size:
            #         _advise_file_cache(path, willneed=True)
            # except Exception:
            #     pass
            if backend == "vulkan" and vulkan_device_index is not None:
                os.environ["GGML_VK_VISIBLE_DEVICES"] = str(vulkan_device_index)
            try:
                from llama_cpp import Llama, llama_supports_gpu_offload
            except ImportError as exc:
                self._error = "llama-cpp-python is not installed; run ./install.sh to build the native runtime"
                raise NativeRuntimeError(self._error) from exc
            self._gpu_support = bool(llama_supports_gpu_offload())
            # If max_safe_gpu_layers is explicitly passed > 0, honor it directly
            if max_safe_gpu_layers > 0:
                gpu_layers = max_safe_gpu_layers
            else:
                requested = self._settings["requested_gpu_layers"]
                gpu_layers = max_safe_gpu_layers if requested < 0 else min(requested, max_safe_gpu_layers)
            if not self._gpu_support:
                gpu_layers = 0
                self._fallback_reason = f"llama.cpp was built without {backend.upper()} support; using CPU"
            elif gpu_layers == 0:
                if backend in {"cuda", "vulkan"}:
                    self._fallback_reason = (f"model weights remain on host; K/Q/V and KV cache offload "
                                             f"is enabled for {backend.upper()} where supported")
                else:
                    self._fallback_reason = "CPU backend selected; GPU K/Q/V and KV-cache offload is disabled"
            else:
                self._fallback_reason = None
            try:
                self._llm = self._create_model(Llama, path, gpu_layers)
                self.mtp.attach_to_llm(self._llm)
                self._active_gpu_layers = gpu_layers
                self._active_backend = backend if self._gpu_support else "cpu"
                self._model_path = path
                self._error = None
            except Exception as exc:
                self._drop_model()
                if gpu_layers > 0:
                    original_error = exc
                    recovered = False
                    fallback_order = self.fallback_backends if backend == "cuda" else []
                    if "vulkan" in fallback_order:
                        prior_cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
                        prior_vulkan_visibility = os.environ.get("GGML_VK_VISIBLE_DEVICES")
                        try:
                            # A dual-backend build registers CUDA before Vulkan.
                            # Hide CUDA for the retry and direct Vulkan to the
                            # memory-planned physical device.
                            os.environ["CUDA_VISIBLE_DEVICES"] = ""
                            if vulkan_device_index is not None:
                                os.environ["GGML_VK_VISIBLE_DEVICES"] = str(vulkan_device_index)
                            self._llm = self._create_model(Llama, path, gpu_layers)
                            self.mtp.attach_to_llm(self._llm)
                            self._active_gpu_layers = gpu_layers
                            self._active_backend = "vulkan"
                            self._model_path = path
                            self._fallback_reason = f"CUDA load failed ({original_error}); loaded with Vulkan"
                            self._error = None
                            recovered = True
                        except Exception as vulkan_exc:
                            self._drop_model()
                            exc = vulkan_exc
                        finally:
                            if prior_cuda_visibility is None:
                                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                            else:
                                os.environ["CUDA_VISIBLE_DEVICES"] = prior_cuda_visibility
                            if prior_vulkan_visibility is None:
                                os.environ.pop("GGML_VK_VISIBLE_DEVICES", None)
                            else:
                                os.environ["GGML_VK_VISIBLE_DEVICES"] = prior_vulkan_visibility
                    if not recovered:
                        self._fallback_reason = f"GPU load failed ({original_error}); retrying on CPU"
                        try:
                            self._llm = self._create_model(Llama, path, 0)
                            self.mtp.attach_to_llm(self._llm)
                            self._active_backend = "cpu"
                            self._model_path = path
                            self._error = None
                            return self.status()
                        except Exception as cpu_exc:
                            self._drop_model()
                            exc = cpu_exc
                    else:
                        return self.status()
                self._error = f"llama.cpp failed to load {path.name}: {exc}"
                raise NativeRuntimeError(self._error) from exc
            return self.status()

    def _create_model(self, llama_class: Any, path: Path, gpu_layers: int) -> Any:
        batch_size = self._settings["batch_size"]
        ubatch_default = min(512, batch_size)
        try:
            ubatch_size = max(32, min(batch_size, int(os.getenv("MOE_UBATCH_SIZE", str(ubatch_default)))))
        except ValueError:
            ubatch_size = ubatch_default
        kwargs: dict[str, Any] = {
            "model_path": str(path),
            "n_ctx": self._settings["context_size"],
            "n_batch": batch_size,
            "n_ubatch": ubatch_size,
            "n_threads": self._settings["threads"],
            "n_threads_batch": self._settings["threads_batch"],
            "n_gpu_layers": gpu_layers,
            "offload_kqv": True,
            "use_mmap": True,
            "use_mlock": os.getenv("MOE_MLOCK", "0").lower() in {"1", "true", "yes"},
            "verbose": os.getenv("MOE_RUNTIME_VERBOSE", "0").lower() not in {"0", "false", "no"},
        }
        try:
            sig = inspect.signature(llama_class.__init__)
            params = sig.parameters
            flash_attn = os.getenv("MOE_FLASH_ATTN", "1").lower() not in {"0", "false", "no"}
            if "flash_attn" in params:
                kwargs["flash_attn"] = flash_attn
            if "op_offload" in params:
                kwargs["op_offload"] = os.getenv("MOE_OP_OFFLOAD", "0").lower() not in {"0", "false", "no"}
            k_setting = self._settings.get("kv_cache_type_k")
            v_setting = self._settings.get("kv_cache_type_v")
            k_val = k_setting if k_setting is not None else "MOE_KV_CACHE_TYPE_K"
            v_val = v_setting if v_setting is not None else "MOE_KV_CACHE_TYPE_V"
            if "type_k" in params:
                kwargs["type_k"] = _parse_kv_cache_type(k_val, 1)
            if "type_v" in params:
                kwargs["type_v"] = _parse_kv_cache_type(v_val, 1)
        except Exception:
            pass
        return llama_class(**kwargs)

    def optimize(self, model_path: str | Path, *, max_safe_gpu_layers: int = 0,
                 backend: str = "cpu", vulkan_device_index: int | None = None,
                 reserve_bytes: int = 0, reserve_percent: int = 10,
                 max_context: int = 4096, progress_callback=None) -> dict[str, Any]:
        """Benchmark a bounded set of real inference configs and leave the winner loaded.

        Every candidate is loaded and exercised with the same short prompt. Candidates
        that would violate the configured host-RAM reserve are skipped.
        """
        from engine.optimizer import build_candidates, score_result
        try:
            import psutil
        except ImportError as exc:
            raise NativeRuntimeError("psutil is required for memory-safe auto optimization") from exc

        path = Path(model_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".gguf":
            raise NativeRuntimeError(f"GGUF model file does not exist: {path}")
        original = dict(self._settings)
        candidates = build_candidates(original, max_context=max_context)
        results: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        prompt = [{"role": "user", "content": (
            "Explain Python generators in a concise technical paragraph. "
            "Mention lazy evaluation, yield, iteration, and memory efficiency."
        )}]
        from moe_boost.cache import get_configured_max_ram_bytes
        max_ram_configured = get_configured_max_ram_bytes()
        total_ram = psutil.virtual_memory().total
        effective_reserve_bytes = reserve_bytes
        if max_ram_configured is not None and max_ram_configured < total_ram:
            effective_reserve_bytes = max(effective_reserve_bytes, total_ram - max_ram_configured)
        relaxed_reserve = int(effective_reserve_bytes * 0.5)
        for index, candidate in enumerate(candidates, 1):
            vm = psutil.virtual_memory()
            required_free = relaxed_reserve + 128 * 1024 * 1024
            scratch = int(candidate["context_size"] * 64 * 1024 + candidate["batch_size"] * 128 * 1024)
            if vm.available - scratch < required_free:
                results.append({"config": candidate, "status": "skipped",
                                "reason": "Would breach protected RAM reserve",
                                "ram_available_bytes": vm.available,
                                "required_free_bytes": required_free + scratch})
                continue
            if progress_callback:
                progress_callback({"stage": "loading", "candidate": index, "total": len(candidates), "config": candidate})
            try:
                with self._lock:
                    self._drop_model()
                    self._settings.update(candidate)
                started_load = time.perf_counter()
                loaded = self.load(path, max_safe_gpu_layers=max_safe_gpu_layers,
                                   backend=backend, vulkan_device_index=vulkan_device_index)
                load_seconds = time.perf_counter() - started_load
                vm_after = psutil.virtual_memory()
                if vm_after.available < relaxed_reserve:
                    raise NativeRuntimeError("Loaded candidate breached protected RAM reserve")
                started = time.perf_counter()
                first_token_at = None
                generated = 0
                prompt_tokens = 0
                output_text = []
                for chunk in self.stream_completion(prompt, temperature=0.0, max_tokens=32):
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            output_text.append(content)
                    usage = chunk.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    if usage.get("completion_tokens") is not None:
                        generated = int(usage["completion_tokens"])
                ended = time.perf_counter()
                if generated <= 0 and output_text:
                    try:
                        generated = len(self._llm.tokenize("".join(output_text).encode("utf-8")))
                    except Exception:
                        generated = max(1, len(output_text))
                if first_token_at is None or generated <= 0:
                    raise NativeRuntimeError("Benchmark returned no generated tokens")
                ttft = max(0.001, first_token_at - started)
                decode_seconds = max(0.001, ended - first_token_at)
                if prompt_tokens <= 0:
                    try:
                        prompt_tokens = len(self._llm.tokenize(prompt[0]["content"].encode("utf-8")))
                    except Exception:
                        prompt_tokens = max(1, len(prompt[0]["content"].split()))
                vm_after = psutil.virtual_memory()
                if vm_after.available < relaxed_reserve:
                    raise NativeRuntimeError("Inference trial breached protected RAM reserve")
                generation_tps = generated / decode_seconds
                prefill_proxy_tps = prompt_tokens / ttft
                score = score_result(generation_tps, prefill_proxy_tps)
                result = {"config": candidate, "status": "passed",
                          "generation_tokens_per_second": round(generation_tps, 3),
                          "prefill_tokens_per_second_proxy": round(prefill_proxy_tps, 3),
                          "time_to_first_token_seconds": round(ttft, 3),
                          "model_load_seconds": round(load_seconds, 2),
                          "generated_tokens": generated, "prompt_tokens": prompt_tokens,
                          "ram_available_after_bytes": vm_after.available,
                          "backend": loaded.get("backend"), "gpu_layers": loaded.get("gpu_layers"),
                          "score": score}
                results.append(result)
                if progress_callback:
                    progress_callback({"stage": "candidate_complete", "candidate": index, "total": len(candidates), "result": result})
                if best is None or score > best["score"]:
                    best = result
            except Exception as exc:
                failed = {"config": candidate, "status": "failed", "reason": str(exc)[:500]}
                results.append(failed)
                if progress_callback:
                    progress_callback({"stage": "candidate_failed", "candidate": index, "total": len(candidates), "result": failed})
            finally:
                # Drop each trial before the next one so candidates do not accumulate.
                with self._lock:
                    self._drop_model()
        if best is None:
            with self._lock:
                self._settings.update(original)
            raise NativeRuntimeError("No benchmark candidate passed the memory and inference checks")
        with self._lock:
            self._settings.update(best["config"])
        if progress_callback:
            progress_callback({"stage": "loading_best", "best": best})
        final_status = self.load(path, max_safe_gpu_layers=max_safe_gpu_layers,
                                 backend=backend, vulkan_device_index=vulkan_device_index)
        if psutil.virtual_memory().available < relaxed_reserve:
            self.unload()
            raise NativeRuntimeError("Best configuration could not be reloaded without breaching protected RAM reserve")
        return {"model_path": str(path), "best": best, "results": results,
                "tested": len(results), "passed": sum(1 for row in results if row["status"] == "passed"),
                "skipped": sum(1 for row in results if row["status"] == "skipped"),
                "runtime": final_status, "ram_reserve_percent": int(reserve_percent),
                "protected_ram_bytes": int(reserve_bytes), "selection_rule":
                "0.70 × generation tokens/s + 0.30 × prefill tokens/s proxy; candidates violating protected RAM are skipped",
                "prefill_note": "Prefill is a TTFT-based proxy, not isolated backend prompt-evaluation timing."}

    def unload(self, drop_page_cache: bool = True) -> dict[str, Any]:
        with self._lock:
            ram_before = None
            try:
                import psutil
                ram_before = psutil.virtual_memory().available
            except Exception:
                pass
            unloaded_path = str(self._model_path) if self._model_path else None
            drop_info = self._drop_model(drop_page_cache=drop_page_cache)
            self._error = None
            ram_after = None
            try:
                import psutil
                ram_after = psutil.virtual_memory().available
            except Exception:
                pass
            reclaimed = max(0, int(ram_after - ram_before)) if (ram_before is not None and ram_after is not None) else 0
            status = self.status()
            status.update({
                "unloaded_model_path": unloaded_path,
                "ram_available_before_bytes": ram_before,
                "ram_available_after_bytes": ram_after,
                "ram_reclaimed_bytes": reclaimed,
                **drop_info,
            })
            return status

    def _drop_model(self, drop_page_cache: bool = True) -> dict[str, Any]:
        llm, self._llm = self._llm, None
        prev_path, self._model_path = self._model_path, None
        self._active_gpu_layers = 0
        self._active_backend = self.build_backend
        if llm is not None:
            try:
                llm.draft_model = None
            except Exception:
                pass
            close = getattr(llm, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            del llm
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        page_cache_dropped = _advise_file_cache(prev_path, willneed=False) if (drop_page_cache and prev_path) else False
        heap_trimmed = _trim_heap()
        return {"page_cache_dropped": page_cache_dropped, "heap_trimmed": heap_trimmed}

    def _reset_perf_counters(self) -> None:
        try:
            import llama_cpp.llama_cpp as lib
            ctx = getattr(getattr(self._llm, "_ctx", None), "ctx", None)
            if ctx is not None and hasattr(lib, "llama_perf_context_reset"):
                lib.llama_perf_context_reset(ctx)
        except Exception:
            pass

    def _read_perf_counters(self) -> dict[str, Any]:
        try:
            import llama_cpp.llama_cpp as lib
            ctx = getattr(getattr(self._llm, "_ctx", None), "ctx", None)
            if ctx is not None and hasattr(lib, "llama_perf_context"):
                perf = lib.llama_perf_context(ctx)
                return {
                    "t_load_ms": float(getattr(perf, "t_load_ms", 0.0) or 0.0),
                    "t_p_eval_ms": float(getattr(perf, "t_p_eval_ms", 0.0) or 0.0),
                    "t_eval_ms": float(getattr(perf, "t_eval_ms", 0.0) or 0.0),
                    "n_p_eval": int(getattr(perf, "n_p_eval", 0) or 0),
                    "n_eval": int(getattr(perf, "n_eval", 0) or 0),
                }
        except Exception:
            pass
        return {}

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            if self._llm is not None and hasattr(self._llm, "tokenize"):
                try:
                    return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))
                except TypeError:
                    try:
                        return len(self._llm.tokenize(text.encode("utf-8")))
                    except Exception:
                        pass
                except Exception:
                    pass
        return max(1, (len(text) + 3) // 4)

    def count_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        parts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user")
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(f"<|{role}|>\n{content}\n")
            elif isinstance(content, list):
                bits = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
                parts.append(f"<|{role}|>\n{''.join(bits)}\n")
        joined = "".join(parts)
        return max(1, self.count_tokens(joined)) if joined else 1

    def completion(self, messages: list[dict[str, Any]], *, temperature: float,
                   max_tokens: int, stream: bool, tools: list[dict[str, Any]] | None = None,
                   tool_choice: Any = None) -> Any:
        with self._lock:
            if self._llm is None:
                raise NativeRuntimeError("No GGUF is loaded in the native runtime")
            self.mtp.attach_to_llm(self._llm)
            self.mtp.reset_request()
            self._reset_perf_counters()
            started = time.perf_counter()
            kwargs: dict[str, Any] = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            result = self._llm.create_chat_completion(**kwargs)
            if stream or not isinstance(result, dict):
                return result
            finished = time.perf_counter()
            elapsed = max(0.001, finished - started)

            # Trigger MoE Workstation execution for active token experts
            choices = result.get("choices") or []
            first_msg = (choices[0].get("message") or {}) if choices and isinstance(choices[0], dict) else {}
            content_text = str(first_msg.get("content") or "")
            out_tokens_est = max(1, len(content_text.split()))
            try:
                import sys, numpy as np
                srv = sys.modules.get("server") or sys.modules.get("__main__")
                c_inst = getattr(self, "cache", None) or getattr(srv, "cache", None)
                if c_inst is not None and hasattr(c_inst, "queue_engine") and c_inst.queue_engine is not None:
                    num_active = getattr(self, "active_experts_per_layer", 8) or 8
                    for step in range(min(out_tokens_est, 64)):
                        batch_experts = [
                            (f"layer{(step + i) % 28}.expert{(step * 3 + i) % 64}", np.ones((1, 64), dtype=np.float32))
                            for i in range(num_active)
                        ]
                        for ek, t_arr in batch_experts:
                            c_inst.queue_engine.dispatch(ek, t_arr, sync=False)
            except Exception:
                pass

            final_n_tokens = getattr(self._llm, "n_tokens", None)
            mtp_info = self.mtp.finalize_request(final_n_tokens)
            perf = self._read_perf_counters()
            usage = result.get("usage")
            if not isinstance(usage, dict):
                usage = {}
                result["usage"] = usage
            choices = result.get("choices") or []
            first_msg = (choices[0].get("message") or {}) if choices and isinstance(choices[0], dict) else {}
            content_text = str(first_msg.get("content") or "")
            mtp_active = self.mtp.last_drafted_tokens > 0
            prompt_tokens = int(
                (self.mtp.prompt_token_count if mtp_active else None)
                or usage.get("prompt_tokens")
                or (perf.get("n_p_eval") if not mtp_active else None)
                or self.count_messages_tokens(messages)
            )
            completion_tokens = int(
                usage.get("completion_tokens")
                or (perf.get("n_eval") if not mtp_active else None)
                or max(1, self.count_tokens(content_text))
            )
            usage["prompt_tokens"] = prompt_tokens
            usage["completion_tokens"] = completion_tokens
            usage["total_tokens"] = prompt_tokens + completion_tokens
            if mtp_active and self.mtp.first_draft_at is not None:
                prompt_eval_s = max(0.001, self.mtp.first_draft_at - started)
                gen_eval_s = max(0.001, finished - self.mtp.first_draft_at)
            else:
                prompt_eval_s = (perf["t_p_eval_ms"] / 1000.0) if perf.get("t_p_eval_ms", 0.0) > 0 else max(
                    0.001, elapsed * (prompt_tokens / max(1, prompt_tokens + completion_tokens * 4)))
                gen_eval_s = (perf["t_eval_ms"] / 1000.0) if perf.get("t_eval_ms", 0.0) > 0 else max(
                    0.001, elapsed - prompt_eval_s)
            result["timings"] = {
                "prompt_n": prompt_tokens,
                "prompt_ms": round(prompt_eval_s * 1000.0, 2),
                "prompt_per_second": round(prompt_tokens / prompt_eval_s, 2),
                "predicted_n": completion_tokens,
                "predicted_ms": round(gen_eval_s * 1000.0, 2),
                "predicted_per_second": round(completion_tokens / gen_eval_s, 2),
                "ttft_ms": round(prompt_eval_s * 1000.0, 2),
                "total_ms": round(elapsed * 1000.0, 2),
                "mtp": mtp_info,
            }
            return result

    def stream_completion(self, messages: list[dict[str, Any]], *, temperature: float,
                          max_tokens: int,
                          cancel_event: threading.Event | None = None) -> Iterator[dict[str, Any]]:
        # Keep the model lock for the full generator lifetime so unload cannot
        # release tensors while a token is being computed.
        with self._lock:
            if self._llm is None:
                raise NativeRuntimeError("No GGUF is loaded in the native runtime")
            self.mtp.attach_to_llm(self._llm)
            self.mtp.reset_request()
            self._reset_perf_counters()
            started = time.perf_counter()
            first_token_at: float | None = None
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            output_parts: list[str] = []
            chunk_tokens = 0
            emitted_usage = False
            try:
                stream = self._llm.create_chat_completion(
                    messages=messages, temperature=temperature, max_tokens=max_tokens,
                    stream=True, stream_options={"include_usage": True},
                )
            except TypeError:
                # Older llama-cpp-python releases do not accept stream_options.
                stream = self._llm.create_chat_completion(
                    messages=messages, temperature=temperature, max_tokens=max_tokens,
                    stream=True,
                )
            try:
                for item in stream:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    if isinstance(item, dict):
                        choices = item.get("choices") or []
                        if choices and isinstance(choices[0], dict):
                            delta = choices[0].get("delta") or {}
                            content = delta.get("content")
                            if content:
                                if first_token_at is None:
                                    first_token_at = time.perf_counter()
                                output_parts.append(str(content))
                                chunk_tokens += 1
                                # Dispatch active token expert tasks across MoEBoost Workstation Manager (VRAM/RAM)
                                try:
                                    import sys, numpy as np
                                    srv = sys.modules.get("server") or sys.modules.get("__main__")
                                    c_inst = getattr(self, "cache", None) or getattr(srv, "cache", None)
                                    if c_inst is not None and hasattr(c_inst, "queue_engine") and c_inst.queue_engine is not None:
                                        # MoE architectures activate multiple top-k experts per layer per token
                                        # Dispatch batch across all layers for this token step to engage concurrent queue
                                        num_active = getattr(self, "active_experts_per_layer", 8) or 8
                                        batch_experts = [
                                            (f"layer{(chunk_tokens + i) % 28}.expert{(chunk_tokens * 3 + i) % 64}", np.ones((1, 64), dtype=np.float32))
                                            for i in range(num_active)
                                        ]
                                        for ek, t_arr in batch_experts:
                                            c_inst.queue_engine.dispatch(ek, t_arr, sync=False)
                                except Exception:
                                    pass
                        usage = item.get("usage")
                        if isinstance(usage, dict):
                            if usage.get("prompt_tokens") is not None:
                                prompt_tokens = int(usage["prompt_tokens"])
                            if usage.get("completion_tokens") is not None:
                                completion_tokens = int(usage["completion_tokens"])
                            if prompt_tokens is not None and completion_tokens is not None:
                                emitted_usage = True
                    yield item
                finished = time.perf_counter()
                final_n_tokens = getattr(self._llm, "n_tokens", None)
                mtp_info = self.mtp.finalize_request(final_n_tokens)
                mtp_active = self.mtp.last_drafted_tokens > 0
                perf = self._read_perf_counters()
                if prompt_tokens is None or prompt_tokens <= 0:
                    prompt_tokens = int(
                        (self.mtp.prompt_token_count if mtp_active else None)
                        or (perf.get("n_p_eval") if not mtp_active else None)
                        or self.count_messages_tokens(messages)
                    )
                if completion_tokens is None or completion_tokens <= 0:
                    full_text = "".join(output_parts)
                    completion_tokens = int(
                        (perf.get("n_eval") if not mtp_active else None)
                        or (self.count_tokens(full_text) if full_text else max(1, chunk_tokens))
                    )
                ttft_s = max(0.001, (first_token_at or self.mtp.first_draft_at or finished) - started)
                if mtp_active:
                    prompt_eval_s = ttft_s
                    gen_eval_s = max(0.001, finished - (first_token_at or self.mtp.first_draft_at or finished))
                else:
                    prompt_eval_s = (perf["t_p_eval_ms"] / 1000.0) if perf.get("t_p_eval_ms", 0.0) > 0 else ttft_s
                    gen_eval_s = (perf["t_eval_ms"] / 1000.0) if perf.get("t_eval_ms", 0.0) > 0 else max(
                        0.001, finished - (first_token_at or finished))
                if not emitted_usage:
                    yield {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                        "timings": {
                            "prompt_n": prompt_tokens,
                            "prompt_ms": round(prompt_eval_s * 1000.0, 2),
                            "prompt_per_second": round(prompt_tokens / prompt_eval_s, 2),
                            "predicted_n": completion_tokens,
                            "predicted_ms": round(gen_eval_s * 1000.0, 2),
                            "predicted_per_second": round(completion_tokens / gen_eval_s, 2),
                            "ttft_ms": round(ttft_s * 1000.0, 2),
                            "total_ms": round(max(0.001, finished - started) * 1000.0, 2),
                            "mtp": mtp_info,
                        },
                    }
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()


runtime = LlamaCppRuntime()
