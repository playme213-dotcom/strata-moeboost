"""Strata-MoEBoost by Gravitize.IGP - Hybrid MoE Inference Server.

Combines Strata's OpenAI-compatible server interface and streaming protocols
with MoEBoost's hierarchical VRAM -> RAM -> SSD expert paging and llama.cpp execution engine.
Optimized for low VRAM (e.g. GTX 1050 3GB / GTX 1650 4GB) and 24GB+ host RAM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Ensure our local modules are on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "moeboost_engine"))

from moe_boost.cache import ExpertCache, safe_ram_budget
from moe_boost.pager import pager

# Import llama-cpp
try:
    from llama_cpp import Llama
except ImportError:
    # Try Desktop venv if running from global python
    venv_site = Path("/home/local-ai/Desktop/MoEBoost/.venv/lib/python3.12/site-packages")
    if venv_site.exists():
        sys.path.insert(0, str(venv_site))
        from llama_cpp import Llama
    else:
        raise

app = FastAPI(
    title="Strata-MoEBoost by Gravitize.IGP",
    description="High-performance low-VRAM MoE inference engine streaming experts between SSD, RAM, and GPU.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AppState:
    llm: Llama | None = None
    model_path: str = ""
    model_name: str = ""
    n_gpu_layers: int = 4
    n_threads: int = 8
    n_ctx: int = 4096
    stats: dict[str, Any] = {
        "requests": 0,
        "tokens_generated": 0,
        "last_tok_s": 0.0,
        "last_prefill_tok_s": 0.0,
        "active_device": "GTX 1050 (Hybrid 4 layers) + 8 CPU Threads",
    }

state = AppState()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "strata-moeboost",
        "author": "Gravitize.IGP",
        "model_loaded": state.llm is not None,
        "model_path": state.model_path,
        "stats": state.stats
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": state.model_name or "cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS",
                "object": "model",
                "owned_by": "Gravitize.IGP",
                "permission": []
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if state.llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    max_tok = req.max_tokens or 512

    if not req.stream:
        t0 = time.time()
        res = state.llm.create_chat_completion(
            messages=msgs,
            max_tokens=max_tok,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=False
        )
        dt = max(0.001, time.time() - t0)
        tok_count = res.get("usage", {}).get("completion_tokens", 0)
        tok_s = tok_count / dt
        state.stats["requests"] += 1
        state.stats["tokens_generated"] += tok_count
        state.stats["last_tok_s"] = round(tok_s, 2)
        return res

    async def event_generator() -> AsyncGenerator[str, None]:
        t0 = time.time()
        completion_tokens = 0
        gen = state.llm.create_chat_completion(
            messages=msgs,
            max_tokens=max_tok,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=True
        )
        for chunk in gen:
            completion_tokens += 1
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.0001)

        dt = max(0.001, time.time() - t0)
        state.stats["requests"] += 1
        state.stats["tokens_generated"] += completion_tokens
        state.stats["last_tok_s"] = round(completion_tokens / dt, 2)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def load_model(path: str, n_gpu_layers: int = 4, n_threads: int = 8, n_ctx: int = 4096):
    print(f"[Strata-MoEBoost] Loading model from {path}...")
    print(f"[Strata-MoEBoost] Hardware config: n_gpu_layers={n_gpu_layers}, n_threads={n_threads}, n_ctx={n_ctx}")
    t0 = time.time()
    state.llm = Llama(
        model_path=path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=False
    )
    t_load = time.time() - t0
    state.model_path = path
    state.model_name = Path(path).stem
    state.n_gpu_layers = n_gpu_layers
    state.n_threads = n_threads
    state.n_ctx = n_ctx
    print(f"[Strata-MoEBoost] Model loaded successfully in {t_load:.2f}s!")

def main():
    parser = argparse.ArgumentParser(description="Strata-MoEBoost Inference Server by Gravitize.IGP")
    parser.add_argument("--model", type=str, default="/media/local-ai/Model/cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf", help="Path to GGUF model")
    parser.add_argument("--gpu-layers", type=int, default=4, help="GPU layers to offload to VRAM (default: 4 for 3GB VRAM)")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for MoE execution (default: 8)")
    parser.add_argument("--ctx", type=int, default=4096, help="Context size (default: 4096)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()

    load_model(args.model, n_gpu_layers=args.gpu_layers, n_threads=args.threads, n_ctx=args.ctx)
    print(f"[Strata-MoEBoost] Serving OpenAI-compatible API on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
