"""Warm-start and Markov-prefetch policy for the expert RAM cache."""
from __future__ import annotations

import asyncio
import math
import os
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from moe_boost.cache import ExpertCache


@dataclass
class PagedExpert:
    """A model-backed expert too large for RAM; compute adapters consume its chunks."""
    cache: ExpertCache
    expert_id: str
    size_bytes: int

    async def iter_chunks(self, chunk_bytes: int = 64 * 1024 * 1024):
        async for offset, array in self.cache.stream_expert(self.expert_id, chunk_bytes):
            yield offset, array


class AdaptiveLearner:
    def __init__(self, cache: ExpertCache, warm_fraction: float = 0.0,
                 max_warm_experts: int = 0, prefetch_width: int = 2):
        self.cache = cache
        self.warm_fraction = min(1.0, max(0.0, float(os.getenv("MOE_WARM_FRACTION", str(warm_fraction)))))
        self.max_warm_experts = max(0, int(os.getenv("MOE_MAX_WARM_EXPERTS", str(max_warm_experts))))
        self.prefetch_width = max(0, prefetch_width)
        self._previous: dict[str, str] = {}
        self._active: set[tuple[str, str]] = set()
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    async def warm_boot(self, expert_ids: set[str] | None = None) -> list[str]:
        """Pre-stage historical hot experts, bounded to a fraction of safe RAM."""
        if not self.cache.manifest or self.max_warm_experts == 0:
            return []
        target_bytes = int(self.cache.budget_bytes * self.warm_fraction)
        staged, used = [], 0
        scope = expert_ids if expert_ids is not None else set(self.cache.manifest)
        ranked = sorted(
            ((key, max(self.cache.hits.get(key, 0), self.cache.invocations.get(key, 0)))
             for key in scope
             if key in self.cache.manifest
             and (self.cache.hits.get(key, 0) > 0 or self.cache.invocations.get(key, 0) > 0)),
            key=lambda row: row[1],
            reverse=True,
        )
        if ranked and self.warm_fraction < 0.95:
            top_count = math.ceil(len(ranked) * self.warm_fraction)
            candidates = [expert_id for expert_id, _ in ranked[:top_count]]
        else:
            # Full model RAM staging: prioritize observed hot experts, followed by all remaining un-staged experts
            ranked_keys = [k for k, _ in ranked]
            remaining_keys = sorted(key for key in scope if key in self.cache.manifest and key not in set(ranked_keys))
            candidates = ranked_keys + remaining_keys
        candidates = candidates[:self.max_warm_experts]
        candidates.sort(key=lambda k: (str(self.cache.manifest[k].get("path", "")),
                                       int(self.cache.manifest[k].get("offset", 0) or 0)))
        for expert_id in candidates:
            if len(staged) >= self.max_warm_experts:
                break
            try:
                _, _, size = self.cache._entry_spec(expert_id)
            except (ValueError, OSError):
                continue
            if size <= 0 or used + size > target_bytes:
                continue
            try:
                if await self.cache.prefetch(expert_id):
                    staged.append(expert_id)
                    used += size
            except (OSError, ValueError, MemoryError):
                continue
        return staged

    def predict_next(self, expert_id: str) -> list[str]:
        transitions = self.cache.transitions.get(str(expert_id), {})
        return [target for target, _ in sorted(transitions.items(), key=lambda row: row[1], reverse=True)
                if target in self.cache.manifest][:self.prefetch_width]

    async def observe(self, session_id: str, expert_ids: list[str]) -> None:
        ids = [str(value) for value in expert_ids]
        async with self._lock:
            previous = self._previous.get(session_id)
            if previous and ids and previous != ids[0]:
                await self.cache.record_transition(previous, ids[0])
            for left, right in zip(ids, ids[1:]):
                if left != right:
                    await self.cache.record_transition(left, right)
            if ids:
                self._previous[session_id] = ids[-1]
        predicted = {target for source in ids for target in self.predict_next(source)} - set(ids)
        for target in predicted:
            task = asyncio.create_task(self.cache.prefetch(target))
            self._tasks.add(task)
            task.add_done_callback(self._prefetch_done)

    def _prefetch_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()  # consume any I/O error; a failed prediction must not affect compute

    @asynccontextmanager
    async def acquire_for_compute(self, session_id: str, expert_ids: list[str],
                                  route_sequence: list[str] | None = None) -> AsyncIterator[dict[str, object]]:
        """Hold cache reference locks for all unique active experts until compute exits."""
        unique = list(dict.fromkeys(str(value) for value in expert_ids))
        await self.observe(session_id, route_sequence or unique)
        leases: dict[str, object] = {}
        budget = (await self.cache.snapshot())["ram_budget_bytes"]
        resident = set(self.cache.entries)
        ranked = sorted(unique, key=lambda key: (key in resident, self.cache.hits.get(key, 0)), reverse=True)
        ram_ids: set[str] = set()
        reserved = 0
        for expert_id in ranked:
            _, _, size = self.cache._entry_spec(expert_id)
            if size <= budget - reserved:
                ram_ids.add(expert_id)
                reserved += size
        async with AsyncExitStack() as stack:
            for expert_id in sorted(unique):
                _, _, size = self.cache._entry_spec(expert_id)
                if expert_id not in ram_ids:
                    await stack.enter_async_context(self.cache.pin_stream(expert_id))
                    leases[expert_id] = PagedExpert(self.cache, expert_id, size)
                else:
                    try:
                        leases[expert_id] = await stack.enter_async_context(self.cache.acquire(expert_id))
                    except MemoryError:
                        await stack.enter_async_context(self.cache.pin_stream(expert_id))
                        leases[expert_id] = PagedExpert(self.cache, expert_id, size)
                self._active.add((session_id, expert_id))
            try:
                yield leases
            finally:
                for expert_id in unique:
                    self._active.discard((session_id, expert_id))

    def active_snapshot(self) -> list[dict[str, str]]:
        result = []
        vram_entries = set(self.cache.vram_cache.entries)
        for session_id, expert_id in sorted(self._active):
            resident = expert_id in self.cache.entries
            tier = "VRAM" if expert_id in vram_entries else ("RAM" if resident else "SSD")
            result.append({"session_id": session_id, "expert_id": expert_id,
                           "residency": tier})
        return result

    async def close(self) -> None:
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)


class ActiveExpertHandler(AdaptiveLearner):
    """Reference-locked adapter for router IDs; backends call this around tensor compute."""

    @staticmethod
    def batch_tokens(token_routes: list[list[str]]) -> dict[str, list[int]]:
        batches: dict[str, list[int]] = {}
        for token_index, expert_ids in enumerate(token_routes):
            for expert_id in dict.fromkeys(map(str, expert_ids)):
                batches.setdefault(expert_id, []).append(token_index)
        return batches
