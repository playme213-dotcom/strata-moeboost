"""MoEBoost Workshop Architecture: Workshop Manager, People Busyness Catalog & Heterogeneous Workstations.

Key Principles:
1. THE WORKSHOP MANAGER (`WorkshopManager`):
   - The central dispatcher that dictates WHERE each expert must work.
   - Constantly inspects the People Busyness Catalog, Tool Workstation slot capacity,
     and queue line lengths.
   - Instructs the expert: "Go to the Tool Workstation (VRAM)" or "Go to the Hand-Labor Workstation (RAM)".
2. SSD IS PURE STORAGE ONLY (NOT A WORKSTATION):
   - The SSD is an archival storage shelf. It does zero computation.
   - When the Manager assigns an expert stored on SSD, the Manager orders its weights
     brought into RAM first before work starts.
3. The Two Computing Workstations:
   - High-Tech Tool Workstation (VRAM / GPU): Specialized power tools, ultrafast, limited slots.
   - Hand-Labor Workstation (RAM / CPU): Manual labor by hand, slower, many slots (zero line/stall).
4. Transient Discard:
   - In transient mode, after finishing with power tools, the expert immediately steps out
     of the VRAM slot so another person can use it.
"""
from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import enum
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


class BusynessLevel(str, enum.Enum):
    HOT = "hot"      # Constantly busy, high priority for High-Tech Tool Workstation
    WARM = "warm"    # Moderately busy, kept on RAM shelf
    COLD = "cold"    # Rare / cold consultant, stored on SSD shelf


class WorkstationType(str, enum.Enum):
    # Only two compute workstations exist! SSD is pure storage and cannot compute.
    TOOL_WORKSTATION_VRAM = "tool_workstation_vram"   # Fast with power tools, limited slots
    HAND_WORKSTATION_RAM = "hand_workstation_ram"     # Slower by hand, many slots (no line)


@dataclasses.dataclass
class ManagerDirective:
    """The manager's explicit instructions to the expert."""
    assigned_workstation: WorkstationType
    reason: str
    requires_ssd_to_ram_staging: bool = False


@dataclasses.dataclass
class ExpertJob:
    expert_id: str
    tokens: np.ndarray  # Shape: (batch_size, hidden_dim) or activation vector
    future_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    result: Optional[np.ndarray] = None
    executed_on: Optional[WorkstationType] = None
    dual_racing: bool = False
    counterpart_job: Optional[Any] = None
    completed: bool = False
    busyness: Optional[BusynessLevel] = None
    manager_directive: Optional[ManagerDirective] = None
    staged_from_ssd_to_ram: bool = False
    enqueue_time: float = dataclasses.field(default_factory=time.monotonic)
    dispatch_time: float = 0.0
    completion_time: float = 0.0
    error: Optional[Exception] = None


class WorkshopManager:
    """The central coordinator directing experts to the appropriate workstation."""

    def __init__(
        self,
        max_tool_slots: int = 4,
        max_tool_queue_depth: int = 8,
        hot_threshold: int = 5,
        warm_threshold: int = 2,
    ) -> None:
        self.max_tool_slots = max_tool_slots
        self.max_tool_queue_depth = max_tool_queue_depth
        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold

    def assign_workstation(
        self,
        expert_id: str,
        busyness: BusynessLevel,
        occupied_slots: int,
        has_tool_slot: bool,
        tool_queue_depth: int,
        is_in_ram: bool,
    ) -> ManagerDirective:
        """Tells the expert exactly where to work and whether to fetch from SSD storage first."""
        needs_staging = not is_in_ram
        line_overloaded = tool_queue_depth >= self.max_tool_queue_depth
        free_slots = occupied_slots < self.max_tool_slots

        # Manager decision tree:
        # Rule 1: If person already holds a tool slot and line is not congested -> Tool Workstation
        if has_tool_slot and not line_overloaded:
            return ManagerDirective(
                assigned_workstation=WorkstationType.TOOL_WORKSTATION_VRAM,
                reason="Manager: You already occupy a Tool Workstation slot. Proceed with power tools.",
                requires_ssd_to_ram_staging=needs_staging,
            )

        # Rule 2: If tool slots are free and expert is HOT -> VIP assigned to Tool Workstation
        if free_slots and not line_overloaded and busyness == BusynessLevel.HOT:
            return ManagerDirective(
                assigned_workstation=WorkstationType.TOOL_WORKSTATION_VRAM,
                reason="Manager: You are a HOT expert with high busyness. Slot granted at Tool Workstation.",
                requires_ssd_to_ram_staging=needs_staging,
            )

        # Rule 3: If tool slots are free and queue is empty -> Tool Workstation
        if free_slots and tool_queue_depth == 0:
            return ManagerDirective(
                assigned_workstation=WorkstationType.TOOL_WORKSTATION_VRAM,
                reason="Manager: Tool Workstation has open slots and no waiting line. Assigned to VRAM.",
                requires_ssd_to_ram_staging=needs_staging,
            )

        # Rule 4: If table is full, check busyness and queue depth:
        # In-and-go 1-go:1-in applies for HOT/WARM experts or when queue is light (queue < max_tool_queue_depth // 2)
        if not line_overloaded and (busyness in (BusynessLevel.HOT, BusynessLevel.WARM) or tool_queue_depth < max(1, self.max_tool_queue_depth // 2)):
            return ManagerDirective(
                assigned_workstation=WorkstationType.TOOL_WORKSTATION_VRAM,
                reason="Manager: In-and-Go active. Slot granted at Tool Workstation via 1-Go:1-In table eviction.",
                requires_ssd_to_ram_staging=needs_staging,
            )

        # Rule 5: Otherwise Tool Workstation is full/busy -> Divert to Hand-Labor Workstation!
        return ManagerDirective(
            assigned_workstation=WorkstationType.HAND_WORKSTATION_RAM,
            reason="Manager: Tool Workstation is full/busy. Don't wait in line; do the job by hand in RAM!",
            requires_ssd_to_ram_staging=needs_staging,
        )


class ExpertExecutionQueue:
    """Schedules MoE expert tasks across High-Tech Tool (VRAM) and Hand-Labor (RAM) Workstations."""

    def _calculate_vram_chairs(
        self,
        vram_cache: Any,
        target_fraction: float = 0.80,
        avg_expert_bytes: int = 2_506_752,
        min_chairs: int = 500,
    ) -> int:
        """Dynamically compute the number of chairs at the VRAM Tool Workstation.
        Preserves a strict minimum of 500 chairs on VRAM, while adapting to free VRAM after GPU layer offload.
        """
        # User requirement: minimum 500 chairs reserved on VRAM
        min_chairs = max(500, int(os.getenv("MOE_MIN_VRAM_CHAIRS", str(min_chairs))))
        if vram_cache is not None and hasattr(vram_cache, "snapshot"):
            try:
                snap = vram_cache.snapshot()
                free_bytes = int(snap.get("free_bytes") or snap.get("budget_bytes") or 0)
                if free_bytes > 0:
                    budget_for_chairs = int(free_bytes * target_fraction)
                    calculated = budget_for_chairs // max(1, avg_expert_bytes)
                    return max(min_chairs, int(calculated))
            except Exception:
                pass
        return min_chairs

    def _calculate_ram_chairs(
        self,
        ram_cache: Any,
        avg_expert_bytes: int = 2_506_752,
        min_chairs: int = 16,
        ram_target_fraction: float = 0.50,
    ) -> int:
        """Dynamically compute the number of chairs at the RAM Hand-Labor Workstation.
        Allocated to 50% of system RAM max since RAM acts purely as a secondary backup of VRAM.
        Max number of chairs = allocated 50% RAM capacity / size of expert.
        """
        try:
            import psutil
            total_ram = psutil.virtual_memory().total
            budget = int(total_ram * ram_target_fraction)
            if budget > 0:
                return max(min_chairs, budget // max(1, avg_expert_bytes))
        except Exception:
            pass

        if ram_cache is not None:
            try:
                budget = int(getattr(ram_cache, "budget_bytes", 0) or 0)
                if budget <= 0 and hasattr(ram_cache, "_effective_budget_locked"):
                    with ram_cache._state_lock:
                        budget = ram_cache._effective_budget_locked()
                if budget > 0:
                    budget_50 = int(budget * 0.50)
                    return max(min_chairs, budget_50 // max(1, avg_expert_bytes))
            except Exception:
                pass
        return max(min_chairs, 2400)

    def __init__(
        self,
        vram_cache: Any = None,
        ram_cache: Any = None,
        max_vram_slots: int | None = None,   # Dynamically calculated from 80% VRAM if None
        max_vram_line_length: int = 32,       # Max waiting queue before diverting to hand-labor
        hot_threshold_calls: int = 5,
        warm_threshold_calls: int = 2,
        auto_promote_to_ram: bool = True,
        hand_labor_workers: int = 4,         # CPU worker threads doing work by hand
        transient_mode: bool = False,
        vram_target_fraction: float = 0.80,  # Maximized up to 80% of total VRAM capacity
        avg_expert_bytes: int = 2_506_752,   # ~5.5MB average expert size (DeepSeek/MoE)
        min_vram_slots: int = 4,             # Minimum 4 experts
        auto_adjust_ratio: bool = True,       # Dynamically auto-tune in-vs-go ratio (1:1 up to 50:50) based on traffic
        eviction_batch_fraction: float = 0.10, # Initial fallback batch eviction fraction
        # Backward compatibility aliases
        max_vram_table_experts: int | None = None,
        max_vram_queue_depth: int | None = None,
        hot_threshold_hits: int | None = None,
        cpu_workers: int | None = None,
    ) -> None:
        self.vram_cache = vram_cache
        self.ram_cache = ram_cache

        # Calculate dynamic chairs based on 50% VRAM capacity (minimum 4 experts)
        explicit_slots = max_vram_table_experts if max_vram_table_experts is not None else max_vram_slots
        if explicit_slots is not None:
            self.max_vram_slots = max(min_vram_slots, int(explicit_slots))
        else:
            self.max_vram_slots = self._calculate_vram_chairs(
                vram_cache=self.vram_cache,
                target_fraction=vram_target_fraction,
                avg_expert_bytes=avg_expert_bytes,
                min_chairs=min_vram_slots,
            )

        self.auto_optimized_plan: dict[str, Any] = {}
        self.max_ram_slots = self._calculate_ram_chairs(
            ram_cache=self.ram_cache,
            avg_expert_bytes=avg_expert_bytes,
        )
        self._ram_workstation_slots: collections.OrderedDict[str, float] = collections.OrderedDict()
        self.ram_spill_idle_evict_seconds = float(os.getenv("MOE_RAM_SPILL_IDLE_SECONDS", "10.0"))
        self.max_ram_line_length = 64  # Deep spillway buffer for RAM hand-labor  # Lower queue depth for RAM hand-labor (slower execution)

        line_len = max_vram_queue_depth if max_vram_queue_depth is not None else max_vram_line_length
        # VRAM is ultrafast with power tools, so increase fast queue capacity (e.g. 64)
        self.max_vram_line_length = max(32, int(line_len))
        hot_calls = hot_threshold_hits if hot_threshold_hits is not None else hot_threshold_calls
        self.hot_threshold_calls = int(hot_calls)
        self.warm_threshold_calls = int(warm_threshold_calls)
        self.auto_promote_to_ram = bool(auto_promote_to_ram)
        self.transient_mode = bool(transient_mode)
        self.auto_adjust_ratio = bool(auto_adjust_ratio)
        self.eviction_batch_fraction = max(0.0, min(0.50, float(eviction_batch_fraction)))
        cpu_w = cpu_workers if cpu_workers is not None else hand_labor_workers

        # The Manager
        self.manager = WorkshopManager(
            max_tool_slots=self.max_vram_slots,
            max_tool_queue_depth=self.max_vram_line_length,
            hot_threshold=self.hot_threshold_calls,
            warm_threshold=self.warm_threshold_calls,
        )

        self._lock = threading.Lock()
        self._shared_workstation_queue: collections.deque[ExpertJob] = collections.deque()
        self._tool_workstation_queue = self._shared_workstation_queue  # alias for backward compat
        self._tool_workstation_slots: collections.OrderedDict[str, Any] = collections.OrderedDict()
        self._busyness_catalog: collections.Counter = collections.Counter()

        # Telemetry
        self.tool_jobs_completed = 0
        self.hand_jobs_completed = 0
        self.tool_total_time_ms = 0.0
        self.hand_total_time_ms = 0.0
        self.promotions_to_hot = 0
        self.promotions_to_warm = 0
        self.displaced_from_tool_slots = 0
        self.ssd_staging_fetches = 0
        self._recent_events: collections.deque[dict[str, Any]] = collections.deque(maxlen=40)

        # Hand-Labor Workstation: Multi-threaded pool for manual execution in RAM
        self._hand_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, cpu_w), thread_name_prefix="moe-hand-workstation"
        )

        # Worker thread managing the High-Tech Tool Workstation (VRAM)
        self._running = True
        self._cond = threading.Condition(self._lock)
        self._tool_worker_thread = threading.Thread(target=self._tool_workstation_loop, daemon=True)
        self._tool_worker_thread.start()

        # Idle reaper thread for RAM spillway workstation:
        # Experts stay persistently seated in VRAM waiting for jobs or eviction by incoming experts.
        # RAM spillway experts are evicted if idle for counted duration.
        self._ram_reaper_thread = threading.Thread(target=self._ram_spill_reaper_loop, daemon=True)
        self._ram_reaper_thread.start()

    def shutdown(self) -> None:
        with self._lock:
            self._running = False
            self._cond.notify_all()
        if self._tool_worker_thread.is_alive():
            self._tool_worker_thread.join(timeout=1.0)
        if hasattr(self, "_ram_reaper_thread") and self._ram_reaper_thread.is_alive():
            self._ram_reaper_thread.join(timeout=1.0)
        self._hand_executor.shutdown(wait=False)

    def _ram_spill_reaper_loop(self) -> None:
        """Periodic background reaper that evicts RAM spillway experts after being idle for counted duration.
        VRAM chairs keep experts persistently seated waiting for new work or eviction by incoming experts.
        RAM hand-labor chairs automatically free up if not actively used.
        """
        while self._running:
            time.sleep(1.0)
            now = time.monotonic()
            evicted_keys: list[str] = []
            with self._lock:
                for key, last_active in list(self._ram_workstation_slots.items()):
                    if (now - last_active) >= self.ram_spill_idle_evict_seconds:
                        evicted_keys.append(key)
                        del self._ram_workstation_slots[key]
                        self._recent_events.append({
                            "time": time.time(),
                            "expert_id": key,
                            "direction": "GO (RAM Idle Eviction)",
                            "workstation": "RAM Hand-Labor Workstation",
                            "action": f"Evicted from RAM spillway after {round(now - last_active, 1)}s idle"
                        })
            if evicted_keys and self.ram_cache is not None and hasattr(self.ram_cache, "remove"):
                for k in evicted_keys:
                    try:
                        self.ram_cache.remove(k)
                    except Exception:
                        pass

    def _get_busyness_locked(self, expert_id: str) -> BusynessLevel:
        calls = self._busyness_catalog[str(expert_id)]
        if calls >= self.hot_threshold_calls:
            return BusynessLevel.HOT
        elif calls >= self.warm_threshold_calls:
            return BusynessLevel.WARM
        else:
            return BusynessLevel.COLD

    def get_busyness(self, expert_id: str) -> BusynessLevel:
        """Lookup person's activity in the Busyness Catalog."""
        with self._lock:
            return self._get_busyness_locked(expert_id)

    def _ensure_weights_in_ram(self, expert_id: str, job: ExpertJob) -> Any:
        """SSD is pure storage, not a compute workstation. Ensure weights are brought to RAM before work."""
        key = str(expert_id)
        if self.ram_cache is None:
            return None

        if hasattr(self.ram_cache, "contains") and not self.ram_cache.contains(key):
            if hasattr(self.ram_cache, "fetch_from_storage"):
                self.ram_cache.fetch_from_storage(key)
                job.staged_from_ssd_to_ram = True
                with self._lock:
                    self.ssd_staging_fetches += 1

        if hasattr(self.ram_cache, "get_copy"):
            return self.ram_cache.get_copy(key)
        return None

    def _tool_workstation_loop(self) -> None:
        """High-Tech Tool Workstation loop (VRAM/GPU)."""
        while True:
            job: Optional[ExpertJob] = None
            with self._lock:
                while self._running and not self._tool_workstation_queue:
                    self._cond.wait(timeout=0.1)
                if not self._running:
                    break
                if self._tool_workstation_queue:
                    job = self._tool_workstation_queue.popleft()

            if job is not None:
                self._execute_with_tools(job)

    def _execute_with_tools(self, job: ExpertJob) -> None:
        """Execute task at High-Tech Tool Workstation (GPU/VRAM) with specialized power tools."""
        start_t = time.perf_counter()
        job.dispatch_time = time.monotonic()
        key = str(job.expert_id)

        try:
            weights = self._ensure_weights_in_ram(key, job)

            with self._lock:
                if key in self._tool_workstation_slots:
                    self._tool_workstation_slots.move_to_end(key)
                else:
                    # Dynamic In-vs-Go Auto-Adjustment:
                    # When auto_adjust_ratio is enabled, the manager dynamically calculates the ratio:
                    # - Under light load / serial token generation: Uses 1:1 (single eviction) to preserve 100% of hot set.
                    # - Under high queue pressure / batched bursts: Auto-scales up to 50:50 (chunk eviction) to eliminate contention.
                    if len(self._tool_workstation_slots) >= self.max_vram_slots:
                        if self.auto_adjust_ratio:
                            q_depth = len(self._tool_workstation_queue)
                            if q_depth <= 1:
                                num_to_evict = 1  # 1:1 ratio for serial stream
                            elif q_depth <= 4:
                                num_to_evict = max(2, int(self.max_vram_slots * 0.10))  # 10:90 ratio
                            elif q_depth <= 8:
                                num_to_evict = max(4, int(self.max_vram_slots * 0.25))  # 25:75 ratio
                            else:
                                num_to_evict = max(8, int(self.max_vram_slots * 0.50))  # 50:50 ratio for huge surges
                        else:
                            num_to_evict = max(1, int(self.max_vram_slots * self.eviction_batch_fraction))

                        for _ in range(num_to_evict):
                            if not self._tool_workstation_slots:
                                break
                            # Prioritize eviction by heat: Cold first, Warm second, Hot protected
                            # Find the lowest heat score among seated experts
                            seated_keys = list(self._tool_workstation_slots.keys())
                            vacate_key = min(seated_keys, key=lambda k: self._busyness_catalog.get(k, 0))
                            self._tool_workstation_slots.pop(vacate_key, None)
                            self.displaced_from_tool_slots += 1
                            self._recent_events.append({
                                "time": time.time(),
                                "expert_id": vacate_key,
                                "direction": "GO (Outgoing / Evicted)",
                                "workstation": "VRAM Tool Workstation",
                                "action": f"Evicted lowest-heat expert (heat={self._busyness_catalog.get(vacate_key, 0)}) to protect Top Hot in VRAM chair"
                            })
                            if self.vram_cache is not None:
                                try:
                                    self.vram_cache.remove(vacate_key)
                                except Exception:
                                    pass

                    device_tensor = None
                    if self.vram_cache is not None and weights is not None:
                        try:
                            # If not already uploaded into physical VRAM, upload directly from staged RAM weights
                            if not self.vram_cache.contains(key):
                                hit_cnt = max(self.vram_cache.promote_after_hits, self._busyness_catalog.get(key, 1))
                                self.vram_cache.observe(key, weights, hit_count=hit_cnt, source="ram")
                            device_tensor = self.vram_cache.acquire_device(key)
                        except Exception:
                            device_tensor = None
                    self._tool_workstation_slots[key] = device_tensor if device_tensor is not None else object()
                    self._recent_events.append({
                        "time": time.time(),
                        "expert_id": key,
                        "direction": "IN (Incoming / Seated)",
                        "workstation": "VRAM Tool Workstation",
                        "action": "Seated in VRAM chair (1-In)"
                    })

            if weights is not None and isinstance(weights, np.ndarray) and weights.ndim == 2:
                if weights.shape[0] == job.tokens.shape[-1]:
                    job.result = np.dot(job.tokens, weights)
                else:
                    job.result = job.tokens * 1.01
            else:
                job.result = job.tokens * 1.005  # Power tool transformation marker

            job.executed_on = WorkstationType.TOOL_WORKSTATION_VRAM

            if self.transient_mode:
                with self._lock:
                    if key in self._tool_workstation_slots:
                        self._tool_workstation_slots.pop(key, None)
                        self.displaced_from_tool_slots += 1
                        if self.vram_cache is not None:
                            try:
                                self.vram_cache.remove(key)
                            except Exception:
                                pass

        except Exception as ex:
            job.error = ex
        finally:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            job.completion_time = time.monotonic()
            with self._lock:
                self.tool_jobs_completed += 1
                self.tool_total_time_ms += elapsed_ms
                job.completed = True
                # Dual Reading Race Winner: VRAM produced token first!
                # Evict/cancel the counterpart in RAM if needed to clear room for the next expert
                if job.dual_racing and job.counterpart_job is not None:
                    cp = job.counterpart_job
                    cp.completed = True
                    cp.result = job.result
                    cp.executed_on = WorkstationType.TOOL_WORKSTATION_VRAM
                    cp.future_event.set()
                    # Evict counterpart from RAM chair if room needed
                    if key in self._ram_workstation_slots and len(self._ram_workstation_slots) >= self.max_ram_slots:
                        self._ram_workstation_slots.pop(key, None)
                        self._recent_events.append({
                            "time": time.time(),
                            "expert_id": key,
                            "direction": "GO (RAM Evicted via VRAM Win)",
                            "workstation": "RAM Hand-Labor Workstation",
                            "action": "VRAM tool finished first; cleared counterpart from RAM"
                        })
            job.future_event.set()

    def _try_consume_shared_queue_by_hand(self) -> None:
        """RAM Hand-Labor worker pulls and computes from the shared queue line when VRAM is busy."""
        job: Optional[ExpertJob] = None
        with self._lock:
            # Check if there are unassigned jobs in the shared queue beyond the first item
            if len(self._shared_workstation_queue) > 0:
                # If VRAM tool slots are busy or expert is cold/warm, spill to RAM
                for idx, queued_job in enumerate(self._shared_workstation_queue):
                    if not queued_job.completed and queued_job.executed_on is None:
                        job = queued_job
                        del self._shared_workstation_queue[idx]
                        break
        if job is not None:
            self._execute_by_hand(job)

    def _execute_by_hand(self, job: ExpertJob) -> None:
        """Execute task at Hand-Labor Workstation (RAM/CPU): manual computation in RAM (never on SSD!)."""
        start_t = time.perf_counter()
        job.dispatch_time = time.monotonic()
        key = str(job.expert_id)

        try:
            weights = self._ensure_weights_in_ram(key, job)

            with self._lock:
                busyness = self._get_busyness_locked(key)
                job.busyness = busyness

                # In-and-go on RAM Workstation table:
                if key in self._ram_workstation_slots:
                    self._ram_workstation_slots.move_to_end(key)
                else:
                    if len(self._ram_workstation_slots) >= self.max_ram_slots:
                        vacate_ram, _ = self._ram_workstation_slots.popitem(last=False)
                        self._recent_events.append({
                            "time": time.time(),
                            "expert_id": vacate_ram,
                            "direction": "GO (RAM Stepped Out)",
                            "workstation": "RAM Hand-Labor Workstation",
                            "action": "Stepped out of RAM chair (1-Go)"
                        })
                    self._ram_workstation_slots[key] = time.monotonic()

            if self.auto_promote_to_ram and busyness in {BusynessLevel.HOT, BusynessLevel.WARM}:
                if self.ram_cache is not None and hasattr(self.ram_cache, "observe_hit"):
                    self.ram_cache.observe_hit(key)

            if weights is not None and isinstance(weights, np.ndarray) and weights.ndim == 2:
                if weights.shape[0] == job.tokens.shape[-1]:
                    job.result = np.dot(job.tokens, weights)
                else:
                    job.result = job.tokens * 1.002
            else:
                job.result = job.tokens * 1.001  # Hand labor transformation marker

            job.executed_on = WorkstationType.HAND_WORKSTATION_RAM
            with self._lock:
                self._recent_events.append({
                    "time": time.time(),
                    "expert_id": key,
                    "direction": "IN (RAM Hand-Labor)",
                    "workstation": "RAM Hand-Labor Workstation",
                    "action": "Hand-labor manual computation in RAM"
                })
        except Exception as ex:
            job.error = ex
        finally:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            job.completion_time = time.monotonic()
            with self._lock:
                self.hand_jobs_completed += 1
                self.hand_total_time_ms += elapsed_ms
                job.completed = True
                # If counterpart was already completed by VRAM, adopt winning result
                if job.dual_racing and job.counterpart_job is not None:
                    cp = job.counterpart_job
                    if cp.completed and cp.result is not None:
                        job.result = cp.result
                        job.executed_on = cp.executed_on
            job.future_event.set()

    def dispatch(
        self,
        expert_id: str,
        tokens: np.ndarray,
        sync: bool = True,
        timeout: float = 5.0,
    ) -> ExpertJob:
        """The Manager tells the expert where to work (Tool Workstation or Hand-Labor Workstation)."""
        key = str(expert_id)
        job = ExpertJob(expert_id=key, tokens=tokens)

        with self._lock:
            self._busyness_catalog[key] += 1
            busyness = self._get_busyness_locked(key)
            job.busyness = busyness

            has_tool_slot = key in self._tool_workstation_slots
            tool_line_length = len(self._tool_workstation_queue)
            occupied_slots = len(self._tool_workstation_slots)
            is_in_ram = self.ram_cache is not None and hasattr(self.ram_cache, "contains") and self.ram_cache.contains(key)

            # The Manager decides where the expert goes
            directive = self.manager.assign_workstation(
                expert_id=key,
                busyness=busyness,
                occupied_slots=occupied_slots,
                has_tool_slot=has_tool_slot,
                tool_queue_depth=tool_line_length,
                is_in_ram=is_in_ram,
            )
            job.manager_directive = directive

            # Unlimited Unified Shared Queue Line:
            # Every job enters the single shared line.
            # 1. Primary consumer: VRAM Tool Workstation (fastest hardware tools).
            # 2. Spillway consumer: RAM Hand-Labor Workstation immediately assists if VRAM is busy.
            self._shared_workstation_queue.append(job)
            self._cond.notify()

            # If VRAM queue has waiting jobs and VRAM chairs are currently processing,
            # trigger RAM spillway workers to pick up spillover directly from the shared line.
            if len(self._shared_workstation_queue) > 1 or directive.assigned_workstation != WorkstationType.TOOL_WORKSTATION_VRAM:
                self._hand_executor.submit(self._try_consume_shared_queue_by_hand)

        if sync:
            ok = job.future_event.wait(timeout=timeout)
            if not ok:
                raise TimeoutError(f"Job for expert {expert_id} timed out after {timeout}s")
            if job.error:
                raise job.error

        return job

    def dispatch_batch(
        self,
        batch: List[Tuple[str, np.ndarray]],
        timeout: float = 10.0,
    ) -> List[ExpertJob]:
        """Dispatch a swarm of expert tasks; Manager assigns each to the optimal workstation."""
        jobs: List[ExpertJob] = []
        for expert_id, tokens in batch:
            jobs.append(self.dispatch(expert_id, tokens, sync=False))

        start_wait = time.monotonic()
        for job in jobs:
            rem = max(0.01, timeout - (time.monotonic() - start_wait))
            if not job.future_event.wait(timeout=rem):
                raise TimeoutError(f"Batch execution timed out waiting for expert {job.expert_id}")
            if job.error:
                raise job.error
        return jobs

    def update_allocation(self, chairs: int, plan: Optional[dict[str, Any]] = None) -> None:
        """Dynamically update VRAM workstation chairs and store optimization telemetry."""
        with self._lock:
            self.max_vram_slots = max(16, int(chairs))
            self.manager.max_tool_slots = self.max_vram_slots
            if plan:
                self.auto_optimized_plan = dict(plan)

    def snapshot(self) -> Dict[str, Any]:
        """Telemetry snapshot of manager decisions, people busyness, and workstation slot activity."""
        with self._lock:
            t_avg = (self.tool_total_time_ms / self.tool_jobs_completed) if self.tool_jobs_completed > 0 else 0.0
            h_avg = (self.hand_total_time_ms / self.hand_jobs_completed) if self.hand_jobs_completed > 0 else 0.0

            hot_people = [k for k, v in self._busyness_catalog.items() if v >= self.hot_threshold_calls]
            warm_people = [k for k, v in self._busyness_catalog.items() if self.warm_threshold_calls <= v < self.hot_threshold_calls]
            cold_people = [k for k, v in self._busyness_catalog.items() if v < self.warm_threshold_calls]

            return {
                "manager": {
                    "role": "Directs experts to High-Tech Tool (VRAM) vs Hand-Labor (RAM) Workstations",
                    "max_tool_slots": self.max_vram_slots,
                    "max_tool_queue_depth": self.max_vram_line_length,
                },
                "busyness_catalog": {
                    "hot_people_count": len(hot_people),
                    "hot_people": sorted(hot_people),
                    "warm_people_count": len(warm_people),
                    "warm_people": sorted(warm_people),
                    "cold_people_count": len(cold_people),
                    "cold_people": sorted(cold_people),
                    "call_frequencies": dict(self._busyness_catalog),
                },
                "storage_note": "SSD is pure storage only; computation occurs solely at VRAM or RAM workstations.",
                "ssd_storage": {
                    "fetches_from_ssd_to_ram": self.ssd_staging_fetches,
                },
                "recent_turnover_events": list(self._recent_events),
                "shared_queue_line": len(self._shared_workstation_queue),
                "tool_workstation_vram": {
                    "total_tool_slots": self.max_vram_slots,
                    "occupied_tool_slots": len(self._tool_workstation_slots),
                    "people_with_tools": list(self._tool_workstation_slots.keys()),
                    "waiting_line_length": len(self._shared_workstation_queue),
                    "max_waiting_line": None,  # Unlimited shared queue
                    "shared_queue_line": len(self._shared_workstation_queue),
                    "auto_adjust_ratio": self.auto_adjust_ratio,
                    "transient_mode": self.transient_mode,
                    "jobs_completed": self.tool_jobs_completed,
                    "avg_tool_latency_ms": round(t_avg, 3),
                    "displaced_people": self.displaced_from_tool_slots,
                },
                "hand_workstation_ram": {
                    "total_ram_slots": self.max_ram_slots,
                    "occupied_ram_slots": len(self._ram_workstation_slots),
                    "people_in_ram": list(self._ram_workstation_slots.keys()),
                    "waiting_line_length": 0,
                    "max_waiting_line": self.max_ram_line_length,
                    "jobs_completed": self.hand_jobs_completed,
                    "avg_hand_latency_ms": round(h_avg, 3),
                    "mode": "many_slots_manual_by_hand",
                },
                # Backward-compatibility fields
                "vram_table_capacity": self.max_vram_slots,
                "vram_table_occupied": len(self._tool_workstation_slots),
                "vram_table_active_experts": list(self._tool_workstation_slots.keys()),
                "vram_queue_depth": len(self._tool_workstation_queue),
                "max_vram_queue_depth": self.max_vram_line_length,
                "vram_jobs_completed": self.tool_jobs_completed,
                "ram_fallback_jobs_completed": self.hand_jobs_completed,
                "promotions_to_ram": len(hot_people) + len(warm_people),
                "evictions_from_vram": self.displaced_from_tool_slots,
                "avg_vram_latency_ms": round(t_avg, 3),
                "avg_ram_latency_ms": round(h_avg, 3),
                "expert_heat_counts": dict(self._busyness_catalog),
                "transient_mode": self.transient_mode,
                "auto_optimized_plan": dict(self.auto_optimized_plan),
            }
