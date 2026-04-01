"""
EvicPressManager — the heart of Machine B.

Responsibilities:
  - Decide where to place a new block (Tier 2 or Tier 3)
  - Evict blocks using the Greedy Knapsack algorithm from the paper
  - Promote Tier 3 blocks to Tier 2 on retrieval (opportunistic)
  - Accept prefetch hints and process them in the background
  - Expose real-time state for the dashboard
"""

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .block import Block
from .config import EvicPressConfig
from .tier_disk import DiskTier
from .tier_ram import RamTier


# ──────────────────────────────────────────────────────────────── #
#  Supporting types                                                 #
# ──────────────────────────────────────────────────────────────── #

@dataclass
class Operation:
    """One entry in the recent-operations ring buffer."""
    op: str          # STORE | RETRIEVE | EVICT | PROMOTE | PREFETCH | MISS | DELETE
    block_id: str
    detail: str
    timestamp: float = field(default_factory=time.time)


class _Stats:
    def __init__(self) -> None:
        self.total_hits:   int = 0
        self.total_misses: int = 0
        self.tier2_hits:   int = 0
        self.tier3_hits:   int = 0
        self.evictions:    int = 0
        self.total_ops:    int = 0
        self._lock = threading.Lock()

    def hit(self, tier: int) -> None:
        with self._lock:
            self.total_hits += 1
            self.total_ops  += 1
            if tier == 2:
                self.tier2_hits += 1
            else:
                self.tier3_hits += 1

    def miss(self) -> None:
        with self._lock:
            self.total_misses += 1
            self.total_ops    += 1

    def evict(self) -> None:
        with self._lock:
            self.evictions += 1


# ──────────────────────────────────────────────────────────────── #
#  Manager                                                          #
# ──────────────────────────────────────────────────────────────── #

class EvicPressManager:
    def __init__(self, config: EvicPressConfig) -> None:
        self.config = config
        self.tier2 = RamTier(config.tier2.capacity_bytes)
        self.tier3 = DiskTier(config.tier3.capacity_bytes, config.tier3.data_dir)
        self.stats = _Stats()

        self._recent_ops: deque[Operation] = deque(maxlen=config.max_recent_ops)
        self._lock = threading.RLock()

        # Filled in main.py once the asyncio loop is running
        self.prefetch_queue: Optional[asyncio.Queue] = None

    # ────────────────────────────────────────── #
    #  Core operations                           #
    # ────────────────────────────────────────── #

    def lookup(self, block_id: str) -> Tuple[bool, int]:
        """Returns (hit, tier). tier=0 on miss."""
        if self.tier2.contains(block_id):
            return True, 2
        if self.tier3.contains(block_id):
            return True, 3
        return False, 0

    def store(self, block_id: str, data: bytes, quality_score: float = 1.0) -> Tuple[bool, int]:
        """
        Store a block. Machine B decides which tier.
        Returns (success, tier_placed).
        """
        block = Block(block_id=block_id, data=data, tier=2, quality_score=quality_score)

        with self._lock:
            # Preserve access history if block already exists
            existing_t2 = self.tier2.remove(block_id)
            existing_t3_removed = False
            if existing_t2:
                block.access_count = existing_t2.access_count
                block.created_at   = existing_t2.created_at
            elif self.tier3.contains(block_id):
                self.tier3.remove(block_id)
                existing_t3_removed = True

            # Try Tier 2 first
            if self.tier2.has_space(block.size_bytes):
                self.tier2.put(block)
                self._log("STORE", block_id, f"tier2 size={_fmt(block.size_bytes)}")
                return True, 2

            # Tier 2 full: run Greedy Knapsack to free space
            if self._evict_tier2_to_make_room(block.size_bytes):
                self.tier2.put(block)
                self._log("STORE", block_id, f"tier2 (after eviction) size={_fmt(block.size_bytes)}")
                return True, 2

            # Fall back to Tier 3
            block.tier = 3
            if not self.tier3.has_space(block.size_bytes):
                self._evict_tier3_to_make_room(block.size_bytes)

            self.tier3.put(block)
            self._log("STORE", block_id, f"tier3 size={_fmt(block.size_bytes)}")
            return True, 3

    def retrieve(self, block_id: str) -> Optional[Tuple[bytes, int]]:
        """
        Fetch a block's data. Returns (data, source_tier) or None on miss.
        Opportunistically promotes from Tier 3 → Tier 2 if RAM has space.
        """
        with self._lock:
            # Tier 2 check
            block = self.tier2.get(block_id)
            if block:
                self.stats.hit(2)
                self._log("RETRIEVE", block_id, "tier2 hit")
                return block.data, 2

            # Tier 3 check
            block = self.tier3.get(block_id)
            if block:
                self.stats.hit(3)
                self._log("RETRIEVE", block_id, "tier3 hit")
                # Opportunistic promotion
                if self.tier2.has_space(block.size_bytes):
                    self.tier3.remove(block_id)
                    block.tier = 2
                    self.tier2.put(block)
                    self._log("PROMOTE", block_id, "tier3→tier2")
                return block.data, 3

            # Miss
            self.stats.miss()
            self._log("MISS", block_id, "not found")
            return None

    def delete(self, block_id: str) -> bool:
        with self._lock:
            removed = self.tier2.remove(block_id) is not None
            removed |= self.tier3.remove(block_id)
            if removed:
                self._log("DELETE", block_id, "removed")
            return removed

    def queue_prefetch(self, block_ids: list[str]) -> int:
        """Push block IDs into the async prefetch queue. Returns how many were queued."""
        if not self.config.prefetch_enabled or self.prefetch_queue is None:
            return 0
        queued = 0
        for bid in block_ids:
            try:
                self.prefetch_queue.put_nowait(bid)
                queued += 1
            except asyncio.QueueFull:
                break
        return queued

    async def prefetch_worker(self) -> None:
        """
        Background asyncio task: moves blocks from Tier 3 → Tier 2 proactively.
        Runs forever; stops only when the event loop shuts down.
        """
        while True:
            block_id = await self.prefetch_queue.get()
            try:
                with self._lock:
                    if self.tier3.contains(block_id) and not self.tier2.contains(block_id):
                        block = self.tier3.get(block_id)
                        if block and self.tier2.has_space(block.size_bytes):
                            self.tier3.remove(block_id)
                            block.tier = 2
                            self.tier2.put(block)
                            self._log("PREFETCH", block_id, "tier3→tier2")
            finally:
                self.prefetch_queue.task_done()

    # ────────────────────────────────────────── #
    #  Greedy Knapsack eviction                  #
    # ────────────────────────────────────────── #

    def _evict_tier2_to_make_room(self, needed_bytes: int) -> bool:
        """
        Move lowest-utility Tier 2 blocks to Tier 3 until `needed_bytes` is free.
        Returns True if enough space was reclaimed.
        """
        total = self.stats.total_hits + self.stats.total_misses
        alpha = self.config.alpha
        t2bw  = self.config.tier2.bandwidth_bytes_per_sec
        t3bw  = self.config.tier3.bandwidth_bytes_per_sec

        candidates = sorted(
            self.tier2.all_blocks(),
            key=lambda b: b.utility(alpha, t2bw, t3bw, total)
        )

        freed = 0
        for block in candidates:
            if freed >= needed_bytes:
                break
            removed = self.tier2.remove(block.block_id)
            if removed is None:
                continue
            freed += removed.size_bytes
            self.stats.evict()

            # Push to Tier 3 (make room there first if needed)
            if not self.tier3.has_space(removed.size_bytes):
                self._evict_tier3_to_make_room(removed.size_bytes)
            removed.tier = 3
            self.tier3.put(removed)
            self._log("EVICT", block.block_id, "tier2→tier3")

        return freed >= needed_bytes

    def _evict_tier3_to_make_room(self, needed_bytes: int) -> None:
        """
        Delete lowest-utility Tier 3 blocks until `needed_bytes` is free.
        (Data is permanently lost — these are the lowest-value blocks.)
        """
        total = self.stats.total_hits + self.stats.total_misses
        alpha = self.config.alpha
        t2bw  = self.config.tier2.bandwidth_bytes_per_sec
        t3bw  = self.config.tier3.bandwidth_bytes_per_sec

        def _meta_utility(m: dict) -> float:
            freq = (m["access_count"] + 1) / (total + 1)
            ttft = m["size_bytes"] / t3bw
            return (alpha * m["quality_score"] - ttft) * freq

        candidates = sorted(self.tier3.all_metas(), key=_meta_utility)

        freed = 0
        for meta in candidates:
            if freed >= needed_bytes:
                break
            if self.tier3.remove(meta["block_id"]):
                freed += meta["size_bytes"]
                self.stats.evict()
                self._log("EVICT", meta["block_id"], "tier3→deleted")

    # ────────────────────────────────────────── #
    #  Dashboard state export                    #
    # ────────────────────────────────────────── #

    def get_state(self) -> dict:
        """Full state snapshot for the dashboard /api/state endpoint."""
        total = self.stats.total_hits + self.stats.total_misses
        hit_rate = self.stats.total_hits / total if total > 0 else 0.0

        ops = [
            {
                "op":       o.op,
                "block_id": o.block_id[:24] + "…",
                "detail":   o.detail,
                "time":     o.timestamp,
            }
            for o in list(self._recent_ops)
        ]

        return {
            "tier2": {
                "used_bytes":    self.tier2.used_bytes,
                "capacity_bytes": self.tier2.capacity_bytes,
                "block_count":   self.tier2.block_count,
                "utilization":   self.tier2.utilization,
            },
            "tier3": {
                "used_bytes":    self.tier3.used_bytes,
                "capacity_bytes": self.tier3.capacity_bytes,
                "block_count":   self.tier3.block_count,
                "utilization":   self.tier3.utilization,
            },
            "stats": {
                "total_hits":   self.stats.total_hits,
                "total_misses": self.stats.total_misses,
                "tier2_hits":   self.stats.tier2_hits,
                "tier3_hits":   self.stats.tier3_hits,
                "evictions":    self.stats.evictions,
                "hit_rate":     hit_rate,
                "total_ops":    self.stats.total_ops,
            },
            "recent_ops": ops,
            "config": {
                "alpha":               self.config.alpha,
                "tier2_capacity_gb":   self.config.tier2.capacity_bytes / 1e9,
                "tier3_capacity_gb":   self.config.tier3.capacity_bytes / 1e9,
                "tier2_bandwidth_gbps": self.config.tier2.bandwidth_bytes_per_sec / 1e9,
                "tier3_bandwidth_gbps": self.config.tier3.bandwidth_bytes_per_sec / 1e9,
                "prefetch_enabled":    self.config.prefetch_enabled,
                "server_port":         self.config.server_port,
                "data_dir":            self.config.tier3.data_dir,
            },
            "prefetch_queue_size": (
                self.prefetch_queue.qsize() if self.prefetch_queue else 0
            ),
        }

    def get_blocks(self) -> list[dict]:
        """Block listing for the dashboard block browser."""
        result = []
        for b in self.tier2.all_blocks():
            result.append(b.to_meta())
        for m in self.tier3.all_metas():
            result.append(m)
        return result

    # ────────────────────────────────────────── #
    #  Internals                                 #
    # ────────────────────────────────────────── #

    def _log(self, op: str, block_id: str, detail: str = "") -> None:
        self._recent_ops.appendleft(Operation(op=op, block_id=block_id, detail=detail))


def _fmt(b: int) -> str:
    """Human-readable byte size for log messages."""
    if b >= 1 << 30:
        return f"{b/(1<<30):.2f}GB"
    if b >= 1 << 20:
        return f"{b/(1<<20):.1f}MB"
    if b >= 1 << 10:
        return f"{b/(1<<10):.1f}KB"
    return f"{b}B"
