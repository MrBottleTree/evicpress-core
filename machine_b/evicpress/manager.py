"""
EvicPressManager — the heart of Machine B.

Responsibilities:
  - Decide where to place a new block (Tier 1 / Tier 2 / Tier 3)
  - Evict blocks using the Greedy Knapsack algorithm from the paper
  - Maintain a local ledger of blocks promoted to Machine A RAM (Tier 1)
    without ever querying Machine A
  - Promote Tier 3 blocks to Tier 2 on retrieval (opportunistic)
  - Accept prefetch hints and process them in the background
  - Expose real-time state for the dashboard

Tier 1 design:
  Machine B controls Tier 1 (Machine A RAM) by returning tier=1 in StoreResponse.
  Machine A's GRPCBackend writes the block to its LocalCPUBackend when it sees tier=1.
  Machine B maintains a local Tier1Ledger as an estimate — it does NOT ping Machine A.
  The ledger may become stale if Machine A evicts entries; it self-corrects on next access.
"""

import asyncio
import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .block import Block
from .config import EvicPressConfig
from .tier_disk import DiskTier
from .tier_ram import RamTier


# ──────────────────────────────────────────────────────────────── #
#  Tier 1 ledger (local accounting of Machine A RAM)               #
# ──────────────────────────────────────────────────────────────── #


def _compute_quality(data: bytes) -> float:
    """Estimate block quality from its compression ratio.
    High ratio (hard to compress) = unique activations = high quality (keep in fast tier).
    Low ratio (easy to compress) = redundant patterns = low quality (evict first).
    """
    if not data:
        return 1.0
    sample = data[:65536]  # 64 KB sample is enough
    compressed_len = len(zlib.compress(sample, level=1))
    ratio = compressed_len / len(sample)
    return round(max(0.1, min(1.0, ratio)), 3)

class Tier1Ledger:
    """
    Local-only accounting of which blocks Machine B has promoted to Machine A RAM.
    Never queries Machine A. May be slightly stale if Machine A evicts entries.

    Admission policy: Greedy Knapsack by quality_score.
    When full, evict the lowest-quality block if the new block is better.
    """

    def __init__(self, capacity_bytes: int, bandwidth_bytes_per_sec: float) -> None:
        self._capacity = capacity_bytes
        self._bandwidth = bandwidth_bytes_per_sec
        # block_id → (size_bytes, quality_score)
        self._entries: dict[str, tuple[int, float]] = {}
        self._used_bytes: int = 0
        self._lock = threading.RLock()

    def try_admit(self, block_id: str, size_bytes: int, quality_score: float) -> bool:
        """
        Try to admit a block into Tier 1. May evict lower-quality blocks to make room.
        Returns True if the block was admitted (Machine A should cache it).
        """
        with self._lock:
            if block_id in self._entries:
                return True  # already tracked

            if size_bytes > self._capacity:
                return False  # block alone exceeds tier capacity

            # Free space by evicting lowest-quality blocks if needed
            if self._used_bytes + size_bytes > self._capacity:
                self._evict_to_make_room(size_bytes, quality_score)

            if self._used_bytes + size_bytes > self._capacity:
                return False  # not worth promoting after eviction attempt

            self._entries[block_id] = (size_bytes, quality_score)
            self._used_bytes += size_bytes
            return True

    def _evict_to_make_room(self, needed_bytes: int, new_quality: float) -> None:
        """Evict lowest-quality entries until enough space is free.
        Among equal-quality blocks, evicts oldest-inserted first (LRU tiebreak).
        Never evicts a block with strictly higher quality than the incoming one."""
        candidates = sorted(
            enumerate(self._entries.items()),
            key=lambda x: (x[1][1][1], x[0])  # (quality_score asc, insertion_idx asc)
        )
        for _, (bid, (sz, qs)) in candidates:
            if self._used_bytes + needed_bytes <= self._capacity:
                break
            if qs > new_quality:
                break  # never displace a strictly better block
            del self._entries[bid]
            self._used_bytes -= sz

    def remove(self, block_id: str) -> int:
        """Remove block from ledger. Returns size_bytes removed (0 if not present)."""
        with self._lock:
            entry = self._entries.pop(block_id, None)
            if entry:
                self._used_bytes -= entry[0]
                return entry[0]
            return 0

    def contains(self, block_id: str) -> bool:
        with self._lock:
            return block_id in self._entries

    def has_space(self, size_bytes: int) -> bool:
        with self._lock:
            return self._used_bytes + size_bytes <= self._capacity

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def block_count(self) -> int:
        return len(self._entries)

    @property
    def utilization(self) -> float:
        if self._capacity == 0:
            return 0.0
        return self._used_bytes / self._capacity

    @property
    def bandwidth_bytes_per_sec(self) -> float:
        return self._bandwidth


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
        self.total_hits:      int = 0
        self.total_misses:    int = 0
        self.tier1_hits:      int = 0  # served from Machine A RAM (local estimate)
        self.tier2_hits:      int = 0
        self.tier3_hits:      int = 0
        self.evictions:       int = 0
        self.tier1_promotions: int = 0  # times Machine B signalled tier=1 to Machine A
        self.total_ops:       int = 0
        self._lock = threading.Lock()

    def hit(self, tier: int) -> None:
        with self._lock:
            self.total_hits += 1
            self.total_ops  += 1
            if tier == 1:
                self.tier1_hits += 1
            elif tier == 2:
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

    def promote_tier1(self) -> None:
        with self._lock:
            self.tier1_promotions += 1


# ──────────────────────────────────────────────────────────────── #
#  Manager                                                          #
# ──────────────────────────────────────────────────────────────── #

class EvicPressManager:
    def __init__(self, config: EvicPressConfig) -> None:
        self.config = config
        self.tier1 = Tier1Ledger(
            config.tier1.capacity_bytes,
            config.tier1.bandwidth_bytes_per_sec,
        )
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
        """Returns (hit, tier). tier=0 on miss, 1=Machine A RAM, 2=Machine B RAM, 3=disk."""
        # Tier 1 check first: if ledger says Machine A has it, return tier=1
        # so GRPCBackend skips any prefetch and serves directly from Machine A RAM.
        if self.tier1.contains(block_id):
            return True, 1
        if self.tier2.contains(block_id):
            return True, 2
        if self.tier3.contains(block_id):
            return True, 3
        return False, 0

    def store(self, block_id: str, data: bytes, quality_score: float = 1.0) -> Tuple[bool, int]:
        """
        Store a block. Machine B decides all tier placement.
        Returns (success, tier_placed).

        tier_placed == 1 means Machine B wants Machine A to also cache this block
        in its local RAM (Tier 1). The block is additionally stored in Tier 2 or 3
        on Machine B as the canonical copy.
        """
        # Override static quality_score with a data-driven estimate based on
        # compression ratio. Blocks that compress well are less unique/important.
        if quality_score >= 1.0:
            quality_score = _compute_quality(data)
        block = Block(block_id=block_id, data=data, tier=2, quality_score=quality_score)

        with self._lock:
            # Preserve access history if block already exists in T2
            existing_t2 = self.tier2.remove(block_id)
            if existing_t2:
                block.access_count = existing_t2.access_count
                block.created_at   = existing_t2.created_at
            # If block is in T3, leave it there until we confirm T2 has room.

            # Place block in Machine B (Tier 2 preferred, Tier 3 fallback)
            if self.tier2.has_space(block.size_bytes):
                self.tier3.remove(block_id)  # upgrading T3 → T2 if present
                self.tier2.put(block)
                canonical_tier = 2
                self._log("STORE", block_id, f"tier2 size={_fmt(block.size_bytes)}")
            elif self._evict_tier2_to_make_room(block.size_bytes):
                self.tier3.remove(block_id)  # upgrading T3 → T2 if present
                self.tier2.put(block)
                canonical_tier = 2
                self._log("STORE", block_id, f"tier2 (after eviction) size={_fmt(block.size_bytes)}")
            else:
                # T2 truly full — store/keep in T3
                if not self.tier3.contains(block_id):
                    block.tier = 3
                    if not self.tier3.has_space(block.size_bytes):
                        self._evict_tier3_to_make_room(block.size_bytes)
                    self.tier3.put(block)
                canonical_tier = 3
                self._log("STORE", block_id, f"tier3 size={_fmt(block.size_bytes)}")

            # Decide whether to also promote to Tier 1 (Machine A RAM).
            # Try to admit: Tier1Ledger uses Greedy Knapsack by quality_score.
            promoted = self.tier1.try_admit(block_id, block.size_bytes, quality_score)
            if promoted:
                self.stats.promote_tier1()
                self._log("PROMOTE", block_id, f"→tier1 quality={quality_score:.2f}")
                # Tiers are exclusive: Machine A holds the only copy.
                # Free Machine B storage entirely — no redundant backup.
                self.tier2.remove(block_id)
                self.tier3.remove(block_id)
                return True, 1  # Signal Machine A to cache in Tier 1

            return True, canonical_tier

    def retrieve(self, block_id: str) -> Optional[Tuple[bytes, int]]:
        """
        Fetch a block's data. Returns (data, source_tier) or None on miss.
        Tier 1 hits are served by Machine A without reaching here, so we only
        see Tier 2 / Tier 3 requests. Opportunistically promotes Tier 3 → Tier 2.
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

            # Miss — tier1 ledger may be stale; Machine A already missed too
            self.stats.miss()
            self._log("MISS", block_id, "not found")
            return None

    def delete(self, block_id: str) -> bool:
        with self._lock:
            in_t1 = self.tier1.remove(block_id) > 0
            in_t2 = self.tier2.remove(block_id) is not None
            in_t3 = self.tier3.remove(block_id)
            removed = in_t1 or in_t2 or in_t3
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
        When a block is demoted from Tier 2, it is also removed from the Tier 1 ledger
        (Machine A's copy becomes lower priority; Machine B's canonical copy is now slower).
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

            # Remove from Tier 1 ledger: canonical copy is going to slower Tier 3
            self.tier1.remove(removed.block_id)

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
        Also removes evicted blocks from the Tier 1 ledger.
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
                self.tier1.remove(meta["block_id"])  # stale ledger entry
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
            "tier1": {
                "used_bytes":    self.tier1.used_bytes,
                "capacity_bytes": self.tier1.capacity_bytes,
                "block_count":   self.tier1.block_count,
                "utilization":   self.tier1.utilization,
            },
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
                "total_hits":       self.stats.total_hits,
                "total_misses":     self.stats.total_misses,
                "tier1_hits":       self.stats.tier1_hits,
                "tier2_hits":       self.stats.tier2_hits,
                "tier3_hits":       self.stats.tier3_hits,
                "tier1_promotions": self.stats.tier1_promotions,
                "evictions":        self.stats.evictions,
                "hit_rate":         hit_rate,
                "total_ops":        self.stats.total_ops,
            },
            "recent_ops": ops,
            "config": {
                "alpha":               self.config.alpha,
                "tier1_capacity_gb":   self.config.tier1.capacity_bytes / 1e9,
                "tier1_bandwidth_gbps": self.config.tier1.bandwidth_bytes_per_sec / 1e9,
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
