"""
EvicPressManager — the heart of Machine B.

Responsibilities:
  - Place every block in Tier 3 (disk) as the canonical copy
  - Additionally mirror into Tier 2 (B RAM) and/or Tier 1 (A RAM) based on utility
  - Evict from T2/T1 by simply dropping the cache copy (T3 still has the block)
  - Evict from T3 only under true pressure (this is the ONLY place data is lost)
  - Accept prefetch hints and process them in the background
  - Expose real-time state for the dashboard

Inclusive tiering:
  Every block always has a T3 copy while it exists in the system. T1 and T2 are
  pure cache layers on top. There is NO T1 return protocol — Machine A's
  LocalCPUBackend can evict its T1 copy on its own; Machine B never loses data
  because T3 remains.

Tier 1 ledger:
  Machine B maintains a local accounting of which block_ids it has asked
  Machine A to cache in its RAM. The ledger may be stale if Machine A evicts
  under its own LRU; it self-corrects on the next Lookup/Store.
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

class Tier1Ledger:
    """
    Local-only accounting of blocks Machine B asked Machine A to cache in T1.
    No return-queue, no ping to Machine A. On T1 pressure we just drop the
    ledger entry — T3 canonical copy remains on Machine B disk.
    """

    def __init__(self, capacity_bytes: int, bandwidth_bytes_per_sec: float) -> None:
        self._capacity = capacity_bytes
        self._bandwidth = bandwidth_bytes_per_sec
        # block_id → (size_bytes, utility, quant_level)
        self._entries: dict[str, tuple[int, float, str]] = {}
        self._used_bytes: int = 0
        self._lock = threading.RLock()

    def admit(
        self,
        block_id: str,
        size_bytes: int,
        utility: float,
        quant_level: str,
    ) -> bool:
        """
        Admit a block into Tier 1. May drop lower-utility entries to make room.
        Returns True if Machine A should cache the block in its local RAM.
        """
        with self._lock:
            if block_id in self._entries:
                # Refresh utility/size in case block changed
                old_size, _, _ = self._entries[block_id]
                self._used_bytes += size_bytes - old_size
                self._entries[block_id] = (size_bytes, utility, quant_level)
                return True

            if size_bytes > self._capacity:
                return False  # block alone exceeds tier capacity

            if self._used_bytes + size_bytes > self._capacity:
                self._drop_to_make_room(size_bytes, utility)

            if self._used_bytes + size_bytes > self._capacity:
                return False  # not worth admitting

            self._entries[block_id] = (size_bytes, utility, quant_level)
            self._used_bytes += size_bytes
            return True

    def _drop_to_make_room(self, needed_bytes: int, new_utility: float) -> None:
        """
        Drop lowest-utility entries until `needed_bytes` is free. Never displaces
        a block with strictly higher utility than the incoming one. These are
        pure drops — T3 still has the canonical copy so no data is lost.
        """
        candidates = sorted(
            self._entries.items(),
            key=lambda kv: kv[1][1],  # utility asc
        )
        for bid, (sz, util, _ql) in candidates:
            if self._used_bytes + needed_bytes <= self._capacity:
                break
            if util > new_utility:
                break
            del self._entries[bid]
            self._used_bytes -= sz

    def all_entries(self) -> list[dict]:
        """Snapshot of all T1 ledger entries for the dashboard block browser."""
        with self._lock:
            return [
                {
                    "block_id":     bid,
                    "size_bytes":   sz,
                    "tier":         1,
                    "quality_score": 0.0,
                    "quant_level":  ql,
                    "access_count": 0,
                    "last_access":  0.0,
                    "created_at":   0.0,
                    "utility":      util,
                }
                for bid, (sz, util, ql) in self._entries.items()
            ]

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

def _compute_quality(data: bytes) -> float:
    """Estimate block uniqueness via zlib compression ratio. High ratio = hard to compress = high quality."""
    if not data:
        return 1.0
    sample = data[:65536]
    compressed_len = len(zlib.compress(sample, level=1))
    ratio = compressed_len / len(sample)
    return round(max(0.1, min(1.0, ratio)), 3)


def _new_block_utility(
    size_bytes: int,
    quality_score: float,
    alpha: float,
    t2_bw: float,
    total_accesses: int,
) -> float:
    """Utility estimate for a freshly-stored block (access_count = 0, tier = 2)."""
    frequency = 1.0 / (total_accesses + 1)
    ttft = size_bytes / t2_bw
    return (alpha * quality_score - ttft) * frequency


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
        if self.tier1.contains(block_id):
            return True, 1
        if self.tier2.contains(block_id):
            return True, 2
        if self.tier3.contains(block_id):
            return True, 3
        return False, 0

    def store(
        self,
        block_id: str,
        data: bytes,
        quality_score: float = 1.0,
    ) -> Tuple[bool, int, str]:
        """
        Store a block under INCLUSIVE tiering semantics:
          1. T3 always receives the canonical copy.
          2. If utility is high enough and T2 has room (or can free it), mirror to T2.
          3. If utility is high enough and T1 ledger admits, promote to T1.

        Returns (success, primary_tier, quant_level).
        primary_tier = 1 if Machine A should cache it, else 2 if mirrored to B RAM, else 3.
        """
        if quality_score >= 1.0:
            quality_score = _compute_quality(data)

        # For M1 there is no placement policy yet — fp16 only, utility drives
        # the optional T2/T1 mirrors. M3 replaces this with config-driven bands.
        quant_level = "fp16"
        payload = data
        size = len(payload)

        with self._lock:
            total = self.stats.total_hits + self.stats.total_misses
            utility = _new_block_utility(
                size,
                quality_score,
                self.config.alpha,
                self.config.tier2.bandwidth_bytes_per_sec,
                total,
            )

            # Preserve access history if block already exists somewhere.
            existing = self.tier2.remove(block_id)
            access_count = existing.access_count if existing else 0
            created_at   = existing.created_at if existing else time.time()

            # Step 1 — T3 canonical write (free space first if needed).
            if not self.tier3.has_space(size):
                self._evict_tier3_to_make_room(size)
            t3_block = Block(
                block_id=block_id,
                data=payload,
                tier=3,
                quality_score=quality_score,
                quant_level=quant_level,
                access_count=access_count,
                created_at=created_at,
            )
            self.tier3.put(t3_block)
            self._log("STORE", block_id, f"tier3 size={_fmt(size)} util={utility:.4f}")

            # Step 2 — optional T2 mirror. Always try; on full, evict
            # lower-utility T2 copies (pure drops, T3 still has them).
            t2_mirrored = False
            if self.tier2.has_space(size):
                self._put_tier2_copy(block_id, payload, quality_score, quant_level,
                                     access_count, created_at)
                t2_mirrored = True
            elif self._evict_tier2_to_make_room(size):
                self._put_tier2_copy(block_id, payload, quality_score, quant_level,
                                     access_count, created_at)
                t2_mirrored = True

            # Step 3 — optional T1 promote.
            promoted = self.tier1.admit(block_id, size, utility, quant_level)
            if promoted:
                self.stats.promote_tier1()
                self._log("PROMOTE", block_id,
                          f"→tier1 util={utility:.4f} quant={quant_level}")
                return True, 1, quant_level

            primary = 2 if t2_mirrored else 3
            return True, primary, quant_level

    def _put_tier2_copy(
        self,
        block_id: str,
        data: bytes,
        quality_score: float,
        quant_level: str,
        access_count: int,
        created_at: float,
    ) -> None:
        """Place a T2 cache copy. T3 canonical copy is untouched."""
        self.tier2.put(Block(
            block_id=block_id,
            data=data,
            tier=2,
            quality_score=quality_score,
            quant_level=quant_level,
            access_count=access_count,
            created_at=created_at,
        ))

    def retrieve(self, block_id: str) -> Optional[Tuple[bytes, int, str]]:
        """
        Fetch a block's data. Returns (data, source_tier, quant_level) or None.
        Tier 1 hits are served by Machine A without reaching here. On T3 hit we
        also COPY (not move) into T2 if space is available, keeping inclusive.
        """
        with self._lock:
            block = self.tier2.get(block_id)
            if block:
                self.stats.hit(2)
                self._log("RETRIEVE", block_id, "tier2 hit")
                return block.data, 2, block.quant_level

            block = self.tier3.get(block_id)
            if block:
                self.stats.hit(3)
                self._log("RETRIEVE", block_id, "tier3 hit")
                # Inclusive-copy into T2 if space, leave T3 intact.
                if not self.tier2.contains(block_id) and self.tier2.has_space(block.size_bytes):
                    self._put_tier2_copy(
                        block.block_id,
                        block.data,
                        block.quality_score,
                        block.quant_level,
                        block.access_count,
                        block.created_at,
                    )
                    self._log("PROMOTE", block_id, "tier3→tier2 (copy)")
                return block.data, 3, block.quant_level

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
        Background asyncio task: copies blocks from Tier 3 → Tier 2 proactively.
        T3 copy is preserved (inclusive). Runs forever; stops only on event-loop shutdown.
        """
        while True:
            block_id = await self.prefetch_queue.get()
            try:
                with self._lock:
                    if self.tier3.contains(block_id) and not self.tier2.contains(block_id):
                        block = self.tier3.get(block_id)
                        if block and self.tier2.has_space(block.size_bytes):
                            self._put_tier2_copy(
                                block.block_id,
                                block.data,
                                block.quality_score,
                                block.quant_level,
                                block.access_count,
                                block.created_at,
                            )
                            self._log("PREFETCH", block_id, "tier3→tier2 (copy)")
            finally:
                self.prefetch_queue.task_done()

    # ────────────────────────────────────────── #
    #  Greedy Knapsack eviction                  #
    # ────────────────────────────────────────── #

    def _evict_tier2_to_make_room(self, needed_bytes: int) -> bool:
        """
        Drop lowest-utility Tier 2 cache copies until `needed_bytes` is free.
        Under inclusive tiering this is a pure drop — T3 retains every block.
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
            self._log("EVICT", block.block_id, "tier2 drop (T3 canonical)")

        return freed >= needed_bytes

    def _evict_tier3_to_make_room(self, needed_bytes: int) -> None:
        """
        Delete lowest-utility Tier 3 blocks until `needed_bytes` is free.
        This is the ONLY place data actually leaves the system — also drop
        any T2/T1 cache copies so stale entries don't linger.
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
            bid = meta["block_id"]
            if self.tier3.remove(bid):
                freed += meta["size_bytes"]
                self.stats.evict()
                self.tier2.remove(bid)
                self.tier1.remove(bid)
                self._log("EVICT", bid, "tier3→deleted")

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
        """Block listing for the dashboard block browser — all tiers."""
        result = self.tier1.all_entries()  # T1 blocks (metadata only)
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
