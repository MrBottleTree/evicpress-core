"""
Block: the fundamental unit managed by EvicPress.

A block represents a contiguous slice of a KV cache (e.g., one paged-attention
block for a specific token range). Machine A sends opaque bytes; Machine B stores,
moves, and evicts them based on utility scores.
"""

from dataclasses import dataclass, field
from time import time as _time


@dataclass
class Block:
    block_id: str       # Deterministic hash of the token sequence (from Machine A)
    data: bytes         # Raw serialized KV tensor — opaque to Machine B
    tier: int           # Current storage tier: 2 = RAM, 3 = Disk
    quality_score: float = 1.0   # 0.0–1.0, provided by Machine A (compression sensitivity)
    quant_level: str    = "fp16"  # "fp16" | "int8" | "int4" — applied by Machine B
    access_count: int   = 0
    last_access: float  = field(default_factory=_time)
    created_at: float   = field(default_factory=_time)

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    def utility(
        self,
        alpha: float,
        tier2_bw: float,
        tier3_bw: float,
        total_accesses: int,
    ) -> float:
        """
        Utility = (alpha * quality - ttft) * frequency

        This matches the EvicPress paper's joint optimization objective.
        Blocks with low utility are evicted first.
        """
        frequency = (self.access_count + 1) / (total_accesses + 1)
        bw = tier2_bw if self.tier == 2 else tier3_bw
        ttft = self.size_bytes / bw          # seconds of loading delay
        return (alpha * self.quality_score - ttft) * frequency

    def touch(self) -> None:
        """Record an access."""
        self.access_count += 1
        self.last_access = _time()

    def to_meta(self) -> dict:
        """Metadata snapshot (no data bytes) for serialization / dashboard."""
        return {
            "block_id":     self.block_id,
            "size_bytes":   self.size_bytes,
            "tier":         self.tier,
            "quality_score": self.quality_score,
            "quant_level":  self.quant_level,
            "access_count": self.access_count,
            "last_access":  self.last_access,
            "created_at":   self.created_at,
        }
