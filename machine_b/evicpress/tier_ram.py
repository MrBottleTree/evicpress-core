"""
Tier 2 — In-memory (RAM) storage.

Simple thread-safe dict. All block data lives in Python process memory.
This is the fast tier; Machine A fetches from here via gRPC with low latency.
"""

import threading
from typing import Optional

from .block import Block


class RamTier:
    def __init__(self, capacity_bytes: int) -> None:
        self._capacity = capacity_bytes
        self._blocks: dict[str, Block] = {}
        self._used_bytes: int = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def contains(self, block_id: str) -> bool:
        with self._lock:
            return block_id in self._blocks

    def get(self, block_id: str) -> Optional[Block]:
        """Return block and record access. None if not present."""
        with self._lock:
            block = self._blocks.get(block_id)
            if block:
                block.touch()
            return block

    def put(self, block: Block) -> bool:
        """
        Insert or replace a block. Returns False if block alone exceeds capacity.
        Caller is responsible for freeing space first if needed.
        """
        if block.size_bytes > self._capacity:
            return False
        with self._lock:
            existing = self._blocks.get(block.block_id)
            if existing:
                self._used_bytes -= existing.size_bytes
            block.tier = 2
            self._blocks[block.block_id] = block
            self._used_bytes += block.size_bytes
            return True

    def remove(self, block_id: str) -> Optional[Block]:
        """Remove and return block, or None if absent."""
        with self._lock:
            block = self._blocks.pop(block_id, None)
            if block:
                self._used_bytes -= block.size_bytes
            return block

    def has_space(self, size_bytes: int) -> bool:
        with self._lock:
            return self._used_bytes + size_bytes <= self._capacity

    def all_blocks(self) -> list[Block]:
        """Snapshot of all blocks (shallow copy of list)."""
        with self._lock:
            return list(self._blocks.values())

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    @property
    def utilization(self) -> float:
        if self._capacity == 0:
            return 0.0
        return self._used_bytes / self._capacity
