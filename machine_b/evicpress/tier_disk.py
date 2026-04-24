"""
Tier 3 — Disk (NVMe SSD) storage.

Each block is stored as two files:
  data/<block_id>.bin   — raw KV bytes
  meta/<block_id>.json  — lightweight metadata (no data in RAM)

Only metadata is kept in memory; data is read from disk on demand.
This means Tier 3 can hold far more blocks than Tier 2 without bloating RAM.
"""

import hashlib
import json
import os
import threading
from typing import Optional

from .block import Block


class DiskTier:
    def __init__(self, capacity_bytes: int, data_dir: str) -> None:
        self._capacity = capacity_bytes
        self._data_dir = data_dir
        self._meta_dir = os.path.join(data_dir, "meta")
        self._blob_dir = os.path.join(data_dir, "data")
        self._index: dict[str, dict] = {}   # block_id -> metadata dict
        self._used_bytes: int = 0
        self._lock = threading.RLock()
        self._init_dirs()
        self._load_existing()

    # ------------------------------------------------------------------ #
    #  Startup                                                             #
    # ------------------------------------------------------------------ #

    def _init_dirs(self) -> None:
        os.makedirs(self._meta_dir, exist_ok=True)
        os.makedirs(self._blob_dir, exist_ok=True)

    def _load_existing(self) -> None:
        """Reconstruct the in-memory index from metadata files on disk."""
        valid_hashes: set[str] = set()
        for fname in os.listdir(self._meta_dir):
            if not fname.endswith(".json"):
                continue
            file_hash = fname[:-5]  # filename IS the md5 hash, not the block_id
            meta_path = os.path.join(self._meta_dir, fname)
            blob_path = os.path.join(self._blob_dir, file_hash + ".bin")
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if os.path.exists(blob_path):
                    self._index[meta["block_id"]] = meta  # key by real block_id
                    self._used_bytes += meta["size_bytes"]
                    valid_hashes.add(file_hash)
                else:
                    os.remove(meta_path)
            except Exception:
                pass  # Corrupted meta; skip silently
        # Remove orphaned .bin files (blob without matching meta)
        for fname in os.listdir(self._blob_dir):
            if not fname.endswith(".bin"):
                continue
            if fname[:-4] not in valid_hashes:
                try:
                    os.remove(os.path.join(self._blob_dir, fname))
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    #  Path helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_name(block_id: str) -> str:
        """Hash block_id to a safe filename — block_ids can contain '/' from model names."""
        return hashlib.md5(block_id.encode()).hexdigest()

    def _meta_path(self, block_id: str) -> str:
        return os.path.join(self._meta_dir, f"{self._safe_name(block_id)}.json")

    def _blob_path(self, block_id: str) -> str:
        return os.path.join(self._blob_dir, f"{self._safe_name(block_id)}.bin")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def contains(self, block_id: str) -> bool:
        with self._lock:
            return block_id in self._index

    def get(self, block_id: str) -> Optional[Block]:
        """Read block from disk. Updates access metadata. None if absent."""
        with self._lock:
            if block_id not in self._index:
                return None
            blob_path = self._blob_path(block_id)
            if not os.path.exists(blob_path):
                # Blob disappeared (manual deletion etc.)
                self._used_bytes -= self._index.pop(block_id, {}).get("size_bytes", 0)
                return None
            with open(blob_path, "rb") as f:
                data = f.read()
            meta = self._index[block_id]
            block = Block(
                block_id=block_id,
                data=data,
                tier=3,
                quality_score=meta["quality_score"],
                quant_level=meta.get("quant_level", "fp16"),
                access_count=meta["access_count"],
                last_access=meta["last_access"],
                created_at=meta["created_at"],
            )
            block.touch()
            self._write_meta(block)   # Persist updated access count
            return block

    def put(self, block: Block) -> bool:
        """
        Write block to disk. Returns False if block alone exceeds capacity.
        Caller must free space first if needed.
        """
        if block.size_bytes > self._capacity:
            return False
        with self._lock:
            old_size = self._index.get(block.block_id, {}).get("size_bytes", 0)
            block.tier = 3
            with open(self._blob_path(block.block_id), "wb") as f:
                f.write(block.data)
            self._write_meta(block)
            self._used_bytes -= old_size
            self._used_bytes += block.size_bytes
            return True

    def remove(self, block_id: str) -> bool:
        """Delete block from disk. Returns True if it existed."""
        with self._lock:
            if block_id not in self._index:
                return False
            meta = self._index.pop(block_id)
            self._used_bytes -= meta["size_bytes"]
            for path in (self._blob_path(block_id), self._meta_path(block_id)):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
            return True

    def has_space(self, size_bytes: int) -> bool:
        with self._lock:
            return self._used_bytes + size_bytes <= self._capacity

    def all_metas(self) -> list[dict]:
        """Snapshot of metadata for all disk blocks (no data loaded)."""
        with self._lock:
            return list(self._index.values())

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
        return len(self._index)

    @property
    def utilization(self) -> float:
        if self._capacity == 0:
            return 0.0
        return self._used_bytes / self._capacity

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _write_meta(self, block: Block) -> None:
        meta = block.to_meta()
        self._index[block.block_id] = meta
        with open(self._meta_path(block.block_id), "w") as f:
            json.dump(meta, f)
