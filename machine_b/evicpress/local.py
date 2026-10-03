# SPDX-License-Identifier: MIT
"""In-process access to the existing manager; no server or RPC dependencies."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, TypeVar

from .config import EvicPressConfig, load_config
from .manager import EvicPressManager

T = TypeVar("T")


@dataclass(frozen=True)
class LookupResult:
    """Presence and policy tier; Tier 1 is a mirror ledger, not a data source."""

    hit: bool
    tier: int


@dataclass(frozen=True)
class StoreResult:
    """Canonical admission, mirror placement, and actual backing precision."""

    success: bool
    tier: int
    quant_level: str


@dataclass(frozen=True)
class RetrieveResult:
    """Restored FP16 payload with its source tier and backing precision."""

    data: bytes
    tier: int
    quant_level: str


class DirectoryOwner:
    """OS-held exclusive writer lock, released even when a process exits."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(
            directory / ".evicpress-local.lock", os.O_CREAT | os.O_RDWR, 0o600
        )
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(self.fd).st_size == 0:
                    os.write(self.fd, b"0")
                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self.fd)
            self.fd = -1
            raise RuntimeError(
                f"EvicPress cache directory already owned: {directory}"
            ) from exc

    def close(self) -> None:
        """Release the OS lock; the lock file intentionally remains in place."""
        if self.fd >= 0:
            if os.name == "nt":
                import msvcrt

                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            os.close(self.fd)
            self.fd = -1


class LocalEvicPressClient:
    """Own one manager and its background prefetch worker in the calling process.

    Args:
        config: Existing EvicPress configuration or explicit YAML path. Disk
            data_dir must be absolute. Server/dashboard fields are not started.
        cache_identity: Optional model/revision/layout fingerprint persisted in
            the cache directory to reject incompatible reuse.

    Raises:
        ValueError: Invalid capacities, directory, or incompatible identity.
        RuntimeError: Another client owns the directory or prefetch fails.

    Operations are synchronous direct calls, except prefetch hints. The OS
    writer lock and lifecycle lock prevent duplicate owners and shutdown races.
    Never construct this client before forking a GPU worker.
    """

    def __init__(
        self,
        config: EvicPressConfig | str | os.PathLike[str],
        *,
        cache_identity: Mapping[str, Any] | None = None,
    ) -> None:
        cfg = deepcopy(
            config if isinstance(config, EvicPressConfig) else load_config(str(config))
        )
        directory = Path(cfg.tier3.data_dir)
        if not directory.is_absolute():
            raise ValueError("local EvicPress data_dir must be absolute")
        for tier in (cfg.tier1, cfg.tier2, cfg.tier3):
            if tier.capacity_bytes < 0 or tier.bandwidth_bytes_per_sec <= 0:
                raise ValueError(
                    "tier capacity must be nonnegative and bandwidth positive"
                )
        if cfg.prefetch_max_queue <= 0:
            raise ValueError("prefetch_max_queue must be positive")
        cfg.tier3.data_dir = str(directory.resolve())
        self._pid = os.getpid()
        self._lifecycle_lock = threading.RLock()
        self._closed = False
        self._closing = False
        self._background_error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._prefetch_task: asyncio.Task | None = None
        self._ready = threading.Event()
        self._owner = DirectoryOwner(directory)
        try:
            if cache_identity is not None:
                identity = dict(cache_identity)
                path = directory / "local_identity.json"
                if path.exists():
                    if json.loads(path.read_text(encoding="utf-8")) != identity:
                        raise ValueError(
                            "local cache identity mismatch; use a fresh data_dir"
                        )
                else:
                    # Adopting untagged legacy blobs could mix model revisions.
                    if any(
                        (directory / name).exists()
                        for name in ("data", "meta", "head_groups")
                    ):
                        raise ValueError(
                            "unidentified cache directory; use a fresh data_dir"
                        )
                    path.write_text(
                        json.dumps(identity, sort_keys=True), encoding="utf-8"
                    )
            self.manager = EvicPressManager(cfg)
            if cfg.prefetch_enabled:
                self._thread = threading.Thread(
                    target=self._run_prefetch,
                    name="evicpress-local-prefetch",
                    daemon=True,
                )
                self._thread.start()
                if not self._ready.wait(10):
                    raise RuntimeError("local prefetch worker failed to start")
                self._check_open()
        except BaseException:
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(10)
            self._owner.close()
            raise

    def lookup(self, block_id: str) -> LookupResult:
        """Return presence/policy placement; no demand access is recorded."""
        return LookupResult(*self._call(self.manager.lookup, block_id))

    def store(
        self, block_id: str, data: bytes, quality_score: float = 1.0
    ) -> StoreResult:
        """Admit canonical bytes using the existing utility and codec policy."""
        return StoreResult(
            *self._call(self.manager.store, block_id, data, quality_score)
        )

    def retrieve(self, block_id: str) -> RetrieveResult | None:
        """Restore from RAM/disk, or return None on a true cache miss."""
        result = self._call(self.manager.retrieve, block_id)
        return RetrieveResult(*result) if result is not None else None

    def delete(self, block_id: str) -> bool:
        """Delete unleased backing using the existing manager's semantics."""
        return self._call(self.manager.delete, block_id)

    def store_heads(
        self, descriptor_json: str, payloads: list[tuple[str, bytes]], client_id: str
    ) -> dict:
        """Atomically publish a complete head group; return its native result."""
        return self._call(
            self.manager.head_groups.store, descriptor_json, payloads, client_id
        )

    def read_heads(
        self,
        items: list[tuple[str, set[str], str]],
        client_id: str,
        *,
        retrieve: bool,
        pin: bool = False,
    ) -> list[dict]:
        """Lookup or retrieve complete groups, preserving read leases."""
        return self._call(
            self.manager.head_groups.read, items, client_id, retrieve=retrieve, pin=pin
        )

    def delete_heads(self, parents: list[str], force: bool = True) -> bool:
        """Delete logical groups; force invalidates their read leases."""
        return self._call(self.manager.head_groups.delete, parents, force)

    def release_heads(self, lease_ids: list[str], client_id: str) -> None:
        """Release only leases owned by the named client."""
        self._call(self.manager.head_groups.release, lease_ids, client_id)

    def sync_head_tier1(self, client_id: str, inventory: dict[str, int]) -> list[str]:
        """Reconcile physical head mirrors and return revoked identities."""
        return self._call(self.manager.head_groups.sync_tier1, client_id, inventory)

    def get_state(self) -> dict:
        """Return a consistent state snapshot with transport-neutral counters."""
        state = self._call(self.manager.get_state)
        state["transport"] = "local"
        return state

    def prefetch(self, block_ids: list[str]) -> Future[int]:
        """Submit a bounded hint on its queue's owning thread, without waiting.

        Returns a Future containing the number admitted; disabled prefetch
        returns an already-completed zero. Demand reads do not wait for hints.
        """
        result: Future[int] = Future()
        with self._lifecycle_lock:
            self._check_open()
            if self._loop is None:
                result.set_result(0)
                return result
            ids = list(block_ids)

            def enqueue() -> None:
                if not result.set_running_or_notify_cancel():
                    return
                try:
                    result.set_result(self.manager.queue_prefetch(ids))
                except BaseException as exc:
                    result.set_exception(exc)

            self._loop.call_soon_threadsafe(enqueue)
        return result

    def drain_prefetch(self, timeout: float = 30) -> None:
        """Wait for prior hints, outside measured demand paths; raise on failure."""
        with self._lifecycle_lock:
            self._check_open()
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(
                    self._join_prefetch(), self._loop
                ).result(timeout)
            self._check_open()

    def close(self, timeout: float = 30) -> None:
        """Drain prefetch, release leases and directory ownership; idempotent.

        Raises TimeoutError if disk I/O prevents safe worker shutdown. In that
        case ownership remains held; callers must not start another manager.
        """
        with self._lifecycle_lock:
            if os.getpid() != self._pid:
                raise RuntimeError(
                    "local client inherited across fork; create it in the worker"
                )
            if self._closed:
                return
            self._closing = True
            if self._loop is not None and self._loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._stop_prefetch(), self._loop
                )
                future.result(timeout)
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout)
                if self._thread.is_alive():
                    raise TimeoutError(
                        "prefetch shutdown incomplete; directory still owned"
                    )
            with self.manager.operation_lock:
                self.manager.head_groups.leases.clear()
                self.manager.prefetch_queue = None
            self._owner.close()
            self._closed = True

    def __enter__(self) -> LocalEvicPressClient:
        self._check_open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _call(self, method: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        with self._lifecycle_lock:
            self._check_open()
            with self.manager.operation_lock:
                return method(*args, **kwargs)

    def _check_open(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError(
                "local client inherited across fork; create it in the worker"
            )
        if self._closed or self._closing:
            raise RuntimeError("local EvicPress client is closed")
        if self._background_error is not None:
            raise RuntimeError(
                "local prefetch worker failed"
            ) from self._background_error

    async def _join_prefetch(self) -> None:
        if self.manager.prefetch_queue is not None:
            await self.manager.prefetch_queue.join()

    async def _stop_prefetch(self) -> None:
        if self._background_error is None:
            await self._join_prefetch()
        if self._prefetch_task is not None:
            self._prefetch_task.cancel()
            await asyncio.gather(self._prefetch_task, return_exceptions=True)

    def _run_prefetch(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self.manager.prefetch_queue = asyncio.Queue(
                maxsize=self.manager.config.prefetch_max_queue
            )
            self._prefetch_task = loop.create_task(self.manager.prefetch_worker())

            def completed(task: asyncio.Task) -> None:
                if not task.cancelled():
                    self._background_error = task.exception() or RuntimeError(
                        "prefetch worker exited unexpectedly"
                    )

            self._prefetch_task.add_done_callback(completed)
            self._ready.set()
            loop.run_forever()
        except BaseException as exc:
            self._background_error = exc
            self._ready.set()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
