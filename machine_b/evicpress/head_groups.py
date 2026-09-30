"""Per-head groups: immutable descriptors, atomic publication and read leases.

The logical-segment/physical-storage split follows RedKnot's SegPaged design.
This is a tiered KV-only adapter, not RedKnot's attention-output reuse plugin.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import pickle
from pathlib import Path
import time
from uuid import uuid4

import torch

from .block import Block
from .quantize import quantize, dequantize

SCHEMA_VERSION = 1
LEASE_SECONDS = 60.0


def segment_id(parent: str, layer: int, head: int, tokens: int) -> str:
    """Stable identity independent of the CPU/disk allocation holding a head."""
    raw = json.dumps([parent, layer, head, tokens], separators=(",", ":"))
    return "head-v1:" + hashlib.sha256(raw.encode()).hexdigest()


def validate_descriptor(raw: str) -> dict:
    """Accept only complete FP16 positions; sparse retention needs a kernel."""
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("head descriptor exceeds 2 MiB")
    d = json.loads(raw)
    if (not isinstance(d, dict) or type(d.get("schema_version")) is not int
            or d["schema_version"] != SCHEMA_VERSION):
        raise ValueError("unsupported head descriptor")
    for field in ("parent_id", "namespace"):
        if not isinstance(d.get(field), str) or not 0 < len(d[field]) <= 4096:
            raise ValueError(f"invalid {field}")
    shape = d.get("shape")
    if (not isinstance(shape, list) or len(shape) != 4 or
            any(type(n) is not int or n <= 0 for n in shape) or shape[0] != 2):
        raise ValueError("expected [2,layers,tokens,KV_heads*head_dim]")
    _, layers, tokens, hidden = shape
    heads, dim = d.get("num_kv_heads"), d.get("head_dim")
    if (type(heads) is not int or type(dim) is not int or heads <= 0 or dim <= 0
            or hidden != heads * dim or layers * heads > 4096):
        raise ValueError("invalid head geometry")
    if d.get("dtype") != "torch.float16" or d.get("format") != "KV_2LTD":
        raise ValueError("head mode currently requires FP16 KV_2LTD")
    if d.get("positions") != {"start": 0, "count": tokens}:
        raise ValueError("all original positions must be retained in head mode")
    expected = [{"block_id": segment_id(d["parent_id"], l, h, tokens),
                 "layer": l, "kv_head": h}
                for l in range(layers) for h in range(heads)]
    if d.get("heads") != expected:
        raise ValueError("missing, duplicated or inconsistent head coordinates")
    return d


def validate_tensor(data: bytes, descriptor: dict) -> torch.Tensor:
    """Check decoded wire geometry before publishing or serving a head."""
    tensor = torch.load(io.BytesIO(data), weights_only=True)
    if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float16
            or tuple(tensor.shape) != (2, descriptor["shape"][2], descriptor["head_dim"])):
        raise ValueError("head tensor shape/dtype disagrees with descriptor")
    return tensor


class HeadGroupStore:
    """Coordinate B's existing byte tiers without changing the legacy RPCs."""

    def __init__(self, manager) -> None:
        self.manager = manager
        self.directory = Path(manager.config.tier3.data_dir) / "head_groups"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.groups: dict[str, dict] = {}
        # lease -> (client, protected IDs, monotonic expiry); not persisted.
        self.leases: dict[str, tuple[str, set[str], float]] = {}
        self.owners: dict[str, str] = {}
        self.counters = {"parent_hits": 0, "parent_misses": 0,
                         "head_rpc_calls": 0, "head_wire_bytes": 0}
        for path in self.directory.glob("*.json"):
            try:
                d = validate_descriptor(path.read_text(encoding="utf-8"))
                self.groups[d["parent_id"]] = d
            except (ValueError, OSError):
                manager.log_operation("HEAD_INVALID", path.name, "descriptor ignored")

    @property
    def parent_requests(self) -> int:
        """Utility frequency is normalized by parent reads, not head/RPC count."""
        return self.counters["parent_hits"] + self.counters["parent_misses"]

    def is_pinned(self, block_id: str) -> bool:
        """Check live canonical-read reservations under the manager's lock."""
        now = time.monotonic()
        for lease in list(self.leases):
            if self.leases[lease][2] <= now:
                del self.leases[lease]
        return any(block_id in ids for _, ids, _ in self.leases.values())

    def release(self, lease_ids: list[str], client: str) -> None:
        """Release only this client's leases; expiry covers disconnected clients."""
        with self.manager.operation_lock:
            for lid in lease_ids:
                if lid in self.leases and self.leases[lid][0] == client:
                    del self.leases[lid]

    def store(self, raw: str, payloads: list[tuple[str, bytes]], client: str) -> dict:
        """Prepare all heads, reserve space, then publish the complete parent."""
        from .manager import _compute_quality, _new_block_utility, _placement_decision

        d = validate_descriptor(raw)
        if not client:
            raise ValueError("client_id is required")
        data = dict(payloads)
        ids = {h["block_id"] for h in d["heads"]}
        if len(data) != len(payloads) or set(data) != ids:
            raise ValueError("payload IDs do not match the descriptor")
        if sum(len(v) for v in data.values()) > self.manager.config.server_max_message_bytes:
            raise ValueError("head group exceeds configured message limit")
        prepared = []
        with self.manager.operation_lock:
            self.counters["head_rpc_calls"] += 1
            self.counters["head_wire_bytes"] += sum(len(v) for v in data.values())
            previous = self.groups.get(d["parent_id"])
            if previous is not None and previous != d:
                raise ValueError("parent descriptor cannot be changed in place")
            if previous is not None and all(self.manager.tier3.contains(i) for i in ids):
                return self.lookup_one(d["parent_id"], set(), client, False)
            for h in d["heads"]:
                original = data[h["block_id"]]
                tensor = validate_tensor(original, d)
                quality = _compute_quality(original)
                utility = _new_block_utility(
                    len(original), quality, self.manager.config.alpha,
                    self.manager.config.tier2.bandwidth_bytes_per_sec,
                    self.parent_requests)
                tiers, level = _placement_decision(utility, self.manager.config.placement)
                if not self.manager.config.quantization.enabled:
                    level = "fp16"
                payload = quantize(original, level)
                old = self.manager.tier3.get_meta(h["block_id"]) or {}
                block = Block(h["block_id"], payload, 3, quality, level,
                              access_count=old.get("access_count", 0),
                              parent_id=d["parent_id"], layer=h["layer"],
                              kv_head=h["kv_head"], token_count=d["shape"][2])
                prepared.append((block, tiers, utility, tensor.numel() * tensor.element_size()))
            required = sum(b.size_bytes for b, _, _, _ in prepared)
            old_bytes = sum((self.manager.tier3.get_meta(i) or {}).get("size_bytes", 0)
                            for i in ids)
            if (required > self.manager.tier3.capacity_bytes or
                    any(self.is_pinned(i) for i in ids) or
                    not self.manager.reserve_head_disk(max(0, required - old_bytes), ids)):
                return {"parent_id": d["parent_id"], "found": False,
                        "message": "canonical capacity or lease prevents complete store"}
            written = []
            try:
                # Repair an incomplete group from scratch. Replacing heads of
                # different sizes in-place can transiently exceed the byte cap.
                # No reader observes this unpublished phase (shared lock).
                for bid in ids:
                    self.manager.delete(bid)
                for block, _, _, _ in prepared:
                    if not self.manager.tier3.put(block):
                        raise OSError("canonical head write rejected")
                    written.append(block.block_id)
                path = self._path(d["parent_id"])
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
                temporary.replace(path)  # Publication is last; no partial parent hit.
                self.groups[d["parent_id"]] = d
            except OSError:
                self.groups.pop(d["parent_id"], None)
                self._path(d["parent_id"]).unlink(missing_ok=True)
                for bid in written:
                    self.manager.delete(bid)
                return {"parent_id": d["parent_id"], "found": False,
                        "message": "canonical write failed; incomplete group discarded"}
            for block, tiers, utility, fp16_bytes in prepared:
                if 2 in tiers:
                    self.manager.mirror_head(block)
                if 1 in tiers and self.manager.tier1.admit(
                        block.block_id, fp16_bytes, utility, block.quant_level):
                    self.owners[block.block_id] = client
                    self.manager.stats.promote_tier1()
            self.manager.log_operation("HEAD_STORE", d["parent_id"], f"{len(ids)} heads")
            return self.lookup_one(d["parent_id"], set(), client, False)

    def lookup_one(self, parent: str, local: set[str], client: str, pin: bool) -> dict:
        """Local inventory, not the Tier 1 ledger, establishes a local hit."""
        d = self.groups.get(parent)
        if d is None:
            return {"parent_id": parent, "found": False}
        expected = {h["block_id"] for h in d["heads"]}
        if not local <= expected:
            raise ValueError("local IDs belong to a different parent")
        status = []
        for h in d["heads"]:
            bid = h["block_id"]
            meta = self.manager.tier3.get_meta(bid)
            tier = (1 if bid in local else 2 if self.manager.tier2.contains(bid)
                    else 3 if meta else 0)
            status.append({"block_id": bid, "found": tier != 0, "tier": tier,
                           "quant_level": (meta or {}).get("quant_level", "fp16"),
                           "cache_in_tier1": self.owners.get(bid) == client
                           and self.manager.tier1.contains(bid)})
        found = all(h["found"] for h in status)
        lid = ""
        if found and pin:
            if len(self.leases) >= 4096:
                self.is_pinned("")  # Expire old reservations before refusing.
            if len(self.leases) >= 4096:
                return {"parent_id": parent, "found": False, "message": "lease limit"}
            lid = uuid4().hex
            self.leases[lid] = (client, expected, time.monotonic() + LEASE_SECONDS)
        return {"parent_id": parent, "found": found,
                "descriptor_json": json.dumps(d, separators=(",", ":")),
                "heads": status, "lease_id": lid}

    def read(self, items: list[tuple[str, set[str], str]], client: str,
             *, retrieve: bool, pin: bool = False) -> list[dict]:
        """One logical demand read per parent, irrespective of its head count."""
        if not client or len(items) > 64:
            raise ValueError("client_id required; maximum 64 parents per batch")
        with self.manager.operation_lock:
            self.counters["head_rpc_calls"] += 1
            if retrieve:
                estimated = sum(
                    4 * d["shape"][2] * d["head_dim"] * max(0, len(d["heads"]) - len(local))
                    + 4096 * len(d["heads"])
                    for parent, local, _ in items if (d := self.groups.get(parent)))
                if estimated > self.manager.config.server_max_message_bytes * 3 // 4:
                    raise ValueError("retrieval batch exceeds safe message limit")
            groups = []
            for parent, local, lid in items:
                group = self.lookup_one(parent, local, client, pin and not retrieve)
                if retrieve:
                    if group["found"]:
                        try:
                            for h in group["heads"]:
                                bid = h["block_id"]
                                if bid in local:
                                    self.manager.tier3.record_access(bid)
                                    self.manager.stats.hit(1)
                                    continue
                                block = self.manager.tier2.get(bid)
                                tier = 2
                                if block is None:
                                    block = self.manager.tier3.get(bid)
                                    tier = 3
                                if tier == 2:
                                    # Keep the canonical frequency current even on RAM hits.
                                    self.manager.tier3.record_access(bid)
                                if block is None:
                                    raise OSError("head disappeared during retrieval")
                                h["data"] = dequantize(block.data, block.quant_level)
                                validate_tensor(h["data"], self.groups[parent])
                                h["tier"] = tier
                                h["quant_level"] = block.quant_level
                                self.manager.stats.hit(tier)
                                self.counters["head_wire_bytes"] += len(h["data"])
                                if tier == 3 and self.manager.tier2.has_space(block.size_bytes):
                                    self.manager.tier2.put(replace(block, tier=2))
                        except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError, TypeError, IndexError):
                            # Corrupt backing must not stay an idempotent "hit"
                            # forever; force invalidation permits recomputed store.
                            self.delete([parent], force=True)
                            group = {"parent_id": parent, "found": False,
                                     "message": "missing or corrupt head"}
                    key = "parent_hits" if group["found"] else "parent_misses"
                    self.counters[key] += 1
                    if not group["found"]:
                        self.manager.stats.miss()
                    self.release([lid], client)
                groups.append(group)
            return groups

    def delete(self, parents: list[str], force: bool) -> bool:
        """Delete complete logical identities; protected groups require force."""
        with self.manager.operation_lock:
            removed = False
            for parent in parents:
                d = self.groups.get(parent)
                if d is None:
                    continue
                ids = {h["block_id"] for h in d["heads"]}
                if not force and any(self.is_pinned(i) for i in ids):
                    continue
                if force:
                    for lid, (_, pinned, _) in list(self.leases.items()):
                        if ids & pinned:
                            del self.leases[lid]
                for bid in ids:
                    self.manager.delete(bid)
                    self.owners.pop(bid, None)
                self.groups.pop(parent, None)
                self._path(parent).unlink(missing_ok=True)
                removed = True
            return removed

    def sync_tier1(self, client: str, inventory: dict[str, int]) -> list[str]:
        """Reconcile actual cache ownership; never delete canonical KV here."""
        if not client or len(inventory) > 65536:
            raise ValueError("invalid Tier 1 inventory")
        with self.manager.operation_lock:
            for bid, owner in list(self.owners.items()):
                if owner == client and bid not in inventory:
                    self.manager.tier1.remove(bid)
                    del self.owners[bid]
            revoked = []
            for bid, size in inventory.items():
                meta = self.manager.tier3.get_meta(bid)
                descriptor = self.groups.get((meta or {}).get("parent_id"), {})
                minimum = 4 * (meta or {}).get("token_count", 0) * descriptor.get("head_dim", 1)
                if (meta is None or self.owners.get(bid) != client or not self.manager.tier1.contains(bid)
                        or type(size) is not int or size < minimum):
                    revoked.append(bid)
                    continue
                utility = self.manager.head_utility(meta)
                # Ledger updates otherwise bypass admission's capacity check.
                self.manager.tier1.remove(bid)
                if not self.manager.tier1.admit(bid, size, utility, meta["quant_level"]):
                    revoked.append(bid)
            # Admission can revoke another entry while adjusting physical sizes.
            return list(set(revoked) | {bid for bid in inventory
                                       if not self.manager.tier1.contains(bid)})

    def forget_empty_parent(self, parent: str) -> None:
        """Reclaim descriptors when canonical eviction removes every child."""
        d = self.groups.get(parent)
        if d and not any(self.manager.tier3.contains(h["block_id"]) for h in d["heads"]):
            self.groups.pop(parent, None)
            self._path(parent).unlink(missing_ok=True)
            for h in d["heads"]:
                self.owners.pop(h["block_id"], None)

    def _path(self, parent: str) -> Path:
        return self.directory / (hashlib.sha256(parent.encode()).hexdigest() + ".json")
