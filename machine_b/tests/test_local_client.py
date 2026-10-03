# SPDX-License-Identifier: MIT
"""GPU-free checks of the installed core library and its local runtime."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import torch

from evicpress.config import (
    EvicPressConfig,
    PlacementBand,
    PlacementConfig,
    QuantConfig,
    Tier1Config,
    Tier2Config,
    Tier3Config,
)
from evicpress.head_groups import segment_id
from evicpress.local import LocalEvicPressClient


def config(directory, *, level="fp16", prefetch=False, ram=0):
    return EvicPressConfig(
        alpha=1.0,
        tier1=Tier1Config(0, 1e9),
        tier2=Tier2Config(ram, 1e9),
        tier3=Tier3Config(1024 * 1024, 1e9, str(Path(directory) / "cache")),
        prefetch_enabled=prefetch,
        prefetch_max_queue=4,
        server_host="127.0.0.1",
        server_port=50051,
        server_max_message_bytes=4 * 1024 * 1024,
        dashboard_host="127.0.0.1",
        dashboard_port=8080,
        max_recent_ops=30,
        quantization=QuantConfig(level != "fp16", {}),
        placement=PlacementConfig([PlacementBand("test", 0, [3], level)]),
    )


def tensor_payload():
    tensor = torch.linspace(-1, 1, 80).reshape(2, 1, 5, 8).half()
    stream = io.BytesIO()
    torch.save(tensor, stream)
    return tensor, stream.getvalue()


class LocalClientTest(unittest.TestCase):
    def test_benchmark_controls_reject_concurrent_direct_work(self):
        with tempfile.TemporaryDirectory() as directory:
            with LocalEvicPressClient(config(directory), benchmark_enabled=True) as client:
                entered, release = threading.Event(), threading.Event()
                original = client.manager.store

                def held_store(*args):
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError("test work did not release")
                    return original(*args)

                with mock.patch.object(client.manager, "store", side_effect=held_store):
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(client.store, "block", b"bytes")
                        self.assertTrue(entered.wait(5))
                        try:
                            for operation in (client.benchmark_reset, client.benchmark_release_tier1,
                                              client.benchmark_snapshot):
                                with self.assertRaises(RuntimeError):
                                    operation()
                        finally:
                            release.set()
                        self.assertTrue(future.result(5).success)

    def test_benchmark_controls_opt_in_preserve_backing_and_reset_history(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory, prefetch=True, ram=1024 * 1024)
            with LocalEvicPressClient(cfg) as client:
                with self.assertRaises(RuntimeError):
                    client.benchmark_reset()
            with LocalEvicPressClient(cfg, benchmark_enabled=True) as client:
                _, payload = tensor_payload()
                client.store("block", payload)
                client.manager.tier1.admit("block", len(payload), 1, "fp16")
                client.prefetch(["block"]).result(10)
                client.drain_prefetch()
                before = client.benchmark_snapshot()
                self.assertEqual(before["canonical_blocks"], 1)
                self.assertEqual(before["disk_files"]["payload_file_bytes"], len(payload))
                self.assertGreater(before["disk_files"]["metadata_file_bytes"], 0)
                self.assertEqual(before["precision_distribution"]["fp16"]["blocks"], 1)
                client.benchmark_release_tier1()
                self.assertEqual(client.benchmark_snapshot()["tier1_ledger_bytes"], 0)
                self.assertEqual(client.retrieve("block").data, payload)
                self.assertGreater(client.get_state()["stats"]["total_hits"], 0)
                client.benchmark_reset()
                self.assertEqual(client.benchmark_snapshot()["canonical_blocks"], 0)
                self.assertEqual(client.get_state()["stats"]["total_hits"], 0)
                self.assertFalse(client.lookup("block").hit)

    def test_benchmark_rejects_lease_and_records_quantization_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with LocalEvicPressClient(config(directory, level="int4"),
                                      benchmark_enabled=True) as client:
                client.store("raw", b"not a tensor")
                state = client.benchmark_snapshot()
                self.assertEqual(state["quantization_fallbacks"], 1)
                self.assertEqual(list(state["precision_distribution"]), ["fp16"])
                client.manager.head_groups.leases["active"] = {}
                with self.assertRaises(RuntimeError):
                    client.benchmark_reset()
                with self.assertRaises(RuntimeError):
                    client.benchmark_release_tier1()
                self.assertTrue(client.lookup("raw").hit)
                client.manager.head_groups.leases.clear()

    def test_installable_library_does_not_import_rpc_or_server(self):
        code = (
            "import sys; from evicpress.local import LocalEvicPressClient; "
            "assert 'grpc' not in sys.modules; "
            "assert 'server.grpc_server' not in sys.modules; "
            "assert 'generated.evicpress_pb2' not in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", code], cwd=tempfile.gettempdir(), check=True
        )

    def test_fp16_disk_restart_ownership_and_closed_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            identity = {"model": "test", "revision": "pinned", "shape": [2, 1, 5, 8]}
            original, payload = tensor_payload()
            client = LocalEvicPressClient(cfg, cache_identity=identity)
            with self.assertRaises(RuntimeError):
                LocalEvicPressClient(cfg, cache_identity=identity)
            stored = client.store("block", payload)
            self.assertTrue(stored.success)
            self.assertEqual(stored.quant_level, "fp16")
            self.assertTrue(client.lookup("block").hit)
            fetched = client.retrieve("block")
            self.assertEqual((fetched.tier, fetched.quant_level), (3, "fp16"))
            self.assertTrue(
                torch.equal(
                    torch.load(io.BytesIO(fetched.data), weights_only=True), original
                )
            )
            client.close()
            client.close()
            with self.assertRaises(RuntimeError):
                client.lookup("block")
            with self.assertRaises(ValueError):
                LocalEvicPressClient(cfg, cache_identity={"model": "other"})
            with LocalEvicPressClient(cfg, cache_identity=identity) as restarted:
                self.assertIsNotNone(restarted.retrieve("block"))
                self.assertTrue(restarted.delete("block"))
                self.assertIsNone(restarted.retrieve("block"))
            self.assertTrue((Path(cfg.tier3.data_dir) / "local_identity.json").exists())

    def test_quantized_backing_reports_real_precision(self):
        for level, tolerance in (("int8", 0.01), ("int4", 0.08)):
            with self.subTest(level=level), tempfile.TemporaryDirectory() as directory:
                original, payload = tensor_payload()
                with LocalEvicPressClient(config(directory, level=level)) as client:
                    stored = client.store("quant", payload)
                    self.assertEqual(stored.quant_level, level)
                    fetched = client.retrieve("quant")
                    self.assertEqual(fetched.quant_level, level)
                    restored = torch.load(io.BytesIO(fetched.data), weights_only=True)
                    self.assertEqual(restored.dtype, torch.float16)
                    self.assertLessEqual(
                        (restored - original).abs().max().item(), tolerance
                    )

    def test_thread_safe_prefetch_and_clean_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory, prefetch=True, ram=1024 * 1024)
            client = LocalEvicPressClient(cfg)
            _, payload = tensor_payload()
            client.store("disk", payload)
            self.assertEqual(client.lookup("disk").tier, 3)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = list(pool.map(lambda _: client.prefetch(["disk"]), range(12)))
            self.assertGreater(sum(f.result(10) for f in futures), 0)
            client.drain_prefetch()
            self.assertEqual(client.retrieve("disk").tier, 2)
            self.assertEqual(client.get_state()["prefetch_queue_size"], 0)
            client.prefetch(["disk"])
            client.close()
            with LocalEvicPressClient(cfg) as restarted:
                self.assertTrue(restarted.lookup("disk").hit)

    def test_disabled_prefetch_rejection_and_invalid_config(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            cfg.tier3.capacity_bytes = 1
            with LocalEvicPressClient(cfg) as client:
                self.assertEqual(client.prefetch(["missing"]).result(), 0)
                self.assertFalse(client.store("too-big", b"1234", 0.5).success)
                self.assertFalse(client.lookup("too-big").hit)
            cfg.tier3.data_dir = "relative/cache"
            with self.assertRaises(ValueError):
                LocalEvicPressClient(cfg)

    def test_head_leases_completeness_and_zero_rpc_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            head_tensor = torch.linspace(-1, 1, 40).reshape(2, 5, 4).half()
            stream = io.BytesIO()
            torch.save(head_tensor, stream)
            descriptor = {
                "schema_version": 1,
                "parent_id": "parent",
                "namespace": "test",
                "shape": [2, 1, 5, 8],
                "num_kv_heads": 2,
                "head_dim": 4,
                "dtype": "torch.float16",
                "format": "KV_2LTD",
                "positions": {"start": 0, "count": 5},
                "heads": [
                    {
                        "block_id": segment_id("parent", 0, h, 5),
                        "layer": 0,
                        "kv_head": h,
                    }
                    for h in range(2)
                ],
            }
            heads = [(h["block_id"], stream.getvalue()) for h in descriptor["heads"]]
            with LocalEvicPressClient(cfg) as client:
                self.assertTrue(
                    client.store_heads(json.dumps(descriptor), heads, "client")["found"]
                )
                lookup = client.read_heads(
                    [("parent", set(), "")], "client", retrieve=False, pin=True
                )[0]
                self.assertFalse(client.delete_heads(["parent"], force=False))
                response = client.read_heads(
                    [("parent", set(), lookup["lease_id"])], "client", retrieve=True
                )[0]
                self.assertTrue(response["found"])
                self.assertEqual(len(response["heads"]), 2)
                self.assertEqual(client.get_state()["stats"]["head_rpc_calls"], 0)
                self.assertGreater(client.get_state()["stats"]["head_payload_bytes"], 0)
                client.sync_head_tier1("client", {})
                client.release_heads([lookup["lease_id"]], "client")
                self.assertTrue(client.delete_heads(["parent"]))
                self.assertFalse(
                    client.read_heads([("parent", set(), "")], "client", retrieve=True)[
                        0
                    ]["found"]
                )
                with self.assertRaises(ValueError):
                    client.store_heads(json.dumps(descriptor), heads[:1], "client")

    def test_forked_client_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with LocalEvicPressClient(config(directory)) as client:
                with mock.patch("evicpress.local.os.getpid", return_value=-1):
                    with self.assertRaises(RuntimeError):
                        client.lookup("block")

    def test_prefetch_failure_is_reported_and_owner_is_released_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory, prefetch=True, ram=1024 * 1024)
            client = LocalEvicPressClient(cfg)
            _, payload = tensor_payload()
            client.store("disk", payload)
            try:
                with mock.patch.object(
                    client.manager.tier3, "get", side_effect=OSError("disk failed")
                ):
                    client.prefetch(["disk"]).result(10)
                    with self.assertRaises(RuntimeError):
                        client.drain_prefetch()
            finally:
                client.close()
            with LocalEvicPressClient(cfg) as restarted:
                self.assertTrue(restarted.lookup("disk").hit)


if __name__ == "__main__":
    unittest.main()
