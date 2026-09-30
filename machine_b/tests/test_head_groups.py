"""CPU service checks; optional loopback integration uses Machine A's real client."""
import asyncio
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import grpc
import torch
import yaml

from evicpress.config import load_config
from evicpress.block import Block
from evicpress.head_groups import segment_id
from evicpress.manager import EvicPressManager
from generated import evicpress_pb2 as pb, evicpress_pb2_grpc as rpc
from server.grpc_server import EvicPressServicer


def make_manager(directory, *, quant=False):
    raw = yaml.safe_load((Path(__file__).parents[1] / "config/config.yaml").read_text())
    raw["tier3"]["data_dir"] = str(Path(directory) / "cache")
    raw["quantization"]["enabled"] = quant
    raw["tier2"]["capacity_bytes"] = 0  # Force retrieval from canonical disk.
    raw["placement"]["bands"] = [{"name": "test", "min_utility": 0,
                                    "tiers": [1, 3], "quant": "int4" if quant else "fp16"}]
    path = Path(directory) / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return EvicPressManager(load_config(str(path)))


def group(parent):
    tensor = torch.linspace(-1, 1, 2 * 5 * 4).reshape(2, 5, 4).half()
    descriptor = {"schema_version": 1, "parent_id": parent, "namespace": "test",
                  "shape": [2, 1, 5, 8], "num_kv_heads": 2, "head_dim": 4,
                  "dtype": "torch.float16", "format": "KV_2LTD",
                  "positions": {"start": 0, "count": 5},
                  "heads": [{"block_id": segment_id(parent, 0, h, 5),
                             "layer": 0, "kv_head": h} for h in range(2)]}
    stream = io.BytesIO()
    torch.save(tensor, stream)
    return json.dumps(descriptor), [(h["block_id"], stream.getvalue()) for h in descriptor["heads"]], tensor


class HeadGroupTest(unittest.TestCase):
    def test_atomic_lease_restart_and_quantization(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = make_manager(temporary)
            raw, payloads, tensor = group("first")
            self.assertTrue(manager.head_groups.store(raw, payloads, "client")["found"])
            before = manager.tier3.used_bytes
            manager.config.tier3.capacity_bytes = before
            # Restart applies the smaller configured canonical capacity.
            manager = EvicPressManager(manager.config)
            lookup = manager.head_groups.read([("first", set(), "")], "client", retrieve=False, pin=True)[0]
            raw2, payloads2, _ = group("second")
            self.assertFalse(manager.head_groups.store(raw2, payloads2, "client")["found"])
            self.assertEqual(manager.tier3.used_bytes, before)
            response = manager.head_groups.read([("first", set(), lookup["lease_id"])], "client", retrieve=True)[0]
            self.assertTrue(response["found"])
            restored = torch.load(io.BytesIO(response["heads"][0]["data"]), weights_only=True)
            self.assertTrue(torch.equal(restored, tensor))
            self.assertTrue(manager.head_groups.store(raw2, payloads2, "client")["found"])
            self.assertFalse(manager.head_groups.read([("first", set(), "")], "client", retrieve=False)[0]["found"])
            with self.assertRaises(ValueError):
                manager.head_groups.store(raw2, payloads2[:1], "client")
            missing = payloads2[0][0]
            manager.delete(missing)
            self.assertFalse(manager.head_groups.read([("second", set(), "")], "client", retrieve=True)[0]["found"])

        with tempfile.TemporaryDirectory() as temporary:
            manager = make_manager(temporary, quant=True)
            raw, payloads, original = group("quantized")
            self.assertTrue(manager.head_groups.store(raw, payloads, "client")["found"])
            response = manager.head_groups.read([("quantized", set(), "")], "client", retrieve=True)[0]
            self.assertEqual(response["heads"][0]["quant_level"], "int4")
            result = torch.load(io.BytesIO(response["heads"][0]["data"]), weights_only=True)
            self.assertLessEqual((result - original).abs().max().item(), 1 / 14 + .001)
            bad = Block(payloads[0][0], b"corrupt", 3, .5, "int4")
            with mock.patch.object(manager.tier3, "get", return_value=bad):
                failed = manager.head_groups.read([("quantized", set(), "")], "client", retrieve=True)[0]
                self.assertFalse(failed["found"])
            # A subsequent recomputation can repair the invalidated identity.
            self.assertTrue(manager.head_groups.store(raw, payloads, "client")["found"])

    def test_real_grpc_client_round_trip(self):
        a = Path(__file__).parents[3] / "distributed-tiered-kv-cache"
        if not (a / "tests/test_head_segments.py").exists():
            self.skipTest("Clone Machine A alongside evicpress-core for paired integration")
        package = ModuleType("paired_head_backend")
        package.__path__ = [str(a / "LMCache/lmcache/v1/storage_backend")]
        sys.modules[package.__name__] = package
        HeadClient = importlib.import_module("paired_head_backend.head_client").HeadClient
        spec = importlib.util.spec_from_file_location("paired_pool", a / "tests/test_head_segments.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        async def check():
            with tempfile.TemporaryDirectory() as temporary:
                manager = make_manager(temporary)
                server = grpc.aio.server()
                rpc.add_EvicPressServiceServicer_to_server(EvicPressServicer(manager), server)
                port = server.add_insecure_port("127.0.0.1:0")
                await server.start()
                def exercise():
                    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
                    pool = module.Pool(4096)
                    client = HeadClient(rpc.EvicPressServiceStub(channel), "test", 1, 2, 4,
                                        256, pool.allocate, 256, 4 * 1024 * 1024)
                    tensor = torch.arange(80).reshape(2, 1, 5, 8).half()
                    self.assertTrue(client.store_head_segment("parent", tensor))
                    self.assertEqual(client.contains_many(["parent"], pin=True), 1)
                    obj = client.gather_segment(["parent"])[0]
                    self.assertTrue(torch.equal(obj.get_tensor(0), tensor))
                    obj.ref_count_down()
                    # Cold-client reconstruction crosses actual protobuf/gRPC/disk.
                    client.close()
                    self.assertEqual(pool.used, 0)
                    client = HeadClient(rpc.EvicPressServiceStub(channel), "test", 1, 2, 4,
                                        256, pool.allocate, 0, 4 * 1024 * 1024)
                    self.assertEqual(client.contains_many(["parent", "absent"], pin=True), 1)
                    obj = client.gather_segment(["parent"])[0]
                    self.assertTrue(torch.equal(obj.get_tensor(0), tensor))
                    obj.ref_count_down()
                    bid = manager.head_groups.groups[client.parent_id("parent")]["heads"][0]["block_id"]
                    manager.delete(bid)
                    self.assertIsNone(client.gather_segment(["parent"])[0])
                    client.close()
                    self.assertEqual(pool.used, 0)
                    channel.close()
                try:
                    await asyncio.to_thread(exercise)
                finally:
                    await server.stop(0)
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
