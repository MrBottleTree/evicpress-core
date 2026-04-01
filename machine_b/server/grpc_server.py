"""
gRPC service implementation for EvicPress Machine B.

Thin adapter layer: translates gRPC request/response objects into calls on
EvicPressManager. All heavy logic lives in the manager, not here.
"""

import sys
import os

# Allow importing generated proto code regardless of CWD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generated"))

import evicpress_pb2        # type: ignore  (generated)
import evicpress_pb2_grpc   # type: ignore  (generated)

from grpc import aio
from evicpress.manager import EvicPressManager


class EvicPressServicer(evicpress_pb2_grpc.EvicPressServiceServicer):
    def __init__(self, manager: EvicPressManager) -> None:
        self._m = manager

    async def Lookup(self, request, context):
        hit, tier = self._m.lookup(request.block_id)
        return evicpress_pb2.LookupResponse(hit=hit, tier=tier)

    async def Store(self, request, context):
        success, tier = self._m.store(
            request.block_id,
            request.data,
            request.quality_score or 1.0,
        )
        return evicpress_pb2.StoreResponse(
            success=success,
            tier=tier,
            message=f"stored in tier{tier}",
        )

    async def Retrieve(self, request, context):
        result = self._m.retrieve(request.block_id)
        if result is None:
            return evicpress_pb2.RetrieveResponse(found=False, data=b"", tier=0)
        data, tier = result
        return evicpress_pb2.RetrieveResponse(found=True, data=data, tier=tier)

    async def Prefetch(self, request, context):
        queued = self._m.queue_prefetch(list(request.block_ids))
        return evicpress_pb2.PrefetchResponse(queued=queued)

    async def Delete(self, request, context):
        success = self._m.delete(request.block_id)
        return evicpress_pb2.DeleteResponse(success=success)

    async def GetStats(self, request, context):
        state = self._m.get_state()
        s  = state["stats"]
        t2 = state["tier2"]
        t3 = state["tier3"]
        return evicpress_pb2.StatsResponse(
            tier2_blocks=t2["block_count"],
            tier3_blocks=t3["block_count"],
            tier2_bytes=t2["used_bytes"],
            tier3_bytes=t3["used_bytes"],
            total_hits=s["total_hits"],
            total_misses=s["total_misses"],
            tier2_hits=s["tier2_hits"],
            tier3_hits=s["tier3_hits"],
            hit_rate=s["hit_rate"],
            evictions=s["evictions"],
        )


async def create_grpc_server(
    manager: EvicPressManager,
    host: str,
    port: int,
    max_msg_bytes: int,
) -> aio.Server:
    options = [
        ("grpc.max_receive_message_length", max_msg_bytes),
        ("grpc.max_send_message_length",    max_msg_bytes),
    ]
    server = aio.server(options=options)
    evicpress_pb2_grpc.add_EvicPressServiceServicer_to_server(
        EvicPressServicer(manager), server
    )
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    print(f"[gRPC] listening on {host}:{port}")
    return server
