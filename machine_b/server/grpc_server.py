"""
gRPC service implementation for EvicPress Machine B.

Thin adapter layer: translates gRPC request/response objects into calls on
EvicPressManager. All heavy logic lives in the manager, not here.
"""

import grpc
from generated import evicpress_pb2, evicpress_pb2_grpc

from grpc import aio
from evicpress.manager import EvicPressManager


class EvicPressServicer(evicpress_pb2_grpc.EvicPressServiceServicer):
    def __init__(self, manager: EvicPressManager) -> None:
        self._m = manager

    async def StoreHeads(self, request, context):
        try:
            result = self._m.head_groups.store(
                request.descriptor_json, [(h.block_id, h.data) for h in request.heads],
                request.client_id)
            return evicpress_pb2.HeadGroupResponse(**result)
        except (ValueError, RuntimeError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

    async def LookupHeads(self, request, context):
        return await self._read_heads(request, context, retrieve=False)

    async def RetrieveHeads(self, request, context):
        return await self._read_heads(request, context, retrieve=True)

    async def DeleteHeads(self, request, context):
        return evicpress_pb2.DeleteResponse(
            success=self._m.head_groups.delete(list(request.parent_ids), request.force))

    async def ReleaseHeads(self, request, context):
        self._m.head_groups.release(list(request.lease_ids), request.client_id)
        return evicpress_pb2.DeleteResponse(success=True)

    async def SyncHeadTier1(self, request, context):
        try:
            revoked = self._m.head_groups.sync_tier1(
                request.client_id, {e.block_id: e.nbytes for e in request.entries})
            return evicpress_pb2.HeadTier1Response(revoked_ids=revoked)
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

    async def _read_heads(self, request, context, *, retrieve: bool):
        try:
            items = [(p.parent_id, set(p.local_ids), p.lease_id) for p in request.parents]
            result = self._m.head_groups.read(
                items, request.client_id, retrieve=retrieve, pin=request.pin)
            return evicpress_pb2.HeadBatchResponse(parents=result)
        except (ValueError, RuntimeError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

    async def Lookup(self, request, context):
        hit, tier = self._m.lookup(request.block_id)
        return evicpress_pb2.LookupResponse(hit=hit, tier=tier)

    async def Store(self, request, context):
        success, tier, _quant = self._m.store(
            request.block_id,
            request.data,
            request.quality_score or 1.0,
        )
        return evicpress_pb2.StoreResponse(
            success=success,
            tier=tier,
            message=f"stored in tier{tier}" if success else "store rejected by tier3",
        )

    async def Retrieve(self, request, context):
        result = self._m.retrieve(request.block_id)
        if result is None:
            return evicpress_pb2.RetrieveResponse(found=False, data=b"", tier=0)
        data, tier, _quant = result
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
        t1 = state["tier1"]
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
            tier1_blocks=t1["block_count"],
            tier1_bytes=t1["used_bytes"],
            tier1_promotions=s["tier1_promotions"],
            head_schema_version=1,
            parent_hits=s["parent_hits"],
            parent_misses=s["parent_misses"],
            head_rpc_calls=s["head_rpc_calls"],
            head_wire_bytes=s["head_wire_bytes"],
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
