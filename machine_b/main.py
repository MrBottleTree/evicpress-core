"""
EvicPress Machine B — Entry Point

Starts two servers in the same asyncio event loop:
  1. gRPC server  — Machine A connects here to store/retrieve KV blocks
  2. FastAPI HTTP  — web dashboard (open in browser for live debug view)

Usage:
  python main.py [--config path/to/config.yaml]
"""

import argparse
import asyncio
import sys

import uvicorn

from evicpress.config import load_config
from evicpress.manager import EvicPressManager
from server.grpc_server import create_grpc_server
from dashboard.app import create_dashboard


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EvicPress Machine B")
    p.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    return p.parse_args()


async def run(config_path: str) -> None:
    # ── Load config ────────────────────────────────────────────
    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 56)
    print("  EvicPress Machine B")
    print("=" * 56)
    print(f"  Config file  : {config_path}")
    print(f"  gRPC         : {cfg.server_host}:{cfg.server_port}")
    print(f"  Dashboard    : http://{cfg.dashboard_host}:{cfg.dashboard_port}")
    print(f"  Tier 2 (RAM) : {cfg.tier2.capacity_bytes/1e9:.1f} GB")
    print(f"  Tier 3 (Disk): {cfg.tier3.capacity_bytes/1e9:.1f} GB  @ {cfg.tier3.data_dir}")
    print(f"  Alpha        : {cfg.alpha}")
    print(f"  Prefetch     : {'enabled' if cfg.prefetch_enabled else 'disabled'}")
    print("=" * 56)

    # ── Create manager ─────────────────────────────────────────
    manager = EvicPressManager(cfg)

    # ── Prefetch queue (asyncio-native) ────────────────────────
    prefetch_q = asyncio.Queue(maxsize=cfg.prefetch_max_queue)
    manager.prefetch_queue = prefetch_q

    # ── gRPC server ────────────────────────────────────────────
    grpc_server = await create_grpc_server(
        manager,
        cfg.server_host,
        cfg.server_port,
        cfg.server_max_message_bytes,
    )

    # ── Dashboard HTTP server ──────────────────────────────────
    dashboard_app = create_dashboard(manager)
    uv_cfg = uvicorn.Config(
        dashboard_app,
        host=cfg.dashboard_host,
        port=cfg.dashboard_port,
        log_level="warning",   # keep console quiet; use the dashboard instead
    )
    uv_server = uvicorn.Server(uv_cfg)
    print(f"[HTTP] dashboard on http://{cfg.dashboard_host}:{cfg.dashboard_port}")

    # ── Run everything concurrently ────────────────────────────
    try:
        await asyncio.gather(
            grpc_server.wait_for_termination(),
            uv_server.serve(),
            manager.prefetch_worker(),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[INFO] Shutting down…")
        await grpc_server.stop(grace=3)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.config))
