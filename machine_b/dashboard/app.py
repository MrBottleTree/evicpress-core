"""
Dashboard FastAPI app — serves the web UI and REST API for the debug dashboard.

Endpoints:
  GET  /             → HTML dashboard page
  GET  /api/state    → Full system state (JSON, polled by the browser every 1s)
  GET  /api/blocks   → All block metadata (for the block browser table)
  POST /api/config   → Live-update tunable params (currently: alpha)
"""

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from evicpress.manager import EvicPressManager


def create_dashboard(manager: EvicPressManager) -> FastAPI:
    app = FastAPI(title="EvicPress Dashboard", docs_url=None, redoc_url=None)

    _static_dir = os.path.join(os.path.dirname(__file__), "static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        with open(os.path.join(_static_dir, "index.html")) as f:
            return f.read()

    @app.get("/api/state")
    async def get_state() -> dict:
        return manager.get_state()

    @app.get("/api/blocks")
    async def get_blocks() -> list:
        return manager.get_blocks()

    @app.post("/api/config")
    async def update_config(body: dict[str, Any]) -> dict:
        """
        Live-update tunable parameters without restarting.
        Currently supports:
          { "alpha": 2.5 }
        """
        if "alpha" in body:
            try:
                manager.config.alpha = float(body["alpha"])
            except (TypeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=str(e))
        # Future: tier capacities, bandwidth overrides, prefetch toggle, etc.
        return {"status": "ok", "applied": body, "config": manager.get_state()["config"]}

    return app
