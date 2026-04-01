"""
Config loader for EvicPress Machine B.
Reads config/config.yaml and exposes strongly-typed dataclasses.
Alpha can be mutated at runtime (dashboard slider updates it directly).
"""

import yaml
from dataclasses import dataclass


@dataclass
class Tier2Config:
    capacity_bytes: int
    bandwidth_bytes_per_sec: float


@dataclass
class Tier3Config:
    capacity_bytes: int
    bandwidth_bytes_per_sec: float
    data_dir: str


@dataclass
class EvicPressConfig:
    alpha: float                    # Mutable at runtime
    tier2: Tier2Config
    tier3: Tier3Config
    prefetch_enabled: bool
    prefetch_max_queue: int
    server_host: str
    server_port: int
    server_max_message_bytes: int
    dashboard_host: str
    dashboard_port: int
    max_recent_ops: int


def load_config(path: str = "config/config.yaml") -> EvicPressConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    return EvicPressConfig(
        alpha=float(raw["evicpress"]["alpha"]),
        tier2=Tier2Config(
            capacity_bytes=int(raw["tier2"]["capacity_bytes"]),
            bandwidth_bytes_per_sec=float(raw["tier2"]["bandwidth_bytes_per_sec"]),
        ),
        tier3=Tier3Config(
            capacity_bytes=int(raw["tier3"]["capacity_bytes"]),
            bandwidth_bytes_per_sec=float(raw["tier3"]["bandwidth_bytes_per_sec"]),
            data_dir=str(raw["tier3"]["data_dir"]),
        ),
        prefetch_enabled=bool(raw["prefetch"]["enabled"]),
        prefetch_max_queue=int(raw["prefetch"]["max_queue_size"]),
        server_host=str(raw["server"]["host"]),
        server_port=int(raw["server"]["port"]),
        server_max_message_bytes=int(raw["server"]["max_message_bytes"]),
        dashboard_host=str(raw["dashboard"]["host"]),
        dashboard_port=int(raw["dashboard"]["port"]),
        max_recent_ops=int(raw["logging"]["max_recent_ops"]),
    )
