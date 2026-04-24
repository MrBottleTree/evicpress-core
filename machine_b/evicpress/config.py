"""
Config loader for EvicPress Machine B.
Reads config/config.yaml and exposes strongly-typed dataclasses.
Alpha can be mutated at runtime (dashboard slider updates it directly).
"""

import yaml
from dataclasses import dataclass, field
from typing import List


_VALID_QUANT = ("fp16", "int8", "int4")


@dataclass
class Tier1Config:
    capacity_bytes: int
    bandwidth_bytes_per_sec: float


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
class QuantConfig:
    enabled: bool
    size_multiplier: dict   # quant level -> float


@dataclass
class PlacementBand:
    name: str
    min_utility: float
    tiers: List[int]        # subset of {1, 2, 3} — T3 is always forced in
    quant: str              # "fp16" | "int8" | "int4"


@dataclass
class PlacementConfig:
    bands: List[PlacementBand]


@dataclass
class EvicPressConfig:
    alpha: float                    # Mutable at runtime
    tier1: Tier1Config
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
    quantization: QuantConfig
    placement: PlacementConfig


def _default_quant() -> QuantConfig:
    return QuantConfig(
        enabled=False,
        size_multiplier={"fp16": 1.0, "int8": 0.5, "int4": 0.25},
    )


def _default_placement() -> PlacementConfig:
    # Conservative fallback: everything fp16 in T2+T3, hot blocks also in T1.
    return PlacementConfig(bands=[
        PlacementBand(name="hot",  min_utility=0.8, tiers=[1, 2, 3], quant="fp16"),
        PlacementBand(name="warm", min_utility=0.0, tiers=[2, 3],    quant="fp16"),
    ])


def _parse_quant(raw: dict) -> QuantConfig:
    if not raw:
        return _default_quant()
    return QuantConfig(
        enabled=bool(raw.get("enabled", False)),
        size_multiplier={
            str(k): float(v)
            for k, v in (raw.get("size_multiplier") or {}).items()
        },
    )


def _parse_placement(raw: dict) -> PlacementConfig:
    if not raw or "bands" not in raw:
        return _default_placement()
    bands = []
    for b in raw["bands"]:
        quant = str(b.get("quant", "fp16"))
        if quant not in _VALID_QUANT:
            raise ValueError(
                f"placement band {b.get('name')!r}: invalid quant {quant!r}"
            )
        bands.append(PlacementBand(
            name=str(b.get("name", "unnamed")),
            min_utility=float(b["min_utility"]),
            tiers=[int(t) for t in b.get("tiers", [3])],
            quant=quant,
        ))
    # Sort descending by min_utility so first match wins as you go down the
    # utility scale. Keeps config author's ordering if already descending.
    bands.sort(key=lambda b: b.min_utility, reverse=True)
    return PlacementConfig(bands=bands)


def load_config(path: str = "config/config.yaml") -> EvicPressConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    return EvicPressConfig(
        alpha=float(raw["evicpress"]["alpha"]),
        tier1=Tier1Config(
            capacity_bytes=int(raw["tier1"]["capacity_bytes"]),
            bandwidth_bytes_per_sec=float(raw["tier1"]["bandwidth_bytes_per_sec"]),
        ),
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
        quantization=_parse_quant(raw.get("quantization") or {}),
        placement=_parse_placement(raw.get("placement") or {}),
    )
