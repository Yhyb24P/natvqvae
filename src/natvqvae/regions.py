"""Residue-level region map for NatV, loaded from the frozen contract."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REGIONS_PATH = REPO_ROOT / "contracts" / "regions.json"


@dataclass(frozen=True)
class Region:
    name: str
    start: int
    end: int

    def __contains__(self, position: int) -> bool:
        return self.start <= position <= self.end

    @property
    def positions(self) -> frozenset[int]:
        return frozenset(range(self.start, self.end + 1))


@lru_cache(maxsize=1)
def load_contract() -> dict:
    with REGIONS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def regions() -> dict[str, Region]:
    raw = load_contract()["regions"]
    return {name: Region(name, int(v["start"]), int(v["end"])) for name, v in raw.items()}


def region_of(position: int) -> str:
    for name, reg in regions().items():
        if position in reg:
            return name
    raise ValueError(f"position {position} is outside 1..341")


def anchors() -> dict[str, tuple[int, ...]]:
    raw = load_contract()["anchors"]
    return {domain: tuple(v["positions"]) for domain, v in raw.items()}


def catalytic_anchors() -> frozenset[int]:
    return frozenset(p for positions in anchors().values() for p in positions)


def validate_partition() -> None:
    """Fail closed if the frozen region map is not a clean partition of 1..341."""
    contract = load_contract()
    total = int(contract["target"]["length"])
    seen: dict[int, str] = {}
    for name, reg in regions().items():
        if reg.start > reg.end:
            raise ValueError(f"region {name} has start > end")
        for p in reg.positions:
            if p in seen:
                raise ValueError(f"position {p} in both {seen[p]} and {name}")
            if p < 1 or p > total:
                raise ValueError(f"position {p} outside 1..{total}")
            seen[p] = name
    missing = sorted(set(range(1, total + 1)) - set(seen))
    if missing:
        raise ValueError(f"regions do not cover positions {missing[:5]}...")
    for domain, positions in anchors().items():
        for p in positions:
            if region_of(p) != domain:
                raise ValueError(f"anchor {p} declared for {domain} but lies in {region_of(p)}")
