"""ADP (average draft position) loader and blend.

ADP drives the forward-model opponent picks in value-of-picking-now.
Each projection system's CSV ships an ``adp`` column (missing for deep
bench players — handled by ADPTable.get via a constant fallback offset).
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd


def blend_adp(per_system: Mapping[str, pd.DataFrame]) -> dict[str, float]:
    """Mean-blend per-system ADP into ``{player_id: adp}``.

    Missing values are skipped, not treated as zero. If a player has ADP
    in only one system, the single value is used.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for df in per_system.values():
        if "adp" not in df.columns or "player_id" not in df.columns:
            continue
        for pid, adp in zip(df["player_id"], df["adp"], strict=False):
            if pd.isna(adp):
                continue
            totals[pid] = totals.get(pid, 0.0) + float(adp)
            counts[pid] = counts.get(pid, 0) + 1
    return {pid: totals[pid] / counts[pid] for pid in totals}


@dataclass
class ADPTable:
    """ADP lookup with a stable fallback for players missing from every system."""

    adp: dict[str, float] = field(default_factory=dict)
    fallback_offset: float = 1000.0  # unknown players sort behind every real ADP

    def get(self, player_id: str) -> float:
        if player_id in self.adp:
            return self.adp[player_id]
        # Deterministic tie-break so unknown players' fallback order is stable
        # across processes (bare `hash()` is randomized per-process since 3.4).
        return self.fallback_offset + zlib.crc32(player_id.encode()) % 1000
