"""General keeper-asset-value metric: discounted multi-year VAR.

Year 2026 uses a player's blended anchor line; out-years scale that anchor
per-stat by ZiPS's own year-over-year ratios (clamped), then run through the
same full-season SGP -> VAR path the draft board uses. Pure math, no I/O.
See docs/superpowers/specs/2026-07-22-keeper-value-design.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fantasy_baseball.utils.constants import safe_float

DEFAULT_DISCOUNT = 0.80
DEFAULT_HORIZON = 3
DEFAULT_RATIO_BAND = (0.25, 2.5)
DEFAULT_MIN_AB = 100.0
DEFAULT_MIN_IP = 20.0
EPS = 1e-6
DEFAULT_EPS_SHARE = 1.0

HITTER_FIELDS = ("r", "hr", "rbi", "sb", "ab", "avg")
PITCHER_FIELDS = ("w", "k", "sv", "ip", "era", "whip")


@dataclass(frozen=True)
class KeeperValueResult:
    player_id: str
    name: str
    per_year_var: dict[int, float]
    total: float
    used_fallback: bool
    flags: list[str]
    pct_from_out_years: float | None
    pct_from_saves: float | None


def _fields_for(player_type: str) -> tuple[str, ...]:
    return HITTER_FIELDS if player_type == "hitter" else PITCHER_FIELDS


def _clamp_ratio(
    numer: float, denom: float, band: tuple[float, float], eps: float
) -> float | None:
    if abs(denom) < eps:
        return None
    lo, hi = band
    return max(lo, min(hi, numer / denom))


def _scale_line(
    anchor: Mapping[str, Any],
    zips_base: Mapping[str, Any],
    zips_y: Mapping[str, Any],
    player_type: str,
    band: tuple[float, float],
    eps: float,
) -> dict[str, Any]:
    out = dict(anchor)
    for field in _fields_for(player_type):
        ratio = _clamp_ratio(
            safe_float(zips_y.get(field, 0)), safe_float(zips_base.get(field, 0)), band, eps
        )
        if ratio is None:
            continue  # undefined ratio -> hold the anchor value flat for this field
        out[field] = safe_float(anchor.get(field, 0)) * ratio
    return out
