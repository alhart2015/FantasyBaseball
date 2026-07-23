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

import pandas as pd

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.var import calculate_var
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


def _line_sgp(line: Mapping[str, Any], player_type: str, scale) -> float:
    series = pd.Series({**dict(line), "player_type": PlayerType(player_type)})
    return calculate_player_sgp(
        series,
        denoms=scale.denoms,
        replacement_avg=scale.repl_rates["avg"],
        replacement_era=scale.repl_rates["era"],
        replacement_whip=scale.repl_rates["whip"],
        team_ab=scale.team_ab,
        team_ip=scale.team_ip,
    )


def _value_of_line(
    line: Mapping[str, Any], positions: list[str], player_type: str, scale
) -> float:
    total_sgp = _line_sgp(line, player_type, scale)
    series = pd.Series(
        {
            **dict(line),
            "player_type": PlayerType(player_type),
            "positions": list(positions),
            "total_sgp": total_sgp,
        }
    )
    return float(calculate_var(series, scale.replacement_levels))


def _below_min_pt(zips_line: Mapping[str, Any], player_type: str, min_pt: float | None) -> bool:
    if player_type == "hitter":
        thresh = min_pt if min_pt is not None else DEFAULT_MIN_AB
        return safe_float(zips_line.get("ab", 0)) < thresh
    thresh = min_pt if min_pt is not None else DEFAULT_MIN_IP
    return safe_float(zips_line.get("ip", 0)) < thresh


def per_year_var(
    anchor_line: Mapping[str, Any],
    positions: list[str],
    player_type: str,
    zips_by_year: Mapping[int, Mapping[str, Any] | None],
    scale,
    *,
    base_year: int = 2026,
    horizon: int = DEFAULT_HORIZON,
    ratio_band: tuple[float, float] = DEFAULT_RATIO_BAND,
    min_pt: float | None = None,
    eps: float = EPS,
) -> tuple[dict[int, float], list[str], bool]:
    pyv: dict[int, float] = {}
    flags: list[str] = []
    used_fallback = False
    zips_base = zips_by_year.get(base_year)

    for k in range(horizon):
        year = base_year + k
        if k == 0:
            if anchor_line:
                pyv[year] = _value_of_line(anchor_line, positions, player_type, scale)
            elif zips_base:
                used_fallback = True
                if "fallback_A" not in flags:
                    flags.append("fallback_A")
                pyv[year] = _value_of_line(zips_base, positions, player_type, scale)
            else:
                pyv[year] = 0.0
                flags.append(f"no_zips_{year}")
            continue

        zips_y = zips_by_year.get(year)
        if not zips_y:
            pyv[year] = 0.0
            flags.append(f"no_zips_{year}")
            continue

        approach_a = (
            (not anchor_line)
            or (not zips_base)
            or _below_min_pt(zips_base, player_type, min_pt)
        )
        if approach_a:
            used_fallback = True
            if "fallback_A" not in flags:
                flags.append("fallback_A")
            line = zips_y
        else:
            line = _scale_line(anchor_line, zips_base, zips_y, player_type, ratio_band, eps)
        pyv[year] = _value_of_line(line, positions, player_type, scale)

    return pyv, flags, used_fallback
