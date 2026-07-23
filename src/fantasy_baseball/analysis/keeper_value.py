"""General keeper-asset-value metric: discounted multi-year VAR.

Year 2026 uses a player's blended anchor line; out-years scale that anchor
per-stat by ZiPS's own year-over-year ratios (clamped), then run through the
same full-season SGP -> VAR path the draft board uses. Pure math, no I/O.
See docs/superpowers/specs/2026-07-22-keeper-value-design.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.player_value import calculate_counting_sgp, calculate_player_sgp
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import Category, safe_float

if TYPE_CHECKING:
    # Type-only import: keeps this module I/O-free at runtime (draft.board pulls
    # data.db) while giving mypy the real ScaleInputs shape for scale.* accesses.
    from fantasy_baseball.draft.board import ScaleInputs

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
    # flags carries the sole fallback signal ("fallback_A") plus any "no_zips_<year>";
    # consumers test membership rather than a separate redundant boolean.
    flags: list[str]
    pct_from_out_years: float | None
    pct_from_saves: float | None


def _fields_for(player_type: str) -> tuple[str, ...]:
    return HITTER_FIELDS if player_type == "hitter" else PITCHER_FIELDS


def _clamp_ratio(numer: float, denom: float, band: tuple[float, float], eps: float) -> float | None:
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
        num = zips_y.get(field)
        # A missing/blank out-year cell carries no aging signal -> hold the anchor
        # flat, symmetric with a missing base (which _clamp_ratio maps to None). A
        # real numerator near 0 is a genuine decline and is clamped to the band low.
        if num is None or pd.isna(num):
            continue
        ratio = _clamp_ratio(safe_float(num), safe_float(zips_base.get(field, 0)), band, eps)
        if ratio is None:
            continue  # undefined ratio (missing base) -> hold the anchor value flat
        out[field] = safe_float(anchor.get(field, 0)) * ratio
    return out


def _line_sgp(line: Mapping[str, Any], player_type: str, scale: ScaleInputs) -> float:
    series = pd.Series({**line, "player_type": PlayerType(player_type)})
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
    line: Mapping[str, Any], positions: list[str], player_type: str, scale: ScaleInputs
) -> float:
    total_sgp = _line_sgp(line, player_type, scale)
    series = pd.Series(
        {
            **line,
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
    scale: ScaleInputs,
    *,
    base_year: int = 2026,
    horizon: int = DEFAULT_HORIZON,
    ratio_band: tuple[float, float] = DEFAULT_RATIO_BAND,
    min_pt: float | None = None,
    eps: float = EPS,
) -> tuple[dict[int, float], list[str]]:
    pyv: dict[int, float] = {}
    flags: list[str] = []
    zips_base = zips_by_year.get(base_year)

    def _flag_fallback() -> None:
        if "fallback_A" not in flags:
            flags.append("fallback_A")

    for k in range(horizon):
        year = base_year + k
        if k == 0:
            if anchor_line:
                pyv[year] = _value_of_line(anchor_line, positions, player_type, scale)
            elif zips_base:
                _flag_fallback()
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
            (not anchor_line) or (not zips_base) or _below_min_pt(zips_base, player_type, min_pt)
        )
        # `zips_base is None` is subsumed by `approach_a` (via `not zips_base`); it is
        # repeated here so the type checker narrows zips_base to non-None in the else.
        if approach_a or zips_base is None:
            _flag_fallback()
            line = zips_y
        else:
            line = _scale_line(anchor_line, zips_base, zips_y, player_type, ratio_band, eps)
        pyv[year] = _value_of_line(line, positions, player_type, scale)

    return pyv, flags


def pct_from_saves(
    anchor_line: Mapping[str, Any],
    player_type: str,
    scale: ScaleInputs,
    *,
    eps_share: float = DEFAULT_EPS_SHARE,
) -> float | None:
    if player_type != "pitcher":
        return 0.0
    sgp = _line_sgp(anchor_line, player_type, scale)
    if abs(sgp) <= eps_share:
        return None
    sv_sgp = calculate_counting_sgp(safe_float(anchor_line.get("sv", 0)), scale.denoms[Category.SV])
    return sv_sgp / sgp


def discounted_total(
    pyv: Mapping[int, float], base_year: int, discount: float, horizon: int
) -> float:
    return sum(discount**k * pyv.get(base_year + k, 0.0) for k in range(horizon))


def out_year_share(
    pyv: Mapping[int, float], base_year: int, total: float, *, eps_share: float = DEFAULT_EPS_SHARE
) -> float | None:
    """Out-year (post-base-year) contribution as a share of ``total``.

    ``total`` is the discounted sum at the discount of interest; the year-0 term
    is undiscounted, so the out-year slice is ``total - pyv[base_year]``. Guarded:
    a near-zero or negative ``total`` (sub-replacement keeper) yields ``None`` so
    the share never explodes or flips sign.
    """
    if total <= eps_share:
        return None
    return (total - pyv.get(base_year, 0.0)) / total


def keeper_value(
    player_id: str,
    name: str,
    anchor_line: Mapping[str, Any],
    positions: list[str],
    player_type: str,
    zips_by_year: Mapping[int, Mapping[str, Any] | None],
    scale: ScaleInputs,
    *,
    base_year: int = 2026,
    discount: float = DEFAULT_DISCOUNT,
    horizon: int = DEFAULT_HORIZON,
    ratio_band: tuple[float, float] = DEFAULT_RATIO_BAND,
    min_pt: float | None = None,
    eps: float = EPS,
    eps_share: float = DEFAULT_EPS_SHARE,
) -> KeeperValueResult:
    # Single-discount snapshot: total and pct_from_out_years are computed at this
    # call's `discount`. A sweep report recomputes both per displayed discount (via
    # discounted_total / out_year_share) rather than reading these stored fields.
    pyv, flags = per_year_var(
        anchor_line,
        positions,
        player_type,
        zips_by_year,
        scale,
        base_year=base_year,
        horizon=horizon,
        ratio_band=ratio_band,
        min_pt=min_pt,
        eps=eps,
    )
    total = discounted_total(pyv, base_year, discount, horizon)
    return KeeperValueResult(
        player_id=player_id,
        name=name,
        per_year_var=pyv,
        total=total,
        flags=flags,
        pct_from_out_years=out_year_share(pyv, base_year, total, eps_share=eps_share),
        pct_from_saves=pct_from_saves(anchor_line, player_type, scale, eps_share=eps_share),
    )
