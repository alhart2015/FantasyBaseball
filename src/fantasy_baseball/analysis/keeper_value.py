"""General keeper-asset-value metric: discounted multi-year VAR.

Year 2026 uses a player's blended anchor line; out-years scale that anchor
per-stat by ZiPS's own year-over-year ratios (clamped), then run through the
same full-season SGP -> VAR path the draft board uses. Pure math, no I/O.
See docs/superpowers/specs/2026-07-22-keeper-value-design.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import pandas as pd

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.player_value import calculate_counting_sgp, calculate_player_sgp
from fantasy_baseball.sgp.rankings import rank_key
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import Category, safe_float

if TYPE_CHECKING:
    # Type-only import: keeps this module I/O-free at runtime (draft.board pulls
    # data.db) while giving mypy the real ScaleInputs shape for scale.* accesses.
    from fantasy_baseball.draft.board import ScaleInputs

DEFAULT_DISCOUNT = 0.80
DEFAULT_HORIZON = 3
# Out-year (2027+) regression toward ZiPS's own forward projection: 0 = pure
# anchor x aging-ratio (over-indexes on the current season), 1 = pure ZiPS
# out-year. 0.6 = "mostly ZiPS" -- keeps ~40% of the realized-2026 signal while
# leaning on ZiPS's regressed multi-year view (inherits ZiPS's skill-vs-luck sort).
DEFAULT_OUT_YEAR_REGRESSION = 0.6
# PT-heal: when a player's current (YTD+ROS) playing time is below his preseason
# PT (an injury, not a talent decline), scale the counting stats up toward the
# healthy PT (rates held) so a lost half-season doesn't negate keeper talent. The
# heal factor is max(1, min(cap, preseason_PT/current_PT)): continuous and monotonic
# in playing time (a mild PT dip gets a mild bump, a severe injury the full cap), so
# there is no threshold cliff. `cap` is the max multiplier -- it bounds how far a
# tiny, noisy sample is extrapolated; cap <= 1.0 disables the heal.
DEFAULT_PT_HEAL_CAP = 2.0
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
    out_year_regression: float = 0.0,
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
            if out_year_regression > 0.0:
                # Regress the current-anchor-scaled out-year toward ZiPS's own
                # (regressed, multi-year) out-year projection, per scored field.
                # lam=0 keeps the pure anchor x ratio; lam=1 is pure ZiPS out-year.
                lam = min(1.0, out_year_regression)
                for f in _fields_for(player_type):
                    zy = zips_y.get(f)
                    if zy is not None and not pd.isna(zy):
                        line[f] = (1.0 - lam) * safe_float(line.get(f, 0)) + lam * safe_float(zy)
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
    out_year_regression: float = 0.0,
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
        out_year_regression=out_year_regression,
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


def overlay_current_anchors(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
    current_by_name: Mapping[str, Mapping[str, Any]],
    *,
    min_ab: float = DEFAULT_MIN_AB,
    min_ip: float = DEFAULT_MIN_IP,
    heal_cap: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Replace each board frame's stat line with the current-talent line when one
    exists for that player (keyed name::player_type) AND clears the min-PT floor.

    ``heal_cap`` PT-heals injury-shortened anchors (up only): a player whose current
    PT (ab/ip) is below his preseason PT has his counting stats scaled by
    ``factor = min(heal_cap, preseason_PT/current_PT)`` toward the healthy PT, with
    rates (avg/era/whip) held. The factor is continuous and monotonic in playing time
    (a mild PT dip -> mild bump, a severe injury -> the full cap), so there is no
    threshold cliff. ``heal_cap <= 1.0`` disables it (use the raw current line).

    KNOWN LIMITATION (heuristic): a PT drop is assumed to be injury -- a genuine
    playing-time *loss* (a platoon/role bat with strong rates but few PA) is
    over-healed, since only IL data could distinguish it (deferred, see the risk
    follow-up). A performance decline is NOT over-healed: rates are held, so scaling
    a low-rate line up to full PT still yields low counting value.

    Returns ``(merged_hitters, merged_pitchers, current_keys)`` where ``current_keys``
    are the ``rank_key(name, player_type)`` values that received the current anchor;
    every other player keeps its preseason line and is flagged by the caller.
    """
    current_keys: set[str] = set()
    out = []
    for df, ptype, fields, vol_field, rate_fields, floor in (
        (hitters, "hitter", HITTER_FIELDS, "ab", ("avg",), min_ab),
        (pitchers, "pitcher", PITCHER_FIELDS, "ip", ("era", "whip"), min_ip),
    ):
        merged = df.copy()
        # Overlaid/healed stats are fractional; float the scored columns so writing a
        # scaled value into an int-typed frame is lossless (and pandas doesn't warn).
        present = [f for f in fields if f in merged.columns]
        if present:
            merged[present] = merged[present].astype(float)
        for idx, name in merged["name"].items():
            key = rank_key(str(name), ptype)
            line = current_by_name.get(key)
            if line is None:
                continue
            cur_pt = safe_float(line.get(vol_field, 0))
            if cur_pt < floor:
                continue
            factor = 1.0
            if heal_cap > 1.0 and cur_pt > 0.0 and vol_field in merged.columns:
                pre_pt = safe_float(merged.at[idx, vol_field])
                if pre_pt > cur_pt:  # played less than projected -> heal up (capped)
                    factor = min(heal_cap, pre_pt / cur_pt)
            # rates (avg/era/whip) carry talent and are held; volume + counting
            # stats scale with the (healed) playing time.
            for f in fields:
                v = safe_float(line.get(f))
                merged.at[idx, f] = v if f in rate_fields else v * factor
            current_keys.add(key)
        out.append(merged)
    return out[0], out[1], current_keys


def mark_preseason_fallback(
    results: list[KeeperValueResult], current_keys: set[str]
) -> list[KeeperValueResult]:
    """Append ``anchor_preseason_fallback`` to every result NOT scored off a current
    anchor (i.e. whose ``name::player_type`` is not in ``current_keys``)."""
    marked = []
    for r in results:
        ptype = r.player_id.rsplit("::", 1)[-1]
        if rank_key(r.name, ptype) in current_keys:
            marked.append(r)
        else:
            marked.append(replace(r, flags=[*r.flags, "anchor_preseason_fallback"]))
    return marked
