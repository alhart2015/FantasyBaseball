"""Keeper breakout/mirage diagnostic: source season Statcast xStats, FanGraphs rates,
and age; regress luck out of the current anchor into a skill-adjusted true-talent line;
rank players by forward keeper value. Shared shapes and pure classifier (no I/O).

See docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fantasy_baseball.analysis import keeper_value as _kv_mod
from fantasy_baseball.utils.constants import INVERSE_STATS, RATE_STATS, safe_float


@dataclass(frozen=True)
class SkillLuckRow:
    """The joined per-player-season underlying signal (one row). All optional except
    pa/player_type: a player may be missing xStats (insufficient batted balls) or be
    a pitcher.
    """

    mlbam: int
    player_type: str  # "hitter" | "pitcher"
    pa: float  # hitter plate appearances (0.0 for pitchers)
    ip: float  # pitcher innings (0.0 for hitters)
    age: float | None
    # hitter confirmations
    barrel_pct: float | None  # brl_percent (share, e.g. 0.12)
    xslg: float | None
    slg: float | None
    xba: float | None
    ba: float | None
    babip: float | None
    xwoba: float | None
    woba: float | None
    k_pct: float | None
    bb_pct: float | None
    # pitcher confirmations (K-BB, xwOBA-against reuse xwoba/woba/k_pct/bb_pct above)
    # HR-confirmation extras (issue #262). Optional: None for pitchers, pre-2016
    # (the xHR leaderboard starts 2016), or unmatched players. Defaults keep the
    # existing keyword constructors valid.
    brl_pa: float | None = None  # barrels per PA (share, e.g. 0.049)
    xhr: float | None = None  # park-adjusted expected HR (season count)


@dataclass(frozen=True)
class BreakoutResult:
    """The classifier output for one player."""

    adjusted_line: dict[str, float]  # counting line, same keys keeper_value consumes
    label: str  # one of LABELS
    reason: str  # short ASCII driver string
    w_by_stat: dict[str, float]  # believed-fraction per adjusted rate, for report/backtest
    confidence: str  # "full" | "low"
    surface_deviation: float  # raw signed aggregate surface-vs-projection deviation
    believed_deviation: float  # w-weighted signed deviation (drives the label)


LABELS = ("real breakout", "lucky mirage", "real decline", "slump", "stable")


HITTER_COUNTING = ("hr", "r", "rbi", "sb")


def line_rates(line, player_type):
    if player_type == "hitter":
        pa = safe_float(line.get("pa", 0))
        rates = {k: (safe_float(line.get(k, 0)) / pa if pa > 0 else 0.0) for k in HITTER_COUNTING}
        rates["avg"] = safe_float(line.get("avg", 0))
        return rates
    ip = safe_float(line.get("ip", 0))
    rates = {
        "k": (safe_float(line.get("k", 0)) / ip if ip > 0 else 0.0),
        "w": (safe_float(line.get("w", 0)) / ip if ip > 0 else 0.0),
        "sv": (safe_float(line.get("sv", 0)) / ip if ip > 0 else 0.0),
        "era": safe_float(line.get("era", 0)),
        "whip": safe_float(line.get("whip", 0)),
    }
    return rates


@dataclass(frozen=True)
class WMapParams:
    """Parameters for w-mapping (reliability x confirmation). Seed defaults calibrated
    for 5x5 roto keeper context: PA stabilization ~300 AB-equiv for hitters, IP for pitchers.
    Per-stat overrides for fast-settling signals (e.g. hr=120, k=60) vs slow-settling
    (avg=800).
    """

    pa_stabilize: float = 300.0
    ip_stabilize: float = 80.0
    confirm_weight: float = 0.5
    stat_stabilize: dict[str, float] = field(
        default_factory=lambda: {
            "hr": 120.0,
            "r": 300.0,
            "rbi": 300.0,
            "sb": 200.0,
            "avg": 800.0,
            "k": 60.0,
            "w": 120.0,
            "sv": 120.0,
            "era": 120.0,
            "whip": 120.0,
        }
    )


DEFAULT_WMAP = WMapParams()

# Roto-VALUE weight per unit of rate deviation, for the label aggregation. The
# label's believed/surface deviations sum `weight * (s_rate - p_rate)` across
# stats so each stat's voice matches its ROTO impact -- NOT its percentage swing.
# (The old `/ |p_rate|` relative form let a tiny-denominator stat dominate: a
# 1-SB dip on a 5-SB projection is a -20% swing that swamped a +25% HR jump, so a
# power breakout mislabeled as a decline.) Seed values approximate a full-season
# ~1-roto-point move mapping to ~1.0 (avg discounted for team-AB dilution);
# pitcher weights are seed/untuned (backtest is hitters-only). Tunable, like the
# w-mapping. KNOWN LIMITATION: a fixed avg/era/whip weight can't see the team-AB
# context the real SGP valuation uses, so it is approximate for those rate stats.
LABEL_WEIGHTS = {
    "hr": 65.0,
    "r": 33.0,
    "rbi": 33.0,
    "sb": 85.0,
    "avg": 40.0,
    "k": 10.0,
    "w": 100.0,
    "sv": 90.0,
    "era": 2.0,
    "whip": 60.0,
}


def _reliability(sample: float, stabilize: float) -> float:
    """Reliability: fraction of total signal attributable to sample vs noise.
    0 when sample=0; 1 when sample >> stabilize.
    """
    return sample / (sample + stabilize) if sample > 0 else 0.0


def _confirm_gap(actual: float | None, expected: float | None, scale: float) -> float:
    """Confirmation: 1.0 when expected matches actual; decays as |actual-expected|
    grows relative to `scale`. No signal returns 0.5 (neutral).
    """
    if actual is None or expected is None:
        return 0.5  # no signal -> neutral
    return max(0.0, 1.0 - abs(actual - expected) / scale)


def barrel_expected_rate(brl_pa: float, slope: float, intercept: float) -> float:
    """Barrel-implied expected HR/PA: the calibrated line HR/PA ~ brl_pa, clamped to
    >= 0 (a degenerate/extreme brl_pa must not yield a negative expected HR rate).
    Shared by the gate backtest (fitted calib) and the live diagnostic (frozen
    constants) so both agree. See issue #262 / backtest_hr_level.py."""
    return max(0.0, intercept + slope * brl_pa)


def w_for_stat(stat: str, row: SkillLuckRow, player_type: str, params: WMapParams) -> float:
    """Believed fraction: reliability * confirmation, in [0, 1].

    reliability = sample / (sample + stabilize[stat])
    confirmation = how well the underlying xStat supports the surface stat:
    slg vs xslg for hr; ba vs xba for avg; woba vs xwoba for r/rbi and all
    pitcher ratios (k/w/sv/era/whip); sb is confirm=1.0 fixed (role/speed
    sticky -- reliability alone governs). barrel_pct/babip/k_pct/bb_pct are
    display-only elsewhere in the report and are not consulted here.

    The blend (1-cw)*reliability + cw*reliability*confirm ensures that huge samples
    have high w even when confirmation is absent, while tiny samples shrink w regardless.
    """
    sample = row.pa if player_type == "hitter" else row.ip
    stabilize = params.stat_stabilize.get(
        stat, params.pa_stabilize if player_type == "hitter" else params.ip_stabilize
    )
    reliability = _reliability(sample, stabilize)

    if stat == "hr":
        confirm = _confirm_gap(row.slg, row.xslg, 0.150)  # slg vs xslg
    elif stat == "avg":
        confirm = _confirm_gap(row.ba, row.xba, 0.060)  # ba vs xba (BABIP luck shows here)
    elif stat in ("r", "rbi"):
        confirm = _confirm_gap(row.woba, row.xwoba, 0.040)  # context stats track overall quality
    elif stat == "sb":
        confirm = 1.0  # role/speed sticky; reliability governs
    else:  # pitcher ratios + sv/w
        confirm = _confirm_gap(row.woba, row.xwoba, 0.040) if row.xwoba is not None else 0.5

    # blend: never let confirmation alone drive w to 0 when sample is huge, nor vice versa
    cw = params.confirm_weight
    return reliability * ((1.0 - cw) + cw * confirm)


# Domain stat-sets projected to breakout's lowercase string keys from the repo's
# canonical Category sets (utils.constants) -- derived, not re-typed, so a change to
# the canonical inverse/rate categories can't leave breakout (or breakout_backtest's
# ruler and _rates_to_line, which import these) stale.
INVERTED_STATS: frozenset[str] = frozenset(c.value.lower() for c in INVERSE_STATS)
RATE_ONLY: frozenset[str] = frozenset(c.value.lower() for c in RATE_STATS)

DEVIATION_THRESHOLD = (
    0.2  # roto-value scale; shared by adjust_line's label + report's deviator flag
)


def _weighted_term(stat: str, s_rate: float, p_rate: float) -> float:
    """Signed, roto-value-weighted deviation for one stat, on the label's scale.
    era/whip are inverted (lower is better), and the weight is a fixed roto value
    (NOT relative to |p_rate|, which would explode tiny-denominator stats). Shared
    by adjust_line's label aggregation and _reason's mover ranking so both agree."""
    direction = -1.0 if stat in INVERTED_STATS else 1.0
    return direction * LABEL_WEIGHTS.get(stat, 0.0) * (s_rate - p_rate)


def adjust_line(
    surface_line,
    projection_line,
    row,
    player_type,
    *,
    params=DEFAULT_WMAP,
    deviation_threshold=DEVIATION_THRESHOLD,
):
    """Skill-adjusted counting line + label + reason.

    Rates are shrunk: adjusted = proj + w*(surface - proj), re-multiplied by the
    surface line's PT (held) to a counting line; avg/era/whip are carried as
    adjusted rates directly. Label from the aggregate signed, ROTO-VALUE-weighted
    deviation (LABEL_WEIGHTS, in ~roto points) vs deviation_threshold; confidence
    "low" when sample below the stabilization sample or xStats absent.
    """
    s_rates = line_rates(surface_line, player_type)
    p_rates = line_rates(projection_line, player_type)
    pt = safe_float(surface_line.get("pa" if player_type == "hitter" else "ip", 0))
    adjusted = dict(surface_line)  # carry non-scored fields (positions, ab, ip, etc.)
    w_by_stat: dict[str, float] = {}
    believed = 0.0  # w-weighted, roto-value-weighted signed deviation -> drives the label
    surface = 0.0  # raw (roto-value-weighted) signed deviation -> drives the deviator flag
    for stat, s_rate in s_rates.items():
        p_rate = p_rates.get(stat, s_rate)
        w = w_for_stat(stat, row, player_type, params)
        w_by_stat[stat] = w
        adj_rate = p_rate + w * (s_rate - p_rate)
        term = _weighted_term(stat, s_rate, p_rate)
        believed += w * term
        surface += term
        if stat in RATE_ONLY:
            adjusted[stat] = adj_rate
        else:
            adjusted[stat] = adj_rate * pt
    label = _label(believed, surface, deviation_threshold)
    reason = _reason(s_rates, p_rates, w_by_stat)
    sample = row.pa if player_type == "hitter" else row.ip
    stab = params.stat_stabilize.get("hr" if player_type == "hitter" else "k", params.pa_stabilize)
    confidence = "low" if sample < stab or row.xwoba is None else "full"
    return BreakoutResult(adjusted, label, reason, w_by_stat, confidence, surface, believed)


MIRAGE_RATIO = 2.0  # a surface move > this x the believed move is mostly luck (mirage/slump)


def _label(believed, surface, thr):
    # CONSERVATIVE label (the mirage-vs-breakout boundary is UNVALIDATED pending
    # the backtest). A 'real' breakout/decline requires believed to (a) AGREE IN
    # SIGN with surface and independently clear the threshold -- so a line the
    # model nets DOWN never reads as a breakout just because its raw magnitude is
    # nonzero -- AND (b) not be a small fraction of the apparent move: when the
    # surface move is more than MIRAGE_RATIO x the believed move, most of it is
    # unbelieved, so it's luck ("mirage" up, "slump" down) even if believed
    # clears the threshold. Symmetric on both signs.
    if abs(surface) < thr:
        return "stable"
    if surface >= thr:
        real = believed >= thr and surface <= MIRAGE_RATIO * abs(believed)
        return "real breakout" if real else "lucky mirage"
    real = believed <= -thr and abs(surface) <= MIRAGE_RATIO * abs(believed)
    return "real decline" if real else "slump"


def _reason(s_rates, p_rates, w_by_stat):
    """Name the largest believed mover, weighted by roto value (same scale the
    label uses -- so the reason matches what drove the label)."""

    def contribution(k):
        s_rate = s_rates[k]
        p_rate = p_rates.get(k, s_rate)
        w = w_by_stat.get(k, 0)
        return abs(w * _weighted_term(k, s_rate, p_rate))

    best = max(s_rates, key=contribution)
    delta = s_rates[best] - p_rates.get(best, s_rates[best])
    dirn = "up" if delta > 0 else "down"
    return f"{best} {dirn}, w={w_by_stat.get(best, 0):.2f}"


def _kv(pid, name, anchor, positions, ptype, zips_by_year, scale, **kw):
    """Indirection so tests can stub the keeper_value call."""
    return _kv_mod.keeper_value(pid, name, anchor, positions, ptype, zips_by_year, scale, **kw)


def _zips_for(row, indices, base_year, horizon):
    """Mirrors scripts/keeper_value.py:_zips_by_year (look up each year's ZiPS
    index by fg_id then name; miss -> None), bounded to [base_year, base_year +
    horizon) so a caller-supplied `indices` wider than the scored horizon can't
    leak extra years into the per_year_var loop."""
    from fantasy_baseball.sgp.rankings import lookup_rank

    fg = row.get("fg_id")
    fgid = str(fg) if fg is not None and str(fg).strip() else None
    ptype = str(row["player_type"])
    return {
        yr: (lookup_rank(indices.get(yr, {}), fgid, row["name"], ptype) or None)
        for yr in range(base_year, base_year + horizon)
    }


def breakout_rows(
    board,
    scale,
    indices,
    skill_luck,
    projections,
    *,
    base_year,
    horizon,
    discount,
    out_year_regression=_kv_mod.DEFAULT_OUT_YEAR_REGRESSION,
):
    """Per-board-player surface-believed vs skill-adjusted keeper value.

    ``out_year_regression`` MUST match scripts/keeper_value.py:build_results so the
    surface value equals today's --anchor current number and surface/adjusted
    differ ONLY in the anchor.

    ``skill_luck`` MUST be keyed by int fg_id (FanGraphs id), NOT MLBAM: the
    data-layer ``build_*_skill_luck`` functions return MLBAM-keyed dicts, so a
    caller (run_breakout_report) must re-key them via the ZiPS bridge before
    passing them in here.
    """
    rows = []
    for _, r in board.iterrows():
        row = r.to_dict()
        ptype = str(row["player_type"])
        fg = row.get("fg_id")
        fgid = int(fg) if fg is not None and str(fg).isdigit() else None
        positions = list(row["positions"])
        zby = _zips_for(row, indices, base_year, horizon)  # mirrors kv_script._zips_by_year

        def _value(anchor, row=row, positions=positions, ptype=ptype, zby=zby):
            return _kv(
                row["player_id"],
                row["name"],
                anchor,
                positions,
                ptype,
                zby,
                scale,
                base_year=base_year,
                horizon=horizon,
                discount=discount,
                out_year_regression=out_year_regression,
            ).total

        surface = _value(row)
        base = {"name": row["name"], "player_type": ptype, "surface_value": surface}
        sl = skill_luck.get(fgid) if fgid is not None else None
        if sl is not None and sl.player_type != ptype:
            # Two-way fg_id collision (e.g. Ohtani): skill_luck is keyed by bare
            # fg_id, so a hitter and pitcher row can share a key. A wrong-type
            # row must not be used -- degrade to the no-data fallback below
            # rather than silently corrupting this board row with the other
            # player_type's Statcast numbers.
            sl = None
        proj = projections.get(f"{fg}::{ptype}") if fg is not None else None
        if sl is not None and proj is not None:
            res = adjust_line(row, proj, sl, ptype, deviation_threshold=DEVIATION_THRESHOLD)
            adjusted = _value(res.adjusted_line)
            gap = (sl.woba - sl.xwoba) if sl.woba is not None and sl.xwoba is not None else None
            under = {
                "woba_xwoba_gap": gap,
                "babip": sl.babip,
                "barrel_pct": sl.barrel_pct,
                "k_pct": sl.k_pct,
                "bb_pct": sl.bb_pct,
            }
            rows.append(
                {
                    **base,
                    "adjusted_value": adjusted,
                    "delta": adjusted - surface,
                    "label": res.label,
                    "reason": res.reason,
                    "confidence": res.confidence,
                    "deviator": abs(res.surface_deviation) >= DEVIATION_THRESHOLD,
                    **under,
                }
            )
        else:
            rows.append(
                {
                    **base,
                    "adjusted_value": surface,
                    "delta": 0.0,
                    "label": "stable",
                    "reason": "no skill/luck data",
                    "confidence": "low",
                    "deviator": False,
                    "woba_xwoba_gap": None,
                    "babip": None,
                    "barrel_pct": None,
                    "k_pct": None,
                    "bb_pct": None,
                }
            )
    rows.sort(key=lambda d: d["adjusted_value"], reverse=True)
    return rows
