"""Keeper breakout/mirage diagnostic: source season Statcast xStats, FanGraphs rates,
and age; regress luck out of the current anchor into a skill-adjusted true-talent line;
rank players by forward keeper value. Shared shapes and pure classifier (no I/O).

See docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fantasy_baseball.utils.constants import safe_float


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
    Per-stat overrides for fast-settling signals (K%, barrel% ~ 60-100) vs slow (BABIP ~800).
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

MIRAGE_RATIO = 2.0  # surface move this many times the believed move -> luck, not skill


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


def w_for_stat(stat: str, row: SkillLuckRow, player_type: str, params: WMapParams) -> float:
    """Believed fraction: reliability * confirmation, in [0, 1].

    reliability = sample / (sample + stabilize[stat])
    confirmation = how well the underlying xStat supports the surface stat
    (barrel/xSLG for hr; xBA vs BABIP for avg; K-BB and xwOBA for pitcher ratios;
    SB from PA-reliability only; SV conservative).

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


_RATE_ONLY = {"avg", "era", "whip"}


def adjust_line(
    surface_line,
    projection_line,
    row,
    player_type,
    *,
    params=DEFAULT_WMAP,
    deviation_threshold=0.12,
):
    """Skill-adjusted counting line + label + reason.

    Rates are shrunk: adjusted = proj + w*(surface - proj), re-multiplied by the
    surface line's PT (held) to a counting line; avg/era/whip are carried as
    adjusted rates directly. Label from the aggregate signed, w-weighted deviation
    vs deviation_threshold; confidence "low" when sample below the stabilization
    sample or xStats absent.
    """
    s_rates = line_rates(surface_line, player_type)
    p_rates = line_rates(projection_line, player_type)
    pt = safe_float(surface_line.get("pa" if player_type == "hitter" else "ip", 0))
    adjusted = dict(surface_line)  # carry non-scored fields (positions, ab, ip, etc.)
    w_by_stat: dict[str, float] = {}
    believed = 0.0  # w-weighted signed deviation -> drives the label
    surface = 0.0  # raw (unweighted) signed deviation -> drives the deviator flag
    for stat, s_rate in s_rates.items():
        p_rate = p_rates.get(stat, s_rate)
        w = w_for_stat(stat, row, player_type, params)
        w_by_stat[stat] = w
        adj_rate = p_rate + w * (s_rate - p_rate)
        # luck-direction aware for era/whip (lower = better)
        direction = -1.0 if stat in ("era", "whip") else 1.0
        denom = abs(p_rate) if abs(p_rate) > 1e-9 else 1.0
        term = direction * (s_rate - p_rate) / denom
        believed += w * term
        surface += term
        if stat in _RATE_ONLY:
            adjusted[stat] = adj_rate
        else:
            adjusted[stat] = adj_rate * pt
    label = _label(believed, surface, deviation_threshold)
    reason = _reason(s_rates, p_rates, w_by_stat)
    sample = row.pa if player_type == "hitter" else row.ip
    stab = params.stat_stabilize.get("hr" if player_type == "hitter" else "k", params.pa_stabilize)
    confidence = "low" if sample < stab or row.xwoba is None else "full"
    return BreakoutResult(adjusted, label, reason, w_by_stat, confidence, surface, believed)


def _label(believed, surface, thr, *, mirage_ratio=MIRAGE_RATIO):
    # believed = w-weighted (skill-backed) deviation; surface = raw deviation.
    # A big surface move that belief mostly regressed away is luck
    # (mirage/slump); one belief largely kept is real (breakout/decline).
    # Symmetric on both signs; multiplication (not division) so a near-zero
    # believed cannot blow up.
    if abs(surface) < thr:
        return "stable"
    luck = abs(surface) > mirage_ratio * abs(believed)
    if surface >= thr:
        return "lucky mirage" if luck else "real breakout"
    return "slump" if luck else "real decline"


def _reason(s_rates, p_rates, w_by_stat):
    """Name the largest-magnitude believed mover (normalized by projection rate)."""

    def contribution(k):
        s_rate = s_rates[k]
        p_rate = p_rates.get(k, s_rate)
        w = w_by_stat.get(k, 0)
        # use direction-aware term like believed calculation
        direction = -1.0 if k in ("era", "whip") else 1.0
        denom = abs(p_rate) if abs(p_rate) > 1e-9 else 1.0
        term = direction * (s_rate - p_rate) / denom
        return abs(w * term)

    best = max(s_rates, key=contribution)
    delta = s_rates[best] - p_rates.get(best, s_rates[best])
    dirn = "up" if delta > 0 else "down"
    return f"{best} {dirn}, w={w_by_stat.get(best, 0):.2f}"
