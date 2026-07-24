"""Keeper breakout/mirage diagnostic: source season Statcast xStats, FanGraphs rates,
and age; regress luck out of the current anchor into a skill-adjusted true-talent line;
rank players by forward keeper value. Shared shapes and pure classifier (no I/O).

See docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

from fantasy_baseball.utils.constants import safe_float


@dataclass(frozen=True)
class SkillLuckRow:
    """The joined per-player-season underlying signal (one row). All optional except
    pa/player_type: a player may be missing xStats (insufficient batted balls) or be
    a pitcher.
    """
    mlbam: int
    player_type: str            # "hitter" | "pitcher"
    pa: float                   # hitter plate appearances (0.0 for pitchers)
    ip: float                   # pitcher innings (0.0 for hitters)
    age: float | None
    # hitter confirmations
    barrel_pct: float | None    # brl_percent (share, e.g. 0.12)
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
    adjusted_line: dict[str, float]   # counting line, same keys keeper_value consumes
    label: str                        # one of LABELS
    reason: str                       # short ASCII driver string
    w_by_stat: dict[str, float]       # believed-fraction per adjusted rate, for report/backtest
    confidence: str                   # "full" | "low"
    surface_deviation: float          # raw signed aggregate surface-vs-projection deviation
    believed_deviation: float         # w-weighted signed deviation (drives the label)


LABELS = ("real breakout", "lucky mirage", "real decline", "slump", "stable")


HITTER_COUNTING = ("hr", "r", "rbi", "sb")


def line_rates(line, player_type):
    if player_type == "hitter":
        pa = safe_float(line.get("pa", 0))
        rates = {k: (safe_float(line.get(k, 0)) / pa if pa > 0 else 0.0) for k in HITTER_COUNTING}
        rates["avg"] = safe_float(line.get("avg", 0))
        return rates
    ip = safe_float(line.get("ip", 0))
    rates = {"k": (safe_float(line.get("k", 0)) / ip if ip > 0 else 0.0),
             "w": (safe_float(line.get("w", 0)) / ip if ip > 0 else 0.0),
             "sv": (safe_float(line.get("sv", 0)) / ip if ip > 0 else 0.0),
             "era": safe_float(line.get("era", 0)), "whip": safe_float(line.get("whip", 0))}
    return rates
