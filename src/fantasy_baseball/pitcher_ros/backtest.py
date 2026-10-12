"""Score pitcher projections against FanGraphs and the simple baselines (#465).

Builds the long "scored" frames (one row per pitcher x system x stat, tagged with
``season`` and, mid-season, ``snapshot``) for:

* Preseason vs. each FanGraphs system in ``data/projections/T/``, their blend
  (``fg_blend``), and the ``marcel`` / ``league_avg`` baselines; actuals are the
  whole season.
* Mid-season vs. each dated ROS snapshot in ``data/projections/T/rest_of_season/<date>/``;
  the baselines use the games before the snapshot date, and actuals are the games on
  or after it.

and renders a markdown summary led by MSE (the main score, ``evaluate``), with a paired
bootstrap of each system vs. the FanGraphs blend.

Every scored row is tagged like the hitters' (``hitter_ros/backtest.py``):

* ``group``: "vet" with at least ``VET_MIN_CAREER_BF`` MLB batters faced before the
  projection date, else "rookie"; "unknown" when the store (from 2008) holds fewer than
  ``MIN_HISTORY_SEASONS`` earlier seasons, so a vet could look like a rookie.
* ``relevant``: among the ``RELEVANT_TOP`` pitchers by fantasy value either in the
  FanGraphs projections compared against or in what actually happened over the span
  (both count, so the set isn't picked on the compared system's own output alone).
  ``RELEVANT_TOP`` is 150: this league starts 9 pitchers on each of 10 teams.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from fantasy_baseball.hitter_ros.backtest import sure_text, to_markdown
from fantasy_baseball.pitcher_ros.baselines import baseline_predictions
from fantasy_baseball.pitcher_ros.evaluate import (
    blend,
    fantasy_value,
    load_systems,
    mse_bootstrap,
    mse_table,
    scored_players,
)
from fantasy_baseball.pitcher_ros.stats import TARGETS, rates_from_counts, totals
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.constants import Category

PRESEASON_MIN_BF = 100
SNAPSHOT_MIN_BF = 40
BLEND = "fg_blend"
VET_MIN_CAREER_BF = 300
MIN_HISTORY_SEASONS = 4
GROUPS = ("vet", "rookie")
RELEVANT_TOP = 150


def _actual(games: pd.DataFrame) -> pd.DataFrame:
    counts = totals(games)
    actual = rates_from_counts(counts)
    actual["bf"] = counts["bf"].astype(float)
    actual["ip"] = counts["outs"] / 3
    return actual


def score_span(
    games: pd.DataFrame,
    season: int,
    before: pd.Timestamp,
    systems: dict[str, pd.DataFrame],
    min_bf: float,
    denoms: dict[Category, float],
) -> pd.DataFrame | None:
    """Score the baselines and ``systems`` (plus their blend, with two or more) on
    ``season``'s games from ``before`` on, the baselines knowing only the games before
    it. None when no pitcher reaches ``min_bf`` (a season not played yet)."""
    in_season = games[games["season"] == season]
    actual = _actual(in_season[in_season["game_date"] >= before])
    if (actual["bf"] >= min_bf).sum() == 0:
        return None
    projections = {**baseline_predictions(actual.index, games, season, before), **systems}
    if len(systems) > 1:
        projections[BLEND] = blend(systems)
    scored = scored_players(projections, actual, min_bf)

    earlier = games[games["game_date"] < before]
    career_bf = earlier.groupby("player_id")["bf"].sum()
    scored["career_bf"] = scored["player_id"].map(career_bf).fillna(0.0)
    vet = scored["career_bf"] >= VET_MIN_CAREER_BF
    known = season - int(games["season"].min()) >= MIN_HISTORY_SEASONS
    scored["group"] = "vet"
    scored.loc[~vet, "group"] = "rookie" if known else "unknown"

    if systems:
        values = pd.concat([fantasy_value(p, denoms) for p in systems.values()], axis=1)
        top = values.mean(axis=1).nlargest(RELEVANT_TOP).index
        top = top.union(fantasy_value(actual, denoms).nlargest(RELEVANT_TOP).index)
        scored["relevant"] = pd.array(scored["player_id"].isin(top), dtype="boolean")
    else:
        scored["relevant"] = pd.array([pd.NA] * len(scored), dtype="boolean")
    return scored


def preseason(
    games: pd.DataFrame,
    season: int,
    projections_dir: Path,
    denoms: dict[Category, float] | None = None,
) -> pd.DataFrame | None:
    """Score the whole of ``season`` against the preseason files (and the baselines,
    which know every earlier season). ``denoms``: SGP denominators for the
    ``relevant`` tag (None: the code defaults)."""
    systems = load_systems(projections_dir / str(season), preseason=True)
    before = pd.Timestamp(season, 1, 1)
    denoms = denoms if denoms is not None else get_sgp_denominators()
    scored = score_span(games, season, before, systems, PRESEASON_MIN_BF, denoms)
    return None if scored is None else scored.assign(season=season)


def snapshots(
    games: pd.DataFrame,
    season: int,
    projections_dir: Path,
    denoms: dict[Category, float] | None = None,
) -> pd.DataFrame | None:
    """Score every dated ROS snapshot of ``season`` that has pitcher files."""
    root = projections_dir / str(season) / "rest_of_season"
    if not root.is_dir():
        return None
    denoms = denoms if denoms is not None else get_sgp_denominators()
    parts = []
    for snap_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        systems = load_systems(snap_dir)
        if not systems:
            continue
        snap = pd.Timestamp(date.fromisoformat(snap_dir.name))
        scored = score_span(games, season, snap, systems, SNAPSHOT_MIN_BF, denoms)
        if scored is not None:
            parts.append(scored.assign(season=season, snapshot=snap.date().isoformat()))
    return pd.concat(parts, ignore_index=True) if parts else None


def systems_in_every_unit(scored: pd.DataFrame, unit: str) -> pd.DataFrame:
    """The rows of systems scored in every season (or snapshot), so their scores are
    over the same units and compare down a column."""
    counts = scored.groupby("system")[unit].nunique()
    return scored[scored["system"].isin(counts.index[counts == scored[unit].nunique()])]


def _vs_blend_lines(frame: pd.DataFrame) -> list[str]:
    """One line per system: its MSE minus the blend's per stat, with a 95% interval and
    how sure it is better or worse."""
    systems = list(dict.fromkeys(frame["system"]))
    if BLEND not in systems:
        return []
    lines = []
    for name in systems:
        if name == BLEND:
            continue
        b = mse_bootstrap(frame, name, BLEND)
        cells = "; ".join(
            f"{s} {r['diff']:+.3g} [{r['lo']:+.3g}, {r['hi']:+.3g}] {sure_text(r['sure'])}"
            for s, r in b.iterrows()
        )
        lines.append(f"* {name} - {BLEND}: {cells}")
    return ["", "MSE minus the blend's (negative = better), 95% interval, how sure:", "", *lines]


def _table(frame: pd.DataFrame) -> str:
    return to_markdown(mse_table(frame), digits=3)


def _typical_miss(frame: pd.DataFrame) -> str:
    """Square root of MSE, back in the stat's units: K/W/SV per pitcher over his
    actual batters faced, ERA/WHIP per inning-weighted pitcher."""
    t = mse_table(frame)
    return to_markdown(t[list(TARGETS)] ** 0.5, digits=2)


def _pooled_block(frame: pd.DataFrame, unit: str, title: str) -> list[str]:
    n = frame.drop_duplicates([unit, "player_id"]).groupby(unit).size().mean()
    return [
        "",
        f"**{title}** ({n:.0f} pitchers per {unit}; MSE pooled over the {unit}s)",
        "",
        _table(frame),
        "",
        "Typical miss (square root of MSE):",
        "",
        _typical_miss(frame),
        *_vs_blend_lines(frame),
    ]


def _tag_blocks(frame: pd.DataFrame, unit: str) -> list[str]:
    md = []
    rel = frame[frame["relevant"].fillna(False).astype(bool)]
    if not rel.empty:
        md += _pooled_block(
            rel,
            unit,
            f"Fantasy-relevant pitchers (top {RELEVANT_TOP} by FanGraphs' projected value "
            "or by actual value)",
        )
    for group in GROUPS:
        g = frame[frame["group"] == group]
        if not g.empty:
            label = (
                f"{VET_MIN_CAREER_BF}+ MLB batters faced"
                if group == "vet"
                else f"under {VET_MIN_CAREER_BF}"
            )
            md += _pooled_block(g, unit, f"{group.capitalize()}s ({label} when projected)")
    unknown = frame.loc[frame["group"] == "unknown", [unit, "player_id"]].drop_duplicates()
    if len(unknown):
        md += [
            "",
            f"{len(unknown)} pitcher-{unit}s are in neither group: under {VET_MIN_CAREER_BF} "
            f"BF with under {MIN_HISTORY_SEASONS} earlier seasons in the store.",
        ]
    return md


def summarize(pre: pd.DataFrame | None, snap: pd.DataFrame | None) -> list[str]:
    """Markdown lines: per-season and per-snapshot MSE tables, then pooled tables with
    each system vs. the blend, for everyone, the fantasy-relevant pitchers, vets and
    rookies."""
    md = [
        "**Main score: MSE**, lower is better. K, W, SV: squared miss in season totals "
        "over the batters each pitcher actually faced. ERA, WHIP: squared miss in the "
        "stat's units, weighted by actual innings. Every system is scored on the same "
        f"pitchers: preseason, those with >= {PRESEASON_MIN_BF} actual batters faced; "
        f"mid-season, >= {SNAPSHOT_MIN_BF} after the snapshot. `{BLEND}` = plain average "
        "of the FanGraphs systems' rates; `marcel` and `league_avg` are simple floors "
        "(pitcher_ros/baselines.py). Intervals resample pitchers; "
        '"N% sure better/worse" is how sure the system really differs from the blend; '
        "95%+ is marked real. It counts only the luck of which pitchers were scored.",
    ]
    if pre is not None:
        md += ["", "#### Preseason"]
        for season, g in pre.groupby("season"):
            md += ["", f"**{season}**", "", _table(g)]
        every = systems_in_every_unit(pre, "season")
        md += _pooled_block(every, "season", "All seasons, systems in every season")
        md += _tag_blocks(every, "season")
    if snap is not None:
        md += ["", "#### Mid-season (ROS snapshots)"]
        for snapshot, g in snap.groupby("snapshot"):
            md += ["", f"**{snapshot}**", "", _table(g)]
        every = systems_in_every_unit(snap, "snapshot")
        md += _pooled_block(every, "snapshot", "All snapshots")
        md += _tag_blocks(every, "snapshot")
    return md
