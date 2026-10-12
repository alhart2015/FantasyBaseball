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

Each season or snapshot is scored twice. Its own table has every FanGraphs system on
file for it. The pooled tables (over seasons, over snapshots) are scored again with only
the systems on file for every one of them, so ``fg_blend`` averages the same systems
throughout and every system is scored on the same pitchers; the summary names the
systems left out of them.

Every scored row is tagged like the hitters' (``hitter_ros/backtest.py``):

* ``group``: "vet" with at least ``VET_MIN_CAREER_BF`` MLB batters faced before the
  projection date, else "rookie"; "unknown" when the store (from 2008) holds fewer than
  ``MIN_HISTORY_SEASONS`` earlier seasons, so a vet could look like a rookie.
* ``relevant``: among the ``RELEVANT_TOP`` pitchers by fantasy value either in the
  FanGraphs projections compared against or in what actually happened over the span
  (``projection_eval.top_by_value``). ``RELEVANT_TOP`` is 150: this league starts 9
  pitchers on each of 10 teams.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from fantasy_baseball.hitter_ros.backtest import sure_text, to_markdown
from fantasy_baseball.pitcher_ros.baselines import baseline_predictions, league_rates
from fantasy_baseball.pitcher_ros.evaluate import (
    blend,
    fantasy_value,
    load_systems,
    mse_bootstrap,
    mse_table,
    scored_players,
)
from fantasy_baseball.pitcher_ros.stats import TARGETS, rates_from_counts, totals
from fantasy_baseball.projection_eval import common_index, systems_in_every_unit, top_by_value
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.constants import Category

logger = logging.getLogger(__name__)

PRESEASON_MIN_BF = 100
SNAPSHOT_MIN_BF = 40
BLEND = "fg_blend"
BASELINES = ("marcel", "league_avg")
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


def _only(
    systems: dict[str, pd.DataFrame], only: Collection[str] | None
) -> dict[str, pd.DataFrame]:
    return systems if only is None else {k: v for k, v in systems.items() if k in only}


def score_span(
    games: pd.DataFrame,
    season: int,
    before: pd.Timestamp,
    systems: dict[str, pd.DataFrame],
    min_bf: float,
    denoms: dict[Category, float],
    league: pd.Series | None = None,
    ours: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame | None:
    """Score ``ours`` (our models' projections, rates indexed by ``player_id``), the
    baselines and ``systems`` (plus their blend, with two or more) on ``season``'s games
    from ``before`` on, the baselines knowing only the games before it. None when no
    pitcher reaches ``min_bf`` (a season not played yet). ``league``:
    ``baselines.league_rates`` for ``season`` if already computed.

    Every system is scored on the same pitchers, so one of ``ours`` without a
    projection for a pitcher the others would score is an error, not a smaller sample
    for everyone; so is one of ``ours`` named like a baseline, the blend or a system."""
    in_season = games[games["season"] == season]
    actual = _actual(in_season[in_season["game_date"] >= before])
    if (actual["bf"] >= min_bf).sum() == 0:
        return None
    _check_ours(ours or {}, systems, actual, min_bf, f"{season} from {before.date()}")
    projections = {
        **(ours or {}),
        **baseline_predictions(actual.index, games, season, before, league),
        **systems,
    }
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
        projected = (fantasy_value(p, denoms) for p in systems.values())
        top = top_by_value(projected, fantasy_value(actual, denoms), RELEVANT_TOP)
        scored["relevant"] = pd.array(scored["player_id"].isin(top), dtype="boolean")
    else:
        scored["relevant"] = pd.array([pd.NA] * len(scored), dtype="boolean")
    return scored


def _check_ours(
    ours: dict[str, pd.DataFrame],
    systems: dict[str, pd.DataFrame],
    actual: pd.DataFrame,
    min_bf: float,
    span: str,
) -> None:
    clash = set(ours) & {*BASELINES, BLEND, *systems}
    if clash:
        raise ValueError(f"our projections are named like other systems: {sorted(clash)}")
    scored = actual.index[actual["bf"] >= min_bf]
    if systems:
        scored = scored.intersection(common_index(systems.values(), TARGETS))
    for name, p in ours.items():
        missing = scored.difference(p.dropna(subset=list(TARGETS)).index)
        if len(missing):
            raise ValueError(
                f"{name} has no projection for {len(missing)} of the {len(scored)} "
                f"pitchers scored in {span}, e.g. player_id {list(missing[:5])}"
            )


def _ours_at(
    ours: dict[str, pd.DataFrame] | None, season: int, when: pd.Timestamp | None
) -> dict[str, pd.DataFrame]:
    """Each of ``ours`` (prediction frames: ``player_id``, ``season``, ``week``,
    ``as_of`` and the ``TARGETS``; one row per pitcher and as-of date) as known at
    ``when``: each pitcher's row from the latest as-of date on or before it in
    ``season``, or week 0 with ``when`` None."""
    out = {}
    for name, p in (ours or {}).items():
        rows = p[p["season"] == season]
        if when is None:
            rows = rows[rows["week"] == 0]
        else:
            rows = rows[rows["as_of"] <= when]
            rows = rows.loc[rows.groupby("player_id")["as_of"].idxmax()]
        if rows["player_id"].duplicated().any():
            raise ValueError(f"{name}: more than one {season} row per pitcher and date")
        out[name] = rows.set_index("player_id")[list(TARGETS)]
    return out


def preseason(
    games: pd.DataFrame,
    season: int,
    projections_dir: Path,
    denoms: dict[Category, float] | None = None,
    only: Collection[str] | None = None,
    ours: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame | None:
    """Score the whole of ``season`` against the preseason files (and the baselines,
    which know every earlier season). ``denoms``: SGP denominators for the
    ``relevant`` tag (None: the code defaults). ``only``: score just these FanGraphs
    systems (None: every one on file). ``ours``: our prediction frames (see
    :func:`_ours_at`); their week-0 rows are scored."""
    systems = _only(load_systems(projections_dir / str(season), preseason=True), only)
    before = pd.Timestamp(season, 1, 1)
    denoms = denoms if denoms is not None else get_sgp_denominators()
    scored = score_span(
        games,
        season,
        before,
        systems,
        PRESEASON_MIN_BF,
        denoms,
        ours=_ours_at(ours, season, None),
    )
    return None if scored is None else scored.assign(season=season)


def snapshots(
    games: pd.DataFrame,
    season: int,
    projections_dir: Path,
    denoms: dict[Category, float] | None = None,
    only: Collection[str] | None = None,
    ours: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame | None:
    """Score every dated ROS snapshot of ``season`` that has pitcher files; ``denoms``,
    ``only`` and ``ours`` as for :func:`preseason` (ours: each pitcher's row from the
    latest as-of date on or before the snapshot). A folder not named for a date is
    skipped, with a warning."""
    root = projections_dir / str(season) / "rest_of_season"
    if not root.is_dir():
        return None
    denoms = denoms if denoms is not None else get_sgp_denominators()
    league = league_rates(games, season)  # the same for every snapshot of the season
    parts = []
    for snap_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            snap = pd.Timestamp(date.fromisoformat(snap_dir.name))
        except ValueError:
            logger.warning("%s: not named for a date (YYYY-MM-DD), skipped", snap_dir)
            continue
        systems = _only(load_systems(snap_dir), only)
        if not systems:
            continue
        scored = score_span(
            games,
            season,
            snap,
            systems,
            SNAPSHOT_MIN_BF,
            denoms,
            league,
            ours=_ours_at(ours, season, snap),
        )
        if scored is not None:
            parts.append(scored.assign(season=season, snapshot=snap.date().isoformat()))
    return pd.concat(parts, ignore_index=True) if parts else None


def fangraphs_systems(scored: pd.DataFrame, ours: Iterable[str] = ()) -> list[str]:
    """The FanGraphs systems in ``scored`` (not ``ours``, the baselines or the blend),
    in order."""
    skip = {*BASELINES, BLEND, *ours}
    return [s for s in dict.fromkeys(scored["system"]) if s not in skip]


def systems_in_every(scored: pd.DataFrame, unit: str, ours: Iterable[str] = ()) -> list[str]:
    """The FanGraphs systems scored in every season (or snapshot) of ``scored``."""
    return fangraphs_systems(systems_in_every_unit(scored, unit), ours)


def _concat(parts: Iterable[pd.DataFrame | None]) -> pd.DataFrame | None:
    frames = [p for p in parts if p is not None]
    return pd.concat(frames, ignore_index=True) if frames else None


@dataclass
class Scores:
    """Scored frames for a set of seasons. ``pre`` and ``snap``: every system on file
    for each season or snapshot. ``pre_pooled`` and ``snap_pooled``: the same, scored
    with only the systems on file for every one, for the pooled tables. ``failures``:
    the seasons that couldn't be scored, with why. ``ours``: the names of our models
    among the systems."""

    pre: pd.DataFrame | None = None
    snap: pd.DataFrame | None = None
    pre_pooled: pd.DataFrame | None = None
    snap_pooled: pd.DataFrame | None = None
    failures: list[str] = field(default_factory=list)
    ours: list[str] = field(default_factory=list)


def score_seasons(
    games: pd.DataFrame,
    seasons: Iterable[int],
    projections_dir: Path,
    denoms: dict[Category, float] | None = None,
    ours: dict[str, pd.DataFrame] | None = None,
) -> Scores:
    """Score every season in ``seasons`` (preseason and its snapshots), with ``ours``
    (our prediction frames by name, see :func:`_ours_at`) beside FanGraphs and the
    baselines in every table. A season that
    fails (a bad file, no earlier season in the store for the baselines) is logged and
    listed in ``failures``, and the rest are still scored. A season with no games yet
    is skipped."""
    out = Scores(ours=list(ours or {}))
    pre: dict[int, pd.DataFrame] = {}
    snap: dict[int, pd.DataFrame] = {}
    for season in sorted(seasons):
        try:
            p = preseason(games, season, projections_dir, denoms, ours=ours)
            if p is None:
                logger.info("%s: no games played yet, skipped", season)
                continue
            s = snapshots(games, season, projections_dir, denoms, ours=ours)
        except (ValueError, KeyError) as exc:
            logger.exception("%s: not scored", season)
            out.failures.append(f"{season}: {exc}")
            continue
        pre[season] = p
        if s is not None:
            snap[season] = s
    out.pre, out.snap = _concat(pre.values()), _concat(snap.values())
    if out.pre is not None:
        core = systems_in_every(out.pre, "season", out.ours)
        out.pre_pooled = _concat(
            preseason(games, season, projections_dir, denoms, only=core, ours=ours)
            for season in pre
        )
    if out.snap is not None:
        core = systems_in_every(out.snap, "snapshot", out.ours)
        out.snap_pooled = _concat(
            snapshots(games, season, projections_dir, denoms, only=core, ours=ours)
            for season in snap
        )
    return out


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


def _pooled_note(
    every: pd.DataFrame, pooled: pd.DataFrame, unit: str, ours: Iterable[str] = ()
) -> str:
    """Which FanGraphs systems the pooled tables use, and which they leave out."""
    used = fangraphs_systems(pooled, ours)
    n = every[unit].nunique()
    left_out = [
        f"{s} (on file for {every.loc[every['system'] == s, unit].nunique()} of {n})"
        for s in fangraphs_systems(every, ours)
        if s not in used
    ]
    note = f"Pooled tables use the FanGraphs systems on file for every {unit}: " + (
        ", ".join(used) or "none"
    )
    if BLEND in set(pooled["system"]):
        note += f"; `{BLEND}` here is their average"
    if left_out:
        note += ". Left out: " + ", ".join(left_out)
    return note + "."


def summarize(scores: Scores) -> list[str]:
    """Markdown lines: per-season and per-snapshot MSE tables (every system on file),
    then pooled tables (the systems on file throughout) with each system vs. the blend,
    for everyone, the fantasy-relevant pitchers, vets and rookies."""
    md = [
        "**Main score: MSE**, lower is better. K, W, SV: squared miss in season totals "
        "over the batters each pitcher actually faced. ERA, WHIP: squared miss in the "
        "stat's units, weighted by actual innings. Every system in a table is scored on "
        f"the same pitchers: preseason, those with >= {PRESEASON_MIN_BF} actual batters "
        f"faced; mid-season, >= {SNAPSHOT_MIN_BF} after the snapshot. `{BLEND}` = plain "
        "average of the FanGraphs systems' rates; `marcel` and `league_avg` are simple "
        "floors (pitcher_ros/baselines.py). Intervals resample pitchers; "
        '"N% sure better/worse" is how sure the system really differs from the blend; '
        "95%+ is marked real. It counts only the luck of which pitchers were scored.",
    ]
    if scores.failures:
        md += ["", "**Not scored:** " + "; ".join(scores.failures)]
    for every, pooled, unit, heading, title in (
        (scores.pre, scores.pre_pooled, "season", "Preseason", "All seasons"),
        (
            scores.snap,
            scores.snap_pooled,
            "snapshot",
            "Mid-season (ROS snapshots)",
            "All snapshots",
        ),
    ):
        if every is None:
            continue
        md += ["", f"#### {heading}"]
        for name, g in every.groupby(unit):
            md += ["", f"**{name}**", "", _table(g)]
        if pooled is not None:
            md += ["", _pooled_note(every, pooled, unit, scores.ours)]
            md += _pooled_block(pooled, unit, title)
            md += _tag_blocks(pooled, unit)
    return md
