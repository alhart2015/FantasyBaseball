"""Score a run's predictions against FanGraphs and the simple baselines (#404, #410).

Given the net's predictions for some test seasons, this builds the long "scored" frames
(one row per player x system x stat, tagged with ``season`` and, mid-season,
``snapshot``) for:

* Preseason (week 0) vs. each FanGraphs system in ``data/projections/T/``, their blend,
  and the ``league_avg`` / ``marcel`` baselines.
* Mid-season vs. each dated ROS snapshot in ``data/projections/T/rest_of_season/<date>/``,
  using each projection's row from the latest as-of date on or before the snapshot;
  actuals are the games on or after the snapshot date.

and renders a markdown summary with MAE tables, a paired bootstrap of ours vs. the
FanGraphs blend, and how spread out each system's projections are.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from fantasy_baseball.hitter_ros.baselines import baseline_predictions
from fantasy_baseball.hitter_ros.evaluate import (
    blend,
    load_systems,
    mae_table,
    paired_bootstrap,
    scored_players,
    spread,
)
from fantasy_baseball.hitter_ros.features import COUNTS, TARGETS, rates_from_counts
from fantasy_baseball.pitch_data.store import connect

PRESEASON_MIN_PA = 300
SNAPSHOT_MIN_PA = 100
OURS = "ours"
BLEND = "fg_blend"


def _with_fangraphs(
    ours: dict[str, pd.DataFrame], systems: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    projections = {**ours, **systems}
    if len(systems) > 1:
        projections[BLEND] = blend(systems)
    return projections


def preseason(
    table: pd.DataFrame,
    candidates: dict[str, pd.DataFrame],
    season: int,
    projections_dir: Path,
) -> pd.DataFrame | None:
    """Score week-0 rows of ``season``. ``candidates``: our predictions and baselines.

    Seasons without FanGraphs files (before 2022) are still scored, on ours and the
    baselines only, so our own variants can be compared over many more seasons.
    """
    folder = projections_dir / str(season)
    systems = load_systems(folder, preseason=True) if folder.is_dir() else {}
    week0 = table[(table["season"] == season) & (table["week"] == 0)].set_index("player_id")
    actual = rates_from_counts(week0.rename(columns=lambda c: c.removeprefix("ros_")))
    actual["pa"] = week0["ros_pa"]
    ours = {
        name: p[p["week"] == 0].set_index("player_id")[list(TARGETS)]
        for name, p in candidates.items()
    }
    scored = scored_players(_with_fangraphs(ours, systems), actual, PRESEASON_MIN_PA)
    return scored.assign(season=season)


def _season_games(store: Path, season: int) -> pd.DataFrame:
    conn = connect(store)
    try:
        games = conn.execute(
            f"""
            SELECT player_id, CAST(game_date AS DATE) AS game_date, {", ".join(COUNTS)}
            FROM lineups WHERE year(CAST(game_date AS DATE)) = ?
            """,
            [season],
        ).df()
    finally:
        conn.close()
    games["game_date"] = pd.to_datetime(games["game_date"])
    return games


def snapshots(
    candidates: dict[str, pd.DataFrame],
    season: int,
    projections_dir: Path,
    store: Path,
) -> pd.DataFrame | None:
    """Score every dated ROS snapshot of ``season``."""
    root = projections_dir / str(season) / "rest_of_season"
    if not root.is_dir():
        return None
    games = _season_games(store, season)
    by_date = {name: p.sort_values("as_of") for name, p in candidates.items()}
    parts = []
    for snap_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        snap = pd.Timestamp(date.fromisoformat(snap_dir.name))
        systems = load_systems(snap_dir)
        if not systems:
            continue
        counts = games[games["game_date"] >= snap].groupby("player_id")[list(COUNTS)].sum()
        actual = rates_from_counts(counts)
        actual["pa"] = counts["pa"]
        if (actual["pa"] >= SNAPSHOT_MIN_PA).sum() == 0:
            continue
        ours = {}
        for name, p in by_date.items():
            known = p[p["as_of"] <= snap]
            ours[name] = known.groupby("player_id").tail(1).set_index("player_id")[list(TARGETS)]
        scored = scored_players(_with_fangraphs(ours, systems), actual, SNAPSHOT_MIN_PA)
        parts.append(scored.assign(season=season, snapshot=snap.date().isoformat()))
    return pd.concat(parts, ignore_index=True) if parts else None


def score_predictions(
    table: pd.DataFrame, preds: pd.DataFrame, projections_dir: Path, store: Path
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(preseason, snapshots) scored frames for every season in ``preds``."""
    pre_parts, snap_parts = [], []
    for season in sorted(int(s) for s in preds["season"].unique()):
        candidates = {OURS: preds[preds["season"] == season], **baseline_predictions(table, season)}
        pre = preseason(table, candidates, season, projections_dir)
        if pre is not None:
            pre_parts.append(pre)
        snap = snapshots(candidates, season, projections_dir, store)
        if snap is not None:
            snap_parts.append(snap)
    return (
        pd.concat(pre_parts, ignore_index=True) if pre_parts else None,
        pd.concat(snap_parts, ignore_index=True) if snap_parts else None,
    )


def write_scores(run_dir: Path, pre: pd.DataFrame | None, snap: pd.DataFrame | None) -> None:
    """Write the scored frames; a kind with nothing scored now has its old file removed."""
    for frame, name in ((pre, "scored_preseason.parquet"), (snap, "scored_snapshots.parquet")):
        path = run_dir / name
        if frame is not None:
            frame.to_parquet(path)
        else:
            path.unlink(missing_ok=True)


def systems_in_every_season(pre: pd.DataFrame) -> pd.DataFrame:
    """The preseason rows of systems scored in every season (so their means compare)."""
    n_seasons = pre["season"].nunique()
    counts = pre.groupby("system")["season"].nunique()
    return pre[pre["system"].isin(counts.index[counts == n_seasons])]


def mean_over_seasons(pre: pd.DataFrame) -> pd.DataFrame:
    """Per-system MAE averaged over seasons (each season counts once), systems x stats."""
    per_season = pre.groupby(["season", "system", "stat"])["abs_err"].mean()
    table = per_season.groupby(level=["system", "stat"]).mean().unstack("stat")
    order = [s for s in dict.fromkeys(pre["system"]) if s in table.index]
    return table.loc[order, list(TARGETS)]


def to_markdown(df: pd.DataFrame, digits: int = 2) -> str:
    cols = list(df.columns)
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, row in df.iterrows():
        cells = [
            str(int(v)) if c == "n" else f"{v:.{digits}f}" if isinstance(v, float) else str(v)
            for c, v in row.items()
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _bootstrap_line(scored: pd.DataFrame) -> str:
    if BLEND not in set(scored["system"]):
        return ""
    b = paired_bootstrap(scored, OURS, BLEND)
    cells = [f"{s} {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]" for s, r in b.iterrows()]
    return "ours - fg_blend, 95% interval: " + "; ".join(cells)


def summarize(pre: pd.DataFrame | None, snap: pd.DataFrame | None) -> list[str]:
    """Markdown lines: per-season and per-snapshot MAE tables with bootstrap and spread."""
    md = [
        "Raw error, lower is better: R/HR/RBI/SB = MAE per 600 PA, AVG = MAE in points. "
        f"Preseason: players with >= {PRESEASON_MIN_PA} actual PA. "
        f"Mid-season: >= {SNAPSHOT_MIN_PA} PA after the snapshot. "
        "`league_avg` and `marcel` are simple floors (see hitter_ros/baselines.py). "
        "Bootstrap: negative = ours better; an interval crossing 0 = can't tell apart."
    ]
    if pre is not None:
        md += ["", "#### Preseason"]
        for season, g in pre.groupby("season"):
            md += ["", f"**{season}**", "", to_markdown(mae_table(g)), "", _bootstrap_line(g)]
        pooled = systems_in_every_season(pre)
        md += ["", "**Mean over seasons** (systems present every season)", ""]
        md.append(to_markdown(mean_over_seasons(pooled)))
        md += [
            "",
            "**Spread of projections** (SD across scored player-seasons; '(actual)' = outcomes)",
            "",
        ]
        md.append(to_markdown(spread(pooled)))
    if snap is not None:
        md += ["", "#### Mid-season (ROS snapshots)"]
        for snapshot, g in snap.groupby("snapshot"):
            md += ["", f"**{snapshot}**", "", to_markdown(mae_table(g)), "", _bootstrap_line(g)]
    return md
