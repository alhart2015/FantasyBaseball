"""Score a run's predictions against FanGraphs and the simple baselines (#404, #410).

Given the net's predictions for some test seasons, this builds the long "scored" frames
(one row per player x system x stat) for:

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
from fantasy_baseball.hitter_ros.features import TARGETS, rates_from_counts
from fantasy_baseball.pitch_data.store import connect

PRESEASON_MIN_PA = 300
SNAPSHOT_MIN_PA = 100
OURS = "ours"
BLEND = "fg_blend"
COUNT_COLS = ["pa", "ab", "h", "r", "hr", "rbi", "sb"]


def _ours_and_baselines(
    table: pd.DataFrame, preds: pd.DataFrame, season: int
) -> dict[str, pd.DataFrame]:
    return {OURS: preds[preds["season"] == season], **baseline_predictions(table, season)}


def _with_fangraphs(
    ours: dict[str, pd.DataFrame], systems: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    projections = {**ours, **systems}
    if len(systems) > 1:
        projections[BLEND] = blend(systems)
    return projections


def preseason(
    table: pd.DataFrame, preds: pd.DataFrame, season: int, projections_dir: Path
) -> pd.DataFrame | None:
    systems = load_systems(projections_dir / str(season), preseason=True)
    if not systems:
        return None
    week0 = table[(table["season"] == season) & (table["week"] == 0)].set_index("player_id")
    actual = rates_from_counts(week0.rename(columns=lambda c: c.removeprefix("ros_")))
    actual["pa"] = week0["ros_pa"]
    ours = {
        name: p[p["week"] == 0].set_index("player_id")[list(TARGETS)]
        for name, p in _ours_and_baselines(table, preds, season).items()
    }
    scored = scored_players(_with_fangraphs(ours, systems), actual, PRESEASON_MIN_PA)
    return scored.assign(season=season)


def snapshots(
    table: pd.DataFrame,
    preds: pd.DataFrame,
    season: int,
    projections_dir: Path,
    store: Path,
) -> pd.DataFrame | None:
    root = projections_dir / str(season) / "rest_of_season"
    if not root.is_dir():
        return None
    conn = connect(store)
    games = conn.execute(
        """
        SELECT player_id, CAST(game_date AS DATE) AS game_date, pa, ab, h, r, hr, rbi, sb
        FROM lineups WHERE year(CAST(game_date AS DATE)) = ?
        """,
        [season],
    ).df()
    conn.close()
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    candidates = _ours_and_baselines(table, preds, season)
    parts = []
    for snap_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        snap = date.fromisoformat(snap_dir.name)
        systems = load_systems(snap_dir)
        if not systems:
            continue
        counts = games[games["game_date"] >= snap].groupby("player_id")[COUNT_COLS].sum()
        actual = rates_from_counts(counts)
        actual["pa"] = counts["pa"]
        if (actual["pa"] >= SNAPSHOT_MIN_PA).sum() == 0:
            continue
        ours = {}
        for name, p in candidates.items():
            known = p[p["as_of"].dt.date <= snap].sort_values("as_of")
            ours[name] = known.groupby("player_id").tail(1).set_index("player_id")[list(TARGETS)]
        scored = scored_players(_with_fangraphs(ours, systems), actual, SNAPSHOT_MIN_PA)
        parts.append(scored.assign(season=season, snapshot=snap.isoformat()))
    return pd.concat(parts, ignore_index=True) if parts else None


def score_predictions(
    table: pd.DataFrame, preds: pd.DataFrame, projections_dir: Path, store: Path
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(preseason, snapshots) scored frames for every season in ``preds``."""
    pre_parts, snap_parts = [], []
    for season in sorted(int(s) for s in preds["season"].unique()):
        pre = preseason(table, preds, season, projections_dir)
        if pre is not None:
            pre_parts.append(pre)
        snap = snapshots(table, preds, season, projections_dir, store)
        if snap is not None:
            snap_parts.append(snap)
    return (
        pd.concat(pre_parts, ignore_index=True) if pre_parts else None,
        pd.concat(snap_parts, ignore_index=True) if snap_parts else None,
    )


def write_scores(run_dir: Path, pre: pd.DataFrame | None, snap: pd.DataFrame | None) -> None:
    if pre is not None:
        pre.to_parquet(run_dir / "scored_preseason.parquet")
    if snap is not None:
        snap.to_parquet(run_dir / "scored_snapshots.parquet")


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
        common = [
            s
            for s, n in pre.groupby("system")["season"].nunique().items()
            if n == pre["season"].nunique()
        ]
        pooled = pre[pre["system"].isin(common)]
        means = pooled.groupby(["season", "system"]).apply(
            lambda g: g.groupby("stat")["abs_err"].mean(), include_groups=False
        )
        mean_table = means.groupby(level="system").mean()[list(TARGETS)]
        order = [s for s in dict.fromkeys(pooled["system"]) if s in mean_table.index]
        md += ["", "**Mean over seasons** (systems present every season)", ""]
        md.append(to_markdown(mean_table.loc[order]))
        md += [
            "",
            "**Spread of projections** (SD across scored players; '(actual)' = outcomes)",
            "",
        ]
        md.append(to_markdown(spread(pooled)))
    if snap is not None:
        md += ["", "#### Mid-season (ROS snapshots)"]
        for snapshot, g in snap.groupby("snapshot"):
            md += ["", f"**{snapshot}**", "", to_markdown(mae_table(g)), "", _bootstrap_line(g)]
    return md
