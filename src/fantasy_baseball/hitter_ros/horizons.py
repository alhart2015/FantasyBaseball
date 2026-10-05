"""Score the short-horizon heads (#419): the next 25 / 100 / 250 PA and rest of season.

For every mid-season row (week 1+) of the test seasons that has an answer for a
horizon, each "system" gives a rate for that hitter's next N PA, scored against what
he did over them:

* ``head``: the net's own head for that horizon (for rest of season, the usual output).
* ``ros_rate``: the net's rest-of-season rate used for the next N PA -- the bar a
  horizon head has to beat to be worth having.
* ``marcel``: the Marcel-style baseline (``baselines.py``).
* ``hot_hand``: his season-to-date rate, the naive "he's hot, he'll stay hot".

Players are compared only within the same season and week (a unit), as in the snapshot
scores, and every score is averaged over units. Only rows where he got at least N PA
(rest of season: ``ROS_MIN_PA``) and every system has a rate are scored.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fantasy_baseball.hitter_ros.backtest import mean_over_seasons, to_markdown
from fantasy_baseball.hitter_ros.baselines import baseline_predictions
from fantasy_baseball.hitter_ros.evaluate import order_scores, order_table, scored_players
from fantasy_baseball.hitter_ros.features import COUNTS, TARGETS, rates_from_counts
from fantasy_baseball.hitter_ros.table import HORIZONS

ROS_MIN_PA = 100
HORIZON_NAMES = (*(f"n{n}" for n in HORIZONS), "ros")


def _actual(rows: pd.DataFrame, horizon: str) -> pd.DataFrame:
    prefix = "ros_" if horizon == "ros" else f"ros_{horizon}_"
    counts = rows[[f"{prefix}{c}" for c in COUNTS]].rename(columns=lambda c: c.removeprefix(prefix))
    actual = rates_from_counts(counts)
    actual["pa"] = counts["pa"]
    return actual


def score_horizons(table: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame | None:
    """One scored frame (``evaluate.scored_players`` rows) for every horizon, tagged with
    ``horizon``, ``season`` and ``snapshot`` (the season and week). None when ``preds``
    has no horizon heads."""
    if f"n{HORIZONS[0]}_{TARGETS[0]}" not in preds.columns:
        return None
    keys = ["player_id", "season", "week"]
    parts = []
    for season in sorted(int(s) for s in preds["season"].unique()):
        rows = table[(table["season"] == season) & (table["week"] > 0)]
        mine = preds[(preds["season"] == season) & (preds["week"] > 0)]
        marcel = baseline_predictions(table, season)["marcel"]
        marcel = marcel[marcel["week"] > 0]
        std = rows[[f"std_{c}" for c in COUNTS]].rename(columns=lambda c: c.removeprefix("std_"))
        hot = pd.concat([rows[keys], rates_from_counts(std)], axis=1)
        for week, wrows in rows.groupby("week"):
            index = wrows.set_index("player_id").index

            def by_player(
                frame: pd.DataFrame, cols: list[str], week: int = week, index: pd.Index = index
            ) -> pd.DataFrame:
                f = frame[frame["week"] == week].set_index("player_id")[cols]
                return f.set_axis(list(TARGETS), axis=1).reindex(index)

            for horizon in HORIZON_NAMES:
                head_cols = (
                    list(TARGETS) if horizon == "ros" else [f"{horizon}_{s}" for s in TARGETS]
                )
                systems = {
                    "head": by_player(mine, head_cols),
                    "marcel": by_player(marcel, list(TARGETS)),
                    "hot_hand": by_player(hot, list(TARGETS)),
                }
                if horizon != "ros":
                    systems["ros_rate"] = by_player(mine, list(TARGETS))
                actual = _actual(wrows, horizon).set_axis(index)
                min_pa = ROS_MIN_PA if horizon == "ros" else int(horizon[1:])
                scored = scored_players(systems, actual.dropna(subset=["pa"]), min_pa)
                if scored.empty:
                    continue
                parts.append(
                    scored.assign(
                        horizon=horizon, season=season, snapshot=f"{season}-w{int(week):02d}"
                    )
                )
    return pd.concat(parts, ignore_index=True) if parts else None


def write_horizon_scores(run_dir: Path, scored: pd.DataFrame | None) -> None:
    """Write ``scored_horizons.parquet``; remove an old one when there's nothing scored."""
    path = run_dir / "scored_horizons.parquet"
    if scored is not None:
        scored.to_parquet(path)
    else:
        path.unlink(missing_ok=True)


def horizon_summary(scored: pd.DataFrame) -> list[str]:
    """Markdown: per horizon, every score averaged over (season, week) units."""
    md = [
        "#### Short horizons (#419), mid-season rows (week 1+)",
        "",
        "Each system's rate for a hitter's next N PA vs what he did, players compared "
        "within the same season and week, averaged over those units. `head` = this "
        "horizon's own head; `ros_rate` = our rest-of-season rate used for the next N PA; "
        "`hot_hand` = his season-to-date rate.",
    ]
    for horizon in HORIZON_NAMES:
        g = scored[scored["horizon"] == horizon]
        if g.empty:
            continue
        per_unit = order_scores(g)
        units = g["snapshot"].nunique()
        md += [
            "",
            f"**{horizon}** ({units} season-weeks, {len(g) // len(TARGETS) // g['system'].nunique():,} player-rows)",
            "",
            "Gap-weighted pairwise accuracy (%) -- main score:",
            "",
            to_markdown(order_table(g, "pairwise_w", per_unit), digits=1),
            "",
            "Pairwise accuracy (%):",
            "",
            to_markdown(order_table(g, "pairwise", per_unit), digits=1),
            "",
            "Raw MAE (per 600 PA; AVG in points):",
            "",
            to_markdown(mean_over_seasons(g, unit="snapshot")),
        ]
    return md
