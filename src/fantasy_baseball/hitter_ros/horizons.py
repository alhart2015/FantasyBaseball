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
scores. MSE (the main score, ``evaluate``; totals over the next N PA) is pooled over
every scored row; the other scores are averaged over units. Only rows where he got at least N PA
(rest of season: ``ROS_MIN_PA``) and every system has a rate are scored.

**vs FanGraphs** (``projections_dir``; only seasons with dated ROS snapshots, i.e. 2026):
a separate set (``comparison == "fangraphs"``) for weeks with a snapshot from the
``FG_FRESH_DAYS`` days up to and including the as-of date. The FanGraphs rest-of-season
rate (the blend when there are several systems) is used for the next N PA, scored with
``head`` and ``ros_rate`` on the players all three cover.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fantasy_baseball.hitter_ros.backtest import mean_over_seasons, to_markdown
from fantasy_baseball.hitter_ros.baselines import baseline_predictions
from fantasy_baseball.hitter_ros.evaluate import (
    blend,
    load_systems,
    mse_table,
    order_scores,
    order_table,
    scored_players,
)
from fantasy_baseball.hitter_ros.features import COUNTS, TARGETS, rates_from_counts
from fantasy_baseball.hitter_ros.table import HORIZONS

ROS_MIN_PA = 100
FG_FRESH_DAYS = 7
HORIZON_NAMES = (*(f"n{n}" for n in HORIZONS), "ros")


def _actual(rows: pd.DataFrame, horizon: str) -> pd.DataFrame:
    prefix = "ros_" if horizon == "ros" else f"ros_{horizon}_"
    counts = rows[[f"{prefix}{c}" for c in COUNTS]].rename(columns=lambda c: c.removeprefix(prefix))
    actual = rates_from_counts(counts)
    actual["pa"] = counts["pa"]
    actual["ab"] = counts["ab"]
    return actual


def _fg_snapshots(projections_dir: Path | None, season: int) -> dict[pd.Timestamp, pd.DataFrame]:
    """FanGraphs rest-of-season rates per snapshot date (the blend of the systems there)."""
    root = None if projections_dir is None else projections_dir / str(season) / "rest_of_season"
    if root is None or not root.is_dir():
        return {}
    out = {}
    for snap_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        systems = load_systems(snap_dir)
        if systems:
            rates = blend(systems) if len(systems) > 1 else next(iter(systems.values()))
            out[pd.Timestamp(snap_dir.name)] = rates[list(TARGETS)]
    return out


def score_horizons(
    table: pd.DataFrame, preds: pd.DataFrame, projections_dir: Path | None = None
) -> pd.DataFrame | None:
    """One scored frame (``evaluate.scored_players`` rows) for every horizon, tagged with
    ``horizon``, ``season``, ``snapshot`` (the season and week) and ``comparison``
    (``"all"``, or ``"fangraphs"`` for the FanGraphs set, see the module doc). None when
    ``preds`` has no horizon heads."""
    if f"n{HORIZONS[0]}_{TARGETS[0]}" not in preds.columns:
        return None
    missing = [f"ros_n{n}_pa" for n in HORIZONS if f"ros_n{n}_pa" not in table.columns]
    if missing:
        raise ValueError(
            f"the table predates the horizon answers {missing}; "
            "run scripts/build_hitter_ros_table.py"
        )
    keys = ["player_id", "season", "week"]
    parts = []
    for season in sorted(int(s) for s in preds["season"].unique()):
        rows = table[(table["season"] == season) & (table["week"] > 0)]
        mine = preds[(preds["season"] == season) & (preds["week"] > 0)]
        marcel = baseline_predictions(table, season)["marcel"]
        marcel = marcel[marcel["week"] > 0]
        std = rows[[f"std_{c}" for c in COUNTS]].rename(columns=lambda c: c.removeprefix("std_"))
        hot = pd.concat([rows[keys], rates_from_counts(std)], axis=1)
        fg = _fg_snapshots(projections_dir, season)
        for week, wrows in rows.groupby("week"):
            index = wrows.set_index("player_id").index
            as_of = pd.Timestamp(wrows["as_of"].iloc[0])
            fresh = [d for d in fg if as_of - pd.Timedelta(days=FG_FRESH_DAYS) <= d <= as_of]
            fg_rates = fg[max(fresh)].reindex(index) if fresh else None

            def by_player(
                frame: pd.DataFrame, cols: list[str], week: int = week, index: pd.Index = index
            ) -> pd.DataFrame:
                f = frame[frame["week"] == week].set_index("player_id")[cols]
                return f.set_axis(list(TARGETS), axis=1).reindex(index)

            # The same for every horizon: the rest-of-season rate, and the baselines.
            ros_rate = by_player(mine, list(TARGETS))
            baselines = {
                "marcel": by_player(marcel, list(TARGETS)),
                "hot_hand": by_player(hot, list(TARGETS)),
            }
            for horizon in HORIZON_NAMES:
                head = (
                    ros_rate
                    if horizon == "ros"
                    else by_player(mine, [f"{horizon}_{s}" for s in TARGETS])
                )
                systems = {"head": head, **baselines}
                if horizon != "ros":
                    systems["ros_rate"] = ros_rate
                actual = _actual(wrows, horizon).set_axis(index).dropna(subset=["pa"])
                min_pa = ROS_MIN_PA if horizon == "ros" else int(horizon[1:])
                tags = {
                    "horizon": horizon,
                    "season": season,
                    "snapshot": f"{season}-w{int(week):02d}",
                }
                scored = scored_players(systems, actual, min_pa)
                if not scored.empty:
                    parts.append(scored.assign(**tags, comparison="all"))
                if fg_rates is not None:
                    vs_fg = {k: v for k, v in systems.items() if k in ("head", "ros_rate")}
                    vs_fg["fg_ros"] = fg_rates
                    scored = scored_players(vs_fg, actual, min_pa)
                    if not scored.empty:
                        parts.append(scored.assign(**tags, comparison="fangraphs"))
    return pd.concat(parts, ignore_index=True) if parts else None


def write_horizon_scores(run_dir: Path, scored: pd.DataFrame | None) -> None:
    """Write ``scored_horizons.parquet``; remove an old one when there's nothing scored."""
    path = run_dir / "scored_horizons.parquet"
    if scored is not None:
        scored.to_parquet(path)
    else:
        path.unlink(missing_ok=True)


def horizon_summary(scored: pd.DataFrame) -> list[str]:
    """Markdown: per horizon, MSE pooled over every scored row, and the other scores
    averaged over (season, week) units."""
    md = [
        "#### Short horizons (#419), mid-season rows (week 1+)",
        "",
        "Each system's rate for a hitter's next N PA vs what he did, players compared "
        "within the same season and week, averaged over those units (MSE, the main score, "
        "is pooled over every scored row). `head` = this "
        "horizon's own head; `ros_rate` = our rest-of-season rate used for the next N PA; "
        "`hot_hand` = his season-to-date rate.",
    ]
    if "comparison" not in scored.columns:  # scored before the FanGraphs set existed
        scored = scored.assign(comparison="all")
    md += _horizon_blocks(scored[scored["comparison"] == "all"])
    fangraphs = scored[scored["comparison"] == "fangraphs"]
    if not fangraphs.empty:
        md += [
            "",
            "#### Short horizons vs FanGraphs (2026)",
            "",
            f"Weeks with a FanGraphs rest-of-season snapshot from the {FG_FRESH_DAYS} days "
            "up to the as-of date; `fg_ros` = that FanGraphs rate (the blend) used for the "
            "next N PA. Only players all three systems cover.",
        ]
        md += _horizon_blocks(fangraphs)
    return md


def _horizon_blocks(scored: pd.DataFrame) -> list[str]:
    md: list[str] = []
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
            "MSE -- main score (totals over the next N PA; AVG in points, AB-weighted):",
            "",
            to_markdown(mse_table(g), digits=2),
            "",
            "Gap-weighted pairwise accuracy (%):",
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
