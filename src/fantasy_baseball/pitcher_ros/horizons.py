"""Score the pitcher net's short-horizon outputs: his next 7 / 30 days (#465).

For every row of the test seasons with a batter faced in the window, two "systems"
give a rate for that pitcher's next N days (``table.HORIZON_DAYS``), scored against
what he did over them:

* ``head``: the net's own output for that horizon (``d{N}_{stat}``).
* ``ros_rate``: the net's rest-of-season rate used for the next N days -- the bar a
  horizon output has to beat to be worth having.

Scored as the backtest scores (``evaluate.squared_error``): K, W, SV ``(bf x err)^2``
over the window's batters faced, ERA, WHIP ``ip x err^2`` weighted by its innings.
Units are the season and week (``snapshot``), so the bootstrap redraws a pitcher once
for every week he is in. Preseason (week 0) and mid-season rows are kept apart
(``span``): week 0's window is the season's first days, read off a preseason projection.
"""

from __future__ import annotations

import pandas as pd

from fantasy_baseball.hitter_ros.backtest import to_markdown
from fantasy_baseball.pitcher_ros.evaluate import mse_bootstrap, mse_table, scored_players
from fantasy_baseball.pitcher_ros.stats import TARGETS, rates_from_counts
from fantasy_baseball.pitcher_ros.table import HORIZON_DAYS
from fantasy_baseball.projection_eval import sure_text

# A row is scored once he faced a batter in the window: the error is in counts over
# the window, so a pitcher who barely pitched barely counts.
MIN_BF = 1
HEAD = "head"
ROS_RATE = "ros_rate"
SPANS = ("preseason", "mid-season")
_KEYS = ["player_id", "season", "week"]
_COUNTS = ("bf", "outs", "k", "w", "sv", "er", "h", "bb")


def _actual(rows: pd.DataFrame, n: int) -> pd.DataFrame:
    counts = pd.DataFrame({c: rows[f"ros_d{n}_{c}"].astype(float) for c in _COUNTS})
    actual = rates_from_counts(counts)[list(TARGETS)]
    actual["bf"] = counts["bf"]
    actual["ip"] = counts["outs"] / 3
    return actual


def score_horizons(table: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame | None:
    """One scored frame (``evaluate.scored_players`` rows) for every horizon, tagged with
    ``horizon`` ("d7", "d30"), ``span`` (``SPANS``), ``season`` and ``snapshot`` (season
    and week). None when ``preds`` has no horizon outputs."""
    if f"d{HORIZON_DAYS[0]}_{TARGETS[0]}" not in preds.columns:
        return None
    missing = [f"ros_d{n}_bf" for n in HORIZON_DAYS if f"ros_d{n}_bf" not in table.columns]
    if missing:
        raise ValueError(
            f"the table predates the short-horizon answers {missing}; "
            "run scripts/build_pitcher_ros_table.py"
        )
    answer_cols = [f"ros_d{n}_{c}" for n in HORIZON_DAYS for c in _COUNTS]
    rows = preds.merge(table[[*_KEYS, *answer_cols]], on=_KEYS, how="left", validate="one_to_one")
    if rows[answer_cols].isna().all(axis=1).any():
        raise ValueError("predictions for rows the table doesn't have")
    parts = []
    for (season, week), unit in rows.groupby(["season", "week"]):
        unit = unit.set_index("player_id")
        ros_rate = unit[list(TARGETS)]
        for n in HORIZON_DAYS:
            head = unit[[f"d{n}_{s}" for s in TARGETS]].set_axis(list(TARGETS), axis=1)
            scored = scored_players({HEAD: head, ROS_RATE: ros_rate}, _actual(unit, n), MIN_BF)
            if not scored.empty:
                parts.append(
                    scored.assign(
                        horizon=f"d{n}",
                        span=SPANS[0] if week == 0 else SPANS[1],
                        season=int(season),
                        snapshot=f"{int(season)}-w{int(week):02d}",
                    )
                )
    return pd.concat(parts, ignore_index=True) if parts else None


def horizon_summary(scored: pd.DataFrame | None) -> list[str]:
    """Markdown lines: per horizon and span, both systems' MSE and how sure the head is
    better than the rest-of-season rate."""
    if scored is None:
        return []
    md = [
        "#### Next 7 / 30 days (pitcher_ros/horizons.py)",
        "",
        f"`{HEAD}` = the net's output for the window; `{ROS_RATE}` = its rest-of-season "
        "rate used for the window. MSE as above, over the window's batters faced and "
        "innings; every pitcher with a batter faced in it. How sure = head better than "
        "ros_rate, 0-100.",
    ]
    for horizon in (f"d{n}" for n in HORIZON_DAYS):
        for span in SPANS:
            frame = scored[(scored["horizon"] == horizon) & (scored["span"] == span)]
            if frame.empty:
                continue
            boot = mse_bootstrap(frame, HEAD, ROS_RATE)
            sure = ", ".join(f"{s} {sure_text(boot.loc[s, 'sure'])}" for s in TARGETS)
            md += [
                "",
                f"**Next {horizon[1:]} days, {span}**",
                "",
                to_markdown(mse_table(frame), digits=3),
                "",
                f"Head vs ros_rate: {sure}.",
            ]
    return md
