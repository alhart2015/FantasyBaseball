"""Appearance sequences for the pitcher ROS net (#465): one token per game pitched.

The window inputs (``features.input_frame``) see a pitcher's history as a few sums --
this season so far, the last 30 days, last season, the last three, his career. A
sequence sees each outing in order, so the net can read things the sums smear out: a
velocity drop over his last few starts, a move from the bullpen to the rotation, a
stretch on the injured list, how a reliever is being used lately.

:func:`build_appearance_tokens` turns the store into one row ("token") per regular-season
appearance, from the box score (``pitching``: role, volume, results) and, where the game
was tracked, the pitches he threw in it (velocity, spin, whiffs, called strikes, zone,
chase, pitch mix). The hitter net's machinery reads them (``hitter_ros.sequence``:
``SequenceBatcher`` with ``TOKEN_FEATURES`` and ``TOKEN_ORDER``, which adds days-ago and
same-season per row, and the encoders): each table row sees the pitcher's last
``seq_len`` appearances strictly before its as-of date.

Results come **against the league** (#483): each game's strikeouts, walks, hit
batters, hits and home runs minus what the league's rates per batter faced would give
over his batters faced (earned runs per out), and his whiff and called-strike-plus-whiff
rates minus the league's per pitch. Raw counts made a high-strikeout season league-wide
read as a good pitcher (the transformer projected 2022, after 2021's high-K year, 5% too
many K vs the plain net's 3.9%). The league's rates are the ones known before the game's
date: last season plus this season's earlier days, pooled, as the net's league forecast
(``features.league_reference``) -- never the game's whole season, which rows in it would
otherwise read their future through. Every pitching line counts (the table's league
leaves out position players pitching: under 1% of batters faced). Volume (batters faced,
outs, pitches), role and velocity stay as they are.

Every feature is on about a 0-1 scale (velocity and spin, centered, clipped to +/-4;
the against-the-league ones are not clipped). A game with
no pitch tracking has ``tracked`` 0 and its tracking features 0; spin is measured from
2015 (``spin_tracked``). A game with no league rate known before it (the store's first
day) has its against-the-league features 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.statcast_sql import SWING_SQL, WHIFF_SQL, sql_in
from fantasy_baseball.pitch_data.store import connect
from fantasy_baseball.pitcher_ros.stats import GAME_DATE_SQL
from fantasy_baseball.pitcher_ros.table import BREAKING, CUTTERS, FASTBALLS, OFFSPEED

# Box-score volume -> the scale each is divided by.
VOLUME_SCALES = {"outs": 27, "bf": 30, "pitches": 100}
# Box-score results -> (what the league's rate is per, the scale the excess over the
# league is divided by). Each token has ``<stat>_vs_lg``.
RESULT_SCALES = {
    "k": ("bf", 10),
    "bb": ("bf", 5),
    "hbp": ("bf", 3),
    "h": ("bf", 10),
    "hr": ("bf", 3),
    "er": ("outs", 7),
}
# Pitch-tracking rates against the league's per pitch: name -> the counts it sums.
PITCH_RATES = {"whiff": ("whiffs",), "csw": ("whiffs", "called")}
BOX_FLAGS = ("started", "finished", "w", "l", "sv", "hld", "bs")
TRACKING = (
    "tracked",
    "fb_velo",  # (mean fastball speed - 93) / 5
    "fb_velo_max",  # (fastest pitch - 95) / 5
    "fb_spin",  # (mean fastball spin - 2250) / 300
    "spin_tracked",  # spin measured (from 2015)
    "whiff_vs_lg",  # whiffs per pitch minus the league's
    "csw_vs_lg",  # called strikes and whiffs per pitch minus the league's
    "zone_rate",
    "chase_rate",  # swings at pitches outside the zone, per such pitch
    "fb_share",
    "fc_share",
    "brk_share",
    "off_share",
)
TOKEN_FEATURES = (
    *VOLUME_SCALES,
    *(f"{c}_vs_lg" for c in RESULT_SCALES),
    "strike_share",  # strikes per pitch, from the box score
    *BOX_FLAGS,
    "rest",  # log1p(days since his previous appearance) / 5; 0 for his first
    "first_seen",
    *TRACKING,
)
# A pitcher's appearances, oldest first; two on one day by game.
TOKEN_ORDER = ("game_date", "game_pk")
# The pitch-tracking sums per pitcher and game (build_appearance_tokens' second query).
_TRACKED_COLS = (
    "player_id",
    "game_pk",
    "n",
    "fb_velo",
    "fb_velo_max",
    "fb_spin",
    "whiffs",
    "called",
    "in_zone",
    "out_zone",
    "chases",
    "fb",
    "fc",
    "brk",
    "off",
)


def build_appearance_tokens(store: Path) -> pd.DataFrame:
    """One row per regular-season appearance (``player_id``, ``game_date``, ``season``,
    ``game_pk`` and ``TOKEN_FEATURES``), sorted by pitcher then time. Dated as the
    training table dates his lines (``stats.GAME_DATE_SQL``: a suspended game on the
    day it was finished)."""

    def group_sql(types: tuple[str, ...]) -> str:
        return f"pitch_type IN {sql_in(types)}"

    conn = connect(store)
    try:
        box = conn.execute(
            f"""
            SELECT player_id, {GAME_DATE_SQL} AS game_date, game_pk, gs, gf, w, l, sv, hld,
                   bs, outs, bf, k, bb, hbp, h, hr, er, pitches, strikes
            FROM pitching
            """
        ).df()
        views = {r[0] for r in conn.execute("SELECT view_name FROM duckdb_views()").fetchall()}
        tracked = (
            conn.execute(
                f"""
                SELECT pitcher AS player_id, game_pk, count(*) AS n,
                       avg(release_speed) FILTER (WHERE {group_sql(FASTBALLS)}) AS fb_velo,
                       max(release_speed) AS fb_velo_max,
                       avg(release_spin_rate) FILTER (WHERE {group_sql(FASTBALLS)}) AS fb_spin,
                       count(*) FILTER (WHERE {WHIFF_SQL}) AS whiffs,
                       count(*) FILTER (WHERE description = 'called_strike') AS called,
                       count(*) FILTER (WHERE zone BETWEEN 1 AND 9) AS in_zone,
                       count(*) FILTER (WHERE zone BETWEEN 11 AND 14) AS out_zone,
                       count(*) FILTER (WHERE zone BETWEEN 11 AND 14 AND {SWING_SQL})
                           AS chases,
                       count(*) FILTER (WHERE {group_sql(FASTBALLS)}) AS fb,
                       count(*) FILTER (WHERE {group_sql(CUTTERS)}) AS fc,
                       count(*) FILTER (WHERE {group_sql(BREAKING)}) AS brk,
                       count(*) FILTER (WHERE {group_sql(OFFSPEED)}) AS off
                FROM pitches WHERE game_type = 'R'
                GROUP BY 1, 2
                """
            ).df()
            if "pitches" in views
            else pd.DataFrame(columns=list(_TRACKED_COLS))
        )
    finally:
        conn.close()
    return appearance_tokens(box, tracked)


def league_known_before(daily: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """``cols`` summed league-wide as known before each date of ``daily`` (one row per
    ``game_date`` with ``cols``): all of last season plus this season's earlier dates.
    Indexed by date."""
    daily = daily.groupby("game_date")[cols].sum().sort_index()
    season = pd.Series(daily.index.year, index=daily.index)
    earlier = daily.groupby(season.to_numpy()).cumsum() - daily
    whole = daily.groupby(season.to_numpy()).sum()
    last = whole.reindex(season.to_numpy() - 1).fillna(0).set_axis(daily.index)
    return earlier + last


def appearance_tokens(box: pd.DataFrame, tracked: pd.DataFrame) -> pd.DataFrame:
    """:func:`build_appearance_tokens` from its two queries: ``box``, one row per pitcher
    and game (box-score counts), and ``tracked``, the pitch-tracking sums per pitcher and
    game (none for an untracked game). The league's rates come from the same rows."""
    box = box.assign(game_date=pd.to_datetime(box["game_date"]))
    tracked = tracked[list(_TRACKED_COLS)].astype(
        {"player_id": box["player_id"].dtype, "game_pk": box["game_pk"].dtype}
    )
    df = box.merge(tracked, on=["player_id", "game_pk"], how="left")
    df = df.sort_values(["player_id", "game_date", "game_pk"]).reset_index(drop=True)
    out = df[["player_id", "game_date", "game_pk"]].copy()
    out["season"] = out["game_date"].dt.year
    for c, scale in VOLUME_SCALES.items():
        out[c] = (df[c].astype(float) / scale).astype(np.float32)
    box_cols = ["bf", "outs", *RESULT_SCALES]
    league = league_known_before(
        df[["game_date", *box_cols]].astype({c: float for c in box_cols}), box_cols
    )
    lg = league.reindex(df["game_date"]).set_axis(df.index)
    for c, (per, scale) in RESULT_SCALES.items():
        rate = lg[c] / lg[per].where(lg[per] > 0)
        excess = (df[c].astype(float) - rate * df[per].astype(float)) / scale
        out[f"{c}_vs_lg"] = excess.fillna(0).astype(np.float32)
    pitches = df["pitches"].astype(float)
    out["strike_share"] = (df["strikes"] / pitches.where(pitches > 0)).fillna(0).astype(np.float32)
    flags = {"started": "gs", "finished": "gf", **{f: f for f in BOX_FLAGS[2:]}}
    for name, col in flags.items():
        out[name] = (df[col].astype(float) > 0).astype(np.float32)
    gap = out.groupby("player_id")["game_date"].diff().dt.days
    out["rest"] = (np.log1p(gap) / 5).fillna(0).astype(np.float32)
    out["first_seen"] = gap.isna().astype(np.float32)

    n = df["n"].astype(float)
    has = n > 0
    out["tracked"] = has.astype(np.float32)

    def scaled(col: str, center: float, scale: float) -> pd.Series:
        # Clipped: a position player lobbing 40 mph would otherwise sit at -10.
        v = ((df[col].astype(float) - center) / scale).clip(-4, 4)
        return v.where(has).fillna(0).astype(np.float32)

    def share(num: str, den: pd.Series) -> pd.Series:
        return (df[num].astype(float) / den.where(den > 0)).fillna(0).astype(np.float32)

    out["fb_velo"] = scaled("fb_velo", 93, 5)
    out["fb_velo_max"] = scaled("fb_velo_max", 95, 5)
    out["fb_spin"] = scaled("fb_spin", 2250, 300)
    out["spin_tracked"] = df["fb_spin"].notna().astype(np.float32)
    pitch_cols = ["n", "whiffs", "called"]
    pitch_league = league_known_before(
        df[["game_date", *pitch_cols]].fillna(0).astype({c: float for c in pitch_cols}),
        pitch_cols,
    )
    plg = pitch_league.reindex(df["game_date"]).set_axis(df.index)
    for name, parts in PITCH_RATES.items():
        own = sum(df[c].astype(float) for c in parts) / n.where(has)
        lg_rate = sum(plg[c] for c in parts) / plg["n"].where(plg["n"] > 0)
        out[f"{name}_vs_lg"] = (own - lg_rate).where(has).fillna(0).astype(np.float32)
    out["zone_rate"] = share("in_zone", n)
    out["chase_rate"] = share("chases", df["out_zone"].astype(float))
    for g in ("fb", "fc", "brk", "off"):
        out[f"{g}_share"] = share(g, n)
    return out
