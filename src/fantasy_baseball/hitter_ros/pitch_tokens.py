"""Pitch tokens for self-supervised pretraining (#415, stage 2 of the sequence track).

One row per regular-season pitch a hitter saw, in order, split into two parts:

* **Context** -- what is known before the hitter decides: pitch type, speed, movement,
  spin, location (relative to the hitter and his strike zone), count, outs, runners,
  handedness. This is the "question" at each step.
* **Outcome** -- what happened: one of ``OUTCOMES`` (ball / called strike / whiff /
  foul / HBP / ball in play by contact quality), plus exit velo and launch angle on
  contact. This is the "answer" the pretraining model predicts for the next pitch, and
  part of the history it reads.

Movement is mirrored for left-handed pitchers and location for left-handed hitters, so
"arm side" and "inside" mean the same thing for everyone. Missing tracking numbers are
0 with a flag.

``gap`` is log1p(days since the hitter's previous pitch) / 5: a purely backward-looking
sense of time (long layoffs, new seasons) that is the same in pretraining and when the
encoder later reads a hitter's history before an as-of date.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.features import (
    FIXED_ZONE,
    ZONES,
    read_build_options,
    zone_options,
)
from fantasy_baseball.pitch_data.store import connect

logger = logging.getLogger(__name__)

PITCH_GROUPS = {
    "ff": ("FF", "FA"),
    "si": ("SI",),
    "fc": ("FC",),
    "sl": ("SL",),
    "sweep": ("ST", "SV"),
    "cu": ("CU", "KC", "CS", "SC"),
    "ch": ("CH",),
    "fs": ("FS", "FO"),
}  # everything else (knuckleball, eephus, intentional, pitchout, unknown) -> "other"

OUTCOMES = (
    "ball",
    "called_strike",
    "whiff",
    "foul",
    "hbp",
    "bip_weak",  # launch_speed_angle 1-3: weak, topped, under
    "bip_flare",  # 4: flare / burner
    "bip_solid",  # 5: solid contact
    "bip_barrel",  # 6: barrel
    "bip_other",  # in play, no classification
)
_DESCRIPTION_OUTCOME = {
    "ball": "ball",
    "blocked_ball": "ball",
    "automatic_ball": "ball",
    "intent_ball": "ball",
    "pitchout": "ball",
    "called_strike": "called_strike",
    "automatic_strike": "called_strike",
    "swinging_strike": "whiff",
    "swinging_strike_blocked": "whiff",
    "missed_bunt": "whiff",
    "swinging_pitchout": "whiff",
    "foul": "foul",
    "foul_tip": "foul",
    "foul_bunt": "foul",
    "bunt_foul_tip": "foul",
    "foul_pitchout": "foul",
    "hit_by_pitch": "hbp",
}
_LSA_OUTCOME = {
    1: "bip_weak",
    2: "bip_weak",
    3: "bip_weak",
    4: "bip_flare",
    5: "bip_solid",
    6: "bip_barrel",
}

CONTEXT_FEATURES = (
    *(f"pt_{g}" for g in (*PITCH_GROUPS, "other")),
    "has_track",  # speed / movement / location measured
    "speed",  # (release_speed - 90) / 5
    "move_arm",  # horizontal break toward the pitcher's arm side, feet
    "move_up",  # induced vertical break, feet
    "has_spin",
    "spin",  # release_spin_rate / 2500
    "loc_in",  # plate_x toward the hitter (inside), feet
    "loc_up",  # height within the strike zone (see ``zone``): 0 = bottom, 1 = top
    "balls",  # / 3
    "strikes",  # / 2
    "outs",  # / 2
    "on_1b",
    "on_2b",
    "on_3b",
    "vs_lhp",
    "bats_left",
    "gap",  # log1p(days since the previous pitch he saw) / 5
)
OUTCOME_FEATURES = (*(f"out_{o}" for o in OUTCOMES), "ev", "la")  # ev / 100, la / 50
TOKEN_FEATURES = (*CONTEXT_FEATURES, *OUTCOME_FEATURES)


def token_path(root: Path, zone: str) -> Path:
    """Where ``scripts/build_hitter_ros_pitch_tokens.py --zone <zone>`` writes the tokens
    under ``root``; the strike zone is recorded next to them (:func:`token_options`)."""
    zone_options(zone)  # validates
    return root / ("pitch_tokens.parquet" if zone == "statcast" else f"pitch_tokens_{zone}.parquet")


def token_options(zone: str) -> dict[str, Any]:
    """The build options a token file records (``<file>.json``): its strike zone."""
    return zone_options(zone)


def recorded_token_options(path: Path) -> dict[str, Any]:
    """A token file's recorded build options. Token files from before #433 recorded none;
    they were all built with Savant's zone."""
    return read_build_options(path) or token_options("statcast")


def outcome_index(description: pd.Series, bb_class: pd.Series, pitch_type: pd.Series) -> np.ndarray:
    """Outcome class index per pitch (see ``OUTCOMES``).

    ``pitch_type`` is Statcast's ``type``: B (ball), S (strike), X (in play). A
    description this module doesn't know falls back on it -- B -> ball, S -> called
    strike -- with a warning, rather than silently becoming a ball. (The store has ~100
    ``type = 'S'`` rows described ``hit_into_play``.) ``foul_tip`` stays a foul: the
    repo counts it as contact, not a whiff (see ``keepers/savant.py``).
    """
    in_play = pitch_type == "X"
    by_desc = description.map(_DESCRIPTION_OUTCOME)
    unknown = by_desc.isna() & ~in_play
    if unknown.any():
        logger.warning(
            "%d pitches with unrecognized descriptions %s; labeled from Statcast type",
            int(unknown.sum()),
            sorted(description[unknown].astype(str).unique())[:10],
        )
        by_desc = by_desc.where(~unknown, pitch_type.map({"B": "ball", "S": "called_strike"}))
    by_bip = bb_class.map(_LSA_OUTCOME).fillna("bip_other")
    label = by_desc.where(~in_play, by_bip).fillna("ball")
    idx: np.ndarray = label.map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy(np.int64)
    return idx


def build_pitch_tokens(store: Path, zone: str = "statcast") -> pd.DataFrame:
    """One row per regular-season pitch, sorted by hitter, then game, PA and pitch.

    Built a season at a time to keep memory down; ``gap`` is recomputed over the whole
    frame at the end so a hitter's first pitch of a season sees the offseason. ``zone``:
    see :func:`tokens_from_pitches`.
    """
    conn = connect(store)
    try:
        seasons = [
            int(r[0])
            for r in conn.execute("SELECT DISTINCT season FROM pitches ORDER BY 1").fetchall()
        ]
        parts = [tokens_from_pitches(_season_pitches(conn, s), zone) for s in seasons]
    finally:
        conn.close()
    tokens = pd.concat(parts, ignore_index=True)
    tokens = tokens.sort_values(
        ["player_id", "game_date", "game_pk", "at_bat_number", "pitch_number"],
        ignore_index=True,
    )
    tokens["gap"] = _gap(tokens["player_id"].to_numpy(), tokens["game_date"])
    return tokens


def _season_pitches(conn: duckdb.DuckDBPyConnection, season: int) -> pd.DataFrame:
    return conn.execute(
        """
            SELECT batter AS player_id, CAST(game_date AS DATE) AS game_date,
                   CAST(season AS INTEGER) AS season, game_pk, at_bat_number, pitch_number,
                   pitch_type, release_speed, pfx_x, pfx_z, release_spin_rate,
                   plate_x, plate_z, sz_top, sz_bot, balls, strikes, outs_when_up,
                   on_1b IS NOT NULL AS on_1b, on_2b IS NOT NULL AS on_2b,
                   on_3b IS NOT NULL AS on_3b, p_throws, stand,
                   description, type, launch_speed_angle, launch_speed, launch_angle
            FROM pitches WHERE game_type = 'R' AND season = ?
            ORDER BY player_id, game_date, game_pk, at_bat_number, pitch_number
            """,
        [season],
    ).df()


def _gap(player: np.ndarray, dates: pd.Series) -> np.ndarray:
    """log1p(days since the same hitter's previous pitch) / 5; 0 for his first pitch."""
    days = pd.to_datetime(dates).to_numpy().astype("datetime64[D]").astype(np.int64)
    prev = np.r_[days[:1], days[:-1]]
    first = np.r_[True, player[1:] != player[:-1]]
    gap = np.where(first, 0, days - prev)
    out: np.ndarray = (np.log1p(gap) / 5).astype(np.float32)
    return out


def tokens_from_pitches(df: pd.DataFrame, zone: str = "statcast") -> pd.DataFrame:
    """The token frame from raw (sorted) pitch rows; split out so tests can feed rows.

    ``zone`` (features.ZONES, #433) is the strike zone ``loc_up`` is measured in:
    "statcast" = each pitch's sz_bot / sz_top, "fixed" = features.FIXED_ZONE for every
    season (2026's ABS-recorded sz_top / sz_bot sit lower than before).
    """
    if zone not in ZONES:
        raise ValueError(f"unknown zone {zone!r}")
    f32 = np.float32
    out = df[["player_id", "game_date", "season", "game_pk", "at_bat_number", "pitch_number"]]
    out = out.copy()
    out["game_date"] = pd.to_datetime(out["game_date"])

    group = pd.Series("other", index=df.index)
    for g, codes in PITCH_GROUPS.items():
        group[df["pitch_type"].isin(codes)] = g
    for g in (*PITCH_GROUPS, "other"):
        out[f"pt_{g}"] = (group == g).astype(f32)

    lefty_p = df["p_throws"] == "L"
    lefty_b = df["stand"] == "L"
    out["has_track"] = df["release_speed"].notna().astype(f32)
    out["speed"] = ((df["release_speed"] - 90) / 5).fillna(0).astype(f32)
    # pfx_x is from the catcher's view (+ = toward first base). A righty's arm side is
    # third base (-), so flip righties' sign; then + means arm side for everyone.
    out["move_arm"] = (df["pfx_x"].where(lefty_p, -df["pfx_x"])).fillna(0).astype(f32)
    out["move_up"] = df["pfx_z"].fillna(0).astype(f32)
    out["has_spin"] = df["release_spin_rate"].notna().astype(f32)
    out["spin"] = (df["release_spin_rate"] / 2500).fillna(0).astype(f32)
    # plate_x + = toward first base = inside to a lefty, outside to a righty.
    out["loc_in"] = (df["plate_x"].where(lefty_b, -df["plate_x"])).fillna(0).astype(f32)
    if zone == "fixed":
        height = (df["plate_z"] - FIXED_ZONE[1]) / (FIXED_ZONE[2] - FIXED_ZONE[1])
    else:
        height = (df["plate_z"] - df["sz_bot"]) / (df["sz_top"] - df["sz_bot"])
    out["loc_up"] = height.replace([np.inf, -np.inf], np.nan).fillna(0.5).astype(f32)
    out["balls"] = (df["balls"] / 3).astype(f32)
    out["strikes"] = (df["strikes"] / 2).astype(f32)
    out["outs"] = (df["outs_when_up"] / 2).astype(f32)
    for c in ("on_1b", "on_2b", "on_3b"):
        out[c] = df[c].astype(f32)
    out["vs_lhp"] = lefty_p.astype(f32)
    out["bats_left"] = lefty_b.astype(f32)
    out["gap"] = _gap(df["player_id"].to_numpy(), out["game_date"])

    in_play = df["type"] == "X"
    idx = outcome_index(df["description"], df["launch_speed_angle"], df["type"])
    out["outcome"] = idx
    for i, o in enumerate(OUTCOMES):
        out[f"out_{o}"] = (idx == i).astype(f32)
    out["ev"] = (df["launch_speed"].where(in_play) / 100).fillna(0).astype(f32)
    out["la"] = (df["launch_angle"].where(in_play) / 50).fillna(0).astype(f32)
    return out
