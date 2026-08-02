"""Normalize the live full-season blend (actual-to-date + rest-of-season) to the
canonical rate/PT schema.

The blob is `CacheKey.FULL_SEASON_PROJECTIONS` in the KV store, written by the ROS
refresh job. Each record is a whole-season estimate for the CURRENT season: what the
player has already done, plus what he is projected to do the rest of the way. That is
the quantity a mid-season keeper decision needs -- a season two-thirds finished is not
comparable to a preseason projection until the remaining third is added back.

Pure and I/O-free: `parse_blend` takes the already-fetched payload dict, so the
network read lives in the calling script and the normalization is testable offline.

Not reusing `draft_value.parse_full_season_lines`: it folds `h`/`ab` into a finished
`avg` and drops `pa` entirely, which is exactly the volume/rate separation this
subsystem cannot lose. The keying and tie-break here deliberately match it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fantasy_baseball.keepers.actuals import (
    HITTER_PT,
    PITCHER_PT,
    index_by_mlbam,
    safe_ratio,
)

# Blob field -> the name `keepers.vintages.decompose_*` expects. The pitcher side calls
# hits allowed `h_allowed`, since `h` on a pitcher record would read as hits taken.
_HITTER_FIELDS = {"pa": "PA", "ab": "AB", "h": "H", "hr": "HR", "r": "R", "rbi": "RBI", "sb": "SB"}
_PITCHER_FIELDS = {
    "ip": "IP",
    "k": "SO",
    "w": "W",
    "sv": "SV",
    "er": "ER",
    "bb": "BB",
    "h_allowed": "H",
}


def parse_blend(payload: dict[str, Any], player_type: str) -> pd.DataFrame:
    """One decomposed rate/PT frame from a full-season payload, indexed by mlbam_id.

    `payload` is the blob's inner `_data` mapping (prod wraps it in a `_meta` envelope;
    unwrapping is the caller's job, because only the caller knows which store it came
    from). Records without an mlbam id are dropped, matching every other join in this
    package. A player appearing twice keeps his highest-volume row, so a mid-season
    trade cannot resolve to the shorter stint.
    """
    if player_type not in {"hitter", "pitcher"}:
        raise ValueError(f"player_type must be 'hitter' or 'pitcher', got {player_type!r}")
    key = "hitters" if player_type == "hitter" else "pitchers"
    fields = _HITTER_FIELDS if player_type == "hitter" else _PITCHER_FIELDS
    records = payload.get(key) or []
    if not records:
        raise ValueError(f"full-season payload carries no {key!r} records")

    raw = pd.DataFrame.from_records(records)
    missing = [src for src in fields if src not in raw.columns]
    if missing:
        raise KeyError(f"{key} records missing {missing}; got {sorted(raw.columns)}")
    renamed = raw.rename(columns=fields)
    renamed["MLBAMID"] = pd.to_numeric(raw["mlbam_id"], errors="coerce")

    frame = index_by_mlbam(renamed.loc[renamed["MLBAMID"].notna()], "MLBAMID")
    pt_col, volume = (HITTER_PT, "PA") if player_type == "hitter" else (PITCHER_PT, "IP")
    frame = frame.sort_values(volume, ascending=False)
    frame = frame.loc[~frame.index.duplicated(keep="first")]

    pt = frame[volume].astype(float)
    if player_type == "hitter":
        ab = frame["AB"].astype(float)
        columns = {
            pt_col: pt,
            "ab_pa": safe_ratio(ab, pt),
            "h_ab": safe_ratio(frame["H"].astype(float), ab),
            "hr_pa": safe_ratio(frame["HR"].astype(float), pt),
            "r_pa": safe_ratio(frame["R"].astype(float), pt),
            "rbi_pa": safe_ratio(frame["RBI"].astype(float), pt),
            "sb_pa": safe_ratio(frame["SB"].astype(float), pt),
        }
    else:
        columns = {
            pt_col: pt,
            "k_ip": safe_ratio(frame["SO"].astype(float), pt),
            "w_ip": safe_ratio(frame["W"].astype(float), pt),
            "sv_ip": safe_ratio(frame["SV"].astype(float), pt),
            "er_ip": safe_ratio(frame["ER"].astype(float), pt),
            "bb_ip": safe_ratio(frame["BB"].astype(float), pt),
            "h_ip": safe_ratio(frame["H"].astype(float), pt),
        }
    return pd.DataFrame(columns, index=frame.index)
