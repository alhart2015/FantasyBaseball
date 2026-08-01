"""Load a ZiPS vintage from disk and decompose it to the canonical rate/PT schema.

Reads the raw CSV rather than going through data.fangraphs.load_projection_set.
The reason is the standalone constraint: increment 1 imports nothing from
fantasy_baseball.data (spec 9). To be clear about what that loader does and does
not do -- it preserves MLBAMID as `mlbam_id`, so the join key would survive; it
is the dependency, not a defect, that rules it out. Filename variants
(year-suffixed, proj-from-dated) are resolved by glob here, matching the
loader's own fallback behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fantasy_baseball.keepers.actuals import (
    HITTER_PT,
    PITCHER_PT,
    index_by_mlbam,
    safe_ratio,
)


def _find(directory: Path, player_type: str) -> Path:
    exact = directory / f"zips-{player_type}.csv"
    if exact.exists():
        return exact
    matches = sorted(directory.glob(f"zips-{player_type}-*.csv"))
    if not matches:
        raise FileNotFoundError(f"no ZiPS {player_type} export under {directory}")
    return matches[-1]


def decompose_hitters(df: pd.DataFrame) -> pd.DataFrame:
    frame = index_by_mlbam(df, "MLBAMID")
    pa, ab = frame["PA"].astype(float), frame["AB"].astype(float)
    return pd.DataFrame(
        {
            HITTER_PT: pa,
            "ab_pa": safe_ratio(ab, pa),
            "h_ab": safe_ratio(frame["H"].astype(float), ab),
            "hr_pa": safe_ratio(frame["HR"].astype(float), pa),
            "r_pa": safe_ratio(frame["R"].astype(float), pa),
            "rbi_pa": safe_ratio(frame["RBI"].astype(float), pa),
            "sb_pa": safe_ratio(frame["SB"].astype(float), pa),
        },
        index=frame.index,
    )


def decompose_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    frame = index_by_mlbam(df, "MLBAMID")
    ip = frame["IP"].astype(float)
    return pd.DataFrame(
        {
            PITCHER_PT: ip,
            "k_ip": safe_ratio(frame["SO"].astype(float), ip),
            "w_ip": safe_ratio(frame["W"].astype(float), ip),
            "sv_ip": safe_ratio(frame["SV"].astype(float), ip),
            "er_ip": safe_ratio(frame["ER"].astype(float), ip),
            "bb_ip": safe_ratio(frame["BB"].astype(float), ip),
            "h_ip": safe_ratio(frame["H"].astype(float), ip),
        },
        index=frame.index,
    )


def load_vintage(year: int, projections_root: Path, player_type: str) -> pd.DataFrame:
    """One decomposed ZiPS vintage frame. Reads only the file it returns."""
    if player_type not in {"hitter", "pitcher"}:
        raise ValueError(f"player_type must be 'hitter' or 'pitcher', got {player_type!r}")
    directory = projections_root / str(year)
    if player_type == "hitter":
        return decompose_hitters(pd.read_csv(_find(directory, "hitters")))
    return decompose_pitchers(pd.read_csv(_find(directory, "pitchers")))
