import pandas as pd
from pathlib import Path

HITTING_COLUMN_MAP: dict[str, str] = {
    "Name": "name",
    "Team": "team",
    "PA": "pa",
    "AB": "ab",
    "H": "h",
    "HR": "hr",
    "R": "r",
    "RBI": "rbi",
    "SB": "sb",
    "AVG": "avg",
    "playerid": "fg_id",
}

PITCHING_COLUMN_MAP: dict[str, str] = {
    "Name": "name",
    "Team": "team",
    "IP": "ip",
    "W": "w",
    "SO": "k",
    "ERA": "era",
    "WHIP": "whip",
    "SV": "sv",
    "ER": "er",
    "BB": "bb",
    "H": "h_allowed",
    "playerid": "fg_id",
}

REQUIRED_HITTING_COLS: list[str] = ["name", "ab", "h", "hr", "r", "rbi", "sb", "avg"]
REQUIRED_PITCHING_COLS: list[str] = ["name", "ip", "w", "k", "era", "whip", "sv"]


def parse_hitting_csv(filepath: Path) -> pd.DataFrame:
    """Parse a FanGraphs hitting projections CSV into normalized columns."""
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    rename = {k: v for k, v in HITTING_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED_HITTING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["player_type"] = "hitter"
    return df


def parse_pitching_csv(filepath: Path) -> pd.DataFrame:
    """Parse a FanGraphs pitching projections CSV into normalized columns."""
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    rename = {k: v for k, v in PITCHING_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED_PITCHING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["player_type"] = "pitcher"
    return df


def load_projection_set(
    projections_dir: Path, system_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a named projection system from the projections directory.

    Expects files named like: steamer_hitters.csv, steamer_pitchers.csv
    """
    hitting_file = projections_dir / f"{system_name}_hitters.csv"
    pitching_file = projections_dir / f"{system_name}_pitchers.csv"
    hitters = parse_hitting_csv(hitting_file) if hitting_file.exists() else pd.DataFrame()
    pitchers = parse_pitching_csv(pitching_file) if pitching_file.exists() else pd.DataFrame()
    return hitters, pitchers
