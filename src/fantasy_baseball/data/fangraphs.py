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
    "ADP": "adp",
    "playerid": "fg_id",
    "PlayerId": "fg_id",
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
    "ADP": "adp",
    "ER": "er",
    "BB": "bb",
    "H": "h_allowed",
    "playerid": "fg_id",
    "PlayerId": "fg_id",
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

    Tries multiple naming conventions:
    - steamer_hitters.csv (simple)
    - fangraphs-leaderboard-projections-steamer-hitters.csv (FanGraphs export)
    """
    hitting_file = _find_file(projections_dir, system_name, "hitters")
    pitching_file = _find_file(projections_dir, system_name, "pitchers")
    hitters = parse_hitting_csv(hitting_file) if hitting_file else pd.DataFrame()
    pitchers = parse_pitching_csv(pitching_file) if pitching_file else pd.DataFrame()
    return hitters, pitchers


def _find_file(directory: Path, system: str, player_type: str) -> Path | None:
    """Find a projection CSV file, trying multiple naming conventions."""
    candidates = [
        directory / f"{system}_{player_type}.csv",
        directory / f"fangraphs-leaderboard-projections-{system}-{player_type}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None
