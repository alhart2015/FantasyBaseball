"""Minimal-but-realistic fixture data for the run_full_refresh
integration test. Returns plain dicts/lists matching the shapes
produced by Yahoo (post-parse) and the projection CSVs (post-blend).
"""
from datetime import date
from typing import Any


TEAM_NAMES = [f"Team {i:02d}" for i in range(1, 13)]  # 12 teams
USER_TEAM_NAME = "Team 01"


def _hitter_proj_row(name: str, fg_id: str, **stats) -> dict:
    """One row of a blended hitter projection table."""
    base = {
        "name": name, "fg_id": fg_id, "team": "TBD",
        "positions": "OF", "ab": 500, "pa": 580,
        "r": 80, "hr": 25, "rbi": 80, "sb": 8, "h": 145, "avg": 0.290,
    }
    base.update(stats)
    return base


def _pitcher_proj_row(name: str, fg_id: str, **stats) -> dict:
    base = {
        "name": name, "fg_id": fg_id, "team": "TBD",
        "positions": "SP", "ip": 180.0,
        "w": 12, "k": 200, "sv": 0, "era": 3.50, "whip": 1.15,
        "er": 70, "bb": 50, "h_allowed": 160,
    }
    base.update(stats)
    return base


def hitter_projections() -> list[dict]:
    """120 hitters — enough to cover 12 rosters x ~6 hitters + spares."""
    rows = []
    for i in range(120):
        rows.append(_hitter_proj_row(
            name=f"Hitter{i:03d}", fg_id=f"fg_h_{i:03d}",
            r=70 + (i % 30), hr=15 + (i % 20), rbi=60 + (i % 30),
            sb=2 + (i % 15), avg=0.250 + (i % 50) / 1000,
        ))
    return rows


def pitcher_projections() -> list[dict]:
    """80 pitchers — covers 12 rosters x ~5 pitchers + spares + closers."""
    rows = []
    for i in range(80):
        is_closer = i < 12  # First 12 are closers
        rows.append(_pitcher_proj_row(
            name=f"Pitcher{i:03d}", fg_id=f"fg_p_{i:03d}",
            positions="RP" if is_closer else "SP",
            ip=70.0 if is_closer else 180.0,
            sv=25 if is_closer else 0,
            w=4 if is_closer else 10 + (i % 8),
            k=80 if is_closer else 180 + (i % 40),
            era=3.0 + (i % 10) / 10, whip=1.10 + (i % 10) / 100,
        ))
    return rows


def standings() -> list[dict]:
    """12 teams with all 10 categories populated."""
    out = []
    for i, name in enumerate(TEAM_NAMES, start=1):
        out.append({
            "name": name,
            "team_key": f"458.l.123.t.{i}",
            "rank": i,
            "stats": {
                "R": 100 + i * 10, "HR": 20 + i, "RBI": 100 + i * 8,
                "SB": 10 + i, "AVG": 0.250 + i / 1000,
                "W": 10 + i, "K": 150 + i * 5, "SV": 5 + i,
                "ERA": 3.50 + i / 100, "WHIP": 1.15 + i / 1000,
            },
        })
    return out


def roster_for_team(team_index: int) -> list[dict]:
    """One team's roster: 6 hitters, 5 pitchers (1 closer + 4 starters)."""
    base_h = team_index * 6
    base_p = team_index * 5
    out = []
    # Hitters in OF / Util / BN slots
    slot_cycle = ["OF", "OF", "OF", "Util", "BN", "BN"]
    for i in range(6):
        idx = base_h + i
        out.append({
            "name": f"Hitter{idx:03d}",
            "positions": ["OF", "Util"] if i < 4 else ["OF"],
            "selected_position": slot_cycle[i],
            "player_id": f"yh_h_{idx:03d}",
            "status": "",
        })
    # Pitchers — first one is a closer (RP), rest are SP
    for i in range(5):
        idx = base_p + i
        is_closer = i == 0
        out.append({
            "name": f"Pitcher{idx:03d}",
            "positions": ["RP", "P"] if is_closer else ["SP", "P"],
            "selected_position": "P",
            "player_id": f"yh_p_{idx:03d}",
            "status": "",
        })
    return out


def all_rosters() -> dict[str, list[dict]]:
    """All 12 teams' rosters by team name."""
    return {name: roster_for_team(i) for i, name in enumerate(TEAM_NAMES)}


def hitter_game_logs() -> dict[str, dict]:
    """Mid-season actuals keyed by normalized name. ~half the league."""
    out = {}
    for i in range(60):
        name = f"hitter{i:03d}"
        out[name] = {
            "r": 40 + (i % 20), "hr": 10 + (i % 12), "rbi": 40 + (i % 20),
            "sb": 1 + (i % 8), "avg": 0.260 + (i % 30) / 1000,
        }
    return out


def pitcher_game_logs() -> dict[str, dict]:
    out = {}
    for i in range(40):
        name = f"pitcher{i:03d}"
        out[name] = {
            "w": 5 + (i % 6), "k": 90 + (i % 40),
            "sv": 12 if i < 12 else 0,
            "era": 3.20 + (i % 10) / 10, "whip": 1.10 + (i % 10) / 100,
        }
    return out


def free_agents() -> list[dict]:
    """20 free agents — players NOT on any roster."""
    out = []
    # Hitters 72..91 (past the 72 used by 12 teams x 6 hitters)
    for i in range(72, 92):
        out.append({
            "name": f"Hitter{i:03d}",
            "positions": ["OF"],
            "selected_position": "BN",
            "player_id": f"yh_h_{i:03d}",
            "status": "",
        })
    return out


def transactions() -> list[dict]:
    """Empty by default — transaction analyzer handles this."""
    return []


def schedule_payload(start_date: date, end_date: date) -> dict:
    """Empty schedule — get_probable_starters tolerates this."""
    return {}


def team_batting_stats() -> dict[str, Any]:
    """Empty team batting stats — matchup factors fall back to defaults."""
    return {}


def scoring_period() -> tuple[str, str]:
    """Sunday-ending scoring week."""
    return ("2026-04-13", "2026-04-19")  # Mon-Sun
