"""Role-aware pitching replacement model for the ERoto draft recommender.

Supplying SP/RP empirical replacements (``empirical_pitcher_replacements``) opts
the recs path into: padding each team's 9 pitcher slots as 7 SP + 2 RP, and
swapping a candidate against its same-role replacement. The RP line carries
saves, so the saves baseline is non-zero -- without this every undrafted team
sits at exactly 0 saves and any trace of saves is a huge tie-break windfall
(a 0.01-SV starter out-ranking real aces).
"""

from __future__ import annotations

from fantasy_baseball.draft.eroto_recs import _pick_replacement
from fantasy_baseball.draft.recs_integration import (
    build_team_rosters,
    empirical_pitcher_replacements,
)
from fantasy_baseball.draft.state import StateKey
from fantasy_baseball.models.player import Player


def _pitcher(name: str, ip: float, sv: float = 0) -> Player:
    return Player.from_dict(
        {
            "name": name,
            "player_id": f"{name}::pitcher",
            "player_type": "pitcher",
            "positions": ["P"],
            "w": 10,
            "k": 150,
            "sv": sv,
            "ip": ip,
            "er": 60,
            "bb": 40,
            "h_allowed": 140,
        }
    )


def test_replacements_sp_has_no_saves_rp_does():
    reps = empirical_pitcher_replacements()
    assert reps["SP"].rest_of_season.sv == 0
    assert reps["RP"].rest_of_season.sv == 8


def test_role_aware_padding_gives_nonzero_saves_baseline():
    reps = empirical_pitcher_replacements()
    state = {StateKey.KEEPERS.value: [], StateKey.PICKS.value: []}
    rosters = build_team_rosters(state, {}, ["A"], {"P": 9}, reps)

    pitchers = rosters["A"]
    assert len(pitchers) == 9
    assert sum(1 for p in pitchers if p.name == "repl RP") == 2  # 2 RP slots
    assert sum(1 for p in pitchers if p.name == "repl SP") == 7  # 7 SP slots
    # The whole point: a team that has drafted no pitchers is NOT at 0 saves.
    assert sum(p.rest_of_season.sv for p in pitchers) == 16


def test_pick_replacement_routes_by_role():
    reps = empirical_pitcher_replacements()
    # Starter displaces the 0-save SP line -> no saves windfall (the Rasmussen fix).
    assert _pick_replacement(_pitcher("Ace", ip=190), reps).name == "repl SP"
    # Closer displaces the saves-carrying RP line.
    assert _pick_replacement(_pitcher("Closer", ip=65, sv=35), reps).name == "repl RP"


def test_pick_replacement_legacy_without_sp_rp():
    """Without SP/RP keys the legacy 'P' lookup still applies (live default)."""
    legacy = {"P": _pitcher("repl P", ip=120)}
    assert _pick_replacement(_pitcher("SomeArm", ip=150), legacy).name == "repl P"
