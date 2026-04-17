import pytest
from fantasy_baseball.models.player import Player, PlayerType, HitterStats, PitcherStats
from fantasy_baseball.models.positions import Position
from fantasy_baseball.lineup.team_optimizer import compute_team_roto, TeamRotoResult
from fantasy_baseball.lineup.optimizer import HitterAssignment, PitcherStarter


CATEGORIES = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]


def _zero_stats():
    return {c: 0.0 for c in CATEGORIES}


def _standing(name, **overrides):
    s = _zero_stats()
    s.update(overrides)
    return {"name": name, "team_key": "", "rank": 0, "stats": s}


def _hitter(name, positions=("OF",), r=70, hr=20, rbi=70, sb=10, h=120, ab=450):
    return Player(
        name=name, player_type=PlayerType.HITTER,
        positions=[Position.parse(p) for p in positions],
        rest_of_season=HitterStats(pa=500, ab=ab, h=h, r=r, hr=hr, rbi=rbi, sb=sb, avg=h/ab),
        selected_position=Position.parse(positions[0]),
    )


def _pitcher(name, positions=("SP",), ip=180, w=12, k=180, sv=0, era=3.50, whip=1.20):
    return Player(
        name=name, player_type=PlayerType.PITCHER,
        positions=[Position.parse(p) for p in positions],
        rest_of_season=PitcherStats(
            ip=ip, w=w, k=k, sv=sv, era=era, whip=whip,
            er=era*ip/9, bb=int(whip*ip*0.3), h_allowed=int(whip*ip*0.7),
        ),
        selected_position=Position.parse(positions[0]),
    )


def test_returns_team_roto_result_with_optimized_lineups():
    roster = [
        _hitter("H1", ["OF"]), _hitter("H2", ["OF"]),
        _pitcher("P1"), _pitcher("P2"),
    ]
    standings = [_standing("Us"), _standing("Rival", R=1, HR=1, RBI=1, W=1, K=1)]
    slots = {"OF": 1, "BN": 1, "P": 1, "IL": 0}
    result = compute_team_roto(
        roster=roster, projected_standings=standings, team_name="Us",
        roster_slots=slots,
    )
    assert isinstance(result, TeamRotoResult)
    assert len(result.hitter_lineup) == 1
    assert len(result.pitcher_starters) == 1
    assert all(isinstance(a, HitterAssignment) for a in result.hitter_lineup)
    assert all(isinstance(s, PitcherStarter) for s in result.pitcher_starters)
    assert result.total_roto > 0
