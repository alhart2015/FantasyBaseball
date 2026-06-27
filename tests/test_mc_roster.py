from fantasy_baseball.mc_roster import PA_PER_GAME, build_effective_roster
from fantasy_baseball.models.player import HitterStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.standings import CategoryStats
from fantasy_baseball.scoring import LeagueContext
from fantasy_baseball.utils.constants import ALL_CATEGORIES


def _h(name, slot, pid, r=80, g=150, pa=600):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[Position.OF],
        selected_position=slot,
        yahoo_id=pid,
        rest_of_season=HitterStats.from_dict(
            {
                "r": r,
                "hr": 20,
                "rbi": 70,
                "sb": 5,
                "h": 150,
                "ab": 550,
                "pa": pa,
                "g": g,
            }
        ),
    )


def _ctx(team="Me", others=("Opp",)):
    base = {t: CategoryStats() for t in others}
    sds = {t: {c: 5.0 for c in ALL_CATEGORIES} for t in (team, *others)}
    return LeagueContext(baseline_other_team_stats=base, team_sds=sds, team_name=team)


def test_active_and_bench_classification():
    roster = [
        _h("Starter", Position.OF, "1"),
        _h("BenchBat", Position.BN, "2"),  # healthy bench hitter -> fill pool
    ]
    eff = build_effective_roster(roster, _ctx())
    assert [b.player.name for b in eff.active] == ["Starter"]
    assert [b.player.name for b in eff.bench] == ["BenchBat"]


def test_bench_body_value_and_games():
    roster = [_h("Starter", Position.OF, "1"), _h("BenchBat", Position.BN, "2", g=120)]
    eff = build_effective_roster(roster, _ctx())
    bench = eff.bench[0]
    assert bench.g_ros_full == 120  # rest_of_season.g
    assert bench.per_game_value > 0  # sgp / g_ros_full, guarded
    assert Position.OF in bench.eligible_positions


def test_missing_g_derives_from_pa():
    roster = [_h("Starter", Position.OF, "1"), _h("BenchBat", Position.BN, "2", g=0, pa=516)]
    eff = build_effective_roster(roster, _ctx())
    assert abs(eff.bench[0].g_ros_full - 516 / PA_PER_GAME) < 1e-6  # not zeroed


def test_il_hitter_in_active_set_with_partial_factor_and_g_ros_adj():
    # IL hitter activates and displaces an active match by its expected ROS PT.
    # IL pa=300 vs active pa=600 -> factor = (600-300)/600 = 0.5 (a PARTIAL
    # factor, so g_ros_adj = 0.5 * g is non-trivial -- NOT a vacuous 0 == 0).
    roster = [
        _h("Star", Position.OF, "1", r=100, pa=600),
        _h("Weak", Position.OF, "2", r=40, pa=600),  # active body, displaced target
        _h("ILbat", Position.IL, "3", r=90, pa=300, g=80),  # IL -> displaces by 300 PA
    ]
    eff = build_effective_roster(roster, _ctx())
    names = {b.player.name for b in eff.active}
    assert "ILbat" in names  # IL body in the active set
    # exactly one active body should be displaced to a PARTIAL (~0.5) factor.
    displaced = [b for b in eff.active if b.factor < 0.999]
    assert len(displaced) == 1, "one active body should be displaced by the IL return"
    b = displaced[0]
    assert 0.0 < b.factor < 1.0, f"expected a partial factor, got {b.factor}"
    # OBSERVED: with a degenerate all-zero other-team baseline, the delta-roto
    # picker displaces "Star" (not "Weak") -- pin to the picker's actual target.
    assert b.player.name == "Star"
    assert b.factor == 0.5  # (active pa 600 - IL pa 300) / 600
    assert abs(b.g_ros_adj - b.factor * b.player.rest_of_season.g) < 1e-6  # 0.5 * 150 = 75
    # undisplaced bodies (the other active hitter and the IL body) keep factor 1.0
    for nm in ("Weak", "ILbat"):
        body = next(x for x in eff.active if x.player.name == nm)
        assert body.factor == 1.0
        assert abs(body.g_ros_adj - body.player.rest_of_season.g) < 1e-6


def test_duplicate_name_in_active_set_guarded():
    # Same-name collision in active+il must not silently mis-scale: the helper
    # raises rather than apply an ambiguous factor.
    import pytest

    roster = [_h("Same", Position.OF, "1"), _h("Same", Position.OF, "2")]
    # Force a collision under displacement: an IL "Same" makes the picker put
    # the name into the factor dict, which then cannot be re-keyed by identity.
    roster.append(_h("Same", Position.IL, "3"))
    with pytest.raises(ValueError):
        build_effective_roster(roster, _ctx())
