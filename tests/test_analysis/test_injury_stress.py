from fantasy_baseball.analysis.injury_stress import HealthProbs, health_probabilities
from fantasy_baseball.models.player import HitterStats, Player, PlayerType
from fantasy_baseball.models.positions import Position


def _hitter(name, *, pa, ab, g):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[Position.OF],
        rest_of_season=HitterStats.from_dict(
            {"r": 80, "hr": 20, "rbi": 70, "sb": 5, "h": 150, "ab": ab, "pa": pa, "g": g}
        ),
        full_season_projection=HitterStats.from_dict(
            {"r": 80, "hr": 20, "rbi": 70, "sb": 5, "h": 150, "ab": ab, "pa": pa, "g": g}
        ),
    )


def test_health_probabilities_sum_to_one_and_ordered():
    players = [_hitter("A", pa=600, ab=550, g=150), _hitter("B", pa=600, ab=550, g=150)]
    hp = health_probabilities(players, 0.5, n_samples=5000, seed=42)
    assert isinstance(hp, HealthProbs)
    assert abs(hp.p_all_healthy + hp.p_one + hp.p_two_plus - 1.0) < 1e-9
    assert 0.0 <= hp.p_two_plus <= hp.p_one  # two-or-more is rarer than exactly-one here
    assert set(hp.per_player) == {"A", "B"}


def test_health_haircut_alone_is_not_significant():
    # A player realizing EXACTLY his expected level (eff_mean) must NOT count as
    # losing significant time -- guards the haircut-vs-injury bug. With threshold
    # 0 no one is ever significant regardless of the systematic mean haircut... so
    # instead assert per-player significance stays well below 1.0 for a healthy
    # full-timer (the haircut does not by itself trip the eff_mean-relative bar).
    players = [_hitter("A", pa=600, ab=550, g=150)]
    hp = health_probabilities(players, 0.5, threshold=0.20, n_samples=20000, seed=1)
    assert hp.per_player["A"] < 0.5
