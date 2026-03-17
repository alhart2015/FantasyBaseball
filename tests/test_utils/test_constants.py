from fantasy_baseball.utils.constants import (
    HITTING_CATEGORIES,
    PITCHING_CATEGORIES,
    ALL_CATEGORIES,
    RATE_STATS,
    INVERSE_STATS,
    ROSTER_SLOTS,
    STARTERS_PER_POSITION,
    DEFAULT_SGP_DENOMINATORS,
    IF_ELIGIBLE,
    NUM_TEAMS,
)


def test_hitting_categories():
    assert HITTING_CATEGORIES == ["R", "HR", "RBI", "SB", "AVG"]


def test_pitching_categories():
    assert PITCHING_CATEGORIES == ["W", "K", "ERA", "WHIP", "SV"]


def test_all_categories_is_union():
    assert ALL_CATEGORIES == HITTING_CATEGORIES + PITCHING_CATEGORIES
    assert len(ALL_CATEGORIES) == 10


def test_rate_stats():
    assert RATE_STATS == {"AVG", "ERA", "WHIP"}


def test_inverse_stats_subset_of_rate():
    assert INVERSE_STATS.issubset(RATE_STATS)
    assert INVERSE_STATS == {"ERA", "WHIP"}


def test_roster_slots_total():
    assert sum(ROSTER_SLOTS.values()) == 25


def test_starters_per_position_total():
    hitter_slots = sum(v for k, v in STARTERS_PER_POSITION.items() if k != "P")
    assert hitter_slots == 120
    assert STARTERS_PER_POSITION["P"] == 90


def test_sgp_denominators_cover_all_categories():
    assert set(DEFAULT_SGP_DENOMINATORS.keys()) == set(ALL_CATEGORIES)


def test_if_eligible_positions():
    assert IF_ELIGIBLE == {"1B", "2B", "3B", "SS"}


def test_num_teams():
    assert NUM_TEAMS == 10
