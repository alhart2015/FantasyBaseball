from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES,
    DEFAULT_SGP_DENOMINATORS,
    HITTING_CATEGORIES,
    IF_ELIGIBLE,
    INVERSE_STATS,
    NUM_TEAMS,
    PITCHING_CATEGORIES,
    RATE_STATS,
    ROSTER_SLOTS,
    STARTERS_PER_POSITION,
    Category,
)


def test_hitting_categories():
    assert HITTING_CATEGORIES == ["R", "HR", "RBI", "SB", "AVG"]


def test_pitching_categories():
    assert PITCHING_CATEGORIES == ["W", "K", "ERA", "WHIP", "SV"]


def test_all_categories_is_union():
    assert ALL_CATEGORIES == HITTING_CATEGORIES + PITCHING_CATEGORIES
    assert len(ALL_CATEGORIES) == 10


def test_rate_stats():
    assert {"AVG", "ERA", "WHIP"} == RATE_STATS


def test_inverse_stats_subset_of_rate():
    assert INVERSE_STATS.issubset(RATE_STATS)
    assert {"ERA", "WHIP"} == INVERSE_STATS


def test_roster_slots_total():
    assert sum(ROSTER_SLOTS.values()) == 25


def test_starters_per_position_total():
    hitter_slots = sum(v for k, v in STARTERS_PER_POSITION.items() if k != "P")
    assert hitter_slots == 120
    assert STARTERS_PER_POSITION["P"] == 90


def test_sgp_denominators_cover_all_categories():
    assert set(DEFAULT_SGP_DENOMINATORS.keys()) == set(ALL_CATEGORIES)


def test_if_eligible_positions():
    assert {"1B", "2B", "3B", "SS"} == IF_ELIGIBLE


def test_num_teams():
    assert NUM_TEAMS == 10


def test_category_members_are_strings():
    # StrEnum members must compare equal to their string values so existing
    # consumers that pass bare strings keep working during the migration.
    assert Category.R == "R"
    assert Category.HR == "HR"
    assert Category.ERA in {"ERA", "WHIP"}
    assert "WHIP" in INVERSE_STATS


def test_category_members_hash_like_strings():
    # Dicts keyed by Category must be lookup-able with bare strings.
    d = {Category.R: 1, Category.HR: 2}
    assert d["R"] == 1
    assert d["HR"] == 2
    assert DEFAULT_SGP_DENOMINATORS["R"] == 20.0
    assert DEFAULT_SGP_DENOMINATORS[Category.AVG] == 0.005


def test_category_enum_covers_all_categories():
    assert {c.value for c in Category} == set(ALL_CATEGORIES)
    assert len(Category) == 10


def test_hitting_and_pitching_are_enum_members():
    assert all(isinstance(c, Category) for c in HITTING_CATEGORIES)
    assert all(isinstance(c, Category) for c in PITCHING_CATEGORIES)
