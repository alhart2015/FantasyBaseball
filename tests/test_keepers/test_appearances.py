import pandas as pd

from fantasy_baseball.keepers.appearances import season_eligibility


def _fielding(rows):
    """rows: list of (player_id, position_abbrev, games)."""
    return pd.DataFrame(
        [{"player.id": pid, "position.abbreviation": pos, "stat.games": g} for pid, pos, g in rows]
    )


def test_ten_games_is_eligible_nine_is_not():
    out = season_eligibility(_fielding([(1, "C", 10), (2, "C", 9)]))
    assert out == {1: {"C"}}  # player 2 has no >=10-game slot at all


def test_outfield_is_combined_across_corners():
    # 6 in LF + 5 in CF = 11 combined -> OF, even though neither corner reached 10.
    out = season_eligibility(_fielding([(1, "LF", 6), (1, "CF", 5)]))
    assert out == {1: {"OF"}}


def test_outfield_under_ten_combined_is_not_eligible():
    out = season_eligibility(_fielding([(1, "LF", 6), (1, "CF", 3)]))
    assert out == {}


def test_a_multi_position_player_gets_every_qualifying_slot():
    out = season_eligibility(_fielding([(1, "C", 12), (1, "1B", 15), (1, "SS", 4)]))
    assert out == {1: {"C", "1B"}}  # SS at 4 games drops out


def test_dh_and_unknown_tokens_map_to_no_slot():
    # A pure DH has a DH fielding row (or none); either way no base slot.
    out = season_eligibility(_fielding([(1, "DH", 100)]))
    assert out == {}


def test_pitchers_map_to_p():
    out = season_eligibility(_fielding([(1, "P", 30)]))
    assert out == {1: {"P"}}


def test_player_id_key_is_int():
    out = season_eligibility(_fielding([(660271, "1B", 20)]))
    assert list(out) == [660271] and isinstance(next(iter(out)), int)
