import pytest
from fantasy_baseball.draft.search import find_player

PLAYER_NAMES = [
    "Aaron Judge", "Mookie Betts", "Adley Rutschman",
    "Marcus Semien", "Gerrit Cole", "Emmanuel Clase",
    "Corbin Burnes", "Juan Soto", "Julio Rodriguez",
]


class TestFindPlayer:
    def test_exact_match(self):
        assert find_player("Aaron Judge", PLAYER_NAMES) == "Aaron Judge"

    def test_case_insensitive(self):
        assert find_player("aaron judge", PLAYER_NAMES) == "Aaron Judge"

    def test_partial_match(self):
        assert find_player("judge", PLAYER_NAMES) == "Aaron Judge"

    def test_misspelling(self):
        assert find_player("aron juge", PLAYER_NAMES) == "Aaron Judge"

    def test_last_name_only(self):
        assert find_player("rutschman", PLAYER_NAMES) == "Adley Rutschman"

    def test_no_match_returns_none(self):
        assert find_player("zzzzzzz", PLAYER_NAMES) is None

    def test_close_match_with_threshold(self):
        assert find_player("Corbin Burns", PLAYER_NAMES) == "Corbin Burnes"

    def test_find_multiple_candidates(self):
        candidates = find_player("ju", PLAYER_NAMES, return_top_n=3)
        assert isinstance(candidates, list)
        assert len(candidates) <= 3
