import pytest

from fantasy_baseball.data.yahoo_players import (
    fetch_missing_keepers,
    load_positions_cache,
    merge_position_maps,
    save_positions_cache,
)


class TestMergePositionMaps:
    def test_merges_two_positions(self):
        maps = [
            {"Player A": ["C"], "Player B": ["1B"]},
            {"Player A": ["1B"], "Player C": ["OF"]},
        ]
        merged = merge_position_maps(maps)
        assert set(merged["Player A"]) == {"C", "1B"}
        assert merged["Player B"] == ["1B"]
        assert merged["Player C"] == ["OF"]

    def test_deduplicates(self):
        maps = [
            {"Player A": ["SS"]},
            {"Player A": ["SS", "2B"]},
        ]
        merged = merge_position_maps(maps)
        assert sorted(merged["Player A"]) == ["2B", "SS"]

    def test_empty_input(self):
        assert merge_position_maps([]) == {}


class TestPositionsCache:
    def test_save_and_load_roundtrip(self, tmp_path):
        positions = {"Aaron Judge": ["OF"], "Gerrit Cole": ["SP"]}
        cache_path = tmp_path / "positions.json"
        save_positions_cache(positions, cache_path)
        loaded = load_positions_cache(cache_path)
        assert loaded == positions

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_positions_cache(tmp_path / "nope.json")


class TestFetchMissingKeepers:
    """Tests for fetch_missing_keepers using a mock league object."""

    def _make_league(self, details_map):
        """Create a mock league whose player_details returns from a dict."""

        class MockLeague:
            def player_details(self, name):
                return details_map.get(name, [])

        return MockLeague()

    def test_finds_missing_keeper(self):
        league = self._make_league(
            {
                "Juan Soto": [
                    {
                        "name": {"full": "Juan Soto"},
                        "eligible_positions": [
                            {"position": "OF"},
                            {"position": "Util"},
                        ],
                    }
                ],
            }
        )
        keepers = [{"name": "Juan Soto", "team": "Hart of the Order"}]
        existing = {}
        result = fetch_missing_keepers(league, keepers, existing)
        assert "Juan Soto" in result
        assert "OF" in result["Juan Soto"]

    def test_stores_under_config_name_not_yahoo_name(self):
        """Accented Yahoo name should be stored under the ASCII config name."""
        league = self._make_league(
            {
                "Jose Ramirez": [
                    {
                        "name": {"full": "José Ramírez"},
                        "eligible_positions": [
                            {"position": "3B"},
                            {"position": "IF"},
                        ],
                    }
                ],
            }
        )
        keepers = [{"name": "Jose Ramirez", "team": "TBD"}]
        result = fetch_missing_keepers(league, keepers, {})
        assert "Jose Ramirez" in result
        assert "José Ramírez" not in result
        assert "3B" in result["Jose Ramirez"]

    def test_batter_pitcher_split(self):
        """Two-way players produce two entries: config name + config name (Pitcher)."""
        league = self._make_league(
            {
                "Shohei Ohtani": [
                    {
                        "name": {"full": "Shohei Ohtani (Batter)"},
                        "eligible_positions": [{"position": "Util"}],
                    },
                    {
                        "name": {"full": "Shohei Ohtani (Pitcher)"},
                        "eligible_positions": [{"position": "P"}],
                    },
                ],
            }
        )
        keepers = [{"name": "Shohei Ohtani", "team": "Work in Progress"}]
        result = fetch_missing_keepers(league, keepers, {})
        assert "Shohei Ohtani" in result
        assert "Util" in result["Shohei Ohtani"]
        assert "Shohei Ohtani (Pitcher)" in result
        assert "P" in result["Shohei Ohtani (Pitcher)"]

    def test_skips_keepers_already_in_cache(self):
        league = self._make_league({})
        keepers = [{"name": "Juan Soto", "team": "Hart of the Order"}]
        existing = {"Juan Soto": ["OF"]}
        result = fetch_missing_keepers(league, keepers, existing)
        assert result == {}

    def test_handles_no_results(self):
        league = self._make_league({"Juan Soto": []})
        keepers = [{"name": "Juan Soto", "team": "Hart of the Order"}]
        result = fetch_missing_keepers(league, keepers, {})
        assert result == {}

    def test_handles_api_exception(self):
        class BrokenLeague:
            def player_details(self, name):
                raise RuntimeError("API error")

        keepers = [{"name": "Juan Soto", "team": "Hart of the Order"}]
        result = fetch_missing_keepers(BrokenLeague(), keepers, {})
        assert result == {}

    def test_handles_string_positions(self):
        """Some Yahoo endpoints return positions as plain strings."""
        league = self._make_league(
            {
                "Cal Raleigh": [
                    {
                        "name": {"full": "Cal Raleigh"},
                        "eligible_positions": ["C", "Util"],
                    }
                ],
            }
        )
        keepers = [{"name": "Cal Raleigh", "team": "TBD"}]
        result = fetch_missing_keepers(league, keepers, {})
        assert "Cal Raleigh" in result
        assert "C" in result["Cal Raleigh"]

    def test_multiple_keepers_mixed(self):
        league = self._make_league(
            {
                "Bobby Witt Jr.": [
                    {
                        "name": {"full": "Bobby Witt Jr."},
                        "eligible_positions": [{"position": "SS"}],
                    }
                ],
                "Ghost Player": [],
            }
        )
        keepers = [
            {"name": "Bobby Witt Jr.", "team": "Team A"},
            {"name": "Ghost Player", "team": "Team B"},
            {"name": "Already Cached", "team": "Team C"},
        ]
        existing = {"Already Cached": ["1B"]}
        result = fetch_missing_keepers(league, keepers, existing)
        assert "Bobby Witt Jr." in result
        assert "SS" in result["Bobby Witt Jr."]
        assert "Ghost Player" not in result
        assert "Already Cached" not in result
