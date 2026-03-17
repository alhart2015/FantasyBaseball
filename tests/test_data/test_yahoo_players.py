import pytest
import json
from pathlib import Path
from fantasy_baseball.data.yahoo_players import (
    merge_position_maps,
    load_positions_cache,
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
