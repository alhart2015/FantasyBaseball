import pytest
import pandas as pd
from pathlib import Path
from fantasy_baseball.draft.board import build_draft_board, apply_keepers
import json


@pytest.fixture
def position_cache(tmp_path):
    positions = {
        "Aaron Judge": ["OF", "DH"],
        "Mookie Betts": ["OF", "SS"],
        "Adley Rutschman": ["C"],
        "Marcus Semien": ["2B", "SS"],
        "Gerrit Cole": ["SP"],
        "Emmanuel Clase": ["RP"],
        "Corbin Burnes": ["SP"],
    }
    cache_path = tmp_path / "positions.json"
    with open(cache_path, "w") as f:
        json.dump(positions, f)
    return cache_path


class TestBuildDraftBoard:
    def test_returns_dataframe_with_required_columns(self, fixtures_dir, position_cache):
        board = build_draft_board(
            projections_dir=fixtures_dir, positions_path=position_cache, systems=["steamer"],
        )
        assert "name" in board.columns
        assert "positions" in board.columns
        assert "total_sgp" in board.columns
        assert "var" in board.columns
        assert "best_position" in board.columns

    def test_players_ranked_by_var_descending(self, fixtures_dir, position_cache):
        board = build_draft_board(
            projections_dir=fixtures_dir, positions_path=position_cache, systems=["steamer"],
        )
        vars_list = board["var"].tolist()
        assert vars_list == sorted(vars_list, reverse=True)

    def test_all_fixture_players_present(self, fixtures_dir, position_cache):
        board = build_draft_board(
            projections_dir=fixtures_dir, positions_path=position_cache, systems=["steamer"],
        )
        assert len(board) == 7

    def test_positions_from_cache(self, fixtures_dir, position_cache):
        board = build_draft_board(
            projections_dir=fixtures_dir, positions_path=position_cache, systems=["steamer"],
        )
        judge = board[board["name"] == "Aaron Judge"].iloc[0]
        assert "OF" in judge["positions"]


class TestApplyKeepers:
    def test_removes_keepers_from_board(self, fixtures_dir, position_cache):
        board = build_draft_board(
            projections_dir=fixtures_dir, positions_path=position_cache, systems=["steamer"],
        )
        keepers = [{"name": "Aaron Judge", "team": "Spacemen"}]
        filtered = apply_keepers(board, keepers)
        assert "Aaron Judge" not in filtered["name"].values
        assert len(filtered) == len(board) - 1

    def test_keeper_not_in_projections_is_ignored(self, fixtures_dir, position_cache):
        board = build_draft_board(
            projections_dir=fixtures_dir, positions_path=position_cache, systems=["steamer"],
        )
        keepers = [{"name": "Nonexistent Player", "team": "Nobody"}]
        filtered = apply_keepers(board, keepers)
        assert len(filtered) == len(board)
