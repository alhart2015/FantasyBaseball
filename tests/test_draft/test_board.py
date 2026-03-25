import pytest
import pandas as pd
from pathlib import Path
from fantasy_baseball.draft.board import build_draft_board, apply_keepers, apply_backfill_blending
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


class TestBackfillBlending:
    def _make_pitcher(self, name, ip, era, sv=0, positions=None):
        er = era * ip / 9
        bb = int(ip * 0.20)
        ha = int(ip * 0.85)
        return {
            "name": name, "player_type": "pitcher",
            "positions": positions or ["SP"],
            "ip": ip, "er": er, "bb": bb, "h_allowed": ha,
            "w": int(ip / 15), "k": int(ip * 0.9), "sv": sv,
            "era": era, "whip": (bb + ha) / ip if ip > 0 else 0,
        }

    def _make_hitter(self, name, ab, avg):
        h = int(ab * avg)
        return {
            "name": name, "player_type": "hitter",
            "positions": ["OF"],
            "ab": ab, "h": h, "r": int(ab * 0.16), "hr": int(ab * 0.05),
            "rbi": int(ab * 0.15), "sb": int(ab * 0.02),
            "avg": avg,
        }

    def test_sp_below_threshold_gets_blended(self):
        pool = pd.DataFrame([self._make_pitcher("Fragile Ace", 145, 3.20)])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["ip"] == pytest.approx(178.0)
        assert result.iloc[0]["er"] > 145 * 3.20 / 9

    def test_sp_above_threshold_unchanged(self):
        pool = pd.DataFrame([self._make_pitcher("Durable SP", 170, 3.60)])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["ip"] == pytest.approx(170.0)

    def test_closer_uses_closer_baseline(self):
        pool = pd.DataFrame([self._make_pitcher("Hurt Closer", 45, 3.00, sv=25, positions=["RP"])])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["ip"] == pytest.approx(60.0)

    def test_middle_reliever_unchanged(self):
        pool = pd.DataFrame([self._make_pitcher("Setup Man", 55, 3.50, sv=5, positions=["RP"])])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["ip"] == pytest.approx(55.0)

    def test_hitter_below_threshold_gets_blended(self):
        pool = pd.DataFrame([self._make_hitter("Fragile Slugger", 520, 0.280)])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["ab"] == pytest.approx(600.0)
        assert result.iloc[0]["h"] / result.iloc[0]["ab"] < 0.280

    def test_hitter_above_threshold_unchanged(self):
        pool = pd.DataFrame([self._make_hitter("Healthy Hitter", 570, 0.280)])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["ab"] == pytest.approx(570.0)

    def test_original_stats_preserved(self):
        pool = pd.DataFrame([self._make_pitcher("Fragile Ace", 145, 3.20)])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["orig_ip"] == pytest.approx(145.0)
        assert result.iloc[0]["orig_era"] == pytest.approx(3.20)
