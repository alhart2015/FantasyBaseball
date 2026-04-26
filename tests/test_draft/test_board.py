import shutil

import pytest

from fantasy_baseball.data.db import (
    create_tables,
    get_connection,
    load_blended_projections,
    load_positions,
)
from fantasy_baseball.draft.board import apply_keepers, build_draft_board


@pytest.fixture
def board_conn(tmp_path, fixtures_dir):
    """Build a SQLite DB from test fixture CSVs + positions."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    create_tables(conn)

    # load_blended_projections expects year subdirectories
    year_dir = tmp_path / "projections" / "2026"
    year_dir.mkdir(parents=True)
    for csv in fixtures_dir.glob("*.csv"):
        shutil.copy(csv, year_dir / csv.name)

    load_blended_projections(conn, tmp_path / "projections", ["steamer"], None)

    positions = {
        "Aaron Judge": ["OF", "DH"],
        "Mookie Betts": ["OF", "SS"],
        "Adley Rutschman": ["C"],
        "Marcus Semien": ["2B", "SS"],
        "Gerrit Cole": ["SP"],
        "Emmanuel Clase": ["RP"],
        "Corbin Burnes": ["SP"],
    }
    load_positions(conn, positions)
    yield conn
    conn.close()


class TestBuildDraftBoard:
    def test_returns_dataframe_with_required_columns(self, board_conn):
        board = build_draft_board(conn=board_conn)
        assert "name" in board.columns
        assert "positions" in board.columns
        assert "total_sgp" in board.columns
        assert "var" in board.columns
        assert "best_position" in board.columns

    def test_players_ranked_by_var_descending(self, board_conn):
        board = build_draft_board(conn=board_conn)
        vars_list = board["var"].tolist()
        assert vars_list == sorted(vars_list, reverse=True)

    def test_all_fixture_players_present(self, board_conn):
        board = build_draft_board(conn=board_conn)
        assert len(board) == 7

    def test_positions_from_cache(self, board_conn):
        board = build_draft_board(conn=board_conn)
        judge = board[board["name"] == "Aaron Judge"].iloc[0]
        assert "OF" in judge["positions"]


class TestApplyKeepers:
    def test_removes_keepers_from_board(self, board_conn):
        board = build_draft_board(conn=board_conn)
        keepers = [{"name": "Aaron Judge", "team": "Spacemen"}]
        filtered = apply_keepers(board, keepers)
        assert "Aaron Judge" not in filtered["name"].values
        assert len(filtered) == len(board) - 1

    def test_keeper_not_in_projections_is_ignored(self, board_conn):
        board = build_draft_board(conn=board_conn)
        keepers = [{"name": "Nonexistent Player", "team": "Nobody"}]
        filtered = apply_keepers(board, keepers)
        assert len(filtered) == len(board)


class TestBoardSgpIntegration:
    def test_durable_sp_has_higher_sgp_than_fragile(self):
        """A workhorse SP with more counting stats has higher SGP than a
        fragile SP at similar rate stats — pure counting-stat math, no
        backfill needed. Regression: backfill blending used to inflate
        fragile pitchers up to healthy-baseline IP, which masked this
        difference. Backfill was removed; verify the comparison still
        holds on the raw projections."""
        fragile = {
            "name": "Fragile Ace",
            "player_type": "pitcher",
            "positions": ["SP"],
            "ip": 145,
            "er": 52,
            "bb": 40,
            "h_allowed": 120,
            "w": 10,
            "k": 160,
            "sv": 0,
            "era": 3.23,
            "whip": 1.10,
        }
        durable = {
            "name": "Durable SP",
            "player_type": "pitcher",
            "positions": ["SP"],
            "ip": 185,
            "er": 66,
            "bb": 51,
            "h_allowed": 153,
            "w": 13,
            "k": 185,
            "sv": 0,
            "era": 3.21,
            "whip": 1.10,
        }
        from fantasy_baseball.sgp.player_value import calculate_player_sgp

        fragile_sgp = calculate_player_sgp(fragile)
        durable_sgp = calculate_player_sgp(durable)
        assert durable_sgp > fragile_sgp
