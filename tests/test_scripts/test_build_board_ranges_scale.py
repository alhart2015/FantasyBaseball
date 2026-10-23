"""scripts/build_board_ranges.board_scale builds the board's scale (#521 review)."""

import sqlite3

import scripts.build_board_ranges as bbr
from fantasy_baseball.config import load_config
from fantasy_baseball.sgp.player_value import REPLACEMENT_RATES


def test_board_scale_runs_and_uses_the_fixed_rates(tmp_path):
    conn = sqlite3.connect(tmp_path / "positions.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE positions (name TEXT, positions TEXT)")
    config = load_config(bbr.PROJECT_ROOT / "config" / "league.yaml")

    scale = bbr.board_scale(2026, None, None, config, conn)

    assert scale.repl_rates == REPLACEMENT_RATES
