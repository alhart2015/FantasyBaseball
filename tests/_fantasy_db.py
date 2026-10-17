"""A seeded ``fantasy.db`` for tests that build the draft board (#496).

Tests never read the machine's own ``data/fantasy.db`` (tests/conftest.py
``_isolate_local_files``). The draft-sim tests need projections and positions, so
they get a database built from committed fixture data: the board rows in
``tests/test_draft/fixtures/recs_golden_state_board.json`` (a real board snapshot:
each player's projected stats, ADP and positions), written back as
``blended_projections`` and ``positions`` rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fantasy_baseball.data.db import (
    _BLENDED_TABLE_COLS,
    _df_to_insert_rows,
    _insert_blended_dfs,
    create_tables,
    get_connection,
    load_positions,
)

BOARD_FIXTURE = Path(__file__).parent / "test_draft" / "fixtures" / "recs_golden_state_board.json"
SEASON = 2026


def seed_fantasy_db(path: Path) -> Path:
    """Write the fixture players to a new database at ``path``; returns ``path``."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; seed a fresh path")
    rows = pd.DataFrame(json.loads(BOARD_FIXTURE.read_text()))
    rows["fg_id"] = rows["player_id"].str.split("::").str[0]
    frames = [
        rows[rows["player_type"] == t].drop(columns="positions") for t in ("hitter", "pitcher")
    ]
    # Names can repeat (two players, one name): keep the longer eligibility list, as
    # the board's own name lookup does (board.build_board_from_frames).
    positions: dict[str, list[str]] = {}
    for name, pos in zip(rows["name"], rows["positions"], strict=True):
        if name not in positions or len(pos) > len(positions[name]):
            positions[name] = list(pos)
    conn = get_connection(path)
    try:
        create_tables(conn)
        _insert_blended_dfs(
            conn,
            "blended_projections",
            frames,
            lambda df: _df_to_insert_rows(df, _BLENDED_TABLE_COLS, year=SEASON),
        )
        load_positions(conn, positions)
        conn.commit()
    finally:
        conn.close()
    return path
