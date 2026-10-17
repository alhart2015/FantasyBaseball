"""A seeded ``fantasy.db`` for tests that build the draft board (#496).

Tests never read the machine's own ``data/fantasy.db`` (tests/conftest.py
``_isolate_local_files``). The draft-sim tests need projections and positions, so
they get a database built from committed fixture data: the board rows in
``tests/test_draft/fixtures/recs_golden_state_board.json`` (a real board snapshot:
each player's projected stats, ADP and positions), written back as
``blended_projections`` and ``positions`` rows that rebuild the same board.
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
from fantasy_baseball.utils.name_utils import normalize_name

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
    positions = _positions_table(rows)
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


def _positions_table(rows: pd.DataFrame) -> dict[str, list[str]]:
    """The ``positions`` rows that rebuild each fixture player's eligibility.

    The table is keyed by name, as production's is (Yahoo eligibility by name), and
    the board looks a player up by normalized name, falling back to OF for hitters
    and SP for pitchers (board._attach_positions). So players who share a normalized
    name must share one entry -- or, when they disagree, be left out so each gets
    its own fallback. Any other disagreement cannot be written; it raises.
    """
    by_norm: dict[str, list[tuple[str, str, list[str]]]] = {}
    for name, ptype, pos in zip(rows["name"], rows["player_type"], rows["positions"], strict=True):
        by_norm.setdefault(normalize_name(name), []).append((name, ptype, list(pos)))
    table: dict[str, list[str]] = {}
    for players in by_norm.values():
        lists = [pos for _, _, pos in players]
        if all(pos == lists[0] for pos in lists):
            table[players[0][0]] = lists[0]
        elif not all(pos == _FALLBACK[ptype] for _, ptype, pos in players):
            raise ValueError(f"players sharing a name need different positions: {players}")
    return table


#: The board's positions for a player with no ``positions`` row.
_FALLBACK = {"hitter": ["OF"], "pitcher": ["SP"]}
