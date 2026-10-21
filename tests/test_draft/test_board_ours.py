"""build_draft_board(with_ours=True): the real path from SQLite to a mixed board."""

import shutil

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.data.db import (
    create_tables,
    get_connection,
    get_id_pairs,
    load_blended_projections,
    load_positions,
)
from fantasy_baseball.draft import our_players
from fantasy_baseball.draft.board import build_draft_board
from fantasy_baseball.hitter_ros import final as hitter_final
from fantasy_baseball.pitcher_ros import final as pitcher_final

YEAR = 2026


def _ids(fixtures_dir, kind):
    """name -> (fg_id, MLBAM id), read from the fixture CSV."""
    raw = pd.read_csv(fixtures_dir / f"steamer_{kind}.csv", dtype={"playerid": str})
    return {r.Name: (r.playerid, int(r.MLBAMID)) for r in raw.itertuples()}


@pytest.fixture
def conn(tmp_path, fixtures_dir):
    db = get_connection(tmp_path / "test.db")
    create_tables(db)
    year_dir = tmp_path / "projections" / str(YEAR)
    year_dir.mkdir(parents=True)
    for csv in fixtures_dir.glob("*.csv"):
        shutil.copy(csv, year_dir / csv.name)
    load_blended_projections(db, tmp_path / "projections", ["steamer"], None)
    # The raw rows' id pairs, as load_raw_projections stores them.
    for kind, player_type in (("hitters", "hitter"), ("pitchers", "pitcher")):
        for name, (fg_id, mlbam) in _ids(fixtures_dir, kind).items():
            db.execute(
                "INSERT INTO raw_projections (year, system, player_type, name, fg_id, mlbam_id) "
                "VALUES (?, 'steamer', ?, ?, ?, ?)",
                (YEAR, player_type, name, fg_id, mlbam),
            )
    db.commit()
    load_positions(db, {"Aaron Judge": ["OF"], "Gerrit Cole": ["SP"]})
    yield db
    db.close()


def _write_ours(player_type, frame):
    path = our_players.ours_path(player_type, YEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.rename_axis("player_id").reset_index().to_parquet(path, index=False)


def _line(board, name):
    return board[board["name"] == name].iloc[0]


def test_get_id_pairs_keeps_one_player_type(conn, fixtures_dir):
    hitters = get_id_pairs(conn, YEAR, "hitter")
    want = {m for _, m in _ids(fixtures_dir, "hitters").values()}
    assert set(hitters["mlbam_id"].astype(int)) == want
    assert get_id_pairs(conn, YEAR + 1, "hitter").empty


def test_with_ours_mixes_hitters_and_pitchers_on_fangraphs_playing_time(conn, fixtures_dir):
    judge = _ids(fixtures_dir, "hitters")["Aaron Judge"][1]
    cole = _ids(fixtures_dir, "pitchers")["Gerrit Cole"][1]
    _write_ours(
        "hitter",
        pd.DataFrame(
            {"r": [0.2], "hr": [0.1], "rbi": [0.2], "sb": [0.0], "avg": [0.3]}, index=[judge]
        ),
    )
    _write_ours(
        "pitcher",
        pd.DataFrame(
            {"k": [0.4], "w": [0.08], "sv": [0.0], "era": [2.0], "whip": [0.9]}, index=[cole]
        ),
    )
    plain = build_draft_board(conn)
    mixed = build_draft_board(conn, with_ours=True)
    fg_j, ours_j = _line(plain, "Aaron Judge"), _line(mixed, "Aaron Judge")
    w = hitter_final.WEIGHTS["hr"]
    assert ours_j["pa"] == fg_j["pa"]
    assert ours_j["hr"] == pytest.approx((w * 0.1 + (1 - w) * fg_j["hr"] / fg_j["pa"]) * fg_j["pa"])
    fg_c, ours_c = _line(plain, "Gerrit Cole"), _line(mixed, "Gerrit Cole")
    w = pitcher_final.WEIGHTS[pitcher_final.BOARD]["era"]
    assert ours_c["ip"] == fg_c["ip"]
    assert ours_c["era"] == pytest.approx(w * 2.0 + (1 - w) * fg_c["era"])
    # Players ours doesn't project keep the FanGraphs line.
    for name in ("Mookie Betts", "Corbin Burnes"):
        assert _line(mixed, name)["total_sgp"] == pytest.approx(_line(plain, name)["total_sgp"])


def test_without_ours_on_file_the_board_is_fangraphs_only(conn):
    plain = build_draft_board(conn)
    mixed = build_draft_board(conn, with_ours=True)
    pd.testing.assert_frame_equal(plain, mixed)


def test_big_year_lines_ride_along_and_reach_the_dashboard(conn, fixtures_dir):
    from fantasy_baseball.draft.board_ranges import HITTING, PITCHING, ranges_path
    from fantasy_baseball.draft.state import serialize_board

    judge = _ids(fixtures_dir, "hitters")["Aaron Judge"][1]
    cole = _ids(fixtures_dir, "pitchers")["Gerrit Cole"][1]
    path = ranges_path(YEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A ranges file has every stat's tails; these two have only a few filled in.
    cols = {f"{s}_p{q}": [np.nan, np.nan] for s in (*HITTING, *PITCHING) for q in (10, 90)}
    cols |= {"hr_p90": [55.0, np.nan], "avg_p90": [0.310, np.nan], "era_p10": [np.nan, 2.4]}
    pd.DataFrame(
        {"player_id": [judge, cole], "player_type": ["hitter", "pitcher"], **cols}
    ).to_parquet(path, index=False)
    board = build_draft_board(conn, with_ours=True)
    j, c = _line(board, "Aaron Judge"), _line(board, "Gerrit Cole")
    assert j["p90_hr"] == 55.0 and c["p90_era"] == 2.4  # ERA's big year is its low tail
    assert pd.isna(_line(board, "Mookie Betts")["p90_hr"])  # no range: no big year
    served = {p["name"]: p for p in serialize_board(board)}
    assert served["Aaron Judge"]["p90"] == {"hr": 55.0, "avg": 0.31}
    assert served["Gerrit Cole"]["p90"] == {"era": 2.4}
    assert "p90" not in served["Mookie Betts"]
