from __future__ import annotations

import pandas as pd
import pytest

from fantasy_baseball.trajectory.panel import (
    load_scored_panel,
    panel_path,
    score,
    season_elapsed_fraction,
    widest_newest,
)


def _hitter_rows(rows: list[dict]) -> pd.DataFrame:
    """A hitter panel with sane defaults; each dict overrides what the test cares about."""
    base = {
        "mlbam_id": 1,
        "season": 2015,
        "observed": True,
        "partial_season": False,
        "scheduled_games": 162,
        "pa": 600.0,
        "games": 150.0,
        "age": 27,
        "is_pitcher": False,
        "ab_pa": 0.9,
        "h_ab": 0.280,
        "hr_pa": 0.05,
        "r_pa": 0.15,
        "rbi_pa": 0.14,
        "sb_pa": 0.02,
    }
    return pd.DataFrame([base | row for row in rows])


def _pitcher_rows(rows: list[dict]) -> pd.DataFrame:
    base = {
        "mlbam_id": 1,
        "season": 2015,
        "observed": True,
        "partial_season": False,
        "scheduled_games": 162,
        "ip": 180.0,
        "games": 32.0,
        "starts": 32.0,
        "age": 27,
        "is_pitcher": True,
        "k_ip": 1.0,
        "w_ip": 0.07,
        "sv_ip": 0.0,
        "er_ip": 0.40,
        "bb_ip": 0.30,
        "h_ip": 0.85,
    }
    return pd.DataFrame([base | row for row in rows])


def _write(tmp_path, frame: pd.DataFrame, kind: str = "hitter", span: str = "2000_2026"):
    path = tmp_path / f"{kind}_pt_panel_{span}.csv"
    frame.to_csv(path, index=False)
    return path


def test_pitchers_batting_are_dropped_from_the_hitter_pool(tmp_path) -> None:
    # Pre-2022 NL pitchers taking a handful of PAs are ~45% of the hitter panel and
    # would halve every pre-2022 age bin.
    _write(tmp_path, _hitter_rows([{"mlbam_id": 1}, {"mlbam_id": 2, "is_pitcher": True, "pa": 10}]))
    out = load_scored_panel("hitter", panel_dir=tmp_path)
    assert list(out["mlbam_id"]) == [1]


def test_short_schedules_are_scaled_to_a_full_season(tmp_path) -> None:
    # 2020's 60 games must not read as a career collapse at whatever age a player was.
    _write(tmp_path, _hitter_rows([{"season": 2020, "scheduled_games": 60, "pa": 222.0}]))
    out = load_scored_panel("hitter", panel_dir=tmp_path)
    assert out.loc[0, "pa"] == pytest.approx(222.0 * 162 / 60)


def test_rates_are_untouched_by_schedule_scaling(tmp_path) -> None:
    _write(tmp_path, _hitter_rows([{"season": 2020, "scheduled_games": 60, "pa": 222.0}]))
    out = load_scored_panel("hitter", panel_dir=tmp_path)
    assert out.loc[0, "hr_pa"] == pytest.approx(0.05)
    assert out.loc[0, "avg"] == pytest.approx(0.280)


def test_in_progress_seasons_are_excluded_unless_asked_for(tmp_path) -> None:
    _write(tmp_path, _hitter_rows([{"season": 2025}, {"season": 2026, "partial_season": True}]))
    assert list(load_scored_panel("hitter", panel_dir=tmp_path)["season"]) == [2025]
    both = load_scored_panel("hitter", panel_dir=tmp_path, include_partial=True)
    assert sorted(both["season"]) == [2025, 2026]


def test_unobserved_and_zero_volume_rows_are_dropped(tmp_path) -> None:
    _write(
        tmp_path,
        _hitter_rows(
            [{"mlbam_id": 1}, {"mlbam_id": 2, "observed": False}, {"mlbam_id": 3, "pa": 0.0}]
        ),
    )
    assert list(load_scored_panel("hitter", panel_dir=tmp_path)["mlbam_id"]) == [1]


def test_an_unknown_position_does_not_crash_the_hitter_filter(tmp_path) -> None:
    # pt_model keeps is_pitcher tri-state (NaN = player absent from `people`), which
    # makes the column object-dtype where `~series` raises TypeError. An unconfirmed
    # position is kept: dropping it would silently lose a real hitter.
    frame = _hitter_rows(
        [
            {"mlbam_id": 1},
            {"mlbam_id": 2, "is_pitcher": float("nan")},
            {"mlbam_id": 3, "is_pitcher": True, "pa": 10.0},
        ]
    )
    _write(tmp_path, frame)
    out = load_scored_panel("hitter", panel_dir=tmp_path)
    assert sorted(out["mlbam_id"]) == [1, 2]


def test_a_pitcher_who_really_hit_stays_in_the_hitter_pool(tmp_path) -> None:
    # MLBAM assigns ONE primary position for a whole career, so a converted player
    # (Jason Lane, Anthony Gose) carries `P` through his hitting seasons. Role is a
    # question about usage: 10 PA is a pitcher taking his turn, 500 is a hitter.
    frame = _hitter_rows(
        [
            {"mlbam_id": 1, "is_pitcher": True, "pa": 10.0},
            {"mlbam_id": 2, "is_pitcher": True, "pa": 500.0},
        ]
    )
    _write(tmp_path, frame)
    out = load_scored_panel("hitter", panel_dir=tmp_path)
    assert list(out["mlbam_id"]) == [2]


def test_mop_up_innings_are_dropped_from_the_pitcher_pool(tmp_path) -> None:
    # A position player's one-inning outing is a guaranteed all-zero forward path: he
    # never pitches again. 655 of them were biasing the low-SGP reliever cohort.
    frame = _pitcher_rows(
        [
            {"mlbam_id": 1, "is_pitcher": True, "ip": 60.0},
            {"mlbam_id": 2, "is_pitcher": False, "ip": 1.0},
            {"mlbam_id": 3, "is_pitcher": float("nan"), "ip": 40.0},
        ]
    )
    _write(tmp_path, frame, kind="pitcher")
    out = load_scored_panel("pitcher", panel_dir=tmp_path)
    assert sorted(out["mlbam_id"]) == [1, 3]


def test_a_two_way_season_lands_in_both_pools(tmp_path) -> None:
    # The league drafts and scores a two-way player as two separate assets, so each
    # half must survive into its own pool. Ohtani's primary position is TWP, not P.
    _write(tmp_path, _hitter_rows([{"mlbam_id": 660271, "is_pitcher": False, "pa": 600.0}]))
    _write(
        tmp_path,
        _pitcher_rows([{"mlbam_id": 660271, "is_pitcher": False, "ip": 166.0}]),
        kind="pitcher",
    )
    assert list(load_scored_panel("hitter", panel_dir=tmp_path)["mlbam_id"]) == [660271]
    assert list(load_scored_panel("pitcher", panel_dir=tmp_path)["mlbam_id"]) == [660271]


def test_an_empty_panel_says_so_instead_of_raising_a_pandas_error(tmp_path) -> None:
    _write(tmp_path, _hitter_rows([{"observed": False}]))
    with pytest.raises(ValueError, match="no usable hitter seasons"):
        load_scored_panel("hitter", panel_dir=tmp_path)


def test_score_handles_an_empty_frame(tmp_path) -> None:
    # A row-wise apply over zero rows returns a DataFrame, and the assignment then dies
    # with "Cannot set a DataFrame with multiple columns to the single column sgp".
    # `.iloc[:0]` keeps the columns, which is what a filtered-to-empty panel looks like.
    out = score(_hitter_rows([{}]).iloc[:0], "hitter")
    assert out.empty
    assert "sgp" in out.columns


def test_sgp_rises_with_production(tmp_path) -> None:
    _write(tmp_path, _hitter_rows([{"mlbam_id": 1}, {"mlbam_id": 2, "hr_pa": 0.09}]))
    out = load_scored_panel("hitter", panel_dir=tmp_path).set_index("mlbam_id")
    assert out.loc[2, "sgp"] > out.loc[1, "sgp"]


def test_score_reconstructs_counting_stats_from_rates() -> None:
    out = score(_hitter_rows([{}]), "hitter")
    assert out.loc[0, "hr"] == pytest.approx(30.0)  # 600 PA * 0.05
    assert out.loc[0, "ab"] == pytest.approx(540.0)


def test_panel_path_prefers_the_newest_end_then_the_widest_span(tmp_path) -> None:
    frame = _hitter_rows([{}])
    _write(tmp_path, frame, span="2010_2026")
    wide = _write(tmp_path, frame, span="2000_2026")
    _write(tmp_path, frame, span="2000_2020")
    assert panel_path("hitter", tmp_path) == wide


def test_panel_path_ignores_files_it_cannot_parse(tmp_path) -> None:
    (tmp_path / "hitter_pt_panel_backup.csv").write_text("junk")
    good = _write(tmp_path, _hitter_rows([{}]))
    assert panel_path("hitter", tmp_path) == good


def test_missing_panel_names_the_command_that_builds_it(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="build_pt_panel"):
        panel_path("hitter", tmp_path)


def test_load_rejects_an_unknown_pool(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"hitter.*pitcher"):
        load_scored_panel("catcher", panel_dir=tmp_path)


def test_season_elapsed_fraction_reads_the_busiest_player(tmp_path) -> None:
    frame = _hitter_rows(
        [
            {"mlbam_id": 1, "season": 2026, "partial_season": True, "games": 113.0},
            {"mlbam_id": 2, "season": 2026, "partial_season": True, "games": 40.0},
        ]
    )
    _write(tmp_path, frame)
    panel = load_scored_panel("hitter", panel_dir=tmp_path, include_partial=True)
    assert season_elapsed_fraction(panel, 2026) == pytest.approx(113 / 162)


def test_season_elapsed_fraction_refuses_the_pitcher_panel() -> None:
    # In the pitcher panel `games` is APPEARANCES, so the busiest arm is a reliever at
    # ~57 and this would read a 70%-elapsed season as 35%, doubling every pitcher's
    # projected pace. Elapsed season is a league fact -- it comes off the hitter panel.
    pitchers = pd.DataFrame(
        {"mlbam_id": [1], "season": [2026], "ip": [80.0], "games": [57.0], "age": [28]}
    )
    with pytest.raises(ValueError, match="HITTER panel"):
        season_elapsed_fraction(pitchers, 2026)


def test_widest_newest_ranks_parsed_years_not_the_filename(tmp_path) -> None:
    # A raw string sort puts "_2010_2026" above "_2000_2026" and silently picks the
    # narrower file -- the bug that made the CLI load the keeper build's people cache.
    narrow = tmp_path / "mlb_people_all_2010_2026.csv"
    wide = tmp_path / "mlb_people_all_2000_2026.csv"
    for p in (narrow, wide):
        p.write_text("id,fullName\n")
    assert sorted(tmp_path.glob("mlb_people_all_*.csv"))[-1] == narrow  # the trap
    assert widest_newest(tmp_path.glob("mlb_people_all_*.csv")) == wide


def test_widest_newest_prefers_the_newest_end_over_the_widest_span(tmp_path) -> None:
    for span in ("2000_2020", "2010_2026", "2005_2026"):
        (tmp_path / f"x_{span}.csv").write_text("")
    assert widest_newest(tmp_path.glob("x_*.csv")).stem == "x_2005_2026"


def test_widest_newest_returns_none_when_nothing_parses(tmp_path) -> None:
    (tmp_path / "x_backup.csv").write_text("")
    assert widest_newest(tmp_path.glob("x_*.csv")) is None


def test_season_elapsed_fraction_rejects_an_all_nan_games_column(tmp_path) -> None:
    # min(max(nan, 1e-6), 1.0) is nan -- both comparisons are False -- so the documented
    # clip to (0, 1] does not hold and the nan reaches the payload as the `season_elapsed`
    # a reader dates the board by, saying nothing about the missing games data.
    frame = _hitter_rows([{"season": 2026, "partial_season": True, "games": float("nan")}])
    _write(tmp_path, frame)
    panel = load_scored_panel("hitter", panel_dir=tmp_path, include_partial=True)
    with pytest.raises(ValueError, match="no usable `games`"):
        season_elapsed_fraction(panel, 2026)


def test_season_elapsed_fraction_caps_a_complete_season_at_one(tmp_path) -> None:
    _write(tmp_path, _hitter_rows([{"season": 2015, "games": 162.0}]))
    panel = load_scored_panel("hitter", panel_dir=tmp_path)
    assert season_elapsed_fraction(panel, 2015) == 1.0
