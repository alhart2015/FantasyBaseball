from datetime import date, timedelta

import pandas as pd
import pytest

from fantasy_baseball.analysis.game_logs import FULL_HITTER_FIELDS
from fantasy_baseball.hitter_ros.table import TARGET_COUNTS, build_table

HITTER, OTHER, PITCHER, BENCH = 1, 2, 3, 4
TEAM_A, TEAM_B = 10, 20


def _line(**stats):
    return {col: stats.get(col, 0) for col in FULL_HITTER_FIELDS.values()}


def _lineup_row(pk, d, player, team, spot=1, sub=0, position="CF", **stats):
    return {
        "game_pk": pk,
        "game_date": d.isoformat(),
        "team_id": team,
        "is_home": True,
        "player_id": player,
        "batting_order": spot * 100 + sub,
        "lineup_spot": spot,
        "sub_index": sub,
        "position": position,
        **_line(**stats),
    }


def _pitch(d, batter, **kw):
    row = {
        "batter": batter,
        "pitcher": 999,
        "game_date": d.isoformat(),
        "game_type": "R",
        "description": "ball",
        "zone": 12,
        "type": "B",
        "launch_speed": None,
        "launch_angle": None,
        "launch_speed_angle": None,
        "bb_type": None,
        "hc_x": None,
        "hc_y": None,
        "stand": "R",
        "p_throws": "R",
        "balls": 0,
        "strikes": 0,
        "estimated_woba_using_speedangle": None,
        "estimated_ba_using_speedangle": None,
        "bat_speed": None,
        "age_bat": 27,
    }
    row.update(kw)
    return row


def _season(year, start, games, hr_per_game=1, ev=100.0, runs=1):
    """`games` daily games for HITTER and OTHER (both TEAM_A) plus PITCHER batting once."""
    lineups, pitches = [], []
    for i in range(games):
        d = start + timedelta(days=i)
        pk = year * 1000 + i
        lineups.append(_lineup_row(pk, d, HITTER, TEAM_A, pa=4, ab=4, h=2, hr=hr_per_game, r=runs))
        lineups.append(_lineup_row(pk, d, OTHER, TEAM_A, spot=2, pa=4, ab=3, h=1, bb=1, rbi=2))
        if i == 0:
            lineups.append(_lineup_row(pk, d, PITCHER, TEAM_B, spot=9, position="P", pa=2))
        if i == 1:
            # The same pitcher pinch-runs once: still not a hitter, because he pitched.
            lineups.append(_lineup_row(pk, d, PITCHER, TEAM_B, spot=9, sub=1, position="PR"))
        if i == 2:
            # A bench player who only pinch-hits and never pitches is a hitter.
            lineups.append(_lineup_row(pk, d, BENCH, TEAM_B, spot=9, sub=1, position="PH", pa=1))
        pitches.append(_pitch(d, OTHER, pitcher=PITCHER))
        for batter in (HITTER, OTHER):
            pitches.append(_pitch(d, batter))
            pitches.append(_pitch(d, batter, description="swinging_strike", zone=5, type="S"))
            # A pulled fly ball for a right-handed hitter: spray toward left field.
            pitches.append(
                _pitch(
                    d,
                    batter,
                    description="hit_into_play",
                    zone=5,
                    type="X",
                    launch_speed=ev,
                    launch_angle=25.0,
                    launch_speed_angle=6,
                    bb_type="fly_ball",
                    hc_x=60.0,
                    hc_y=100.0,
                    estimated_woba_using_speedangle=1.2,
                    estimated_ba_using_speedangle=0.7,
                )
            )
    return lineups, pitches


def _write(root, seasons, scheduled_last=None):
    """Write a store. The schedule spans each season's games unless ``scheduled_last``
    (year -> date) says the season runs longer than what has been played."""
    scheduled_last = scheduled_last or {}
    for year, (lineups, pitches) in seasons.items():
        path = root / "pitches" / f"season={year}" / "chunk.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pitches).to_parquet(path, index=False)
        path = root / "lineups" / f"{year}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(lineups).to_parquet(path, index=False)
        dates = [date.fromisoformat(r["game_date"]) for r in lineups]
        path = root / "schedule" / f"{year}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        last = scheduled_last.get(year, max(dates))
        pd.DataFrame(
            {"season": [year], "first_date": [min(dates)], "last_date": [last]}
        ).to_parquet(path, index=False)


@pytest.fixture
def store(tmp_path):
    _write(
        tmp_path,
        {
            2024: _season(2024, date(2024, 4, 1), 10),
            2025: _season(2025, date(2025, 4, 1), 21),
        },
    )
    return tmp_path


def _row(df, player, season, week):
    match = df[(df.player_id == player) & (df.season == season) & (df.week == week)]
    assert len(match) == 1
    return match.iloc[0]


def test_grid_and_population(store):
    df = build_table(store)
    hitter_2025 = df[(df.player_id == HITTER) & (df.season == 2025)]
    assert sorted(hitter_2025.week) == [0, 1, 2]
    assert list(hitter_2025.sort_values("week").as_of.dt.date) == [
        date(2025, 4, 1),
        date(2025, 4, 8),
        date(2025, 4, 15),
    ]
    # A pitcher (even one who pinch-ran) is not a hitter row; a pinch-hitter is.
    assert PITCHER not in set(df.player_id)
    assert BENCH in set(df.player_id)


def test_windows_add_up(store):
    df = build_table(store)
    w0 = _row(df, HITTER, 2025, 0)
    assert w0.std_pa == 0 and w0.std_pitches == 0
    assert w0.ros_pa == 21 * 4
    assert w0.p1_pa == 10 * 4 and w0.p1_hr == 10
    assert w0.p3_pa == 10 * 4

    w1 = _row(df, HITTER, 2025, 1)
    assert w1.std_pa == 7 * 4 and w1.std_games == 7
    assert w1.ros_pa == 14 * 4
    assert w1.std_pa + w1.ros_pa == w0.ros_pa
    assert w1.std_spot_sum == 7 and w1.std_starts == 7


def test_pitch_counts(store):
    w1 = _row(build_table(store), HITTER, 2025, 1)
    assert w1.std_pitches == 21
    assert w1.std_swings == 14 and w1.std_whiffs == 7
    assert w1.std_chase_pitches == 7 and w1.std_chase_swings == 0
    assert w1.std_bip == 7 and w1.std_barrels == 7
    assert w1.std_ev_sum == 700.0 and w1.std_ev_sq_sum == 70000.0
    assert w1.std_fb == 7 and w1.std_pulled_air == 7 and w1.std_oppo == 0
    assert w1.std_xwoba_sum == pytest.approx(8.4)
    assert w1.std_xba_n == 7 and w1.std_xba_sum == pytest.approx(4.9)
    assert w1.age == 27


def test_team_context_uses_only_games_before_the_date(store):
    w1 = _row(build_table(store), HITTER, 2025, 1)
    assert w1.team_id == TEAM_A
    # TEAM_A: HITTER r=1 + OTHER r=0 per game.
    assert w1.std_team_games == 7 and w1.std_team_r == 7
    assert w1.p1_team_games == 10 and w1.p1_team_r == 10


def test_no_input_uses_data_on_or_after_the_as_of_date(tmp_path):
    """Change everything on or after week 1's date; no input column at week <= 1 may move."""
    base = tmp_path / "base"
    changed = tmp_path / "changed"
    first, cutoff = date(2025, 4, 1), date(2025, 4, 8)
    _write(base, {2024: _season(2024, date(2024, 4, 1), 10), 2025: _season(2025, first, 21)})

    lineups, pitches = _season(2025, first, 21)
    future_lineups, future_pitches = _season(2025, first, 21, hr_per_game=4, ev=60.0, runs=3)
    lineups = [
        f if date.fromisoformat(f["game_date"]) >= cutoff else p
        for p, f in zip(lineups, future_lineups, strict=True)
    ]
    pitches = [
        f if date.fromisoformat(f["game_date"]) >= cutoff else p
        for p, f in zip(pitches, future_pitches, strict=True)
    ]
    _write(changed, {2024: _season(2024, date(2024, 4, 1), 10), 2025: (lineups, pitches)})

    a, b = build_table(base), build_table(changed)
    inputs = [c for c in a.columns if not c.startswith("ros_")]
    early_a = a[(a.season == 2025) & (a.week <= 1)].reset_index(drop=True)
    early_b = b[(b.season == 2025) & (b.week <= 1)].reset_index(drop=True)
    pd.testing.assert_frame_equal(early_a[inputs], early_b[inputs])
    # The answer must see the change, or the test proves nothing.
    assert not early_a["ros_hr"].equals(early_b["ros_hr"])


def test_targets_are_all_present(store):
    df = build_table(store)
    assert {f"ros_{c}" for c in TARGET_COUNTS} <= set(df.columns)


def test_history_coverage_flags_and_career_window(store):
    df = build_table(store)
    first = _row(df, HITTER, 2024, 0)
    assert (first.p1_in_store, first.p3_seasons_in_store, first.car_seasons_in_store) == (
        False,
        0,
        0,
    )
    second = _row(df, HITTER, 2025, 0)
    assert (second.p1_in_store, second.p3_seasons_in_store, second.car_seasons_in_store) == (
        True,
        1,
        1,
    )
    assert second.car_pa == 10 * 4 and second.car_barrels == 10


def test_season_in_progress_is_flagged_and_uses_the_scheduled_end(tmp_path):
    # 21 games played, but the schedule runs to Apr 30: the season is not over.
    _write(
        tmp_path,
        {2025: _season(2025, date(2025, 4, 1), 21)},
        scheduled_last={2025: date(2025, 4, 30)},
    )
    df = build_table(tmp_path)
    assert not df.season_complete.any()
    w1 = _row(df, HITTER, 2025, 1)
    assert w1.frac_season_left == pytest.approx(23 / 30)


def test_complete_season_is_flagged(store):
    assert build_table(store).season_complete.all()


def test_sprint_columns_exist_without_sprint_files(store):
    df = build_table(store)
    assert df.p1_sprint_speed.isna().all() and df.p2_sprint_runs.isna().all()


def test_sprint_speed_joins_the_previous_season(store):
    path = store / "sprint_speed" / "2024.parquet"
    path.parent.mkdir()
    pd.DataFrame(
        {"player_id": [HITTER], "season": [2024], "sprint_speed": [28.5], "competitive_runs": [40]}
    ).to_parquet(path, index=False)
    df = build_table(store)
    assert _row(df, HITTER, 2025, 0).p1_sprint_speed == 28.5
    assert pd.isna(_row(df, OTHER, 2025, 0).p1_sprint_speed)


def test_spray_uses_atan2_behind_home():
    import duckdb

    from fantasy_baseball.hitter_ros.table import _OPPO, _PULLED

    # A right-handed hitter's dribbler to the left side, fielded behind home's y origin.
    row = "SELECT 'R' AS stand, 100.0 AS hc_x, 210.0 AS hc_y"
    pulled, oppo = duckdb.sql(f"SELECT {_PULLED}, {_OPPO} FROM ({row})").fetchone()
    assert (pulled, oppo) == (True, False)


def test_missing_store_says_how_to_fill_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_pitch_data"):
        build_table(tmp_path)


def test_league_context_uses_only_games_before_the_date(store):
    w1 = _row(build_table(store), HITTER, 2025, 1)
    # Before Apr 8, 2025: 7 days x (HITTER 4 + OTHER 4 PA) and BENCH's 1 PA as a pinch
    # hitter. PITCHER's 2 PA batting as P are not league offense.
    assert w1.lg_std_pa == 7 * 8 + 1
    assert w1.lg_std_hr == 7  # HITTER 1 HR a day
    # 2024: 10 days x 8 PA + BENCH 1.
    assert w1.lg_p1_pa == 10 * 8 + 1 and w1.lg_p3_pa == w1.lg_p1_pa == w1.lg_car_pa


def test_pitchers_batting_are_not_league_or_team_offense(store):
    w0 = _row(build_table(store), BENCH, 2025, 0)
    # BENCH plays for TEAM_B, whose only other batter is PITCHER (batting as P).
    assert w0.p1_team_pa == 1
