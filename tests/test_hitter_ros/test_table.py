from datetime import date, timedelta

import pandas as pd
import pytest

from fantasy_baseball.analysis.game_logs import FULL_HITTER_FIELDS
from fantasy_baseball.hitter_ros.table import TARGET_COUNTS, build_table

HITTER, OTHER, PITCHER = 1, 2, 3
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


def _write(root, seasons):
    lineups_by_year = {}
    for year, (lineups, pitches) in seasons.items():
        lineups_by_year[year] = lineups
        path = root / "pitches" / f"season={year}" / "chunk.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pitches).to_parquet(path, index=False)
    for year, lineups in lineups_by_year.items():
        path = root / "lineups" / f"{year}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(lineups).to_parquet(path, index=False)


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
    # Someone who only ever batted as a pitcher is not a hitter row.
    assert PITCHER not in set(df.player_id)


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
