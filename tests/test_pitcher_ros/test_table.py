from datetime import date, timedelta

import pandas as pd
import pytest

from fantasy_baseball.pitch_data.store import PITCHING_STAT_COLUMNS
from fantasy_baseball.pitcher_ros.table import TARGET_COUNTS, build_table

STARTER, RELIEVER, MOPUP, TWOWAY, BATTER = 1, 2, 3, 4, 9
TEAM_A, TEAM_B = 10, 20


def _line(pk, d, player, team, appearance=1, resumed=None, **stats):
    row = {c: 0 for c in PITCHING_STAT_COLUMNS}
    row.update(
        game_pk=pk,
        game_date=d.isoformat(),
        team_id=team,
        is_home=team == TEAM_A,
        player_id=player,
        appearance=appearance,
        resumed_date=resumed,
        **stats,
    )
    return row


def _lineup(pk, d, player, team, position, sub=0, **stats):
    return {
        "game_pk": pk,
        "game_date": d.isoformat(),
        "team_id": team,
        "is_home": team == TEAM_A,
        "player_id": player,
        "batting_order": 100 + sub,
        "lineup_spot": 1,
        "sub_index": sub,
        "position": position,
        "pa": 4,
        "r": stats.get("r", 0),
    }


def _pitch(d, pk, pitcher, **kw):
    row = {
        "pitcher": pitcher,
        "batter": BATTER,
        "game_pk": pk,
        "game_date": d.isoformat(),
        "game_type": "R",
        "pitch_type": "FF",
        "release_speed": 95.0,
        "release_spin_rate": 2300.0,
        "release_extension": 6.5,
        "arm_angle": 40.0,
        "pfx_x": -0.5,
        "pfx_z": 1.25,
        "description": "swinging_strike",
        "zone": 5,
        "type": "S",
        "stand": "L",
        "balls": 0,
        "strikes": 0,
        "launch_speed": None,
        "launch_angle": None,
        "launch_speed_angle": None,
        "bb_type": None,
        "woba_value": None,
        "woba_denom": None,
        "estimated_woba_using_speedangle": None,
        "delta_pitcher_run_exp": 0.1,
        "age_pit": 30,
    }
    row.update(kw)
    return row


def _season(year, start, days, k=5, velo=95.0, er=1):
    """``days`` daily games: STARTER starts for TEAM_A (6 innings), RELIEVER closes for
    TEAM_B (1 inning, a save), TWOWAY starts at DH for TEAM_B, and MOPUP, a first
    baseman, pitches the first game. STARTER throws two tracked pitches a game."""
    lines, lineups, pitches = [], [], []
    for i in range(days):
        d = start + timedelta(days=i)
        pk = year * 1000 + i
        lines.append(
            _line(pk, d, STARTER, TEAM_A, gs=1, outs=18, bf=24, k=k, er=er, h=5, bb=2, w=1)
        )
        lines.append(_line(pk, d, RELIEVER, TEAM_B, outs=3, bf=4, k=1, sv=1, svo=1, gf=1))
        lineups.append(_lineup(pk, d, STARTER, TEAM_A, "P", r=1))
        lineups.append(_lineup(pk, d, TWOWAY, TEAM_B, "DH", r=2))
        lineups.append(_lineup(pk, d, MOPUP, TEAM_B, "1B"))
        if i == 0:
            lines.append(_line(pk, d, MOPUP, TEAM_B, appearance=2, outs=3, bf=6, er=3))
            lines.append(_line(pk, d, TWOWAY, TEAM_B, appearance=1, gs=1, outs=15, bf=20))
        pitches.append(_pitch(d, pk, STARTER, release_speed=velo))
        pitches.append(
            _pitch(
                d,
                pk,
                STARTER,
                pitch_type="SL",
                release_speed=85.0,
                description="hit_into_play",
                type="X",
                stand="R",
                launch_speed=100.0,
                launch_angle=10.0,
                launch_speed_angle=4,
                bb_type="ground_ball",
                woba_value=0.9,
                woba_denom=1,
                estimated_woba_using_speedangle=0.5,
            )
        )
    return lines, lineups, pitches


def _write(root, seasons, scheduled=None):
    """Write a store; the schedule spans each season's games unless ``scheduled``
    (year -> (first, last)) says otherwise."""
    scheduled = scheduled or {}
    for year, (lines, lineups, pitches) in seasons.items():
        for kind, rows in (("pitching", lines), ("lineups", lineups)):
            (root / kind).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(root / kind / f"{year}.parquet", index=False)
        path = root / "pitches" / f"season={year}" / "chunk.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pitches).to_parquet(path, index=False)
        dates = [date.fromisoformat(r["game_date"]) for r in lines]
        first, last = scheduled.get(year, (min(dates), max(dates)))
        _schedule(root, year, first, last)


def _schedule(root, year, first, last):
    (root / "schedule").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"season": [year], "first_date": [first], "last_date": [last]}).to_parquet(
        root / "schedule" / f"{year}.parquet", index=False
    )


def _seasons():
    return {
        2024: _season(2024, date(2024, 4, 1), 10),
        2025: _season(2025, date(2025, 4, 1), 21),
    }


@pytest.fixture
def store(tmp_path):
    _write(tmp_path, _seasons())
    return tmp_path


def _row(df, player, season, week):
    match = df[(df.player_id == player) & (df.season == season) & (df.week == week)]
    assert len(match) == 1
    return match.iloc[0]


def test_population_is_pitcher_seasons(store):
    df = build_table(store)
    # A two-way starter is a pitcher; a first baseman who mopped up is not.
    assert {STARTER, RELIEVER, TWOWAY} <= set(df.player_id)
    assert MOPUP not in set(df.player_id)
    assert sorted(df[(df.player_id == STARTER) & (df.season == 2025)].week) == [0, 1, 2]


def test_windows_add_up(store):
    df = build_table(store)
    w0, w1 = _row(df, STARTER, 2025, 0), _row(df, STARTER, 2025, 1)
    assert w0.std_outs == 0 and w0.ros_outs == 21 * 18 and w0.ros_gs == 21
    assert w0.p1_k == 10 * 5 and w0.p3_k == 10 * 5 and w0.car_bf == 10 * 24
    assert w1.std_outs == 7 * 18 and w1.std_outs + w1.ros_outs == w0.ros_outs
    assert w1.l14_g == 7 and w1.l30_g == 7
    closer = _row(df, RELIEVER, 2025, 1)
    assert closer.std_sv == 7 and closer.std_rel_outs == 7 * 3 and closer.std_g == 7
    assert _row(df, STARTER, 2025, 1).std_rel_outs == 0


def test_statcast_counts(store):
    w1 = _row(build_table(store), STARTER, 2025, 1)
    assert w1.std_sc_pitches == 14 and w1.std_sc_fb_pitches == 7 and w1.std_sc_brk_pitches == 7
    assert w1.std_sc_fb_velo_sum / w1.std_sc_fb_velo_n == 95.0
    assert w1.std_sc_fb_ivb_sum / w1.std_sc_fb_ivb_n == pytest.approx(15.0)
    assert w1.std_sc_fb_hbreak_sum / w1.std_sc_fb_hbreak_n == pytest.approx(6.0)
    assert w1.std_sc_whiffs == 7 and w1.std_sc_swings == 14 and w1.std_sc_fb_whiffs == 7
    assert w1.std_sc_bip == 7 and w1.std_sc_hard_hit == 7 and w1.std_sc_gb == 7
    assert w1.std_sc_woba_n == 7 and w1.std_sc_woba_sum == pytest.approx(6.3)
    assert w1.std_sc_xwoba_sum / w1.std_sc_xwoba_n == pytest.approx(0.5)
    assert w1.std_sc_pitches_vs_lhb == 7 and w1.std_sc_rv_sum == pytest.approx(1.4)
    assert w1.p1_sc_pitches == 20 and w1.age == 30


def test_no_input_uses_data_on_or_after_the_as_of_date(tmp_path):
    """Change everything on or after week 1's date; no input at week <= 1 may move."""
    first, cutoff = date(2025, 4, 1), date(2025, 4, 8)
    base, changed = tmp_path / "base", tmp_path / "changed"
    _write(base, _seasons())
    now = _season(2025, first, 21)
    future = _season(2025, first, 21, k=9, velo=99.0, er=4)
    mixed = tuple(
        [
            f if date.fromisoformat(f["game_date"]) >= cutoff else p
            for p, f in zip(a, b, strict=True)
        ]
        for a, b in zip(now, future, strict=True)
    )
    _write(changed, {2024: _seasons()[2024], 2025: mixed})
    a, b = build_table(base), build_table(changed)
    inputs = [c for c in a.columns if not c.startswith("ros_")]
    early_a = a[(a.season == 2025) & (a.week <= 1)].reset_index(drop=True)
    early_b = b[(b.season == 2025) & (b.week <= 1)].reset_index(drop=True)
    pd.testing.assert_frame_equal(early_a[inputs], early_b[inputs])
    # The answer must see the change, or the test proves nothing.
    assert not early_a["ros_k"].equals(early_b["ros_k"])


def test_a_suspended_game_counts_on_the_day_it_finished(tmp_path):
    seasons = _seasons()
    lines, _, pitches = seasons[2025]
    # A game that started before week 1's date and finished after it: its line and its
    # pitches are history only from the resume date on.
    d, resumed = date(2025, 4, 6), "2025-04-12"
    pk = 2025999
    lines.append(_line(pk, d, RELIEVER, TEAM_B, outs=6, bf=8, k=4, resumed=resumed))
    pitches.append(_pitch(d, pk, RELIEVER))
    _write(tmp_path, seasons)
    df = build_table(tmp_path)
    w1, w2 = _row(df, RELIEVER, 2025, 1), _row(df, RELIEVER, 2025, 2)
    assert w1.std_k == 7 and w1.std_sc_pitches == 0
    assert w2.std_k == 14 + 4 and w2.std_sc_pitches == 1


def test_team_context_uses_only_games_before_the_date(store):
    w1 = _row(build_table(store), RELIEVER, 2025, 1)
    assert w1.team_id == TEAM_B
    # TEAM_B: TWOWAY scores 2 a game; RELIEVER saves every game.
    assert w1.std_team_games == 7 and w1.std_team_r == 14
    assert w1.std_team_sv == 7 and w1.std_team_svo == 7
    assert w1.p1_team_games == 10 and w1.p1_team_sv == 10


def test_league_context_is_pitcher_seasons_before_the_date(store):
    w1 = _row(build_table(store), STARTER, 2025, 1)
    # 7 days of STARTER (5 K) and RELIEVER (1 K); no TWOWAY K. MOPUP's line left out.
    assert w1.lg_std_k == 7 * 6 and w1.lg_std_er == 7 * 1
    assert w1.lg_p1_bf == 10 * (24 + 4) + 20


def test_targets_and_coverage_flags(store):
    df = build_table(store)
    assert {f"ros_{c}" for c in TARGET_COUNTS} <= set(df.columns)
    first, later = _row(df, STARTER, 2024, 0), _row(df, STARTER, 2025, 0)
    assert not first.p1_in_store and first.car_seasons_in_store == 0
    assert later.p1_in_store and later.sc_p1_in_store and later.sc_car_seasons_in_store == 1


def test_an_unstarted_season_gets_preseason_rows_to_project_from(store):
    before = build_table(store)
    _schedule(store, 2026, date(2026, 3, 26), date(2026, 9, 27))
    df = build_table(store)
    pd.testing.assert_frame_equal(df[df.season < 2026].reset_index(drop=True), before)
    new = df[df.season == 2026]
    assert set(new.player_id) == {STARTER, RELIEVER, TWOWAY}
    assert set(new.week) == {0} and not new.season_complete.any()
    w0 = _row(df, STARTER, 2026, 0)
    assert w0.team_id == TEAM_A and w0.age == 31  # last team; last age plus one
    assert w0.ros_outs == 0 and w0.p1_outs == 21 * 18


def test_missing_store_says_how_to_fill_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_pitch_data"):
        build_table(tmp_path)
