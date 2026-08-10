"""The in-progress season's anchor: YTD actuals plus a rest-of-season projection (#348).

The rules being pinned here are the four the issue named as the ones that go wrong
silently: SGP is combined on the LINE and never as two scores, the conversion back to
the panel's rate x volume schema preserves AVG/ERA/WHIP, the join is id-keyed, and a
player with no ROS row is dropped explicitly rather than priced off half a season.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fantasy_baseball.trajectory.panel import score
from fantasy_baseball.trajectory.ros_anchor import anchor_full_season

#: A 300-PA half-season: .300/600 AB-rate hitter with 15 HR, 40 R, 45 RBI, 10 SB.
YTD_HITTER = {
    "mlbam_id": 1,
    "season": 2026,
    "age": 27,
    "partial_season": True,
    "pa": 300.0,
    "ab_pa": 0.9,  # 270 AB
    "h_ab": 0.300,  # 81 H
    "hr_pa": 0.05,  # 15 HR
    "r_pa": 40.0 / 300,
    "rbi_pa": 45.0 / 300,
    "sb_pa": 10.0 / 300,
    "games": 70.0,
}

#: 100 IP: 110 K, 8 W, 0 SV, 40 ER (3.60 ERA), 30 BB + 90 H (1.20 WHIP).
YTD_PITCHER = {
    "mlbam_id": 1,
    "season": 2026,
    "age": 27,
    "partial_season": True,
    "ip": 100.0,
    "k_ip": 1.10,
    "w_ip": 0.08,
    "sv_ip": 0.0,
    "er_ip": 0.40,
    "bb_ip": 0.30,
    "h_ip": 0.90,
    "games": 18.0,
    "starts": 18.0,
}


def _panel(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _ros_hitter(**over) -> pd.DataFrame:
    """A 200-PA remainder: 180 AB, 45 H (.250), 10 HR, 30 R, 28 RBI, 5 SB."""
    base = {
        "mlbam_id": 1,
        "pa": 200.0,
        "ab": 180.0,
        "h": 45.0,
        "hr": 10.0,
        "r": 30.0,
        "rbi": 28.0,
        "sb": 5.0,
        "g": 50.0,
    }
    return pd.DataFrame([{**base, **over}])


def _ros_pitcher(**over) -> pd.DataFrame:
    """A 60-IP remainder: 66 K, 5 W, 1 SV, 30 ER (4.50 ERA), 20 BB, 60 H."""
    base = {
        "mlbam_id": 1,
        "ip": 60.0,
        "k": 66.0,
        "w": 5.0,
        "sv": 1.0,
        "er": 30.0,
        "bb": 20.0,
        "h_allowed": 60.0,
        "g": 11.0,
        "gs": 11.0,
    }
    return pd.DataFrame([{**base, **over}])


def test_the_hitter_line_is_the_sum_of_the_two_halves() -> None:
    """Volume and every counting stat ADD. The panel stores rate x volume, so what is
    pinned is the counting line the rates reconstruct."""
    out, dropped = anchor_full_season(_panel(YTD_HITTER), _ros_hitter(), kind="hitter", season=2026)
    row = out.iloc[0]
    assert dropped == []
    assert row["pa"] == pytest.approx(500.0)
    assert row["pa"] * row["ab_pa"] == pytest.approx(450.0)  # 270 + 180
    assert row["pa"] * row["hr_pa"] == pytest.approx(25.0)  # 15 + 10
    assert row["pa"] * row["r_pa"] == pytest.approx(70.0)
    assert row["pa"] * row["rbi_pa"] == pytest.approx(73.0)
    assert row["pa"] * row["sb_pa"] == pytest.approx(15.0)


def test_batting_average_survives_the_round_trip() -> None:
    """AVG is a RATE, so the two halves' averages do not add and neither do their SGP
    scores. .300 over 270 AB and .250 over 180 AB is 126/450 = .280, and the only way to
    get there is to combine the LINE and re-divide."""
    out, _ = anchor_full_season(_panel(YTD_HITTER), _ros_hitter(), kind="hitter", season=2026)
    row = out.iloc[0]
    assert row["h_ab"] == pytest.approx(126.0 / 450.0)
    # Same number reached the other way -- through the schema the panel actually stores.
    ab = row["pa"] * row["ab_pa"]
    assert ab * row["h_ab"] == pytest.approx(126.0)


def test_era_and_whip_survive_the_round_trip() -> None:
    """40 ER over 100 IP with 30 more over 60 is 70/160 -- a 3.94 ERA, not the 4.05 the
    two halves' ERAs average to. WHIP the same way."""
    out, _ = anchor_full_season(_panel(YTD_PITCHER), _ros_pitcher(), kind="pitcher", season=2026)
    row = out.iloc[0]
    assert row["ip"] == pytest.approx(160.0)
    assert row["er_ip"] * 9.0 == pytest.approx(70.0 * 9.0 / 160.0)
    assert row["bb_ip"] + row["h_ip"] == pytest.approx(200.0 / 160.0)
    assert row["ip"] * row["k_ip"] == pytest.approx(176.0)
    assert row["ip"] * row["sv_ip"] == pytest.approx(1.0)


def test_a_player_with_no_ros_row_is_dropped_and_named() -> None:
    """Silence here reads as "the model priced him low". He leaves the board and is
    counted, so the page can say how many went and why."""
    panel = _panel(YTD_HITTER, {**YTD_HITTER, "mlbam_id": 2})
    out, dropped = anchor_full_season(panel, _ros_hitter(), kind="hitter", season=2026)
    assert dropped == [2]
    assert list(out["mlbam_id"]) == [1]


def test_the_join_is_keyed_on_the_id_and_not_the_name() -> None:
    """A ROS row for a DIFFERENT player must not anchor this one. `name` is carried
    through the blend and would join 58 hitters onto a namesake."""
    ros = _ros_hitter(mlbam_id=999).assign(name="Alpha")
    panel = _panel({**YTD_HITTER, "name": "Alpha"})
    out, dropped = anchor_full_season(panel, ros, kind="hitter", season=2026)
    assert dropped == [1]
    assert out.empty


def test_settled_seasons_are_left_alone() -> None:
    """Only the in-progress season gets an anchor. Every other row is the training data
    the fit runs against, and a projection must never reach it."""
    prior = {**YTD_HITTER, "season": 2025, "age": 26, "partial_season": False}
    out, _ = anchor_full_season(
        _panel(prior, YTD_HITTER), _ros_hitter(), kind="hitter", season=2026
    )
    settled = out[out["season"] == 2025].iloc[0]
    assert settled["pa"] == pytest.approx(300.0)
    assert settled["h_ab"] == pytest.approx(0.300)


def test_appearances_are_not_projected_forward() -> None:
    """`games` dates the season (`season_elapsed_fraction` reads the busiest player's
    games) and `starts / games` routes a pitcher to the SP or RP floor from a SETTLED
    season. Both are calendar and role facts about what HAPPENED; writing a projection
    into them would read the season as complete and let a forecast pick the floor."""
    out, _ = anchor_full_season(_panel(YTD_HITTER), _ros_hitter(), kind="hitter", season=2026)
    assert out.iloc[0]["games"] == pytest.approx(70.0)
    out, _ = anchor_full_season(_panel(YTD_PITCHER), _ros_pitcher(), kind="pitcher", season=2026)
    assert (out.iloc[0]["games"], out.iloc[0]["starts"]) == (18.0, 18.0)


def test_the_anchored_season_is_still_flagged_partial() -> None:
    """The comp pool is `~partial_season`. An anchored row that lost the flag would be
    fitted against as though it were a realized career year."""
    out, _ = anchor_full_season(_panel(YTD_HITTER), _ros_hitter(), kind="hitter", season=2026)
    assert bool(out.iloc[0]["partial_season"]) is True


def test_two_ros_rows_for_one_id_are_refused() -> None:
    """Adding both would inflate the remainder by a whole projection, and every number
    downstream would render normally."""
    ros = pd.concat([_ros_hitter(), _ros_hitter()], ignore_index=True)
    with pytest.raises(ValueError, match="two ROS rows"):
        anchor_full_season(_panel(YTD_HITTER), ros, kind="hitter", season=2026)


def test_a_split_in_progress_season_is_refused() -> None:
    """A traded player with two half-rows would take the full remainder onto EACH. The
    board collapses split seasons downstream, so this has to be caught here."""
    panel = _panel(YTD_HITTER, YTD_HITTER)
    with pytest.raises(ValueError, match="split"):
        anchor_full_season(panel, _ros_hitter(), kind="hitter", season=2026)


def test_a_missing_ros_column_is_named() -> None:
    """A schema shift in the FanGraphs export is the reachable cause, and a KeyError
    three frames down names a column the operator never asked for."""
    with pytest.raises(KeyError, match="sb"):
        anchor_full_season(
            _panel(YTD_HITTER), _ros_hitter().drop(columns=["sb"]), kind="hitter", season=2026
        )


def test_a_season_with_no_rows_changes_nothing() -> None:
    """A pitcher panel ending a year early is a real configuration -- `_require_scored_pool`
    is the guard for it -- and this must not be where it fails."""
    prior = {**YTD_HITTER, "season": 2025, "partial_season": False}
    out, dropped = anchor_full_season(_panel(prior), _ros_hitter(), kind="hitter", season=2026)
    assert dropped == []
    assert len(out) == 1


def test_a_finished_base_season_is_not_anchored() -> None:
    """`build_pt_panel._live_seasons` flags a season partial iff `year >= today.year`, so
    a panel rebuilt in January un-flags the season that just ended while leaving it the
    newest one in the file. Anchoring it would bolt a whole remaining-season projection
    onto a complete year -- and the snapshot would be last season's."""
    done = {**YTD_HITTER, "partial_season": False}
    out, dropped = anchor_full_season(_panel(done), _ros_hitter(), kind="hitter", season=2026)
    assert dropped == []
    assert out.iloc[0]["pa"] == pytest.approx(300.0)
    assert out.iloc[0]["h_ab"] == pytest.approx(0.300)


# --- the ordering, which is the requirement and is silent when wrong -----------------


def _actual_panel() -> pd.DataFrame:
    """A panel spanning the 2023-2025 reference window plus an in-progress 2026.

    2026 is a DEPRESSED home-run environment (.030/PA against a ~.055 reference), so its
    era factor is well above 1 and a factor taken off projected rows instead would be
    visibly smaller.
    """
    base = {
        "ab_pa": 0.9,
        "h_ab": 0.280,
        "r_pa": 0.15,
        "rbi_pa": 0.14,
        "sb_pa": 0.02,
        "pa": 600.0,
        "games": 150.0,
        "partial_season": False,
    }
    rows = [
        {**base, "mlbam_id": 1, "season": 2023, "age": 24, "hr_pa": 0.050},
        {**base, "mlbam_id": 1, "season": 2024, "age": 25, "hr_pa": 0.060},
        {**base, "mlbam_id": 1, "season": 2025, "age": 26, "hr_pa": 0.055},
        # 300 PA so far this year, 9 home runs, and still playing.
        {
            **base,
            "mlbam_id": 1,
            "season": 2026,
            "age": 27,
            "hr_pa": 0.030,
            "pa": 300.0,
            "games": 70.0,
            "partial_season": True,
        },
    ]
    # SCORED, like `load_scored_panel` returns it: `era.league_rates` weights by `ab`,
    # which `score` reconstructs, and refuses a frame without it.
    return score(pd.DataFrame(rows), "hitter")


def _pitcher_panel() -> pd.DataFrame:
    """The other pool, present only so `load_anchored_panels` can load both.

    It always loads both -- the base season is a property of the HITTER panel, so even a
    pitcher-only board has to read it -- and `era.league_rates` refuses a frame whose
    denominators belong to the other pool, so a stub returning hitters for both would
    fail on `ip` rather than on anything the test is about.
    """
    base = {
        "mlbam_id": 9,
        "ip": 180.0,
        "k_ip": 1.0,
        "w_ip": 0.07,
        "sv_ip": 0.0,
        "er_ip": 0.40,
        "bb_ip": 0.30,
        "h_ip": 0.85,
        "games": 30.0,
        "starts": 30.0,
        "partial_season": False,
    }
    rows = [{**base, "season": s, "age": 24 + i} for i, s in enumerate((2023, 2024, 2025))]
    rows.append({**base, "season": 2026, "age": 27, "ip": 100.0, "partial_season": True})
    return score(pd.DataFrame(rows), "pitcher")


def _stub_load(monkeypatch, panel: pd.DataFrame, ros: pd.DataFrame) -> None:
    from datetime import date

    from fantasy_baseball.trajectory import ros_anchor

    pitchers = _pitcher_panel()
    ros_pitchers = pd.DataFrame(
        [
            {
                "mlbam_id": 9,
                "ip": 60.0,
                "k": 60.0,
                "w": 4.0,
                "sv": 0.0,
                "er": 24.0,
                "bb": 18.0,
                "h_allowed": 51.0,
            }
        ]
    )
    monkeypatch.setattr(
        "fantasy_baseball.trajectory.panel.load_scored_panel",
        lambda kind, **_kw: (panel if kind == "hitter" else pitchers).copy(),
    )
    monkeypatch.setattr(
        ros_anchor,
        "load_ros_blend",
        lambda *_a, **_kw: ros_anchor.RosBlend(
            date(2026, 7, 21), {"hitter": ros.copy(), "pitcher": ros_pitchers.copy()}
        ),
    )


def test_the_anchor_is_injected_before_the_panel_is_era_normalized(monkeypatch) -> None:
    """Every training row lives on the 2023-2025 reference scale and a raw FanGraphs line
    lives on the raw current-season run environment. Combining after normalization would
    put a half-normalized anchor against fully-normalized comps -- silent, and it would
    bias every fit.

    Pinned on the arithmetic rather than on call order: the 2026 rate that comes out has
    to be the COMBINED line's rate multiplied by the era factor. Normalize-then-inject
    would leave the raw combined rate sitting there unscaled.
    """
    from fantasy_baseball.trajectory.era import era_factors
    from fantasy_baseball.trajectory.ros_anchor import load_anchored_panels

    panel = _actual_panel()
    # 200 PA of remainder with 11 more home runs.
    ros = pd.DataFrame(
        [
            {
                "mlbam_id": 1,
                "pa": 200.0,
                "ab": 180.0,
                "h": 50.0,
                "hr": 11.0,
                "r": 30.0,
                "rbi": 28.0,
                "sb": 4.0,
            }
        ]
    )
    _stub_load(monkeypatch, panel, ros)

    loaded = load_anchored_panels(systems=["steamer"], weights={"steamer": 1.0})
    out = loaded.panels["hitter"]
    row = out[out["season"] == 2026].iloc[0]

    factor = era_factors(panel, "hitter").loc[2026, "hr_pa"]
    combined = (9.0 + 11.0) / 500.0  # 20 HR in 500 PA
    assert row["hr_pa"] == pytest.approx(combined * factor)
    assert loaded.snapshot_date.isoformat() == "2026-07-21"


def test_the_base_season_era_factor_comes_from_the_actual_rows(monkeypatch) -> None:
    """`era_normalize` derives each season's factor from the league rates of the rows it
    is handed. Projections are regressed toward the mean, so a projected league HR/600
    sits closer to the reference than the season really did -- and the factor for the one
    season the whole board is anchored on would quietly shrink."""
    from fantasy_baseball.trajectory.era import era_factors
    from fantasy_baseball.trajectory.ros_anchor import load_anchored_panels

    panel = _actual_panel()
    # A remainder projected right at the reference environment. Anchored, 2026's league
    # rate is dragged most of the way back to normal; the FACTOR must not follow it.
    ros = pd.DataFrame(
        [
            {
                "mlbam_id": 1,
                "pa": 200.0,
                "ab": 180.0,
                "h": 50.0,
                "hr": 15.0,
                "r": 30.0,
                "rbi": 28.0,
                "sb": 4.0,
            }
        ]
    )
    _stub_load(monkeypatch, panel, ros)

    loaded = load_anchored_panels(systems=["steamer"], weights={"steamer": 1.0})
    out = loaded.panels["hitter"]
    applied = out.loc[out["season"] == 2026, "era_factor_hr_pa"].iloc[0]

    from_actual = era_factors(panel, "hitter").loc[2026, "hr_pa"]
    assert applied == pytest.approx(from_actual)
    # And that the distinction is not vacuous: the injected rows imply a smaller factor.
    anchored, _ = anchor_full_season(panel, ros, kind="hitter", season=2026)
    assert era_factors(anchored, "hitter").loc[2026, "hr_pa"] < from_actual - 0.1


def test_a_complete_base_season_reads_no_snapshot(monkeypatch) -> None:
    """In the offseason the newest snapshot belongs to the season that just finished.
    Reading it would bolt a remainder onto a year that has none."""
    from fantasy_baseball.trajectory import ros_anchor
    from fantasy_baseball.trajectory.ros_anchor import load_anchored_panels

    panel = _actual_panel()
    panel["partial_season"] = False
    pitchers = _pitcher_panel()
    pitchers["partial_season"] = False

    monkeypatch.setattr(
        "fantasy_baseball.trajectory.panel.load_scored_panel",
        lambda kind, **_kw: (panel if kind == "hitter" else pitchers).copy(),
    )

    def _refuse(*_a, **_kw):
        raise AssertionError("a complete base season must not read a ROS snapshot")

    monkeypatch.setattr(ros_anchor, "load_ros_blend", _refuse)

    loaded = load_anchored_panels(systems=["steamer"], weights={"steamer": 1.0})
    assert loaded.snapshot_date is None
    assert loaded.no_ros == {"hitter": [], "pitcher": []}
    out = loaded.panels["hitter"]
    assert out[out["season"] == 2026].iloc[0]["pa"] == pytest.approx(300.0)
