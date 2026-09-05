"""Realized value bars: the thresholds the board's probabilities are measured against."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fantasy_baseball.trajectory.value_bars import (
    BUST_RANK,
    ValueBars,
    build_bars,
    league_ranks,
    realized_var,
    windows_for,
)

RANKS = {"elite": 2, "keeper": 4, "bust": 6}


def _panels() -> dict[str, pd.DataFrame]:
    """Eight hitters with flat, separated SGP so a realized rank is checkable by hand."""
    rows = []
    for pid in range(1, 9):
        for season in (2022, 2023, 2024):
            rows.append({"mlbam_id": pid, "season": season, "sgp": float(20 - pid)})
    return {"hitter": pd.DataFrame(rows), "pitcher": pd.DataFrame(columns=rows[0].keys())}


def test_realized_var_nets_the_starting_floor_off_every_year() -> None:
    """The floor is fixed at the start season and applied to all `span` years, matching
    `SweptPlayer.points("var")`. A moving floor would not be the board's quantity."""
    panels = _panels()
    floors = {"hitter": {1: 5.0, 2: 5.0}, "pitcher": {}}
    got = realized_var(panels, floors, 2022, 3)
    # Player 1 is 19 SGP a year, so 3 * (19 - 5) = 42; player 2 is 3 * (18 - 5) = 39.
    assert list(got) == [42.0, 39.0]


def test_a_season_the_player_missed_is_a_real_zero() -> None:
    """Out of the league is an outcome, so that year contributes `-floor`, not nothing."""
    panels = _panels()
    panels["hitter"] = panels["hitter"][
        ~((panels["hitter"].mlbam_id == 1) & (panels["hitter"].season == 2023))
    ]
    got = realized_var(panels, {"hitter": {1: 5.0}, "pitcher": {}}, 2022, 3)
    # 14 + (0 - 5) + 14 = 23, not 28.
    assert got.iloc[0] == pytest.approx(23.0)


def test_windows_need_the_whole_span_complete(tmp_path) -> None:
    for season in (2022, 2023, 2024, 2025, 2026):
        (tmp_path / f"mlb_fielding_{season}.csv").write_text("x", encoding="utf-8")
    assert windows_for(1, tmp_path, 2025) == [2022, 2023, 2024, 2025]
    assert windows_for(3, tmp_path, 2025) == [2022, 2023]
    assert windows_for(5, tmp_path, 2025) == [], "no complete 5-year window exists"


def test_a_span_with_no_window_carries_no_bar(tmp_path) -> None:
    """The board must not headline a probability it cannot measure, and bars are NOT
    linear in span, so a missing one cannot be filled in from a shorter one."""
    for season in (2022, 2023):
        (tmp_path / f"mlb_fielding_{season}.csv").write_text("x", encoding="utf-8")
    panels = _panels()
    floors = {s: {"hitter": dict.fromkeys(range(1, 9), 5.0), "pitcher": {}} for s in (2022, 2023)}
    bars = build_bars(
        panels,
        floors,
        panel_vintage="v",
        ranks=RANKS,
        cache_dir=tmp_path,
        last_complete=2023,
        max_span=3,
    )
    assert bars.bar("s1", "keeper") is not None
    assert bars.bar("s3", "keeper") is None, "2022-24 is not complete against last=2023"
    assert "s1" in bars.spans() and "s3" not in bars.spans()


def test_the_ranks_follow_the_league_rules() -> None:
    """Elite and keeper are DERIVED, so a rule change in league.yaml moves the bars."""
    assert league_ranks(10, 3) == {"elite": 10, "keeper": 30, "bust": BUST_RANK}
    assert league_ranks(12, 2) == {"elite": 12, "keeper": 24, "bust": BUST_RANK}


def test_a_mismatched_panel_vintage_is_refused(tmp_path) -> None:
    path = tmp_path / "value_bars.json"
    ValueBars(panel_vintage="old.csv", bars={}, windows={}, ranks=RANKS).save(path)
    assert ValueBars.load(path, panel_vintage="old.csv")
    with pytest.raises(ValueError, match=r"[Rr]egenerate"):
        ValueBars.load(path, panel_vintage="new.csv")


def test_the_artifact_round_trips(tmp_path) -> None:
    path = tmp_path / "value_bars.json"
    original = ValueBars(
        panel_vintage="v", bars={"s3": {"30": 14.2}}, windows={"s3": [2022, 2023]}, ranks=RANKS
    )
    original.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["windows"]["s3"] == [2022, 2023]
    assert ValueBars.load(path).bars == original.bars


def test_the_shipped_bars_are_ordered_and_realistic() -> None:
    """elite > keeper > bust at every span, and every bar above replacement.

    A realized top-100 bar BELOW zero would mean the 100th best player was worse than a
    waiver pickup, which is the reading that made the projected pool the wrong source.
    """
    from fantasy_baseball.trajectory.value_bars import load_shipped_bars

    bars = load_shipped_bars()
    if bars is None:
        pytest.skip("no value_bars.json built in this checkout")
    for span in bars.spans():
        elite, keeper, bust = (bars.bar(span, n) for n in ("elite", "keeper", "bust"))
        assert elite > keeper > bust > 0, f"{span}: {elite}/{keeper}/{bust}"
