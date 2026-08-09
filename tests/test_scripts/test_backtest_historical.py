"""Guards on the historical mode added for #325.

These assert the things whose failure produces a PLAUSIBLE WRONG NUMBER rather than an
exception -- a 2026 lag inside a 2022 forecast, a panel that still contains the future,
a query player who can match himself. None of them would raise; all of them would
silently change the verdict this backtest exists to produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _pt_panel(seasons: list[int]) -> pd.DataFrame:
    """Minimal playing-time panel: one row per (player, season) for two players."""
    rows = []
    for pid in (1, 2):
        for season in seasons:
            rows.append(
                {
                    "mlbam_id": pid,
                    "season": season,
                    "pa": 600.0,
                    "ip": 0.0,
                    "games": 150,
                    "starts": 0,
                    "age": 25 + (season - min(seasons)),
                    "partial_season": False,
                }
            )
    return pd.DataFrame(rows)


class _StubCurve:
    """Stands in for the fitted playing-time curve; returns a flat projection."""

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(600.0, index=features.index)


def test_volume_forecast_reads_no_season_after_its_base_year(monkeypatch) -> None:
    """BASE_YEAR was a module constant read in SIX places inside volume_forecast.

    Threading five of the six leaves a 2026 lag in a 2022 forecast, and nothing raises
    -- the forecast simply uses a season the estimator was never supposed to see.
    """
    import keeper_forecast

    seen: list[int] = []
    panel = _pt_panel(list(range(2018, 2027)))

    monkeypatch.setattr(keeper_forecast, "_panel_path", lambda kind: Path("fake.csv"))
    monkeypatch.setattr(keeper_forecast.pd, "read_csv", lambda *_a, **_k: panel)
    monkeypatch.setattr(
        keeper_forecast,
        "lag_panel",
        lambda *_a, **_k: pd.DataFrame(
            {feature: [1.0, 2.0] for feature in keeper_forecast.FEATURES["hitter"]}
            | {"target": [600.0, 610.0]}
        ),
    )
    monkeypatch.setattr(keeper_forecast, "fit_curve", lambda *_a, **_k: _StubCurve())

    real_series_for = keeper_forecast._series_for

    def spy(panel_arg, year, column, index):
        seen.append(year)
        return real_series_for(panel_arg, year, column, index)

    monkeypatch.setattr(keeper_forecast, "_series_for", spy)

    observed = pd.Series([600.0, 600.0], index=pd.Index([1, 2], name="mlbam_id"))
    keeper_forecast.volume_forecast("hitter", 2022, 2023, observed)

    assert seen, "volume_forecast never consulted the panel"
    assert max(seen) <= 2022, f"read seasons after the base year: {sorted(set(seen))}"


def test_volume_forecast_walks_age_to_the_target_year(monkeypatch) -> None:
    """A two-years-out target applies the curve twice, walking age forward each step.
    The base year is now a parameter, so the step count must come from it and not from
    a constant that happens to be 2026."""
    import keeper_forecast

    ages: list[float] = []
    panel = _pt_panel(list(range(2018, 2027)))

    class _RecordingCurve:
        def predict(self, features: pd.DataFrame) -> pd.Series:
            ages.append(float(features["age"].iloc[0]))
            return pd.Series(600.0, index=features.index)

    monkeypatch.setattr(keeper_forecast, "_panel_path", lambda kind: Path("fake.csv"))
    monkeypatch.setattr(keeper_forecast.pd, "read_csv", lambda *_a, **_k: panel)
    monkeypatch.setattr(
        keeper_forecast,
        "lag_panel",
        lambda *_a, **_k: pd.DataFrame(
            {feature: [1.0, 2.0] for feature in keeper_forecast.FEATURES["hitter"]}
            | {"target": [600.0, 610.0]}
        ),
    )
    monkeypatch.setattr(keeper_forecast, "fit_curve", lambda *_a, **_k: _RecordingCurve())

    observed = pd.Series([600.0, 600.0], index=pd.Index([1, 2], name="mlbam_id"))
    keeper_forecast.volume_forecast("hitter", 2022, 2024, observed)

    # Two steps for a +2 target, and the first projects 2023 (one year younger than
    # the 2024 target), not the target itself.
    assert len(ages) == 2
    assert ages[1] == pytest.approx(ages[0] + 1.0)


def test_transitions_for_matches_the_counts_the_spec_discloses() -> None:
    """The leakage disclosure in the writeup is computed from these.

    Base 2024 is the year where loto and causal COINCIDE, which is why the sensitivity
    check runs on 2023. Running it on 2024 would return a difference of exactly zero
    and read as "the leakage is negligible" when nothing had been measured.
    """
    from backtest_trajectory import transitions_for

    assert transitions_for(2022, "loto") == ((2023, 2024), (2024, 2025))
    assert transitions_for(2023, "loto") == ((2022, 2023), (2024, 2025))
    assert transitions_for(2024, "loto") == ((2022, 2023), (2023, 2024))

    assert transitions_for(2022, "causal") == ()
    assert transitions_for(2023, "causal") == ((2022, 2023),)
    assert transitions_for(2024, "causal") == ((2022, 2023), (2023, 2024))

    # The sensitivity check is only meaningful where the two differ.
    assert transitions_for(2024, "loto") == transitions_for(2024, "causal")
    assert transitions_for(2023, "loto") != transitions_for(2023, "causal")


def test_loto_never_includes_the_transition_being_predicted() -> None:
    """That is the ONE thing leave-one-transition-out guarantees. It does not make the
    fit causal -- for base 2022 both survivors are LATER than the transition predicted,
    which is why the writeup lists it as a third advantage keeper-value keeps."""
    from backtest_trajectory import transitions_for

    for base in (2022, 2023, 2024):
        assert (base, base + 1) not in transitions_for(base, "loto")


def test_future_transition_counts_are_2_1_0() -> None:
    """Printed per base year in the report. If these ever change, the leakage
    disclosure in the PR body is wrong."""
    from backtest_trajectory import transitions_for

    counts = {
        base: sum(1 for _, end in transitions_for(base, "loto") if end > base + 1)
        for base in (2022, 2023, 2024)
    }
    assert counts == {2022: 2, 2023: 1, 2024: 0}


def test_transitions_for_rejects_an_unknown_mode() -> None:
    from backtest_trajectory import transitions_for

    with pytest.raises(ValueError, match="loto"):
        transitions_for(2023, "whatever")


def test_volume_forecast_censors_the_training_panel_to_the_base_year(monkeypatch) -> None:
    """A curve fit on seasons after Y has seen the future it is asked to predict.

    Regression guard: the censor itself landed with the base-year parameterization,
    since threading base_year without it would have been a half-change.
    """
    import keeper_forecast

    captured: dict[str, pd.DataFrame] = {}

    def fake_lag_panel(panel, kind, **kwargs):
        captured["panel"] = panel
        return pd.DataFrame(
            {feature: [1.0, 2.0] for feature in keeper_forecast.FEATURES["hitter"]}
            | {"target": [600.0, 610.0]}
        )

    monkeypatch.setattr(keeper_forecast, "_panel_path", lambda kind: Path("fake.csv"))
    monkeypatch.setattr(
        keeper_forecast.pd, "read_csv", lambda *_a, **_k: _pt_panel(list(range(2018, 2027)))
    )
    monkeypatch.setattr(keeper_forecast, "lag_panel", fake_lag_panel)
    monkeypatch.setattr(keeper_forecast, "fit_curve", lambda *_a, **_k: _StubCurve())

    observed = pd.Series([600.0, 600.0], index=pd.Index([1, 2], name="mlbam_id"))
    keeper_forecast.volume_forecast("hitter", 2022, 2023, observed)

    assert "panel" in captured, "volume_forecast never reached lag_panel"
    assert int(captured["panel"]["season"].max()) <= 2022


def test_fallback_report_counts_per_player_misses() -> None:
    import keeper_forecast

    report = keeper_forecast.FallbackReport(whole_pool=False, per_player=30, total=100)
    assert report.share == pytest.approx(0.30)
    assert report.exceeds_headline_threshold is True


def test_fallback_report_tolerates_a_share_at_the_threshold() -> None:
    import keeper_forecast

    report = keeper_forecast.FallbackReport(whole_pool=False, per_player=25, total=100)
    assert report.exceeds_headline_threshold is False


def test_fallback_report_flags_a_whole_pool_miss_regardless_of_share() -> None:
    """No panel at all fails the base year outright: there is no curve to measure, so
    the number would be about the gap model rather than about keeper-value."""
    import keeper_forecast

    report = keeper_forecast.FallbackReport(whole_pool=True, per_player=0, total=100)
    assert report.share == 0.0
    assert report.exceeds_headline_threshold is True


def test_fallback_report_handles_an_empty_pool_without_dividing_by_zero() -> None:
    import keeper_forecast

    report = keeper_forecast.FallbackReport(whole_pool=False, per_player=0, total=0)
    assert report.share == 0.0
    assert report.exceeds_headline_threshold is False


def test_volume_forecast_reports_a_whole_pool_fallback(monkeypatch) -> None:
    """The missing-panel path returns None; the caller has to be able to tell that
    apart from a curve that simply scored nobody."""
    import keeper_forecast

    monkeypatch.setattr(keeper_forecast, "_panel_path", lambda kind: None)

    observed = pd.Series([600.0], index=pd.Index([1], name="mlbam_id"))
    projected, whole_pool = keeper_forecast.volume_forecast("hitter", 2022, 2023, observed)

    assert projected is None
    assert whole_pool is True


def _scored_panel(rows: list[dict], kind: str = "hitter") -> pd.DataFrame:
    """A scored trajectory panel from partial rows, defaults filled in."""
    from fantasy_baseball.trajectory.panel import score

    base = {
        "mlbam_id": 1,
        "season": 2024,
        "age": 27,
        "pa": 600.0,
        "ab_pa": 0.9,
        "h_ab": 0.280,
        "hr_pa": 0.05,
        "r_pa": 0.15,
        "rbi_pa": 0.14,
        "sb_pa": 0.02,
    }
    return score(pd.DataFrame([base | row for row in rows]), kind)


def _panel_2000_to_2026() -> pd.DataFrame:
    """Three careers spanning 2000-2026, so the 2023-2025 reference window is present."""
    rows = []
    for pid in (1, 2, 3):
        for season in range(2000, 2027):
            rows.append({"mlbam_id": pid, "season": season, "age": 22 + (season - 2000) % 15})
    return _scored_panel(rows)


def test_historical_panel_normalizes_before_truncating() -> None:
    """The order is load-bearing. era_normalize RAISES without the 2023-2025 reference
    seasons, so truncating first aborts base 2022 and 2023 outright -- two of the three
    base years in scope. This test is the only thing standing between a plausible
    restructure and losing two thirds of the evaluation."""
    from backtest_trajectory import historical_panel

    out = historical_panel(_panel_2000_to_2026(), "hitter", 2022, sgp_overrides=None)

    assert not out.empty
    assert int(out["season"].max()) <= 2022
    assert "era_factor_hr_pa" in out.columns


def test_without_player_removes_the_query_player() -> None:
    """No self-matching. An in-sample comparison flatters shape, which fits a model."""
    from backtest_trajectory import historical_panel, without_player

    truncated = historical_panel(_panel_2000_to_2026(), "hitter", 2024, sgp_overrides=None)
    out = without_player(truncated, query_id=1)

    assert 1 in set(truncated["mlbam_id"]), "fixture must contain the player held out"
    assert 1 not in set(out["mlbam_id"])
    assert set(out["mlbam_id"]) == {2, 3}


def test_horizons_for_drops_the_plus_two_run_where_2026_would_be_the_target() -> None:
    """2026 is in progress and is never an outcome year: prorate_partial is
    straight-line and assumes health, so pacing an outcome season would scale an
    injured player UP -- the confound the injury-excluded view exists to remove."""
    from backtest_trajectory import horizons_for

    assert horizons_for(2022) == (1, 2)
    assert horizons_for(2023) == (1, 2)
    assert horizons_for(2024) == (1,)
    assert horizons_for(2025) == ()


def test_keeper_value_sgp_uses_the_panels_own_scorer() -> None:
    """One scorer, or the two estimators are not on one scale.

    A forecast frame is already in the panel's rate schema (keepers.actuals.HITTER_PT
    == 'pa'), so this is a hand-off rather than a translation. Do NOT route through
    keeper_forecast.to_counting, which renames to PA/IP and finishes AVG/ERA/WHIP.
    """
    from backtest_trajectory import keeper_value_sgp

    from fantasy_baseball.trajectory.panel import score

    frame = pd.DataFrame(
        {
            "pa": [600.0],
            "ab_pa": [0.90],
            "h_ab": [0.280],
            "hr_pa": [0.05],
            "r_pa": [0.15],
            "rbi_pa": [0.14],
            "sb_pa": [0.02],
        },
        index=pd.Index([12345], name="mlbam_id"),
    )
    expected = score(frame.reset_index(), "hitter")["sgp"].iloc[0]

    assert keeper_value_sgp(frame, "hitter", None).loc[12345] == pytest.approx(expected)


def test_var_uses_year_Y_eligibility_not_the_outcome_years(tmp_path: Path) -> None:
    """A catcher who stops catching in the outcome year must still be priced against
    the catcher floor -- that is the information the keeper decision had."""
    from backtest_trajectory import var_for

    from fantasy_baseball.config import load_config
    from fantasy_baseball.sgp.denominators import get_sgp_denominators
    from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
    from fantasy_baseball.trajectory.board import season_slots

    # season_slots is @lru_cache(maxsize=4) on (cache_dir, season); without this a
    # previous test's tmp_path can answer for this one.
    season_slots.cache_clear()

    (tmp_path / "mlb_fielding_2023.csv").write_text(
        "player.id,position.abbreviation,stat.games\n12345,C,100\n", encoding="utf-8"
    )
    (tmp_path / "mlb_fielding_2024.csv").write_text(
        "player.id,position.abbreviation,stat.games\n12345,1B,100\n", encoding="utf-8"
    )
    sgp = pd.Series([10.0], index=pd.Index([12345], name="mlbam_id"))
    # The REAL table, not a fixture -- a fixture would be free to drift from the floors
    # the draft board actually nets against, which is the point of using them.
    levels = position_aware_replacement_levels(
        get_sgp_denominators(load_config(PROJECT_ROOT / "config" / "league.yaml").sgp_overrides)
    )

    as_catcher = var_for(sgp, "hitter", 2023, tmp_path, levels)
    as_first_baseman = var_for(sgp, "hitter", 2024, tmp_path, levels)

    # Directional against config/league.yaml's configured floors: the catcher floor is
    # the lowest, so the same SGP is worth MORE as a catcher. An inequality rather than
    # a number, because the floors are re-derived from the denominators and move.
    assert as_catcher.loc[12345] > as_first_baseman.loc[12345]


def test_var_degrades_a_missing_eligibility_cache_to_the_util_floor(tmp_path: Path) -> None:
    """UTIL is the HIGHEST hitter floor, so an unknown player is UNDERSTATED rather
    than credited with scarcity he may not have."""
    from backtest_trajectory import var_for

    from fantasy_baseball.config import load_config
    from fantasy_baseball.sgp.denominators import get_sgp_denominators
    from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
    from fantasy_baseball.trajectory.board import season_slots

    season_slots.cache_clear()
    levels = position_aware_replacement_levels(
        get_sgp_denominators(load_config(PROJECT_ROOT / "config" / "league.yaml").sgp_overrides)
    )
    sgp = pd.Series([10.0], index=pd.Index([12345], name="mlbam_id"))

    out = var_for(sgp, "hitter", 1999, tmp_path, levels)

    assert out.loc[12345] == pytest.approx(10.0 - levels["UTIL"])


CENSOR_CASES = [
    # (anchor_volume, outcome_volumes, expected_censored, why)
    (600.0, {2024: 600.0, 2025: 600.0}, False, "healthy both years"),
    (600.0, {2024: 300.0, 2025: 600.0}, False, "exactly 50% is NOT under the cut"),
    (600.0, {2024: 299.0, 2025: 600.0}, True, "just under 50% in one year censors both"),
    (600.0, {2024: 0.0, 2025: 600.0}, True, "zero volume is censored, by explicit decision"),
    (600.0, {2025: 600.0}, True, "a MISSING row is zero volume, not a skipped year"),
]


@pytest.mark.parametrize("anchor,volumes,expected,why", CENSOR_CASES)
def test_censoring_boundaries(anchor, volumes, expected, why) -> None:
    from backtest_trajectory import Outcome, censored

    outcome = Outcome(mlbam_id=1, sgp_by_year={}, volume_by_year=volumes, anchor_volume=anchor)
    assert censored(outcome, [2024, 2025]) is expected, why


def test_the_ratio_is_against_the_anchor_year_not_the_previous_outcome() -> None:
    """A wrecked Y+1 must not redefine 'normal' for Y+2. Against Y+1 (100 PA) a 500-PA
    Y+2 would look like a 5x recovery and pass; the anchor is what decides."""
    from backtest_trajectory import Outcome, censored

    outcome = Outcome(
        mlbam_id=1, sgp_by_year={}, volume_by_year={2024: 100.0, 2025: 500.0}, anchor_volume=600.0
    )
    assert censored(outcome, [2024, 2025]) is True
    assert censored(outcome, [2025]) is False


def test_a_missing_outcome_row_scores_zero_in_the_ALL_view() -> None:
    """The 0 a vanished player is worth to a roster slot -- which is a different
    question from whether he was wrecked, hence two methods."""
    from backtest_trajectory import Outcome

    outcome = Outcome(mlbam_id=1, sgp_by_year={}, volume_by_year={}, anchor_volume=600.0)
    assert outcome.realized([2024, 2025]) == 0.0


def test_a_zero_anchor_is_censored_rather_than_dividing_by_zero() -> None:
    from backtest_trajectory import Outcome, censored

    outcome = Outcome(mlbam_id=1, sgp_by_year={}, volume_by_year={2024: 600.0}, anchor_volume=0.0)
    assert censored(outcome, [2024]) is True


def test_outcomes_collapse_a_traded_players_split_season() -> None:
    """Two rows, one season. Uncollapsed, a healthy 600-PA year reads as 310 + 290 and
    the injury view censors him as wrecked -- a false positive landing on every player
    who changed teams, correlated with nothing the view claims to control for."""
    from backtest_trajectory import censored, outcomes_for

    panel = _scored_panel(
        [
            {"mlbam_id": 1, "season": 2024, "pa": 310.0},
            {"mlbam_id": 1, "season": 2024, "pa": 290.0},
        ]
    )
    out = outcomes_for(panel, "hitter", 2023, (1,), anchor_volume={1: 600.0})

    assert out[1].volume_by_year[2024] == pytest.approx(600.0)
    assert censored(out[1], [2024]) is False


def test_outcomes_omit_a_season_the_player_did_not_appear_in() -> None:
    from backtest_trajectory import outcomes_for

    panel = _scored_panel([{"mlbam_id": 1, "season": 2024, "pa": 600.0}])
    out = outcomes_for(panel, "hitter", 2023, (1, 2), anchor_volume={1: 600.0})

    assert 2024 in out[1].volume_by_year
    assert 2025 not in out[1].volume_by_year
    assert out[1].realized([2024, 2025]) == pytest.approx(out[1].sgp_by_year[2024])


def test_outcome_sgp_matches_the_shared_collapse_definition() -> None:
    """collapse_split_seasons SUMS sgp across the fragments (comps.py). The point of
    routing through it is not that summing is obviously right -- it is that both
    estimators already fit on that definition, so the outcome side must not carry a
    second one. If the collapse ever changes, this fails instead of diverging."""
    from backtest_trajectory import outcomes_for

    from fantasy_baseball.trajectory.comps import collapse_split_seasons

    panel = _scored_panel(
        [
            {"mlbam_id": 1, "season": 2024, "pa": 310.0},
            {"mlbam_id": 1, "season": 2024, "pa": 290.0},
        ]
    )
    collapsed = collapse_split_seasons(panel)
    expected = float(collapsed.loc[collapsed["season"] == 2024, "sgp"].iloc[0])

    out = outcomes_for(panel, "hitter", 2023, (1,), anchor_volume={1: 600.0})
    assert out[1].sgp_by_year[2024] == pytest.approx(expected)


def test_volume_survives_a_panel_with_nothing_split() -> None:
    """collapse_split_seasons returns the panel UNTOUCHED when there are no duplicates
    and a four-column aggregate when there are, so its schema varies by data. Code
    reading `pa` off its result works on most fixtures and breaks on exactly the split
    season it exists for -- volume must come from its own groupby on the raw panel."""
    from backtest_trajectory import outcomes_for

    panel = _scored_panel([{"mlbam_id": 1, "season": 2024, "pa": 600.0}])
    out = outcomes_for(panel, "hitter", 2023, (1,), anchor_volume={1: 600.0})

    assert out[1].volume_by_year[2024] == pytest.approx(600.0)


def test_resolve_draft_reports_an_unresolved_name_instead_of_dropping_it() -> None:
    """A silent drop thins roster pools NON-randomly -- toward the fringe players a
    keeper board is right to ignore -- which flatters both estimators and quietly
    shrinks the decision being measured."""
    from backtest_trajectory import resolve_draft

    people = pd.DataFrame({"id": [1], "fullName": ["Yordan Alvarez"]})
    result = resolve_draft(
        [
            {"team": "Hart of the Order", "player": "Yordan Alvarez"},
            {"team": "Hart of the Order", "player": "Nobody At All"},
        ],
        people,
        {1: "hitter"},
    )

    assert result.by_team["Hart of the Order"] == [1]
    assert ("Hart of the Order", "Nobody At All") in result.unresolved


def test_resolve_draft_reports_an_ambiguous_name_rather_than_picking_one() -> None:
    """Two hitters called Max Muncy is a REAL case (roster_join.py). Picking one
    silently means a roster is scored against the wrong player's career."""
    from backtest_trajectory import resolve_draft

    people = pd.DataFrame({"id": [1, 2], "fullName": ["Max Muncy", "Max Muncy"]})
    result = resolve_draft(
        [{"team": "Spacemen", "player": "Max Muncy"}], people, {1: "hitter", 2: "hitter"}
    )

    assert result.by_team.get("Spacemen", []) == []
    assert ("Spacemen", "Max Muncy") in result.ambiguous


def test_resolve_draft_normalizes_accents() -> None:
    """The people cache carries the accented spelling and the draft file does not.
    Written as an escape so the source stays ASCII while still exercising the join --
    two identical ASCII strings would pass against naive exact matching."""
    from backtest_trajectory import resolve_draft

    people = pd.DataFrame({"id": [7], "fullName": ["Jesús Luzardo"]})
    result = resolve_draft(
        [{"team": "Spacemen", "player": "Jesus Luzardo"}], people, {7: "pitcher"}
    )

    assert result.by_team["Spacemen"] == [7]


def test_a_two_way_player_enters_a_roster_once() -> None:
    """The league scores a two-way player once per POOL, but a keeper decision is for
    one roster SPOT. Entering him twice would let one player consume two of a team's
    three keeper slots and double-count him in the ex-post optimum."""
    from backtest_trajectory import resolve_draft

    people = pd.DataFrame({"id": [42], "fullName": ["Shohei Ohtani"]})
    result = resolve_draft(
        [{"team": "Spacemen", "player": "Shohei Ohtani"}],
        people,
        {42: "both"},
        var_by_pool={("hitter", 42): 12.0, ("pitcher", 42): 8.0},
    )

    assert result.by_team["Spacemen"] == [42]
    assert result.pool_of[42] == "hitter"  # the higher year-Y VAR


def test_resolve_draft_ignores_a_player_the_panel_cannot_score() -> None:
    """Present in the people cache but absent from both panels is not 'unresolved' --
    the name resolved fine, the player simply has no scoreable seasons."""
    from backtest_trajectory import resolve_draft

    people = pd.DataFrame({"id": [1, 9], "fullName": ["Yordan Alvarez", "Bench Guy"]})
    result = resolve_draft(
        [
            {"team": "Spacemen", "player": "Yordan Alvarez"},
            {"team": "Spacemen", "player": "Bench Guy"},
        ],
        people,
        {1: "hitter"},
    )

    assert result.by_team["Spacemen"] == [1]
    assert ("Spacemen", "Bench Guy") in result.unscoreable


def test_triple_regret_is_zero_when_the_forecast_picks_the_ex_post_best() -> None:
    from backtest_trajectory import triple_regret

    forecast = {1: 30.0, 2: 20.0, 3: 10.0, 4: 5.0, 5: 1.0}
    realized = {1: 30.0, 2: 20.0, 3: 10.0, 4: 5.0, 5: 1.0}
    picked, regret = triple_regret([1, 2, 3, 4, 5], forecast, realized)

    assert picked == (1, 2, 3)
    assert regret == pytest.approx(0.0)


def test_triple_regret_is_the_realized_shortfall_not_the_forecast_error() -> None:
    """Ranking wrong only costs what it actually cost."""
    from backtest_trajectory import triple_regret

    forecast = {1: 30.0, 2: 20.0, 3: 10.0, 4: 9.0, 5: 1.0}
    realized = {1: 5.0, 2: 20.0, 3: 10.0, 4: 25.0, 5: 1.0}
    picked, regret = triple_regret([1, 2, 3, 4, 5], forecast, realized)

    assert picked == (1, 2, 3)
    # best available was 25 + 20 + 10 = 55; picked 5 + 20 + 10 = 35
    assert regret == pytest.approx(20.0)


def test_triple_regret_breaks_ties_deterministically() -> None:
    from backtest_trajectory import triple_regret

    forecast = {5: 10.0, 3: 10.0, 1: 10.0, 9: 1.0}
    realized = dict.fromkeys((1, 3, 5, 9), 1.0)
    assert triple_regret([9, 5, 3, 1], forecast, realized)[0] == (1, 3, 5)


def test_agreement_rate_counts_identical_triples() -> None:
    """The likeliest outcome is that both estimators name the same three. Those rows
    contribute zero to the difference while counting toward n, so 18-of-20 agreement
    would report a tight interval around zero that MEANS two informative rows."""
    from backtest_trajectory import agreement_rate

    shape = [(1, 2, 3), (1, 2, 3), (4, 5, 6)]
    keeper = [(1, 2, 3), (1, 2, 3), (7, 8, 9)]
    assert agreement_rate(shape, keeper) == pytest.approx(2 / 3)


def test_agreement_rate_refuses_unequal_lengths() -> None:
    """Both estimators pick from the same candidate pool within a view, so the lists
    are equal by construction. zip would truncate silently and report a rate over a
    shorter list than either input."""
    from backtest_trajectory import agreement_rate

    with pytest.raises(ValueError, match="same length"):
        agreement_rate([(1, 2, 3)], [(1, 2, 3), (4, 5, 6)])


def test_usable_draft_years_pins_the_headline_decision_counts() -> None:
    """10 multi-year decisions and 20 one-year ones are the headline of the slice the
    spec ranks most trustworthy. Derived rather than typed, so a base year silently
    dropping out changes the count and something notices."""
    from backtest_trajectory import usable_draft_years

    available = [2023, 2024, 2025]
    assert usable_draft_years(2, available) == [2023]
    assert usable_draft_years(1, available) == [2023, 2024]

    teams_per_year = 10
    assert len(usable_draft_years(2, available)) * teams_per_year == 10
    assert len(usable_draft_years(1, available)) * teams_per_year == 20


def test_intersect_keeps_only_players_both_estimators_scored() -> None:
    from backtest_trajectory import intersect

    assert intersect([1, 2, 3], [2, 3, 4]) == [2, 3]


def test_top_of_board_scores_the_realized_value_of_the_forecast_top_n() -> None:
    from backtest_trajectory import top_of_board

    forecast = {1: 50.0, 2: 40.0, 3: 30.0, 4: 1.0}
    realized = {1: 10.0, 2: 10.0, 3: 10.0, 4: 99.0}
    picked, total = top_of_board(forecast, realized, n=3)

    assert picked == (1, 2, 3)
    assert total == pytest.approx(30.0)


def test_breakout_mask_selects_a_season_25_percent_over_the_prior() -> None:
    from backtest_trajectory import breakout_mask

    # `current`, not `now`: these are build_history's column names, and the
    # fixture previously invented a name the real frame does not carry.
    anchors = pd.DataFrame({"current": [13.0, 12.4, 4.0], "prior": [10.0, 10.0, 10.0]})
    assert list(breakout_mask(anchors)) == [True, False, False]


def test_the_roster_floor_is_applied_per_view_and_names_the_difference() -> None:
    """Censored players leave the candidate pool in the injury view, so a roster can
    clear the floor in ALL and fail it in the other. Unreported, a between-view
    difference reads as 'excluding injuries changed the answer' when it means
    'different teams were scored'."""
    from backtest_trajectory import eligible_rosters

    by_team = {"Spacemen": [1, 2, 3, 4, 5, 6], "Hart of the Order": [7, 8, 9, 10, 11]}

    all_view, all_dropped = eligible_rosters(by_team, scoreable=set(range(1, 12)), floor=5)
    injury_view, injury_dropped = eligible_rosters(
        by_team, scoreable={1, 2, 3, 4, 7, 8, 9, 10, 11}, floor=5
    )

    assert set(all_view) == {"Spacemen", "Hart of the Order"}
    assert all_dropped == []
    assert set(injury_view) == {"Hart of the Order"}
    assert injury_dropped == ["Spacemen"]


def test_bootstrap_difference_separates_an_obvious_gap() -> None:
    from backtest_trajectory import bootstrap_difference

    _lo, hi, share = bootstrap_difference([1.0] * 40, [5.0] * 40)
    assert hi < 0
    assert share == pytest.approx(1.0)


def test_bootstrap_difference_reports_a_null_as_straddling_zero() -> None:
    """TWO independent samples, not the same array twice. The resample is PAIRED, so
    passing one array as both sides makes every draw exactly 0.0 -- the interval
    collapses to a point and the assertion holds whether or not the bootstrap works."""
    import numpy as np
    from backtest_trajectory import bootstrap_difference

    rng = np.random.default_rng(0)
    lo, hi, _ = bootstrap_difference(list(rng.normal(size=200)), list(rng.normal(size=200)))

    assert lo < 0 < hi, "a genuine null must produce a WIDE interval, not a point"


def test_bootstrap_win_share_is_even_when_the_means_are_equal_by_construction() -> None:
    """The interval is the 'cannot separate' signal; the win share is NOT, because a
    PAIRED bootstrap centres on the OBSERVED difference rather than on zero. Two
    independent draws from one distribution differ by ~1 SE and legitimately give a
    share near 0.15. Equal means by construction is what pins the share at a half."""
    import numpy as np
    from backtest_trajectory import bootstrap_difference

    rng = np.random.default_rng(0)
    values = list(rng.normal(size=200))
    shuffled = list(rng.permutation(values))

    lo, hi, share = bootstrap_difference(values, shuffled)

    assert lo < 0 < hi
    assert 0.4 < share < 0.6


def test_bootstrap_difference_is_deterministic_for_a_seed() -> None:
    from backtest_trajectory import bootstrap_difference

    a, b = [1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 9.0]
    assert bootstrap_difference(a, b, seed=3) == bootstrap_difference(a, b, seed=3)


def test_var_scales_the_floor_by_the_number_of_summed_seasons(tmp_path: Path) -> None:
    """A two-season SGP total must be netted against TWO seasons of replacement.

    Subtracting one year's floor from a two-year sum halves the position scarcity
    credit -- the C-to-OF spread goes from 4.5 SGP to 2.3 -- which reorders catchers
    against outfielders in exactly the +2 slices the multi-year claim rests on.
    """
    from backtest_trajectory import var_for

    from fantasy_baseball.trajectory.board import season_slots

    season_slots.cache_clear()
    (tmp_path / "mlb_fielding_2023.csv").write_text(
        "player.id,position.abbreviation,stat.games\n12345,C,100\n", encoding="utf-8"
    )
    levels = {"C": 7.70, "UTIL": 9.96}
    sgp = pd.Series([20.0], index=pd.Index([12345], name="mlbam_id"))

    one_year = var_for(sgp, "hitter", 2023, tmp_path, levels, seasons=1)
    two_year = var_for(sgp, "hitter", 2023, tmp_path, levels, seasons=2)

    assert one_year.loc[12345] == pytest.approx(20.0 - 7.70)
    assert two_year.loc[12345] == pytest.approx(20.0 - 2 * 7.70)
