import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.baselines import (
    REGRESS_PA,
    baseline_predictions,
    league_rates,
    marcel,
)
from fantasy_baseball.hitter_ros.evaluate import mae_table, paired_bootstrap, scored_players, spread
from fantasy_baseball.hitter_ros.features import TARGETS

COUNTS = ("pa", "ab", "h", "r", "hr", "rbi", "sb")


def _row(player, season, week, **counts):
    row = {"player_id": player, "season": season, "week": week, "season_complete": True}
    row["as_of"] = pd.Timestamp(f"{season}-04-01") + pd.Timedelta(weeks=week)
    for window in ("std", "p1", "p3", "ros"):
        for c in COUNTS:
            row[f"{window}_{c}"] = counts.get(f"{window}_{c}", 0)
    return row


def _league_table():
    # One complete season, 2024: 1000 PA, 900 AB, 270 H, 120 R, 30 HR, 110 RBI, 10 SB.
    return pd.DataFrame(
        [
            _row(
                1,
                2024,
                0,
                ros_pa=1000,
                ros_ab=900,
                ros_h=270,
                ros_r=120,
                ros_hr=30,
                ros_rbi=110,
                ros_sb=10,
            )
        ]
    )


def test_league_rates_come_from_earlier_complete_seasons():
    lg = league_rates(_league_table(), 2025)
    assert lg["hr"] == pytest.approx(0.03) and lg["avg"] == pytest.approx(0.3)
    assert lg["ab_per_pa"] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        league_rates(_league_table(), 2024)  # nothing before 2024


def test_marcel_with_no_history_is_league_average():
    lg = league_rates(_league_table(), 2025)
    rookie = pd.DataFrame([_row(9, 2025, 0)])
    out = marcel(rookie, lg).iloc[0]
    for s in TARGETS:
        assert out[s] == pytest.approx(lg[s])


def test_marcel_weights_this_season_last_season_and_older():
    lg = league_rates(_league_table(), 2025)
    # 100 PA / 10 HR this season, 600 / 30 last season, 1200 / 36 over the two before.
    rows = pd.DataFrame(
        [
            _row(
                9,
                2025,
                5,
                std_pa=100,
                std_hr=10,
                p1_pa=600,
                p1_hr=30,
                p3_pa=1800,
                p3_hr=66,
            )
        ]
    )
    expected_hr = (6 * 10 + 5 * 30 + 3.5 * 36 + REGRESS_PA * 0.03) / (
        6 * 100 + 5 * 600 + 3.5 * 1200 + REGRESS_PA
    )
    assert marcel(rows, lg).iloc[0]["hr"] == pytest.approx(expected_hr)


def test_baseline_predictions_have_the_prediction_shape():
    table = pd.concat([_league_table(), pd.DataFrame([_row(9, 2025, 0), _row(9, 2025, 1)])])
    preds = baseline_predictions(table.reset_index(drop=True), 2025)
    assert set(preds) == {"league_avg", "marcel"}
    for p in preds.values():
        assert list(p.columns) == ["player_id", "season", "week", "as_of", *TARGETS]
        assert len(p) == 2


def _scored(errs_a, errs_b):
    n = len(errs_a)
    actual = pd.DataFrame({s: [0.1] * n for s in TARGETS}, index=range(n))
    actual["pa"] = 500
    a = pd.DataFrame({s: 0.1 + np.array(errs_a) / 600 for s in TARGETS}, index=range(n))
    b = pd.DataFrame({s: 0.1 + np.array(errs_b) / 600 for s in TARGETS}, index=range(n))
    return scored_players({"a": a, "b": b}, actual, min_pa=1)


def test_paired_bootstrap_sign_and_interval():
    rng = np.random.default_rng(1)
    scored = _scored(rng.uniform(0, 2, 400), rng.uniform(3, 5, 400))
    r = paired_bootstrap(scored, "a", "b", n_boot=500).loc["hr"]
    assert r["diff"] == pytest.approx(-3.0, abs=0.2)
    assert r["lo"] < r["diff"] < r["hi"] < 0  # a clearly better: interval below 0


def test_paired_bootstrap_cannot_separate_equal_systems():
    rng = np.random.default_rng(2)
    scored = _scored(rng.uniform(0, 4, 400), rng.uniform(0, 4, 400))
    r = paired_bootstrap(scored, "a", "b", n_boot=500).loc["r"]
    assert r["lo"] < 0 < r["hi"]


def test_spread_reports_projection_and_outcome_sd():
    scored = _scored([0.0, 6.0], [3.0, 3.0])
    sd = spread(scored)
    assert sd.loc["a", "hr"] == pytest.approx(np.std([0.0, 6.0], ddof=1) * 600 / 600)
    assert sd.loc["b", "hr"] == pytest.approx(0.0)
    assert "(actual)" in sd.index


def _stacked_units():
    """Two players scored in two seasons, with different outcomes each season."""
    parts = []
    for season, (a_hr, b_hr) in ((2024, (0.0, 0.1)), (2025, (0.2, 0.3))):
        actual = pd.DataFrame({s: [0.1, 0.1] for s in TARGETS}, index=[1, 2])
        actual["hr"] = [a_hr, b_hr]
        actual["pa"] = 500
        proj = pd.DataFrame({s: [0.1, 0.1] for s in TARGETS}, index=[1, 2])
        parts.append(
            scored_players({"x": proj, "y": proj + 0.01}, actual, min_pa=1).assign(season=season)
        )
    return pd.concat(parts, ignore_index=True)


def test_spread_actual_row_uses_every_season():
    sd = spread(_stacked_units())
    expected = np.std(np.array([0.0, 0.1, 0.2, 0.3]) * 600, ddof=1)
    assert sd.loc["(actual)", "hr"] == pytest.approx(expected)


def test_scored_units_are_player_seasons():
    scored = _stacked_units()
    assert mae_table(scored).loc["x", "n"] == 4
    b = paired_bootstrap(scored, "x", "y", n_boot=50)
    assert b.loc["hr", "diff"] == pytest.approx(
        scored[(scored.stat == "hr") & (scored.system == "x")].abs_err.mean()
        - scored[(scored.stat == "hr") & (scored.system == "y")].abs_err.mean()
    )


def test_write_scores_removes_a_stale_file(tmp_path):
    from fantasy_baseball.hitter_ros.backtest import write_scores

    frame = _stacked_units()
    write_scores(tmp_path, frame, frame)
    assert (tmp_path / "scored_snapshots.parquet").exists()
    write_scores(tmp_path, frame, None)
    assert not (tmp_path / "scored_snapshots.parquet").exists()


def test_mean_over_seasons_weights_each_season_once():
    from fantasy_baseball.hitter_ros.backtest import mean_over_seasons

    scored = _stacked_units()
    # Drop one player from 2025 so the seasons have different sizes.
    scored = scored[~((scored.season == 2025) & (scored.player_id == 2))]
    per_season = (
        scored[(scored.system == "x") & (scored.stat == "hr")].groupby("season").abs_err.mean()
    )
    assert mean_over_seasons(scored).loc["x", "hr"] == pytest.approx(per_season.mean())


def test_preseason_scores_a_season_without_fangraphs_files(tmp_path):
    from fantasy_baseball.hitter_ros.backtest import preseason

    table = pd.concat(
        [
            _league_table(),
            pd.DataFrame(
                [
                    _row(
                        9,
                        2025,
                        0,
                        ros_pa=500,
                        ros_ab=450,
                        ros_h=135,
                        ros_r=70,
                        ros_hr=20,
                        ros_rbi=70,
                        ros_sb=5,
                    ),
                ]
            ),
        ]
    ).reset_index(drop=True)
    preds = table[table.season == 2025][["player_id", "season", "week", "as_of"]].assign(
        **{s: 0.1 for s in TARGETS}
    )
    from fantasy_baseball.hitter_ros.baselines import baseline_predictions

    candidates = {"ours": preds, **baseline_predictions(table, 2025)}
    scored = preseason(table, candidates, 2025, tmp_path)  # tmp_path has no projections
    assert scored is not None
    assert set(scored.system) == {"ours", "league_avg", "marcel"}


def test_summary_keeps_fangraphs_when_older_seasons_have_no_files():
    from fantasy_baseball.hitter_ros.backtest import summarize

    old = _stacked_units()
    old = old[old.season == 2024].assign(
        system=lambda d: d.system.map({"x": "ours", "y": "marcel"})
    )
    new = _stacked_units()
    new = new[new.season == 2025].assign(
        system=lambda d: d.system.map({"x": "ours", "y": "fg_blend"})
    )
    md = "\n".join(summarize(pd.concat([old, new], ignore_index=True), None))
    assert "seasons with FanGraphs files** (2025)" in md


def _career_table(seasons_in_store=10, std_pa_by_week5=200, **career_pa):
    """Season 2025 rows per player: week 0 (as of 2025-03-25) with the given career PA,
    and week 5 (as of 2025-04-29) with ``std_pa_by_week5`` PA this season so far."""
    rows = []
    for player, pa in career_pa.items():
        pid = int(player.removeprefix("p"))
        for week, as_of, std_pa in ((0, "2025-03-25", 0), (5, "2025-04-29", std_pa_by_week5)):
            rows.append(
                {
                    "player_id": pid,
                    "season": 2025,
                    "week": week,
                    "as_of": pd.Timestamp(as_of),
                    "car_pa": float(pa),
                    "std_pa": float(std_pa),
                    "car_seasons_in_store": seasons_in_store,
                }
            )
    # The real table stores season as int32; the scored frames use int64.
    return pd.DataFrame(rows).astype({"season": "int32"})


def test_tag_experience_splits_at_300_career_pa_before_the_season():
    from fantasy_baseball.hitter_ros.backtest import VET_MIN_CAREER_PA, tag_experience

    assert VET_MIN_CAREER_PA == 300
    table = _career_table(p1=0, p2=299, p3=300, p4=4000)
    scored = pd.DataFrame(
        {"player_id": [1, 2, 3, 4, 1], "season": 2025, "system": ["ours"] * 4 + ["fg_blend"]}
    )
    tagged = tag_experience(scored, table)
    # Preseason reads the week-0 row, not the week-5 one with 200 more PA.
    assert list(tagged["group"]) == ["rookie", "rookie", "vet", "vet", "rookie"]
    assert list(tagged["career_pa"]) == [0, 299, 300, 4000, 0]
    # Re-tagging replaces the old columns instead of duplicating them.
    again = tag_experience(tagged, table)
    assert list(again.columns) == list(tagged.columns)


def test_tag_experience_counts_pa_up_to_the_snapshot():
    from fantasy_baseball.hitter_ros.backtest import tag_experience

    table = _career_table(p1=150, p2=50)
    scored = pd.DataFrame(
        {
            "player_id": [1, 1, 2],
            "season": 2025,
            "system": "ours",
            # Before week 5: still the week-0 row. On or after it: 200 more PA.
            "snapshot": ["2025-04-28", "2025-04-29", "2025-06-01"],
        }
    )
    tagged = tag_experience(scored, table)
    assert list(tagged["career_pa"]) == [150, 350, 250]
    assert list(tagged["group"]) == ["rookie", "vet", "rookie"]


def test_tag_experience_calls_short_history_unknown_not_rookie():
    from fantasy_baseball.hitter_ros.backtest import MIN_HISTORY_SEASONS, tag_experience

    scored = pd.DataFrame({"player_id": [1, 2], "season": 2025, "system": "ours"})
    short = _career_table(seasons_in_store=MIN_HISTORY_SEASONS - 1, p1=100, p2=500)
    assert list(tag_experience(scored, short)["group"]) == ["unknown", "vet"]
    enough = _career_table(seasons_in_store=MIN_HISTORY_SEASONS, p1=100, p2=500)
    assert list(tag_experience(scored, enough)["group"]) == ["rookie", "vet"]


def test_tag_experience_refuses_a_player_without_a_table_row():
    from fantasy_baseball.hitter_ros.backtest import tag_experience

    scored = pd.DataFrame({"player_id": [1, 7], "season": 2025, "system": "ours"})
    with pytest.raises(ValueError, match="no table row"):
        tag_experience(scored, _career_table(p1=500))
    early = scored.assign(player_id=1, snapshot="2025-03-01")  # before any as-of date
    with pytest.raises(ValueError, match="no table row"):
        tag_experience(early, _career_table(p1=500))


def _grouped_units():
    """One season: vets 1-3 and rookies 4-6. Ours orders the vets right and the rookies
    backwards; the blend does the opposite."""
    actual = pd.DataFrame({s: [0.1, 0.2, 0.3] * 2 for s in TARGETS}, index=range(1, 7))
    actual["pa"] = 500
    right = actual[list(TARGETS)]
    flipped = right.copy()
    flipped.loc[[1, 2, 3]] = right.loc[[3, 2, 1]].to_numpy()
    flipped.loc[[4, 5, 6]] = right.loc[[6, 5, 4]].to_numpy()
    ours = pd.concat([right.loc[[1, 2, 3]], flipped.loc[[4, 5, 6]]])
    blend = pd.concat([flipped.loc[[1, 2, 3]], right.loc[[4, 5, 6]]])
    scored = scored_players({"ours": ours, "fg_blend": blend}, actual, 1)
    return scored.assign(
        season=2025, group=scored.player_id.map(lambda p: "vet" if p <= 3 else "rookie")
    )


def test_summary_scores_vets_and_rookies_separately():
    from fantasy_baseball.hitter_ros.backtest import summarize

    md = "\n".join(summarize(_grouped_units(), None))
    assert "**Vets vs rookies**" in md
    vets = md.split("Vets, 3 players per season -- gap-weighted pairwise (%):")[1]
    rookies = md.split("Rookies, 3 players per season -- gap-weighted pairwise (%):")[1]
    # Pairs only inside a group: ours is perfect on vets and backwards on rookies.
    assert "| ours | 100.0 |" in vets.split("raw MAE")[0]
    assert "| ours | 0.0 |" in rookies.split("raw MAE")[0]
    # Each group's MSE (the main score) comes with its own ours-minus-blend luck line.
    for group in ("Vets", "Rookies"):
        mse = md.split(f"{group}, 3 players per season -- MSE (main score):")[1]
        mse = mse.split("gap-weighted pairwise")[0]
        assert "ours - fg_blend (negative = ours better)" in mse
    # Unknown players are in neither group, and the summary says how many.
    unknown = _grouped_units().assign(group="unknown")
    assert "6 player-seasons are in neither group" in "\n".join(summarize(unknown, None))
    # A frame scored before the tag existed just has no group section.
    assert "Vets vs rookies" not in "\n".join(
        summarize(_grouped_units().drop(columns="group"), None)
    )
