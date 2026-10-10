"""The arm comparison: seed-averaged runs scored on actuals, with luck ranges."""

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.features import TARGETS
from scripts import compare_hitter_ros_arms as arms

WEEKS = (1, 8, 15, 22)


def _truth(seed=0, n=25):
    """Actual rates per (season, week or "pre"), the same in every run."""
    rng = np.random.default_rng(seed)
    return {
        (season, unit): rng.random((n, len(TARGETS)))
        for season in (2024, 2025)
        for unit in (*WEEKS, "pre")
    }


def _run(root, name, truth, noise, seed):
    """A finished run whose projections are the actuals plus ``noise``."""
    rng = np.random.default_rng(seed)
    hz, pre = [], []
    for (season, unit), act in truth.items():
        proj = act + rng.normal(0, noise, act.shape)
        for k, s in enumerate(TARGETS):
            rows = pd.DataFrame(
                {
                    "player_id": range(len(act)),
                    "stat": s,
                    "projected": proj[:, k],
                    "actual": act[:, k],
                    "season": season,
                }
            )
            if unit == "pre":
                pre.append(rows.assign(system="ours"))
            else:
                snapshot = f"{season}-w{unit:02d}"
                for system in ("head", "marcel"):  # only the head is ours
                    hz.append(
                        rows.assign(
                            system=system, horizon="ros", comparison="all", snapshot=snapshot
                        )
                    )
                hz.append(
                    rows.assign(system="head", horizon="n25", comparison="all", snapshot=snapshot)
                )
    run = root / name
    run.mkdir()
    pd.concat(hz).to_parquet(run / "scored_horizons.parquet")
    pd.concat(pre).to_parquet(run / "scored_preseason.parquet")
    (run / "summary.md").write_text("done\n")


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(arms, "RUNS", tmp_path)
    truth = _truth()
    for seed in (0, 1):
        _run(tmp_path, f"base-s{seed}", truth, 0.6, seed)
        _run(tmp_path, f"good-s{seed}", truth, 0.1, 10 + seed)
    _run(tmp_path, "single", truth, 0.6, 99)
    (tmp_path / "base-s9").mkdir()  # still training: no summary.md, so not a seed
    (tmp_path / "base-sx").mkdir()
    return tmp_path


def test_arm_runs_finds_finished_seeds(runs):
    assert list(arms.arm_runs("base")) == ["0", "1"]
    assert list(arms.arm_runs("single")) == [""]
    with pytest.raises(ValueError, match="no finished run"):
        arms.arm_runs("missing")


def test_bands():
    assert arms.band_of("pre") == "preseason"
    assert [arms.band_of(f"2025-w{w:02d}") for w in (1, 6, 7, 13, 14, 20, 21, 27)] == [
        "wk1-6", "wk1-6", "wk7-13", "wk7-13", "wk14-20", "wk14-20", "wk21+", "wk21+",
    ]  # fmt: skip


def test_run_scores_keep_only_the_rest_of_season_head(runs):
    s = arms.run_scores(runs / "base-s0")
    assert set(s["band"]) == set(arms.BANDS)
    assert len(s) == 2 * (len(WEEKS) + 1) * 25 * len(TARGETS)


def test_seed_mean_averages_the_projections(runs):
    scores = arms.arm_scores("base", None)
    mean = arms.seed_mean(scores)
    keyed = [s.set_index(arms.KEYS)["projected"] for s in scores.values()]
    expected = (keyed[0] + keyed[1]) / 2
    got = mean.set_index(arms.KEYS)["projected"]
    pd.testing.assert_series_equal(got.sort_index(), expected.sort_index(), check_names=False)
    broken = dict(scores)
    broken["1"] = broken["1"].iloc[1:]
    with pytest.raises(ValueError, match="different rows"):
        arms.seed_mean(broken)


def test_a_better_arm_wins_every_band_on_every_seed(runs):
    out = arms.compare_arm(arms.arm_scores("base", None), arms.arm_scores("good", None), 100)
    assert list(out.index) == list(arms.BANDS) and list(out.columns) == list(TARGETS)
    for cell in out.to_numpy().ravel():
        assert cell.startswith("+") and cell.endswith(" 100% sure better (real), 2/2 seeds")


def test_seasons_filter_and_mismatched_arms(runs):
    only = arms.arm_scores("base", [2025])
    assert set(only["0"]["season"]) == {2025}
    with pytest.raises(ValueError, match="different rows"):
        arms.compare_arm(only, arms.arm_scores("good", None), 10)


def test_an_unseeded_arm_compares_without_a_seed_count(runs):
    out = arms.compare_arm(arms.arm_scores("base", None), arms.arm_scores("single", None), 20)
    assert not out.loc["wk1-6", "r"].endswith("seeds")
