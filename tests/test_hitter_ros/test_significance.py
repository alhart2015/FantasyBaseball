"""Luck ranges and P(better) on every comparison: resampling hitters, one draw shared
across every season or snapshot (so overlapping snapshots aren't independent)."""

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.evaluate import (
    order_table,
    paired_bootstrap,
    pairwise_accuracy,
    pairwise_bootstrap,
    scored_players,
    sure,
)
from fantasy_baseball.hitter_ros.features import TARGETS


def test_sure_reads_as_is():
    # 97.5% of draws on one side = the 95% interval just clears 0 = 95% sure.
    draws = np.r_[np.ones(975), -np.ones(25)]
    assert sure(draws) == pytest.approx(0.95)
    assert sure(-draws) == pytest.approx(-0.95)  # sure it's worse
    assert sure(np.array([1.0, -1.0])) == 0.0  # no lean
    assert sure(np.array([1.0, -1.0, 0.0, 2.0])) == pytest.approx(0.25)  # ties count half
    assert np.isnan(sure(np.array([])))


def test_sure_text():
    from fantasy_baseball.hitter_ros.backtest import sure_text

    assert sure_text(0.96) == "96% sure better (real)"
    assert sure_text(0.95) == "95% sure better (real)"
    assert sure_text(-0.9) == "90% sure worse"
    assert sure_text(0.0) == "no lean"
    assert sure_text(float("nan")) == "n/a"


def _snapshots(n_units=3, n=30, noise_a=0.2, noise_b=0.6, seed=0):
    """Overlapping snapshots: the same players (plus one extra each) in every one."""
    rng = np.random.default_rng(seed)
    parts, truth = [], {}
    for u in range(n_units):
        ids = [*range(n), 100 + u]
        actual = pd.DataFrame({s: rng.random(len(ids)) for s in TARGETS}, index=ids)
        actual["pa"] = 500
        a = actual[list(TARGETS)] + rng.normal(0, noise_a, (len(ids), len(TARGETS)))
        b = actual[list(TARGETS)] + rng.normal(0, noise_b, (len(ids), len(TARGETS)))
        snapshot = f"2026-0{u + 6}-01"
        parts.append(
            scored_players({"a": a, "b": b}, actual, 1).assign(season=2026, snapshot=snapshot)
        )
        truth[snapshot] = (actual, a, b)
    return pd.concat(parts, ignore_index=True), truth


def test_pooled_bootstrap_scores_the_mean_over_snapshots():
    scored, _ = _snapshots()
    table = order_table(scored, "pairwise_w")
    boot = pairwise_bootstrap(scored, "a", "b", n_boot=50)
    for s in TARGETS:
        assert boot.loc[s, "diff"] == pytest.approx(table.loc["a", s] - table.loc["b", s])
    assert (boot["sure"] > 0.9).all()  # the less noisy one


def test_pooled_bootstrap_redraws_each_player_once_for_every_snapshot():
    """The same as drawing players one draw at a time, keeping each drawn player's rows
    in every snapshot he is in, and averaging the snapshots' scores."""
    scored, truth = _snapshots(n=12)
    fast = pairwise_bootstrap(scored, "a", "b", n_boot=40, seed=5)
    rng = np.random.default_rng(5)
    for s in TARGETS:
        players = np.array(sorted(scored.loc[scored["stat"] == s, "player_id"].unique()))
        slow = []
        for _ in range(40):
            drawn = players[rng.integers(0, len(players), len(players))]
            per_unit = []
            for actual, a, b in truth.values():
                ids = [p for p in drawn if p in actual.index]
                act = actual.loc[ids, s].to_numpy()
                per_unit.append(
                    100
                    * (
                        pairwise_accuracy(a.loc[ids, s].to_numpy(), act, weighted=True)
                        - pairwise_accuracy(b.loc[ids, s].to_numpy(), act, weighted=True)
                    )
                )
            slow.append(np.mean(per_unit))
        assert fast.loc[s, "lo"] == pytest.approx(np.percentile(slow, 2.5))
        assert fast.loc[s, "hi"] == pytest.approx(np.percentile(slow, 97.5))
        assert fast.loc[s, "sure"] == pytest.approx(sure(np.array(slow)))


def test_shared_draws_are_wider_than_treating_snapshots_as_independent():
    """Ten copies of one snapshot hold no more evidence than one: the interval mustn't
    shrink as if they were ten independent samples."""
    one, _ = _snapshots(n_units=1, n=40, noise_a=0.5, noise_b=0.55, seed=3)
    copies = pd.concat([one.assign(snapshot=f"2026-06-{d:02d}") for d in range(1, 11)])
    width_one = pairwise_bootstrap(one, "a", "b", n_boot=200).eval("hi - lo")
    width_ten = pairwise_bootstrap(copies, "a", "b", n_boot=200).eval("hi - lo")
    np.testing.assert_allclose(width_ten, width_one)


def test_paired_bootstrap_sure_is_for_lower_error():
    n = 300
    rng = np.random.default_rng(1)
    actual = pd.DataFrame({s: [0.1] * n for s in TARGETS})
    actual["pa"] = 500
    a = pd.DataFrame({s: 0.1 + rng.uniform(0, 2, n) / 600 for s in TARGETS})
    b = pd.DataFrame({s: 0.1 + rng.uniform(3, 5, n) / 600 for s in TARGETS})
    r = paired_bootstrap(scored_players({"a": a, "b": b}, actual, 1), "a", "b", n_boot=200)
    assert (r["diff"] < 0).all() and (r["sure"] == 1.0).all()


def test_summary_shows_how_sure_for_one_snapshot_and_the_mean():
    from fantasy_baseball.hitter_ros.backtest import summarize

    scored, _ = _snapshots()
    scored["system"] = scored["system"].map({"a": "ours", "b": "fg_blend"})
    md = "\n".join(summarize(None, scored))
    first, mean = md.split("**Mean over snapshots**")
    assert "how sure" in first and "100% sure better (real)" in first
    luck = [line for line in mean.splitlines() if line.startswith("ours - fg_blend")]
    assert len(luck) == 1 and "100% sure better (real)" in luck[0]  # main score only


def test_summary_mean_has_no_luck_line_without_the_blend():
    from fantasy_baseball.hitter_ros.backtest import summarize

    scored, _ = _snapshots()
    scored["system"] = scored["system"].map({"a": "ours", "b": "marcel"})
    md = "\n".join(summarize(None, scored))
    assert "ours - fg_blend" not in md
