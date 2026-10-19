import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.pt_model.ranges import (
    FEATURES,
    GRID,
    MIN_RATE_PA,
    STATS,
    BandPT,
    LuckPool,
    QuantilePT,
    backtest,
    fit,
    history,
    pa_by_rank,
    pinball,
    simulate,
    top_ranked,
    with_rank_pa,
)


def test_top_ranked_takes_the_most_projected_pa_ties_to_the_lower_id():
    pa = pd.Series({7: 600.0, 3: 650.0, 5: 600.0, 9: np.nan, 1: 100.0})
    rank = top_ranked(pa, top_n=3)
    assert rank.to_dict() == {3: 1, 5: 2, 7: 3}


def test_pa_by_rank_averages_each_seasons_sorted_pa_padded_with_zero():
    by_rank = pa_by_rank(
        {2024: np.array([500.0, 700.0]), 2025: np.array([600.0, 650.0, 100.0])}, [2024, 2025], 3
    )
    np.testing.assert_allclose(by_rank, [(700 + 650) / 2, (500 + 600) / 2, (0 + 100) / 2])
    with pytest.raises(ValueError, match="no seasons"):
        pa_by_rank({}, [], 3)


def test_with_rank_pa_maps_ranks_and_the_ratio_to_actual():
    rows = pd.DataFrame({"proj_rank": [1, 2], "a_pa": [350.0, 600.0]})
    out = with_rank_pa(rows, np.array([700.0, 600.0]))
    assert out["rank_pa"].tolist() == [700.0, 600.0] and out["ratio"].tolist() == [0.5, 1.0]
    assert "ratio" not in with_rank_pa(rows.drop(columns="a_pa"), np.array([700.0, 600.0]))
    with pytest.raises(ValueError, match="proj_rank must be"):
        with_rank_pa(rows, np.array([700.0]))
    with pytest.raises(ValueError, match="0 PA"):
        with_rank_pa(rows, np.array([700.0, 0.0]))


def test_history_is_zero_after_his_debut_and_unknown_before_it():
    lines = {y: pd.DataFrame({"g": [0.9], "pa": [4.0]}, index=[10]) for y in (2023, 2024, 2025)}
    lines[2024] = pd.DataFrame({"g": [0.5], "pa": [2.0]}, index=[10])
    ids = pd.Index([10, 20, 30])
    # 20 debuted in 2024 and missed 2025; 30 never debuted.
    out = history(ids, 2026, lines, pd.Series({10: 2015, 20: 2024}))
    assert out.loc[10, ["g1", "g2", "g3"]].tolist() == [0.9, 0.5, 0.9]
    assert out.loc[20, "g1"] == 0.0 and out.loc[20, "g2"] == 0.0 and np.isnan(out.loc[20, "g3"])
    assert out.loc[30].isna().all()
    with pytest.raises(ValueError, match="no season lines for 2022"):
        history(ids, 2025, lines, pd.Series(dtype=float))


def _rows(rng, n, season, frail_share=0.5):
    """Synthetic hitters: the frail ones (low g1) have a much wider, lower PT range."""
    frail = rng.uniform(size=n) < frail_share
    g = np.where(frail, rng.uniform(0.2, 0.5, n), rng.uniform(0.85, 1.0, n))
    ratio = np.where(frail, rng.uniform(0.1, 1.1, n), rng.uniform(0.85, 1.15, n))
    rank = rng.permutation(n) + 1
    rows = pd.DataFrame(
        {
            "season": season,
            "proj_rank": rank,
            "ab_pa": 0.9,
            "age": rng.uniform(22, 36, n),
            "catcher": 0.0,
            "dh": 0.0,
            **{f"{c}{k}": g if c == "g" else 4 * g for k in (1, 2, 3) for c in ("g", "pa")},
            "p_r": 0.12,
            "p_hr": 0.04,
            "p_rbi": 0.12,
            "p_sb": 0.02,
            "p_avg": 0.25,
        },
        index=pd.Index(np.arange(n) + season * 1000, name="player_id"),
    )
    by_rank = np.linspace(700, 200, n)
    pa = np.round(ratio * by_rank[rank - 1])
    ab = np.round(0.9 * pa)
    rows["a_pa"], rows["a_ab"] = pa, ab
    rows["a_h"] = rng.binomial(ab.astype(int), 0.25)
    for c, p in (("r", 0.12), ("hr", 0.04), ("rbi", 0.12), ("sb", 0.02)):
        rows[f"a_{c}"] = rng.binomial(pa.astype(int), p)
    return rows, pa


def test_quantile_pt_gives_the_frail_a_lower_floor_and_ordered_quantiles():
    rng = np.random.default_rng(0)
    rows, _ = _rows(rng, 600, 2025)
    rows = rows.assign(rank_pa=np.linspace(700, 200, 600)[rows["proj_rank"] - 1])
    rows["ratio"] = rows["a_pa"] / rows["rank_pa"]
    q = QuantilePT().fit(rows).predict(rows)
    assert q.shape == (600, len(GRID)) and (np.diff(q, axis=1) >= 0).all() and (q >= 0).all()
    frail = rows["g1"].to_numpy() < 0.6
    p10 = q[:, list(GRID).index(0.1)]
    assert p10[frail].mean() < 0.35 < 0.8 < p10[~frail].mean()
    # A blank input (no history) still predicts, with its flag.
    blank = rows.head(3).assign(g1=np.nan, pa1=np.nan)
    assert np.isfinite(QuantilePT().fit(rows).predict(blank)).all()


def test_band_pt_is_the_same_range_for_everyone_in_a_band():
    rows = pd.DataFrame({"rank_pa": [600.0, 620.0, 200.0, 210.0], "ratio": [1.0, 0.5, 0.2, 0.4]})
    q = BandPT().fit(rows).predict(rows)
    np.testing.assert_allclose(q[0], q[1])
    np.testing.assert_allclose(q[2], q[3])
    assert not np.allclose(q[0], q[2])


def _past(n_short):
    """Past seasons for a luck pool: 4 full seasons, then n_short under MIN_RATE_PA."""
    pa = np.array([600.0, 600.0, 600.0, 600.0, *([10.0] * n_short)])
    rows = pd.DataFrame(
        {
            "ratio": pa / 600,
            "a_pa": pa,
            "a_ab": 0.9 * pa,
            "a_h": 0.25 * 0.9 * pa,
            **{f"a_{c}": 0.1 * pa for c in ("r", "hr", "rbi", "sb")},
            **{f"p_{c}": 0.1 for c in ("r", "hr", "rbi", "sb")},
            "p_avg": 0.25,
        }
    )
    q = np.tile(np.linspace(0.0, 1.2, len(GRID)), (len(rows), 1))
    return rows, q


def test_luck_pool_keeps_every_seasons_pt_and_lends_short_ones_a_rate_miss():
    rows, q = _past(n_short=3)
    rows.loc[0, "a_hr"] = 66.0  # one full season that beat its projection
    pool = LuckPool.from_rows(rows, q, np.random.default_rng(0))
    assert len(pool.u) == len(rows)  # the short seasons' PT stays in the pool
    assert pool.u[4:].max() < pool.u[:4].min()
    full = pool.z["hr"][:4]
    assert set(np.round(pool.z["hr"][4:], 9)) <= set(np.round(full, 9))  # borrowed, not their own
    with pytest.raises(ValueError, match=f"{MIN_RATE_PA}"):
        LuckPool.from_rows(rows.iloc[4:], q[4:], np.random.default_rng(0))


def test_simulate_with_no_luck_is_rate_times_his_pt_quantile():
    pool = LuckPool(u=np.full(10, 0.5), z={c: np.zeros(10) for c in STATS})
    rows = pd.DataFrame({"rank_pa": [600.0], "ab_pa": [0.9], **{f"p_{c}": [0.1] for c in STATS}})
    q = np.tile(np.linspace(0.5, 1.5, len(GRID)), (1, 1))
    median_ratio = np.interp(0.5, GRID, q[0])
    out = simulate(rows, q, pool, (0.1, 0.9), np.random.default_rng(0), n_draws=50)
    np.testing.assert_allclose(out["pa"][0], 600 * median_ratio)
    np.testing.assert_allclose(out["hr"][0], 0.1 * 600 * median_ratio)
    np.testing.assert_allclose(out["avg"][0], 0.1)
    # Next to no at-bats: AVG is his projected AVG, not a draw over a handful of AB.
    tiny = simulate(
        rows.assign(rank_pa=20.0), q, pool, (0.1, 0.9), np.random.default_rng(0), n_draws=50
    )
    np.testing.assert_allclose(tiny["avg"][0], 0.1)


def test_backtest_covers_about_ten_percent_each_side_on_a_known_generator():
    rng = np.random.default_rng(1)
    parts, pa = [], {}
    for season in (2023, 2024, 2025, 2026):
        rows, season_pa = _rows(rng, 250, season)
        parts.append(rows)
        pa[season] = season_pa
    rows = pd.concat(parts)
    scored = backtest(rows, pa, QuantilePT, np.random.default_rng(2), (0.1, 0.9), n_draws=300)
    assert set(scored["stat"]) == {*STATS, "pa"} and {"q10", "q90", "actual", "rank_pa"} <= set(
        scored
    )
    s = scored[scored["stat"].isin(["pa", "hr", "r"])]
    below, above = (s["actual"] < s["q10"]).mean(), (s["actual"] > s["q90"]).mean()
    assert 0.05 < below < 0.16 and 0.05 < above < 0.16, (below, above)


def test_no_draw_plays_more_than_the_cap():
    pool = LuckPool(u=np.full(10, 0.98), z={c: np.zeros(10) for c in STATS})
    rows = pd.DataFrame({"rank_pa": [600.0], "ab_pa": [0.9], **{f"p_{c}": [0.1] for c in STATS}})
    q = np.tile(np.linspace(0.5, 1.5, len(GRID)), (1, 1))  # his top draws: 900 PA
    out = simulate(rows, q, pool, (0.5,), np.random.default_rng(0), n_draws=20, pa_cap=750.0)
    assert out["pa"][0, 0] == 750.0 and out["hr"][0, 0] == pytest.approx(75.0)


def test_fit_caps_pa_at_the_most_any_training_hitter_had():
    rng = np.random.default_rng(3)
    parts, pa = [], {}
    for season in (2024, 2025):
        rows, season_pa = _rows(rng, 120, season)
        parts.append(rows)
        pa[season] = season_pa
    ranges = fit(pd.concat(parts), pa, QuantilePT, rng)
    assert ranges.pa_cap == max(pa[2024].max(), pa[2025].max())


def test_fit_needs_two_seasons_and_the_rows_it_reads():
    rng = np.random.default_rng(0)
    rows, pa = _rows(rng, 50, 2025)
    with pytest.raises(ValueError, match="2\\+ seasons"):
        fit(rows, {2025: pa}, QuantilePT, rng)
    with pytest.raises(ValueError, match="lack"):
        fit(rows.drop(columns="g1"), {2025: pa}, QuantilePT, rng)
    with pytest.raises(ValueError, match="projected rates"):
        fit(rows.assign(p_hr=np.nan), {2025: pa}, QuantilePT, rng)


def test_pinball_charges_misses_by_their_side():
    np.testing.assert_allclose(
        pinball(np.array([10.0, 0.0]), np.array([5.0, 5.0]), 0.1), [0.5, 4.5]
    )


def test_every_feature_is_built_by_history_or_the_rows():
    assert {f"{c}{k}" for k in (1, 2, 3) for c in ("g", "pa")} <= set(FEATURES)
