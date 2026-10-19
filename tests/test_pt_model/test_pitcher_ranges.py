import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.pt_model import range_inputs as ri
from fantasy_baseball.pt_model.range_specs import HITTER, RELIEVER, STARTER
from fantasy_baseball.pt_model.ranges import (
    GRID,
    BandPT,
    LuckPool,
    QuantilePT,
    actual_totals,
    backtest,
    drawn_rate,
    pt_model,
    rate_miss,
    simulate,
)

ERA = STARTER.stats[STARTER.stat_names.index("era")]
K = STARTER.stats[STARTER.stat_names.index("k")]


def test_pitcher_specs_count_k_w_sv_per_batter_and_era_whip_per_inning():
    assert STARTER.stat_names == ("k", "w", "sv", "era", "whip")
    assert {s.trials for s in STARTER.stats} == {"bf", "ip"}
    assert STARTER.actuals == ("a_ip", "a_bf", "a_k", "a_w", "a_sv", "a_er", "a_h", "a_bb")
    assert STARTER.rank_col == "rank_ip" and RELIEVER.top_n > STARTER.top_n


def test_each_role_uses_the_pt_range_its_spec_says():
    assert isinstance(pt_model(STARTER), BandPT)
    assert isinstance(pt_model(RELIEVER), QuantilePT) and isinstance(pt_model(HITTER), QuantilePT)


def test_a_ratio_stats_miss_shrinks_with_innings():
    # ERA 3.00 projected, 4.00 over 100 IP: a miss of 10 in sqrt(IP) units.
    miss = rate_miss(ERA, np.array([4.0]), np.array([3.0]), np.array([100.0]))
    assert miss[0] == pytest.approx(10.0)
    # The same luck over 25 IP moves ERA twice as far; never below 0.
    assert drawn_rate(ERA, 3.0, miss, np.array([25.0]))[0] == pytest.approx(5.0)
    assert drawn_rate(ERA, 3.0, np.array([-100.0]), np.array([25.0]))[0] == 0.0
    # A binomial stat's miss is in coin flips.
    k = rate_miss(K, np.array([0.30]), np.array([0.25]), np.array([700.0]))
    assert k[0] == pytest.approx(0.05 / np.sqrt(0.25 * 0.75 / 700))


def test_actual_era_and_whip_come_from_er_hits_and_walks():
    rows = pd.DataFrame(
        {
            "a_ip": [90.0, 0.0],
            "a_bf": [380.0, 0.0],
            "a_k": [100.0, 0.0],
            "a_w": [6.0, 0.0],
            "a_sv": [0.0, 0.0],
            "a_er": [30.0, 0.0],
            "a_h": [80.0, 0.0],
            "a_bb": [25.0, 0.0],
        }
    )
    out = actual_totals(rows, STARTER)
    assert out["era"][0] == pytest.approx(3.0) and out["whip"][0] == pytest.approx(105 / 90)
    assert out["k"][0] == 100.0 and np.isnan(out["era"][1])  # no innings: no ERA


def test_simulate_pitchers_k_over_batters_faced_and_era_needs_innings():
    pool = LuckPool(u=np.full(10, 0.5), z={c: np.zeros(10) for c in STARTER.stat_names})
    rows = pd.DataFrame(
        {
            "rank_ip": [180.0],
            "bf_ip": [4.2],
            "p_k": [0.25],
            "p_w": [0.05],
            "p_sv": [0.0],
            "p_era": [3.5],
            "p_whip": [1.2],
        }
    )
    q = np.tile(np.linspace(0.5, 1.5, len(GRID)), (1, 1))
    ip = 180.0 * np.interp(0.5, GRID, q[0])
    out = simulate(rows, q, pool, (0.5,), np.random.default_rng(0), STARTER, n_draws=20)
    assert out["ip"][0, 0] == pytest.approx(ip)
    assert out["k"][0, 0] == pytest.approx(0.25 * 4.2 * ip)
    assert out["era"][0, 0] == pytest.approx(3.5)
    # Next to no innings: ERA is his projected ERA, not a draw over 2 IP.
    tiny = simulate(
        rows.assign(rank_ip=3.0), q, pool, (0.5,), np.random.default_rng(0), STARTER, 20
    )
    assert tiny["era"][0, 0] == pytest.approx(3.5)


def _leaderboard(rows):
    """A raw MLB pitching leaderboard: (id, G, GS, IP in baseball notation, BF)."""
    return pd.DataFrame(
        [
            {
                "player.id": pid,
                "stat.gamesPitched": g,
                "stat.gamesStarted": gs,
                "stat.inningsPitched": ip,
                "stat.battersFaced": bf,
                "stat.strikeOuts": 0.25 * bf,
                "stat.wins": 5,
                "stat.saves": 0 if gs else 20,
                "stat.earnedRuns": 30,
                "stat.hits": 70,
                "stat.baseOnBalls": 25,
            }
            for pid, g, gs, ip, bf in rows
        ]
    )


def test_pitching_frame_reads_innings_in_thirds():
    frame = ri.pitching_frame(_leaderboard([(1, 30, 30, "150.2", 640), (2, 60, 0, "58.1", 240)]))
    assert frame.loc[1, "ip"] == pytest.approx(150 + 2 / 3) and frame.loc[2, "ip"] == pytest.approx(
        58 + 1 / 3
    )
    assert frame.loc[2, "sv"] == 20.0 and frame.loc[1, "gs"] == 30.0


def _setup():
    seasons = {
        y: ri.pitching_frame(
            _leaderboard(
                [(1, 30, 30, "170.0", 700), (2, 60, 0, "60.0", 250), (3, 25, 20, "110.0", 470)]
            )
        )
        for y in (2023, 2024, 2025, 2026)
    }
    info = ri.player_info(
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "birthDate": ["1995-06-30"] * 4,
                "mlbDebutDate": ["2018-04-01", "2019-04-01", "2020-04-01", "2026-04-01"],
            }
        )
    )
    pt = pd.DataFrame(
        {
            "ip": [180.0, 65.0, 120.0, 40.0],
            "bf": [740.0, 270.0, 510.0, 175.0],
            "gs_share": [1.0, 0.0, 0.8, np.nan],
        },
        index=[1, 2, 3, 4],
    )
    rates = pd.DataFrame(
        {f"p_{c}": [0.1] * 4 for c in STARTER.stat_names},
        index=pd.MultiIndex.from_tuples(
            [(i, 2026) for i in (1, 2, 3, 4)], names=["player_id", "season"]
        ),
    )
    return seasons, info, pt, rates


def test_build_pitcher_rows_splits_the_roles_by_projected_starts():
    seasons, info, pt, rates = _setup()
    sp, _ = ri.build_pitcher_rows(2026, STARTER, pt, rates, seasons, info, with_actuals=True)
    rp, _ = ri.build_pitcher_rows(2026, RELIEVER, pt, rates, seasons, info, with_actuals=True)
    assert sp.index.tolist() == [1, 3] and rp.index.tolist() == [2]  # 4: no role, left out
    assert sp["proj_rank"].tolist() == [1, 2] and rp.loc[2, "proj_rank"] == 1
    assert sp.loc[1, "bf_ip"] == pytest.approx(740 / 180)
    assert sp.loc[1, "ip1"] == pytest.approx(170 / 162) and sp.loc[3, "gs1"] == pytest.approx(
        20 / 162
    )
    assert rp.loc[2, "a_sv"] == 20.0 and sp.loc[1, "a_ip"] == 170.0


def test_role_ip_and_last_seasons_line_read_the_actual_starts():
    seasons, *_ = _setup()
    assert sorted(ri.actual_ip(seasons, [2025], STARTER)[2025]) == [110.0, 170.0]
    assert ri.actual_ip(seasons, [2025], RELIEVER)[2025].tolist() == [60.0]
    last = ri.last_season_pitcher_pt(seasons, 2026)
    assert last.loc[3, "gs_share"] == pytest.approx(0.8) and last.loc[2, "ip"] == 60.0


def test_load_pitcher_pt_adds_the_start_share(tmp_path):
    path = tmp_path / "steamer-pitchers.csv"
    pd.DataFrame(
        {
            "Name": ["A", "B"],
            "MLBAMID": [1, 2],
            "IP": [180.0, 65.0],
            "TBF": [740, 270],
            "G": [30, 60],
            "GS": [30, 0],
            "SO": [180, 70],
            "W": [12, 4],
            "SV": [0, 25],
            "ERA": [3.5, 3.0],
            "WHIP": [1.2, 1.1],
        }
    ).to_csv(path, index=False)
    pt = ri.load_pitcher_pt(path)
    assert (
        pt.loc[1, "gs_share"] == 1.0 and pt.loc[2, "gs_share"] == 0.0 and pt.loc[2, "bf"] == 270.0
    )


def _synthetic(rng, n, season):
    """Starters with ERA luck shrinking as 1/sqrt(IP) and K per batter faced."""
    ratio = rng.uniform(0.3, 1.1, n)
    rank = rng.permutation(n) + 1
    by_rank = np.linspace(200, 80, n)
    ip = np.round(ratio * by_rank[rank - 1])
    bf = np.round(4.2 * ip)
    era = np.clip(3.8 + rng.normal(0, 9.0, n) / np.sqrt(np.maximum(ip, 1)), 0, None)
    rows = pd.DataFrame(
        {
            "season": season,
            "proj_rank": rank,
            "bf_ip": 4.2,
            "age": rng.uniform(23, 35, n),
            **{f"{c}{k}": rng.uniform(0.5, 1.2, n) for k in (1, 2, 3) for c in ("g", "ip", "gs")},
            "p_k": 0.24,
            "p_w": 0.05,
            "p_sv": 0.001,
            "p_era": 3.8,
            "p_whip": 1.25,
            "a_ip": ip,
            "a_bf": bf,
            "a_k": rng.binomial(bf.astype(int), 0.24),
            "a_w": rng.binomial(bf.astype(int), 0.05),
            "a_sv": 0.0,
            "a_er": era * ip / 9,
            "a_h": 0.95 * ip,
            "a_bb": 0.3 * ip,
        },
        index=pd.Index(np.arange(n) + season * 1000, name="player_id"),
    )
    return rows, ip


def test_pitcher_backtest_covers_about_ten_percent_each_side():
    rng = np.random.default_rng(5)
    parts, ip = [], {}
    for season in (2023, 2024, 2025, 2026):
        rows, season_ip = _synthetic(rng, 150, season)
        parts.append(rows)
        ip[season] = season_ip
    scored = backtest(
        pd.concat(parts), ip, pt_model, np.random.default_rng(6), STARTER, (0.1, 0.9), 300
    )
    for stat in ("ip", "k", "era"):
        s = scored[(scored["stat"] == stat) & (scored["trials"] >= 20)]
        below, above = (s["actual"] < s["q10"]).mean(), (s["actual"] > s["q90"]).mean()
        assert 0.04 < below < 0.17 and 0.04 < above < 0.17, (stat, below, above)


def test_reliever_luck_is_drawn_from_his_saves_band():
    # Pool: half setup men (tiny projected SV, a big lucky saves miss), half closers.
    n = 40
    by = np.r_[np.full(n // 2, 0.002), np.full(n // 2, 0.12)]
    z = {c: np.zeros(n) for c in RELIEVER.stat_names}
    z["sv"] = np.r_[np.full(n // 2, 13.5), np.zeros(n // 2)]
    pool = LuckPool(u=np.full(n, 0.5), z=z, by=by)
    band, members = pool.bands(RELIEVER, np.array([0.12, 0.002]))
    assert set(members[band[0]]) <= set(range(n // 2, n))  # a closer draws closers only
    rows = pd.DataFrame(
        {
            "rank_ip": [65.0],
            "bf_ip": [4.1],
            "p_k": [0.3],
            "p_w": [0.04],
            "p_sv": [0.12],
            "p_era": [3.0],
            "p_whip": [1.1],
        }
    )
    q = np.tile(np.linspace(0.5, 1.5, len(GRID)), (1, 1))
    out = simulate(rows, q, pool, (0.9,), np.random.default_rng(0), RELIEVER, n_draws=200)
    assert out["sv"][0, 0] < 40  # no 100-save seasons from setup men's luck


def test_the_cap_covers_a_reliever_who_moved_to_the_rotation():
    rng = np.random.default_rng(7)
    parts, ip = [], {}
    for season in (2025, 2026):
        rows, season_ip = _synthetic(rng, 60, season)
        parts.append(rows)
        ip[season] = season_ip * 0.5  # the role's own IP, all lower than these rows
    from fantasy_baseball.pt_model.ranges import fit

    ranges = fit(pd.concat(parts), ip, pt_model, rng, STARTER)
    assert ranges.cap == max(r["a_ip"].max() for r in parts)


def test_rows_need_a_usable_per_volume_rate_and_a_pitcher_role():
    seasons, info, pt, rates = _setup()
    rows, _ = ri.build_pitcher_rows(2026, STARTER, pt, rates, seasons, info, with_actuals=True)
    from fantasy_baseball.pt_model.ranges import check_rows

    with pytest.raises(ValueError, match="bf_ip"):
        check_rows(rows.assign(bf_ip=np.inf), STARTER, actuals=True)
    with pytest.raises(ValueError, match="not a pitcher role"):
        ri.build_pitcher_rows(2026, HITTER, pt, rates, seasons, info, with_actuals=True)
    # A 0-IP projection isn't ranked (it has no BF per inning).
    rows0, _ = ri.build_pitcher_rows(
        2026,
        STARTER,
        pt.assign(ip=[0.0, 65.0, 120.0, 40.0]),
        rates,
        seasons,
        info,
        with_actuals=True,
    )
    assert rows0.index.tolist() == [3]


def test_no_counting_total_beats_the_best_training_season():
    pool = LuckPool(u=np.full(10, 0.5), z={c: np.full(10, 5.0) for c in RELIEVER.stat_names})
    rows = pd.DataFrame(
        {
            "rank_ip": [70.0],
            "bf_ip": [4.1],
            "p_k": [0.3],
            "p_w": [0.04],
            "p_sv": [0.15],
            "p_era": [3.0],
            "p_whip": [1.1],
        }
    )
    q = np.tile(np.linspace(0.5, 1.5, len(GRID)), (1, 1))
    out = simulate(
        rows, q, pool, (0.9,), np.random.default_rng(0), RELIEVER, 50, stat_caps={"sv": 49.0}
    )
    assert out["sv"][0, 0] == 49.0 and out["k"][0, 0] > 49.0  # only the capped stat
