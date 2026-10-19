from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.parks import PARK_FEATURES
from fantasy_baseball.pitcher_ros import backtest
from fantasy_baseball.pitcher_ros.features import (
    ERA_PIECES,
    answers,
    input_frame,
    league_answer_rates,
    league_reference,
)
from fantasy_baseball.pitcher_ros.model import (
    APPLIED,
    DEFAULT_ERA_PIECES,
    DEFAULT_HORIZONS,
    DEFAULT_RELATIVE_TARGET,
    ID_COLS,
    LeagueRates,
    applied_settings,
    average_seeds,
    check_plain,
    check_week_zero,
    fit_season,
    league_rates,
    net_config,
    plain_config,
    with_age,
)
from fantasy_baseball.pitcher_ros.stats import TARGETS
from fantasy_baseball.pitcher_ros.table import build_table

from .test_table import RELIEVER, STARTER, _season, _seasons, _write


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    _write(root, _seasons())
    return build_table(root)


def _starter_week0(table):
    return table[(table.player_id == STARTER) & (table.season == 2025) & (table.week == 0)].index[0]


def test_answers_and_their_weights(table):
    y, w = answers(table)
    row = _starter_week0(table)
    # 21 starts: 5 K and 24 BF, 18 outs, 1 ER, 5 H, 2 BB each.
    assert y.loc[row, "k"] == pytest.approx(5 / 24)
    assert y.loc[row, "era"] == pytest.approx(27 * 1 / 18)
    assert y.loc[row, "whip"] == pytest.approx(3 * 7 / 18)
    assert w.loc[row, "k"] == 21 * 24 and w.loc[row, "era"] == pytest.approx(21 * 6)
    # The scorer charges (bf x err)^2 on K/W/SV: "bf2" weights each row by bf^2.
    _, w2 = answers(table, "bf2")
    assert w2.loc[row, "k"] == (21 * 24) ** 2 and w2.loc[row, "era"] == w.loc[row, "era"]
    with pytest.raises(ValueError, match="per_bf_weight"):
        answers(table, "pa")


def test_era_pieces_are_extra_answers_weighted_as_k(table):
    y, w = answers(table, era_pieces=True)
    row = _starter_week0(table)
    assert list(y.columns) == [*TARGETS, *ERA_PIECES]
    # 21 starts of 24 BF, 2 BB, no HBP or HR each.
    assert y.loc[row, "bb_bf"] == pytest.approx(2 / 24)
    assert y.loc[row, "hbp_bf"] == 0 and y.loc[row, "hr_bf"] == 0
    assert (w.loc[row, list(ERA_PIECES)] == w.loc[row, "k"]).all()
    plain_y, plain_w = answers(table)
    assert list(plain_y.columns) == list(TARGETS)
    pd.testing.assert_frame_equal(y[list(TARGETS)], plain_y)
    pd.testing.assert_frame_equal(w[list(TARGETS)], plain_w)


def test_fit_season_predicts_era_pieces_beside_the_targets(table):
    x, (y, w) = input_frame(table), answers(table, era_pieces=True)
    config = plain_config(hidden=[8], max_epochs=3, patience=1, val_frac=0.34)
    preds, _ = fit_season(table, x, y, w, 2025, config)
    assert list(preds.columns) == [*ID_COLS, *TARGETS, *ERA_PIECES]
    assert (preds[list(ERA_PIECES)] >= 0).all().all()
    # Relative to the league: the fixture's league walks people but never hits one or
    # gives up a home run, so only BB has a league rate to learn against.
    with_bb = [*TARGETS, "bb_bf"]
    relative, _ = fit_season(
        table, x, y[with_bb], w[with_bb], 2025, replace(config, relative_target="answer")
    )
    assert relative["bb_bf"].notna().all()
    with pytest.raises(ValueError, match="hbp_bf"):
        fit_season(table, x, y, w, 2025, replace(config, relative_target="answer"))
    with pytest.raises(ValueError, match="TARGETS then"):
        fit_season(table, x, y[["era", *TARGETS[1:]]].join(y["k"]), w, 2025, config)


def test_short_horizons_are_the_targets_over_the_next_days(table):
    from fantasy_baseball.pitcher_ros.features import column_horizon, target_stat

    y, w = answers(table, era_pieces=True, horizons=(7, 30))
    short = [f"d{n}_{s}" for n in (7, 30) for s in TARGETS]
    assert list(y.columns) == [*TARGETS, *ERA_PIECES, *short]
    row = _starter_week0(table)
    # Daily starts of 24 BF: 7 in the next 7 days; the 21-day season ends inside 30.
    assert w.loc[row, "d7_k"] == 7 * 24 and w.loc[row, "d30_k"] == 21 * 24
    assert w.loc[row, "d7_era"] == pytest.approx(7 * 6)
    assert y.loc[row, "d7_k"] == pytest.approx(5 / 24)
    plain_y, plain_w = answers(table, era_pieces=True)
    pd.testing.assert_frame_equal(y[list(plain_y.columns)], plain_y)
    pd.testing.assert_frame_equal(w[list(plain_w.columns)], plain_w)
    assert [column_horizon(c) for c in ("d7_k", "d30_whip", "k", "bb_bf")] == [
        "d7",
        "d30",
        "ros",
        "ros",
    ]
    assert target_stat("d30_whip") == "whip" and target_stat("hr_bf") == "hr_bf"
    with pytest.raises(ValueError, match="horizons"):
        answers(table, horizons=(14,))


def test_short_horizon_league_rates_are_the_leagues_over_those_days(table):
    league = league_rates(table, (7,))
    pooled = table.groupby(["season", "week"])[["ros_d7_k", "ros_d7_bf"]].transform("sum")
    expected = pooled["ros_d7_k"] / pooled["ros_d7_bf"]
    pd.testing.assert_series_equal(league.answer["d7_k"], expected, check_names=False)
    # Forecast: nothing known on the date tells next week's league rate from the season's.
    pd.testing.assert_series_equal(
        league.reference["d7_k"], league.reference["k"], check_names=False
    )


def test_fit_season_predicts_the_short_horizons_beside_the_targets(table):
    x, (y, w) = input_frame(table), answers(table, horizons=(7, 30))
    config = plain_config(hidden=[8], max_epochs=3, patience=1, val_frac=0.34)
    short = [f"d{n}_{s}" for n in (7, 30) for s in TARGETS]
    relative = replace(config, relative_target="answer")
    preds, _ = fit_season(table, x, y, w, 2025, relative, league_rates(table, (7, 30)))
    assert list(preds.columns) == [*ID_COLS, *TARGETS, *short]
    assert np.isfinite(preds[short].to_numpy()).all() and (preds[short] >= 0).all().all()
    with pytest.raises(ValueError, match="same horizons"):
        fit_season(table, x, y, w, 2025, relative, league_rates(table))
    with pytest.raises(ValueError, match="TARGETS then"):  # a horizon before the targets
        fit_season(table, x, y[[short[0], *TARGETS]], w[[short[0], *TARGETS]], 2025, config)


def test_a_row_with_no_answer_has_no_weight(table):
    empty = table.copy()
    row = _starter_week0(table)
    for c in (c for c in table.columns if c.startswith("ros_")):
        empty.loc[row, c] = 0
    y, w = answers(empty, "bf2")
    assert y.loc[row].isna().all() and (w.loc[row] == 0).all()


def test_inputs_never_read_the_answer(table):
    """Scramble every ros_ column: not one input may move."""
    scrambled = table.copy()
    rng = np.random.default_rng(0)
    for c in (c for c in table.columns if c.startswith("ros_")):
        scrambled[c] = rng.permutation(table[c].to_numpy())
    pd.testing.assert_frame_equal(input_frame(table), input_frame(scrambled))
    assert not (input_frame(table) == np.inf).any().any()


def test_without_his_own_windows_only_those_inputs_go(table):
    from fantasy_baseball.pitcher_ros.features import COVERAGE, INPUT_RECENT, WINDOWS

    full, lean = input_frame(table), input_frame(table, own_windows=False)
    prefixes = tuple(f"{w}_" for w in (*WINDOWS, *INPUT_RECENT))

    def own(c):
        return c.startswith(prefixes) and "_team_" not in c and c not in COVERAGE

    gone = set(full.columns) - set(lean.columns)
    assert gone and all(own(c) for c in gone)
    assert not any(own(c) for c in lean.columns)
    # Minor leagues, team, league and context stay, unchanged.
    for c in ("milb_p1_k_bf", "std_team_r_per_g", "lg_p1_k_bf", "age", "p1_in_store"):
        pd.testing.assert_series_equal(lean[c], full[c])


def test_without_his_role_windows_only_the_role_rates_go(table):
    from fantasy_baseball.pitcher_ros.features import INPUT_RECENT, ROLE_RATES, WINDOWS

    full, lean = input_frame(table), input_frame(table, role_windows=False)
    role = {f"{w}_{r}" for w in (*WINDOWS, *INPUT_RECENT) for r in ROLE_RATES}
    assert role <= set(full.columns)
    assert set(full.columns) - set(lean.columns) == role
    pd.testing.assert_frame_equal(lean, full.drop(columns=sorted(role)))


def test_fit_season_predicts_every_test_row_as_rates(table):
    x, (y, w) = input_frame(table), answers(table)
    config = plain_config(hidden=[8], max_epochs=3, patience=1, val_frac=0.34)
    preds, info = fit_season(table, x, y, w, 2025, config)
    assert list(preds.columns) == [*ID_COLS, *TARGETS]
    assert len(preds) == (table.season == 2025).sum()
    assert (preds[list(TARGETS)] >= 0).all().all() and preds[list(TARGETS)].notna().all().all()
    assert info["test_season"] == 2025 and info["train_rows"] > 0


def test_what_if_inputs_are_predicted_by_the_same_trained_net(table):
    x, (y, w) = input_frame(table), answers(table)
    config = plain_config(hidden=[8], max_epochs=3, patience=1, val_frac=0.34)
    older = with_age(x, 45)
    preds, info = fit_season(table, x, y, w, 2025, config, what_if={"same": x, "old": older})
    # Unchanged inputs through the same net give the same predictions; only age moved.
    pd.testing.assert_frame_equal(info["what_if"]["same"], preds)
    assert list(info["what_if"]["old"].columns) == list(preds.columns)
    assert not np.allclose(info["what_if"]["old"][list(TARGETS)], preds[list(TARGETS)])
    assert "what_if" not in fit_season(table, x, y, w, 2025, config)[1]
    with pytest.raises(ValueError, match="rows and columns"):
        fit_season(table, x, y, w, 2025, config, what_if={"bad": x.drop(columns="age")})


def test_with_age_sets_only_the_known_ages():
    x = pd.DataFrame({"age": [24.0, np.nan, 37.0], "k": [0.2, 0.3, 0.25]})
    out = with_age(x, 27)
    assert out["age"].tolist()[0::2] == [27.0, 27.0] and np.isnan(out["age"].iloc[1])
    pd.testing.assert_series_equal(out["k"], x["k"])
    with pytest.raises(ValueError, match="no age"):
        with_age(x.drop(columns="age"), 27)


def test_only_settings_the_pitcher_net_uses_are_accepted_and_recorded():
    c = plain_config(seed=3, dropout=0.2)
    assert (c.seq, c.relative_target, c.count_loss, c.avg_pieces) == ("none",) * 4
    check_plain(c)
    assert set(applied_settings(c)) == set(APPLIED) and applied_settings(c)["seed"] == 3
    with pytest.raises(ValueError, match="split"):
        check_plain(plain_config(split=True))


def test_week_zero_must_be_the_whole_season_left(table):
    check_week_zero(table)
    bad = table.copy()
    bad.loc[_starter_week0(table), "frac_season_left"] = 0.9
    with pytest.raises(ValueError, match="week 0"):
        check_week_zero(bad)


def _preds(player, weeks, dates):
    return pd.DataFrame(
        {
            "player_id": [player] * len(weeks),
            "season": [2025] * len(weeks),
            "week": weeks,
            "as_of": pd.to_datetime(dates),
            **{s: [0.1 * (i + 1) for i in range(len(weeks))] for s in TARGETS},
        }
    )


def test_our_rows_are_picked_as_known_on_the_date():
    preds = _preds(RELIEVER, [0, 1, 2], ["2025-04-01", "2025-04-08", "2025-04-15"])
    pre = backtest._ours_at({"ours": preds}, 2025, None)["ours"]
    assert pre.loc[RELIEVER, "k"] == pytest.approx(0.1)
    mid = backtest._ours_at({"ours": preds}, 2025, pd.Timestamp("2025-04-14"))["ours"]
    assert mid.loc[RELIEVER, "k"] == pytest.approx(0.2)  # week 2's row is after the date
    twice = pd.concat([preds, preds])
    with pytest.raises(ValueError, match="more than one"):
        backtest._ours_at({"ours": twice}, 2025, None)


def test_ours_must_cover_every_scored_pitcher_and_not_borrow_a_name():
    actual = pd.DataFrame(
        {"bf": [200.0, 200.0], **{s: [0.2, 0.2] for s in TARGETS}},
        index=pd.Index([STARTER, RELIEVER], name="player_id"),
    )
    systems = {"steamer": actual[list(TARGETS)]}
    partial = {"ours": actual.loc[[STARTER], list(TARGETS)]}
    with pytest.raises(ValueError, match="no projection for 1 of the 2"):
        backtest._check_ours(partial, systems, actual, 100, "2025")
    with pytest.raises(ValueError, match="named like"):
        backtest._check_ours({"marcel": actual[list(TARGETS)]}, systems, actual, 100, "2025")
    backtest._check_ours({"ours": actual[list(TARGETS)]}, systems, actual, 100, "2025")


def _seed_preds(rows, k):
    ids = pd.DataFrame(rows, columns=ID_COLS)
    return ids.assign(**{t: k if t == "k" else 0.1 for t in TARGETS})


def test_seed_average_is_row_for_row_whatever_the_order():
    rows = [(1, 2025, 0, "2025-03-27"), (2, 2025, 0, "2025-03-27")]
    a = _seed_preds(rows, [0.20, 0.30])
    b = _seed_preds(rows[::-1], [0.10, 0.40])  # same rows, other order: player 2 then 1
    out = average_seeds([a, b]).set_index("player_id")
    assert out.loc[1, "k"] == pytest.approx(0.30)  # 0.20 and 0.40
    assert out.loc[2, "k"] == pytest.approx(0.20)  # 0.30 and 0.10
    assert list(out.columns) == [c for c in ID_COLS if c != "player_id"] + list(TARGETS)
    assert len(out) == 2


def test_seeds_that_predicted_different_rows_are_refused():
    a = _seed_preds([(1, 2025, 0, "2025-03-27"), (2, 2025, 0, "2025-03-27")], [0.2, 0.3])
    b = _seed_preds([(1, 2025, 0, "2025-03-27")], [0.2])
    with pytest.raises(ValueError, match="different rows"):
        average_seeds([a, b])
    with pytest.raises(ValueError, match="no predictions"):
        average_seeds([])


def test_seed_average_carries_era_pieces_and_refuses_a_blank_in_one_seed():
    rows = [(1, 2025, 0, "2025-03-27")]
    a = _seed_preds(rows, [0.2]).assign(bb_bf=0.08)
    b = _seed_preds(rows, [0.4]).assign(bb_bf=0.10)
    assert average_seeds([a, b])["bb_bf"].iloc[0] == pytest.approx(0.09)
    # Blank in every seed (no league rate to scale it back by) stays blank.
    blank = average_seeds([a.assign(bb_bf=np.nan), b.assign(bb_bf=np.nan)])
    assert blank["bb_bf"].isna().all()
    with pytest.raises(ValueError, match="blank"):
        average_seeds([a, b.assign(bb_bf=np.nan)])


def test_a_repeated_row_is_refused_not_merged():
    row = (1, 2025, 0, "2025-03-27")
    twice = _seed_preds([row, row], [0.2, 0.3])
    with pytest.raises(ValueError, match="repeated a row"):
        average_seeds([twice, twice])


def test_league_reference_pools_last_season_and_this_one_before_the_date(table):
    ref = league_reference(table)
    # 2024: 10 games of STARTER (24 BF, 5 K) and RELIEVER (4 BF, 1 K), plus TWOWAY's
    # 20-BF start (MOPUP, a first baseman, isn't a pitcher-season). 2025 week 1: the
    # first 7 days of the same, with TWOWAY's start on day one.
    k, bf = 10 * 6 + 7 * 6, (10 * 28 + 20) + (7 * 28 + 20)
    assert ref.loc[_row_index(table, STARTER, 2025, 1), "k"] == pytest.approx(k / bf)
    # The store's first season has no last season to forecast from.
    assert ref.loc[_row_index(table, STARTER, 2024, 0)].isna().all()


def test_league_rates_include_era_pieces(table):
    row = _row_index(table, STARTER, 2025, 1)
    for frame in (league_reference(table), league_answer_rates(table)):
        assert set(ERA_PIECES) <= set(frame.columns)
    # Last season plus this one before the date, as for K: only STARTER walks anyone.
    bb, bf = 10 * 2 + 7 * 2, (10 * 28 + 20) + (7 * 28 + 20)
    assert league_reference(table).loc[row, "bb_bf"] == pytest.approx(bb / bf)


def test_league_answer_rates_pool_the_rest_of_the_season(table):
    ans = league_answer_rates(table)
    # 2025 from opening day: 21 games of STARTER and RELIEVER, and TWOWAY's start.
    assert ans.loc[_row_index(table, RELIEVER, 2025, 0), "k"] == pytest.approx(
        21 * 6 / (21 * 28 + 20)
    )


def test_park_inputs_are_trained_on_only_when_the_config_says_so(table):
    """A run is never credited to park inputs it didn't have, or the reverse."""
    x, (y, w) = input_frame(table), answers(table)
    with_parks = pd.concat([x, pd.DataFrame(1.0, index=x.index, columns=PARK_FEATURES)], axis=1)
    tiny = plain_config(hidden=[8], max_epochs=3, patience=1, val_frac=0.34)
    with pytest.raises(ValueError, match="park inputs"):
        fit_season(table, x, y, w, 2025, replace(tiny, parks="p3"))
    with pytest.raises(ValueError, match="park inputs"):
        fit_season(table, with_parks, y, w, 2025, tiny)
    _, info = fit_season(table, with_parks, y, w, 2025, replace(tiny, parks="p3"))
    assert info["n_features"] == fit_season(table, x, y, w, 2025, tiny)[1]["n_features"] + len(
        PARK_FEATURES
    )
    assert applied_settings(replace(tiny, parks="p3"))["parks"] == "p3"


def _tiny(relative_target):
    return plain_config(
        hidden=[8], max_epochs=3, patience=1, val_frac=0.34, relative_target=relative_target
    )


def test_answer_predictions_are_scaled_back_by_the_forecast(table):
    """ "answer" learns against the league's actual rates and scales back by the forecast
    known on the date: doubling the forecast doubles every prediction, and the answers'
    league rates never reach a prediction."""
    x, (y, w) = input_frame(table), answers(table)
    league = league_rates(table)
    preds, _ = fit_season(table, x, y, w, 2025, _tiny("answer"), league)
    doubled = LeagueRates(league.reference * 2, league.answer)
    twice, _ = fit_season(table, x, y, w, 2025, _tiny("answer"), doubled)
    rates = list(TARGETS)
    assert preds[rates].notna().all().all()
    pd.testing.assert_frame_equal(twice[rates], preds[rates] * 2)
    # Built here when not passed in: the same predictions.
    again, _ = fit_season(table, x, y, w, 2025, _tiny("answer"))
    pd.testing.assert_frame_equal(again, preds)


@pytest.fixture(scope="module")
def table3(tmp_path_factory):
    """Three seasons: "known" can't learn from the first (no forecast), so it needs two
    before the test season."""
    root = tmp_path_factory.mktemp("store3")
    _write(root, {2023: _season(2023, date(2023, 4, 1), 10, k=4), **_seasons()})
    return build_table(root)


def test_known_predictions_learn_and_scale_back_by_the_same_forecast(table3):
    """With "known", targets are divided by the forecast and predictions multiplied back
    by it: doubling it leaves the standardized targets, so the predictions, unchanged."""
    table = table3
    x, (y, w) = input_frame(table), answers(table)
    league = league_rates(table)
    preds, _ = fit_season(table, x, y, w, 2025, _tiny("known"), league)
    doubled = LeagueRates(league.reference * 2, league.answer)
    same, _ = fit_season(table, x, y, w, 2025, _tiny("known"), doubled)
    rates = list(TARGETS)
    assert preds[rates].notna().all().all()
    pd.testing.assert_frame_equal(same[rates], preds[rates], rtol=1e-5)


def test_a_season_with_no_usable_forecast_is_refused(table):
    x, (y, w) = input_frame(table), answers(table)
    # The store's first season has no last season to forecast from.
    with pytest.raises(ValueError, match="no usable league forecast"):
        fit_season(table, x, y, w, 2024, _tiny("answer"))
    # A league rate of 0 can't scale a prediction back: refused, not multiplied by 0.
    league = league_rates(table)
    no_saves = LeagueRates(league.reference.assign(sv=0.0), league.answer)
    with pytest.raises(ValueError, match="no usable league forecast"):
        fit_season(table, x, y, w, 2025, _tiny("answer"), no_saves)


def test_the_net_as_trained_learns_relative_to_the_league():
    assert DEFAULT_RELATIVE_TARGET == "answer"
    assert net_config(seed=2).relative_target == "answer"
    assert net_config().seq == "transformer" and net_config(seq="none").seq == "none"
    assert plain_config().seq == "none"  # the plain baseline stays plain
    assert net_config(relative_target="none").relative_target == "none"
    assert plain_config().relative_target == "none"  # the plain baseline stays plain


def test_the_net_as_trained_predicts_era_pieces():
    assert DEFAULT_ERA_PIECES is True


def test_the_net_as_trained_predicts_the_next_week_and_month():
    assert DEFAULT_HORIZONS is True


def test_the_net_as_trained_reads_park_inputs():
    assert net_config().parks == "p3" and net_config(parks="none").parks == "none"
    assert plain_config().parks == "none"  # the plain baseline stays plain


def _row_index(table, player, season, week):
    match = table.index[
        (table.player_id == player) & (table.season == season) & (table.week == week)
    ]
    assert len(match) == 1
    return match[0]


def test_train_on_limits_the_training_rows_not_the_predictions(table):
    x, (y, w) = input_frame(table), answers(table)
    config = plain_config(hidden=[8], max_epochs=2, patience=1, val_frac=0.34)
    keep = pd.Series(table["player_id"] != RELIEVER, index=table.index)
    preds, info = fit_season(table, x, y, w, 2025, config, train_on=keep)
    eligible = table["season_complete"] & (table["season"] < 2025) & (w.sum(axis=1) > 0)
    assert info["train_rows"] + info["val_rows"] == (eligible & keep).sum()
    assert (eligible & ~keep).any()  # the reliever had rows it could have trained on
    # Every test row is predicted, the reliever's included.
    assert len(preds) == (table["season"] == 2025).sum()
    assert RELIEVER in set(preds["player_id"])
    with pytest.raises(ValueError, match="one entry per table row"):
        fit_season(table, x, y, w, 2025, config, train_on=keep.iloc[1:])


def test_swap_saves_takes_only_the_saves_outputs():
    from fantasy_baseball.pitcher_ros.model import DEFAULT_SAVES_NET, ID_COLS, swap_saves

    assert DEFAULT_SAVES_NET is True
    ids = pd.DataFrame({"player_id": [1, 2], "season": 2025, "week": 0, "as_of": "2025-03-27"})
    main = ids.assign(k=[0.2, 0.3], sv=[0.0, 0.01], d7_sv=[0.0, 0.02], d7_k=[0.25, 0.3])
    saves = ids.assign(k=[9.0, 9.0], sv=[0.05, 0.06], d7_sv=[0.07, 0.08], d7_k=[9.0, 9.0])
    out = swap_saves(main, saves)
    assert out["sv"].tolist() == [0.05, 0.06] and out["d7_sv"].tolist() == [0.07, 0.08]
    assert out["k"].tolist() == [0.2, 0.3] and out["d7_k"].tolist() == [0.25, 0.3]
    assert main["sv"].tolist() == [0.0, 0.01]  # not changed in place
    with pytest.raises(ValueError, match="different rows"):
        swap_saves(main, saves.iloc[::-1])
    with pytest.raises(ValueError, match="different rows"):
        swap_saves(main, saves.iloc[:1])
    with pytest.raises(ValueError, match="saves outputs differ"):
        swap_saves(main, saves.drop(columns="d7_sv"))
    assert list(out.columns) == list(main.columns) and out[ID_COLS].equals(main[ID_COLS])
