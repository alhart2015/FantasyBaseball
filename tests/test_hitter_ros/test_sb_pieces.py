"""#413: SB as its pieces -- opportunities per PA, attempts per opportunity, SB per attempt."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.features import (
    SB_PIECES,
    TARGETS,
    horizon_columns,
    input_frame,
    league_answer_rates,
    league_reference,
    sb_from_pieces,
    sb_piece_rates,
    target_frame,
    target_stat,
)
from fantasy_baseball.hitter_ros.table import build_table
from tests.test_hitter_ros.test_table import HITTER, _row, _season, _write


def _with_steals(season):
    """test_table's season, where HITTER (on first with second open once a game) steals
    in every third game and is caught in the game after. Plus a strikeout in every game
    of 3+ AB: AVG's pieces (on by default) need a league K rate above 0."""
    lineups, pitches = season
    for row in lineups:
        row["k"] = 1 if row["ab"] >= 3 else 0
    hitter_games = [r for r in lineups if r["player_id"] == HITTER]
    for i, row in enumerate(hitter_games):
        row["sb"] = 1 if i % 3 == 0 else 0
        row["cs"] = 1 if i % 3 == 1 else 0
    return lineups, pitches


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    seasons = {2024: _season(2024, date(2024, 4, 1), 10), 2025: _season(2025, date(2025, 4, 1), 21)}
    _write(root, {year: _with_steals(s) for year, s in seasons.items()})
    return build_table(root)


def test_pieces_rebuild_sb_exactly():
    rng = np.random.default_rng(0)
    pa = rng.integers(1, 700, size=200)
    opp2 = rng.integers(0, pa // 3 + 1)
    opp3 = rng.integers(0, pa // 10 + 1)
    attempts = rng.integers(0, opp2 + opp3 + 1)
    sb = rng.integers(0, attempts + 1)
    counts = pd.DataFrame(
        {"pa": pa, "sb": sb, "cs": attempts - sb, "steal_opp2": opp2, "steal_opp3": opp3}
    )
    p = sb_piece_rates(counts)
    assert list(p.columns) == list(SB_PIECES)
    known = p.notna().all(axis=1)
    np.testing.assert_allclose(
        sb_from_pieces(p.opp_pa, p.att_opp, p.sb_att)[known], (sb / pa)[known]
    )
    # No opportunities: attempts per opportunity unknown; no attempts: success unknown.
    none = sb_piece_rates(
        pd.DataFrame(
            {"pa": [4, 4], "sb": [0, 0], "cs": [0, 0], "steal_opp2": [0, 2], "steal_opp3": [0, 0]}
        )
    )
    assert none.opp_pa.tolist() == [0.0, 0.5]
    assert np.isnan(none.att_opp[0]) and none.att_opp[1] == 0.0
    assert none.sb_att.isna().all()


def test_piece_column_names():
    assert target_stat("n25_sb_att") == "sb_att" and target_stat("opp_pa") == "opp_pa"
    assert target_stat("n100_att_opp") == "att_opp"


def test_table_counts_steal_chances_and_caught_stealing_going_forward(table):
    w1 = _row(table, HITTER, 2025, 1)  # as of 2025-04-08: 7 of 21 games played
    assert w1.ros_steal_opp2 == 14 and w1.ros_steal_opp3 == 0
    w0 = _row(table, HITTER, 2025, 0)
    # Next 25 PA: 7 games, one chance each; SB in games 0, 3, 6, caught in 1, 4.
    assert w0.ros_n25_steal_opp2 == 7 and w0.ros_n25_sb == 3 and w0.ros_n25_cs == 2
    # From week 1 (7 games in): the next 7 games are games 7-13.
    assert w1.ros_n25_steal_opp2 == 7 and w1.ros_n25_sb == 2 and w1.ros_n25_cs == 3
    # The league's chances before the date (HITTER is the only runner).
    assert w1.lg_std_steal_opp2 == 7 and w1.lg_p1_steal_opp2 == 10


def test_targets_with_sb_pieces(table):
    plain_rates, plain_weights = target_frame(table, (25,), pieces=True)
    rates, weights = target_frame(table, (25,), pieces=True, sb_pieces=True)
    sb_cols = horizon_columns((25,), SB_PIECES)
    assert list(rates.columns) == [*plain_rates.columns, *sb_cols] == list(weights.columns)
    pd.testing.assert_frame_equal(plain_rates, rates[list(plain_rates.columns)])
    pd.testing.assert_frame_equal(plain_weights, weights[list(plain_weights.columns)])
    row = (table.player_id == HITTER) & (table.season == 2025) & (table.week == 0)
    got, w = rates.loc[row].iloc[0], weights.loc[row].iloc[0]
    assert got["n25_opp_pa"] == pytest.approx(7 / 28) and w["n25_opp_pa"] == 28
    assert got["n25_att_opp"] == pytest.approx(5 / 7) and w["n25_att_opp"] == 7
    assert got["n25_sb_att"] == pytest.approx(3 / 5) and w["n25_sb_att"] == 5
    assert sb_from_pieces(
        got["n25_opp_pa"], got["n25_att_opp"], got["n25_sb_att"]
    ) == pytest.approx(got["n25_sb"])
    # OTHER is never on base with a base open: no chances, so no attempt rate, weight 0.
    other = (table.player_id != HITTER) & (table.season == 2025) & (table.week == 0)
    assert rates.loc[other, "att_opp"].isna().all() and (weights.loc[other, "att_opp"] == 0).all()


def test_league_rates_with_sb_pieces(table):
    for rates in (league_answer_rates, league_reference):
        plain, both = rates(table), rates(table, sb_pieces=True)
        assert list(both.columns) == [*TARGETS, *SB_PIECES]
        pd.testing.assert_frame_equal(plain, both[list(TARGETS)])
        assert list(rates(table, pieces=True, sb_pieces=True).columns)[-3:] == list(SB_PIECES)
    a = league_answer_rates(table, sb_pieces=True)
    rows = (table.season == 2025) & (table.week == 0)
    first = a.loc[rows].iloc[0]
    assert sb_from_pieces(first.opp_pa, first.att_opp, first.sb_att) == pytest.approx(first.sb)
    # The reference's pieces use SB's window too, so they multiply back to its SB.
    ref = league_reference(table, sb_pieces=True)
    known = ref[list(SB_PIECES)].notna().all(axis=1)
    assert known.any()
    np.testing.assert_allclose(
        sb_from_pieces(ref.opp_pa, ref.att_opp, ref.sb_att)[known], ref.sb[known]
    )


def test_sb_piece_config_and_loss():
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig, count_loss, device

    assert NetConfig().sb_pieces == "none"
    with pytest.raises(ValueError):
        NetConfig(sb_pieces="all")
    stats = [*TARGETS, *SB_PIECES]
    y = np.full((10, len(stats)), 0.25)
    y[:5] = 0.2
    w = np.ones_like(y)
    spec = count_loss(NetConfig(sb_pieces="extra", avg_pieces="none"), y, w, device(), stats)
    # Opportunities and attempts are Poisson counts (attempts per opportunity can pass
    # 1: two tries on one opportunity); SB per attempt is binomial.
    assert spec.mask.tolist() == [s in ("r", "hr", "rbi", "sb", "opp_pa", "att_opp") for s in stats]
    assert spec.binomial.tolist() == [s == "sb_att" for s in stats]


def _config(**kw):
    from fantasy_baseball.hitter_ros.net import NetConfig

    return NetConfig(
        hidden=[4],
        max_epochs=2,
        probes="none",
        milb="none",
        parks="none",
        steal_inputs=False,
        val_frac=0.0,
        **kw,
    )


def test_fit_season_refuses_mismatched_targets(table):
    pytest.importorskip("torch")
    from scripts.train_hitter_ros import fit_season

    y_all, w_all = target_frame(table, (25,), pieces=True)
    with pytest.raises(ValueError, match="sb_pieces"):
        fit_season(table, input_frame(table), y_all, w_all, 2025, _config(sb_pieces="extra"))
    y_all, w_all = target_frame(table, (25,), pieces=True, sb_pieces=True)
    with pytest.raises(ValueError, match="sb_pieces"):
        fit_season(table, input_frame(table), y_all, w_all, 2025, _config())


@pytest.mark.parametrize("mode", ["extra", "derived"])
def test_fit_season_with_sb_pieces(table, mode):
    pytest.importorskip("torch")
    from scripts.train_hitter_ros import fit_season

    # The test seasons are too short to reach 100 PA: next 25 PA only.
    y_all, w_all = target_frame(table, (25,), pieces=True, sb_pieces=True)
    preds, _ = fit_season(table, input_frame(table), y_all, w_all, 2025, _config(sb_pieces=mode))
    assert list(preds.columns) == ["player_id", "season", "week", "as_of", *y_all.columns]
    for tag in ("", "n25_"):
        counts = preds[[f"{tag}opp_pa", f"{tag}att_opp"]].stack()
        assert (counts > 0).all()  # rates from a log
        success = preds[f"{tag}sb_att"]
        assert ((success > 0) & (success < 1)).all()  # a rate from log-odds
        derived = sb_from_pieces(*(preds[f"{tag}{p}"] for p in SB_PIECES))
        if mode == "derived":
            pd.testing.assert_series_equal(preds[f"{tag}sb"], derived, check_names=False)
        else:
            assert not np.allclose(preds[f"{tag}sb"], derived)  # its own output


def test_steal_chances_only_on_dates_with_a_lineup_row(tmp_path):
    """A steal chance on a date with no lineup row for the runner would sit in the
    counts before the date but never in the horizon's running totals, so the horizon
    count would come out too low. Such dates are left out everywhere."""
    lineups, pitches = _with_steals(_season(2025, date(2025, 4, 1), 21))
    gap = date(2025, 4, 4).isoformat()  # game 3, before week 1 (2025-04-08)
    lineups = [r for r in lineups if not (r["player_id"] == HITTER and r["game_date"] == gap)]
    seasons = {2024: _with_steals(_season(2024, date(2024, 4, 1), 10)), 2025: (lineups, pitches)}
    _write(tmp_path, seasons)
    t = build_table(tmp_path)
    w1 = _row(t, HITTER, 2025, 1)
    assert w1.std_steal_opp2 == 6  # games 0-6 less game 3
    # Next 25 PA from week 1: games 7-13, one chance each.
    assert w1.ros_n25_steal_opp2 == 7
    horizon = [c for c in t.columns if c.startswith("ros_n") and "steal_opp" in c]
    assert (t[horizon].fillna(0) >= 0).all().all()


def test_sb_piece_weights_are_never_negative():
    t = pd.DataFrame({f"ros_{c}": [0] for c in ("pa", "ab", "h", "r", "hr", "rbi", "sb", "k")})
    for c in ("cs", "steal_opp2", "steal_opp3"):
        t[f"ros_{c}"] = [-1]
    t["ros_pa"], t["ros_sb"] = [10], [0]
    _, weights = target_frame(t, sb_pieces=True)
    assert (weights[list(SB_PIECES)] >= 0).all().all()


def test_usable_league_rates_blank_rates_that_would_be_infinite():
    from scripts.train_hitter_ros import usable_league_rates

    rates = pd.DataFrame(
        {"sb": [0.0, 0.01, 0.02], "sb_att": [0.0, 1.0, 0.75], "avg": [0.25, 0.25, np.nan]}
    )
    got = usable_league_rates(rates, ("sb_att",))
    # A count rate needs only to be above 0; a binomial one also below 1.
    assert got["sb"].isna().tolist() == [True, False, False]
    assert got["sb_att"].isna().tolist() == [True, True, False]
    assert got["avg"].isna().tolist() == [False, False, True]


def test_sb_piece_loss_split_is_named():
    from fantasy_baseball.hitter_ros.features import SB_BINOMIAL_PIECES, SB_POISSON_PIECES

    assert (*SB_POISSON_PIECES, *SB_BINOMIAL_PIECES) == SB_PIECES
    assert SB_BINOMIAL_PIECES == ("sb_att",)


def test_no_milb_sb_without_milb_is_refused(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    from scripts import train_hitter_ros

    monkeypatch.setattr(train_hitter_ros, "RUNS", tmp_path / "runs")
    monkeypatch.setattr("sys.argv", ["train", "--name", "x", "--milb", "none", "--no-milb-sb"])
    with pytest.raises(SystemExit):
        train_hitter_ros.main()
    assert "--no-milb-sb" in capsys.readouterr().err
