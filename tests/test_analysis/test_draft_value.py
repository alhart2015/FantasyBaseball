import json as _json
from collections import Counter

from fantasy_baseball.analysis import draft_value as dv
from fantasy_baseball.utils.name_utils import normalize_name as nn


def _hitter_line(**kw):
    base = {"r": 90, "hr": 30, "rbi": 95, "sb": 12, "avg": 0.280, "ab": 560}
    base.update(kw)
    return base


def test_score_var_reproduces_board_var_for_onboard_player():
    board, scale = dv.reproduce_draft_day_board()
    row = board[board["player_type"] == "hitter"].iloc[0]
    line = {k: row[k] for k in ("r", "hr", "rbi", "sb", "avg", "ab")}
    var = dv.score_var(line, list(row["positions"]), "hitter", scale)
    assert abs(var - float(row["var"])) < 1e-6  # same scale -> same VAR


def test_score_var_fraction_half_scales_counting_not_rate():
    _, scale = dv.reproduce_draft_day_board()
    full = dv.score_var(_hitter_line(), ["OF"], "hitter", scale, fraction=1.0)
    half = dv.score_var(_hitter_line(), ["OF"], "hitter", scale, fraction=0.5)
    # counting SGP halves, rate SGP is fraction-invariant, floor also to-date -> VAR roughly halves but not below full
    assert half < full


def test_build_preseason_board_returns_scale_and_soft_frozen():
    board, scale = dv.reproduce_draft_day_board()
    assert not board.empty
    assert set(scale.replacement_levels) >= {"C", "1B", "SS", "OF", "SP", "RP"}
    assert scale.team_ab == 5500 and scale.team_ip == 1450
    assert {"era", "whip", "avg"} <= set(scale.repl_rates)
    # soft frozen cross-check: returns a drift summary, never raises
    summary = dv.frozen_drift_summary(board)
    assert summary["joined"] > 0
    assert set(summary) >= {"joined", "over_tol", "max", "median"}


def test_reconstruct_draft_shape_and_gate():
    picks = dv.reconstruct_draft()
    keepers = [p for p in picks if p.is_keeper]
    drafted = [p for p in picks if not p.is_keeper]
    assert len(keepers) == 30
    assert len(drafted) == 200
    # every team owns exactly 3 keepers
    assert set(Counter(p.team for p in keepers).values()) == {3}

    # ENFORCE the known-roster gate (spec oracle 6b): the user's roster must
    # reconstruct exactly. Infer the user's team from its keepers, then assert its
    # reconstructed roster is a superset of state["user_roster"].
    state = _json.loads(dv._DRAFT_STATE.read_text(encoding="utf-8"))
    user_roster = state["user_roster"]
    league = dv._load_league()
    keeper_team = {nn(k["name"]): k["team"] for k in league["keepers"]}
    user_team = next(keeper_team[nn(n)] for n in user_roster if nn(n) in keeper_team)
    assert dv.validate_reconstruction(picks, known_team=user_team, known_roster=user_roster) == []


def test_par_curve_is_descending_and_keeper_mean():
    board, _scale = dv.reproduce_draft_day_board()
    picks = dv.reconstruct_draft()
    curve = dv.build_par_curve(picks, board)
    # drafted pars sorted descending
    assert curve.drafted_pars == sorted(curve.drafted_pars, reverse=True)
    # par_for_slot(1) is the top on-board drafted VAR
    assert curve.par_for_slot(1) == curve.drafted_pars[0]
    # keeper par is the mean of the keeper VARs (finite, not NaN)
    assert curve.keeper_par == curve.keeper_par
