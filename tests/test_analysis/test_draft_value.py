import json as _json
from collections import Counter

import pytest

from fantasy_baseball.analysis import draft_value as dv
from fantasy_baseball.utils.name_utils import normalize_name as nn


def _hitter_line(**kw):
    base = {"r": 90, "hr": 30, "rbi": 95, "sb": 12, "avg": 0.280, "ab": 560}
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def synthetic_scale():
    # build the real board/scale once for all oracle tests in this module
    _board, scale = dv.reproduce_draft_day_board()
    return scale


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


def test_full_season_and_actual_loaders_shape():
    full_by_mlbam, full_by_name = dv.load_full_season_lines()
    assert full_by_name, "no full-season lines (KV store not synced?)"
    assert full_by_mlbam, "no mlbam-keyed full-season lines (KV store not synced?)"
    # name map is keyed name_normalized::player_type; mlbam map is keyed
    # (int mlbam id, player_type) so a two-way player's hitter and pitcher
    # records do not collide under one id.
    k = next(iter(full_by_name))
    assert "::" in k
    line = full_by_name[k]
    assert any(s in line for s in ("hr", "k"))
    mk = next(iter(full_by_mlbam))
    assert isinstance(mk, tuple) and isinstance(mk[0], int) and mk[1] in ("hitter", "pitcher")
    assert any(s in full_by_mlbam[mk] for s in ("hr", "k"))
    # actual-to-date loader has the same tuple shape
    td_by_mlbam, td_by_name = dv.load_actual_to_date_lines()
    assert td_by_mlbam and td_by_name, "no game-log lines (KV store not synced?)"
    assert "::" in next(iter(td_by_name))
    tk = next(iter(td_by_mlbam))
    assert isinstance(tk, tuple) and isinstance(tk[0], int) and tk[1] in ("hitter", "pitcher")


def test_season_fraction_in_unit_range():
    f = dv.season_fraction()
    assert 0.0 <= f <= 1.0


def test_ytd_fraction_is_not_linear_in_f(synthetic_scale):
    # Guards the f*floor_full bug: rate SGP is f-invariant while counting scales by f,
    # so a to-date VAR does NOT simply equal f * full VAR. This is oracle 5 at the
    # score_var level (a distinct, non-tautological check that f=1 convergence cannot see).
    scale = synthetic_scale
    full = dv.score_var(_hitter_line(), ["OF"], "hitter", scale, fraction=1.0)
    half = dv.score_var(_hitter_line(), ["OF"], "hitter", scale, fraction=0.5)
    assert half < full  # counting-dominated: to-date VAR is smaller
    assert abs(half - 0.5 * full) > 1e-6  # but NOT linear in f (rate component invariant)


def test_value_decomposition_identity(synthetic_scale):
    scale = synthetic_scale
    line = _hitter_line()
    pv = dv.compute_player_value(
        team="Hart of the Order",
        name="Test Bat",
        player_type="hitter",
        positions=["OF"],
        baseline_proj=5.0,
        baseline_ytd=2.5,
        baseline_kind="drafted",
        preseason_var=8.0,
        full_line=line,
        todate_line=line,
        scale=scale,
        fraction=0.5,
    )
    # projected decomposition holds exactly
    assert abs((pv.skill + pv.luck) - pv.value_proj) < 1e-9
    # YTD is value-only
    assert pv.value_ytd is not None


def test_offboard_waiver_gem_skill_luck_na(synthetic_scale):
    scale = synthetic_scale
    line = _hitter_line()
    pv = dv.compute_player_value(
        team="Hart of the Order",
        name="Gem",
        player_type="hitter",
        positions=["OF"],
        baseline_proj=0.0,
        baseline_ytd=0.0,
        baseline_kind="waiver",
        preseason_var=None,
        full_line=line,
        todate_line=line,
        scale=scale,
        fraction=0.5,
    )
    assert pv.skill is None and pv.luck is None
    assert pv.value_proj is not None  # value still computed vs replacement (0)


def test_missing_line_est_scores_at_replacement(synthetic_scale):
    # A drafted/kept player with NO stat line (never played) is scored at the
    # replacement estimate (0.0), so value == 0 - par == -par (wasted pick penalized).
    scale = synthetic_scale
    pv = dv.compute_player_value(
        team="Hart of the Order",
        name="Never Played",
        player_type="hitter",
        positions=["OF"],
        baseline_proj=5.0,
        baseline_ytd=2.5,
        baseline_kind="drafted",
        preseason_var=None,
        full_line=None,
        todate_line=None,
        scale=scale,
        fraction=0.5,
        missing_line_est=0.0,
    )
    assert pv.value_proj == -5.0
    assert pv.est_var_proj == 0.0


def test_convergence_ytd_equals_proj_at_f1(synthetic_scale):
    # spec oracle 3: at f=1 (ROS->0), with full_line == todate_line and matching
    # baselines, the YTD value converges to the projected value. Non-tautological:
    # it exercises BOTH horizon paths (est_proj at fraction=1.0 vs est_ytd at fraction=1.0)
    # and both baselines through the real value computation.
    scale = synthetic_scale
    line = _hitter_line()
    pv = dv.compute_player_value(
        team="Hart of the Order",
        name="Test Bat",
        player_type="hitter",
        positions=["OF"],
        baseline_proj=5.0,
        baseline_ytd=5.0,
        baseline_kind="drafted",
        preseason_var=8.0,
        full_line=line,
        todate_line=line,
        scale=scale,
        fraction=1.0,
    )
    assert abs(pv.value_proj - pv.value_ytd) < 1e-9


def test_run_draft_value_end_to_end_and_known_pick():
    from fantasy_baseball.utils.name_utils import normalize_name

    players, teams = dv.run_draft_value()
    assert players and teams
    # every graded pick is a keeper or a drafted pick -- no waiver
    assert all(t.credited_count >= 0 for t in teams)
    assert all(p.baseline_kind in ("keeper", "drafted") for p in players)
    # every team drafted/kept roughly a full roster's worth of picks (3 keepers + 20
    # drafted); credited_count counts only picks with a finite value, so allow slack.
    assert all(18 <= t.credited_count <= 25 for t in teams), {
        t.team: t.credited_count for t in teams
    }
    # known-pick sanity: a specific keeper resolves with a finite projected value
    soto = next((p for p in players if normalize_name(p.name) == "juan soto"), None)
    assert soto is not None and soto.value_proj == soto.value_proj  # not NaN
    assert soto.baseline_kind == "keeper"

    # Mason Miller (the closer) was DRAFTED and is scored regardless of any later
    # drop/trade. Namesake-collision guard (mlbam-id join): the drafted Mason Miller
    # is the A's/Padres closer (mlbam 695243, ~37 SV projected), NOT the scrub
    # namesake (mlbam 692223, ~2 IP). His projected estVAR must be clearly positive.
    miller = next((p for p in players if normalize_name(p.name) == "mason miller"), None)
    assert miller is not None, "Mason Miller not among drafted picks -- update this guard"
    assert miller.baseline_kind == "drafted"
    assert miller.est_var_proj is not None
    assert miller.est_var_proj > 2.0, (
        f"Mason Miller estVAR {miller.est_var_proj} looks like the scrub namesake's "
        "line, not the closer's -- mlbam-id join regressed"
    )

    # Two-way player: Shohei Ohtani appears TWICE -- once as a keeper hitter (his
    # "batter only" keeper note) and once as a drafted pitcher (the remaining board
    # type after the keeper claimed hitter).
    ohtani = [p for p in players if normalize_name(p.name) == "shohei ohtani"]
    assert len(ohtani) == 2, f"expected 2 Ohtani rows, got {len(ohtani)}"
    kinds = {(o.baseline_kind, o.player_type) for o in ohtani}
    assert kinds == {("keeper", "hitter"), ("drafted", "pitcher")}, kinds


def test_team_rollup_sum_avg_count():
    # PlayerValue has 12 fields; construct with keywords to avoid positional drift.
    pvs = [
        dv.PlayerValue(
            team="Hart of the Order",
            name="A",
            player_type="hitter",
            slot=1,
            baseline_kind="drafted",
            preseason_var=8.0,
            est_var_proj=10.0,
            est_var_ytd=6.0,
            value_proj=4.0,
            value_ytd=2.5,
            skill=2.0,
            luck=2.0,
        ),
        dv.PlayerValue(
            team="Hart of the Order",
            name="B",
            player_type="hitter",
            slot=None,
            baseline_kind="waiver",
            preseason_var=None,
            est_var_proj=3.0,
            est_var_ytd=1.5,
            value_proj=3.0,
            value_ytd=1.4,
            skill=None,
            luck=None,
        ),
    ]
    r = dv.roll_up_team("Hart of the Order", pvs, horizon="proj")
    assert r.credited_count == 2
    assert abs(r.sum_value - 7.0) < 1e-9  # value_proj: 4.0 + 3.0
    assert abs(r.avg_value - 3.5) < 1e-9
