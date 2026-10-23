import dataclasses
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
def _season_payloads():
    """Live-season KV payloads built from committed data (#496), never the machine's
    own KV store, which holds whatever it last synced.

    The draft-day blend of data/projections/<season>/ (every player, with mlbam ids)
    is split 60/40: season-to-date game-log totals are 60% of each counting line, and
    the rest-of-season projection is the other 40%. The full-season payload is then
    made the way the ROS job makes it (ros_pipeline.derive_full_season, then
    ``to_dict(orient="records")``), so it has production's shape.
    """
    from fantasy_baseball.data.projections import (
        HITTING_COUNTING_COLS,
        PITCHING_COUNTING_COLS,
        blend_projections,
    )
    from fantasy_baseball.data.ros_pipeline import derive_full_season

    config = dv.load_config(dv._CONFIG)
    hitters, pitchers, _ = blend_projections(
        dv._REPO_ROOT / "data" / "projections" / str(config.season_year),
        config.projection_systems,
        config.projection_weights,
    )

    def split(frame, cols):
        cols = [c for c in cols if c in frame.columns]
        ros = frame.copy()
        ros[cols] = frame[cols].astype("float64") * 0.4
        played = frame[frame["mlbam_id"].notna()]
        to_date = {
            str(int(mid)): {"name": name, **{c: 0.6 * float(v) for c, v in line.items()}}
            for mid, name, line in zip(
                played["mlbam_id"],
                played["name"],
                played[cols].fillna(0).to_dict("records"),
                strict=True,
            )
        }
        return ros, to_date

    hitters_ros, hitter_totals = split(hitters, HITTING_COUNTING_COLS)
    pitchers_ros, pitcher_totals = split(pitchers, PITCHING_COUNTING_COLS)
    full_hitters, full_pitchers = derive_full_season(
        hitters_ros, pitchers_ros, hitter_totals, pitcher_totals
    )
    full = {
        "hitters": full_hitters.to_dict(orient="records"),
        "pitchers": full_pitchers.to_dict(orient="records"),
    }
    return full, {"hitters": hitter_totals, "pitchers": pitcher_totals}


@pytest.fixture
def season_kv(_season_payloads):
    """This test's own local KV (tests/conftest.py) seeded with the season payloads."""
    from fantasy_baseball.data.cache_keys import CacheKey
    from fantasy_baseball.data.kv_store import get_kv
    from fantasy_baseball.data.redis_store import set_game_log_totals
    from fantasy_baseball.web.season_data import write_cache

    full, to_date = _season_payloads
    write_cache(CacheKey.FULL_SEASON_PROJECTIONS, full)
    for kind, totals in to_date.items():
        set_game_log_totals(get_kv(), kind, totals)


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


def test_to_date_floors_golden_and_delegates_to_replacement_helper(synthetic_scale):
    # M10: _to_date_floors delegates to sgp.replacement.position_aware_replacement_levels
    # with a fraction, instead of re-encoding the counting-scaling recipe + UTIL rule.
    # Pin the pre-refactor floor values so the delegation cannot silently shift a floor
    # (every YTD grade nets against these).
    from fantasy_baseball.sgp.replacement import position_aware_replacement_levels

    scale = synthetic_scale
    # f=1.0 returns the board's own floors unchanged (cheap projected-side path).
    assert dv._to_date_floors(scale, 1.0) is scale.replacement_levels
    # f<1 golden values (captured from the pre-M10 hand-rolled implementation).
    gold_04 = {
        "1B": 3.498867,
        "2B": 3.622112,
        "3B": 3.559660,
        "C": 2.973834,
        "OF": 3.844582,
        "RP": 2.909401,
        "SP": 3.005698,
        "SS": 3.643083,
        "UTIL": 3.844582,
    }
    floors = dv._to_date_floors(scale, 0.4)
    for pos, want in gold_04.items():
        assert abs(floors[pos] - want) < 1e-5, (pos, floors[pos], want)
    # UTIL mirrors the best hitter floor (rule now lives in the shared helper).
    assert floors["UTIL"] == max(floors[p] for p in ("C", "1B", "2B", "3B", "SS", "OF"))
    # delegation identity: the helper called directly reproduces _to_date_floors.
    direct = position_aware_replacement_levels(
        scale.denoms,
        scale.repl_rates,
        team_ab=scale.team_ab,
        team_ip=scale.team_ip,
        fraction=0.4,
    )
    assert direct == floors


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
    # LITERAL pin, deliberately not the live DEFAULT_TEAM_* constants: the
    # draft-day scale is frozen (frozen_draft_2026/meta.json) so historical
    # par curves survive future recalibrations of the live defaults.
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
    cfg = dv.load_config(dv._CONFIG)
    state = _json.loads(dv._draft_state_path(cfg).read_text(encoding="utf-8"))
    user_roster = state["user_roster"]
    keepers = dv.load_config(dv._CONFIG).keepers
    keeper_team = {nn(k["name"]): k["team"] for k in keepers}
    user_team = next(keeper_team[nn(n)] for n in user_roster if nn(n) in keeper_team)
    assert dv.validate_reconstruction(picks, known_team=user_team, known_roster=user_roster) == []


def test_par_curve_is_descending_and_keeper_mean():
    board, _scale = dv.reproduce_draft_day_board()
    picks = dv.reconstruct_draft()
    bindex = dv._board_index(board)
    keepers = dv.load_config(dv._CONFIG).keepers
    typed_picks = dv._assign_pick_types(picks, bindex, keepers)
    curve = dv.build_par_curve(typed_picks, bindex)
    # drafted pars sorted descending
    assert curve.drafted_pars == sorted(curve.drafted_pars, reverse=True)
    # par_for_slot(1) is the top on-board drafted VAR
    assert curve.par_for_slot(1) == curve.drafted_pars[0]
    # keeper par is the mean of the keeper VARs (finite, not NaN)
    assert curve.keeper_par == curve.keeper_par


def test_full_season_and_actual_loaders_shape(season_kv):
    full_by_mlbam, full_by_name = dv.load_full_season_lines()
    assert full_by_name, "no full-season lines"
    assert full_by_mlbam, "no mlbam-keyed full-season lines"
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
    assert td_by_mlbam and td_by_name, "no game-log lines"
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


def test_frozen_var_loader_and_anchor():
    # M7: preseason_var / skill / luck / projected par curve must anchor to the frozen
    # draft-day VAR, not the drifted rebuild. Unit-test the loader + the anchor join.
    import pandas as pd

    fv = dv._frozen_var_by_player_id()
    assert fv, "frozen board did not load"
    k = next(iter(fv))
    assert isinstance(k, str) and "::" in k and isinstance(fv[k], float)
    # missing/malformed file -> empty dict, never raises (soft contract)
    assert dv._frozen_var_by_player_id("does_not_exist_board.json") == {}

    board = pd.DataFrame(
        {
            "player_id": ["100::hitter", "200::pitcher", "999::hitter"],
            "name": ["A", "B", "C"],
            "player_type": ["hitter", "pitcher", "hitter"],
            "var": [1.0, 2.0, 3.0],
        }
    )
    frozen = {"100::hitter": 5.5, "200::pitcher": 6.5}  # 999 absent
    out = dv._anchor_board_var_to_frozen(board, frozen)
    got = dict(zip(out["player_id"], out["var"], strict=True))
    assert got["100::hitter"] == 5.5  # anchored to frozen
    assert got["200::pitcher"] == 6.5  # anchored to frozen
    assert got["999::hitter"] == 3.0  # kept rebuilt VAR (no frozen anchor)
    # empty frozen -> fail loud (missing/all-corrupt board), NOT a silent ship of drifted VAR
    with pytest.raises(RuntimeError, match="no frozen VAR to anchor"):
        dv._anchor_board_var_to_frozen(board, {})
    # catastrophic mismatch (player_id key formats diverged) -> fail loud, NOT a silent
    # revert to drifted rebuilt VAR for every player.
    with pytest.raises(RuntimeError, match="key formats have likely diverged"):
        dv._anchor_board_var_to_frozen(board, {"deadbeef::hitter": 1.0})


def test_frozen_var_loader_skips_corrupt_rows(tmp_path):
    # Soft contract: a non-numeric / null / id-less var row is skipped, never raises.
    import json

    p = tmp_path / "board.json"
    p.write_text(
        json.dumps(
            [
                {"player_id": "1::hitter", "var": 3.5},
                {"player_id": "2::hitter", "var": "not_a_number"},
                {"player_id": "3::hitter", "var": None},
                {"player_id": None, "var": 9.9},
                {"player_id": "4::hitter", "var": float("nan")},  # non-finite -> skipped
                {"player_id": "5::hitter", "var": float("inf")},  # non-finite -> skipped
            ]
        ),
        encoding="utf-8",
    )
    assert dv._frozen_var_by_player_id(p) == {"1::hitter": 3.5}


def test_config_and_season_year_threading():
    # M9: config threads through instead of each call re-reading league.yaml, and the
    # preseason CSV dir is derived from config.season_year (no hardcoded 2026).
    import dataclasses

    cfg = dv.load_config(dv._CONFIG)
    assert 0.0 <= dv.season_fraction(cfg) <= 1.0
    picks = dv.reconstruct_draft(cfg)
    assert len(picks) == 230
    assert dv.validate_reconstruction(picks, config=cfg) == []
    # default season_year resolves to a real CSV dir (board builds)
    board, _scale = dv.reproduce_draft_day_board(cfg)
    assert not board.empty
    # a bogus season_year points the loader at a missing dir -> raises (proves derivation)
    bogus = dataclasses.replace(cfg, season_year=1999)
    with pytest.raises(FileNotFoundError):
        dv.reproduce_draft_day_board(bogus)


def test_run_draft_value_end_to_end_and_known_pick(season_kv):
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


def _pv(team, name, value_proj, **kw):
    """Construct a PlayerValue with sensible finite defaults; override via kw."""
    fields = dict(
        player_type="hitter",
        slot=None,
        baseline_kind="drafted",
        preseason_var=10.0,
        est_var_proj=12.0,
        est_var_ytd=6.0,
        value_ytd=2.0,
        skill=1.0,
        luck=1.0,
    )
    fields.update(kw)
    return dv.PlayerValue(team=team, name=name, value_proj=value_proj, **fields)


def test_build_cache_groups_and_sorts_teams_and_players():
    players = [
        _pv("Bravo", "B1", 1.0),
        _pv("Bravo", "B2", 5.0),
        _pv("Alpha", "A1", 3.0),
    ]
    teams = [
        dv.TeamRollup("Alpha", 3.0, 3.0, 1),
        dv.TeamRollup("Bravo", 6.0, 3.0, 2),
    ]
    out = dv.build_draft_value_cache(players, teams)
    assert out["horizon"] == "proj"
    names = [t["team"] for t in out["teams"]]
    # Equal avg_value (3.0, 3.0) -> stable order preserves input (Alpha, Bravo)
    assert names == ["Alpha", "Bravo"]
    bravo = next(t for t in out["teams"] if t["team"] == "Bravo")
    # players sorted by value_proj desc within team
    assert [p["name"] for p in bravo["players"]] == ["B2", "B1"]
    assert bravo["credited_count"] == 2


def test_build_cache_nan_avg_team_sinks():
    players = [_pv("Good", "G", 4.0), _pv("Empty", "E", float("nan"))]
    teams = [
        dv.TeamRollup("Empty", 0.0, float("nan"), 0),
        dv.TeamRollup("Good", 4.0, 4.0, 1),
    ]
    out = dv.build_draft_value_cache(players, teams)
    assert out["teams"][0]["team"] == "Good"
    assert out["teams"][-1]["team"] == "Empty"
    assert out["teams"][-1]["avg_value"] is None  # NaN -> null


def test_build_cache_nonfinite_to_null_and_strict_json():
    players = [_pv("T", "P", float("nan"), skill=float("inf"), luck=float("-inf"))]
    teams = [dv.TeamRollup("T", 0.0, 0.0, 0)]
    out = dv.build_draft_value_cache(players, teams)
    p = out["teams"][0]["players"][0]
    assert p["value_proj"] is None
    assert p["skill"] is None
    assert p["luck"] is None
    # No non-finite float leaks -> strict JSON succeeds.
    _json.dumps(out, allow_nan=False)


def test_build_cache_off_board_flier_nulls_but_finite_value():
    players = [_pv("T", "Flier", 0.0, preseason_var=None, skill=None, luck=None)]
    teams = [dv.TeamRollup("T", 0.0, 0.0, 1)]
    out = dv.build_draft_value_cache(players, teams)
    p = out["teams"][0]["players"][0]
    assert p["preseason_var"] is None
    assert p["skill"] is None
    assert p["luck"] is None
    assert p["value_proj"] == 0.0  # finite, still present


def test_build_cache_field_mapping():
    players = [_pv("T", "P", 3.0, baseline_kind="keeper")]
    teams = [dv.TeamRollup("T", 3.0, 3.0, 1)]
    p = dv.build_draft_value_cache(players, teams)["teams"][0]["players"][0]
    assert "est_var_ytd" not in p  # dropped
    assert p["value_ytd"] == 2.0  # kept
    assert p["kind"] == "keeper"  # baseline_kind -> kind
    assert isinstance(p["player_type"], str)


def test_build_cache_credited_count_may_be_below_player_count():
    # Two rows, one ungradeable (NaN value_proj); rollup credits only 1.
    players = [_pv("T", "Good", 3.0), _pv("T", "NaNrow", float("nan"))]
    teams = [dv.TeamRollup("T", 3.0, 3.0, 1)]
    team = dv.build_draft_value_cache(players, teams)["teams"][0]
    assert team["credited_count"] == 1
    assert len(team["players"]) == 2
    nan_row = next(p for p in team["players"] if p["name"] == "NaNrow")
    assert nan_row["value_proj"] is None


def test_build_cache_two_way_display_name_per_team():
    # Same name, both types, SAME team -> suffixed; identical name solo on
    # ANOTHER team -> no suffix (per-team scope).
    players = [
        _pv("T1", "Shohei Ohtani", 5.0, player_type="hitter"),
        _pv("T1", "Shohei Ohtani", 4.0, player_type="pitcher"),
        _pv("T2", "Shohei Ohtani", 3.0, player_type="hitter"),
    ]
    teams = [
        dv.TeamRollup("T1", 9.0, 4.5, 2),
        dv.TeamRollup("T2", 3.0, 3.0, 1),
    ]
    out = dv.build_draft_value_cache(players, teams)
    t1 = next(t for t in out["teams"] if t["team"] == "T1")
    t2 = next(t for t in out["teams"] if t["team"] == "T2")
    disp_t1 = sorted(p["display_name"] for p in t1["players"])
    assert disp_t1 == ["Shohei Ohtani (H)", "Shohei Ohtani (P)"]
    assert t2["players"][0]["display_name"] == "Shohei Ohtani"  # solo -> no suffix


def test_parse_full_season_lines_keys_by_name_and_mlbam():
    from fantasy_baseball.analysis.draft_value import parse_full_season_lines
    from fantasy_baseball.sgp.rankings import rank_key

    payload = {
        "hitters": [
            {
                "name": "Al Star",
                "mlbam_id": 111,
                "r": 90,
                "hr": 40,
                "rbi": 100,
                "sb": 8,
                "ab": 550,
                "h": 165,
            }
        ],
        "pitchers": [
            {
                "name": "Ace One",
                "mlbam_id": 222,
                "w": 15,
                "k": 200,
                "sv": 0,
                "ip": 190,
                "er": 60,
                "bb": 40,
                "h_allowed": 150,
            }
        ],
    }
    by_mlbam, by_name = parse_full_season_lines(payload)
    assert by_name[rank_key("Al Star", "hitter")]["hr"] == 40
    assert by_name[rank_key("Ace One", "pitcher")]["k"] == 200
    assert (111, "hitter") in by_mlbam


def test_parse_full_season_lines_empty_payload():
    from fantasy_baseball.analysis.draft_value import parse_full_season_lines

    assert parse_full_season_lines({}) == ({}, {})


def test_parse_full_season_lines_namesake_keeps_higher_volume():
    from fantasy_baseball.analysis.draft_value import parse_full_season_lines
    from fantasy_baseball.sgp.rankings import rank_key

    # two "Mason Miller" pitchers, distinct mlbam; by_name must keep the higher-IP one
    payload = {
        "hitters": [],
        "pitchers": [
            {
                "name": "Mason Miller",
                "mlbam_id": 1,
                "ip": 12,
                "k": 20,
                "w": 0,
                "sv": 5,
                "er": 4,
                "bb": 3,
                "h_allowed": 8,
            },
            {
                "name": "Mason Miller",
                "mlbam_id": 2,
                "ip": 180,
                "k": 210,
                "w": 14,
                "sv": 0,
                "er": 60,
                "bb": 40,
                "h_allowed": 150,
            },
        ],
    }
    _by_mlbam, by_name = parse_full_season_lines(payload)
    assert by_name[rank_key("Mason Miller", "pitcher")]["ip"] == 180  # higher-volume wins


def test_projected_par_curve_matches_inline_construction():
    import json
    import math

    from fantasy_baseball.analysis import draft_value as dv
    from fantasy_baseball.config import load_config

    par = dv.projected_par_curve()
    # non-empty and sorted descending (par_for_slot(k) = k-th best drafted VAR)
    assert len(par.drafted_pars) > 0
    assert par.drafted_pars == sorted(par.drafted_pars, reverse=True)

    # cross-check: the standalone helper reproduces the exact same curve the
    # metric builds inline, guarding against setup-sequence drift (anchor before
    # board index, fraction=1.0, keepers passed to _assign_pick_types).
    config = load_config(dv._CONFIG)
    board, _scale = dv.reproduce_draft_day_board(config)
    board = dv._anchor_board_var_to_frozen(board, dv._frozen_var_by_player_id())
    state = json.loads(dv._draft_state_path(config).read_text(encoding="utf-8"))
    picks = dv.reconstruct_draft(config, state=state)
    bindex = dv._board_index(board)
    typed = dv._assign_pick_types(picks, bindex, config.keepers)
    expected = dv.build_par_curve(typed, bindex, fraction=1.0)

    assert par.drafted_pars == expected.drafted_pars
    assert (math.isnan(par.keeper_par) and math.isnan(expected.keeper_par)) or (
        par.keeper_par == expected.keeper_par
    )


# --- #244: a frozen draft per season, readable in both saved formats ---


def _as_dashboard_state(config):
    """The frozen 2026 draft rewritten in the draft dashboard's format, with each
    pick's type taken from the legacy path's own type assignment."""
    board, _scale = dv.reproduce_draft_day_board(config)
    bindex = dv._board_index(board)
    typed = dv._assign_pick_types(dv.reconstruct_draft(config), bindex, config.keepers)

    def entry(pick, ptype, i):
        return {
            "pick_number": pick.slot,
            "round": 0 if pick.is_keeper else (pick.slot - 1) // config.num_teams + 1,
            "team": pick.team,
            "player_id": f"fg{i}::{ptype.value}",
            "player_name": pick.player_name,
            "position": "",
            "timestamp": 0.0,
            "undone": False,
        }

    keepers = [entry(p, t, i) for i, (p, t) in enumerate(typed) if p.is_keeper]
    picks = [entry(p, t, i) for i, (p, t) in enumerate(typed) if not p.is_keeper]
    picks.reverse()  # order on disk must not matter: pick_number decides
    return {"version": 7, "keepers": keepers, "picks": picks, "undo_stack": []}, typed


def test_dashboard_format_reconstructs_the_same_draft_as_the_legacy_format():
    config = dv.load_config(dv._CONFIG)
    state, legacy_typed = _as_dashboard_state(config)

    picks = dv.reconstruct_draft(config, state=state)

    def shape(p):
        return (p.slot, p.team, p.player_name, p.is_keeper)

    assert [shape(p) for p in picks] == [shape(p) for p, _t in legacy_typed]
    assert dv.validate_reconstruction(picks, config=config) == []
    bindex = dv._board_index(dv.reproduce_draft_day_board(config)[0])
    assert dv._assign_pick_types(picks, bindex, config.keepers) == [
        (dataclasses.replace(p, player_type=t), t) for p, t in legacy_typed
    ]


def test_run_draft_value_grades_a_dashboard_draft_like_the_legacy_one(season_kv, monkeypatch):
    legacy_players, legacy_teams = dv.run_draft_value(fraction=0.5)
    state, _typed = _as_dashboard_state(dv.load_config(dv._CONFIG))
    monkeypatch.setattr(dv, "_load_draft_state", lambda _config: state)

    players, teams = dv.run_draft_value(fraction=0.5)

    assert players == legacy_players
    assert teams == legacy_teams


def test_dashboard_pick_keeps_its_own_type_and_skips_undone_picks():
    state = {
        "keepers": [],
        "picks": [
            {"pick_number": 2, "team": "B", "player_id": "9::pitcher", "player_name": "X"},
            {"pick_number": 1, "team": "A", "player_id": "8::hitter", "player_name": "X"},
            {
                "pick_number": 3,
                "team": "C",
                "player_id": "7::hitter",
                "player_name": "Y",
                "undone": True,
            },
        ],
    }
    picks = dv.reconstruct_draft(dv.load_config(dv._CONFIG), state=state)
    assert [(p.slot, p.team, p.player_type) for p in picks] == [
        (1, "A", dv.PlayerType.HITTER),
        (2, "B", dv.PlayerType.PITCHER),
    ]


def test_dashboard_pick_without_a_type_suffix_is_refused():
    state = {
        "keepers": [],
        "picks": [
            {"pick_number": 1, "team": "A", "player_id": "Juan Soto", "player_name": "Juan Soto"}
        ],
    }
    with pytest.raises(ValueError, match="no '::hitter' / '::pitcher' suffix"):
        dv.reconstruct_draft(dv.load_config(dv._CONFIG), state=state)


def test_keeper_type_that_contradicts_its_note_is_refused():
    pick = dv.DraftPick(None, "T", "Shohei Ohtani", True, dv.PlayerType.PITCHER)
    keepers = [{"name": "Shohei Ohtani", "team": "T", "note": "batter only"}]
    with pytest.raises(ValueError, match="batter only"):
        dv._assign_pick_types([pick], {}, keepers)


def test_unrecognized_draft_state_names_both_formats():
    with pytest.raises(ValueError, match=r"'keepers' \+ 'picks'.*'drafted_players'"):
        dv.reconstruct_draft(dv.load_config(dv._CONFIG), state={"version": 1})


def test_a_season_with_no_frozen_draft_says_how_to_freeze_it():
    cfg = dataclasses.replace(dv.load_config(dv._CONFIG), season_year=1999)
    with pytest.raises(FileNotFoundError, match="answer yes"):
        dv._load_draft_state(cfg)


def test_frozen_draft_files_are_not_the_dashboards_working_files():
    """The dashboard rewrites its state and board. If draft-value read the same
    paths, any dashboard or mock-draft session broke it (#244)."""
    import scripts.run_draft_dashboard as dash
    from fantasy_baseball.web.app import DEFAULT_STATE_PATH

    cfg = dv.load_config(dv._CONFIG)
    frozen = {dv._draft_state_path(cfg).resolve(), dv._frozen_board_path(cfg).resolve()}
    assert DEFAULT_STATE_PATH.resolve() not in frozen
    assert dash.BOARD_PATH.resolve() not in frozen


@pytest.fixture
def dashboard_draft(tmp_path, monkeypatch):
    """A finished dashboard draft on disk, and a frozen root under tmp_path."""
    config = dv.load_config(dv._CONFIG)
    state, _typed = _as_dashboard_state(config)
    # As stamped at New Draft; deliberately not the live defaults, so the test
    # proves the freeze records the board's volumes, not today's constants.
    state["board_scale"] = {
        "team_ab": 5400,
        "team_ip": 1350,
        "repl_rates": {"avg": 0.251, "era": 4.4, "whip": 1.31},
    }
    state_path = tmp_path / "live" / "draft_state.json"
    board_path = tmp_path / "live" / "draft_state_board.json"
    state_path.parent.mkdir()
    state_path.write_text(_json.dumps(state), encoding="utf-8")
    board_path.write_text(_json.dumps([{"player_id": "1::hitter", "var": 1.0}]), encoding="utf-8")
    monkeypatch.setattr(dv, "frozen_draft_dir", lambda year: tmp_path / f"frozen_draft_{year}")
    return config, state, state_path, board_path


def test_freeze_draft_copies_state_and_board_and_records_volumes(dashboard_draft):
    config, state, state_path, board_path = dashboard_draft
    dest = dv.freeze_draft(config, state_path, board_path)

    assert _json.loads((dest / "draft_state.json").read_text(encoding="utf-8")) == state
    assert (dest / "draft_state_board.json").read_bytes() == board_path.read_bytes()
    assert dv._draft_day_scale(config) == (5400, 1350, {"avg": 0.251, "era": 4.4, "whip": 1.31})
    assert dv._load_draft_state(config) == state
    assert not dest.with_name(dest.name + ".tmp").exists()
    assert not dest.with_name(dest.name + ".old").exists()


def test_freeze_draft_refuses_a_draft_without_board_volumes(dashboard_draft):
    config, state, state_path, board_path = dashboard_draft
    del state["board_scale"]
    state_path.write_text(_json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="Start it with New Draft"):
        dv.freeze_draft(config, state_path, board_path)


def test_freeze_draft_never_replaces_a_legacy_draft(dashboard_draft):
    config, _state, state_path, board_path = dashboard_draft
    dest = dv.frozen_draft_dir(config.season_year)
    dest.mkdir()
    legacy = dv._REPO_ROOT / "data" / "frozen_draft_2026" / "draft_state.json"
    (dest / "draft_state.json").write_bytes(legacy.read_bytes())
    with pytest.raises(ValueError, match="never replaced"):
        dv.freeze_draft(config, state_path, board_path, force=True)
    assert (dest / "draft_state.json").read_bytes() == legacy.read_bytes()


def test_saved_draft_status_tells_this_draft_from_another(dashboard_draft):
    config, state, state_path, board_path = dashboard_draft
    year = config.season_year
    assert dv.saved_draft_status(year, state) == "none"
    dv.freeze_draft(config, state_path, board_path)
    assert dv.saved_draft_status(year, state) == "same"
    other = {
        **state,
        "picks": [{**state["picks"][0], "player_id": "x::hitter"}, *state["picks"][1:]],
    }
    assert dv.saved_draft_status(year, other) == "different"


def test_freeze_draft_refuses_to_overwrite_without_force(dashboard_draft):
    config, _state, state_path, board_path = dashboard_draft
    dv.freeze_draft(config, state_path, board_path)
    with pytest.raises(FileExistsError):
        dv.freeze_draft(config, state_path, board_path)
    dv.freeze_draft(config, state_path, board_path, force=True)


def test_freeze_draft_refuses_an_unfinished_draft(dashboard_draft):
    config, state, state_path, board_path = dashboard_draft
    state["picks"] = state["picks"][1:]
    state_path.write_text(_json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="non-keeper pick count 199"):
        dv.freeze_draft(config, state_path, board_path)
    assert not dv.frozen_draft_dir(config.season_year).exists()


def test_freeze_draft_refuses_the_legacy_format(dashboard_draft):
    config, _state, state_path, board_path = dashboard_draft
    legacy_path = dv._REPO_ROOT / "data" / "frozen_draft_2026" / "draft_state.json"
    legacy = legacy_path.read_text(encoding="utf-8")
    state_path.write_text(legacy, encoding="utf-8")
    with pytest.raises(ValueError, match="not the draft dashboard's state"):
        dv.freeze_draft(config, state_path, board_path)


def test_board_floors_match_the_floors_every_other_model_uses():
    """#521: the board's replacement levels no longer depend on its own pool, so they
    equal the plain measured floors (what the trajectory board and keeper report use)
    at the same denominators and team volumes."""
    from fantasy_baseball.data.projections import blend_projections
    from fantasy_baseball.data.yahoo_players import load_positions_cache
    from fantasy_baseball.draft.board import build_board_from_frames
    from fantasy_baseball.sgp.player_value import REPLACEMENT_RATES
    from fantasy_baseball.sgp.replacement import (
        league_replacement_levels,
        position_aware_replacement_levels,
    )

    cfg = dv.load_config(dv._CONFIG)
    hitters, pitchers, _ = blend_projections(
        dv._REPO_ROOT / "data" / "projections" / "2026",
        cfg.projection_systems,
        cfg.projection_weights,
    )
    _board, scale = build_board_from_frames(
        hitters, pitchers, load_positions_cache(dv._POSITIONS_JSON)
    )
    assert scale.repl_rates == REPLACEMENT_RATES
    plain = position_aware_replacement_levels(
        scale.denoms, team_ab=scale.team_ab, team_ip=scale.team_ip
    )
    assert scale.replacement_levels == pytest.approx(plain)
    # ...and the live board, at the league's denominators, is the one shared source.
    _board, live = build_board_from_frames(
        hitters, pitchers, load_positions_cache(dv._POSITIONS_JSON), sgp_overrides=cfg.sgp_overrides
    )
    assert live.replacement_levels == pytest.approx(league_replacement_levels(cfg.sgp_overrides))


def test_draft_value_rebuilds_2026_on_its_draft_day_rates():
    """The 2026 board was scored on rates from its own pool, before #521. Draft-value
    must rebuild that scale, or luck (est_proj - frozen preseason VAR) would mix two
    zero points."""
    meta = _json.loads(
        (dv._REPO_ROOT / "data" / "frozen_draft_2026" / "meta.json").read_text(encoding="utf-8")
    )
    _board, scale = dv.reproduce_draft_day_board()
    assert scale.repl_rates == meta["repl_rates"]
    assert scale.repl_rates["era"] == pytest.approx(4.0317, abs=1e-4)
