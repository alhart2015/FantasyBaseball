"""Unit tests for the pure pricing helpers in the keeper-rankings script.

`_slots_for` exists because a pitcher was being priced against a hitter floor.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.positions import HITTER_ELIGIBLE, Position
from fantasy_baseball.sgp.var import calculate_var
from scripts import keeper_rankings as module
from scripts.keeper_rankings import FALLBACK_POS, _dedupe_two_way, _slots_for

POSITIONS = {
    "shohei ohtani": ["UTIL"],
    "tarik skubal": ["P"],
    "ivan herrera": ["C", "UTIL"],
    "otto lopez": ["2B", "SS", "IF", "UTIL"],
    "someone hurt": ["OF", "IL"],
}


def test_a_util_only_pitcher_is_still_priced_as_a_pitcher():
    """Yahoo lists Ohtani as UTIL. In the pitcher pool that sent his pitching
    projection to the UTIL floor via calculate_var's hitter branch."""
    assert _slots_for(POSITIONS, "Shohei Ohtani", PlayerType.PITCHER) == FALLBACK_POS["pitcher"]


def test_a_pitcher_only_player_is_not_priced_as_a_hitter():
    assert _slots_for(POSITIONS, "Tarik Skubal", PlayerType.HITTER) == FALLBACK_POS["hitter"]


def test_a_hitter_keeps_only_his_batting_slots():
    assert _slots_for(POSITIONS, "Otto Lopez", PlayerType.HITTER) == ["2B", "SS", "IF", "UTIL"]


def test_a_bench_or_il_token_is_not_a_position_to_price_against():
    """An allowlist, not a denylist: IL is not in HITTER_ELIGIBLE."""
    assert _slots_for(POSITIONS, "Someone Hurt", PlayerType.HITTER) == ["OF"]


def test_an_unknown_player_falls_back_to_the_deepest_floor():
    assert _slots_for(POSITIONS, "Nobody At All", PlayerType.HITTER) == FALLBACK_POS["hitter"]


def _board(ids, names, proj_var):
    return pd.DataFrame({"name": names, "proj_var": proj_var}, index=pd.Index(ids, name="mlbam_id"))


def test_a_two_way_player_collapses_to_his_better_side():
    board = _board([660271, 660271, 592450], ["Ohtani", "Ohtani", "Judge"], [11.8, 7.8, 10.0])
    out = _dedupe_two_way(board)
    assert len(out) == 2
    assert out.loc[out["name"] == "Ohtani", "proj_var"].tolist() == [11.8]


def test_two_different_players_sharing_a_name_both_survive():
    """2022 had two Will Smiths, two Diego Castillos and two Luis Garcias. A
    name-keyed drop deletes a real rival, and probability_top_n then spreads the
    same slot mass over fewer people -- inflating everyone while the sum-to-slots
    check still passes."""
    board = _board([519293, 669257], ["Will Smith", "Will Smith"], [9.0, 8.0])
    assert len(_dedupe_two_way(board)) == 2


def test_dedupe_leaves_a_board_with_no_duplicates_alone():
    board = _board([1, 2, 3], ["A", "B", "C"], [3.0, 2.0, 1.0])
    assert _dedupe_two_way(board)["name"].tolist() == ["A", "B", "C"]


def test_dedupe_refuses_a_frame_that_lost_its_id_index():
    """The dedupe's correctness rests on the index being mlbam_id, and a synthetic
    index passes the tests above just as well -- so a reset_index upstream would
    silently restore the double-count with everything green."""
    board = pd.DataFrame({"name": ["A", "A"], "proj_var": [2.0, 1.0]})
    with pytest.raises(ValueError, match="mlbam_id"):
        _dedupe_two_way(board)


def test_every_real_hitter_slot_has_a_floor():
    """Drop or rename a key in `keepers.scarcity.NATIVE_CREDITS` and
    `calculate_var` silently falls back for that slot rather than raising, so every
    player eligible there is quietly repriced against UTIL with the suite green.
    DH and IF are excluded deliberately: they are Yahoo aggregates, not scarcity
    positions, and have no floor by design."""
    _, floors = module.pricing_table()
    real = set(HITTER_ELIGIBLE) - {Position.DH, Position.IF}
    missing = sorted(str(slot) for slot in real - set(floors))
    assert not missing, f"_slots_for can emit {missing}, which no floor prices"
    assert set(FALLBACK_POS["hitter"]) <= set(floors)


def test_an_aggregate_only_slot_is_charged_the_deepest_floor_not_zero():
    """IF and DH have no floor, so they must fall through to UTIL -- the harshest
    hitter floor -- rather than scoring as a free 0.0 credit, which would hand an
    unmapped slot the best price on the board."""
    _, floors = module.pricing_table()
    var, pos = calculate_var(
        pd.Series({"total_sgp": 0.0, "positions": [Position.IF], "ip": 0.0}), floors, True
    )
    assert pos == "UTIL"
    assert var == pytest.approx(-floors["UTIL"])
    assert var < 0.0


class _FakeKV:
    """Stands in for the live KV, keyed the way `redis_key` spells it."""

    def __init__(self, values):
        self.values = values
        self.asked = []

    def get(self, key):
        self.asked.append(key)
        return self.values.get(key)


def _fake_kv(monkeypatch, values):
    kv = _FakeKV(values)
    monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", lambda: kv)
    return kv


def test_the_league_loader_unions_my_roster_with_the_opponents(monkeypatch):
    """`cache:roster` holds only my team and `cache:opp_rosters` only the other
    nine, so a league report that read either alone would silently drop a team.
    Also pins the KEYS: reading one wrong would return None and lose a roster
    without raising."""
    kv = _fake_kv(
        monkeypatch,
        {
            "cache:roster": [{"name": "Juan Soto", "player_type": "hitter"}],
            "cache:opp_rosters": {"Rivals": [{"name": "Bobby Witt Jr.", "player_type": "hitter"}]},
        },
    )
    rosters = module.load_league_rosters("Mine")
    assert rosters == {"Rivals": [("bobby witt jr.", "hitter")], "Mine": [("juan soto", "hitter")]}
    assert set(kv.asked) == {"cache:roster", "cache:opp_rosters"}


def test_an_unreachable_kv_yields_no_league_rather_than_raising(monkeypatch):
    """The rest of the script runs offline; only this report needs the network, so
    a dead KV has to degrade to an empty league instead of taking the run down."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", boom)
    assert module.load_league_rosters("Mine") == {}


def test_the_data_envelope_is_unwrapped_but_a_bare_payload_survives(monkeypatch):
    """A cache value arrives as a JSON string, a dict wrapped in `_data`, or the
    bare value. All three have to reach the caller as the same thing. A CORRUPT blob
    degrades to None (json.loads is inside the try), not a crashed report."""
    for stored, expected in (
        ('{"_data": [{"name": "A"}]}', [{"name": "A"}]),
        ({"_data": [{"name": "A"}]}, [{"name": "A"}]),
        ([{"name": "A"}], [{"name": "A"}]),
        ({"Team": [{"name": "A"}]}, {"Team": [{"name": "A"}]}),
        (None, None),
        ("", None),
        ('{"_data": [1, 2,', None),  # truncated / invalid JSON -> degrade, do not crash
    ):
        _fake_kv(monkeypatch, {"cache:roster": stored})
        assert module._kv_payload(module.CacheKey.ROSTER) == expected


def test_roster_entries_disambiguate_by_type_and_keep_duplicate_spots():
    """Entries key on (normalized_name, player_type) -- names accent-fold to join the board,
    the type distinguishes a hitter from a same-named pitcher, and it stays a LIST so two
    same-name same-type spots (two Luis Garcias) are not collapsed (which made unscored
    go negative). Blank/malformed entries are skipped; a non-list yields []."""
    assert module._roster_entries([{"name": "Julio Rodríguez", "player_type": "hitter"}]) == [
        ("julio rodriguez", "hitter")
    ]
    # same normalized name, different types -> two distinct keys
    assert module._roster_entries(
        [
            {"name": "Will Smith", "player_type": "hitter"},
            {"name": "Will Smith", "player_type": "pitcher"},
        ]
    ) == [("will smith", "hitter"), ("will smith", "pitcher")]
    # two same-name same-type spots survive as duplicates (list, not set)
    assert module._roster_entries(
        [
            {"name": "Luis Garcia", "player_type": "pitcher"},
            {"name": "Luis Garcia", "player_type": "pitcher"},
        ]
    ) == [("luis garcia", "pitcher"), ("luis garcia", "pitcher")]
    assert module._roster_entries([{"name": ""}, {}]) == []
    assert module._roster_entries("not a list") == []


def test_build_fails_loud_if_the_credits_table_gains_sp_rp_keys():
    """build passes calculate_var no role_ip, so if the credits ever split P into SP/RP,
    role_from_ip(0.0)='RP' would silently price every starter (incl a 200-IP ace) at the
    reliever floor. Fail loud -- and BEFORE the expensive projected build, so it is fast."""
    with pytest.raises(ValueError, match="role_ip"):
        module.build(2026, "hitter", {}, {}, pricing=({}, {"SP": 1.0, "RP": 2.0}))


def test_denoms_key_is_order_stable_and_distinguishes_denominators():
    """The out-year SGP cache keys on denoms because the memoized series is scored THROUGH
    them; two denominator sets must not collide, and dict order must not matter."""
    a = module._denoms_key({"R": 30.0, "HR": 12.0})
    assert a == module._denoms_key({"HR": 12.0, "R": 30.0})  # order-stable
    assert a != module._denoms_key({"R": 31.0, "HR": 12.0})  # distinguishes a changed value


def test_league_report_refuses_a_partial_league_rather_than_mislead(monkeypatch, capsys):
    """One present KV blob unions to a few teams; a 'LEAGUE KEEPER BOARD' header over a
    partial league is worse than none. Refuse when fewer than num_teams rosters loaded --
    before the expensive board build."""
    monkeypatch.setattr(
        module, "load_config", lambda _p: SimpleNamespace(team_name="Me", num_teams=10)
    )
    monkeypatch.setattr(module, "load_league_rosters", lambda _t: {"Me": [("juan soto", "hitter")]})
    assert module.league_report(2026, {}, {}, 3, 20) == 1
    assert "partial league" in capsys.readouterr().out


def _synth_board(rows):
    """rows: (mlbam_id, name, kind, proj_var). Fills the columns build() adds."""
    df = pd.DataFrame(
        {
            "name": [r[1] for r in rows],
            "kind": [r[2] for r in rows],
            "proj_var": [r[3] for r in rows],
            "proj_sgp": [r[3] + 1.0 for r in rows],
            "sd": [2.0] * len(rows),
            "pos": ["C"] * len(rows),
            "age": [27] * len(rows),
            "keeper_of": [""] * len(rows),
        },
        index=pd.Index([r[0] for r in rows], name="mlbam_id"),
    )
    return df.sort_values("proj_var", ascending=False)


def test_board_row_keys_join_a_row_to_a_roster_entry_by_name_and_type():
    """The board carries name + kind (a PlayerType) and no roster id, so a row joins to a
    roster entry on (normalized_name, str(kind)) -- accent-folded, and typed so a hitter
    and a same-named pitcher are distinct. #282."""
    board = _synth_board(
        [(1, "Julio Rodríguez", PlayerType.HITTER, 9.0), (2, "Will Smith", PlayerType.PITCHER, 8.0)]
    )
    assert set(module._board_row_keys(board)) == {
        ("julio rodriguez", "hitter"),
        ("will smith", "pitcher"),
    }


def test_roster_report_scores_only_my_side_of_a_shared_name(monkeypatch, capsys):
    """I roster the hitter Will Smith, not the reliever. The (name, type) filter scores only
    my hitter -- a bare-name filter would pull in the reliever I don't own and dilute every
    P(keep). #282."""
    board = _synth_board(
        [(1, "Will Smith", PlayerType.HITTER, 10.0), (2, "Will Smith", PlayerType.PITCHER, 9.0)]
    )
    monkeypatch.setattr(module, "_scored_board", lambda *a, **k: board)
    monkeypatch.setattr(module, "pricing_table", lambda: ({}, {}))
    monkeypatch.setattr(module, "load_roster_keys", lambda: [("will smith", "hitter")])
    assert module.roster_report(2026, {}, {}, 3) == 0
    assert "1 scoreable players" in capsys.readouterr().out  # the hitter only, not the reliever


def test_league_report_attributes_a_shared_name_and_keeps_a_split_two_way(monkeypatch, capsys):
    """Two teams roster the same normalized name (hitter on A, pitcher on B); the (name,
    type) owner map credits each its own instead of handing both to whoever iterates last.
    A two-way player split across teams (same mlbam, two rows) keeps BOTH rows because the
    dedupe is per-team, not league-wide -- so all 4 rows survive (the old league-wide dedupe
    would have collapsed the two-way pair to 3 before ownership). #282."""
    board = _synth_board(
        [
            (1, "Will Smith", PlayerType.HITTER, 10.0),
            (2, "Will Smith", PlayerType.PITCHER, 9.0),
            (660271, "Shohei Ohtani", PlayerType.HITTER, 12.0),
            (660271, "Shohei Ohtani", PlayerType.PITCHER, 11.0),
        ]
    )
    monkeypatch.setattr(module, "_scored_board", lambda *a, **k: board)
    monkeypatch.setattr(module, "pricing_table", lambda: ({}, {}))
    monkeypatch.setattr(
        module, "load_config", lambda _p: SimpleNamespace(team_name="A", num_teams=2)
    )
    monkeypatch.setattr(
        module,
        "load_league_rosters",
        lambda _t: {
            "A": [("will smith", "hitter"), ("shohei ohtani", "hitter")],
            "B": [("will smith", "pitcher"), ("shohei ohtani", "pitcher")],
        },
    )
    assert module.league_report(2026, {}, {}, 3, 20) == 0
    assert "of 4 scoreable rostered players" in capsys.readouterr().out


def test_qualified_families_emits_pt_and_batted_ball_columns():
    """The pt/batted_ball column wiring the backtest and board both read. Runs on a
    synthetic frame, so it pins the family computation without the skills cache."""
    frame = pd.DataFrame(
        {
            "sgp": [20.0, 5.0, 12.0],
            "age": [25.0, 31.0, 28.0],
            "pt": [600.0, 550.0, 300.0],  # all above the 250 hitter floor
            "avg": [0.310, 0.240, 0.250],
            "xba": [0.250, 0.245, 0.280],  # player 0 overperformed (+.060), player 2 under
            "barrel_pct": [12.0, 6.0, 9.0],
            "barrel_pa_pct": [5.0, 3.0, 4.0],
            "xwoba": [0.380, 0.300, 0.330],
            "wrc_plus": [150.0, 95.0, 115.0],
        },
        index=pd.Index([1, 2, 3], name="mlbam_id"),
    )
    out = module._qualified_families(frame, "hitter")
    assert {"pt_pct", "batted_ball_pct"} <= set(out.columns)
    # Player 0 overperformed his xBA the most, so he tops batted_ball; player 2 is last.
    assert out["batted_ball_pct"].idxmax() == 1
    assert out["batted_ball_pct"].idxmin() == 3
    # pt is monotone in PA within the pool.
    assert out.loc[1, "pt_pct"] == pytest.approx(1.0)


def _pct_frame(**all_nan):
    """A qualified-like frame with every family `_pct` column present and non-NaN,
    setting the named families all-NaN (e.g. `_pct_frame(future=True)`)."""
    families = ("skill", "luck", "batted_ball", "future", "age")
    return pd.DataFrame(
        {
            f"{fam}_pct": [float("nan"), float("nan")] if all_nan.get(fam) else [0.3, 0.7]
            for fam in families
        }
    )


def test_an_all_nan_mandatory_future_fails_loud_not_silently_dropped():
    """The live board runs composite strict=False, so an all-NaN family is silently
    dropped. This guard makes the MANDATORY `future` family's absence a loud, ZiPS-
    source-pointing failure instead of a valid-looking board that ignores it."""
    with pytest.raises(FileNotFoundError, match="future"):
        module._require_mandatory_families(_pct_frame(future=True), None, "hitter", 2026)


def test_an_all_nan_mandatory_batted_ball_fails_loud_not_silently_dropped():
    """Same trap as `future`: an xba/Savant outage makes batted_ball all-NaN, which a
    silent drop would revert to the pre-#277 blend that over-sells lucky bats. Fail
    loud, naming the Savant/skills pull, instead of shipping the degraded board."""
    with pytest.raises(ValueError, match="batted-ball"):
        module._require_mandatory_families(_pct_frame(batted_ball=True), None, "hitter", 2026)


def test_an_all_nan_low_weight_family_still_fails_loud_not_silently_dropped():
    """Every shipped family is guarded, not just future/batted_ball: `age` carries a
    small weight but still coerces to all-NaN on a malformed BBRef Age column, and a
    silent drop would ship a board ranked on the wrong weights. Generic message names
    the offending family."""
    with pytest.raises(ValueError, match="age"):
        module._require_mandatory_families(_pct_frame(age=True), None, "hitter", 2026)


def test_a_candidate_board_that_omits_batted_ball_is_not_guarded_for_it():
    """baseline/A candidates legitimately blend without batted_ball, so its all-NaN
    column must not fail their boards -- the guard fires only for families the active
    `family_order` actually uses."""
    module._require_mandatory_families(
        _pct_frame(batted_ball=True), ("skill", "luck", "future", "age"), "hitter", 2026
    )


def test_present_mandatory_families_pass_the_guard():
    module._require_mandatory_families(_pct_frame(), None, "hitter", 2026)


def test_the_family_guard_no_ops_on_an_empty_pool():
    """An empty pool is not a family outage (`.isna().all()` is vacuously True on empty
    columns). The guard must no-op, not raise about its first family, so intentional
    empty pools -- an early-season board, a `--study` truncation sub-pool -- flow through
    as an empty board rather than crashing the study/backtest."""
    module._require_mandatory_families(_pct_frame().iloc[0:0], None, "hitter", 2026)


def test_qualified_families_returns_empty_for_a_below_floor_pool_not_raises():
    """A pool where nobody clears MIN_PT flows through as an empty frame, not a raise:
    `--study` builds intentional sub-pools (`nlargest(0)`) it expects to skip, so raising
    here would defeat `_print_truncation`'s `if shift.empty: continue`. The columns are
    present (as `_observed` always supplies them) -- only the rows are filtered out."""
    below_floor = pd.DataFrame(
        {
            "pt": [10.0],  # below MIN_PT hitter (250)
            "sgp": [5.0],
            "age": [27.0],
            "avg": [0.250],
            "xba": [0.245],
            "barrel_pct": [8.0],
            "barrel_pa_pct": [4.0],
            "xwoba": [0.320],
            "wrc_plus": [100.0],
        },
        index=pd.Index([1], name="mlbam_id"),
    )
    assert module._qualified_families(below_floor, "hitter").empty


def test_an_empty_live_board_fails_loud_not_silently(capsys):
    """The shared math tolerates empty pools (so diagnostics can build empty sub-pools),
    but a LIVE board that came back empty -- a broken actuals/skills join, or a season too
    early for MIN_PT -- must fail rather than write a silent empty CSV read as 'no keepers
    qualify'. A populated board passes through untouched."""
    assert module._fail_if_empty_board(pd.DataFrame(), 2026) is True
    assert "keeper board is empty" in capsys.readouterr().out
    assert (
        module._fail_if_empty_board(pd.DataFrame({"name": ["A"], "proj_var": [5.0]}), 2026) is False
    )


def test_a_partial_live_board_missing_a_whole_pool_fails_loud(capsys):
    """The combined roster/league board merges both pools, so a one-pool join break (drifted
    mlbam ids on just one side) leaves the merged board non-empty. Without the `pools` check
    it would silently render a keeper board missing an entire pool -- and unlike the too-early
    case, a join break hits mid-season, when keeper decisions are made."""
    hitters_only = pd.DataFrame({"name": ["A"], "proj_var": [5.0], "kind": [PlayerType.HITTER]})
    assert module._fail_if_empty_board(hitters_only, 2026, module.POOLS) is True
    assert "pool is empty" in capsys.readouterr().out
    both = pd.DataFrame(
        {
            "name": ["A", "B"],
            "proj_var": [5.0, 4.0],
            "kind": [PlayerType.HITTER, PlayerType.PITCHER],
        }
    )
    assert module._fail_if_empty_board(both, 2026, module.POOLS) is False


def test_composite_pct_rejects_a_typod_family_order_with_a_clear_message():
    """The production wrapper materializes `{family}_pct` columns, so a typo'd family_order
    must be caught BEFORE that -- else `frame['sklll_pct']` raises an opaque pandas KeyError
    instead of the actionable 'unknown families' message. composite's own guard runs after
    the column access, too late for this path, so composite_pct checks first."""
    frame = pd.DataFrame({"skill_pct": [0.3, 0.7]})
    # Match "unknown families", NOT "sklll": the latter also matches the opaque pandas
    # `KeyError: 'sklll_pct'` from `_family_columns`, so it would stay green even if the
    # pre-check were deleted -- failing to pin the actionable-message behavior it guards.
    with pytest.raises(KeyError, match="unknown families"):
        module.composite_pct(frame, "hitter", weights=(1.0, 1.0), family_order=("skill", "sklll"))


def test_projected_actually_calls_the_mandatory_family_guard(monkeypatch):
    """The guard only helps if `projected` invokes it. Drive `projected` with a
    batted_ball column that comes back all-NaN (blank avg) and confirm it raises, so a
    later edit that deletes or moves the call cannot silently reintroduce the pre-#277
    over-selling bug with the isolated helper tests still green."""
    observed = pd.DataFrame(
        {
            "sgp": [20.0, 5.0, 12.0],
            "age": [25.0, 31.0, 28.0],
            "pt": [600.0, 550.0, 300.0],
            "avg": [float("nan")] * 3,  # -> batted_ball all-NaN
            "xba": [0.250, 0.245, 0.280],
            "barrel_pct": [12.0, 6.0, 9.0],
            "barrel_pa_pct": [5.0, 3.0, 4.0],
            "xwoba": [0.380, 0.300, 0.330],
            "wrc_plus": [150.0, 95.0, 115.0],
        },
        index=pd.Index([1, 2, 3], name="mlbam_id"),
    )
    monkeypatch.setattr(module, "_observed", lambda year, kind, denoms: observed)
    monkeypatch.setattr(
        module,
        "zips_out_year_sgp",
        lambda year, kind, denoms: pd.Series([10.0, 8.0, 6.0], index=[1, 2, 3]),
    )
    with pytest.raises(ValueError, match="batted-ball"):
        module.projected(2026, "hitter", {})


def test_projected_renders_an_empty_board_for_an_empty_pool(monkeypatch):
    """An empty pool (nobody cleared MIN_PT -- e.g. `--backtest`'s 2026 watchlist board
    run before anyone reaches 250 PA) must render an empty board, not crash. Before
    #277's all-NaN drop `composite` returned empty for empty; this pins that base behavior
    is restored end to end, so an empty current pool cannot abort `--backtest`/`--study`."""
    below_floor = pd.DataFrame(
        {
            "sgp": [5.0],
            "age": [27.0],
            "pt": [10.0],  # below MIN_PT hitter (250) -> empty after the floor
            "avg": [0.250],
            "xba": [0.245],
            "barrel_pct": [8.0],
            "barrel_pa_pct": [4.0],
            "xwoba": [0.320],
            "wrc_plus": [100.0],
        },
        index=pd.Index([1], name="mlbam_id"),
    )
    monkeypatch.setattr(module, "_observed", lambda year, kind, denoms: below_floor)
    monkeypatch.setattr(
        module, "zips_out_year_sgp", lambda year, kind, denoms: pd.Series([10.0], index=[1])
    )
    assert module.projected(2026, "hitter", {}).empty


def test_season_value_keeps_a_blank_rate_as_nan_not_a_phantom_zero(monkeypatch):
    """`batted_ball` differences avg against xba, so a blank BA must reach the family as
    NaN (mean-filled to neutral), NOT the fillna(0.0) `lines` value that would read as a
    fake .000 -- a phantom 'unlucky' extreme mis-ranking the player to the bottom."""
    raw = pd.DataFrame(
        {
            "mlbID": [1, 2],
            "PA": [600, 550],
            "R": [90, 70],
            "HR": [30, 12],
            "RBI": [95, 60],
            "SB": [10, 4],
            "AB": [540, 500],
            "BA": ["0.280", ""],  # player 2's rate is blank
            "Age": [27, 31],
        }
    )
    monkeypatch.setattr(module, "_raw", lambda year, table: raw)
    monkeypatch.setattr(module, "_sgp", lambda lines, denoms: pd.Series(0.0, index=lines.index))
    out = module.season_value(2026, "hitter", {})
    assert out.loc[1, "avg"] == pytest.approx(0.280)
    assert pd.isna(out.loc[2, "avg"])  # blank -> NaN, not 0.0
