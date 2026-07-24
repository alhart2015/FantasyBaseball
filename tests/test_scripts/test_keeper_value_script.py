import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import keeper_value as script


def _kvr_pyv(name, y0, y1, y2):
    from fantasy_baseball.analysis.keeper_value import KeeperValueResult

    return KeeperValueResult(
        player_id=f"{name}::hitter",
        name=name,
        per_year_var={2026: y0, 2027: y1, 2028: y2},
        total=y0 + y1 + y2,
        flags=[],
        pct_from_out_years=0.5,
        pct_from_saves=None,
    )


def _kvr(i):
    v = 100.0 - i
    return _kvr_pyv(f"p{i}", v, v, v)


def _row_count(out: str) -> int:
    # body rows render as "<mark> <name>..." with fixture names "p<n>"; match that
    # row identity rather than the rank-cell format (a render tweak could change the
    # cell). The title/header lines don't start with a mark + lowercase "p<digit>".
    return sum(bool(re.match(r"[ *] p\d", ln)) for ln in out.splitlines())


def test_render_truncates_to_limit():
    results = [_kvr(i) for i in range(10)]
    out = script.render(results, [0.8], set(), limit=3)
    assert _row_count(out) == 3  # only 3 body rows
    assert "showing top 3 of 10" in out


def test_render_limit_zero_shows_all():
    results = [_kvr(i) for i in range(10)]
    out = script.render(results, [0.8], set(), limit=0)
    assert _row_count(out) == 10
    assert "showing top" not in out


def test_render_ranks_reflect_full_pool_not_truncated_slice():
    # The feature's core invariant: the (#rank) column is leaguewide -- computed over
    # the FULL pool -- even when --limit truncates the printed rows. Two discounts that
    # reorder the pool expose it: "front" is shown (top-2 by the primary/last discount)
    # but ranks LAST (#4) at the other discount. A regression that ranked over the
    # truncated `shown` slice would render "front" as #2 at d=0.60 and fail here.
    results = [
        _kvr_pyv("front", 50, 50, 50),  # top at d=0.90 (out-years weighted), last at d=0.60
        _kvr_pyv("p1", 90, 20, 5),
        _kvr_pyv("p2", 100, 5, 2),
        _kvr_pyv("p3", 95, 10, 3),
    ]
    out = script.render(results, [0.60, 0.90], set(), limit=2)
    front_row = next((ln for ln in out.splitlines() if ln.lstrip().startswith("front")), "")
    assert front_row, "front should be shown (top-2 by the primary discount)"
    assert re.search(r"\(#\s*4\)", front_row)  # full-pool rank at d=0.60, not slice-relative #2


def test_render_orders_by_largest_discount_regardless_of_input_order():
    # Row order (and thus which top-N --limit keeps) must follow the most dynasty-weighted
    # (largest) discount even when --discount is passed descending. "front" is top by
    # d=0.90 but near-last by d=0.60; with limit=2 it must still be shown.
    results = [
        _kvr_pyv("front", 50, 50, 50),
        _kvr_pyv("p1", 90, 20, 5),
        _kvr_pyv("p2", 100, 5, 2),
        _kvr_pyv("p3", 95, 10, 3),
    ]
    out = script.render(results, [0.90, 0.60], set(), limit=2)  # descending on purpose
    assert any(ln.lstrip().startswith("front") for ln in out.splitlines())


def test_render_suppresses_note_when_limit_covers_all():
    # The "(showing top N of M)" note must appear only when truncation actually happens
    # (0 < limit < len). limit == len and limit > len show everything -> no note.
    results = [_kvr(i) for i in range(5)]
    assert "showing top" not in script.render(results, [0.8], set(), limit=5)  # limit == len
    assert "showing top" not in script.render(results, [0.8], set(), limit=9)  # limit > len


def test_parse_args_limit_default_is_100():
    assert script._parse_args([]).limit == 100
    assert script._parse_args(["--limit", "0"]).limit == 0


def test_parse_args_limit_rejects_negative():
    with pytest.raises(SystemExit):
        script._parse_args(["--limit", "-1"])


def test_nonneg_int_validates():
    assert script._nonneg_int("0") == 0
    assert script._nonneg_int("100") == 100
    for bad in ["-1", "3.5", "abc", ""]:
        with pytest.raises(argparse.ArgumentTypeError):
            script._nonneg_int(bad)


def test_discounts_arg_parses_and_validates():
    assert script._discounts_arg("0.6,0.8,0.9") == [0.6, 0.8, 0.9]
    assert script._discounts_arg("0.5") == [0.5]
    assert script._discounts_arg("1.0") == [1.0]  # 1.0 = no discount, allowed
    for bad in ["0", "1.5", "-0.2", "abc", ""]:
        with pytest.raises(argparse.ArgumentTypeError):
            script._discounts_arg(bad)


def test_parse_args_defaults_and_overrides():
    default = script._parse_args([])
    assert default.horizon == 3
    assert default.discount == [0.60, 0.70, 0.80, 0.90]
    custom = script._parse_args(["--horizon", "2", "--discount", "0.7,0.95"])
    assert custom.horizon == 2
    assert custom.discount == [0.7, 0.95]


def test_load_zips_year_missing_raises_with_url(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        script.load_zips_year(tmp_path, 2027)
    assert "fangraphs.com" in str(exc.value)
    assert "2027" in str(exc.value)


def test_load_zips_year_loads_present(tmp_path):
    d = tmp_path / "2027"
    d.mkdir()
    pd.DataFrame(
        [{"Name": "A B", "AB": 500, "H": 150, "HR": 30, "R": 90, "RBI": 95, "SB": 10, "AVG": 0.300}]
    ).to_csv(d / "zips-hitters.csv", index=False)
    # FanGraphs (and ZiPS) exports use "SO" for strikeouts; PITCHING_COLUMN_MAP
    # normalizes SO -> k. A "K" header would NOT be recognized.
    pd.DataFrame(
        [{"Name": "C D", "IP": 180, "W": 14, "SO": 200, "ERA": 3.2, "WHIP": 1.05, "SV": 0}]
    ).to_csv(d / "zips-pitchers.csv", index=False)
    hitters, pitchers = script.load_zips_year(tmp_path, 2027)
    assert not hitters.empty and not pitchers.empty


def test_resolve_candidate_ids_is_collision_safe_by_var():
    # Two same-normalized-name players; find_keeper_match must pick the higher-VAR
    # one, so the highlight resolves to exactly that player_id (never both).
    board = pd.DataFrame(
        [
            {
                "name": "Star Guy",
                "player_id": "star::hitter",
                "name_normalized": "star guy",
                "var": 12.0,
            },
            {
                "name": "Star Guy",
                "player_id": "scrub::hitter",
                "name_normalized": "star guy",
                "var": 1.0,
            },
            {"name": "Other", "player_id": "other::hitter", "name_normalized": "other", "var": 5.0},
        ]
    )
    ids = script.resolve_candidate_ids(board, ["Star Guy"])
    assert ids == {"star::hitter"}  # higher-VAR match only, not the namesake


def test_zips_index_disambiguates_same_name_by_fg_id():
    # Two same-name/same-type hitters distinguished only by fg_id must NOT collapse:
    # each resolves to its own line via lookup_rank(fg_id, ...).
    hitters = pd.DataFrame(
        [
            {"name": "Max Muncy", "fg_id": "111", "hr": 35, "ab": 500},
            {"name": "Max Muncy", "fg_id": "222", "hr": 10, "ab": 300},
        ]
    )
    idx = script.zips_index(hitters, pd.DataFrame())
    indices = {2027: idx}
    a = script._zips_by_year("111", "Max Muncy", "hitter", indices)[2027]
    b = script._zips_by_year("222", "Max Muncy", "hitter", indices)[2027]
    assert a["hr"] == 35 and b["hr"] == 10  # distinct lines, no last-write-wins collapse


def test_zips_by_year_missing_player_is_none():
    idx = script.zips_index(
        pd.DataFrame([{"name": "Someone", "fg_id": "999", "hr": 20, "ab": 400}]), pd.DataFrame()
    )
    got = script._zips_by_year("000", "Nobody Here", "hitter", {2027: idx})
    assert got[2027] is None  # unknown fg_id + unknown name -> None (per_year_var flags it)


def _fake_kv(raw):
    class _KV:
        def get(self, _key):
            return raw

    return _KV()


def test_load_current_lines_fails_loud_when_missing(monkeypatch):
    monkeypatch.setattr(script, "build_explicit_upstash_kv", lambda: _fake_kv(None))
    with pytest.raises(SystemExit):
        script.load_current_full_season_lines()


def test_load_current_lines_fails_loud_when_empty(monkeypatch):
    envelope = {"_meta": {}, "_data": {"hitters": [], "pitchers": []}}
    monkeypatch.setattr(script, "build_explicit_upstash_kv", lambda: _fake_kv(envelope))
    with pytest.raises(SystemExit):
        script.load_current_full_season_lines()


def test_load_current_lines_parses_present_blob(monkeypatch):
    from fantasy_baseball.sgp.rankings import rank_key

    envelope = {
        "_meta": {},
        "_data": {
            "hitters": [{"name": "Al Star", "mlbam_id": 111, "hr": 40, "ab": 550, "h": 165}],
            "pitchers": [],
        },
    }
    monkeypatch.setattr(script, "build_explicit_upstash_kv", lambda: _fake_kv(envelope))
    by_name = script.load_current_full_season_lines()
    assert by_name[rank_key("Al Star", "hitter")]["hr"] == 40


def test_parse_args_anchor_default_is_current():
    assert script._parse_args([]).anchor == "current"
    assert script._parse_args(["--anchor", "preseason"]).anchor == "preseason"


def test_parse_args_anchor_rejects_invalid():
    with pytest.raises(SystemExit):
        script._parse_args(["--anchor", "bogus"])


def test_parse_args_out_year_regression_default_and_bounds():
    assert script._parse_args([]).out_year_regression == 0.6
    assert script._parse_args(["--out-year-regression", "0"]).out_year_regression == 0.0
    with pytest.raises(SystemExit):
        script._parse_args(["--out-year-regression", "1.5"])


def test_unit_float_validates():
    assert script._unit_float("0") == 0.0
    assert script._unit_float("0.6") == 0.6
    assert script._unit_float("1") == 1.0
    for bad in ["-0.1", "1.5", "abc", ""]:
        with pytest.raises(argparse.ArgumentTypeError):
            script._unit_float(bad)


def test_parse_args_pt_heal_cap_default_and_bounds():
    assert script._parse_args([]).pt_heal_cap == 2.0
    assert script._parse_args(["--pt-heal-cap", "1"]).pt_heal_cap == 1.0
    with pytest.raises(SystemExit):
        script._parse_args(["--pt-heal-cap", "0.5"])


def test_min_one_float_validates():
    assert script._min_one_float("1") == 1.0
    assert script._min_one_float("2.5") == 2.5
    for bad in ["0.9", "0", "-1", "abc", ""]:
        with pytest.raises(argparse.ArgumentTypeError):
            script._min_one_float(bad)
