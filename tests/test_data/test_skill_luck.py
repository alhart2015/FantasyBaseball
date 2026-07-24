from pathlib import Path

import pandas as pd

from fantasy_baseball.data import skill_luck


def _fake_register():
    return pd.DataFrame(
        {
            "key_mlbam": [665742, 700, float("nan")],  # Soto, junk, unmatched
            "key_fangraphs": [20123, float("nan"), 55],
            "name_first": ["Juan", "No", "No"],
            "name_last": ["Soto", "Fg", "Mlbam"],
        }
    )


def test_load_id_map_drops_unmatched_and_caches(tmp_path: Path):
    m = skill_luck.load_id_map(tmp_path, fetcher=_fake_register)
    # only the fully-identified row survives
    assert list(m["key_mlbam"]) == [665742]
    assert list(m["key_fangraphs"]) == [20123]

    # second call with a fetcher that would raise must hit the cache, not the network
    def _boom():
        raise AssertionError("must not refetch")

    m2 = skill_luck.load_id_map(tmp_path, fetcher=_boom)
    assert list(m2["key_mlbam"]) == [665742]


def test_fetch_or_cache_refuses_empty_and_reuses(tmp_path: Path):
    calls = {"n": 0}
    good = pd.DataFrame({"a": [1, 2]})

    def _fetch_good():
        calls["n"] += 1
        return good

    p = tmp_path / "x.csv"
    out = skill_luck.fetch_or_cache(p, _fetch_good)
    assert list(out["a"]) == [1, 2] and calls["n"] == 1
    # cache hit: fetcher not called again
    skill_luck.fetch_or_cache(p, _fetch_good)
    assert calls["n"] == 1
    # empty fetch to a fresh path raises and writes nothing
    import pytest

    q = tmp_path / "y.csv"
    with pytest.raises(RuntimeError):
        skill_luck.fetch_or_cache(q, lambda: pd.DataFrame())
    assert not q.exists()


def test_load_fg_hitters_renames_and_fails_loud_on_schema_drift(tmp_path: Path):
    src = pd.DataFrame(
        {
            "IDfg": [20123],
            "Age": [26.0],
            "K%": [0.20],
            "BB%": [0.15],
            "BABIP": [0.34],
            "HR/FB": [0.18],
            "Contact%": [0.78],
            "PA": [600],
        }
    )
    out = skill_luck.load_fg_hitters(tmp_path, 2024, fetcher=lambda: src)
    row = out.iloc[0]
    assert row["key_fangraphs"] == 20123 and row["k_pct"] == 0.20 and row["age"] == 26.0
    import pytest

    with pytest.raises(KeyError):
        skill_luck.load_fg_hitters(tmp_path, 2025, fetcher=lambda: pd.DataFrame({"IDfg": [1]}))


def test_build_hitter_skill_luck_joins_and_reports_coverage(tmp_path: Path):
    from fantasy_baseball.data import skill_luck

    id_map = pd.DataFrame({"key_mlbam": [665742, 700], "key_fangraphs": [20123, 800]})
    fg = pd.DataFrame(
        {
            "IDfg": [20123, 800],
            "Age": [26.0, 30.0],
            "K%": [0.2, 0.3],
            "BB%": [0.15, 0.05],
            "BABIP": [0.34, 0.28],
            "HR/FB": [0.2, 0.1],
            "Contact%": [0.78, 0.7],
            "PA": [600, 550],
        }
    )
    sc = pd.DataFrame(
        {
            "player_id": [665742],
            "woba": [0.400],
            "est_woba": [0.360],
            "ba": [0.300],
            "est_ba": [0.270],
            "slg": [0.600],
            "est_slg": [0.520],
        }
    )
    brl = pd.DataFrame({"player_id": [665742], "brl_percent": [14.0]})
    rows, cov = skill_luck.build_hitter_skill_luck(
        tmp_path,
        2024,
        fetchers={
            "id_map": lambda: id_map,
            "fg": lambda: fg,
            "sc_x": lambda: sc,
            "sc_brl": lambda: brl,
        },
    )
    soto = rows[20123]
    assert soto.mlbam == 665742 and soto.woba == 0.400 and soto.xwoba == 0.360
    assert soto.barrel_pct == 0.14 and soto.k_pct == 0.2
    # fg 800 has no statcast -> present but xStats None
    assert rows[800].xwoba is None and rows[800].woba is None
    assert cov.matched == 1 and cov.fg_only == 1


def test_build_pitcher_skill_luck_joins_and_reports_coverage(tmp_path: Path):
    from fantasy_baseball.data import skill_luck

    id_map = pd.DataFrame({"key_mlbam": [543037, 900], "key_fangraphs": [10028, 950]})
    fg = pd.DataFrame(
        {
            "IDfg": [10028, 950],
            "Age": [28.0, 32.0],
            "K%": [0.30, 0.22],
            "BB%": [0.07, 0.09],
            "IP": [180.0, 150.0],
        }
    )
    sc = pd.DataFrame({"player_id": [543037], "woba": [0.290], "est_woba": [0.300]})
    rows, cov = skill_luck.build_pitcher_skill_luck(
        tmp_path,
        2024,
        fetchers={"id_map": lambda: id_map, "fg": lambda: fg, "sc_x": lambda: sc},
    )
    ace = rows[10028]
    assert ace.mlbam == 543037 and ace.woba == 0.290 and ace.xwoba == 0.300
    assert ace.k_pct == 0.30 and ace.bb_pct == 0.07 and ace.ip == 180.0
    assert ace.pa == 0.0
    assert ace.barrel_pct is None and ace.xslg is None and ace.slg is None
    assert ace.xba is None and ace.ba is None and ace.babip is None
    # fg 950 has no statcast -> present but xwoba/woba None
    assert rows[950].xwoba is None and rows[950].woba is None
    assert cov.matched == 1 and cov.fg_only == 1
