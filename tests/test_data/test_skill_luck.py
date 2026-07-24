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
