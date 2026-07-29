from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.keepers.cache import fetch_or_cache


def test_miss_fetches_and_writes(tmp_path: Path):
    path = tmp_path / "x.csv"
    df = pd.DataFrame({"a": [1, 2]})
    out = fetch_or_cache(path, lambda: df)
    pd.testing.assert_frame_equal(out, df)
    assert path.exists()


def test_hit_reads_cache_without_fetching(tmp_path: Path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)

    def boom() -> pd.DataFrame:
        raise AssertionError("fetcher must not be called on a cache hit")

    out = fetch_or_cache(path, boom)
    assert list(out["a"]) == [1]


def test_empty_pull_raises_and_writes_nothing(tmp_path: Path):
    path = tmp_path / "x.csv"
    with pytest.raises(RuntimeError):
        fetch_or_cache(path, lambda: pd.DataFrame())
    assert not path.exists()


def test_tolerate_empty_returns_without_writing(tmp_path: Path):
    path = tmp_path / "x.csv"
    empty = pd.DataFrame()
    out = fetch_or_cache(path, lambda: empty, tolerate_empty=True)
    assert out.empty
    assert not path.exists()


def test_version_bump_ignores_the_previous_cache(tmp_path: Path):
    """A fetcher that transforms its response must not keep serving pre-transform
    data. This is how repaired names got silently un-repaired on a later run."""
    path = tmp_path / "pull.csv"
    stale = pd.DataFrame({"Name": ["Acu\xc3\xb1a"]})
    fetch_or_cache(path, lambda: stale, version=1)
    assert path.exists()

    fresh = pd.DataFrame({"Name": ["Acuna"]})
    out = fetch_or_cache(path, lambda: fresh, version=2)
    assert out["Name"].tolist() == ["Acuna"]
    assert (tmp_path / "pull.v2.csv").exists()
    assert path.exists()  # v1 left alone, not clobbered


def test_version_one_keeps_the_bare_filename(tmp_path: Path):
    """Adding a version to an untransformed pull must not orphan its cache."""
    path = tmp_path / "pull.csv"
    fetch_or_cache(path, lambda: pd.DataFrame({"a": [1]}), version=1)
    assert path.exists()
