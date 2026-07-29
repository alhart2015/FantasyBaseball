import os
from datetime import datetime, timedelta
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
    fetch_or_cache(path, lambda: pd.DataFrame({"Name": ["mojibake"]}))
    assert path.exists()

    out = fetch_or_cache(path, lambda: pd.DataFrame({"Name": ["repaired"]}), version=2)
    assert out["Name"].tolist() == ["repaired"]
    assert (tmp_path / "pull.v2.csv").exists()
    assert path.exists()  # the unversioned cache is left alone, not clobbered


def test_no_version_keeps_the_bare_filename(tmp_path: Path):
    """Declaring no version must not orphan an existing unversioned cache."""
    path = tmp_path / "pull.csv"
    fetch_or_cache(path, lambda: pd.DataFrame({"a": [1]}))
    assert path.exists()
    assert not list(tmp_path.glob("*.v*.csv"))


def test_cache_older_than_max_age_is_refetched(tmp_path: Path):
    """Season-to-date pulls go stale daily and nothing about that announces
    itself -- the version int cannot see this failure mode at all."""
    path = tmp_path / "pull.csv"
    fetch_or_cache(path, lambda: pd.DataFrame({"ip": [10]}))
    old = (datetime.now() - timedelta(days=3)).timestamp()
    os.utime(path, (old, old))

    out = fetch_or_cache(path, lambda: pd.DataFrame({"ip": [40]}), max_age=timedelta(days=1))
    assert out["ip"].tolist() == [40]


def test_cache_within_max_age_is_served(tmp_path: Path):
    path = tmp_path / "pull.csv"
    fetch_or_cache(path, lambda: pd.DataFrame({"ip": [10]}))

    def boom() -> pd.DataFrame:
        raise AssertionError("should not refetch a fresh cache")

    out = fetch_or_cache(path, boom, max_age=timedelta(days=1))
    assert out["ip"].tolist() == [10]


def test_max_age_zero_always_refetches(tmp_path: Path):
    """--refresh passes max_age=0. st_mtime resolves finer than time.time()'s
    tick, so a just-written cache can compute a negative age and be served."""
    path = tmp_path / "pull.csv"
    fetch_or_cache(path, lambda: pd.DataFrame({"a": [1]}))
    out = fetch_or_cache(path, lambda: pd.DataFrame({"a": [2]}), max_age=timedelta(0))
    assert out["a"].tolist() == [2]
