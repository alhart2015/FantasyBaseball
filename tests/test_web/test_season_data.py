import json
from pathlib import Path

from fantasy_baseball.web.season_data import read_cache, write_cache, read_meta


def test_write_and_read_cache(tmp_path):
    data = {"teams": [{"name": "Hart of the Order", "total": 67}]}
    write_cache("standings", data, cache_dir=tmp_path)
    result = read_cache("standings", cache_dir=tmp_path)
    assert result == data


def test_read_cache_missing_file(tmp_path):
    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None


def test_read_cache_corrupt_json(tmp_path):
    path = tmp_path / "standings.json"
    path.write_text("not json", encoding="utf-8")
    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None


def test_read_meta_missing(tmp_path):
    result = read_meta(cache_dir=tmp_path)
    assert result == {}


def test_write_cache_overwrites(tmp_path):
    write_cache("standings", {"v": 1}, cache_dir=tmp_path)
    write_cache("standings", {"v": 2}, cache_dir=tmp_path)
    result = read_cache("standings", cache_dir=tmp_path)
    assert result == {"v": 2}
