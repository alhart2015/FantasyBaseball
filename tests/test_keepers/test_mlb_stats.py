from pathlib import Path
from typing import Any

import pandas as pd

from fantasy_baseball.keepers import mlb_stats
from fantasy_baseball.keepers.mlb_stats import _fetch_mlb_season, fetch_mlb_season


class _FakeResp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


def _page(splits: list[dict[str, Any]]) -> dict[str, Any]:
    return {"stats": [{"splits": splits}]}


def test_fetch_mlb_season_passthrough_and_caches(tmp_path: Path):
    raw = pd.DataFrame({"player.id": [1], "player.fullName": ["A"], "stat.homeRuns": [10]})
    out = fetch_mlb_season(tmp_path, 2024, "hitting", fetcher=lambda: raw)
    pd.testing.assert_frame_equal(out, raw)
    assert (tmp_path / "mlb_hitting_2024.csv").exists()


def test_fetch_mlb_season_paginates_and_keeps_all_columns(monkeypatch):
    monkeypatch.setattr(mlb_stats, "_MLB_PAGE", 2)
    pages = [
        _page(
            [
                {
                    "player": {"id": 1, "fullName": "A"},
                    "stat": {"homeRuns": 10},
                    "team": {"name": "X"},
                },
                {
                    "player": {"id": 2, "fullName": "B"},
                    "stat": {"homeRuns": 5},
                    "team": {"name": "Y"},
                },
            ]
        ),
        _page(
            [
                {
                    "player": {"id": 3, "fullName": "C"},
                    "stat": {"homeRuns": 1},
                    "team": {"name": "Z"},
                },
            ]
        ),
    ]
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, *, params: dict[str, Any], timeout: int) -> _FakeResp:
        calls.append(params)
        return _FakeResp(pages[params["offset"] // 2])

    df = _fetch_mlb_season("hitting", 2024, get=fake_get)

    assert len(df) == 3
    assert len(calls) == 2  # stopped after the short (< _MLB_PAGE) page
    # nothing dropped -- name and team survive (the old {mlbam, **stat} dropped them)
    assert "player.fullName" in df.columns
    assert "team.name" in df.columns
    assert "stat.homeRuns" in df.columns
