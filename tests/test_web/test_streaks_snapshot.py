"""Snapshot test for the rendered /streaks page.

Catches unintentional structural drift in the HTML. The snapshot is a
plain text file under ``tests/test_web/snapshots/``. On first run it
writes the snapshot and SKIPs; subsequent runs compare. To regenerate,
delete the snapshot file.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.streaks.dashboard import serialize_report
from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
from fantasy_baseball.streaks.reports.sunday import (
    DriverLine,
    Report,
    ReportRow,
)
from fantasy_baseball.web.season_app import create_app
from fantasy_baseball.web.season_data import write_cache

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SNAPSHOT_PATH = SNAPSHOT_DIR / "streaks.html"


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        with c.session_transaction() as sess:
            sess["authenticated"] = True
        yield c


@pytest.fixture
def kv_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "test.db"))
    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


def _seed_canonical_report() -> None:
    score = PlayerCategoryScore(
        player_id=1,
        category="hr",
        label="hot",
        probability=0.62,
        drivers=(Driver(feature="barrel_pct", z_score=1.8),),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="Canon Player",
        positions=("OF",),
        player_id=1,
        composite=1,
        scores={"hr": score},
        max_probability=0.62,
    )
    dl = DriverLine(
        player_name="Canon Player",
        category="hr",
        label="hot",
        probability=0.62,
        drivers=(Driver(feature="barrel_pct", z_score=1.8),),
    )
    rpt = Report(
        report_date=date(2026, 5, 11),
        window_end=date(2026, 5, 10),
        team_name="Hart of the Order",
        league_id=5652,
        season_set_train="2023-2025",
        roster_rows=(row,),
        fa_rows=(),
        driver_lines=(dl,),
        skipped=(),
    )
    write_cache(CacheKey.STREAK_SCORES, serialize_report(rpt))


def test_streaks_html_snapshot(client, kv_isolation) -> None:
    _seed_canonical_report()
    resp = client.get("/streaks")
    assert resp.status_code == 200
    actual = resp.data.decode()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    if not SNAPSHOT_PATH.exists():
        SNAPSHOT_PATH.write_text(actual, encoding="utf-8")
        pytest.skip("Snapshot created; rerun.")
    expected = SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert actual == expected, (
        "Streaks HTML drift detected. "
        f"Diff the response against {SNAPSHOT_PATH} and either fix the route/template "
        "or delete the snapshot to regenerate."
    )
