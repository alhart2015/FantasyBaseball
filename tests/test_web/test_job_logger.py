import json
from unittest.mock import MagicMock, patch

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.web.job_logger import JobLogger, get_all_logs


@pytest.fixture(autouse=True)
def _local_kv(tmp_path, monkeypatch):
    """Per-test isolated SQLite KV.

    After unifying on ``get_kv()`` (removing the old ``_get_redis()`` that
    returned ``None`` off-Render), ``finish()`` and ``get_all_logs()``
    persist to and read from the local backend, so each test needs its own
    empty store.
    """
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "kv.db"))
    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


def test_job_logger_records_entries():
    logger = JobLogger("refresh")
    logger.log("Authenticating...")
    logger.log("Fetching standings...")
    assert len(logger._entries) == 2
    assert logger._entries[0]["msg"] == "Authenticating..."
    assert logger._entries[1]["msg"] == "Fetching standings..."
    # Each entry must have a "time" field
    for entry in logger._entries:
        assert "time" in entry


def test_job_logger_finish_persists_to_local_kv():
    """Off-Render, finish() persists to the local SQLite KV.

    Previously a no-op: the old ``_get_redis()`` returned ``None`` off-Render
    so local job logs were silently dropped. Unifying on ``get_kv()`` means
    the local backend now holds them, readable back via ``get_all_logs()``.
    """
    logger = JobLogger("refresh")
    logger.log("Step one")
    logger.finish("ok")

    logs = get_all_logs()
    assert len(logs) == 1
    assert logs[0]["job"] == "refresh"
    assert logs[0]["status"] == "ok"
    assert logs[0]["error"] is None
    assert logs[0]["entries"][0]["msg"] == "Step one"


def test_job_logger_finish_writes_key_and_ttl():
    # Mocks the KV specifically to inspect the `ex` (TTL) kwarg, which the
    # SQLite backend's read API doesn't expose; the other tests use the real store.
    mock_kv = MagicMock()
    with patch("fantasy_baseball.data.kv_store.get_kv", return_value=mock_kv):
        logger = JobLogger("refresh")
        logger.log("Step one")
        logger.finish("ok")

    mock_kv.set.assert_called_once()
    call_args = mock_kv.set.call_args

    # Key pattern: job_log:refresh:<date>:<timestamp>
    assert call_args[0][0].startswith("job_log:refresh:")

    # Payload shape
    data = json.loads(call_args[0][1])
    assert data["job"] == "refresh"
    assert data["status"] == "ok"
    assert len(data["entries"]) == 1
    assert "started_at" in data
    assert "finished_at" in data
    assert "duration_seconds" in data

    # 30-day TTL
    assert call_args[1]["ex"] == 30 * 86400


def test_job_logger_finish_records_error():
    logger = JobLogger("refresh")
    logger.log("Something went wrong")
    logger.finish("error", error="Connection refused")

    logs = get_all_logs()
    assert len(logs) == 1
    assert logs[0]["status"] == "error"
    assert logs[0]["error"] == "Connection refused"


def test_job_logger_finish_swallows_kv_errors():
    """finish() must never crash the job, even if the KV write raises."""
    mock_kv = MagicMock()
    mock_kv.set.side_effect = RuntimeError("kv down")
    with patch("fantasy_baseball.data.kv_store.get_kv", return_value=mock_kv):
        logger = JobLogger("refresh")
        logger.log("Step one")
        logger.finish("ok")  # should not raise


def test_get_all_logs_returns_sorted():
    # Seed the real KV directly so this exercises the production keys()/mget()
    # path, then assert get_all_logs sorts most-recent-first.
    kv = kv_store.get_kv()
    kv.set(
        "job_log:refresh:2026-03-29:111",
        json.dumps({"job": "refresh", "started_at": "2026-03-29 10:00:00", "status": "ok"}),
    )
    kv.set(
        "job_log:refresh:2026-03-30:222",
        json.dumps({"job": "refresh", "started_at": "2026-03-30 08:00:00", "status": "ok"}),
    )

    logs = get_all_logs()

    assert len(logs) == 2
    assert logs[0]["started_at"] == "2026-03-30 08:00:00"
    assert logs[1]["started_at"] == "2026-03-29 10:00:00"


def test_get_all_logs_empty_store_returns_empty_list():
    assert get_all_logs() == []
