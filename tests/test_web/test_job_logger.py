import json
from unittest.mock import MagicMock, patch

from fantasy_baseball.web.job_logger import JobLogger, get_all_logs


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


def test_job_logger_finish_writes_to_redis():
    mock_redis = MagicMock()
    with patch("fantasy_baseball.web.job_logger._get_redis", return_value=mock_redis):
        logger = JobLogger("refresh")
        logger.log("Step one")
        logger.finish("ok")

    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args

    # Verify key pattern: job_log:refresh:<date>:<timestamp>
    key = call_args[0][0]
    assert key.startswith("job_log:refresh:")

    # Verify JSON structure
    raw = call_args[0][1]
    data = json.loads(raw)
    assert data["job"] == "refresh"
    assert data["status"] == "ok"
    assert data["error"] is None
    assert len(data["entries"]) == 1
    assert data["entries"][0]["msg"] == "Step one"
    assert "started_at" in data
    assert "finished_at" in data
    assert "duration_seconds" in data

    # Verify 30-day TTL
    assert call_args[1]["ex"] == 30 * 86400


def test_job_logger_finish_error():
    mock_redis = MagicMock()
    with patch("fantasy_baseball.web.job_logger._get_redis", return_value=mock_redis):
        logger = JobLogger("refresh")
        logger.log("Something went wrong")
        logger.finish("error", error="Connection refused")

    call_args = mock_redis.set.call_args
    raw = call_args[0][1]
    data = json.loads(raw)
    assert data["status"] == "error"
    assert data["error"] == "Connection refused"


def test_job_logger_finish_no_redis():
    with patch("fantasy_baseball.web.job_logger._get_redis", return_value=None):
        logger = JobLogger("refresh")
        logger.log("Step one")
        # Should not raise
        logger.finish("ok")


def test_get_all_logs_returns_sorted():
    older_log = json.dumps(
        {
            "job": "refresh",
            "started_at": "2026-03-29 10:00:00",
            "status": "ok",
        }
    )
    newer_log = json.dumps(
        {
            "job": "refresh",
            "started_at": "2026-03-30 08:00:00",
            "status": "ok",
        }
    )

    mock_redis = MagicMock()
    mock_redis.keys.return_value = [
        "job_log:refresh:2026-03-29:111",
        "job_log:refresh:2026-03-30:222",
    ]
    mock_redis.mget.return_value = [older_log, newer_log]

    with patch("fantasy_baseball.web.job_logger._get_redis", return_value=mock_redis):
        logs = get_all_logs()

    assert len(logs) == 2
    assert logs[0]["started_at"] == "2026-03-30 08:00:00"
    assert logs[1]["started_at"] == "2026-03-29 10:00:00"


def test_get_all_logs_no_redis():
    with patch("fantasy_baseball.web.job_logger._get_redis", return_value=None):
        logs = get_all_logs()
    assert logs == []
