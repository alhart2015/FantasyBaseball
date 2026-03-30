"""Persistent job logging to Upstash Redis."""

import json
import time
from datetime import datetime, date


def _get_redis():
    """Lazy Redis client — reuses the season_data helper."""
    from fantasy_baseball.web.season_data import _get_redis as get_redis
    return get_redis()


class JobLogger:
    """Accumulates verbose log entries during a job run and writes to Redis.

    Usage:
        logger = JobLogger("refresh")
        logger.log("Authenticating...")
        logger.log("Fetching standings...")
        logger.finish("ok")  # writes complete log to Redis
    """

    def __init__(self, job_name: str):
        self.job_name = job_name
        self._start = time.time()
        self._started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._entries: list[dict] = []

    def log(self, msg: str) -> None:
        """Append a timestamped log entry."""
        self._entries.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "msg": msg,
        })

    def finish(self, status: str, error: str | None = None) -> None:
        """Write the complete log to Redis."""
        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = round(time.time() - self._start)
        today = date.today().isoformat()
        timestamp = int(self._start)
        key = f"job_log:{self.job_name}:{today}:{timestamp}"

        log_data = json.dumps({
            "job": self.job_name,
            "started_at": self._started_at,
            "finished_at": finished_at,
            "status": status,
            "duration_seconds": duration,
            "error": error,
            "entries": self._entries,
        })

        redis = _get_redis()
        if redis is None:
            return
        try:
            redis.set(key, log_data, ex=30 * 86400)  # 30 day TTL
        except Exception:
            pass  # don't crash the job if logging fails


def get_all_logs() -> list[dict]:
    """Read all job logs from Redis, sorted by most recent first."""
    redis = _get_redis()
    if redis is None:
        return []
    try:
        keys = redis.keys("job_log:*")
        if not keys:
            return []
        logs = []
        for key in keys:
            raw = redis.get(key)
            if raw:
                logs.append(json.loads(raw))
        logs.sort(key=lambda l: l.get("started_at", ""), reverse=True)
        return logs
    except Exception:
        return []
