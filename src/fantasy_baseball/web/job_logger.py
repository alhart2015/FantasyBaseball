"""Persistent job logging to the KV store (Upstash on Render, SQLite locally)."""

import json
import time

from fantasy_baseball.utils.time_utils import local_now, local_today


class JobLogger:
    """Accumulates verbose log entries during a job run and writes to the KV store.

    Usage:
        logger = JobLogger("refresh")
        logger.log("Authenticating...")
        logger.log("Fetching standings...")
        logger.finish("ok")  # writes complete log to the KV store
    """

    def __init__(self, job_name: str):
        self.job_name = job_name
        self._start = time.time()
        self._started_at = local_now().strftime("%Y-%m-%d %H:%M:%S")
        self._entries: list[dict] = []

    def log(self, msg: str) -> None:
        """Append a timestamped log entry."""
        self._entries.append(
            {
                "time": local_now().strftime("%H:%M:%S"),
                "msg": msg,
            }
        )

    def finish(self, status: str, error: str | None = None) -> None:
        """Write the complete log to the KV store. Never raises."""
        try:
            finished_at = local_now().strftime("%Y-%m-%d %H:%M:%S")
            duration = round(time.time() - self._start)
            today = local_today().isoformat()
            timestamp = int(self._start)
            key = f"job_log:{self.job_name}:{today}:{timestamp}"

            log_data = json.dumps(
                {
                    "job": self.job_name,
                    "started_at": self._started_at,
                    "finished_at": finished_at,
                    "status": status,
                    "duration_seconds": duration,
                    "error": error,
                    "entries": self._entries,
                }
            )

            from fantasy_baseball.data.kv_store import get_kv

            get_kv().set(key, log_data, ex=30 * 86400)  # 30 day TTL
        except Exception:
            pass  # never crash the job if logging fails


def get_all_logs() -> list[dict]:
    """Read all job logs from the KV store, sorted by most recent first.

    Routes through ``kv_store.get_kv()`` (Upstash on Render, SQLite
    locally), so logs persist and read back in both environments. Uses
    KEYS to find log entries (fine for small keyspaces; this store only
    holds dashboard cache + job logs) and MGET to batch reads into a
    single round-trip.
    """
    from fantasy_baseball.data.kv_store import get_kv

    try:
        kv = get_kv()
        keys = kv.keys("job_log:*")
        if not keys:
            return []
        values = kv.mget(*keys)
        logs = []
        for raw in values:
            if raw:
                logs.append(json.loads(raw))
        logs.sort(key=lambda log: log.get("started_at", ""), reverse=True)
        return logs
    except Exception:
        return []
