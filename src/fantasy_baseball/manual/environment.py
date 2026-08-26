"""Point a process at the isolated manual KV store.

The manual pipeline's isolation is by whole store, not by key prefix: a manual
run writes hand-typed data into ``cache:standings``, ``weekly_rosters_history``
and the rest of the ``cache:*`` family, the same keys the Yahoo pipeline owns.
Nothing separates them but the file, so every entry point that wants manual mode
has to bind this process to ``data/manual.db`` before anything resolves a store.

One copy of that sequence lives here because getting it subtly wrong is silent:
the store binds on the FIRST ``get_kv()``, and a process that imported something
which already resolved one keeps the old binding with no error.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Repo root, from ``src/fantasy_baseball/manual/environment.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: The isolated store the Yahoo-free pipeline reads and writes. Never
#: ``data/local.db``, which is the only copy of the pre-outage Yahoo history.
DEFAULT_MANUAL_KV_PATH = PROJECT_ROOT / "data" / "manual.db"


def activate_manual_environment(kv_path: Path | None = None) -> Path:
    """Bind this process to the manual store and disable every Yahoo step.

    Call BEFORE anything resolves a KV store. Returns the absolute path bound,
    so a caller can print it rather than re-deriving it and risking a banner
    that names a different store than the one in use.

    ``FANTASY_LOCAL_KV_PATH`` is set to an ABSOLUTE path deliberately:
    ``kv_store`` resolves it against the current working directory, so a
    relative value silently creates a second, empty store when the process is
    launched from anywhere but the repo root.
    """
    resolved = (kv_path or DEFAULT_MANUAL_KV_PATH).resolve()
    os.environ["FANTASY_LOCAL_KV_PATH"] = str(resolved)
    os.environ["FB_SKIP_YAHOO"] = "1"

    src = str(PROJECT_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    # Belt and braces: if anything in this process already built the KV
    # singleton (an ambient import, an earlier run in the same interpreter, a
    # test harness), discard it so the next get_kv() rebinds to the env var
    # just set.
    from fantasy_baseball.data import kv_store

    kv_store._reset_singleton()
    return resolved


def deactivate_manual_environment() -> dict[str, str]:
    """Unbind this process from the manual store; return what was cleared.

    The mirror of :func:`activate_manual_environment`, and the reason both
    exist: these variables are EXPORTED into a shell, so they outlive the
    command that set them. Without an explicit clear, a launcher run with no
    manual flag inherits whatever the last manual session left behind and
    serves the hand-transcribed store while the caller believes it is reading
    the Yahoo baseline. The banner shows the path, but nothing contradicts the
    expectation.

    Clearing is the only way the flag can be the single control. Returns the
    variables actually removed, so a caller can say so rather than changing the
    environment silently.

    No-op on Render: the KV is Upstash there, ``FANTASY_LOCAL_KV_PATH`` cannot
    reach it, and ``FB_SKIP_YAHOO`` may be a deliberate service setting that
    this function has no business overriding.
    """
    from fantasy_baseball.data.kv_store import is_remote

    if is_remote():
        return {}

    cleared = {
        name: os.environ.pop(name)
        for name in ("FANTASY_LOCAL_KV_PATH", "FB_SKIP_YAHOO")
        if name in os.environ
    }
    if cleared:
        from fantasy_baseball.data import kv_store

        kv_store._reset_singleton()
    return cleared
