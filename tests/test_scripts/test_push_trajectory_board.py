"""Where `push_trajectory_board` writes, and what provenance it stamps.

The script had no tests at all. These cover the destination decision, which is the
part that can quietly send an unverified board to production.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from fantasy_baseball.data.kv_store import SqliteKVStore

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location(
        "push_trajectory_board", PROJECT_ROOT / "scripts" / "push_trajectory_board.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_stays_local_even_when_render_is_already_set(monkeypatch, tmp_path) -> None:
    """`--local` exists to keep an unverified board OFF Render.

    `get_kv()` resolves through `is_remote()`, which reads RENDER at call time. The
    script used to SET that variable on the prod path and never clear it, so running
    --local in a shell where RENDER was already "true" -- the state several repo scripts
    and the documented "read prod locally" workflow both create -- silently wrote the
    board to PRODUCTION while printing "wrote to local SQLite".
    """
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "local.db"))

    target = _script()._target_store(local=True)

    assert isinstance(target, SqliteKVStore), (
        "--local must pick the local store from the FLAG, never from the environment"
    )


def test_the_push_never_mutates_render(monkeypatch) -> None:
    """The prod path selects its client explicitly, so it has no reason to touch RENDER.

    Setting it was copied from refresh_remote.py, where it IS load-bearing because that
    script then runs code resolving through get_kv(). Here nothing after the flip reads
    get_kv(), and the flip inverted the provenance: with RENDER set, `_code_sha()` takes
    the is_remote() branch, skips the git fallback, finds no RENDER_GIT_COMMIT on a
    laptop and stamps `_sha: "unknown"` -- so the PROD blob recorded "unknown" while
    --local recorded the real commit.
    """
    monkeypatch.delenv("RENDER", raising=False)
    module = _script()

    with pytest.raises(Exception):  # noqa: B017 - refuses to build an Upstash client here
        module._target_store(local=False)

    assert "RENDER" not in __import__("os").environ, (
        "selecting the prod store must not leave RENDER set for whatever runs next"
    )
