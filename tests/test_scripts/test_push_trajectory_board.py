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


def test_an_empty_pool_refuses_the_push(monkeypatch) -> None:
    """A pool that scores nobody must not overwrite a good board.

    `panel_path` picks the newest panel per kind INDEPENDENTLY and the base season comes
    off the HITTER panel, so a stale pitcher panel makes `board_inputs(season=2026)` hit
    `current.empty` and return [] -- the run then prints "0 players with a 2026 line" and
    pushes a hitters-only board over a complete one. Same shape as the 2026-06-04 ROS
    incident, and guarded the same way: refuse before touching the KV.
    """
    module = _script()
    with pytest.raises(module.EmptyPoolError) as exc:
        module._require_scored_pool("pitcher", [], 2026, "pitcher_pt_panel_2000_2025.csv")

    message = str(exc.value)
    assert "pitcher" in message and "2026" in message
    assert "pitcher_pt_panel_2000_2025.csv" in message, "name the panel that came up short"

    # A pool that scored is silent.
    module._require_scored_pool("hitter", [object()], 2026, "hitter_pt_panel_2000_2026.csv")


def test_the_guard_checks_what_the_sweep_produced_not_what_it_was_given(monkeypatch) -> None:
    """Candidate rows are not scored rows, and the guard has to look at the latter.

    `sweep_pool` independently drops every player whose VAR path has no observable point.
    A panel whose COMPLETE seasons do not span horizon 1 -- a pitcher panel built only
    through the in-progress year, say -- yields hundreds of candidates, sails past a
    guard that inspects its INPUT, and then returns [] with no exception. The push
    proceeds and overwrites a complete board with a pool-less one that renders normally,
    which is verbatim what EmptyPoolError's docstring says it prevents.

    Drives `build_payload` rather than the guard directly: the defect was the guard's
    POSITION, so calling it with an empty list would pass either way.
    """
    module = _script()
    # build_payload imports sweep_pool inside the function, so patch it at the source.
    monkeypatch.setattr("fantasy_baseball.trajectory.sweep.sweep_pool", lambda *a, **k: [])

    with pytest.raises(module.EmptyPoolError) as exc:
        module.build_payload(max_horizon=1, panel_dir=None)

    assert "scored 0 players" in str(exc.value)


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
