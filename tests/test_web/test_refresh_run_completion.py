"""`run()` returning is not `run()` having run.

`RefreshRun.run()` acquires the cross-instance durable lock and, when another
instance already holds it, logs "skipping this run" and returns NORMALLY having
executed zero steps. Any caller that then reads the caches gets whatever the
last successful run wrote -- so a manual run would render a stale audit under a
fresh provenance header. The lock has a 30-minute TTL and is released in a
`finally`, so an interrupted run leaves it set and the natural response (fix the
input, re-run) walks straight into it.
"""

from __future__ import annotations

import pytest

from fantasy_baseball.web import refresh_pipeline as rp
from fantasy_baseball.web.refresh_pipeline import RefreshRun


@pytest.fixture(autouse=True)
def _slot_is_left_free(monkeypatch):
    """Leave the in-process refresh slot free, whatever this test does to it.

    `run()` writes `_refresh_status["running"] = True` into a MODULE-LEVEL dict
    and relies on its own `finally: release_refresh_slot()` to put it back. That
    global outlives the test: stubbing the release out left `running` True for
    the rest of the worker's session, and `/api/refresh-status` -- which just
    reports the dict -- then answered "a refresh is running" in an unrelated
    file. Under xdist the two land on the same worker only sometimes, so it
    surfaced as an ordering-dependent failure rather than as this file's fault.

    So: do NOT stub the release, and belt-and-braces release again on the way
    out for the paths that raise before the finally is reached.
    """
    yield
    rp.release_refresh_slot()


class _Logger:
    def start(self, *a, **k):
        return None

    def finish(self, *a, **k):
        return None


def _run(monkeypatch, *, got_lock: bool, steps_ran: list):
    import contextlib

    @contextlib.contextmanager
    def _lock():
        yield got_lock

    run = RefreshRun(skip_yahoo=True)
    monkeypatch.setattr(rp, "durable_refresh_lock", _lock)
    monkeypatch.setattr(run, "logger", _Logger(), raising=False)
    monkeypatch.setattr(run, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(run, "_run_pipeline_steps", lambda: steps_ran.append(True))
    return run


def test_completed_is_false_when_the_lock_was_not_acquired(monkeypatch):
    steps: list = []
    run = _run(monkeypatch, got_lock=False, steps_ran=steps)

    run.run()

    assert steps == [], "no step may run without the lock"
    assert run.completed is False, (
        "a caller reading the caches after this would render the PREVIOUS run's output"
    )


def test_completed_is_true_after_a_full_run(monkeypatch):
    steps: list = []
    run = _run(monkeypatch, got_lock=True, steps_ran=steps)

    run.run()

    assert steps == [True]
    assert run.completed is True


def test_completed_stays_false_when_a_step_raises(monkeypatch):
    """A partial run is not a completed one, and `run()` re-raises."""

    def _boom():
        raise RuntimeError("step exploded")

    run = _run(monkeypatch, got_lock=True, steps_ran=[])
    monkeypatch.setattr(run, "_run_pipeline_steps", _boom)

    with pytest.raises(RuntimeError):
        run.run()

    assert run.completed is False


def test_a_fresh_run_starts_incomplete():
    assert RefreshRun().completed is False
