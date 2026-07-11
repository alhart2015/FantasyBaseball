from datetime import date

import pytest


@pytest.fixture
def patched(monkeypatch):
    import scripts.send_daily_summary as mod

    sent = {}
    monkeypatch.setattr(mod, "read_meta", lambda: {"last_refresh": "2026-07-11 08:00"})
    monkeypatch.setattr(mod, "send_email", lambda **kw: sent.update(kw) or "msg_1")
    written = {}
    monkeypatch.setattr(mod, "_write_snapshot", lambda meta: written.update({"done": True}))
    return mod, sent, written


def test_run_summary_stale_refresh_skips_send(monkeypatch, patched):
    mod, sent, _written = patched
    monkeypatch.setattr(mod, "read_meta", lambda: {"last_refresh": "2026-07-09 08:00"})
    rc = mod.run_summary(
        _cfg(), _root(), api_key="k", league=object(), team_key="t", today=date(2026, 7, 11)
    )
    assert rc != 0
    assert sent == {}  # never sent


def test_run_summary_missing_meta_skips_send(monkeypatch, patched):
    mod, sent, _written = patched
    monkeypatch.setattr(mod, "read_meta", lambda: {})
    rc = mod.run_summary(
        _cfg(), _root(), api_key="k", league=object(), team_key="t", today=date(2026, 7, 11)
    )
    assert rc != 0
    assert sent == {}


def _fresh_summary():
    from fantasy_baseball.summary.models import DailySummary, StandingsDelta

    return DailySummary(
        as_of=date(2026, 7, 10),
        last_night=[],
        unmatched=[],
        streaks=[],
        standings_delta=StandingsDelta(is_first_run=True, user_team_name="T"),
        lineup_moves=[],
        injuries=[],
        probables=[],
        section_errors=[],
    )


def test_run_summary_fresh_sends_and_writes_snapshot(monkeypatch, patched):
    mod, sent, written = patched
    monkeypatch.setattr(mod, "build_daily_summary", lambda *a, **k: _fresh_summary())
    rc = mod.run_summary(
        _cfg(), _root(), api_key="k", league=object(), team_key="t", today=date(2026, 7, 11)
    )
    assert rc == 0
    assert sent["subject"]
    assert written.get("done") is True


def test_run_summary_failed_send_does_not_advance_snapshot(monkeypatch, patched):
    mod, _sent, written = patched
    monkeypatch.setattr(mod, "build_daily_summary", lambda *a, **k: _fresh_summary())

    def _boom(**kw):
        raise RuntimeError("resend down")

    monkeypatch.setattr(mod, "send_email", _boom)
    rc = mod.run_summary(
        _cfg(), _root(), api_key="k", league=object(), team_key="t", today=date(2026, 7, 11)
    )
    assert rc != 0
    assert written == {}  # snapshot NOT advanced on failed send


def _cfg():
    from fantasy_baseball.config import LeagueConfig

    c = LeagueConfig.__new__(LeagueConfig)
    c.team_name = "T"
    c.season_year = 2026
    c.summary = {"recipients": ["me@x.com"], "from_address": "d@x.com"}
    return c


def _root():
    from pathlib import Path

    return Path(".")
