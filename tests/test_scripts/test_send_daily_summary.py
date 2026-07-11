from datetime import date

import pytest

# Noon UTC on 2026-07-11 = 08:00 ET -> local date 2026-07-11 (fresh vs the
# date(2026, 7, 11) the tests pass as "today").
_FRESH_ENVELOPE = ({"last_refresh": "9:00 AM"}, {"_written_at": "2026-07-11T12:00:00+00:00"})


@pytest.fixture
def patched(monkeypatch):
    import scripts.send_daily_summary as mod

    sent = {}
    monkeypatch.setattr(mod, "read_cache_with_meta", lambda key: _FRESH_ENVELOPE)
    monkeypatch.setattr(mod, "send_email", lambda **kw: sent.update(kw) or "msg_1")
    written = {}
    monkeypatch.setattr(mod, "_write_snapshot", lambda wa: written.update({"done": True}))
    return mod, sent, written


def test_run_summary_stale_refresh_skips_send(monkeypatch, patched):
    mod, sent, _written = patched
    stale = ({"last_refresh": "9:00 AM"}, {"_written_at": "2026-07-09T12:00:00+00:00"})
    monkeypatch.setattr(mod, "read_cache_with_meta", lambda key: stale)
    rc = mod.run_summary(
        _cfg(), _root(), api_key="k", league=object(), team_key="t", today=date(2026, 7, 11)
    )
    assert rc != 0
    assert sent == {}  # never sent


def test_run_summary_missing_meta_skips_send(monkeypatch, patched):
    mod, sent, _written = patched
    monkeypatch.setattr(mod, "read_cache_with_meta", lambda key: (None, {}))
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
