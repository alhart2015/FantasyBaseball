from datetime import date

from fantasy_baseball.summary.assemble import refresh_is_fresh


def test_refresh_is_fresh_true_when_written_today():
    # Noon UTC on 2026-07-11 is 08:00 ET -> local date 2026-07-11.
    assert refresh_is_fresh("2026-07-11T12:00:00+00:00", date(2026, 7, 11)) is True


def test_refresh_is_fresh_false_when_stale():
    assert refresh_is_fresh("2026-07-10T12:00:00+00:00", date(2026, 7, 11)) is False


def test_refresh_is_fresh_false_when_missing_or_malformed():
    assert refresh_is_fresh(None, date(2026, 7, 11)) is False
    assert refresh_is_fresh("", date(2026, 7, 11)) is False
    assert refresh_is_fresh("garbage", date(2026, 7, 11)) is False


def test_refresh_is_fresh_uses_local_not_utc_date():
    # 02:00 UTC on 2026-07-11 is 22:00 ET on 2026-07-10 -> NOT fresh for the
    # 2026-07-11 local day. Guards the "UTC midnight reads as tomorrow" trap.
    assert refresh_is_fresh("2026-07-11T02:00:00+00:00", date(2026, 7, 11)) is False


def test_build_daily_summary_isolates_a_failing_builder(monkeypatch):
    """One raising builder degrades to an empty section + a section_errors note;
    the rest of the summary still assembles (spec error-isolation requirement)."""
    from datetime import date
    from pathlib import Path

    import fantasy_baseball.summary.assemble as asm
    from fantasy_baseball.config import LeagueConfig

    # Stub every external read so only build_streaks raises.
    monkeypatch.setattr(asm, "get_kv", lambda: object())
    monkeypatch.setattr(asm, "fetch_roster", lambda league, tk: [])
    monkeypatch.setattr(asm, "fetch_injuries", lambda league, tk: [])
    monkeypatch.setattr(asm, "build_typed_name_to_mlbam", lambda root, *, season: {})
    monkeypatch.setattr(asm, "read_cache", lambda key: None)
    monkeypatch.setattr(asm, "read_cache_dict", lambda key: None)
    monkeypatch.setattr(asm, "read_cache_list", lambda key: None)

    def _boom(_payload):
        raise RuntimeError("streaks exploded")

    monkeypatch.setattr(asm, "build_streaks", _boom)

    cfg = LeagueConfig.__new__(LeagueConfig)
    cfg.team_name = "My Team"
    cfg.season_year = 2026

    summary = asm.build_daily_summary(
        cfg, Path("."), today=date(2026, 7, 11), league=object(), team_key="t"
    )
    assert "build_streaks" in summary.section_errors
    assert summary.streaks == []  # degraded to empty, not fatal
    assert summary.as_of == date(2026, 7, 10)
