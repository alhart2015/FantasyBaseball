"""Tests for streaks/payload.py — the STREAK_SCORES payload (de)serializers.

Formerly the serialization half of test_dashboard.py (#175 moved the
serializers out of the duckdb-tainted streaks/dashboard.py).
"""

from __future__ import annotations

from fantasy_baseball.streaks.payload import (
    _deserialize_score,
    deserialize_report,
    load_report,
    serialize_report,
)
from tests.test_streaks.conftest import example_report, example_report_inactive


def test_serialize_report_round_trips() -> None:
    original = example_report()
    payload = serialize_report(original)
    rebuilt = deserialize_report(payload)
    assert rebuilt == original


def test_serialize_report_round_trips_inactive_row() -> None:
    original = example_report_inactive()
    payload = serialize_report(original)
    assert payload["roster_rows"][0]["days_since_last_game"] == 30
    rebuilt = deserialize_report(payload)
    assert rebuilt == original
    assert rebuilt.roster_rows[0].days_since_last_game == 30


def test_deserialize_score_tolerates_legacy_payload_without_baserate() -> None:
    """Payloads cached before probability_baserate existed must still load."""
    legacy = {
        "player_id": 7,
        "category": "hr",
        "label": "hot",
        "probability": 0.62,
        "drivers": [],
        "window_end": None,
    }
    score = _deserialize_score(legacy)
    assert score.probability == 0.62
    assert score.probability_baserate is None


def test_load_report_is_none_for_no_cache_and_for_an_unreadable_one(caplog) -> None:
    """The dashboard must not 500 on a malformed cache: no chips instead."""
    assert load_report(None) is None
    broken = serialize_report(example_report())
    del broken["roster_rows"][0]["composite"]
    assert load_report(broken) is None
    assert "unreadable STREAK_SCORES payload" in caplog.text
    assert load_report(serialize_report(example_report())) == example_report()
