"""Tests for draft.keepers — the shared keeper-resolution helpers."""

from __future__ import annotations

from fantasy_baseball.draft.keepers import find_keeper_match, index_by_normalized_name


def test_index_groups_namesakes_under_the_same_key():
    rows = [
        {"name": "Jose Ramirez", "name_normalized": "jose ramirez", "var": 9.0},
        {"name": "Jose Ramirez", "name_normalized": "jose ramirez", "var": 0.5},
        {"name": "Aaron Judge", "name_normalized": "aaron judge", "var": 11.0},
    ]
    by_norm = index_by_normalized_name(rows)
    assert len(by_norm["jose ramirez"]) == 2
    assert len(by_norm["aaron judge"]) == 1


def test_index_falls_back_to_normalizing_name_when_field_missing():
    rows = [{"name": "Juan Soto", "var": 12.0}]  # no name_normalized
    by_norm = index_by_normalized_name(rows)
    assert "juan soto" in by_norm


def test_find_keeper_match_picks_highest_var():
    by_norm = {
        "jose ramirez": [
            {"name": "Jose Ramirez", "var": 0.5, "player_id": "low"},
            {"name": "Jose Ramirez", "var": 9.0, "player_id": "high"},
        ]
    }
    best = find_keeper_match("Jose Ramirez", by_norm)
    assert best is not None
    assert best["player_id"] == "high"


def test_find_keeper_match_handles_accented_input():
    """Resolver normalizes the query, not just the index."""
    by_norm = index_by_normalized_name([{"name": "Ronald Acuna Jr.", "var": 10.0}])
    # Accented form should hit the same normalized key (NFKD strips
    # combining marks).
    assert find_keeper_match("Ronald Acuña Jr.", by_norm) is not None


def test_find_keeper_match_returns_none_for_missing_name():
    assert find_keeper_match("Nobody", {}) is None


def test_find_keeper_match_tolerates_missing_var():
    """Rows without ``var`` shouldn't crash the max() — fall back to 0."""
    by_norm = {
        "x player": [
            {"name": "X Player", "player_id": "novars"},
            {"name": "X Player", "player_id": "withvar", "var": 1.0},
        ]
    }
    best = find_keeper_match("X Player", by_norm)
    assert best is not None
    assert best["player_id"] == "withvar"
