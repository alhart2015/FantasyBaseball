"""Cheap regression guard: dataclass field surface stays in lockstep with the DDL.

If this fails after a column add, update both `models.py` and `schema.py`
together — they are co-load-bearing for `load.py`'s attrgetter-based upsert.
"""

from __future__ import annotations

from dataclasses import fields

from fantasy_baseball.streaks.models import HitterGame, HitterStatcastPA


def test_hitter_game_fields_in_expected_order() -> None:
    expected = (
        "player_id",
        "game_pk",
        "name",
        "team",
        "season",
        "date",
        "pa",
        "ab",
        "h",
        "hr",
        "r",
        "rbi",
        "sb",
        "bb",
        "k",
        "b2",
        "b3",
        "sf",
        "hbp",
        "ibb",
        "cs",
        "gidp",
        "sh",
        "ci",
        "is_home",
    )
    assert tuple(f.name for f in fields(HitterGame)) == expected


def test_hitter_statcast_pa_fields_in_expected_order() -> None:
    expected = (
        "player_id",
        "date",
        "pa_index",
        "event",
        "launch_speed",
        "launch_angle",
        "estimated_woba_using_speedangle",
        "barrel",
        "at_bat_number",
        "bb_type",
        "estimated_ba_using_speedangle",
        "hit_distance_sc",
    )
    assert tuple(f.name for f in fields(HitterStatcastPA)) == expected
