from __future__ import annotations

from fantasy_baseball.data.rosters import RosterSpot, owner_map, parse_rosters

MINE = "Hart of the Order"


def _player(name: str, kind: str = "hitter", **extra) -> dict:
    return {"name": name, "player_type": kind, "player_id": 12345, **extra}


def test_your_own_roster_is_a_bare_list_and_still_gets_your_team_name() -> None:
    """`cache:roster` carries no team name -- it is implicitly yours. Treating it like
    `cache:opp_rosters`, which IS a dict, drops your whole roster silently, and a board
    that shows nine teams reads as complete rather than broken."""
    spots = parse_rosters([_player("Juan Soto")], {"Spacemen": [_player("Brice Turang")]}, MINE)
    assert {s.team for s in spots} == {MINE, "Spacemen"}
    assert next(s for s in spots if s.name == "Juan Soto").team == MINE


def test_the_cache_envelope_is_unwrapped() -> None:
    """Prod blobs are wrapped as {"_meta": ..., "_data": ...}. Reading the wrapper as the
    payload yields zero players, which looks like an empty league rather than an error."""
    spots = parse_rosters(
        {"_meta": {"_written_at": "..."}, "_data": [_player("Juan Soto")]},
        {"_meta": {}, "_data": {"Spacemen": [_player("Brice Turang")]}},
        MINE,
    )
    assert sorted(s.name for s in spots) == ["Brice Turang", "Juan Soto"]


def test_a_missing_blob_is_survivable() -> None:
    """A refresh that has not run yet should cost you the opposing rosters, not crash the
    board you asked for."""
    spots = parse_rosters([_player("Juan Soto")], None, MINE)
    assert [s.team for s in spots] == [MINE]


def test_rows_without_a_name_are_skipped_rather_than_keyed_on_empty() -> None:
    """An empty name normalizes to "", which would collide every malformed row onto one
    owner entry and silently reassign a real player."""
    spots = parse_rosters([_player("Juan Soto"), {"player_type": "hitter"}, {}], {}, MINE)
    assert [s.name for s in spots] == ["Juan Soto"]


def test_the_join_key_is_normalized_and_typed() -> None:
    """Accents differ between Yahoo and MLBAM spellings, and a two-way player is two
    separate assets -- so the key is (accent-stripped name, player_type), not the name."""
    spots = parse_rosters(
        [_player("Iván Herrera"), _player("Shohei Ohtani"), _player("Shohei Ohtani", "pitcher")],
        {},
        MINE,
    )
    owners = owner_map(spots)
    assert owners[("ivan herrera", "hitter")] == MINE
    # Both halves of the two-way player are present and distinct.
    assert owners[("shohei ohtani", "hitter")] == MINE
    assert owners[("shohei ohtani", "pitcher")] == MINE


def test_status_is_carried_because_an_injured_player_is_still_owned() -> None:
    spots = parse_rosters([_player("Juan Soto", status="IL10")], {}, MINE)
    assert spots[0].status == "IL10"


def test_a_spot_records_the_yahoo_id_even_though_the_join_cannot_use_it() -> None:
    """Yahoo's id IS unique where the name is not. Carrying it means the #284 fix -- an
    mlbam_id at ingest -- can upgrade the join without re-reading these blobs."""
    spots = parse_rosters([_player("Juan Soto")], {}, MINE)
    assert spots[0].yahoo_id == "12345"


def test_owner_map_is_keyed_the_way_the_board_looks_players_up() -> None:
    spots = [
        RosterSpot("A B", "a b", "hitter", "Spacemen", "1", ""),
        RosterSpot("C D", "c d", "pitcher", MINE, "2", "IL10"),
    ]
    assert owner_map(spots) == {("a b", "hitter"): "Spacemen", ("c d", "pitcher"): MINE}
