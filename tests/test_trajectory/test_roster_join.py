from __future__ import annotations

from fantasy_baseball.data.rosters import RosterSpot
from fantasy_baseball.trajectory.roster_join import index_rosters


def _spot(name: str, team: str, pool: str = "hitter", status: str = "") -> RosterSpot:
    return RosterSpot(
        name=name,
        normalized=name.lower(),
        player_type=pool,
        team=team,
        yahoo_id="0",
        status=status,
    )


def _row(name: str, pool: str = "hitter") -> dict:
    return {"name": name, "pool": pool}


def test_my_team_sorts_first_and_the_rest_alphabetically() -> None:
    """The dropdown order. Mine first because it is the roster I act on."""
    spots = [_spot("A", "Zebras"), _spot("B", "Mine"), _spot("C", "Aardvarks")]
    index = index_rosters([], spots, "Mine")
    assert index.teams == ("Mine", "Aardvarks", "Zebras")


def test_teams_are_plain_alphabetical_when_my_team_is_unknown() -> None:
    """`my_team` is None when the config read fails, and can name a team that is
    not in the blob after a rename. Neither may promote a phantom entry."""
    spots = [_spot("A", "Zebras"), _spot("C", "Aardvarks")]
    assert index_rosters([], spots, None).teams == ("Aardvarks", "Zebras")
    assert index_rosters([], spots, "Nobody").teams == ("Aardvarks", "Zebras")


def test_a_name_matching_two_board_rows_is_ambiguous() -> None:
    """The live board carries two hitters called Max Muncy. Roster blobs have no
    mlbam_id (#284), so the join cannot tell them apart and must say so."""
    rows = [_row("Max Muncy"), _row("Max Muncy"), _row("Elly De La Cruz")]
    index = index_rosters(rows, [_spot("Max Muncy", "Mine")], "Mine")
    assert index.is_ambiguous("Max Muncy", "hitter")
    assert not index.is_ambiguous("Elly De La Cruz", "hitter")


def test_two_teams_rostering_one_key_BOTH_own_the_row() -> None:
    """The SPOT side of the collision, with one board row to fight over.

    This asserted the opposite until 2026-08-07: that one spot won the key and the
    LOSER was listed as unscored. That was the bug. Ownership derived from a single
    winner meant my own rostered player rendered under the other team, unhighlighted
    and unflagged, and the list that would have named him only appears on a filtered
    view. Membership is not a contest -- both teams roster the name, neither is
    missing him, and the (?) flag is what says the board cannot tell which is which.
    """
    rows = [_row("Max Muncy")]
    spots = [_spot("Max Muncy", "Zebras"), _spot("Max Muncy", "Aardvarks")]
    index = index_rosters(rows, spots, "Zebras")

    assert index.is_ambiguous("Max Muncy", "hitter")
    assert index.owners_for("Max Muncy", "hitter") == frozenset({"Zebras", "Aardvarks"})
    assert index.unscored_for("Zebras") == [], "a matched key is not a missing player"
    assert index.unscored_for("Aardvarks") == []
    assert index.matched_teams == frozenset({"Zebras", "Aardvarks"})

    # `team_for` still answers with ONE team, for the CSV column and the [IL10]
    # suffix, and it stays a fixed function of the data rather than roster order.
    winner = index.team_for("Max Muncy", "hitter")
    assert winner in ("Zebras", "Aardvarks")
    assert index_rosters(rows, list(reversed(spots)), "Zebras").team_for("Max Muncy", "hitter") == (
        winner
    )


def test_one_team_rostering_two_players_under_a_key_is_not_missing_either() -> None:
    """Both halves colliding on ONE roster -- two Luis Garcias, same team.

    The loser used to land in `unscored`, so the page listed a player as missing
    directly beneath his own visible row.
    """
    rows = [_row("Luis Garcia", "pitcher"), _row("Luis Garcia", "pitcher")]
    spots = [
        _spot("Luis Garcia", "Mine", pool="pitcher"),
        _spot("Luis Garcia", "Mine", pool="pitcher"),
    ]
    index = index_rosters(rows, spots, "Mine")

    assert index.is_ambiguous("Luis Garcia", "pitcher")
    assert index.owners_for("Luis Garcia", "pitcher") == frozenset({"Mine"})
    assert index.unscored_for("Mine") == [], "his row is on screen; he is not missing"


def test_unscored_can_be_filtered_to_the_pool_on_screen() -> None:
    """The list renders under a table that may be showing one pool, so naming a
    player from the other reads as a hole in that pool rather than as a note."""
    spots = [
        _spot("Unpriced Bat", "Mine"),
        _spot("Unpriced Arm", "Mine", pool="pitcher"),
    ]
    index = index_rosters([], spots, "Mine")

    assert index.unscored_for("Mine") == ["Unpriced Arm", "Unpriced Bat"]
    assert index.unscored_for("Mine", "hitter") == ["Unpriced Bat"]
    assert index.unscored_for("Mine", "pitcher") == ["Unpriced Arm"]


def test_the_same_name_in_two_pools_is_not_ambiguous() -> None:
    """A two-way player is two assets in this league, so (name, pool) is the key
    and a hitter does not collide with a pitcher."""
    rows = [_row("Shohei Ohtani", "hitter"), _row("Shohei Ohtani", "pitcher")]
    index = index_rosters(rows, [_spot("Shohei Ohtani", "Mine")], "Mine")
    assert not index.is_ambiguous("Shohei Ohtani", "hitter")


def test_roster_players_with_no_scored_row_are_grouped_by_team() -> None:
    """The list the UI must render -- "the model could not price these", not
    "these are not worth listing"."""
    rows = [_row("Scored Guy")]
    spots = [
        _spot("Scored Guy", "Mine"),
        _spot("Rookie", "Mine"),
        _spot("Their Rookie", "Theirs"),
    ]
    index = index_rosters(rows, spots, "Mine")
    assert index.unscored_for("Mine") == ["Rookie"]
    assert index.unscored_for("Theirs") == ["Their Rookie"]


def test_team_and_status_are_looked_up_by_normalized_name_and_pool() -> None:
    spots = [_spot("Yoan Moncada", "Mine", status="IL10")]
    index = index_rosters([_row("Yoan Moncada")], spots, "Mine")
    assert index.team_for("Yoan Moncada", "hitter") == "Mine"
    assert index.status_for("Yoan Moncada", "hitter") == "IL10"
    assert index.team_for("Yoan Moncada", "pitcher") is None
    assert index.status_for("Nobody", "hitter") == ""


def test_the_input_rows_are_not_mutated() -> None:
    """The entire reason this returns a lookup instead of stamping rows.

    The web's rows come from `_ranked_rows`, a cross-request cache whose comment
    states they are never mutated after `add_ranks`. A helper that stamped them
    would be correct in the CLI and a data race on the web.
    """
    rows = [_row("Scored Guy")]
    before = [dict(r) for r in rows]
    index_rosters(rows, [_spot("Scored Guy", "Mine")], "Mine")
    assert rows == before
