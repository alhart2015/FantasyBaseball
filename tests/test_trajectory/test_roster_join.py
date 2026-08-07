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
    assert index.unscored == {"Mine": ["Rookie"], "Theirs": ["Their Rookie"]}


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
