"""Contract tests for the hand-transcribed Yahoo YAML loaders.

Two jobs here:

1. Pin the transcription schema -- the worked example round-trips into
   exactly the ``weekly_rosters_history`` row shape the Yahoo pipeline
   writes, and every documented validation failure is caught BEFORE any
   KV write.
2. Guard the real ``data/manual/standings.yaml`` snapshot. The
   counting-stat categories (R HR RBI SB W SV K) must re-derive from the
   transcribed totals to exactly the ``category_points`` Yahoo shows. A
   future re-transcription that fat-fingers a total breaks this test,
   which is the whole point.
"""

from __future__ import annotations

import copy
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from fantasy_baseball.manual.transcripts import (
    ManualRosterSnapshot,
    ManualTranscriptError,
    load_fa_exclusions,
    load_manual_rosters,
    load_manual_standings,
    validate_transcripts,
)
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.standings import CategoryStats, Standings, StandingsEntry
from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import Category, OpportunityStat
from fantasy_baseball.utils.time_utils import local_today

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_STANDINGS = REPO_ROOT / "data" / "manual" / "standings.yaml"
SHIPPED_ROSTERS = REPO_ROOT / "data" / "manual" / "rosters.yaml"
SHIPPED_EXCLUSIONS = REPO_ROOT / "data" / "manual" / "fa_exclusions.yaml"

USER_TEAM = "Hart of the Order"

# Yahoo's own counting-stat columns: no rounding happens on the way to
# the screen, so their roto points must be exactly re-derivable.
COUNTING_CATEGORIES = [
    Category.R,
    Category.HR,
    Category.RBI,
    Category.SB,
    Category.W,
    Category.SV,
    Category.K,
]

# config/league.yaml's roster_slots, inlined so a config edit does not
# silently change what these tests assert.
ROSTER_SLOTS = {
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "IF": 1,
    "OF": 4,
    "UTIL": 2,
    "P": 9,
    "BN": 2,
    "IL": 2,
}


def _write(tmp_path: Path, name: str, payload: Any) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _roster_payload() -> dict[str, Any]:
    """The build plan's worked example: 2 teams, one IL, one numbered slot."""
    return {
        "snapshot_date": local_today().isoformat(),
        "teams": [
            {
                "name": USER_TEAM,
                "players": [
                    {"name": "Bobby Witt Jr.", "slot": "SS", "positions": "SS, IF, Util"},
                    {
                        "name": "Spencer Strider",
                        "slot": "IL",
                        "positions": "P, IL",
                        "status": "IL60",
                    },
                ],
            },
            {
                "name": "Send in the Cavalli",
                "players": [
                    {
                        "name": "Mookie Betts",
                        "slot": "OF2",
                        "positions": "2B, SS, IF, OF, Util",
                    }
                ],
            },
        ],
    }


def _real_standings_payload() -> dict[str, Any]:
    return copy.deepcopy(yaml.safe_load(REAL_STANDINGS.read_text(encoding="utf-8")))


def _standings_from_names(names: list[str]) -> Standings:
    return Standings(
        effective_date=local_today(),
        entries=[
            StandingsEntry(team_name=n, team_key="", rank=i + 1, stats=CategoryStats())
            for i, n in enumerate(names)
        ],
    )


# ------------------------------------------------------------- roster shape


def test_load_rosters_produces_weekly_history_row_shape(tmp_path: Path) -> None:
    path = _write(tmp_path, "rosters.yaml", _roster_payload())
    snap = load_manual_rosters(path)

    assert snap.snapshot_date == local_today()
    assert list(snap.rows_by_team) == [USER_TEAM, "Send in the Cavalli"]

    witt = snap.rows_by_team[USER_TEAM][0]
    # Exactly the keys refresh_pipeline writes into weekly_rosters_history.
    assert set(witt) == {"slot", "player_name", "positions", "status", "yahoo_id"}
    assert witt == {
        "slot": "SS",
        "player_name": "Bobby Witt Jr.",
        "positions": "SS, IF, Util",
        "status": "",
        "yahoo_id": "",
    }
    # yahoo_id stays blank on purpose: the audit falls back to player_key,
    # and inventing a Yahoo id would be a recalled identifier.
    assert all(r["yahoo_id"] == "" for rows in snap.rows_by_team.values() for r in rows)


def test_numbered_slot_collapses_to_base_slot(tmp_path: Path) -> None:
    path = _write(tmp_path, "rosters.yaml", _roster_payload())
    snap = load_manual_rosters(path)
    betts = snap.rows_by_team["Send in the Cavalli"][0]
    assert betts["slot"] == "OF"
    assert Position.parse(betts["slot"]) is Position.OF


def test_multi_position_string_parses_to_position_list(tmp_path: Path) -> None:
    path = _write(tmp_path, "rosters.yaml", _roster_payload())
    snap = load_manual_rosters(path)
    betts = snap.rows_by_team["Send in the Cavalli"][0]
    # Stored verbatim (Yahoo's casing included) so the round-trip through
    # League.from_redis matches the Yahoo-written rows byte for byte.
    assert betts["positions"] == "2B, SS, IF, OF, Util"
    assert Position.parse_list(betts["positions"]) == [
        Position.SECOND_BASE,
        Position.SS,
        Position.IF,
        Position.OF,
        Position.UTIL,
    ]


def test_il_player_with_bn_slot_is_detected_via_status(tmp_path: Path) -> None:
    payload = _roster_payload()
    payload["teams"][0]["players"].append(
        {"name": "Sandy Alcantara", "slot": "BN", "positions": "P", "status": "IL15"}
    )
    path = _write(tmp_path, "rosters.yaml", payload)
    snap = load_manual_rosters(path)

    parked = snap.rows_by_team[USER_TEAM][-1]
    assert parked["slot"] == "BN"
    # The ONLY signal that this bench player is hurt.
    assert parked["status"] == "IL15"


def test_status_defaults_to_empty_string_not_none(tmp_path: Path) -> None:
    path = _write(tmp_path, "rosters.yaml", _roster_payload())
    snap = load_manual_rosters(path)
    assert snap.rows_by_team[USER_TEAM][0]["status"] == ""


# ------------------------------------------------------- roster validation


@pytest.mark.parametrize("bad_slot", ["", "   "])
def test_blank_slot_rejected(tmp_path: Path, bad_slot: str) -> None:
    payload = _roster_payload()
    payload["teams"][0]["players"][0]["slot"] = bad_slot
    path = _write(tmp_path, "rosters.yaml", payload)
    with pytest.raises(ManualTranscriptError) as exc:
        load_manual_rosters(path)
    assert any("'slot' is required" in e for e in exc.value.errors)


def test_missing_slot_key_rejected(tmp_path: Path) -> None:
    payload = _roster_payload()
    del payload["teams"][0]["players"][0]["slot"]
    path = _write(tmp_path, "rosters.yaml", payload)
    with pytest.raises(ManualTranscriptError):
        load_manual_rosters(path)


def test_unknown_slot_rejected(tmp_path: Path) -> None:
    payload = _roster_payload()
    payload["teams"][0]["players"][0]["slot"] = "LF"
    path = _write(tmp_path, "rosters.yaml", payload)
    with pytest.raises(ManualTranscriptError) as exc:
        load_manual_rosters(path)
    assert any("bad slot" in e for e in exc.value.errors)


def test_missing_positions_rejected(tmp_path: Path) -> None:
    payload = _roster_payload()
    payload["teams"][0]["players"][0]["positions"] = ""
    path = _write(tmp_path, "rosters.yaml", payload)
    with pytest.raises(ManualTranscriptError) as exc:
        load_manual_rosters(path)
    assert any("'positions' is required" in e for e in exc.value.errors)


def test_duplicate_player_within_a_team_rejected(tmp_path: Path) -> None:
    payload = _roster_payload()
    payload["teams"][0]["players"].append(
        {"name": "bobby witt jr.", "slot": "BN", "positions": "SS, IF, Util"}
    )
    path = _write(tmp_path, "rosters.yaml", payload)
    with pytest.raises(ManualTranscriptError) as exc:
        load_manual_rosters(path)
    assert any("duplicate player" in e for e in exc.value.errors)


def test_the_same_player_on_two_teams_is_rejected(tmp_path: Path) -> None:
    """A player can only be on one roster, and the consequence of missing it is
    not a cosmetic duplicate.

    ``League.from_redis`` gives him to both teams and
    ``ProjectedStandings.from_rosters`` counts his rest-of-season line twice,
    inflating two teams' projected totals and shifting every roto comparison --
    including the DeltaRoto behind the audit's recommendations. Every per-team
    slot count stays legal, so nothing else catches it.

    Reachable the obvious way: a trade lands between two roster screenshots and
    the player is typed onto both the old team and the new one.
    """
    payload = _roster_payload()
    moved = dict(payload["teams"][0]["players"][0])
    moved["slot"] = "BN"
    payload["teams"][1]["players"].append(moved)
    path = _write(tmp_path, "rosters.yaml", payload)

    with pytest.raises(ManualTranscriptError) as exc:
        load_manual_rosters(path)

    joined = " ".join(exc.value.errors)
    assert "on more than one team" in joined
    assert payload["teams"][0]["name"] in joined and payload["teams"][1]["name"] in joined


def test_a_two_way_player_may_be_on_two_teams_as_his_two_halves(tmp_path: Path) -> None:
    """Ohtani is two rostered entities and this league splits them across teams.

    The cross-team check must therefore key on (name, player type) and not on the
    bare name -- keying on the name alone would reject the real 2026 rosters,
    where Work in Progress holds the batter and Tortured Baseball Department the
    pitcher.
    """
    payload = _roster_payload()
    payload["teams"][0]["players"].append(
        {"name": "Shohei Ohtani", "slot": "UTIL", "positions": "Util"}
    )
    payload["teams"][1]["players"].append({"name": "Shohei Ohtani", "slot": "P", "positions": "P"})
    path = _write(tmp_path, "rosters.yaml", payload)

    snapshot = load_manual_rosters(path)

    names = [r["player_name"] for rows in snapshot.rows_by_team.values() for r in rows]
    assert names.count("Shohei Ohtani") == 2


def test_future_snapshot_date_rejected(tmp_path: Path) -> None:
    payload = _roster_payload()
    payload["snapshot_date"] = (local_today() + timedelta(days=1)).isoformat()
    path = _write(tmp_path, "rosters.yaml", payload)
    with pytest.raises(ManualTranscriptError) as exc:
        load_manual_rosters(path)
    assert any("in the future" in e for e in exc.value.errors)


def test_every_problem_is_reported_not_just_the_first(tmp_path: Path) -> None:
    payload = _roster_payload()
    payload["teams"][0]["players"][0]["slot"] = ""
    payload["teams"][0]["players"][1]["positions"] = ""
    payload["teams"][1]["players"][0]["slot"] = "LF"
    path = _write(tmp_path, "rosters.yaml", payload)
    with pytest.raises(ManualTranscriptError) as exc:
        load_manual_rosters(path)
    assert len(exc.value.errors) == 3


def test_missing_file_is_a_transcript_error(tmp_path: Path) -> None:
    with pytest.raises(ManualTranscriptError):
        load_manual_rosters(tmp_path / "nope.yaml")


def test_shipped_roster_template_parses(tmp_path: Path) -> None:
    """The version-controlled template must always be a valid instance."""
    snap = load_manual_rosters(SHIPPED_ROSTERS)
    assert snap.rows_by_team
    assert all(
        set(row) == {"slot", "player_name", "positions", "status", "yahoo_id"}
        for rows in snap.rows_by_team.values()
        for row in rows
    )


# ------------------------------------------------------------- cross-checks


def test_roster_team_names_must_equal_standings_team_names(tmp_path: Path) -> None:
    path = _write(tmp_path, "rosters.yaml", _roster_payload())
    rosters = load_manual_rosters(path)
    # A trailing space is invisible in a screenshot and silently orphans a team.
    standings = _standings_from_names([USER_TEAM, "Send in the Cavalli "])
    errors = validate_transcripts(
        standings, rosters, team_name=USER_TEAM, roster_slots=ROSTER_SLOTS
    )
    assert any("Send in the Cavalli " in e for e in errors)
    assert any("Send in the Cavalli'" in e for e in errors)


def test_matching_team_names_produce_no_errors(tmp_path: Path) -> None:
    path = _write(tmp_path, "rosters.yaml", _roster_payload())
    rosters = load_manual_rosters(path)
    standings = _standings_from_names([USER_TEAM, "Send in the Cavalli"])
    assert (
        validate_transcripts(standings, rosters, team_name=USER_TEAM, roster_slots=ROSTER_SLOTS)
        == []
    )


def test_user_team_must_be_present(tmp_path: Path) -> None:
    payload = _roster_payload()
    payload["teams"] = payload["teams"][1:]
    path = _write(tmp_path, "rosters.yaml", payload)
    rosters = load_manual_rosters(path)
    standings = _standings_from_names(["Send in the Cavalli"])
    errors = validate_transcripts(
        standings, rosters, team_name=USER_TEAM, roster_slots=ROSTER_SLOTS
    )
    assert any("config team_name" in e and "standings.yaml" in e for e in errors)
    assert any("config team_name" in e and "rosters.yaml" in e for e in errors)


def test_slot_overflow_is_flagged(tmp_path: Path) -> None:
    payload = _roster_payload()
    payload["teams"][0]["players"] = [
        {"name": f"Outfielder {i}", "slot": f"OF{i}", "positions": "OF, Util"} for i in range(1, 7)
    ]
    path = _write(tmp_path, "rosters.yaml", payload)
    rosters = load_manual_rosters(path)
    standings = _standings_from_names([USER_TEAM, "Send in the Cavalli"])
    errors = validate_transcripts(
        standings, rosters, team_name=USER_TEAM, roster_slots=ROSTER_SLOTS
    )
    assert any("6 players in slot OF" in e for e in errors)


def _cavalli_shaped_team(name: str) -> dict[str, Any]:
    """A team shaped like the real Yahoo 2026-07-21 snapshot for this league.

    23 active players against ``ROSTER_SLOTS``' 23 active slots, but with an OF
    and a UTIL slot left EMPTY and the two displaced bats on a nominally
    2-deep bench: BN=4, OF=3, UTIL=1. This is what the Yahoo API actually
    returned for "Send in the Cavalli" in ``weekly_rosters_history``, so it is
    the shape a faithful transcription has to be allowed to express.
    """
    players: list[dict[str, Any]] = []
    for slot, n, positions in (
        ("C", 1, "C, Util"),
        ("1B", 1, "1B, IF, Util"),
        ("2B", 1, "2B, IF, Util"),
        ("3B", 1, "3B, IF, Util"),
        ("SS", 1, "SS, IF, Util"),
        ("IF", 1, "1B, IF, Util"),
        ("OF", 3, "OF, Util"),
        ("UTIL", 1, "OF, Util"),
        ("P", 9, "P"),
        ("BN", 4, "OF, Util"),
    ):
        for i in range(n):
            players.append({"name": f"{slot} Player {i}", "slot": slot, "positions": positions})
    return {"name": name, "players": players}


def test_an_overfull_bench_with_empty_starting_slots_is_accepted() -> None:
    """Yahoo's bench is elastic -- a hard BN cap would reject real data.

    Regression guard for a validator that refused the shipped transcription
    and blocked the whole manual run. The BN=2 in roster_slots is not a
    ceiling on how many players may sit on the bench; it is one term in the
    total active capacity, and Yahoo pushes whoever is not in a named slot
    onto the bench regardless.
    """
    rosters = ManualRosterSnapshot(
        snapshot_date=local_today(),
        rows_by_team={
            USER_TEAM: [
                {
                    "slot": p["slot"],
                    "player_name": p["name"],
                    "positions": p["positions"],
                    "status": "",
                    "yahoo_id": "",
                }
                for p in _cavalli_shaped_team(USER_TEAM)["players"]
            ]
        },
    )
    standings = _standings_from_names([USER_TEAM])
    assert (
        validate_transcripts(standings, rosters, team_name=USER_TEAM, roster_slots=ROSTER_SLOTS)
        == []
    )


def test_more_active_players_than_active_slots_is_still_flagged() -> None:
    """Exempting BN from the per-slot cap must not exempt it from the total.

    The invariant that replaces the BN cap has to keep catching a genuine
    transcription error -- a row typed twice, or a dropped player still
    listed -- which shows up as one more active body than the roster holds.
    """
    team = _cavalli_shaped_team(USER_TEAM)
    team["players"].append({"name": "One Too Many", "slot": "BN", "positions": "OF, Util"})
    rosters = ManualRosterSnapshot(
        snapshot_date=local_today(),
        rows_by_team={
            USER_TEAM: [
                {
                    "slot": p["slot"],
                    "player_name": p["name"],
                    "positions": p["positions"],
                    "status": "",
                    "yahoo_id": "",
                }
                for p in team["players"]
            ]
        },
    )
    standings = _standings_from_names([USER_TEAM])
    errors = validate_transcripts(
        standings, rosters, team_name=USER_TEAM, roster_slots=ROSTER_SLOTS
    )
    assert any("24 active players but roster_slots allows 23 active" in e for e in errors)


def test_il_slots_keep_their_hard_cap() -> None:
    """IL is a real fixed compartment -- only the bench is elastic."""
    team = _cavalli_shaped_team(USER_TEAM)
    team["players"] = team["players"][:5] + [
        {"name": f"Hurt {i}", "slot": "IL", "positions": "P, IL"} for i in range(3)
    ]
    rosters = ManualRosterSnapshot(
        snapshot_date=local_today(),
        rows_by_team={
            USER_TEAM: [
                {
                    "slot": p["slot"],
                    "player_name": p["name"],
                    "positions": p["positions"],
                    "status": "",
                    "yahoo_id": "",
                }
                for p in team["players"]
            ]
        },
    )
    standings = _standings_from_names([USER_TEAM])
    errors = validate_transcripts(
        standings, rosters, team_name=USER_TEAM, roster_slots=ROSTER_SLOTS
    )
    assert any("3 players in slot IL" in e for e in errors)


# --------------------------------------------------------------- standings


def test_standings_team_key_resolved_by_lookup_not_literal() -> None:
    """team_key comes from the caller's lookup table, never a literal."""
    payload = _real_standings_payload()
    names = [t["name"] for t in payload["teams"]]
    # Stand-in for what read_team_keys pulls off the seeded cache:standings.
    lookup = {name: f"mlb.l.5652.t.{i + 1}" for i, name in enumerate(names)}

    standings = load_manual_standings(REAL_STANDINGS, team_keys=lookup)
    by_team = standings.by_team()
    for name in names:
        assert by_team[name].team_key == lookup[name]


def test_standings_team_key_in_file_wins_over_lookup(tmp_path: Path) -> None:
    payload = _real_standings_payload()
    payload["teams"][0]["team_key"] = "from.the.file"
    path = _write(tmp_path, "standings.yaml", payload)
    standings = load_manual_standings(path, team_keys={payload["teams"][0]["name"]: "from.lookup"})
    assert standings.by_team()[payload["teams"][0]["name"]].team_key == "from.the.file"


def test_unresolvable_team_key_is_blank_not_invented(tmp_path: Path) -> None:
    payload = _real_standings_payload()
    path = _write(tmp_path, "standings.yaml", payload)
    standings = load_manual_standings(path, team_keys={})
    assert {e.team_key for e in standings.entries} == {""}


def test_ip_notation_preserved_verbatim() -> None:
    """Yahoo's innings.outs display format survives the load untouched.

    Scope, stated narrowly on purpose: this guards the LOADER, not the
    transcription. It cannot catch a mistyped IP -- 1094.0 entered as 1049.0 is
    a well-formed innings.outs value and no property of the file contradicts
    it. An earlier version of this docstring claimed otherwise.

    The expected digits come from the raw YAML TEXT rather than from the parsed
    payload, so the comparison is against what a human typed rather than against
    another read of the same parse. The previous literal (``1079.1``) pinned one
    snapshot's value and broke on the next re-transcription, which asserts the
    snapshot date rather than any behaviour.

    What is load-bearing: the digit after the point counts OUTS, so it is
    always 0, 1 or 2, and the loader must carry it through as itself rather
    than normalising ``.1`` into a decimal third.
    """
    raw = REAL_STANDINGS.read_text(encoding="utf-8")
    # name -> the IP literal exactly as typed, e.g. "1057.1"
    typed: dict[str, str] = {}
    current: str | None = None
    for line in raw.splitlines():
        name = re.search(r'^\s*-\s*name:\s*"(.+)"\s*$', line)
        if name:
            current = name.group(1)
        ip = re.search(r"\bIP:\s*([0-9]+\.[0-9]+)", line)
        if ip and current is not None:
            typed[current] = ip.group(1)

    by_team = load_manual_standings(REAL_STANDINGS, team_keys={}).by_team()
    assert set(typed) == set(by_team), (
        f"IP literals scraped from the YAML text {sorted(typed)} do not cover "
        f"every loaded team {sorted(by_team)}"
    )

    for team, literal in typed.items():
        whole_text, outs_text = literal.split(".")
        assert outs_text in ("0", "1", "2"), (
            f"{team}: IP {literal} is not Yahoo innings.outs notation -- the "
            "digit after the point counts outs"
        )
        loaded = by_team[team].extras[OpportunityStat.IP]
        # The loader must reproduce the typed digits, NOT convert them: an
        # outs digit of 1 stays .1 and never becomes .333.
        assert loaded == pytest.approx(float(literal)), (
            f"{team}: loader returned {loaded} for a typed {literal}"
        )
        if outs_text != "0":
            third = int(whole_text) + int(outs_text) / 3
            assert loaded != pytest.approx(third), (
                f"{team}: IP was normalised to decimal thirds ({third}); "
                "that diverges from the Yahoo path"
            )


def test_points_for_maps_onto_yahoo_points_for() -> None:
    payload = _real_standings_payload()
    standings = load_manual_standings(REAL_STANDINGS, team_keys={})
    by_team = standings.by_team()
    for row in payload["teams"]:
        assert by_team[row["name"]].yahoo_points_for == pytest.approx(float(row["points_for"]))


def test_league_points_for_totals_the_roto_maximum() -> None:
    standings = load_manual_standings(REAL_STANDINGS, team_keys={})
    n = len(standings.entries)
    total = sum(e.yahoo_points_for for e in standings.entries if e.yahoo_points_for is not None)
    assert total == pytest.approx(n * (n + 1) / 2 * 10)


def test_category_points_not_summing_to_points_for_is_rejected(tmp_path: Path) -> None:
    payload = _real_standings_payload()
    payload["teams"][0]["category_points"]["R"] += 1
    path = _write(tmp_path, "standings.yaml", payload)
    with pytest.raises(ManualTranscriptError) as exc:
        load_manual_standings(path, team_keys={})
    assert any("category_points sum to" in e and "points_for" in e for e in exc.value.errors)


def test_category_points_league_total_is_validated(tmp_path: Path) -> None:
    """Per-team consistency is not enough -- the league must balance too."""
    payload = _real_standings_payload()
    # Keep the team's own arithmetic self-consistent so ONLY the
    # league-wide check can catch it.
    payload["teams"][0]["category_points"]["R"] -= 1
    payload["teams"][0]["points_for"] -= 1
    path = _write(tmp_path, "standings.yaml", payload)
    with pytest.raises(ManualTranscriptError) as exc:
        load_manual_standings(path, team_keys={})
    assert any("category_points for R sum to 54" in e for e in exc.value.errors)


def test_category_points_out_of_range_is_rejected(tmp_path: Path) -> None:
    payload = _real_standings_payload()
    payload["teams"][0]["category_points"]["R"] = 11
    path = _write(tmp_path, "standings.yaml", payload)
    with pytest.raises(ManualTranscriptError) as exc:
        load_manual_standings(path, team_keys={})
    assert any("outside the legal roto range" in e for e in exc.value.errors)


def test_partially_transcribed_points_fields_are_rejected(tmp_path: Path) -> None:
    payload = _real_standings_payload()
    del payload["teams"][3]["category_points"]
    del payload["teams"][3]["points_for"]
    path = _write(tmp_path, "standings.yaml", payload)
    with pytest.raises(ManualTranscriptError) as exc:
        load_manual_standings(path, team_keys={})
    assert any("'points_for' is present on 9 of 10 teams" in e for e in exc.value.errors)
    assert any("'category_points' is present on 9 of 10 teams" in e for e in exc.value.errors)


def test_older_transcription_without_points_fields_still_loads(tmp_path: Path) -> None:
    payload = _real_standings_payload()
    for row in payload["teams"]:
        row.pop("points_for", None)
        row.pop("category_points", None)
    path = _write(tmp_path, "standings.yaml", payload)
    standings = load_manual_standings(path, team_keys={})
    assert len(standings.entries) == 10
    assert all(e.yahoo_points_for is None for e in standings.entries)


def test_missing_stat_category_is_rejected(tmp_path: Path) -> None:
    """CategoryStats.from_dict defaults silently; the loader must not."""
    payload = _real_standings_payload()
    del payload["teams"][0]["stats"]["SB"]
    path = _write(tmp_path, "standings.yaml", payload)
    with pytest.raises(ManualTranscriptError) as exc:
        load_manual_standings(path, team_keys={})
    assert any("'stats' is missing SB" in e for e in exc.value.errors)


# ------------------------------------- the regression guard on real numbers


def test_counting_category_points_re_derive_from_the_transcribed_stats() -> None:
    """R HR RBI SB W SV K must reproduce Yahoo's category_points exactly.

    Yahoo prints these seven with no rounding, so re-scoring the
    transcribed totals with the repo's own ``score_roto`` (team_sds=None,
    i.e. exact ranks with the averaged-ranks tie convention) has to agree
    with Yahoo digit for digit. A disagreement means a stat total was
    mistyped -- which would otherwise flow straight into the audit.

    The three rate categories are deliberately NOT checked: Yahoo shows
    AVG/ERA/WHIP rounded but ranks on full precision, which is exactly
    why ``points_for`` / ``category_points`` are transcribed at all.
    """
    payload = _real_standings_payload()
    standings = load_manual_standings(REAL_STANDINGS, team_keys={})
    expected = {row["name"]: row["category_points"] for row in payload["teams"]}

    scored = score_roto(standings)
    mismatches = []
    for name, points in scored.items():
        for cat in COUNTING_CATEGORIES:
            got = points.values[cat]
            want = float(expected[name][cat.value])
            if abs(got - want) > 1e-9:
                mismatches.append(f"{name} {cat.value}: derived {got} vs Yahoo {want}")
    assert mismatches == []


# -------------------------------------------------------------- exclusions


def test_fa_exclusions_absent_file_is_empty(tmp_path: Path) -> None:
    assert load_fa_exclusions(None) == frozenset()
    assert load_fa_exclusions(tmp_path / "nope.yaml") == frozenset()


def test_fa_exclusions_are_normalized(tmp_path: Path) -> None:
    path = _write(tmp_path, "fa_exclusions.yaml", {"names": ["  Jose Ramirez ", "MOOKIE BETTS"]})
    assert load_fa_exclusions(path) == frozenset({"jose ramirez", "mookie betts"})


def test_fa_exclusions_empty_names_key_is_empty(tmp_path: Path) -> None:
    path = _write(tmp_path, "fa_exclusions.yaml", {"names": None})
    assert load_fa_exclusions(path) == frozenset()


def test_fa_exclusions_bad_entry_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "fa_exclusions.yaml", {"names": ["Real Person", 42]})
    with pytest.raises(ManualTranscriptError):
        load_fa_exclusions(path)


def test_shipped_exclusions_file_loads() -> None:
    assert load_fa_exclusions(SHIPPED_EXCLUSIONS) == frozenset()


class TestYamlSyntaxErrorsArriveAsTranscriptErrors:
    """A bad indent is the likeliest mistake in a hand-typed thousand-line file.

    `yaml.YAMLError` is not a `ManualTranscriptError`, so it escaped the driver's
    handler and landed as a traceback -- the one thing that error path exists to
    prevent.
    """

    def test_a_malformed_roster_file(self, tmp_path):
        from fantasy_baseball.manual.transcripts import ManualTranscriptError, load_manual_rosters

        path = tmp_path / "rosters.yaml"
        path.write_text("teams:\n  - name: Alpha\n   players: []\n", encoding="utf-8")

        with pytest.raises(ManualTranscriptError) as excinfo:
            load_manual_rosters(path)

        assert any("not valid YAML" in e for e in excinfo.value.errors)
        # PyYAML marks the line and column; that is more useful than anything this
        # layer could write, so it must survive into the message.
        assert any("line" in e for e in excinfo.value.errors)

    def test_a_malformed_exclusions_file(self, tmp_path):
        from fantasy_baseball.manual.transcripts import ManualTranscriptError, load_fa_exclusions

        path = tmp_path / "fa_exclusions.yaml"
        path.write_text("names:\n  - A\n - B\n", encoding="utf-8")

        with pytest.raises(ManualTranscriptError) as excinfo:
            load_fa_exclusions(path)

        assert any("not valid YAML" in e for e in excinfo.value.errors)
