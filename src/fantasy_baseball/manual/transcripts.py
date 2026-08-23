"""Load and validate the hand-transcribed Yahoo YAML into repo types.

Three input files live under ``data/manual/``:

- ``standings.yaml``  -- the ROTO > Stats tab, one row per team.
- ``rosters.yaml``    -- the Team > Roster tab, one block per team.
- ``fa_exclusions.yaml`` (optional) -- names to keep out of the
  synthesized free-agent pool.

Everything here is pure I/O plus validation. No KV, no network, no
pipeline imports: a transcription typo must surface as an exception (or
an error string) BEFORE anything is written to the isolated KV store,
because a half-seeded store is worse than an unseeded one.

Parsing reuses the repo's existing primitives -- :meth:`Position.parse`,
:meth:`Position.parse_list`, :func:`normalize_name`,
:meth:`CategoryStats.from_dict` -- rather than hand-rolling slot or name
handling, so the manual rows are byte-compatible with what the Yahoo
path writes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from fantasy_baseball.models.positions import IL_SLOTS, Position
from fantasy_baseball.models.standings import CategoryStats, Standings, StandingsEntry
from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category, OpportunityStat
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.time_utils import local_today

log = logging.getLogger(__name__)

# Roto points are exact rationals (an integer, or x.5 on a tie), so a
# tight tolerance is right; anything looser hides a real typo.
_POINTS_TOLERANCE = 1e-6

# The slot column Yahoo prints on the roster page, for error messages.
_SLOT_HINT = "C 1B 2B 3B SS IF OF UTIL P BN IL"


class ManualTranscriptError(ValueError):
    """A transcription file is missing, malformed, or self-inconsistent.

    Carries the full list of problems rather than the first one: a
    hand-typed file usually has several, and fixing them one exception
    per run is miserable.
    """

    def __init__(self, path: Path | str, errors: list[str]) -> None:
        self.path = str(path)
        self.errors = list(errors)
        body = "\n".join(f"  - {e}" for e in self.errors)
        super().__init__(f"{self.path}: {len(self.errors)} transcription problem(s)\n{body}")


@dataclass(frozen=True)
class ManualRosterSnapshot:
    """One day's transcription of all ten Yahoo rosters.

    ``rows_by_team`` maps the exact Yahoo team name to rows already in
    the ``weekly_rosters_history`` shape the Yahoo pipeline writes
    (``slot`` / ``player_name`` / ``positions`` / ``status`` /
    ``yahoo_id``), so the seeder can hand them straight to
    ``redis_store.write_roster_snapshot`` with no further reshaping.

    ``yahoo_id`` is deliberately the empty string. The audit uses it only
    as a dedup key that falls back to ``player_key`` when it is blank,
    and inventing a Yahoo id would violate the ids-are-looked-up rule.
    """

    snapshot_date: date
    rows_by_team: dict[str, list[dict[str, str]]]


# ---------------------------------------------------------------- helpers


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file that must contain a top-level mapping."""
    if not path.exists():
        raise ManualTranscriptError(path, ["file does not exist"])
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raise ManualTranscriptError(path, ["file is empty"])
    if not isinstance(raw, Mapping):
        raise ManualTranscriptError(
            path, [f"top level must be a mapping, got {type(raw).__name__}"]
        )
    return dict(raw)


def _parse_date(value: Any, *, field: str, errors: list[str]) -> date | None:
    """Parse an ISO date that must not be in the future."""
    if value is None:
        errors.append(f"{field!r} is required (ISO date, e.g. '2026-08-22')")
        return None
    text = str(value).strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        errors.append(f"{field!r} must be an ISO date (YYYY-MM-DD), got {text!r}")
        return None
    today = local_today()
    if parsed > today:
        errors.append(f"{field!r} {parsed.isoformat()} is in the future (today is {today})")
        return None
    return parsed


def _as_float(value: Any, *, where: str, field: str, errors: list[str]) -> float | None:
    """Coerce a YAML scalar to float, rejecting bools and junk.

    ``isinstance(True, int)`` is True in Python, so bools are screened
    out explicitly -- an unquoted YAML ``on`` / ``no`` would otherwise
    silently become 1.0 / 0.0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{where}: {field} must be a number, got {value!r}")
        return None
    return float(value)


def _nonempty_str(value: Any, *, where: str, field: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{where}: {field!r} is required and must be a non-empty string")
        return None
    return value.strip()


# ---------------------------------------------------------------- rosters


def _parse_player(row: Any, *, team: str, index: int, errors: list[str]) -> dict[str, str] | None:
    """Turn one transcribed player block into a weekly_rosters_history row."""
    where = f"team {team!r} player #{index + 1}"
    if not isinstance(row, Mapping):
        errors.append(f"{where}: entry must be a mapping, got {type(row).__name__}")
        return None

    name = _nonempty_str(row.get("name"), where=where, field="name", errors=errors)
    if name is None:
        return None
    where = f"team {team!r} player {name!r}"

    slot_raw = row.get("slot")
    slot = ""
    if not isinstance(slot_raw, str) or not slot_raw.strip():
        errors.append(
            f"{where}: 'slot' is required and must be non-empty ({_SLOT_HINT}). "
            "A blank slot becomes selected_position=None, which the scoring layer "
            "counts as ACTIVE."
        )
    else:
        try:
            slot = Position.parse(slot_raw).value
        except ValueError as exc:
            errors.append(f"{where}: bad slot -- {exc}")

    positions_raw = row.get("positions")
    positions = ""
    if not isinstance(positions_raw, str) or not positions_raw.strip():
        errors.append(
            f"{where}: 'positions' is required -- the grey eligibility string, verbatim "
            "(e.g. '2B, SS, IF, Util')"
        )
    else:
        positions = positions_raw.strip()
        try:
            parsed = Position.parse_list(positions)
        except ValueError as exc:
            errors.append(f"{where}: bad positions {positions!r} -- {exc}")
            parsed = []
        if not parsed:
            errors.append(f"{where}: 'positions' parsed to an empty list ({positions!r})")
            positions = ""

    status_raw = row.get("status", "")
    status = ""
    if status_raw is None:
        status = ""
    elif isinstance(status_raw, str):
        status = status_raw.strip()
    else:
        errors.append(f"{where}: 'status' must be a string when present, got {status_raw!r}")

    if not slot or not positions:
        return None
    return {
        "slot": slot,
        "player_name": name,
        "positions": positions,
        "status": status,
        "yahoo_id": "",
    }


def load_manual_rosters(path: Path) -> ManualRosterSnapshot:
    """Load ``data/manual/rosters.yaml`` into weekly_rosters_history rows.

    Raises:
        ManualTranscriptError: listing every problem found, never just
            the first. Nothing is returned unless the file is clean.
    """
    raw = _load_yaml_mapping(path)
    errors: list[str] = []

    snapshot_date = _parse_date(raw.get("snapshot_date"), field="snapshot_date", errors=errors)

    teams_raw = raw.get("teams")
    if not isinstance(teams_raw, list) or not teams_raw:
        errors.append("'teams' must be a non-empty list of team blocks")
        teams_raw = []

    rows_by_team: dict[str, list[dict[str, str]]] = {}
    for i, team_raw in enumerate(teams_raw):
        where = f"teams[{i}]"
        if not isinstance(team_raw, Mapping):
            errors.append(f"{where}: must be a mapping, got {type(team_raw).__name__}")
            continue
        team = _nonempty_str(team_raw.get("name"), where=where, field="name", errors=errors)
        if team is None:
            continue
        if team in rows_by_team:
            errors.append(f"duplicate team block for {team!r}")
            continue

        players_raw = team_raw.get("players")
        if not isinstance(players_raw, list) or not players_raw:
            errors.append(f"team {team!r}: 'players' must be a non-empty list")
            rows_by_team[team] = []
            continue

        rows: list[dict[str, str]] = []
        seen_names: dict[str, str] = {}
        for j, player_raw in enumerate(players_raw):
            row = _parse_player(player_raw, team=team, index=j, errors=errors)
            if row is None:
                continue
            key = normalize_name(row["player_name"])
            if key in seen_names:
                errors.append(
                    f"team {team!r}: duplicate player {row['player_name']!r} "
                    f"(already transcribed as {seen_names[key]!r})"
                )
                continue
            seen_names[key] = row["player_name"]
            rows.append(row)
        rows_by_team[team] = rows

    if errors:
        raise ManualTranscriptError(path, errors)
    assert snapshot_date is not None  # unreachable otherwise: a miss appends an error
    return ManualRosterSnapshot(snapshot_date=snapshot_date, rows_by_team=rows_by_team)


# -------------------------------------------------------------- standings


def _parse_category_points(
    raw: Any, *, where: str, errors: list[str], num_teams: int
) -> dict[Category, float] | None:
    """Parse Yahoo's per-category points for one team.

    Present-but-broken is an error; absent is fine (an older
    transcription predates the field).
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        errors.append(f"{where}: 'category_points' must be a mapping, got {type(raw).__name__}")
        return None
    known = {c.value for c in ALL_CATEGORIES}
    unknown = sorted(str(k) for k in raw if str(k) not in known)
    if unknown:
        errors.append(f"{where}: 'category_points' has unknown categories {unknown}")
    out: dict[Category, float] = {}
    for cat in ALL_CATEGORIES:
        if cat.value not in raw:
            errors.append(f"{where}: 'category_points' is missing {cat.value}")
            continue
        val = _as_float(
            raw[cat.value], where=where, field=f"category_points.{cat.value}", errors=errors
        )
        if val is None:
            continue
        if val < 1.0 or val > float(num_teams):
            errors.append(
                f"{where}: category_points.{cat.value} = {val:g} is outside the legal "
                f"roto range 1..{num_teams}"
            )
            continue
        out[cat] = val
    if len(out) != len(ALL_CATEGORIES):
        return None
    return out


def _parse_standings_row(
    row: Any,
    *,
    index: int,
    num_teams: int,
    team_keys: Mapping[str, str],
    errors: list[str],
) -> tuple[StandingsEntry, dict[Category, float] | None] | None:
    """Parse one transcribed standings row into a :class:`StandingsEntry`."""
    where = f"teams[{index}]"
    if not isinstance(row, Mapping):
        errors.append(f"{where}: must be a mapping, got {type(row).__name__}")
        return None
    name = _nonempty_str(row.get("name"), where=where, field="name", errors=errors)
    if name is None:
        return None
    where = f"team {name!r}"

    rank_raw = row.get("rank")
    rank = 0
    if isinstance(rank_raw, bool) or not isinstance(rank_raw, int):
        errors.append(f"{where}: 'rank' is required and must be an integer, got {rank_raw!r}")
    else:
        rank = rank_raw

    stats_raw = row.get("stats")
    stats = CategoryStats()
    if not isinstance(stats_raw, Mapping):
        errors.append(f"{where}: 'stats' must be a mapping of the ten roto categories")
    else:
        clean: dict[str, float] = {}
        for cat in ALL_CATEGORIES:
            if cat.value not in stats_raw:
                errors.append(f"{where}: 'stats' is missing {cat.value}")
                continue
            val = _as_float(
                stats_raw[cat.value], where=where, field=f"stats.{cat.value}", errors=errors
            )
            if val is not None:
                clean[cat.value] = val
        if len(clean) == len(ALL_CATEGORIES):
            stats = CategoryStats.from_dict(clean)

    extras: dict[OpportunityStat, float] = {}
    extras_raw = row.get("extras")
    if extras_raw is not None:
        if not isinstance(extras_raw, Mapping):
            errors.append(f"{where}: 'extras' must be a mapping, got {type(extras_raw).__name__}")
        else:
            for key, value in extras_raw.items():
                try:
                    stat = OpportunityStat(str(key))
                except ValueError:
                    errors.append(f"{where}: unknown extras key {key!r}")
                    continue
                extra_val = _as_float(value, where=where, field=f"extras.{key}", errors=errors)
                if extra_val is not None:
                    extras[stat] = extra_val

    points_for: float | None = None
    if row.get("points_for") is not None:
        points_for = _as_float(row["points_for"], where=where, field="points_for", errors=errors)

    cat_points = _parse_category_points(
        row.get("category_points"), where=where, errors=errors, num_teams=num_teams
    )
    if cat_points is not None and points_for is not None:
        total = sum(cat_points.values())
        if abs(total - points_for) > _POINTS_TOLERANCE:
            errors.append(
                f"{where}: category_points sum to {total:g} but points_for is "
                f"{points_for:g} -- one of the two is mistyped"
            )

    team_key = ""
    row_key = row.get("team_key")
    if isinstance(row_key, str) and row_key.strip():
        team_key = row_key.strip()
    elif name in team_keys:
        team_key = team_keys[name]
    else:
        log.warning(
            "manual standings: no team_key for %r (absent from the file and from the "
            "seeded cache:standings); writing an empty key",
            name,
        )

    entry = StandingsEntry(
        team_name=name,
        team_key=team_key,
        rank=rank,
        stats=stats,
        yahoo_points_for=points_for,
        extras=extras,
    )
    return entry, cat_points


def _check_league_points(
    entries: list[StandingsEntry],
    cat_points_by_team: Mapping[str, dict[Category, float]],
    *,
    num_teams: int,
) -> list[str]:
    """League-wide arithmetic on the two optional points fields."""
    errors: list[str] = []
    if not entries:
        return errors

    with_points_for = [e for e in entries if e.yahoo_points_for is not None]
    if with_points_for and len(with_points_for) != len(entries):
        missing = sorted(e.team_name for e in entries if e.yahoo_points_for is None)
        errors.append(
            f"'points_for' is present on {len(with_points_for)} of {len(entries)} teams "
            f"-- transcribe it for all teams or none (missing: {missing})"
        )

    if not cat_points_by_team:
        return errors
    if len(cat_points_by_team) != len(entries):
        missing = sorted(e.team_name for e in entries if e.team_name not in cat_points_by_team)
        errors.append(
            f"'category_points' is present on {len(cat_points_by_team)} of {len(entries)} "
            f"teams -- transcribe it for all teams or none (missing: {missing})"
        )
        return errors

    per_category_total = float(num_teams * (num_teams + 1)) / 2.0
    for cat in ALL_CATEGORIES:
        got = sum(pts[cat] for pts in cat_points_by_team.values())
        if abs(got - per_category_total) > _POINTS_TOLERANCE:
            errors.append(
                f"category_points for {cat.value} sum to {got:g} across the league; "
                f"{num_teams} teams must split {per_category_total:g}"
            )

    grand = sum(sum(pts.values()) for pts in cat_points_by_team.values())
    expected = per_category_total * len(ALL_CATEGORIES)
    if abs(grand - expected) > _POINTS_TOLERANCE:
        errors.append(
            f"category_points sum to {grand:g} league-wide; {num_teams} teams over "
            f"{len(ALL_CATEGORIES)} categories must sum to {expected:g}"
        )
    return errors


def load_manual_standings(path: Path, *, team_keys: Mapping[str, str]) -> Standings:
    """Load ``data/manual/standings.yaml`` into a :class:`Standings`.

    ``points_for`` maps onto :attr:`StandingsEntry.yahoo_points_for`.
    Yahoo displays AVG/ERA/WHIP rounded but ranks on full precision, so
    re-deriving the total from the transcribed (rounded) stats splits
    every display tie at x.5 and disagrees with Yahoo's real standings by
    up to a full point per team.

    ``category_points`` is not carried on the entry (``StandingsEntry``
    has no field for it) but IS validated here, and that validation is
    what makes the transcription trustworthy: per team it must sum to
    ``points_for``, per category it must sum to ``n*(n+1)/2`` across the
    league, and the grand total must be ``n*(n+1)/2 * 10``.

    Both fields are optional so an older transcription still loads, but a
    partially filled-in file is an error rather than a silent half
    validation.

    Args:
        path: the YAML file.
        team_keys: name -> Yahoo team_key, resolved by the caller from
            the seeded ``cache:standings`` so the key is looked up and
            never hand-typed. A per-team ``team_key`` in the file wins;
            an unresolvable name gets ``""`` plus a log warning.

    Raises:
        ManualTranscriptError: listing every problem found.
    """
    raw = _load_yaml_mapping(path)
    errors: list[str] = []

    effective_date = _parse_date(raw.get("effective_date"), field="effective_date", errors=errors)

    teams_raw = raw.get("teams")
    if not isinstance(teams_raw, list) or not teams_raw:
        errors.append("'teams' must be a non-empty list of team rows")
        teams_raw = []
    num_teams = len(teams_raw)

    entries: list[StandingsEntry] = []
    cat_points_by_team: dict[str, dict[Category, float]] = {}
    seen: set[str] = set()
    for i, row in enumerate(teams_raw):
        parsed = _parse_standings_row(
            row, index=i, num_teams=num_teams, team_keys=team_keys, errors=errors
        )
        if parsed is None:
            continue
        entry, cat_points = parsed
        if entry.team_name in seen:
            errors.append(f"duplicate standings row for team {entry.team_name!r}")
            continue
        seen.add(entry.team_name)
        entries.append(entry)
        if cat_points is not None:
            cat_points_by_team[entry.team_name] = cat_points

    errors.extend(_check_league_points(entries, cat_points_by_team, num_teams=num_teams))

    if errors:
        raise ManualTranscriptError(path, errors)
    assert effective_date is not None  # unreachable otherwise: a miss appends an error
    return Standings(effective_date=effective_date, entries=entries)


# ------------------------------------------------------------- exclusions


def load_fa_exclusions(path: Path | None) -> frozenset[str]:
    """Load the optional free-agent exclusion list as normalized names.

    A missing path, a missing file, or an empty ``names`` list all mean
    "exclude nobody" -- absent is a valid state, not an error.
    """
    if path is None or not path.exists():
        return frozenset()
    raw = _load_yaml_mapping(path)
    names = raw.get("names")
    if names is None:
        return frozenset()
    if not isinstance(names, list):
        raise ManualTranscriptError(
            path, [f"'names' must be a list of player names, got {type(names).__name__}"]
        )
    errors: list[str] = []
    out: set[str] = set()
    for i, entry in enumerate(names):
        if not isinstance(entry, str) or not entry.strip():
            errors.append(f"names[{i}]: must be a non-empty string, got {entry!r}")
            continue
        out.add(normalize_name(entry))
    if errors:
        raise ManualTranscriptError(path, errors)
    return frozenset(out)


# ------------------------------------------------------------ cross-check


def validate_transcripts(
    standings: Standings,
    rosters: ManualRosterSnapshot,
    *,
    team_name: str,
    roster_slots: Mapping[str, int],
) -> list[str]:
    """Cross-check the two transcriptions against each other and config.

    The loaders each validate their own file in isolation; this catches
    what only shows up when the pair is viewed together. Returns a list
    of human-readable error strings -- empty means OK. It never raises
    and never writes anything, so the caller can print every problem at
    once and exit before touching the KV store.

    Team names are compared byte-for-byte on purpose. ``League.from_redis``
    joins rosters to standings on the raw name, so "SkeleThor " with a
    trailing space would produce a team with rosters but no standings and
    the audit would quietly score against nine opponents.
    """
    errors: list[str] = []

    standings_names = [e.team_name for e in standings.entries]
    roster_names = list(rosters.rows_by_team)
    missing = sorted(set(standings_names) - set(roster_names))
    extra = sorted(set(roster_names) - set(standings_names))
    if missing:
        errors.append(f"teams in standings.yaml but not in rosters.yaml: {missing}")
    if extra:
        errors.append(f"teams in rosters.yaml but not in standings.yaml: {extra}")
    if len(standings_names) != len(roster_names) and not (missing or extra):
        errors.append(
            f"standings.yaml has {len(standings_names)} teams but rosters.yaml has "
            f"{len(roster_names)}"
        )

    if team_name not in standings_names:
        errors.append(f"config team_name {team_name!r} is not in standings.yaml: {standings_names}")
    if team_name not in roster_names:
        errors.append(f"config team_name {team_name!r} is not in rosters.yaml: {roster_names}")

    limits: dict[Position, int] = {}
    for key, count in roster_slots.items():
        try:
            limits[Position.parse(key)] = int(count)
        except ValueError:
            errors.append(f"config roster_slots has an unparseable slot key {key!r}")

    # Total ACTIVE capacity: every configured non-IL slot, bench included.
    # This -- not a fixed BN count -- is the ceiling Yahoo actually enforces.
    active_capacity = sum(n for slot, n in limits.items() if slot not in IL_SLOTS)

    for team, rows in rosters.rows_by_team.items():
        used: dict[Position, int] = {}
        for row in rows:
            try:
                slot = Position.parse(row["slot"])
            except ValueError as exc:
                errors.append(f"team {team!r} player {row['player_name']!r}: {exc}")
                continue
            used[slot] = used.get(slot, 0) + 1
        for slot, count in sorted(used.items(), key=lambda kv: kv[0].value):
            if slot not in limits:
                errors.append(
                    f"team {team!r}: slot {slot.value} is not configured in roster_slots "
                    f"({sorted(k.value for k in limits)})"
                )
                continue
            # BN is EXEMPT from the per-slot cap; every other slot keeps it.
            #
            # Yahoo's bench is elastic: it holds whoever is not sitting in a
            # named starting slot, so leaving an OF and a UTIL slot empty puts
            # four bodies on a nominally 2-deep bench. That is not a
            # transcription error and not a hypothetical -- the real Yahoo API
            # reported exactly it for a team in this league
            # (weekly_rosters_history 2026-07-21: BN=4, OF=3, Util=1, 23
            # active, against roster_slots BN=2). A hard BN cap therefore
            # rejects FAITHFUL data and blocks the run, while catching nothing
            # a real transcription error would not also trip below.
            #
            # What Yahoo does enforce is the TOTAL, which is checked after this
            # loop -- so an actual extra body (a row typed twice, a dropped
            # player still listed) is still caught, just by the invariant that
            # holds rather than by one that does not.
            if slot == Position.BN:
                continue
            if count > limits[slot]:
                errors.append(
                    f"team {team!r}: {count} players in slot {slot.value} but roster_slots "
                    f"allows {limits[slot]} -- check the transcription"
                )

        active_used = sum(n for slot, n in used.items() if slot not in IL_SLOTS)
        if active_capacity and active_used > active_capacity:
            errors.append(
                f"team {team!r}: {active_used} active players but roster_slots allows "
                f"{active_capacity} active (bench included) -- check the transcription"
            )

    return errors
