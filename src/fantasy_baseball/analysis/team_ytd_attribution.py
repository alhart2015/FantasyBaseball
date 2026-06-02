"""Team-YTD stat attribution via ownership-period x per-game log intersection.

Same attribution model as :mod:`fantasy_baseball.analysis.spoe` -- walk
``Team.ownership_periods()`` and credit each (player, window) to its team --
but sums ACTUAL per-game AB from ``data/roster_game_logs.json`` instead of
SPoE's projection-scaled expected stats.

Used by :mod:`fantasy_baseball.web.refresh_pipeline` to populate
``StandingsEntry.extras[OpportunityStat.AB]`` before
``ProjectedStandings.from_rosters`` so the team-YTD projection can recombine
AVG correctly.

The companion file ``data/roster_game_logs.json`` is the canonical per-game
source; it carries hitter games with ``date``, ``ab``, ``h``, ``pa``, etc.

Active-slot filter
------------------
Only games played while the player was in an ACTIVE HITTER slot (not BN/IL/
IL+/DL/DL+ and not P/SP/RP) count toward team AB. Yahoo's team AVG is
computed over active-slot ABs only; summing bench/IL games into the derived
total would inflate ``H = AVG * AB`` downstream of ``ytd_components``.

Pitcher slots are excluded too: name-normalized lookups against the
hitter-only per-game dict would otherwise attribute Ohtani's hitter ABs to
his pitcher slot (double count).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from fantasy_baseball.models.league import League
from fantasy_baseball.models.positions import BENCH_SLOTS, IL_SLOTS, Position
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import PITCHER_POSITIONS
from fantasy_baseball.utils.time_utils import local_today

log = logging.getLogger(__name__)

# parents[0]=analysis/, [1]=fantasy_baseball/, [2]=src/, [3]=repo root
_DEFAULT_GAME_LOGS_PATH = Path(__file__).resolve().parents[3] / "data" / "roster_game_logs.json"


def _load_per_game_hitter_ab(
    game_logs: dict[str, Any] | None = None,
    path: Path = _DEFAULT_GAME_LOGS_PATH,
) -> dict[str, list[tuple[date, int]]]:
    """Return ``{normalized_name: [(game_date, ab), ...]}`` for hitters only.

    Pitcher entries and entries missing dates are skipped. AB values that
    fail to parse as int are skipped (defensive against malformed rows).

    Defensive against on-disk corruption:
    - missing file -> log warning, return {}
    - corrupt JSON -> log warning, return {}
    - non-dict top-level entry -> skip
    - ``games: null`` -> treat as empty
    - missing name -> skip

    Normalized-name collisions (two distinct ``mlbam_id``s whose names
    normalize to the same key -- e.g. the two MLB "Will Smith"s, or an
    accented vs ASCII spelling) are EXCLUDED from attribution, not merged.
    The Yahoo roster side has only a name, so a colliding name can't be
    resolved to one player; merging both players' games would credit a team
    owning "that name" with the union of two players' ABs (a silent
    double-count). Excluding is a logged undercount instead -- preferable to
    a silent inflation per the repo's "no plausible-wrong answer" rule.
    Detection is by ``mlbam_id`` (the game-log key), so two players sharing
    an identical name string are caught too -- the prior name-string compare
    missed those.
    """
    if game_logs is None:
        if not path.exists():
            log.warning(
                "Game logs file not found at %s; team YTD AB will be zero",
                path,
            )
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                game_logs = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Failed to load game logs from %s: %s", path, exc)
            return {}

    # Accumulate games per (normalized name, mlbam_id) so a normalized name
    # mapping to more than one distinct id can be detected and excluded rather
    # than silently merged. ``ids_by_norm`` keeps the source names for the
    # warning.
    games_by_norm_id: dict[str, dict[str, list[tuple[date, int]]]] = defaultdict(dict)
    names_by_norm_id: dict[str, dict[str, str]] = defaultdict(dict)

    for mid, entry in (game_logs or {}).items():
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "hitter":
            continue
        name = entry.get("name") or ""
        if not name:
            continue
        norm = normalize_name(name)
        names_by_norm_id[norm][mid] = name

        games = games_by_norm_id[norm].setdefault(mid, [])
        for game in entry.get("games") or []:
            if not isinstance(game, dict):
                continue
            try:
                game_date = date.fromisoformat(game.get("date", ""))
                ab = int(game.get("ab", 0))
            except (TypeError, ValueError):
                continue
            games.append((game_date, ab))

    out: dict[str, list[tuple[date, int]]] = {}
    collisions: list[tuple[str, list[str]]] = []
    for norm, by_id in games_by_norm_id.items():
        if len(by_id) > 1:
            # Ambiguous: can't map the Yahoo name to one of these ids. Exclude
            # rather than merge (which would double-count). Record for warning.
            collisions.append((norm, sorted(names_by_norm_id[norm].values())))
            continue
        (games,) = by_id.values()
        if games:
            out[norm] = games

    if collisions:
        log.warning(
            "Game logs have %d normalized-name collisions (multiple mlbam_ids "
            "per name); these names are EXCLUDED from team AB attribution to "
            "avoid double-counting (no MLB-id to Yahoo-id disambiguator): %s",
            len(collisions),
            collisions[:5],
        )

    return out


def _is_active_hitter_slot(slot: Position) -> bool:
    """A slot counts toward team AB only when it is an active HITTER slot.

    Excludes bench (BN), all IL flavors (IL/IL+/DL/DL+), and every pitcher
    slot (P/SP/RP). The pitcher exclusion is load-bearing for Ohtani-like
    cases: ``_load_per_game_hitter_ab`` returns a hitter-only dict; a
    pitcher entry whose normalized name collides with a hitter would
    otherwise silently inherit the hitter's ABs (double count).

    BENCH_SLOTS already includes IL_SLOTS in the canonical model, but the
    redundant IL check makes the intent explicit and survives any future
    refactor that splits the two sets.
    """
    return slot not in BENCH_SLOTS and slot not in IL_SLOTS and slot not in PITCHER_POSITIONS


def compute_team_ytd_ab(
    league: League,
    season_start: date,
    season_end: date,
    *,
    today: date | None = None,
    game_logs: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Sum each team's YTD AB attributed by ownership window.

    For every team, every roster snapshot, every entry whose slot is an
    active hitter slot, walks ``Team.ownership_periods`` to get the
    ``(period_start, period_end)`` window and sums ``ab`` from the player's
    per-game logs whose ``date`` falls in the window. Windows are
    half-open ``[period_start, period_end)`` EXCEPT the last (current)
    window, where ``period_end == today``; for that window the comparison
    is closed-right ``[period_start, period_end]`` so today's completed
    games are included (Yahoo's stats.avg already counts them, and an
    underderived AB would inflate ``H = AVG * AB``).

    Adjacent windows are ``[s1, e1)`` and ``[e1, e2)`` (the next snapshot's
    effective_date is the previous window's end). Using ``<=`` only on the
    last window therefore cannot double-count: every non-last window is
    strictly half-open, so a game on the boundary date e1 attributes to
    the next window (the one whose ``period_start == e1``).

    Returns a dict keyed by team name. Every team in ``league.teams`` appears
    in the output, even teams with zero attributed AB.

    Pre-loaded ``game_logs`` (the parsed ``data/roster_game_logs.json``
    payload) may be passed to skip the disk read; tests use this. Production
    calls leave it as ``None`` to load from the default path.
    """
    today = today or local_today()
    name_to_games = _load_per_game_hitter_ab(game_logs)

    ab_by_team: dict[str, float] = defaultdict(float)
    for team in league.teams:
        # Ensure every team appears in the output, even with zero AB.
        ab_by_team[team.name] += 0.0
        for entry, period_start, period_end in team.ownership_periods(
            season_start=season_start,
            season_end=season_end,
            today=today,
        ):
            if not _is_active_hitter_slot(entry.selected_position):
                continue
            games = name_to_games.get(normalize_name(entry.name), [])
            # Closed-right comparison ONLY on the last (current) window so
            # today's completed games attribute to today's owner. Every
            # other window stays half-open to prevent double-count on the
            # adjacent boundary.
            is_last_window = period_end == today
            for game_date, ab in games:
                if game_date < period_start:
                    continue
                if is_last_window:
                    if game_date > period_end:
                        continue
                else:
                    if game_date >= period_end:
                        continue
                ab_by_team[team.name] += ab
    return dict(ab_by_team)
