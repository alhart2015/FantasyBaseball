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

    Logs a warning when two distinct source names normalize to the same
    key (e.g. Will Smith C / Will Smith OF). There's no MLB-id to Yahoo-id
    disambiguator available, so the lists are still merged -- the warning
    just surfaces that per-team AB attribution may double-count for the
    affected players.
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

    out: dict[str, list[tuple[date, int]]] = defaultdict(list)
    seen_norm_names: dict[str, str] = {}  # norm -> first source name encountered
    collisions: list[tuple[str, str]] = []

    for _mid, entry in (game_logs or {}).items():
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "hitter":
            continue
        name = entry.get("name") or ""
        if not name:
            continue
        norm = normalize_name(name)

        # Track normalized-name collisions across distinct source names.
        prior = seen_norm_names.get(norm)
        if prior is not None and prior != name:
            collisions.append((prior, name))
        else:
            seen_norm_names[norm] = name

        for game in entry.get("games") or []:
            if not isinstance(game, dict):
                continue
            try:
                game_date = date.fromisoformat(game.get("date", ""))
                ab = int(game.get("ab", 0))
            except (TypeError, ValueError):
                continue
            out[norm].append((game_date, ab))

    if collisions:
        log.warning(
            "roster_game_logs.json has %d normalized-name collisions: %s. "
            "Per-team AB attribution may double-count for affected players "
            "(no MLB-id to Yahoo-id disambiguator available).",
            len(collisions),
            collisions[:5],
        )

    return dict(out)


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
