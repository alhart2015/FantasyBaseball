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
Only games played while the player was in an ACTIVE slot (not BN/IL/IL+/DL/
DL+) count toward team AB. Yahoo's team AVG is computed over active-slot
ABs only; summing bench/IL games into the derived total would inflate
``H = AVG * AB`` downstream of ``ytd_components``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from fantasy_baseball.models.league import League
from fantasy_baseball.models.positions import BENCH_SLOTS, IL_SLOTS, Position
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.time_utils import local_today

# parents[0]=analysis/, [1]=fantasy_baseball/, [2]=src/, [3]=repo root
_DEFAULT_GAME_LOGS_PATH = Path(__file__).resolve().parents[3] / "data" / "roster_game_logs.json"


def _load_per_game_hitter_ab(
    game_logs: dict[str, Any] | None = None,
    path: Path = _DEFAULT_GAME_LOGS_PATH,
) -> dict[str, list[tuple[date, int]]]:
    """Return ``{normalized_name: [(game_date, ab), ...]}`` for hitters only.

    Pitcher entries and entries missing dates are skipped. AB values that
    fail to parse as int are skipped (defensive against malformed rows).
    """
    if game_logs is None:
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            game_logs = json.load(f)

    out: dict[str, list[tuple[date, int]]] = defaultdict(list)
    for _mid, entry in game_logs.items():
        if entry.get("type") != "hitter":
            continue
        name = entry.get("name") or ""
        if not name:
            continue
        norm = normalize_name(name)
        for game in entry.get("games", []):
            try:
                game_date = date.fromisoformat(game.get("date", ""))
                ab = int(game.get("ab", 0))
            except (TypeError, ValueError):
                continue
            out[norm].append((game_date, ab))
    return dict(out)


def _is_active_slot(slot: Position) -> bool:
    """A slot counts toward team AB only when it is neither bench nor IL.

    BENCH_SLOTS already includes IL_SLOTS in the canonical model, but the
    redundant IL check makes the intent explicit and survives any future
    refactor that splits the two sets.
    """
    return slot not in BENCH_SLOTS and slot not in IL_SLOTS


def compute_team_ytd_ab(
    league: League,
    season_start: date,
    season_end: date,
    *,
    today: date | None = None,
    game_logs: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Sum each team's YTD AB attributed by ownership window.

    For every team, every roster snapshot, every entry whose slot is active,
    walks ``Team.ownership_periods`` to get the ``(period_start, period_end)``
    window and sums ``ab`` from the player's per-game logs whose ``date``
    falls in ``[period_start, period_end)`` (half-open).

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
            if not _is_active_slot(entry.selected_position):
                continue
            games = name_to_games.get(normalize_name(entry.name), [])
            for game_date, ab in games:
                if period_start <= game_date < period_end:
                    ab_by_team[team.name] += ab
    return dict(ab_by_team)
