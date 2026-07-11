"""Section builders for the daily summary email.

Each builder is a pure function returning a typed section model (or an empty
list). Builders never raise for "no data" -- that is an empty section. They read
KV payloads the morning refresh produced, plus (for last night) per-player game
logs. No builder imports the streaks/dashboard module (it pulls in duckdb).
"""

from __future__ import annotations

from datetime import date
from typing import Any, cast

from fantasy_baseball.data.redis_store import get_player_game_log
from fantasy_baseball.summary.crosswalk import player_group
from fantasy_baseball.summary.models import (
    InjuryItem,
    LineupMove,
    PlayerLine,
    ProbableMatchup,
    StandingsDelta,
    StreakItem,
    TeamDelta,
)
from fantasy_baseball.utils.name_utils import normalize_name

_HITTER_FIELDS = ("pa", "ab", "h", "hr", "r", "rbi", "sb")
_PITCHER_FIELDS = ("ip", "k", "er", "bb", "w", "sv", "h_allowed")


def build_last_night(
    roster: list[dict[str, Any]],
    xmap: dict[tuple[str, str], int],
    client: Any,
    season: int,
    yesterday: date,
) -> tuple[list[PlayerLine], list[str]]:
    """Box-score lines for rostered players who played on ``yesterday``.

    Returns ``(lines, unmatched_names)``. A player whose name+type is not in the
    crosswalk goes into ``unmatched``; a matched player with no game row for
    ``yesterday`` is omitted (did not play).
    """
    lines: list[PlayerLine] = []
    unmatched: list[str] = []
    target = yesterday.isoformat()

    for entry in roster:
        name = entry.get("name", "")
        positions = entry.get("positions", []) or []
        groups = player_group(positions)

        # A two-way player resolves under whichever type namespace matches; the
        # same person-level MLBAM id serves both game-log groups.
        norm = normalize_name(name)
        mlbam: int | None = None
        for group in groups:
            key = (norm, "pitcher" if group == "pitching" else "hitter")
            if key in xmap:
                mlbam = xmap[key]
                break
        if mlbam is None:
            unmatched.append(name)
            continue

        for group in groups:
            log = get_player_game_log(client, season, str(mlbam), group)
            if not log:
                continue
            for row in log.get("games", []):
                if row.get("date") != target:
                    continue
                fields = _HITTER_FIELDS if group == "hitting" else _PITCHER_FIELDS
                stats = {f: _num(row.get(f)) for f in fields}
                lines.append(PlayerLine(name=name, group=group, stats=stats))

    return lines, unmatched


def build_streaks(streak_payload: dict[str, Any] | None) -> list[StreakItem]:
    """Hot/cold hitter streaks from the serialized STREAK_SCORES report.

    Reads the serialized dict directly (no duckdb import). Hitters-only,
    single-window -- matches the underlying report; emits one item per category
    labelled "hot" or "cold".
    """
    if not streak_payload:
        return []
    items: list[StreakItem] = []
    for row in streak_payload.get("roster_rows", []):
        name = row.get("name", "")
        for category, score in (row.get("scores") or {}).items():
            label = score.get("label")
            if label not in ("hot", "cold"):
                continue
            prob = score.get("probability")
            items.append(
                StreakItem(
                    name=name,
                    category=str(category),
                    label=str(label),
                    probability=float(prob) if prob is not None else 0.0,
                )
            )
    return items


def build_lineup_moves(optimal_payload: dict[str, Any] | None) -> list[LineupMove]:
    """Flatten LINEUP_OPTIMAL["moves"] into start/sit LineupMove rows."""
    if not optimal_payload:
        return []
    moves = optimal_payload.get("moves") or {}
    out: list[LineupMove] = []

    def _start(row: dict[str, Any]) -> LineupMove:
        rd = row.get("roto_delta")
        return LineupMove(
            player=row.get("player", ""),
            action="start",
            from_slot=row.get("from", ""),
            to_slot=row.get("to", ""),
            roto_delta=float(rd) if rd is not None else 0.0,
        )

    def _sit(row: dict[str, Any]) -> LineupMove:
        return LineupMove(
            player=row.get("player", ""),
            action="sit",
            from_slot=row.get("from", ""),
            to_slot=row.get("to", ""),
            roto_delta=0.0,
        )

    for swap in moves.get("swaps", []):
        if swap.get("start"):
            out.append(_start(swap["start"]))
        if swap.get("bench"):
            out.append(_sit(swap["bench"]))
    for row in moves.get("unpaired_starts", []):
        out.append(_start(row))
    for row in moves.get("unpaired_benches", []):
        out.append(_sit(row))
    return out


def build_injuries(injury_rows: list[dict[str, Any]]) -> list[InjuryItem]:
    """Map fetch_injuries rows to InjuryItem (injury_note carries the news)."""
    return [
        InjuryItem(
            name=row.get("name", ""),
            status=row.get("status", ""),
            note=row.get("injury_note", "") or "",
        )
        for row in injury_rows
    ]


def build_probables(probable_rows: list[dict[str, Any]] | None) -> list[ProbableMatchup]:
    """Map PROBABLE_STARTERS rollup rows to ProbableMatchup. Absent -> []."""
    if not probable_rows:
        return []
    out: list[ProbableMatchup] = []
    for row in probable_rows:
        starts = row.get("starts")
        out.append(
            ProbableMatchup(
                pitcher=row.get("pitcher", ""),
                starts=int(starts) if starts is not None else 0,
                days=row.get("days", ""),
                opponents=row.get("opponents", ""),
                quality=row.get("matchup_quality", ""),
            )
        )
    return out


def build_standings_delta(
    current_raw: dict[str, Any] | None,
    snapshot_payload: dict[str, Any] | None,
    user_team_name: str,
) -> StandingsDelta:
    """Overnight roto movement vs. the prior snapshot.

    Reconstructs both standings and re-scores per-category roto points (the
    stored payload holds raw totals, not place points). Freshness is enforced
    up-front by the orchestrator; this function assumes current is fresh.
    """
    from fantasy_baseball.models.standings import Standings
    from fantasy_baseball.scoring import score_roto

    if current_raw is None or snapshot_payload is None:
        return StandingsDelta(is_first_run=True, user_team_name=user_team_name)

    current = Standings.from_json(current_raw)
    prior = Standings.from_json(snapshot_payload["standings"])

    cur_roto = score_roto(cast("Any", current))
    prev_roto = score_roto(cast("Any", prior))
    # Join current and prior on the STABLE team_key, not the mutable display
    # name: a team renamed between the snapshot and today would otherwise miss
    # its prior row and silently vanish from the delta (score_roto keys by name).
    cur_key_of = {e.team_name: e.team_key for e in current.entries}
    prev_name_of = {e.team_key: e.team_name for e in prior.entries}
    cur_rank = {e.team_key: e.rank for e in current.entries}
    prev_rank = {e.team_key: e.rank for e in prior.entries}

    teams: list[TeamDelta] = []
    for name, cur_points in cur_roto.items():
        key = cur_key_of.get(name)
        prev_name = prev_name_of.get(key) if key is not None else None
        prev_points = prev_roto.get(prev_name) if prev_name is not None else None
        if key is None or prev_points is None:
            continue
        cat_delta = {
            str(getattr(cat, "value", cat)): cur_points.values[cat]
            - prev_points.values.get(cat, 0.0)
            for cat in cur_points.values
        }
        teams.append(
            TeamDelta(
                name=name,
                rank_prev=prev_rank.get(key, cur_rank.get(key, 0)),
                rank_now=cur_rank.get(key, 0),
                points_prev=prev_points.total,
                points_now=cur_points.total,
                category_points_delta=cat_delta,
            )
        )

    return StandingsDelta(
        is_first_run=False,
        user_team_name=user_team_name,
        teams=teams,
    )


def _num(value: Any) -> float:
    return float(value) if value is not None else 0.0
