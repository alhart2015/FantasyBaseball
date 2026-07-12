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
    CategoryEroto,
    InjuryItem,
    LineupMove,
    PlayerLine,
    ProbableMatchup,
    ProjectionDelta,
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
    prev_points_by_key = {
        e.team_key: prev_roto[e.team_name] for e in prior.entries if e.team_name in prev_roto
    }
    prev_rank = {e.team_key: e.rank for e in prior.entries}

    teams: list[TeamDelta] = []
    for e in current.entries:
        cur_points = cur_roto.get(e.team_name)
        prev_points = prev_points_by_key.get(e.team_key)
        if cur_points is None or prev_points is None:
            continue
        cat_delta = {
            str(getattr(cat, "value", cat)): cur_points.values[cat]
            - prev_points.values.get(cat, 0.0)
            for cat in cur_points.values
        }
        teams.append(
            TeamDelta(
                name=e.team_name,
                rank_prev=prev_rank.get(e.team_key, e.rank),
                rank_now=e.rank,
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


def _eroto_for(
    proj: dict[str, Any] | None, user_team_name: str
) -> tuple[dict[str, float], float] | None:
    """Per-category expected-roto points + total for the user's team on the
    PROJECTED end-of-season standings (Gaussian score_roto with team SDs), or
    None if the projections payload is missing/invalid or the team is absent.
    Matches the dashboard's format_standings_for_display(..., team_sds=...)."""
    if not proj:
        return None
    ps_json = proj.get("projected_standings")
    sds_json = proj.get("team_sds")
    if not ps_json or not sds_json:
        return None

    from fantasy_baseball.models.standings import ProjectedStandings
    from fantasy_baseball.scoring import score_roto, team_sds_from_json

    proj_standings = ProjectedStandings.from_json(ps_json)
    team_sds = team_sds_from_json(sds_json)
    roto = score_roto(cast("Any", proj_standings), team_sds=team_sds)
    mine = roto.get(user_team_name)
    if mine is None:
        return None
    # score_roto builds values in canonical category order; preserve it.
    values = {str(getattr(cat, "value", cat)): float(v) for cat, v in mine.values.items()}
    return values, float(mine.total)


def _champ_pct(mc: dict[str, Any] | None, user_team_name: str) -> float | None:
    """Monte Carlo championship odds (first_pct) for the user's team, or None."""
    if not mc:
        return None
    team_results = (mc.get("rest_of_season") or {}).get("team_results") or {}
    res = team_results.get(user_team_name)
    if not res:
        return None
    fp = res.get("first_pct")
    return float(fp) if fp is not None else None


def build_projection_delta(
    proj_current: dict[str, Any] | None,
    proj_prev: dict[str, Any] | None,
    mc_current: dict[str, Any] | None,
    mc_prev: dict[str, Any] | None,
    user_team_name: str,
) -> ProjectionDelta:
    """Projected-finish movement for the user's team: per-category eRoto (on the
    projected EOS standings) and MC championship odds, each with its overnight
    change vs. the prior snapshot. Freshness is enforced up-front; both current
    payloads are assumed fresh.
    """
    cur = _eroto_for(proj_current, user_team_name)
    prev = _eroto_for(proj_prev, user_team_name)
    champ_now = _champ_pct(mc_current, user_team_name)
    champ_prev = _champ_pct(mc_prev, user_team_name)
    first_run = prev is None and champ_prev is None

    if cur is None:
        # No current eRoto (projections absent) -- still surface champ odds if present.
        return ProjectionDelta(
            is_first_run=first_run, champ_pct_now=champ_now, champ_pct_prev=champ_prev
        )

    cur_vals, cur_total = cur
    prev_vals = prev[0] if prev else None
    eroto = [
        CategoryEroto(category=cat, now=now, prev=(prev_vals.get(cat) if prev_vals else None))
        for cat, now in cur_vals.items()
    ]
    return ProjectionDelta(
        is_first_run=first_run,
        eroto=eroto,
        eroto_total_now=cur_total,
        eroto_total_prev=(prev[1] if prev else None),
        champ_pct_now=champ_now,
        champ_pct_prev=champ_prev,
    )


def _num(value: Any) -> float:
    return float(value) if value is not None else 0.0
