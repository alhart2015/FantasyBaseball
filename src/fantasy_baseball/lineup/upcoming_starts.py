"""Build upcoming projected starts for roster pitchers.

For each in-scope starting pitcher we find the rotation anchor (their
most recent past start within a 14-day lookback) and project the next
starts at every 5th team game until past the scoring window. MLB-
announced probable starters override projections for the same game.

Each start is decorated with a difficulty color (Tough/Fair/Great) based
on a park-adjusted rank of the opponent's offense. See ``_matchup_quality``
for the ranking math.

Public API:
    build_team_game_index(probable_pitchers, team_abbrev) -> list[GameSlot]
    find_anchor_index(team_games, pitcher_name, today) -> int | None
    project_start_indices(anchor_index, total_games, step=5) -> list[int]
    compose_pitcher_entries(...)  -> list[StartEntry]
    MatchupContext (dataclass with the precomputed ranking distributions)

All functions are pure -- no I/O, no global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from datetime import datetime as _datetime
from typing import Any

import pandas as pd

from fantasy_baseball.data.park_factors import get_park_factor
from fantasy_baseball.utils.name_utils import normalize_name


@dataclass(frozen=True)
class GameSlot:
    """One scheduled team-game from the perspective of a single team.

    ``announced_starter`` is the name MLB has listed (or empty string if
    "TBD" / unset). For completed past games, MLB populates this with
    the actual starter, which is the signal used to find rotation anchors.
    """

    date: str  # YYYY-MM-DD
    game_number: int  # 1 for normal games, >1 for second game of doubleheader
    opponent: str  # FanGraphs-normalized opponent abbreviation
    indicator: str  # "@" if away, "vs" if home
    announced_starter: str = ""


@dataclass
class StartEntry:
    """One projected or announced start for a roster pitcher."""

    date: str
    day: str  # "Mon", "Tue", ...
    opponent: str
    indicator: str
    announced: bool = False
    matchup_quality: str = "Fair"  # "Great" | "Fair" | "Tough"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchupContext:
    """Precomputed league distributions used to color each start.

    Built once per refresh by ``matchups.get_probable_starters`` and
    passed into every ``compose_pitcher_entries`` call.

    - ``team_stats`` keeps the raw season OPS/K% per team so the tooltip
      can show a recognizable "season OPS .740" alongside the
      park-adjusted color.
    - ``neutral_ops`` / ``neutral_k_pct`` are each team's stats with
      their home-park inflation backed out, so the park factor at the
      *actual* venue can be re-applied cleanly.
    - ``neutral_ops_sorted_desc`` / ``neutral_k_pct_sorted_asc`` form the
      league baseline against which each start's park-adjusted value is
      ranked. OPS descending so rank 1 is highest (toughest); K% ascending
      so rank 1 is lowest K% (toughest -- contact-heavy lineup).
    - ``ops_rank_map`` / ``k_rank_map`` are the raw season ranks, kept
      around for the tooltip so the user can see how the team ranks
      before the park nudge.
    """

    team_stats: dict[str, dict[str, float]]
    neutral_ops: dict[str, float]
    neutral_k_pct: dict[str, float]
    neutral_ops_sorted_desc: tuple[float, ...]
    neutral_k_pct_sorted_asc: tuple[float, ...]
    ops_rank_map: dict[str, int]
    k_rank_map: dict[str, int]


def build_team_game_index(
    probable_pitchers: list[dict[str, Any]],
    team_abbrev: str,
) -> list[GameSlot]:
    """Filter the league-wide probable_pitchers list to one team's games.

    Returns a chronological list (by date, then game_number). Each
    entry exposes the opponent and the announced starter for that team.
    """
    slots: list[GameSlot] = []
    for game in probable_pitchers:
        if game["away_team"] == team_abbrev:
            opponent = game["home_team"]
            indicator = "@"
            starter = game.get("away_pitcher", "") or ""
        elif game["home_team"] == team_abbrev:
            opponent = game["away_team"]
            indicator = "vs"
            starter = game.get("home_pitcher", "") or ""
        else:
            continue

        if starter == "TBD":
            starter = ""

        slots.append(
            GameSlot(
                date=game["date"],
                game_number=int(game.get("game_number", 1) or 1),
                opponent=opponent,
                indicator=indicator,
                announced_starter=starter,
            )
        )

    slots.sort(key=lambda s: (s.date, s.game_number))
    return slots


def find_anchor_index(
    team_games: list[GameSlot],
    pitcher_name: str,
    today: date,
) -> int | None:
    """Most recent index in ``team_games`` where ``pitcher_name`` started.

    Only considers games strictly before ``today``. Name comparison is
    accent/case-insensitive (delegates to normalize_name). Returns
    ``None`` if the pitcher has no eligible past start in the index.
    """
    target = normalize_name(pitcher_name)
    today_iso = today.isoformat()
    anchor: int | None = None
    for i, slot in enumerate(team_games):
        if slot.date >= today_iso:
            continue
        if not slot.announced_starter:
            continue
        if normalize_name(slot.announced_starter) == target:
            anchor = i
    return anchor


def project_start_indices(
    anchor_index: int,
    total_games: int,
    step: int = 5,
) -> list[int]:
    """Return the projected start indices in the team's game stream.

    Starts at ``anchor_index + step`` and steps by ``step`` until
    exceeding ``total_games - 1``. Returns an empty list if anchor_index
    is negative.
    """
    if anchor_index < 0:
        return []
    indices: list[int] = []
    nxt = anchor_index + step
    while nxt < total_games:
        indices.append(nxt)
        nxt += step
    return indices


def _day_name(date_iso: str) -> str:
    return _datetime.strptime(date_iso, "%Y-%m-%d").strftime("%a")


def _venue_for(pitcher_team: str, opponent: str, indicator: str) -> str:
    """The team whose home park hosts a given start.

    ``@`` means our pitcher is on the road -- venue is opponent's park.
    ``vs`` means our pitcher is home -- venue is our pitcher's team park.
    """
    return opponent if indicator == "@" else pitcher_team


def _rank_against(value: float, sorted_values: tuple[float, ...], descending: bool) -> int:
    """Rank ``value`` (1..30) inside a baseline distribution.

    For descending baselines (OPS): rank 1 = highest, so we count how
    many baseline entries are strictly greater than ``value``.
    For ascending baselines (K%): rank 1 = lowest, so we count how many
    are strictly less.

    Naturally clamps to [1, N+1]; callers can clip to [1, 30] if they
    want a strict 1-30 band, but for color bucketing values outside the
    league band correctly fall into the most extreme bucket anyway.
    """
    if descending:
        return 1 + sum(1 for v in sorted_values if v > value)
    return 1 + sum(1 for v in sorted_values if v < value)


def _matchup_quality(
    opponent: str,
    venue: str,
    ctx: MatchupContext,
) -> tuple[str, int, int]:
    """Color-band, effective OPS rank, effective K rank for one start.

    Park-neutralizes the opponent's season stats (already done in
    ``ctx``), applies the venue's park factor, ranks the effective
    values against the league-wide park-neutral baseline, then buckets
    the average rank into Tough (1-10) / Fair (11-20) / Great (21-30).

    Returns ``("Fair", 0, 0)`` when the opponent isn't in the rank
    context -- typically very early in the season before any team
    batting data has been pulled.
    """
    if opponent not in ctx.neutral_ops or opponent not in ctx.neutral_k_pct:
        return ("Fair", 0, 0)

    venue_ops_pf = get_park_factor(venue, "ops")
    venue_k_pf = get_park_factor(venue, "k")

    effective_ops = ctx.neutral_ops[opponent] * venue_ops_pf
    effective_k = ctx.neutral_k_pct[opponent] * venue_k_pf

    ops_rank = _rank_against(effective_ops, ctx.neutral_ops_sorted_desc, descending=True)
    k_rank = _rank_against(effective_k, ctx.neutral_k_pct_sorted_asc, descending=False)
    # Clip to 1..30 so extreme values (e.g. Rockies-at-Coors lands above
    # the league's #1 park-neutral OPS) display cleanly in the tooltip.
    ops_rank = max(1, min(30, ops_rank))
    k_rank = max(1, min(30, k_rank))

    avg_rank = (ops_rank + k_rank) / 2.0
    if avg_rank <= 10:
        return ("Tough", ops_rank, k_rank)
    if avg_rank <= 20:
        return ("Fair", ops_rank, k_rank)
    return ("Great", ops_rank, k_rank)


def _build_detail(
    ctx: MatchupContext,
    opponent: str,
    venue: str,
    effective_ops_rank: int,
    effective_k_rank: int,
) -> dict[str, Any]:
    """Tooltip payload for one start.

    Reports both the raw season stats (so the user sees how good the
    opponent is overall) and the park-adjusted effective ranks (which
    is what the color is actually keyed off of). The venue's park
    factor is exposed too so the user can sanity-check the nudge.
    """
    opp = ctx.team_stats.get(opponent, {})
    raw_k = opp.get("k_pct", 0.0)
    k_display = round(raw_k * 100, 1) if raw_k < 1 else round(raw_k, 1)
    return {
        "ops": round(opp.get("ops", 0.0), 3),
        "ops_rank": ctx.ops_rank_map.get(opponent, 0),
        "k_pct": k_display,
        "k_rank": ctx.k_rank_map.get(opponent, 0),
        "venue": venue,
        "park_ops_factor": round(get_park_factor(venue, "ops"), 2),
        "park_k_factor": round(get_park_factor(venue, "k"), 2),
        "effective_ops_rank": effective_ops_rank,
        "effective_k_rank": effective_k_rank,
    }


def compose_pitcher_entries(
    pitcher_name: str,
    pitcher_team: str,
    team_games: list[GameSlot],
    today: date,
    window_start: date,
    window_end: date,
    ctx: MatchupContext,
) -> list[StartEntry]:
    """Build the full list of StartEntry rows for one pitcher.

    Combines:
      - announced starts in ``[window_start, window_end]`` where this
        pitcher is the starter,
      - projected starts (anchor + 5*N) that land inside the window,
        excluding any team-game whose announced starter is someone else.

    Each entry is decorated by ``_matchup_quality``: the opponent's
    park-neutral OPS/K% (precomputed in ``ctx``) get the venue's park
    factor applied, and the resulting effective values are ranked
    against the league's park-neutral distribution.

    ``pitcher_team`` is needed so ``vs`` games (home games) resolve to
    the right venue -- the pitcher's own park, not the opponent's.

    Rows are returned sorted by date.
    """
    target = normalize_name(pitcher_name)
    win_start_iso = window_start.isoformat()
    win_end_iso = window_end.isoformat()

    in_window = lambda d: win_start_iso <= d <= win_end_iso  # noqa: E731

    used_indices: set[int] = set()
    entries: list[StartEntry] = []

    def make_entry(slot: GameSlot, announced: bool) -> StartEntry:
        venue = _venue_for(pitcher_team, slot.opponent, slot.indicator)
        quality, eff_ops_rank, eff_k_rank = _matchup_quality(slot.opponent, venue, ctx)
        return StartEntry(
            date=slot.date,
            day=_day_name(slot.date),
            opponent=slot.opponent,
            indicator=slot.indicator,
            announced=announced,
            matchup_quality=quality,
            detail=_build_detail(ctx, slot.opponent, venue, eff_ops_rank, eff_k_rank),
        )

    for i, slot in enumerate(team_games):
        if not in_window(slot.date):
            continue
        if not slot.announced_starter:
            continue
        if normalize_name(slot.announced_starter) != target:
            continue
        used_indices.add(i)
        entries.append(make_entry(slot, announced=True))

    anchor = find_anchor_index(team_games, pitcher_name, today)
    if anchor is not None:
        for idx in project_start_indices(anchor, len(team_games), step=5):
            if idx in used_indices:
                continue
            slot = team_games[idx]
            if not in_window(slot.date):
                continue
            if slot.announced_starter and normalize_name(slot.announced_starter) != target:
                continue
            entries.append(make_entry(slot, announced=False))

    entries.sort(key=lambda e: e.date)
    return entries


def filter_starting_pitchers(
    roster: list[Any],
    pitchers_proj: pd.DataFrame,
) -> list[Any]:
    """Keep only roster pitchers projected to start at least one game.

    Pitcher-eligible (P / SP / RP) AND projection ``gs > 0``. The
    eligibility gate uses ``is_pitcher`` rather than a strict
    ``Position.SP`` check because some Yahoo leagues use a single
    universal ``P`` slot -- every pitcher's eligibility list is just
    ``[P]`` regardless of starter/reliever role, so an SP-only check
    drops the entire pitcher staff. ``gs > 0`` does the real work of
    excluding closers/setup men.

    Players missing from the projection frame, or with a projection row
    that has no ``gs`` column / non-positive gs, are dropped.
    """
    from fantasy_baseball.utils.positions import is_pitcher

    if pitchers_proj is None or pitchers_proj.empty or "gs" not in pitchers_proj.columns:
        return []
    if "_name_norm" not in pitchers_proj.columns:
        # Defensive: refresh pipeline always attaches _name_norm, but tests may not.
        pitchers_proj = pitchers_proj.copy()
        pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)

    gs_by_name = dict(zip(pitchers_proj["_name_norm"], pitchers_proj["gs"], strict=False))

    kept: list[Any] = []
    for player in roster:
        if not is_pitcher(player.positions):
            continue
        gs = gs_by_name.get(normalize_name(player.name), 0.0) or 0.0
        if gs > 0:
            kept.append(player)
    return kept
