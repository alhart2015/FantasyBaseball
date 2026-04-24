"""Glue between the live draft state and the ERoto recommender.

Converts the on-disk draft board + state into the inputs
``eroto_recs.rank_candidates`` needs: per-team rosters, per-position
replacement-level Players, ProjectedStandings, and team_sds. Kept
outside ``draft_controller`` so the controller stays pure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from fantasy_baseball.draft.adp import ADPTable, blend_adp
from fantasy_baseball.draft.state import StateKey, read_board
from fantasy_baseball.models.player import Player
from fantasy_baseball.models.standings import ProjectedStandings, ProjectedStandingsEntry
from fantasy_baseball.scoring import build_team_sds, project_team_stats, score_roto
from fantasy_baseball.utils.constants import ALL_CATEGORIES, INVERSE_STATS, Category

_NON_ACTIVE_SLOTS: frozenset[str] = frozenset({"BN", "IL", "IL+", "DL", "DL+"})


def load_board_rows(board_path: Path) -> list[dict[str, Any]]:
    """Load the cached draft board as a list of row dicts."""
    return read_board(board_path)


def rows_to_players(rows: list[dict[str, Any]]) -> list[Player]:
    """Convert board rows (flat stat format) to Player objects."""
    return [Player.from_dict(r) for r in rows]


def drafted_ids(state: dict[str, Any]) -> set[str]:
    """All player_ids already claimed by keepers or picks."""
    keepers = state.get(StateKey.KEEPERS) or []
    picks = state.get(StateKey.PICKS) or []
    return {p["player_id"] for p in (keepers + picks)}


def partition_available(players: list[Player], drafted: set[str]) -> list[Player]:
    """Filter the board down to players not yet drafted."""
    return [p for p in players if (p.yahoo_id or "") not in drafted]


def build_replacements_by_position(
    rows: list[dict[str, Any]],
    roster_slots: Mapping[str, int],
    num_teams: int,
) -> dict[str, Player]:
    """Per-position replacement-level Player.

    For each position with ``capacity`` starters per team, pick the
    (capacity * num_teams + 1)-th best player (by ``var``) who is
    eligible there — roughly the first player beyond the league's
    collective demand.
    """
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    if df.empty or "var" not in df.columns or "positions" not in df.columns:
        return {}
    df = df.sort_values("var", ascending=False).reset_index(drop=True)
    out: dict[str, Player] = {}
    for pos, capacity in roster_slots.items():
        if pos in _NON_ACTIVE_SLOTS or not capacity:
            continue
        demand = int(capacity) * max(int(num_teams), 1)
        eligible = df[df["positions"].apply(lambda ps, p=pos: p in (ps or []))]
        if eligible.empty:
            continue
        if len(eligible) <= demand:
            # Not enough players at this position to define a true
            # replacement — fall back to the worst eligible one.
            choice = eligible.iloc[-1]
        else:
            choice = eligible.iloc[demand]
        out[pos] = Player.from_dict(choice.to_dict())
    return out


def _generic_replacement(replacements: Mapping[str, Player]) -> Player | None:
    """Pick a single 'generic' replacement to pad unfilled bench slots.

    Uses the highest-VAR ROS-stat replacement among all positions —
    falls back to the first available value if no ROS stats are present.
    """
    if not replacements:
        return None

    def _score(p: Player) -> float:
        ros = p.rest_of_season
        if ros is None:
            return 0.0
        sgp = getattr(ros, "sgp", None)
        return float(sgp) if sgp is not None else 0.0

    return max(replacements.values(), key=_score)


def build_team_rosters(
    state: dict[str, Any],
    board_by_id: Mapping[str, Player],
    teams: list[str],
    roster_slots: Mapping[str, int],
    replacements: Mapping[str, Player],
) -> dict[str, list[Player]]:
    """Collect each team's keepers + picks as Player objects.

    Pads each roster with a generic replacement-level Player up to the
    league's active roster size so every team has comparable depth when
    aggregating stats. Keepers/picks not on the board (rare) fall back
    to the replacement at the keeper's declared position.
    """
    team_picks: dict[str, list[Player]] = {t: [] for t in teams}
    for entry in (state.get(StateKey.KEEPERS) or []) + (state.get(StateKey.PICKS) or []):
        team = entry["team"]
        pid = entry["player_id"]
        player = board_by_id.get(pid)
        if player is not None:
            team_picks.setdefault(team, []).append(player)
        elif entry.get("position") in replacements:
            team_picks.setdefault(team, []).append(replacements[entry["position"]])

    total_slots = sum(int(v) for k, v in roster_slots.items() if k not in _NON_ACTIVE_SLOTS and v)
    generic_rep = _generic_replacement(replacements)
    for roster in team_picks.values():
        while generic_rep is not None and len(roster) < total_slots:
            roster.append(generic_rep)
    return team_picks


def build_projected_standings(
    team_rosters: Mapping[str, list[Player]],
    effective_date: str = "2026-01-01",
) -> ProjectedStandings:
    """Project each team's roster into a ProjectedStandings."""
    entries = [
        ProjectedStandingsEntry(team_name=team, stats=project_team_stats(roster))
        for team, roster in team_rosters.items()
    ]
    return ProjectedStandings(
        effective_date=date.fromisoformat(effective_date),
        entries=entries,
    )


def build_adp_table(rows: list[dict[str, Any]]) -> ADPTable:
    """Construct an ADPTable from the board's ``adp`` column (if present)."""
    if not rows:
        return ADPTable()
    df = pd.DataFrame(rows)
    if df.empty or "adp" not in df.columns or "player_id" not in df.columns:
        return ADPTable()
    blended = blend_adp({"board": df[["player_id", "adp"]]})
    return ADPTable(adp=blended)


def _league_teams(league_yaml: Mapping[str, Any]) -> list[str]:
    """Return team names from league.yaml in pick order."""
    teams_raw = (league_yaml.get("draft") or {}).get("teams") or {}
    if isinstance(teams_raw, Mapping):
        return [teams_raw[k] for k in sorted(teams_raw.keys())]
    return list(teams_raw)


@dataclass(frozen=True)
class RecInputs:
    """Bundle of inputs ``eroto_recs.rank_candidates`` consumes.

    Returned by :func:`compute_rec_inputs` so callers don't have to
    unpack a 5-tuple by position. Pass ``inputs.candidates``,
    ``inputs.replacements``, etc., to ``rank_candidates`` keyword args.
    """

    candidates: list[Player]
    replacements: dict[str, Player]
    projected_standings: ProjectedStandings
    team_sds: dict[str, dict[Category, float]]
    adp_table: ADPTable


def compute_rec_inputs(
    state: dict[str, Any],
    board_path: Path,
    league_yaml: Mapping[str, Any],
) -> RecInputs:
    """Assemble the inputs ``rank_candidates`` needs."""
    rows = load_board_rows(board_path)
    players = rows_to_players(rows)
    drafted = drafted_ids(state)
    candidates = partition_available(players, drafted)

    roster_slots = league_yaml.get("roster_slots") or {}
    teams = _league_teams(league_yaml)

    replacements = build_replacements_by_position(rows, roster_slots, len(teams))

    board_by_id: dict[str, Player] = {}
    for p in players:
        if p.yahoo_id:
            board_by_id[p.yahoo_id] = p

    team_rosters = build_team_rosters(state, board_by_id, teams, roster_slots, replacements)
    projected_standings = build_projected_standings(team_rosters)
    team_sds = build_team_sds(team_rosters, sd_scale=1.0)
    adp_table = build_adp_table(rows)

    return RecInputs(
        candidates=candidates,
        replacements=replacements,
        projected_standings=projected_standings,
        team_sds=team_sds,
        adp_table=adp_table,
    )


def monte_carlo_roto_totals(
    projected_standings: ProjectedStandings,
    team_sds: Mapping[str, Mapping[Category, float]],
    *,
    n_iters: int = 500,
    seed: int | None = None,
) -> dict[str, tuple[float, float]]:
    """Monte-Carlo simulate final roto totals under projection uncertainty.

    For each iteration: perturb each team's raw category total by a
    Gaussian with the team's projection SD for that category, rank teams
    per category (averaging ties, reversing for ERA/WHIP), and sum the
    resulting per-category ranks (1..n_teams) as that team's total.

    Returns ``{team: (mean_total, sd_total)}`` across ``n_iters`` draws.
    The mean approximates ``score_roto``'s EV (averaged ranks converge to
    the Gaussian pairwise smooth approximation); the SD is the honest
    uncertainty on the total roto points — the quantity a naive
    quadrature-sum of per-category projection SDs does not produce.
    """
    rng = np.random.default_rng(seed)
    teams = [e.team_name for e in projected_standings.entries]
    if not teams:
        return {}
    stats_by_team = {e.team_name: e.stats for e in projected_standings.entries}

    totals = np.zeros((n_iters, len(teams)))
    for cat in ALL_CATEGORIES:
        means = np.array([stats_by_team[t][cat] for t in teams], dtype=float)
        sds = np.array(
            [float(team_sds.get(t, {}).get(cat, 0.0)) for t in teams],
            dtype=float,
        )
        samples = rng.normal(loc=means, scale=sds, size=(n_iters, len(teams)))
        if cat in INVERSE_STATS:
            samples = -samples  # lower is better → flip sign for ranking
        # rankdata returns 1..n with averaged ties. Applied per iteration.
        ranks = np.apply_along_axis(rankdata, 1, samples)
        totals += ranks

    means_out = totals.mean(axis=0)
    sds_out = totals.std(axis=0, ddof=1) if n_iters > 1 else np.zeros(len(teams))
    return {t: (float(m), float(s)) for t, m, s in zip(teams, means_out, sds_out, strict=True)}


@dataclass(frozen=True)
class TeamStandingsRow:
    """One team's entry in the ``projected_standings_cache``.

    ``total`` / ``total_sd`` describe the team's projected total roto
    points (from a Monte-Carlo over projection uncertainty). ``categories``
    maps each roto category to its fractional EV points (from
    ``score_roto``) for per-category drill-down.
    """

    total: float
    total_sd: float
    categories: dict[Category, float]

    def to_json(self) -> dict[str, Any]:
        """Serialize to the shape stored in ``draft_state.json``."""
        return {
            "total": self.total,
            "total_sd": self.total_sd,
            "categories": {cat.value: pts for cat, pts in self.categories.items()},
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> TeamStandingsRow:
        """Parse from the JSON shape written by :meth:`to_json`."""
        return cls(
            total=float(data["total"]),
            total_sd=float(data["total_sd"]),
            categories={Category(k): float(v) for k, v in (data.get("categories") or {}).items()},
        )


def compute_standings_cache(
    projected_standings: ProjectedStandings,
    team_sds: Mapping[str, Mapping[Category, float]],
    *,
    mc_iters: int = 500,
    mc_seed: int | None = None,
) -> dict[str, TeamStandingsRow]:
    """Build the standings cache (one :class:`TeamStandingsRow` per team).

    Per-team totals and SDs come from a Monte-Carlo over projection
    uncertainty (see :func:`monte_carlo_roto_totals`). Per-category EVs
    come from ``score_roto`` so the drill-down matches the numbers
    :func:`rank_candidates` sees.
    """
    # ProjectedStandings structurally satisfies TeamStatsTable, but mypy
    # can't see the protocol variance through list[ProjectedStandingsEntry]
    # vs Sequence[TeamStatsRow]. Same cast pattern web/season_data.py uses.
    points = score_roto(cast("Any", projected_standings), team_sds=team_sds)
    mc = monte_carlo_roto_totals(
        projected_standings,
        team_sds,
        n_iters=mc_iters,
        seed=mc_seed,
    )
    cache: dict[str, TeamStandingsRow] = {}
    for team, cat_points in points.items():
        mean_total, sd_total = mc.get(team, (float(cat_points.total), 0.0))
        cache[team] = TeamStandingsRow(
            total=mean_total,
            total_sd=sd_total,
            categories={cat: float(pts) for cat, pts in cat_points.values.items()},
        )
    return cache


def serialize_standings_cache(
    cache: Mapping[str, TeamStandingsRow],
) -> dict[str, dict[str, Any]]:
    """Serialize a standings cache for persistence in ``draft_state.json``."""
    return {team: row.to_json() for team, row in cache.items()}


def deserialize_standings_cache(
    data: Mapping[str, Mapping[str, Any]],
) -> dict[str, TeamStandingsRow]:
    """Parse a persisted standings cache back into typed rows.

    Entries that fail to parse (legacy schema, missing keys, etc.) are
    silently dropped — the next pick refreshes the cache.
    """
    out: dict[str, TeamStandingsRow] = {}
    for team, raw in data.items():
        try:
            out[team] = TeamStandingsRow.from_json(raw)
        except (KeyError, ValueError, TypeError):
            continue
    return out
