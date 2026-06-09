from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fantasy_baseball.draft import eroto_recs
from fantasy_baseball.draft.eroto_recs import RecRow
from fantasy_baseball.draft.recommender import Recommendation, get_recommendations
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.positions import Position

if TYPE_CHECKING:
    import pandas as pd

    from fantasy_baseball.draft.recs_integration import RecInputs


@dataclass
class RankedPick:
    """One ranked draft candidate, uniform across every scoring mode.

    ``score`` is the active mode's primary metric. ``metrics`` carries every
    mode-native metric (deltaRoto modes populate both ``immediate_delta`` and
    ``value_of_picking_now`` so the dashboard can toggle between them).
    """

    player_id: str
    name: str
    positions: list[Position]
    player_type: PlayerType
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    per_category: dict[str, float] = field(default_factory=dict)
    note: str = ""
    need_flag: bool = False

    def position_strings(self) -> list[str]:
        """Position codes as plain strings (for JSON / display)."""
        return [p.value if isinstance(p, Position) else str(p) for p in self.positions]


def from_recommendation(rec: Recommendation, *, player_id: str) -> RankedPick:
    """Adapt a VAR/VONA ``Recommendation`` into a ``RankedPick``.

    ``Recommendation`` carries no player_id, so callers pass it in (the
    board lookup already has it).
    """
    return RankedPick(
        player_id=player_id,
        name=rec.name,
        positions=list(rec.positions),
        player_type=rec.player_type,
        score=rec.var,
        metrics={"var": rec.var},
        note=rec.note,
        need_flag=rec.need_flag,
    )


_DELTAROTO_METRICS = ("immediate_delta", "value_of_picking_now")

_DELTAROTO_MODES = {
    "deltaroto_immediate": "immediate_delta",
    "deltaroto_vopn": "value_of_picking_now",
}

_VARVONA_MODES = ("var", "vona")


def from_recrow(row: RecRow, *, metric: str, player_type: PlayerType) -> RankedPick:
    """Adapt a deltaRoto ``RecRow`` into a ``RankedPick``.

    ``metric`` selects which native metric becomes ``score``; both are kept
    in ``metrics`` so the dashboard can display/toggle both.
    """
    if metric not in _DELTAROTO_METRICS:
        raise ValueError(f"metric must be one of {_DELTAROTO_METRICS}, got {metric!r}")
    metrics = {
        "immediate_delta": row.immediate_delta,
        "value_of_picking_now": row.value_of_picking_now,
    }
    return RankedPick(
        player_id=row.player_id,
        name=row.name,
        positions=[Position.parse(p) for p in row.positions],
        player_type=player_type,
        score=metrics[metric],
        metrics=metrics,
        per_category=dict(row.per_category),
    )


@dataclass
class RecommendContext:
    """Everything either ranker needs for one pick.

    deltaRoto modes use ``inputs`` (a ``RecInputs``); var/vona modes use the
    pandas ``board`` + ``drafted`` + ``filled_positions`` + ``config``. The
    caller fills whichever the active mode requires; ``rank_for_mode`` validates.
    """

    scoring_mode: str
    team_name: str
    picks_until_next: int
    inputs: RecInputs | None = None
    board: pd.DataFrame | None = None
    drafted: list[str] = field(default_factory=list)
    filled_positions: dict[str, int] | None = None
    config: Any = None


def _rank_deltaroto(ctx: RecommendContext) -> list[RankedPick]:
    if ctx.inputs is None:
        raise ValueError(f"scoring_mode {ctx.scoring_mode!r} requires inputs (RecInputs)")
    metric = _DELTAROTO_MODES[ctx.scoring_mode]
    rows = eroto_recs.rank_candidates(
        candidates=ctx.inputs.candidates,
        replacements=ctx.inputs.replacements,
        team_name=ctx.team_name,
        projected_standings=ctx.inputs.projected_standings,
        team_sds=ctx.inputs.team_sds,
        picks_until_next_turn=ctx.picks_until_next,
        adp_table=ctx.inputs.adp_table,
        user_rp_filled=ctx.inputs.rp_filled_by_team.get(ctx.team_name, 0),
    )
    type_by_id = {eroto_recs._candidate_id(c): c.player_type for c in ctx.inputs.candidates}
    picks: list[RankedPick] = []
    for r in rows:
        pt = type_by_id.get(r.player_id)
        if pt is None:
            # Fail loud rather than mislabel a pitcher as a hitter in overlays.
            raise KeyError(f"candidate id {r.player_id!r} ({r.name}) absent from board candidates")
        picks.append(from_recrow(r, metric=metric, player_type=pt))
    # Sort by the active metric's score unconditionally. rank_candidates returns
    # immediate_delta order, but relying on that for immediate mode is a hidden
    # coupling (the kind of sort assumption that caused pre-PR #127 bugs); score
    # already mirrors the active metric, so sorting here is correct for all modes.
    picks.sort(key=lambda p: p.score, reverse=True)
    return picks


def _rank_var_vona(ctx: RecommendContext) -> list[RankedPick]:
    if ctx.board is None:
        raise ValueError(f"scoring_mode {ctx.scoring_mode!r} requires board (DataFrame)")
    if ctx.config is None:
        raise ValueError(f"scoring_mode {ctx.scoring_mode!r} requires config (LeagueConfig)")
    # Same-name players exist on the board; the recommender returns names only,
    # so resolve to player_id by keeping the higher-VAR entry per name (the
    # repo convention for name collisions). See CLAUDE.md cross-cutting rules.
    id_by_name: dict[str, str] = {}
    best_var: dict[str, float] = {}
    for name, pid, var in zip(
        ctx.board["name"], ctx.board["player_id"], ctx.board["var"], strict=False
    ):
        if name not in id_by_name or var > best_var[name]:
            id_by_name[name] = pid
            best_var[name] = var
    recs = get_recommendations(
        ctx.board,
        drafted=ctx.drafted,
        user_roster=[],
        n=15,
        filled_positions=ctx.filled_positions,
        picks_until_next=ctx.picks_until_next,
        roster_slots=ctx.config.roster_slots,
        num_teams=ctx.config.num_teams,
        scoring_mode=ctx.scoring_mode,
    )
    out: list[RankedPick] = []
    for rec in recs:
        pid = id_by_name.get(rec.name)
        if pid is None:
            # rec.name came from the same board, so a miss is a real logic error.
            raise KeyError(f"recommendation {rec.name!r} has no player_id on the board")
        rp = from_recommendation(rec, player_id=str(pid))
        if ctx.scoring_mode == "vona":
            vona = rec.score if rec.score is not None else rec.var
            rp.metrics = {"vona": vona}
            rp.score = vona
        out.append(rp)
    return out


def rank_for_mode(ctx: RecommendContext) -> list[RankedPick]:
    """Single dispatcher: rank the candidate pool for ``ctx.scoring_mode``."""
    if ctx.scoring_mode in _DELTAROTO_MODES:
        return _rank_deltaroto(ctx)
    if ctx.scoring_mode in _VARVONA_MODES:
        return _rank_var_vona(ctx)
    raise ValueError(f"unknown scoring_mode {ctx.scoring_mode!r}")
