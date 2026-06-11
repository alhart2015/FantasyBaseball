from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fantasy_baseball.draft import eroto_recs
from fantasy_baseball.draft.eroto_recs import RecRow
from fantasy_baseball.draft.recommender import Recommendation, get_recommendations
from fantasy_baseball.draft.strategy import OVERLAYS, select_from_ranked
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

# Experimental adaptive mode: VOPN ("value of picking now") while building, then
# immediate marginal roto once the roster is loaded. Round-based trigger for now
# (a roster-strength-relative-to-field trigger is the planned v2). Flip round is
# overridable via the ADAPTIVE_K env var for sweeps.
ADAPTIVE_MODE = "deltaroto_adaptive"
ADAPTIVE_DEFAULT_K = 8

_VARVONA_MODES = ("var", "vona")


def _adaptive_metric(current_round: int) -> str:
    k = int(os.environ.get("ADAPTIVE_K", str(ADAPTIVE_DEFAULT_K)))
    return "immediate_delta" if current_round >= k else "value_of_picking_now"


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
    # 1-indexed draft round, used only by the experimental "deltaroto_adaptive"
    # mode to switch metric (VOPN while building -> immediate once loaded).
    current_round: int = 0


def _rank_deltaroto(ctx: RecommendContext) -> list[RankedPick]:
    if ctx.inputs is None:
        raise ValueError(f"scoring_mode {ctx.scoring_mode!r} requires inputs (RecInputs)")
    if ctx.scoring_mode == ADAPTIVE_MODE:
        metric = _adaptive_metric(ctx.current_round)
    else:
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
        # Coerce NaN/None to -inf so a real var always wins the tie-break (Fix 5).
        v = float("-inf") if var is None or (isinstance(var, float) and math.isnan(var)) else var
        if name not in id_by_name or v > best_var[name]:
            id_by_name[name] = pid
            best_var[name] = v
    recs = get_recommendations(
        ctx.board,
        drafted=ctx.drafted,
        user_roster=[],
        n=40,
        filled_positions=ctx.filled_positions,
        picks_until_next=ctx.picks_until_next,
        roster_slots=ctx.config.roster_slots,
        num_teams=ctx.config.num_teams,
        scoring_mode=ctx.scoring_mode,
    )
    # Build a pid -> sv lookup so closer-family overlays can inspect SV
    # projections in var/vona mode (per_category["SV"] is empty without this).
    sv_by_pid: dict[str, float] = {}
    if "sv" in ctx.board.columns:
        for pid_col, sv_val in zip(ctx.board["player_id"], ctx.board["sv"], strict=False):
            with contextlib.suppress(TypeError, ValueError):
                sv_float = float(sv_val)
                if not math.isnan(sv_float):
                    sv_by_pid[str(pid_col)] = sv_float

    out: list[RankedPick] = []
    for rec in recs:
        pid = id_by_name.get(rec.name)
        if pid is None:
            # rec.name came from the same board, so a miss is a real logic error.
            raise KeyError(f"recommendation {rec.name!r} has no player_id on the board")
        rp = from_recommendation(rec, player_id=str(pid))
        # Populate per_category["SV"] from board so closer-family overlays fire
        # correctly in var/vona mode (same signal as deltaroto per_category).
        sv = sv_by_pid.get(str(pid))
        if sv is not None:
            rp.per_category = {"SV": sv}
        if ctx.scoring_mode == "vona":
            vona = rec.score if rec.score is not None else rec.var
            rp.metrics = {"vona": vona}
            rp.score = vona
        out.append(rp)
    return out


def rank_for_mode(ctx: RecommendContext) -> list[RankedPick]:
    """Single dispatcher: rank the candidate pool for ``ctx.scoring_mode``."""
    if ctx.scoring_mode in _DELTAROTO_MODES or ctx.scoring_mode == ADAPTIVE_MODE:
        return _rank_deltaroto(ctx)
    if ctx.scoring_mode in _VARVONA_MODES:
        return _rank_var_vona(ctx)
    raise ValueError(f"unknown scoring_mode {ctx.scoring_mode!r}")


def to_recs_json(pick: RankedPick) -> dict:
    """Serialize a deltaRoto-mode RankedPick into the exact /api/recs shape
    the dashboard JS expects (immediate_delta + value_of_picking_now top-level)."""
    if "immediate_delta" not in pick.metrics:
        raise ValueError(
            f"to_recs_json requires a deltaRoto-mode pick; got metrics keys {list(pick.metrics)}"
        )
    return {
        "player_id": pick.player_id,
        "name": pick.name,
        "positions": pick.position_strings(),
        "immediate_delta": pick.metrics["immediate_delta"],
        "value_of_picking_now": pick.metrics["value_of_picking_now"],
        "per_category": pick.per_category,
    }


def recommend(
    ctx: RecommendContext,
    *,
    strategy: str,
    open_starters: set,
    roster_state=None,
    pick_rank: int = 0,
    ranked: list[RankedPick] | None = None,
    **overlay_kwargs,
) -> RankedPick | None:
    """Rank for ctx.scoring_mode, apply the strategy overlay, slot-gate.

    Serves all four modes because rank_for_mode(ctx) does the dispatch; the
    overlay and slot-gate are mode-agnostic (they consume RankedPick).

    Overlay-specific context (e.g. current_round, closer_count for the closer
    overlays) is passed via **overlay_kwargs by the caller.  The sim and
    dashboard supply these from draft state; recommend() forwards them
    unchanged to the overlay so closer-family overlays fire correctly.

    ``ranked`` may be supplied by the caller to skip re-ranking (avoids a
    redundant rank_for_mode call when the caller already ranked for the same
    ctx, e.g. the noise-alternate path in simulate_draft).
    """
    if ranked is None:
        ranked = rank_for_mode(ctx)
    if strategy not in OVERLAYS:
        raise ValueError(f"unknown strategy {strategy!r}; valid: {sorted(OVERLAYS)}")
    chosen = OVERLAYS[strategy](
        ranked, roster_state=roster_state, config=ctx.config, **overlay_kwargs
    )
    if chosen is not None:
        return chosen
    # Overlay deferred -> plain slot-gated selection.
    return select_from_ranked(ranked, open_starters, pick_rank)
