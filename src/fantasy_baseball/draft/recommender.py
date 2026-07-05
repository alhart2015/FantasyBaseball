from dataclasses import dataclass
from typing import cast

import pandas as pd

# compute_slot_scarcity_order, get_filled_positions, and get_roster_by_position
# are re-exported below for backward compatibility with simulator code and tests
# that imported these helpers from draft.recommender. New live-draft code should
# import directly from draft.roster_state.
from fantasy_baseball.draft.roster_state import (
    RosterState,
    compute_slot_scarcity_order,  # noqa: F401
    get_filled_positions,  # noqa: F401
    get_roster_by_position,  # noqa: F401
)
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.sgp.denominators import SgpOverrides, get_sgp_denominators
from fantasy_baseball.sgp.replacement import (
    calculate_replacement_rates,
    position_aware_replacement_levels,
)
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import (
    CLOSER_SV_THRESHOLD,
    DEFAULT_ROSTER_SLOTS,
    compute_starters_per_position,
)
from fantasy_baseball.utils.positions import can_fill_slot


@dataclass
class Recommendation:
    """A single draft pick recommendation.

    ``score`` is the scoring-mode-specific ranking value (VAR or VONA) used
    to sort recommendations; it is ``None`` for synthetic entries like the
    strategy-layer closer alert in ``run_draft.py``.

    ``best_position`` and ``positions`` are ``Position`` enum values; the
    ``__post_init__`` coerces raw strings (as stored on the board DataFrame)
    so callers can pass either.
    """

    name: str
    var: float
    score: float | None
    best_position: Position
    positions: list[Position]
    player_type: PlayerType
    need_flag: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.best_position, Position):
            self.best_position = Position.parse(self.best_position)
        self.positions = [
            p if isinstance(p, Position) else Position.parse(p) for p in self.positions
        ]


def calculate_vona_scores(
    available: pd.DataFrame,
    picks_until_next: int | None = None,
) -> dict[str, float]:
    """Compute Value Over Next Available for each player.

    For each player, estimates what the best player of the same bucket
    (hitter / SP / closer) will still be available after opponents make
    ``picks_until_next`` picks, drafting by ADP.

    Returns dict of player_id -> VONA score.

    Note: position-level VONA (per-position hitter buckets) was tested
    and regressed badly — it over-values positional scarcity, causing the
    recommender to reach for scarce positions at the expense of overall
    value.  The 3-bucket approach keeps hitter VONA balanced.
    """
    if picks_until_next is None or picks_until_next < 1:
        picks_until_next = 10  # sensible default

    # Sort the full pool by ADP — opponents draft roughly by ADP
    adp_sorted = available.sort_values("adp", ascending=True)

    # The next N picks (by ADP) are the ones opponents will take
    gone_ids = set(adp_sorted.head(picks_until_next)["player_id"])

    # What remains after opponents pick
    remaining = available[~available["player_id"].isin(gone_ids)]

    # Assign buckets vectorized
    is_hitter = remaining["player_type"] == PlayerType.HITTER
    sv = (
        remaining["sv"].fillna(0)
        if "sv" in remaining.columns
        else pd.Series(0, index=remaining.index)
    )
    remaining_buckets = pd.Series("sp", index=remaining.index)
    remaining_buckets[is_hitter] = "hitter"
    remaining_buckets[(~is_hitter) & (sv >= CLOSER_SV_THRESHOLD)] = "closer"

    sgp = remaining["total_sgp"].fillna(0)
    best_remaining = sgp.groupby(remaining_buckets).max().to_dict()
    for b in ("hitter", "sp", "closer"):
        best_remaining.setdefault(b, 0)

    # VONA = player SGP - best remaining in same bucket (vectorized)
    is_hitter_a = available["player_type"] == PlayerType.HITTER
    sv_a = (
        available["sv"].fillna(0)
        if "sv" in available.columns
        else pd.Series(0, index=available.index)
    )
    avail_buckets = pd.Series("sp", index=available.index)
    avail_buckets[is_hitter_a] = "hitter"
    avail_buckets[(~is_hitter_a) & (sv_a >= CLOSER_SV_THRESHOLD)] = "closer"

    avail_sgp = available["total_sgp"].fillna(0)
    best_for_bucket = avail_buckets.map(best_remaining)
    vona_series = avail_sgp - best_for_bucket

    return dict(zip(available["player_id"], vona_series, strict=False))


def _replacement_levels_for_board(
    board: pd.DataFrame,
    roster_slots: dict[str, int],
    num_teams: int | None,
    sgp_overrides: SgpOverrides | None = None,
) -> dict[str, float]:
    """Position-aware empirical replacement floors for ``board``.

    The floors are a pure function of the FULL board's rate baselines + the
    slot config, so they are constant for the whole draft -- but
    ``get_recommendations`` runs once per pick (and per strategy-opponent pick
    in the simulator, ~250 picks x 100+ sims). Cache the result on
    ``board.attrs`` keyed by the slot signature: the cache is per-board-object
    (a freshly built board gets fresh floors) and recomputes if the slot config
    changes, turning hundreds of redundant full-board pandas scans into one.
    """
    # Key the cache on the RESOLVED denominators, not the raw override
    # pairs: get_sgp_denominators normalizes enum/string keys (a plain-Enum
    # sort would TypeError) and resolves duplicate spellings last-write-wins,
    # so two dicts that resolve differently can never share a cache entry.
    denoms = get_sgp_denominators(sgp_overrides)
    overrides_key = (
        tuple(sorted((c.value, v) for c, v in denoms.items())) if sgp_overrides else None
    )
    cache_key = (num_teams, tuple(sorted(roster_slots.items())), overrides_key)
    cached = board.attrs.get("_repl_levels_cache")
    if cached is not None and cached[0] == cache_key:
        return cast(dict[str, float], cached[1])
    starters = compute_starters_per_position(roster_slots, num_teams)
    repl_rates = calculate_replacement_rates(board, starters)
    repl_levels = position_aware_replacement_levels(denoms, repl_rates)
    board.attrs["_repl_levels_cache"] = (cache_key, repl_levels)
    return repl_levels


def get_recommendations(
    board: pd.DataFrame,
    drafted: list[str],
    user_roster: list[str],  # noqa: ARG001  (kept for API; used by callers positionally)
    n: int = 5,
    filled_positions: dict[str, int] | None = None,
    picks_until_next: int | None = None,
    roster_slots: dict[str, int] | None = None,
    num_teams: int | None = None,
    scoring_mode: str = "var",
    sgp_overrides: SgpOverrides | None = None,
) -> list[Recommendation]:
    """Get top draft pick recommendations.

    Replacement floors are position-aware empirical waiver lines
    (:func:`position_aware_replacement_levels`) -- static per-position and
    SP/RP constants, not the old per-pick demand floors. Only the rate
    baselines (AVG/ERA/WHIP) are recomputed from the remaining pool, so
    in-draft positional scarcity (e.g. a run on catchers) is no longer
    reflected in the VAR floor itself; it is surfaced via the scarcity note
    below and, in "vona" mode, by VONA.

    *scoring_mode*: "var" (default) uses Value Above Replacement for
    ranking; "vona" uses Value Over Next Available, which accounts for
    talent depth at each player type (hitter/SP/closer).

    *sgp_overrides*: league-specific SGP denominator overrides (from
    ``config.sgp_overrides``) so live-VAR floors are computed on the same
    denominators the board was built with. None keeps the code defaults.
    """
    if roster_slots is None:
        roster_slots = DEFAULT_ROSTER_SLOTS
    available = board[~board["player_id"].isin(drafted)]

    # Empirical waiver floors (a pure function of denoms + rate baselines). The
    # AVG/ERA/WHIP baselines come from the FULL board, not the shrinking
    # ``available`` pool, so they match the basis the board's frozen
    # ``total_sgp`` was scored on and don't drift as the draft drains. Cached
    # on the board so the per-pick call doesn't re-scan the full board.
    repl_levels = _replacement_levels_for_board(board, roster_slots, num_teams, sgp_overrides)

    # Only recompute VAR for top candidates (by pre-computed VAR).
    # The full pool sets replacement levels accurately, but iterating
    # all ~3000 players per pick is the main performance bottleneck.
    # 300 covers all draftable players (10 teams x 23 slots = 230) plus
    # padding for positional scarcity shifts at C/SS/1B.
    _VAR_CANDIDATE_LIMIT = 300
    candidates = available.nlargest(_VAR_CANDIDATE_LIMIT, "var")
    live_var = {}
    live_pos = {}
    for idx, row in candidates.iterrows():
        var, pos = calculate_var(row, repl_levels, return_position=True)
        live_var[idx] = var
        live_pos[idx] = pos
    available = candidates.copy()
    available["var"] = available.index.map(live_var)
    available["best_position"] = available.index.map(live_pos)

    # Compute VONA if requested.  Use the full undrafted pool (not the
    # top-150 VAR candidates) so that "best remaining in bucket" reflects
    # true talent depth.  VONA only needs total_sgp and adp, both static
    # columns, so iterating the full pool is cheap (~O(n)).
    vona_scores = None
    if scoring_mode == "vona":
        full_available = board[~board["player_id"].isin(drafted)]
        vona_scores = calculate_vona_scores(full_available, picks_until_next)
        available["vona"] = available["player_id"].map(vona_scores).fillna(0)

    # Sort by the active scoring mode
    sort_col = "vona" if scoring_mode == "vona" else "var"
    available = available.sort_values(sort_col, ascending=False)

    if filled_positions is None:
        filled_positions = {}
    roster_state = RosterState.from_dicts(filled_positions, roster_slots)

    # Filter out players who have no open roster slot (including BN).
    # E.g. if all OF, UTIL, and BN spots are full, don't suggest more OFs.
    available = _filter_rosterable(available, roster_state)
    available = available.sort_values(sort_col, ascending=False)

    # Use a wider window for scarcity checks, narrower for rec candidates
    scarcity_pool = available.head(50)
    candidates = available.head(n * 3)
    unfilled = roster_state.unfilled_starter_slots()

    # Ensure the best available player at each unfilled position is included
    # so the user always sees their positional options, not just raw VAR.
    candidate_ids = set(candidates["player_id"])
    for slot in unfilled:
        for _, row in available.iterrows():
            if row["player_id"] in candidate_ids:
                continue
            if can_fill_slot(row["positions"], slot):
                candidates = pd.concat([candidates, row.to_frame().T], ignore_index=True)
                candidate_ids.add(row["player_id"])
                break  # only need the best one per slot

    recs: list[Recommendation] = []
    for _, player in candidates.iterrows():
        raw_score = player.get("vona", 0) if scoring_mode == "vona" else player["var"]
        positions = player["positions"]
        if not isinstance(positions, list):
            positions = [positions]
        rec = Recommendation(
            name=player["name"],
            var=float(player["var"]),
            score=round(float(raw_score), 2),
            best_position=player["best_position"],
            positions=list(positions),
            player_type=player["player_type"],
        )
        # Check specific positional slots before flex (IF/UTIL) so the note
        # shows "fills 3B need" rather than "fills IF need" when both are open.
        specific_unfilled = [s for s in unfilled if s not in ("IF", "UTIL")]
        flex_unfilled = [s for s in unfilled if s in ("IF", "UTIL")]
        for slot in specific_unfilled + flex_unfilled:
            if can_fill_slot(rec.positions, slot):
                rec.need_flag = True
                rec.note = f"fills {slot} need"
                break
        if picks_until_next and picks_until_next > 8:
            pos = rec.best_position
            remaining_at_pos = len(scarcity_pool[scarcity_pool["best_position"] == pos])
            if remaining_at_pos <= 3:
                scarcity = f"scarce position — only {remaining_at_pos} left in top tier"
                rec.note = f"{rec.note}; {scarcity}" if rec.note else scarcity
        recs.append(rec)
    # Guarantee at least one player per unfilled position makes the final list.
    # Split into need-fills and pure-score, then merge.
    need_recs: list[Recommendation] = []
    other_recs: list[Recommendation] = []
    seen_need_slots: set[Position] = set()
    # Sort all by score (VAR or VONA)
    recs.sort(key=lambda r: r.score if r.score is not None else float("-inf"), reverse=True)
    for rec in recs:
        if rec.need_flag and rec.best_position not in seen_need_slots:
            need_recs.append(rec)
            seen_need_slots.add(rec.best_position)
        else:
            other_recs.append(rec)
    # Fill remaining slots with best-score players
    result = need_recs + other_recs
    return result[:n]


def _filter_rosterable(
    available: pd.DataFrame,
    roster_state: RosterState,
) -> pd.DataFrame:
    """Remove players who cannot fit in any open roster slot (including BN)."""
    open_slots = roster_state.open_slots()
    if not open_slots:
        return available.iloc[0:0]  # no room at all

    def has_open_slot(positions):
        return any(can_fill_slot(positions, slot) for slot in open_slots)

    mask = available["positions"].apply(has_open_slot)
    return available[mask]
