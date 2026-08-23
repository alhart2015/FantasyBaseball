"""Synthesize the free-agent pool Yahoo used to supply.

The season pipeline asks Yahoo "who is available?" once, in
``web.refresh_pipeline.RefreshRun._audit_roster``, and that single question is
the only reason the roster audit needs the API at all. This module answers it
from data we already hold:

    FA pool = every row in the ROS projection frames
              MINUS everyone on the ten transcribed rosters
              MINUS implausible candidates
              MINUS the manual exclusions list

and then hands the survivors to the SAME matcher the Yahoo path uses
(:func:`data.projections.match_roster_to_projections`), so suffix stripping,
accent normalization, same-name tie-break by playing time, and identity-keyed
preseason attach all behave identically. Downstream -- ``build_positions_map``,
``audit_roster``, ``score_stash_candidates`` -- cannot tell the two pools apart
by shape.

Three properties of this derivation are worth stating plainly, because each one
is a place where a plausible-looking shortcut produces a wrong answer:

1. **Subtraction is per player type.** The rostered set is keyed on
   ``(normalized name, player_type)``, never a bare name. Shohei Ohtani is two
   rows in the projection frames; in this league he is kept as a batter only, so
   removing the hitter must leave the pitcher in the pool. The same trap catches
   the catcher and the pitcher both named Will Smith.

2. **The pool is filtered, not merely subtracted.** Yahoo's endpoint returns a
   pre-ranked, ownership-filtered slice. "All projections minus rostered" is
   ~9,000 rows including every AAA depth arm, and ``audit_roster`` takes the
   MAXIMUM DeltaRoto free agent per slot -- so one fluky 12-IP rate line becomes
   a headline "upgrade". Pool quality is directly load-bearing on the output,
   which is why the volume floor and the per-position rank cap exist.

3. **Eligibility is never invented.** A hitter with no known position data is
   DROPPED, not defaulted to UTIL. A synthetic UTIL bat would be eligible for a
   real lineup slot on the strength of a projection alone and could displace a
   rostered player -- a fabricated upgrade. Pitchers with no entry default to
   ``["P"]``, which is what Yahoo returns for every pitcher in this league
   anyway (``roster_audit`` splits SP/RP on projected saves, not on positions).

INVARIANT (see the package docstring): this module acquires and shapes data. It
computes no SGP, no DeltaRoto, and no replacement level.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from fantasy_baseball.lineup.waivers import FreeAgentRequest
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import HITTER_ELIGIBLE, PITCHER_ELIGIBLE, Position
from fantasy_baseball.sgp.rankings import lookup_rank
from fantasy_baseball.utils.name_utils import normalize_name

logger = logging.getLogger(__name__)

#: Buckets the per-position rank cap is applied to, for hitters. These are
#: exactly the six hitter positions
#: ``lineup.waivers.fetch_and_match_free_agents`` asks Yahoo for, so the
#: synthesized pool has the same shape as the real one. A consequence worth
#: knowing: a UTIL/DH-only bat belongs to no bucket and so never enters the pool
#: -- but neither does he enter Yahoo's, because Yahoo is queried by those same
#: six positions plus SP/RP.
HITTER_RANK_BUCKETS: tuple[Position, ...] = (
    Position.C,
    Position.FIRST_BASE,
    Position.SECOND_BASE,
    Position.THIRD_BASE,
    Position.SS,
    Position.OF,
)

#: Default per-position cap, matching ``fa_per_position`` on the Yahoo path.
DEFAULT_PER_POSITION_CAP = 100

#: Volume floors. A rest-of-season line below these is either a September
#: call-up with no role or a projection-system artifact; either way it is not a
#: player this league can add and start.
DEFAULT_MIN_ROS_PA = 40.0
DEFAULT_MIN_ROS_IP = 12.0

#: How many names to name in an aggregate warning before saying "and N more".
_WARN_SAMPLE = 10


@dataclass(frozen=True)
class _Candidate:
    """One surviving projection row, with the identity used to key it."""

    name: str
    name_norm: str
    fg_id: str | None
    player_type: PlayerType
    positions: tuple[str, ...]


def build_manual_free_agents(
    req: FreeAgentRequest,
    *,
    positions_by_name: Mapping[str, Sequence[str]],
    excluded_names: frozenset[str] = frozenset(),
    per_position_cap: int = DEFAULT_PER_POSITION_CAP,
    min_ros_pa: float = DEFAULT_MIN_ROS_PA,
    min_ros_ip: float = DEFAULT_MIN_ROS_IP,
) -> list[Player]:
    """Build the free-agent pool from projections and the rostered-name sets.

    Args:
        req: The pool request. ``req.rostered_hitters`` /
            ``req.rostered_pitchers`` must be normalized names split by player
            type -- see :class:`lineup.waivers.FreeAgentRequest` for why the
            split matters.
        positions_by_name: normalized name -> eligible slots, as produced by
            :func:`keepers.positions.load_positions` (the committed
            ``data/player_positions.json``, which
            ``scripts/fetch_positions_mlb.py`` backfills from the MLB Stats API,
            merged under the frozen ``cache:positions`` blob, blob winning).
            Slots are filtered to the requested player type, so an ``IL`` token
            carried in the blob never reaches a ``Player``, and a two-way
            player's hitter slots never make him look like a hitter in the
            pitcher pool.
        excluded_names: already-normalized names to drop entirely, from
            ``data/manual/fa_exclusions.yaml`` via
            :func:`manual.transcripts.load_fa_exclusions`. Matched by name only,
            so an entry removes both types of a shared name.
        per_position_cap: candidates kept per hitter bucket. Pitchers get one
            combined bucket at twice this, mirroring the Yahoo path's two
            pitcher fetches (SP and RP).
        min_ros_pa: hitter volume floor, in projected rest-of-season PA.
        min_ros_ip: pitcher volume floor, in projected rest-of-season IP.

    Returns:
        ``list[Player]`` in the same shape ``fetch_and_match_free_agents``
        returns: ``.rest_of_season`` from the ROS frames, ``.preseason``
        attached when the preseason frames are supplied, ``.status`` empty.

    Raises:
        ValueError: if ``req.rankings_lookup`` is missing or empty. The rank cap
            is the only thing standing between the audit and a 9,000-row pool of
            fluky small-sample lines, and silently returning an unfiltered (or
            empty) pool would read as a result rather than as the failure it is.
    """
    if not req.rankings_lookup:
        raise ValueError(
            "build_manual_free_agents needs a populated rankings_lookup: it is the "
            "substitute for Yahoo's ownership filter. Run the pipeline's ranking "
            "step (RefreshRun._compute_rankings) before the roster audit."
        )

    candidates: list[_Candidate] = []
    for df, ptype, rostered, volume_col, min_volume in (
        (req.hitters_proj, PlayerType.HITTER, req.rostered_hitters, "pa", min_ros_pa),
        (req.pitchers_proj, PlayerType.PITCHER, req.rostered_pitchers, "ip", min_ros_ip),
    ):
        rows = _candidate_rows(df, rostered, min_volume, volume_col, excluded_names)
        typed = _with_positions(rows, ptype, positions_by_name)
        cap = per_position_cap if ptype == PlayerType.HITTER else per_position_cap * 2
        candidates.extend(_cap_by_position(typed, req.rankings_lookup, cap, ptype))

    fa_dicts: list[dict[str, Any]] = [
        {"name": c.name, "positions": list(c.positions), "status": ""} for c in candidates
    ]
    logger.info(
        "Manual FA pool: %d candidate(s) (%d hitters, %d pitchers) before projection matching",
        len(fa_dicts),
        sum(1 for c in candidates if c.player_type == PlayerType.HITTER),
        sum(1 for c in candidates if c.player_type == PlayerType.PITCHER),
    )

    # Imported here, not at module scope, to mirror the Yahoo path's lazy import
    # and keep this module importable from an offline script.
    from fantasy_baseball.data.projections import match_roster_to_projections

    return match_roster_to_projections(
        fa_dicts,
        req.hitters_proj,
        req.pitchers_proj,
        preseason_hitters_proj=req.preseason_hitters_proj,
        preseason_pitchers_proj=req.preseason_pitchers_proj,
        warn_unmatched=False,
        context="fa:manual",
    )


def _volume(row: Mapping[str, Any], column: str) -> float:
    """The row's playing-time value, with a missing/NaN cell read as 0.0.

    Written as an explicit ``is None`` / ``isna`` test rather than
    ``row.get(column) or 0.0``: a legitimate 0.0 and an absent cell are
    different facts, and the ``or`` idiom erases the difference (see CLAUDE.md).
    Here the two happen to land on the same number, which is exactly why the
    forbidden idiom would survive review -- so the correct one is written out.
    """
    raw = row.get(column)
    if raw is None or pd.isna(raw):
        return 0.0
    return float(raw)


def _candidate_rows(
    df: pd.DataFrame,
    rostered: frozenset[str],
    min_volume: float,
    volume_col: str,
    excluded: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Rows of ``df`` that are unrostered, not excluded, and clear the floor.

    ``rostered`` and ``excluded`` are normalized names; ``rostered`` must
    already be restricted to this frame's player type by the caller.

    Exclusions are applied HERE, before the rank cap, rather than after it: an
    excluded name must not consume one of the per-position cap slots and thereby
    shrink the usable pool. (The build plan sketched exclusions as a final pass;
    this ordering is a strict improvement and changes nothing else.)

    A frame that does not carry ``volume_col`` at all is a projection-schema
    change, not a pool of zero-playing-time players: the floor is skipped with a
    loud warning rather than silently emptying the pool.
    """
    if df.empty:
        return []
    if "_name_norm" not in df.columns:
        raise ValueError(
            "projection frame is missing the _name_norm column -- callers must "
            "precompute it (see lineup.waivers.fetch_and_match_free_agents)"
        )
    apply_floor = volume_col in df.columns
    if not apply_floor:
        logger.warning(
            "Projection frame has no %r column -- skipping the volume floor for it "
            "(possible projection-schema change).",
            volume_col,
        )

    kept: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        name_norm = str(row.get("_name_norm", ""))
        if not name_norm:
            continue
        if name_norm in rostered or name_norm in excluded:
            continue
        if apply_floor and _volume(row, volume_col) < min_volume:
            continue
        kept.append(row)
    return kept


def _eligible_positions(
    name: str,
    player_type: PlayerType,
    positions_by_name: Mapping[str, Sequence[str]],
) -> list[str]:
    """Canonical slots for ``name``, restricted to ``player_type``.

    Returns ``[]`` when a hitter has no usable entry -- the caller drops him.
    A pitcher with no usable entry falls back to ``["P"]``: that is what Yahoo
    reports for every pitcher in this league, and it is also the right answer
    for the pitcher half of a two-way player whose only stored eligibility is
    the hitter slot he is rostered at.
    """
    allowed = HITTER_ELIGIBLE if player_type == PlayerType.HITTER else PITCHER_ELIGIBLE
    slots: list[str] = []
    for token in positions_by_name.get(normalize_name(name), ()):
        try:
            parsed = Position.parse(str(token))
        except ValueError:
            continue
        if parsed in allowed and parsed.value not in slots:
            slots.append(parsed.value)
    if slots:
        return slots
    if player_type == PlayerType.PITCHER:
        return [Position.P.value]
    return []


def _with_positions(
    rows: Sequence[Mapping[str, Any]],
    player_type: PlayerType,
    positions_by_name: Mapping[str, Sequence[str]],
) -> list[_Candidate]:
    """Attach eligibility, dropping hitters we have no position data for.

    The drop is aggregated into one warning rather than one line per player: the
    unresolved tail is thousands of names long (the position sources cover
    rostered players and last season's free agents, not every projected minor
    leaguer), and a per-player warning would bury the rest of the refresh log.
    """
    out: list[_Candidate] = []
    unknown: list[str] = []
    for row in rows:
        name = str(row.get("name", ""))
        if not name:
            continue
        positions = _eligible_positions(name, player_type, positions_by_name)
        if not positions:
            unknown.append(name)
            continue
        raw_fg = row.get("fg_id")
        # The same expression ``sgp.rankings.compute_sgp_rankings`` uses to build
        # the fg_id key, so the two sides cannot drift into a silent lookup miss.
        fg_id = str(raw_fg) if raw_fg is not None and pd.notna(raw_fg) else None
        out.append(
            _Candidate(
                name=name,
                name_norm=str(row.get("_name_norm", normalize_name(name))),
                fg_id=fg_id,
                player_type=player_type,
                positions=tuple(positions),
            )
        )
    if unknown:
        sample = sorted(unknown)[:_WARN_SAMPLE]
        logger.warning(
            "Manual FA pool: dropped %d %s candidate(s) with no position data "
            "(not defaulted to UTIL -- a synthetic slot would fake an upgrade). "
            "Sample: %s%s",
            len(unknown),
            player_type,
            ", ".join(sample),
            f" and {len(unknown) - len(sample)} more" if len(unknown) > len(sample) else "",
        )
    return out


def _cap_by_position(
    candidates: Sequence[_Candidate],
    rankings_lookup: Mapping[str, dict[str, Any]],
    per_position_cap: int,
    player_type: PlayerType,
) -> list[_Candidate]:
    """Keep the best ``per_position_cap`` candidates per eligible-position bucket.

    "Best" is the rest-of-season SGP ordinal from
    :func:`sgp.rankings.lookup_rank` -- rank 1 is best, so the sort is
    ascending. Ties break on normalized name so the pool is reproducible run to
    run.

    A candidate with no stored rank is dropped: ``compute_sgp_rankings`` ranks
    every row of the same frames these candidates came from, so a missing rank
    means the row could not be scored at all. Note the test on ``is None``: rank
    is an ordinal whose best value is 1, and a truthiness test would eventually
    discard whichever player a future ranker numbers 0 -- the best one.

    Hitters bucket by :data:`HITTER_RANK_BUCKETS`; pitchers all share one bucket
    (the caller passes twice the cap for them). A multi-position hitter competes
    in -- and consumes a slot in -- every bucket he is eligible for, exactly as
    he would occupy a slot in each of Yahoo's per-position queries, and is
    de-duplicated afterwards on ``(normalized name, player_type)``.

    ``player_type`` selects the bucket set rather than being sniffed from the
    candidates, so an empty list still behaves like the pool it came from.
    """
    ranked: list[tuple[int, _Candidate]] = []
    for cand in candidates:
        rank_data = lookup_rank(rankings_lookup, cand.fg_id, cand.name, cand.player_type)
        rank = rank_data.get("rest_of_season")
        if rank is None:
            continue
        ranked.append((int(rank), cand))
    ranked.sort(key=lambda pair: (pair[0], pair[1].name_norm))

    buckets: list[str | None]
    if player_type == PlayerType.PITCHER:
        buckets = [None]  # one combined pitcher bucket, matched by every candidate
    else:
        buckets = [p.value for p in HITTER_RANK_BUCKETS]

    kept: dict[tuple[str, str], _Candidate] = {}
    for bucket in buckets:
        taken = 0
        for _rank, cand in ranked:
            if taken >= per_position_cap:
                break
            if bucket is not None and bucket not in cand.positions:
                continue
            kept.setdefault((cand.name_norm, cand.player_type), cand)
            taken += 1
    return list(kept.values())
