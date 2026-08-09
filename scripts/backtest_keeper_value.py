"""Out-of-sample bake-off: `shape` against the keeper-value chain (#325).

Answers the question #325's Phase 1 exists to answer -- does the
`keeper_forecast` -> `keeper_value` chain actually predict the KEEPER DECISION better
than `shape`? Not pool-wide rank, which the keeper-value teardown calls "nearly a
tautology": which three of a 23-man roster you would hold, and what that cost.

    python scripts/backtest_keeper_value.py                          # both pools, 2022-2024
    python scripts/backtest_keeper_value.py --base-year 2023
    python scripts/backtest_keeper_value.py --base-year 2023 --causal-check

Both estimators are scored by `trajectory.panel.score`, which is what puts them on one
SCALE rather than merely in one unit.

Five things here are load-bearing and were each got wrong once before being fixed:

  * era factors come from the FULL panel and truncation happens AFTER -- `era_normalize`
    refuses a panel missing its 2023-2025 reference window, so the other order aborts
    base years 2022 and 2023 outright
  * targets are CUMULATIVE per horizon, so +1 and +2 are different questions
  * the keeper decision is scored at the ROSTER level across BOTH pools; a team keeps
    three players, not three hitters and three pitchers
  * `breakout_mask` needs its positivity guard -- `prior = 0` means "out of the league",
    and without it 38-43% of the breakout slice was returns from absence
  * a two-way player is resolved to ONE pool before anything is merged, or an `|=` on
    id-keyed dicts silently deletes his bat

Keeper-value keeps three declared advantages (out-year vintage leakage, it reads ZiPS at
all, and a persistence fit that for base 2022 trains on LATER transitions). They are
declared rather than removed, so a shape win is the strong form of the result.

DELETED WITH THE CHAIN. This script exists to justify #325's retirement and goes with it;
nothing here is meant to survive PR 3.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from keeper_forecast import forecast_pool
from keeper_persistence import TRANSITIONS as KEEPER_TRANSITIONS
from keeper_persistence import load_rates

from fantasy_baseball.config import load_config
from fantasy_baseball.sgp.denominators import SgpOverrides, get_sgp_denominators
from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
from fantasy_baseball.trajectory.board import people, player_names, season_slots
from fantasy_baseball.trajectory.era import era_factors, era_normalize
from fantasy_baseball.trajectory.model import MIN_LOCAL_SUPPORT, collapse_split_seasons
from fantasy_baseball.trajectory.panel import DEFAULT_PANEL_DIR, load_scored_panel
from fantasy_baseball.trajectory.panel import score as panel_score
from fantasy_baseball.trajectory.shape import build_history, shape_trajectory
from fantasy_baseball.trajectory.value import best_floor, resolve_slots
from fantasy_baseball.utils.name_utils import normalize_name

#: Every year-pair the persistence fit could use, bounded by actuals coverage
#: (`data/stats/{pool}-{Y}.csv` exists for 2022-2025). Imported rather than restated so
#: this cannot drift from the fit `keeper_persistence` actually validated.
ALL_TRANSITIONS = KEEPER_TRANSITIONS


def transitions_for(base_year: int, mode: str) -> tuple[tuple[int, int], ...]:
    """Which (year, year+1) transitions the persistence fit may use for `base_year`.

    `loto` drops ONLY the transition being predicted. It does not make the fit causal,
    and the difference matters: for base 2022 both survivors are LATER than the
    transition predicted, so the fit trains on the future. That is disclosed in the
    writeup as a third advantage keeper-value keeps, rather than silently corrected,
    because a strictly causal rule leaves base 2022 with nothing to fit on at all and
    base 2023 with one transition -- which would delete the +2 horizon and with it the
    multi-year claim this evaluation exists to make.

    `causal` is the sensitivity variant, and it is only informative for base 2023:

        base 2022  loto = 2 transitions, both future   causal = 0   not computable
        base 2023  loto = 2, one future                causal = 1   THIS is the check
        base 2024  loto = 2, none future               causal = 2   identical, measures nothing
    """
    if mode not in {"loto", "causal"}:
        raise ValueError(f"mode must be 'loto' or 'causal', got {mode!r}")
    if mode == "loto":
        predicted = (base_year, base_year + 1)
        return tuple(t for t in ALL_TRANSITIONS if t != predicted)
    return tuple(t for t in ALL_TRANSITIONS if t[1] <= base_year)


#: The last season that can serve as an OUTCOME. 2026 is in progress, and the only
#: tool for comparing it against full seasons -- `panel.prorate_partial` -- is
#: straight-line and explicitly assumes the player stays healthy. Pacing an outcome
#: season would scale an injured player's line up as if he had not been hurt, which is
#: exactly the confound the injury-excluded view exists to remove.
LAST_OUTCOME_SEASON = 2025


def horizons_for(base_year: int) -> tuple[int, ...]:
    """Which forward years are scoreable from `base_year`.

    The single source of truth for this, so the shape side, the keeper side and the
    slice counts cannot disagree about which base years support a multi-year target.
    """
    return tuple(h for h in (1, 2) if base_year + h <= LAST_OUTCOME_SEASON)


def without_player(panel: pd.DataFrame, query_id: int) -> pd.DataFrame:
    """The panel both estimators see for one query: no self-matching.

    Cheap by design and called in the inner loop. An in-sample comparison would flatter
    `shape`, which fits a model, over an estimator that averages.
    """
    return panel[panel["mlbam_id"] != query_id]


#: Outcome-year volume below this share of the ANCHOR year's counts as wrecked and
#: leaves the injury-excluded view. Chosen by @alhart2015: injury is close to random,
#: and charging an otherwise-correct keeper decision for it confounds the comparison.
CENSOR_THRESHOLD = 0.5


@dataclass(frozen=True)
class Outcome:
    """What a player actually did in the outcome years, and how much he played.

    `sgp_by_year` and `volume_by_year` are SPARSE: a season the player did not appear
    in is ABSENT, not zero, so "played badly" and "was not there" stay distinguishable
    all the way to the censoring rule. `realized` collapses absence to the 0 a vanished
    player is worth to a roster slot; `censored` treats it as zero volume. Those are
    different questions and the two methods answer them differently on purpose.

    `frozen=True` blocks attribute reassignment only -- the dicts remain mutable and the
    generated `__hash__` would raise on them. Outcomes are never hashed and each owns
    its dicts; do not start sharing them.
    """

    mlbam_id: int
    sgp_by_year: dict[int, float]
    volume_by_year: dict[int, float]
    anchor_volume: float

    def realized(self, years: Sequence[int]) -> float:
        return sum(self.sgp_by_year.get(y, 0.0) for y in years)


def censored(outcome: Outcome, years: Sequence[int], threshold: float = CENSOR_THRESHOLD) -> bool:
    """True if ANY outcome year falls under `threshold` of the ANCHOR year's volume.

    ANY, not all: a one-year sum and a two-year sum are not the same target, so a
    player wrecked in one of two years leaves the multi-year metric entirely rather
    than contributing a shorter one.

    The ratio is against the ANCHOR (year Y) for every outcome year, never against the
    preceding outcome. A wrecked Y+1 must not be allowed to redefine "normal" for Y+2 --
    against a 100-PA Y+1, a 500-PA Y+2 would read as a recovery and pass.

    Censoring is a property of the realized outcome, not of either forecast, so both
    estimators lose identical rows and it cannot favour one.
    """
    if outcome.anchor_volume <= 0:
        return True
    return any(
        outcome.volume_by_year.get(year, 0.0) < threshold * outcome.anchor_volume for year in years
    )


def outcomes_for(
    panel: pd.DataFrame,
    kind: str,
    base_year: int,
    horizons: tuple[int, ...],
    anchor_volume: Mapping[int, float],
    ids: Sequence[int] | None = None,
) -> dict[int, Outcome]:
    """What each player actually did in `base_year + h` for every h in `horizons`.

    SGP and volume come from two DIFFERENT places, and that is not an accident:

      * SGP -> `collapse_split_seasons`, which aggregates `sgp=("sgp", "sum")`. Used
        because it is the definition both estimators already fit on, not because
        summing is self-evidently correct for rate categories.
      * volume -> a separate groupby on the RAW panel.

    The collapse drops `pa` and `ip` entirely, and it returns the panel *untouched*
    when no season is split. So its output schema differs by data -- full columns when
    nothing is split, four columns when something is -- and reading volume off it works
    on ordinary fixtures while failing on precisely the split season it exists to
    handle. `_ROLE_SUMS` above re-sums the pitcher role columns for the same reason.

    Getting this wrong is not a crash. A traded player's 600-PA season reads as 310 +
    290, both under half a 600-PA anchor, and the injury-excluded view censors him as
    wrecked -- a false positive concentrated on players who changed teams.
    """
    volume_col = "pa" if kind == "hitter" else "ip"
    collapsed = collapse_split_seasons(panel).set_index(["mlbam_id", "season"])["sgp"]
    volumes = panel.groupby(["mlbam_id", "season"])[volume_col].sum()

    years = [base_year + h for h in horizons]
    wanted = panel["mlbam_id"].unique() if ids is None else ids
    out: dict[int, Outcome] = {}
    for pid in wanted:
        pid = int(pid)
        sgp_by_year = {y: float(collapsed[(pid, y)]) for y in years if (pid, y) in collapsed.index}
        volume_by_year = {y: float(volumes[(pid, y)]) for y in years if (pid, y) in volumes.index}
        out[pid] = Outcome(
            mlbam_id=pid,
            sgp_by_year=sgp_by_year,
            volume_by_year=volume_by_year,
            # Passed in rather than re-derived: the anchor is year Y and this function
            # only ever looks forward.
            anchor_volume=float(anchor_volume.get(pid, 0.0)),
        )
    return out


@dataclass(frozen=True)
class RosterResolution:
    """Drafted names resolved to mlbam ids, with every miss accounted for.

    Four buckets, not two, because the misses mean different things and lumping them
    would hide the one that matters:

      `by_team`      team -> the ids that resolved AND are scoreable
      `pool_of`      id -> "hitter"/"pitcher", the pool his roster spot is judged in
      `unresolved`   (team, name) the people cache has no id for at all
      `ambiguous`    (team, name) matching more than one scoreable id
      `unscoreable`  (team, name) that resolved fine but has no panel seasons
    """

    by_team: dict[str, list[int]]
    pool_of: dict[int, str]
    unresolved: list[tuple[str, str]]
    ambiguous: list[tuple[str, str]]
    unscoreable: list[tuple[str, str]]


def resolve_draft(
    draft: Sequence[Mapping[str, str]],
    people: pd.DataFrame,
    pool_by_id: Mapping[int, str],
    var_by_pool: Mapping[tuple[str, int], float] | None = None,
) -> RosterResolution:
    """Resolve `data/historical_drafts_resolved.json` records to mlbam ids.

    THIS IS THE RISK IN THE WHOLE HARNESS. The draft file carries bare names
    (`"player": "Yordan Alvarez"`) and everything else here is keyed on `mlbam_id`.
    `CLAUDE.md` names bare-name joins as a defect class, and
    `trajectory/roster_join.py` records that `(normalized_name, pool)` is not unique --
    the live board has two hitters called Max Muncy.

    So nothing is dropped silently. An unresolved or ambiguous name is REPORTED, because
    a silent drop thins roster pools toward the fringe -- flattering both estimators and
    shrinking the decision being measured -- and would look exactly like a clean run.

    `pool_by_id` comes from PANEL MEMBERSHIP (the draft records carry no position), and
    an id in both panels is `"both"`. Such a player enters his roster ONCE, under
    whichever pool gives the higher year-Y VAR: the league scores him twice but a keeper
    decision is for one roster spot, and entering him twice would let one player consume
    two of a team's three slots.
    """
    by_name: dict[str, list[int]] = {}
    for pid, name in zip(people["id"], people["fullName"], strict=False):
        by_name.setdefault(normalize_name(str(name)), []).append(int(pid))

    by_team: dict[str, list[int]] = {}
    pool_of: dict[int, str] = {}
    unresolved: list[tuple[str, str]] = []
    ambiguous: list[tuple[str, str]] = []
    unscoreable: list[tuple[str, str]] = []

    for record in draft:
        team, name = str(record["team"]), str(record["player"])
        by_team.setdefault(team, [])
        candidates = by_name.get(normalize_name(name), [])
        if not candidates:
            unresolved.append((team, name))
            continue
        scoreable = [pid for pid in candidates if pid in pool_by_id]
        if not scoreable:
            # The NAME resolved; the player simply has no panel seasons. A different
            # failure from "no such name", and conflating them would overstate the join.
            unscoreable.append((team, name))
            continue
        if len(scoreable) > 1:
            ambiguous.append((team, name))
            continue
        pid = scoreable[0]
        pool = pool_by_id[pid]
        if pool == "both":
            lookups = var_by_pool or {}
            pool = max(
                ("hitter", "pitcher"),
                key=lambda p: lookups.get((p, pid), float("-inf")),
            )
        pool_of[pid] = pool
        by_team[team].append(pid)
    return RosterResolution(by_team, pool_of, unresolved, ambiguous, unscoreable)


#: How many players a team may retain. Ten teams x three keepers is also where the
#: top-of-board slice's 30 comes from.
KEEP_SLOTS = 3


#: Ten teams. The pooled top-30 is KEEP_SLOTS x this.
LEAGUE_TEAMS = 10


#: A slice thinner than this is reported as unmeasurable rather than scored.
MIN_REPORTABLE_SLICE = 10


#: How many censored players and join misses to name before summarizing.
CENSORED_LIST_LIMIT = 12


JOIN_MISS_LIMIT = 8


#: A roster thinned below this many candidates leaves the triple slice: picking 3 from
#: 4 is not the decision being measured.
CANDIDATE_FLOOR = 5


def usable_draft_years(horizon: int, available: Sequence[int]) -> list[int]:
    """Draft years whose keeper decision can be scored at `horizon`.

    Derived from `horizons_for` rather than typed out, so a base year dropping out
    changes the decision count and something notices. The headline counts are 10
    (multi-year, draft 2023 only) and 20 (one-year, 2023 and 2024).
    """
    return [year for year in available if horizon in horizons_for(year)]


def intersect(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Players BOTH estimators can score. Slices run on this or they compare two
    different populations and the difference is partly a population difference."""
    return sorted(set(a) & set(b))


def eligible_rosters(
    by_team: Mapping[str, Sequence[int]], scoreable: set[int], floor: int = CANDIDATE_FLOOR
) -> tuple[dict[str, list[int]], list[str]]:
    """Rosters with enough scoreable candidates to pose the keeper question.

    Applied PER VIEW. Censored players leave the candidate pool in the injury-excluded
    view, so a roster can clear the floor in ALL and fail it there -- and if that is not
    reported, a difference between the views reads as "excluding injuries changed the
    answer" when it means "different teams were scored", which is the one thing running
    two views is supposed to separate.
    """
    kept, dropped = {}, []
    for team, ids in by_team.items():
        candidates = [pid for pid in ids if pid in scoreable]
        if len(candidates) >= floor:
            kept[team] = candidates
        else:
            dropped.append(team)
    return kept, sorted(dropped)


def triple_regret(
    candidates: Sequence[int],
    forecast: Mapping[int, float],
    realized: Mapping[int, float],
    keep: int = KEEP_SLOTS,
) -> tuple[tuple[int, ...], float]:
    """The keeper decision: pick `keep` on FORECAST, score them on REALIZED.

    Regret is the realized shortfall against the ex-post best `keep` from the same
    roster -- ranking wrong only costs what it actually cost. Ties break on `mlbam_id`
    so a rerun cannot silently pick a different triple.
    """
    ranked = sorted(candidates, key=lambda pid: (-forecast.get(pid, 0.0), pid))
    picked = tuple(sorted(ranked[:keep]))
    best = sorted(candidates, key=lambda pid: (-realized.get(pid, 0.0), pid))[:keep]
    shortfall = sum(realized.get(pid, 0.0) for pid in best) - sum(
        realized.get(pid, 0.0) for pid in picked
    )
    return picked, shortfall


def agreement_rate(a: Sequence[tuple[int, ...]], b: Sequence[tuple[int, ...]]) -> float:
    """Share of decisions where both estimators named the same players.

    Report this or the bootstrap lies about what it measured. The best three on a
    23-man roster are usually not close, so most decisions agree -- and every agreeing
    decision contributes exactly zero to the difference while still counting toward n.
    18 agreements out of 20 produce a tight interval around zero that reads as "cannot
    separate" when it means the slice had two informative rows.
    """
    if len(a) != len(b):
        raise ValueError(
            f"pick lists must be the same length ({len(a)} vs {len(b)}); both estimators "
            "pick from one candidate pool per view, so a mismatch is a bug rather than "
            "something to zip over"
        )
    if not a:
        return 0.0
    return sum(set(x) == set(y) for x, y in zip(a, b, strict=True)) / len(a)


def top_of_board(
    forecast: Mapping[int, float], realized: Mapping[int, float], n: int
) -> tuple[tuple[int, ...], float]:
    """Realized value of the `n` players an estimator ranked highest."""
    ranked = sorted(forecast, key=lambda pid: (-forecast[pid], pid))[:n]
    return tuple(ranked), sum(realized.get(pid, 0.0) for pid in ranked)


def breakout_mask(anchors: pd.DataFrame, factor: float = 1.25) -> pd.Series:
    """Query seasons at least `factor` above the prior one.

    `current`/`prior` are `build_history`'s own column names; do not rename them here.

    Estimator-neutral: both anchors are REALIZED seasons, so this selects a population
    rather than favouring the side that happens to model breakouts.

    **The positivity guard is load-bearing, not defensive.** `current > factor * prior`
    only means "stepped up" when `prior > 0`:

      * `build_history` assigns `prior = 0` to a season the player was ABSENT for, and
        every positive `current` clears `1.25 * 0`. On the real panel that admitted
        144 of 333 hitters and 168 of 405 pitchers -- 38-43% of the slice -- none of
        them breakouts, all of them returns from absence.
      * a negative prior inverts the inequality: `-2.2 > 1.25 * -2.0` is true, so a
        season that got WORSE qualified. Rare (0-2 per cell) but wrong.

    The first case is the dangerous one, because a zero prior is exactly where `shape`
    is structurally weakest -- its prior anchor carries no information there, while the
    keeper chain still has a real ZiPS projection. Admitting them loads the slice
    against shape for a reason that has nothing to do with breakouts.
    """
    return (anchors["prior"] > 0) & (anchors["current"] > factor * anchors["prior"])


def bootstrap_difference(
    a: Sequence[float], b: Sequence[float], *, draws: int = 10_000, seed: int = 7
) -> tuple[float, float, float]:
    """Paired bootstrap on `mean(a) - mean(b)`.

    Returns `(lo, hi, share_a_lower)` at the 2.5/97.5 percentiles. **PAIRED**: one index
    draw indexes both samples, so the two estimators are always compared on the same
    resampled decisions.

    `share_a_lower` is the fraction of draws where `a`'s mean is BELOW `b`'s. Lower is
    better for regret and for error, so that is "a wins" for every metric here -- but
    the direction inverts if this is ever handed a metric where high is good, so check
    before reusing it.

    The RESAMPLING UNIT is the caller's choice and it matters: team-decisions for the
    triple slice (resampling players inside a roster would change the roster, which
    changes the ex-post optimum and leaves regret undefined), players elsewhere.
    """
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap needs equal lengths, got {len(a)} and {len(b)}")
    if not a:
        return float("nan"), float("nan"), float("nan")
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(left), size=(draws, len(left)))
    diffs = left[idx].mean(axis=1) - right[idx].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))
    return lo, hi, float((diffs < 0).mean())


def _two_way_owner(runs: Sequence[PoolRun]) -> dict[int, str]:
    """`mlbam_id -> owning pool`, for ids that appear in more than one panel.

    The league scores a two-way player twice; a keeper decision is for one roster spot.
    Highest base-year SGP wins, with the hitter side breaking ties -- that is where the
    roster spot conventionally sits. Only contested ids appear in the result.
    """
    seen: dict[int, list[tuple[float, str]]] = {}
    for run in runs:
        for pid in run.common:
            best = max(run.shape_sgp.get(pid, {0: 0.0}).values(), default=0.0)
            seen.setdefault(pid, []).append((best, run.kind))
    return {
        pid: max(sides, key=lambda s: (s[0], s[1] == "hitter"))[1]
        for pid, sides in seen.items()
        if len(sides) > 1
    }


def merge_pools(
    per_pool: Sequence[tuple[str, Mapping[int, float]]],
    prefer: Mapping[int, str],
) -> tuple[dict[int, float], dict[int, str], list[tuple[int, str]]]:
    """Merge the pools' id-keyed maps, resolving a two-way player to ONE pool.

    A plain `|=` is last-wins, and `runs` is ordered hitter-then-pitcher, so a two-way
    id kept only its pitcher row -- the bat's forecast and realized VAR deleted with no
    count and no warning. On the real panel that is Shohei Ohtani in base 2022 and
    2023, the two years carrying the multi-year claim, entering the pooled top-30 and
    the roster triple valued as a 130-inning pitcher.

    `prefer` says which pool each contested id belongs to; the discarded side is
    RETURNED rather than dropped, so the caller can report it.
    """
    merged: dict[int, float] = {}
    pool_of: dict[int, str] = {}
    dropped: list[tuple[int, str]] = []
    for kind, values in per_pool:
        for pid, value in values.items():
            chosen = prefer.get(pid, kind)
            if chosen == kind:
                merged[pid] = value
                pool_of[pid] = kind
            else:
                dropped.append((pid, kind))
    return merged, pool_of, sorted(dropped)


def keeper_value_sgp(
    frame: pd.DataFrame, kind: str, sgp_overrides: SgpOverrides | None
) -> pd.Series:
    """SGP for a keeper-value forecast frame, scored by the PANEL's own scorer.

    This is what puts the two estimators on one scale rather than merely in one unit.
    `forecast_pool` emits the canonical rate/PT schema -- `keepers.actuals.HITTER_PT`
    is literally `"pa"` and `HITTER_RATES` are character-for-character the columns
    `panel.score` reconstructs from -- so the forecast can be handed to the scorer the
    realized seasons went through, with no translation step to disagree about.

    Deliberately NOT via `keeper_forecast.to_counting`, which renames to `PA`/`IP` and
    finishes `AVG`/`ERA`/`WHIP`: that output is for display and would need a second
    scoring path.
    """
    scored = panel_score(frame.reset_index(), kind, sgp_overrides)
    return scored.set_index("mlbam_id")["sgp"]


def slots_for(
    kind: str,
    base_year: int,
    raw_panel: pd.DataFrame,
    cache_dir: Path,
) -> dict[int, set[str]]:
    """The slot each player is priced at, from the base year.

    HITTERS route through `board.season_slots` -- the fielding leaderboard's 10-game
    rule -- and fall back to UTIL, the highest floor, when the cache cannot answer.

    PITCHERS route on ROLE, from the base season's starts and games, because
    `resolve_slots`' pitcher branch decides SP vs RP that way and its defaults are
    `starts=0, games=0`. Calling it without them puts every arm on the reliever floor.
    That was harmless while the pools were scored separately and never compared; it
    became load-bearing the moment a roster decision put hitters and pitchers up for
    the same three slots, because a starter then carries 1.87 SGP a season of credit
    (3.7 at +2) that no hitter gets.
    """
    if kind == "pitcher":
        year = raw_panel[raw_panel["season"] == base_year]
        agg = year.groupby("mlbam_id")[["starts", "games"]].sum()
        return {
            int(pid): resolve_slots(
                None, "pitcher", starts=float(row.starts), games=float(row.games)
            )
            for pid, row in agg.iterrows()
        }
    eligibility = season_slots(cache_dir, base_year)
    return {int(pid): resolve_slots(set(slots), "hitter") for pid, slots in eligibility.items()}


def var_for(
    sgp_by_id: pd.Series,
    kind: str,
    base_year: int,
    cache_dir: Path,
    levels: dict[str, float],
    seasons: int = 1,
    slots_by_id: Mapping[int, set[str]] | None = None,
) -> pd.Series:
    """SGP above the position-aware floor, using year-`base_year` eligibility.

    `seasons` is how many years `sgp_by_id` sums, and the floor is charged once per
    year. A two-season total netted against ONE season of replacement halves the
    scarcity credit -- the catcher-to-outfield spread collapses from 4.5 SGP to 2.3 --
    which reorders catchers against outfielders in precisely the +2 slices the
    multi-year claim rests on.

    Year Y, not the outcome year: Y is the information set the keeper decision actually
    has, and outcome-year eligibility would be hindsight. The catcher-to-outfield floor
    spread is 2.3 SGP a year, larger than the margins this backtest is trying to
    resolve, so the choice is not cosmetic.

    Reuses `trajectory.board.season_slots` rather than re-reading the fielding cache: a
    second eligibility path would be free to price the backtest's catchers differently
    from the live board's. Its existing fallback carries through -- a missing or corrupt
    cache degrades to UTIL, the HIGHEST hitter floor, so an unknown player is
    understated rather than credited with scarcity he may not have.
    """
    lookup = (
        slots_by_id
        if slots_by_id is not None
        else slots_for(
            kind,
            base_year,
            pd.DataFrame(columns=["mlbam_id", "season", "starts", "games"]),
            cache_dir,
        )
    )
    floors = {}
    for pid in sgp_by_id.index:
        floors[pid] = best_floor(lookup.get(int(pid), set()), levels)[1]
    return sgp_by_id - seasons * pd.Series(floors, dtype=float).reindex(sgp_by_id.index)


DRAFT_FILE = PROJECT_ROOT / "data" / "historical_drafts_resolved.json"


FIELDING_CACHE = PROJECT_ROOT / "data" / "cache" / "keeper_skills"


@dataclass
class PoolRun:
    """One pool's contribution to one base year, before the pools are merged.

    A keeper decision is for a 23-man ROSTER spanning both pools -- a team keeps three
    PLAYERS, not three hitters and three pitchers. So no slice may run inside the
    per-pool loop; each pool produces one of these and the reporting merges them.
    """

    kind: str
    common: list[int]
    shape_sgp: dict[int, dict[int, float]]
    keeper_sgp: dict[int, dict[int, float]]
    outcomes: dict[int, Outcome]
    low_support: set[int]
    excluded: bool
    #: base-year anchors (`current`/`prior`), for the breakout slice
    anchors: pd.DataFrame
    #: mlbam_id -> the slot it is priced at, from the BASE year
    slots: dict[int, set[str]]


@dataclass
class PoolPanels:
    """Panels for one pool, loaded and normalized once and reused across base years."""

    raw: pd.DataFrame
    factors: pd.DataFrame
    full: pd.DataFrame


def _anchor_volume(raw_panel: pd.DataFrame, kind: str, base_year: int) -> dict[int, float]:
    """Year-Y playing time per player, summed across a split season."""
    volume = "pa" if kind == "hitter" else "ip"
    year = raw_panel[raw_panel["season"] == base_year]
    return year.groupby("mlbam_id")[volume].sum().to_dict()


def _shape_forecasts(
    truncated: pd.DataFrame,
    anchors: pd.DataFrame,
    kind: str,
    ids: Sequence[int],
    horizons: tuple[int, ...],
) -> tuple[dict[int, dict[int, float]], set[int]]:
    """`{mlbam_id: {horizon: mean}}` with the query player held out each time.

    Returns the low-support IDS, not a count, so the sensitivity line can actually
    exclude them. Those rows stay in the headline: dropping them would flatter shape by
    removing exactly the predictions it is least sure of.
    """
    lookup = anchors.set_index("mlbam_id")
    out: dict[int, dict[int, float]] = {}
    low_support: set[int] = set()
    for i, pid in enumerate(ids, start=1):
        if i % 100 == 0:
            print(f"      shape {i}/{len(ids)}...", flush=True)
        row = lookup.loc[pid]
        traj, _ = shape_trajectory(
            without_player(truncated, pid),
            kind=kind,
            age=int(row["age"]),
            sgp=float(row["current"]),
            prior_sgp=float(row["prior"]),
            horizons=horizons,
        )
        means = {h: float(point.mean) for h, point in zip(horizons, traj.path, strict=True)}
        if any(np.isnan(v) for v in means.values()):
            continue
        if traj.local_support < MIN_LOCAL_SUPPORT:
            low_support.add(int(pid))
        out[int(pid)] = means
    return out, low_support


def score_pool(
    kind: str,
    base_year: int,
    horizons: tuple[int, ...],
    panels: PoolPanels,
    args: argparse.Namespace,
    overrides: SgpOverrides | None,
) -> PoolRun | None:
    """Both estimators' forecasts for one pool and base year, on their intersection."""
    print(f"\n  [{kind}]")
    mode = "causal" if args.causal_check else "loto"
    transitions = transitions_for(base_year, mode)
    future = sum(1 for _, end in transitions if end > base_year + 1)
    print(f"    persistence fit: {mode}, {len(transitions)} transitions, {future} FUTURE")
    if not transitions:
        print("    -- no transitions to fit on; not scoreable in this mode")
        return None

    observed = load_rates(base_year, kind, source="actual", factors=panels.factors)
    keeper_sgp: dict[int, dict[int, float]] = {}
    fallbacks = []
    for h in horizons:
        frame, fallback = forecast_pool(
            kind,
            base_year,
            base_year + h,
            observed,
            args,
            transitions=transitions,
            factors=panels.factors,
        )
        fallbacks.append(fallback)
        for pid, value in keeper_value_sgp(frame, kind, overrides).items():
            keeper_sgp.setdefault(int(pid), {})[h] = float(value)
    worst = max(fallbacks, key=lambda f: (f.whole_pool, f.share))
    print(
        f"    gap-model fallback: {worst.per_player}/{worst.total}"
        f" ({worst.share:.0%}), whole-pool={worst.whole_pool}"
    )
    excluded = worst.exceeds_headline_threshold
    if excluded:
        print("    ** FLAGGED: mostly gap model; split out of the headline **")

    # `panels.full` is ALREADY era-normalized on the full panel, so slicing it here is
    # the normalize-then-truncate order without a second 18,000-row rescoring.
    truncated = panels.full[panels.full["season"] <= base_year]
    anchors = build_history(truncated)
    anchors = anchors[anchors["season"] == base_year]
    common = intersect(list(anchors["mlbam_id"]), list(keeper_sgp))
    print(
        f"    coverage: shape {len(anchors)}, keeper-value {len(keeper_sgp)},"
        f" intersection {len(common)}"
    )
    if not common:
        print("    -- empty intersection; nothing to compare")
        return None

    shape_sgp, low_support = _shape_forecasts(truncated, anchors, kind, common, horizons)
    common = intersect(common, list(shape_sgp))
    print(f"    scored by both: {len(common)}   low-support shape rows: {len(low_support)}")

    outcomes = outcomes_for(
        panels.full, kind, base_year, horizons, _anchor_volume(panels.raw, kind, base_year), common
    )
    return PoolRun(
        kind,
        common,
        shape_sgp,
        keeper_sgp,
        outcomes,
        low_support,
        excluded,
        anchors[anchors["mlbam_id"].isin(common)].copy(),
        slots_for(kind, base_year, panels.raw, FIELDING_CACHE),
    )


def run_historical(args: argparse.Namespace) -> int:
    """The #325 head-to-head: shape against the keeper-value chain, out of sample.

    Base year is the OUTER loop and the pools are merged before any slice runs, because
    the decision being measured -- which three of my 23 do I keep -- does not respect
    the hitter/pitcher split.
    """
    overrides = load_config(PROJECT_ROOT / "config" / "league.yaml").sgp_overrides
    levels = position_aware_replacement_levels(get_sgp_denominators(overrides))
    drafts = json.loads(DRAFT_FILE.read_text(encoding="utf-8"))
    pools = [args.pool] if args.pool else ["hitter", "pitcher"]
    if len(pools) == 1:
        print(
            f"NOTE: scoring {pools[0]}s only. A keeper roster spans both pools, so the\n"
            "      triple slice below is 'best 3 within this pool', not the real\n"
            "      decision. Drop --pool for the roster-level answer."
        )

    panels = {}
    for kind in pools:
        raw = load_scored_panel(kind, panel_dir=args.panel_dir, sgp_overrides=overrides)
        panels[kind] = PoolPanels(
            raw=raw,
            factors=era_factors(raw, kind),
            full=era_normalize(raw, kind, sgp_overrides=overrides),
        )

    pooled: list[dict] = []
    for base_year in args.base_year:
        horizons = horizons_for(base_year)
        if not horizons:
            print(f"\nbase {base_year}: no scoreable horizon, skipped")
            continue
        print(f"\n{'=' * 88}\nBASE {base_year} -- horizons {horizons}\n{'=' * 88}")
        runs = [
            run
            for kind in pools
            if (run := score_pool(kind, base_year, horizons, panels[kind], args, overrides))
        ]
        if runs:
            pooled += report_base_year(base_year, horizons, runs, levels, drafts, args)
    report_pooled(pooled, args)
    return 0


def _var(
    sgp_by_id: Mapping[int, float],
    kind: str,
    base_year: int,
    levels: dict[str, float],
    seasons: int,
    slots_by_id: Mapping[int, set[str]],
) -> dict[int, float]:
    """VAR for a total that sums `seasons` years. See `var_for` on why that matters."""
    series = pd.Series(dict(sgp_by_id), dtype=float)
    if series.empty:
        return {}
    return var_for(
        series, kind, base_year, FIELDING_CACHE, levels, seasons=seasons, slots_by_id=slots_by_id
    ).to_dict()


def report_base_year(
    base_year: int,
    horizons: tuple[int, ...],
    runs: Sequence[PoolRun],
    levels: dict[str, float],
    drafts: dict,
    args: argparse.Namespace,
) -> list[dict]:
    """Every slice for one base year, across BOTH pools, in both views.

    The pools are merged before anything is sliced. A keeper decision is for a 23-man
    roster that spans hitters and pitchers -- a team keeps three PLAYERS -- so scoring
    "best 3 hitters" and "best 3 pitchers" separately answers a question nobody asks
    and doubles the decision count.

    Wiring only: every number comes from a helper that has its own tests.
    """
    collected: list[dict] = []
    low_support = {pid for run in runs for pid in run.low_support}
    flagged_pools = {run.kind for run in runs if run.excluded}
    # A two-way id is in BOTH panels. Decide ONCE, before anything is merged, which
    # pool owns his roster spot -- the league scores him twice but a keeper decision is
    # for one slot. Highest base-year SGP wins; ties go to the hitter side, which is
    # where a two-way player's roster spot conventionally sits.
    prefer = _two_way_owner(runs)

    for horizon in horizons:
        # CUMULATIVE: +1 is Y+1, +2 is Y+1 and Y+2 summed. One summed target reused for
        # every horizon made the two rows identical, which the first smoke run showed.
        years = [base_year + h for h in horizons if h <= horizon]
        wrecked: set[int] = set()
        per_pool_var: list[tuple[str, dict[int, float]]] = []
        shape_parts: list[tuple[str, dict[int, float]]] = []
        keeper_parts: list[tuple[str, dict[int, float]]] = []
        censor_payload = []

        for run in runs:
            usable = [
                pid
                for pid in run.common
                if pid in run.outcomes
                and _finite(run.shape_sgp.get(pid), horizon)
                and _finite(run.keeper_sgp.get(pid), horizon)
            ]
            realized = {pid: run.outcomes[pid].realized(years) for pid in usable}
            shape_tot = {
                pid: sum(v for h, v in run.shape_sgp[pid].items() if h <= horizon) for pid in usable
            }
            keeper_tot = {
                pid: sum(v for h, v in run.keeper_sgp[pid].items() if h <= horizon)
                for pid in usable
            }
            per_pool_var.append(
                (run.kind, _var(realized, run.kind, base_year, levels, horizon, run.slots))
            )
            shape_parts.append(
                (run.kind, _var(shape_tot, run.kind, base_year, levels, horizon, run.slots))
            )
            keeper_parts.append(
                (run.kind, _var(keeper_tot, run.kind, base_year, levels, horizon, run.slots))
            )
            pool_wrecked = {
                pid for pid in usable if censored(run.outcomes[pid], years, args.censor_threshold)
            }
            wrecked |= pool_wrecked
            if horizon == horizons[-1]:
                censor_payload.append((run, usable, pool_wrecked))

        realized_var, pool_of, dropped = merge_pools(per_pool_var, prefer)
        shape_var, _, _ = merge_pools(shape_parts, prefer)
        keeper_var, _, _ = merge_pools(keeper_parts, prefer)
        per_pool = {
            kind: [pid for pid in realized_var if pool_of[pid] == kind]
            for kind in {run.kind for run in runs}
        }
        # Wrecked is a property of the OUTCOME, so a two-way id could be censored on the
        # side he does not own. Keep only the owning pool's verdict.
        wrecked &= set(realized_var)

        label = "multi-year" if horizon > 1 else "one-year"
        print(f"\n== target +{horizon} ({label}) ==")
        if flagged_pools:
            print(f"     NOTE: {sorted(flagged_pools)} flagged as mostly gap model,")
            print("           and NOT excluded -- their players are in every slice below")
        if dropped:
            names = player_names(FIELDING_CACHE)
            for pid, side in dropped:
                print(
                    f"     two-way: {str(names.get(pid, pid))[:24]} scored as"
                    f" {prefer.get(pid, '?')}; his {side} line is out of the pool"
                )
        for run, usable, pool_wrecked in censor_payload:
            _report_censored(run, usable, pool_wrecked, years, args)
        for view in ("ALL", "INJURY-EXCLUDED"):
            ids = [pid for pid in realized_var if view == "ALL" or pid not in wrecked]
            print(f"  -- {view} ({len(ids)} players) --")
            if len(ids) < MIN_REPORTABLE_SLICE:
                print(f"     under {MIN_REPORTABLE_SLICE}, not reported")
                continue
            _report_top_of_board(ids, per_pool, shape_var, keeper_var, realized_var, args)
            _report_low_support_sensitivity(
                ids, low_support, shape_var, keeper_var, realized_var, args
            )
            _report_breakout(ids, runs, base_year, shape_var, keeper_var, realized_var, args)
            collected += _report_triples(
                base_year,
                horizon,
                view,
                ids,
                pool_of,
                shape_var,
                keeper_var,
                realized_var,
                drafts,
                args,
            )
    return collected


def _finite(by_horizon: dict[int, float] | None, horizon: int) -> bool:
    """True when every horizon up to `horizon` has a usable number.

    Both sides are filtered by this. The shape side used to drop NaNs while the keeper
    side kept them, which is worse than asymmetric coverage: a NaN forecast reaches
    `triple_regret`'s sort key, every comparison against it is False, and the triple
    then depends on input order rather than on the forecast.
    """
    if not by_horizon:
        return False
    values = [v for h, v in by_horizon.items() if h <= horizon]
    return len(values) == horizon and not any(np.isnan(v) for v in values)


def _report_censored(run: PoolRun, ids: Sequence[int], wrecked_set, years, args) -> None:
    """Counts, and the wrecked players BY NAME with their volumes.

    Takes the already-computed set rather than recomputing `censored` over the same ids
    and threshold, so the report and the slice population cannot disagree.
    """
    wrecked = sorted(wrecked_set)
    zero = [pid for pid in wrecked if not any(run.outcomes[pid].volume_by_year.values())]
    at_20 = [pid for pid in ids if censored(run.outcomes[pid], years, 0.2)]
    print(
        f"    [{run.kind}] censored at {args.censor_threshold:.0%}: {len(wrecked)}"
        f" of {len(ids)} ({len(zero)} zero-volume, {len(wrecked) - len(zero)} played but"
        f" wrecked); at 20%: {len(at_20)}"
    )
    if not wrecked:
        return
    names = player_names(FIELDING_CACHE)
    shown = sorted(wrecked, key=lambda p: -run.outcomes[p].anchor_volume)[:CENSORED_LIST_LIMIT]
    for pid in shown:
        out = run.outcomes[pid]
        got = ", ".join(f"{y}={out.volume_by_year.get(y, 0.0):.0f}" for y in years)
        print(
            f"        {str(names.get(pid, pid))[:24]:24s} anchor {out.anchor_volume:6.0f}   {got}"
        )
    if len(wrecked) > CENSORED_LIST_LIMIT:
        print(f"        ... and {len(wrecked) - CENSORED_LIST_LIMIT} more")


def _report_top_of_board(ids, per_pool, shape_var, keeper_var, realized_var, args) -> None:
    """Pooled top-30 across BOTH pools, plus per-pool top-15 computed independently.

    The pooled 30 is the league's actual keeper count (3 x 10 teams). The per-pool
    tables are computed on each pool's own ranking rather than sliced out of the pooled
    one -- slicing would just report whichever pool dominated the pooled list.
    """
    _top_line("pooled", ids, shape_var, keeper_var, realized_var, KEEP_SLOTS * LEAGUE_TEAMS, args)
    if len(per_pool) > 1:
        for kind, pool_ids in sorted(per_pool.items()):
            usable = [pid for pid in pool_ids if pid in realized_var and pid in ids]
            if len(usable) >= MIN_REPORTABLE_SLICE:
                _top_line(kind, usable, shape_var, keeper_var, realized_var, 15, args)


def _top_line(label, ids, shape_var, keeper_var, realized_var, n, args) -> None:
    n = min(n, len(ids))
    s_pick, s_total = top_of_board({p: shape_var[p] for p in ids}, realized_var, n)
    k_pick, k_total = top_of_board({p: keeper_var[p] for p in ids}, realized_var, n)
    union = sorted(set(s_pick) | set(k_pick))
    # NO interval on the VAR total. The two top-n lists are DIFFERENT players ordered by
    # different forecasts, so `bootstrap_difference` -- which is paired, one index draw
    # indexing both samples -- has no common unit to pair on. It happily returns a
    # non-degenerate interval even when both estimators pick the identical 30 players in
    # a different order, i.e. when the true difference is exactly zero. The union MAE
    # below IS paired (same ids both sides) and is the honest interval for this slice.
    print(
        f"     top-{n} {label:7s} realized VAR: shape {s_total:7.1f}  keeper {k_total:7.1f}"
        f"  (share {len(set(s_pick) & set(k_pick))}/{n}; no paired interval exists"
        f" -- different player sets)"
    )
    s_err = [abs(shape_var[p] - realized_var[p]) for p in union]
    k_err = [abs(keeper_var[p] - realized_var[p]) for p in union]
    lo, hi, share = bootstrap_difference(s_err, k_err, draws=args.draws)
    print(
        f"        union-of-top MAE  shape {np.mean(s_err):5.2f}  keeper {np.mean(k_err):5.2f}"
        f"   shape-minus-keeper 95% [{lo:+.2f}, {hi:+.2f}]  shape better in {share:.0%}"
        f"  [paired on {len(union)} players]"
    )


def _report_low_support_sensitivity(
    ids, low_support, shape_var, keeper_var, realized_var, args
) -> None:
    """The headline KEEPS low-support shape rows; this is the excluded-rows variant.

    Dropping them from the headline would flatter shape by removing exactly the
    predictions it is least sure of, so they stay -- but the spec asks what the answer
    would be without them, and that is a different number, not a caveat.
    """
    kept = [pid for pid in ids if pid not in low_support]
    dropped = len(ids) - len(kept)
    if dropped == 0 or len(kept) < MIN_REPORTABLE_SLICE:
        print(f"        low-support rows: {dropped} (no sensitivity line)")
        return
    s_err = [abs(shape_var[p] - realized_var[p]) for p in kept]
    k_err = [abs(keeper_var[p] - realized_var[p]) for p in kept]
    lo, hi, share = bootstrap_difference(s_err, k_err, draws=args.draws)
    print(
        f"        low-support rows: {dropped}; EXCLUDING them, MAE shape"
        f" {np.mean(s_err):5.2f} keeper {np.mean(k_err):5.2f}"
        f"  95% [{lo:+.2f}, {hi:+.2f}]  shape better in {share:.0%}"
    )


def _report_breakout(ids, runs, base_year, shape_var, keeper_var, realized_var, args) -> None:
    """Slice 3: players whose base season was 25%+ above their prior one.

    This is where 2a's open question gets answered -- `persistence.S` regresses a
    breakout against ZiPS, shape regresses it against how comparable shapes actually
    played out. Estimator-neutral: both anchors are realized seasons.
    """
    anchors = pd.concat([run.anchors for run in runs], ignore_index=True)
    anchors = anchors[anchors["mlbam_id"].isin(ids)]
    if anchors.empty:
        return
    breakout = set(anchors.loc[breakout_mask(anchors), "mlbam_id"].astype(int))
    subset = [pid for pid in ids if pid in breakout]
    if len(subset) < MIN_REPORTABLE_SLICE:
        print(
            f"        breakout (now > 1.25x prior): {len(subset)} players, under"
            f" {MIN_REPORTABLE_SLICE}, not reported"
        )
        return
    s_err = [abs(shape_var[p] - realized_var[p]) for p in subset]
    k_err = [abs(keeper_var[p] - realized_var[p]) for p in subset]
    lo, hi, share = bootstrap_difference(s_err, k_err, draws=args.draws)
    print(
        f"        breakout (now > 1.25x prior), n={len(subset)}: MAE shape"
        f" {np.mean(s_err):5.2f} keeper {np.mean(k_err):5.2f}"
        f"  95% [{lo:+.2f}, {hi:+.2f}]  shape better in {share:.0%}"
    )


def _report_triples(
    base_year, horizon, view, ids, pool_of, shape_var, keeper_var, realized_var, drafts, args
) -> list[dict]:
    """Keeper-triple regret on the REAL roster: 23 players spanning both pools.

    `pool_of` maps each scoreable id to the pool it was priced in, which is what makes
    the cross-pool join possible; a team keeps three players, not three of each.
    """
    if base_year not in usable_draft_years(horizon, [int(y) for y in drafts]):
        print(f"     triples: no draft file for base {base_year}, slice skipped")
        return []
    scoreable = {pid: pool_of[pid] for pid in ids if pid in pool_of}
    resolution = resolve_draft(drafts[str(base_year)], people(FIELDING_CACHE), scoreable)
    rosters, dropped = eligible_rosters(resolution.by_team, set(scoreable))
    if not rosters:
        print(f"     triples: no roster kept {CANDIDATE_FLOOR}+ candidates")
        return []

    records = []
    for team, roster in sorted(rosters.items()):
        s_pick, s_r = triple_regret(roster, shape_var, realized_var)
        k_pick, k_r = triple_regret(roster, keeper_var, realized_var)
        records.append(
            {
                "base_year": base_year,
                "horizon": horizon,
                "view": view,
                "team": team,
                "shape_regret": s_r,
                "keeper_regret": k_r,
                "agree": set(s_pick) == set(k_pick),
            }
        )
    print(
        f"     triples: {len(records)} decisions"
        f"{f', {len(dropped)} rosters under the floor: {dropped}' if dropped else ''}"
    )
    print(
        f"        join: {len(resolution.unresolved)} unresolved,"
        f" {len(resolution.ambiguous)} ambiguous,"
        f" {len(resolution.unscoreable)} resolved-but-unscoreable"
    )
    for label, misses in (
        ("unresolved", resolution.unresolved),
        ("ambiguous", resolution.ambiguous),
    ):
        for team, name in misses[:JOIN_MISS_LIMIT]:
            print(f"           {label:11s} {team:22s} {name}")
        if len(misses) > JOIN_MISS_LIMIT:
            print(f"           {label:11s} ... and {len(misses) - JOIN_MISS_LIMIT} more")
    _print_regret_block(records, args, indent="        ")
    return records


def _print_regret_block(records, args, indent: str) -> None:
    """Regret, its bootstrap interval, the agreement rate, and the disagreeing subset.

    The agreement rate is not decoration. Most keeper decisions agree -- the best three
    on a 23-man roster are rarely close -- and an agreeing decision contributes exactly
    zero to the difference while still counting toward n. Without it, 18-of-20 agreement
    reports a tight interval around zero that reads as "cannot separate" when it means
    the slice had two informative rows.
    """
    s = [r["shape_regret"] for r in records]
    k = [r["keeper_regret"] for r in records]
    lo, hi, share = bootstrap_difference(s, k, draws=args.draws)
    agree = sum(r["agree"] for r in records) / len(records)
    print(
        f"{indent}regret  shape {np.mean(s):6.2f}  keeper {np.mean(k):6.2f}"
        f"   diff 95% [{lo:+.2f}, {hi:+.2f}]  shape better in {share:.0%}"
    )
    print(f"{indent}identical triples: {agree:.0%}")
    dis = [r for r in records if not r["agree"]]
    if not dis:
        print(f"{indent}every decision agreed -- the slice carries no information")
        return
    ds = [r["shape_regret"] for r in dis]
    dk = [r["keeper_regret"] for r in dis]
    dlo, dhi, dshare = bootstrap_difference(ds, dk, draws=args.draws)
    print(
        f"{indent}on the {len(dis)} that DISAGREE: shape {np.mean(ds):6.2f}"
        f"  keeper {np.mean(dk):6.2f}  95% [{dlo:+.2f}, {dhi:+.2f}]"
        f"  shape better in {dshare:.0%}"
    )


def report_pooled(records: list[dict], args: argparse.Namespace) -> None:
    """Triple decisions pooled ACROSS base years -- where the headline counts live.

    A per-base-year block can only ever show ten, since each draft year contributes ten
    teams. Decisions are roster-level (both pools), so there is no per-pool grouping.
    """
    if not records:
        return
    print(f"\n{'=' * 88}")
    print("POOLED KEEPER-TRIPLE DECISIONS (across base years)")
    print("=" * 88)
    for horizon in sorted({r["horizon"] for r in records}):
        for view in ("ALL", "INJURY-EXCLUDED"):
            subset = [r for r in records if r["horizon"] == horizon and r["view"] == view]
            if not subset:
                continue
            years = sorted({r["base_year"] for r in subset})
            print(f"\n+{horizon} {view}: {len(subset)} decisions from base {years}")
            _print_regret_block(subset, args, indent="     ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        choices=("hitter", "pitcher"),
        help="score ONE pool only. A keeper roster spans both, so the triple slice then "
        "answers 'best 3 within this pool' rather than the real decision.",
    )
    parser.add_argument(
        "--base-year",
        type=int,
        action="append",
        help="repeatable; defaults to 2022 2023 2024",
    )
    parser.add_argument(
        "--causal-check",
        action="store_true",
        help="fit the persistence share strictly causally; only informative for base 2023",
    )
    parser.add_argument("--censor-threshold", type=float, default=CENSOR_THRESHOLD)
    parser.add_argument("--draws", type=int, default=10_000, help="bootstrap resamples")
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    # `forecast_pool` reads these off the namespace; they must match keeper_forecast's
    # own defaults or this scores a different population than the live board.
    parser.add_argument("--min-pa", type=float, default=300)
    parser.add_argument("--min-ip", type=float, default=50)
    parser.add_argument("--min-next-pa", type=float, default=250)
    parser.add_argument("--min-next-ip", type=float, default=50)
    parser.add_argument("--no-aging", action="store_true")
    args = parser.parse_args()

    if not args.panel_dir.is_absolute():
        args.panel_dir = PROJECT_ROOT / args.panel_dir
    args.base_year = args.base_year or [2022, 2023, 2024]
    return run_historical(args)


if __name__ == "__main__":
    raise SystemExit(main())
