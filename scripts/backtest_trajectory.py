"""Out-of-sample bake-off between the trajectory matchers (#312, #313).

Answers the question the default rests on: does `shape` actually predict better than
level matching, and on which players? Every claim in `trajectory/__init__.py` and in
`shape.py`'s docstring comes from this script -- it exists so those numbers can be
re-measured rather than trusted, and so a regression in the estimator shows up as a
changed table instead of a stale docstring.

**The query player is removed from the panel entirely** before either estimator is
built, so neither can match him to himself. That is the whole point: an in-sample
comparison would flatter `shape`, which fits a model, over `comps`, which averages.

Slice, always. A random sample of the panel is dominated by fringe players -- 173 of
249 in the first run -- and the pooled number said 3% RMSE where the decision-relevant
slice said 18-20%. Pooled accuracy is not the thing being bought.

Usage:
    python scripts/backtest_trajectory.py                      # hitters, elite slices
    python scripts/backtest_trajectory.py --pool pitcher       # the #313 question
    python scripts/backtest_trajectory.py --sample 400         # random rather than elite
    python scripts/backtest_trajectory.py --horizon 2 --elite-floor 12

`--historical` runs a SECOND, different bake-off (#325): `shape` against the
`keeper_forecast` -> `keeper_value` chain, on the keeper decision rather than on
pool-wide rank. It exists to answer whether the chain being retired is actually worse.

    python scripts/backtest_trajectory.py --historical --pool hitter
    python scripts/backtest_trajectory.py --historical --base-year 2023 --causal-check

Three things about that mode are load-bearing and easy to undo by accident:

  * era factors are computed on the FULL panel and the truncation happens AFTER --
    `era_normalize` refuses a panel missing its 2023-2025 reference window, so the
    other order aborts base years 2022 and 2023 outright
  * targets are CUMULATIVE per horizon, so +1 and +2 are different questions
  * keeper-value keeps three advantages (out-year vintage leakage, it reads ZiPS at
    all, and a persistence fit that for base 2022 trains on LATER transitions). They
    are declared rather than removed, so a shape win is the strong form of the result

Both estimators are scored by `trajectory.panel.score`, which is what puts them on one
SCALE rather than merely in one unit.
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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from keeper_forecast import forecast_pool
from keeper_persistence import TRANSITIONS as KEEPER_TRANSITIONS
from keeper_persistence import load_rates

from fantasy_baseball.config import load_config
from fantasy_baseball.sgp.denominators import SgpOverrides, get_sgp_denominators
from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
from fantasy_baseball.trajectory.board import people, season_slots
from fantasy_baseball.trajectory.comps import (
    MIN_LOCAL_SUPPORT,
    collapse_split_seasons,
    comp_trajectory,
)
from fantasy_baseball.trajectory.era import era_factors, era_normalize
from fantasy_baseball.trajectory.panel import DEFAULT_PANEL_DIR, load_scored_panel
from fantasy_baseball.trajectory.panel import score as panel_score
from fantasy_baseball.trajectory.shape import build_history, shape_trajectory
from fantasy_baseball.trajectory.value import STARTER_SHARE, best_floor, resolve_slots
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD
from fantasy_baseball.utils.name_utils import normalize_name

#: Columns the role bucket needs, and the rule for a season split across two rows.
#: `collapse_split_seasons` keeps only `sgp` and `age`, so a traded pitcher's counting
#: columns have to be re-summed here or a mid-season trade reads as two half-roles --
#: the same reason `trajectory.board._SPLIT_RULES` re-sums `starts`/`games`.
_ROLE_SUMS = ("starts", "games", "sv")

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


def historical_panel(
    raw_panel: pd.DataFrame,
    kind: str,
    base_year: int,
    sgp_overrides: SgpOverrides | None,
) -> pd.DataFrame:
    """Era-normalize on the FULL panel, THEN truncate to `base_year`.

    Not the other order, and this is not a stylistic preference. `era_normalize` raises
    when any of `REFERENCE_SEASONS = (2023, 2024, 2025)` is missing -- deliberately, so
    a partial window cannot silently restate every season into units the output never
    mentions. A panel truncated to `season <= 2022` contains none of them, so computing
    factors after truncation aborts base years 2022 and 2023 outright.

    The factor table is therefore informed by seasons after `base_year`. That is a
    limitation, not an advantage to either estimator: a run environment is a league-wide
    fact and both sides are restated by the same one. It is also what the shipped
    harness already does -- it normalizes the full panel and filters queries afterwards.

    Called once per base year. `era_normalize` re-scores every one of ~18,000 seasons
    row-wise, so calling it per query would be hours of identical work; `without_player`
    is the cheap per-query half.
    """
    normalized = era_normalize(raw_panel, kind, sgp_overrides=sgp_overrides)
    return normalized[normalized["season"] <= base_year].copy()


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
    out: dict[int, Outcome] = {}
    for pid in panel["mlbam_id"].unique():
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

    Estimator-neutral: both anchors are REALIZED seasons, so this selects a population
    rather than favouring the side that happens to model breakouts.
    """
    return anchors["now"] > factor * anchors["prior"]


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


def var_for(
    sgp_by_id: pd.Series,
    kind: str,
    base_year: int,
    cache_dir: Path,
    levels: dict[str, float],
) -> pd.Series:
    """SGP above the position-aware floor, using year-`base_year` eligibility.

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
    eligibility = season_slots(cache_dir, base_year)
    floors = {}
    for pid in sgp_by_id.index:
        slots = resolve_slots(set(eligibility.get(int(pid), frozenset())), kind)
        floors[pid] = best_floor(slots, levels)[1]
    return sgp_by_id - pd.Series(floors, dtype=float).reindex(sgp_by_id.index)


def roles(panel: pd.DataFrame) -> pd.Series:
    """``(mlbam_id, season) -> "SP" / "closer" / "RP"``.

    #313 asks for the pitcher result split by role, because a closer's SGP is
    saves-dominated and saves are a job rather than a skill: a pooled pitcher number can
    average two opposite effects into a null.

    The cuts are BORROWED, not invented. `STARTER_SHARE` is the same `starts / games`
    split `trajectory.value` routes a pitcher's replacement floor on, and
    `CLOSER_SV_THRESHOLD` is the same save count the draft board buckets closers at. A
    third rule defined here would be one more thing to disagree with them.

    **Pass the RAW panel, not the era-normalized one.** `era_normalize` rescales
    `sv_ip` and `panel.score` then rebuilds `sv` from it, so a 20-save threshold on a
    normalized frame is a threshold on restated saves -- which is meaningless, because
    a closer is a JOB and 20 saves is a count of real ones. Measured on the live panel,
    that mistake moves 8 of 17,947 seasons across the bucket line. Refused below rather
    than documented, since the two frames are otherwise interchangeable to look at.
    """
    normalized = [c for c in panel.columns if c.startswith("era_factor_")]
    if normalized:
        raise ValueError(
            "roles() needs the RAW panel: this frame is era-normalized "
            f"(carries {normalized[:3]}...), so its `sv` has been restated into the "
            "reference run environment and a 20-save cut no longer means 20 saves. "
            "Pass the frame from load_scored_panel, before era_normalize."
        )
    missing = [c for c in _ROLE_SUMS if c not in panel.columns]
    if missing:
        raise KeyError(f"pitcher panel is missing role columns {missing}")
    agg = panel.groupby(["mlbam_id", "season"])[list(_ROLE_SUMS)].sum()
    games = agg["games"].to_numpy(dtype=float)
    starts = agg["starts"].to_numpy(dtype=float)
    saves = agg["sv"].to_numpy(dtype=float)
    # games == 0 cannot be a starter; guard the divide rather than letting it warn.
    share = np.divide(starts, games, out=np.zeros_like(starts), where=games > 0)
    bucket = np.where(
        share >= STARTER_SHARE, "SP", np.where(saves >= CLOSER_SV_THRESHOLD, "closer", "RP")
    )
    return pd.Series(bucket, index=agg.index, name="role")


def score(
    panel: pd.DataFrame,
    queries: pd.DataFrame,
    kind: str,
    horizon: int,
    role_by_season: pd.Series | None = None,
) -> pd.DataFrame:
    """Predict `horizon` years ahead for each query, with that player held out."""
    index = panel.set_index(["mlbam_id", "season"])["sgp"]
    rows = []
    for i, q in enumerate(queries.itertuples(index=False), start=1):
        if i % 100 == 0:
            print(f"  {i}/{len(queries)}...", flush=True)
        actual = float(index.get((q.mlbam_id, q.season + horizon), 0.0))
        # No self-matching: the player is gone from the panel both estimators see.
        clean = panel[panel["mlbam_id"] != q.mlbam_id]
        age, current, prior = int(q.age), float(q.current), float(q.prior)
        level = comp_trajectory(clean, kind=kind, age=age, sgp=current, horizons=(horizon,))
        curve, _ = shape_trajectory(
            clean, kind=kind, age=age, sgp=current, prior_sgp=prior, horizons=(horizon,)
        )
        if level.path[0].n == 0 or np.isnan(curve.path[0].mean):
            continue
        # `track` is `current` plus a HARD band on the prior season (#305) -- the same
        # two anchors shape uses, bounded instead of kernel-weighted. Passing prior_sgp
        # is what selects it; `comp_trajectory` defaults to level matching without it.
        #
        # Fitted AFTER the guard so a row that is about to be discarded does not pay for
        # a third full-panel scan. Its own emptiness is deliberately NOT part of that
        # guard: the two-mode comparison was already published from this harness, and
        # dropping rows track cannot score would silently change the current-vs-shape
        # population. Track records NaN there and is reported on its own defined subset.
        tracked = comp_trajectory(
            clean, kind=kind, age=age, sgp=current, prior_sgp=prior, horizons=(horizon,)
        )
        rows.append(
            {
                "mlbam_id": q.mlbam_id,
                "season": q.season,
                "age": age,
                "prior": prior,
                "now": current,
                "actual": actual,
                "current": level.path[0].mean,
                "shape": curve.path[0].mean,
                "track": (float("nan") if tracked.path[0].n == 0 else tracked.path[0].mean),
                # The role of the QUERY season -- the one both anchors describe.
                "role": (
                    role_by_season.get((q.mlbam_id, q.season), "")
                    if role_by_season is not None
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str) -> dict | None:
    if len(df) < 10:
        # Say so rather than printing nothing. A slice that silently vanishes reads as
        # "not applicable" when it means "too thin to measure" -- which for the role
        # splits in #313 is itself the finding.
        print(f"  {label:30s} n={len(df):4d}   (under 10, not reported)")
        return None
    out = {}
    for mode in ("current", "shape"):
        err = df[mode] - df["actual"]
        out[mode] = (float(np.sqrt((err**2).mean())), float(err.abs().mean()), float(err.mean()))
    wins = float(((df["shape"] - df["actual"]).abs() < (df["current"] - df["actual"]).abs()).mean())
    print(
        f"  {label:30s} n={len(df):4d}   "
        f"RMSE {out['current'][0]:5.2f} -> {out['shape'][0]:5.2f}   "
        f"MAE {out['current'][1]:5.2f} -> {out['shape'][1]:5.2f}   "
        f"bias {out['current'][2]:+5.2f} -> {out['shape'][2]:+5.2f}   "
        f"shape wins {wins:.0%}"
    )
    return {"slice": label, "n": len(df), "wins": wins}


def report_track(df: pd.DataFrame, label: str) -> None:
    """Three-way on the subset where `track` found any comps.

    Separate from `report` on purpose. `track`'s hard prior band leaves some queries
    with an empty cohort, and folding those drops into the shared row filter would move
    the current-vs-shape population that was already measured and published. So the
    three-way runs on track's own defined subset, and the coverage is printed rather
    than left for the reader to infer from a shrinking n.
    """
    defined = df.dropna(subset=["track"])
    coverage = f"{len(defined)}/{len(df)}"
    if len(defined) < 10:
        print(f"  {label:30s} track scored {coverage:>9}   (under 10, not reported)")
        return
    stats = {}
    for mode in ("current", "track", "shape"):
        err = defined[mode] - defined["actual"]
        stats[mode] = (float(np.sqrt((err**2).mean())), float(err.mean()))
    beats_track = float(
        (
            (defined["shape"] - defined["actual"]).abs()
            < (defined["track"] - defined["actual"]).abs()
        ).mean()
    )
    print(
        f"  {label:30s} track scored {coverage:>9}   "
        f"RMSE cur {stats['current'][0]:5.2f} / track {stats['track'][0]:5.2f} / "
        f"shape {stats['shape'][0]:5.2f}   "
        f"bias track {stats['track'][1]:+5.2f} shape {stats['shape'][1]:+5.2f}   "
        f"shape beats track {beats_track:.0%}"
    )


def _var(sgp_by_id: Mapping[int, float], kind: str, base_year: int, levels) -> dict[int, float]:
    series = pd.Series(dict(sgp_by_id), dtype=float)
    if series.empty:
        return {}
    return var_for(series, kind, base_year, FIELDING_CACHE, levels).to_dict()


def report_base_year(
    kind,
    base_year,
    horizons,
    common,
    shape_sgp,
    keeper_sgp,
    raw_panel,
    full,
    levels,
    drafts,
    args,
) -> list[dict]:
    """Every slice, in both views. Returns the triple decisions for pooled reporting.

    Wiring only -- each number comes from a helper that has its own tests.
    """
    anchor_vol = _anchor_volume(raw_panel, kind, base_year)
    outcomes = outcomes_for(full, kind, base_year, horizons, anchor_vol)
    collected: list[dict] = []

    for horizon in horizons:
        # CUMULATIVE, not the single year: +1 means "Y+1", +2 means "Y+1 and Y+2
        # summed". Reusing one summed target for every horizon made the rows identical,
        # which is what the first smoke run showed.
        years = [base_year + h for h in horizons if h <= horizon]
        realized_sgp = {pid: outcomes[pid].realized(years) for pid in common if pid in outcomes}
        shape_total = {
            pid: sum(v for h, v in shape_sgp[pid].items() if h <= horizon) for pid in common
        }
        keeper_total = {
            pid: sum(v for h, v in keeper_sgp[pid].items() if h <= horizon) for pid in common
        }
        realized_var = _var(realized_sgp, kind, base_year, levels)
        shape_var = _var(shape_total, kind, base_year, levels)
        keeper_var = _var(keeper_total, kind, base_year, levels)

        wrecked = {
            pid
            for pid in common
            if pid in outcomes and censored(outcomes[pid], years, args.censor_threshold)
        }
        if horizon == horizons[-1]:
            zero_vol = sum(
                1
                for pid in wrecked
                if pid in outcomes and not any(outcomes[pid].volume_by_year.values())
            )
            at_20 = {
                pid for pid in common if pid in outcomes and censored(outcomes[pid], years, 0.2)
            }
            print(
                f"  censored at {args.censor_threshold:.0%}: {len(wrecked)} of {len(common)}"
                f" ({zero_vol} zero-volume, {len(wrecked) - zero_vol} played but wrecked);"
                f" at 20%: {len(at_20)}"
            )

        label = "multi-year" if horizon > 1 else "one-year"
        print(f"\n  == target +{horizon} ({label}) ==")
        for view, pool_ids in (
            ("ALL", [pid for pid in common if pid in realized_var]),
            (
                "INJURY-EXCLUDED",
                [pid for pid in common if pid in realized_var and pid not in wrecked],
            ),
        ):
            print(f"  -- {view} ({len(pool_ids)} players) --")
            if len(pool_ids) < 10:
                print("     under 10, not reported")
                continue
            _report_top_of_board(pool_ids, shape_var, keeper_var, realized_var, args)
            collected += _report_triples(
                kind,
                base_year,
                horizon,
                view,
                pool_ids,
                shape_var,
                keeper_var,
                realized_var,
                drafts,
                args,
            )
    return collected


def _report_top_of_board(pool_ids, shape_var, keeper_var, realized_var, args) -> None:
    n = min(KEEP_SLOTS * 10, len(pool_ids))
    s_pick, s_total = top_of_board({p: shape_var[p] for p in pool_ids}, realized_var, n)
    k_pick, k_total = top_of_board({p: keeper_var[p] for p in pool_ids}, realized_var, n)
    overlap = len(set(s_pick) & set(k_pick))
    print(
        f"     top-{n} realized VAR: shape {s_total:7.1f}  keeper {k_total:7.1f}"
        f"  (they share {overlap}/{n})"
    )
    union = sorted(set(s_pick) | set(k_pick))
    s_err = [abs(shape_var[p] - realized_var[p]) for p in union]
    k_err = [abs(keeper_var[p] - realized_var[p]) for p in union]
    lo, hi, share = bootstrap_difference(s_err, k_err, draws=args.draws)
    print(
        f"     union-of-top MAE  shape {np.mean(s_err):5.2f}  keeper {np.mean(k_err):5.2f}"
        f"   diff 95% [{lo:+.2f}, {hi:+.2f}]  shape better in {share:.0%} of draws"
    )


def _report_triples(
    kind, base_year, horizon, view, pool_ids, shape_var, keeper_var, realized_var, drafts, args
) -> list[dict]:
    """Keeper-triple regret for one base year, horizon and view."""
    scoreable = set(pool_ids)
    if base_year not in usable_draft_years(horizon, [int(y) for y in drafts]):
        return []
    resolution = resolve_draft(
        drafts[str(base_year)], people(FIELDING_CACHE), dict.fromkeys(scoreable, kind)
    )
    rosters, dropped = eligible_rosters(resolution.by_team, scoreable)
    if not rosters:
        print(f"     triples: no roster kept {CANDIDATE_FLOOR}+ candidates")
        return []

    records = []
    for team, ids in sorted(rosters.items()):
        s_pick, s_r = triple_regret(ids, shape_var, realized_var)
        k_pick, k_r = triple_regret(ids, keeper_var, realized_var)
        records.append(
            {
                "pool": kind,
                "base_year": base_year,
                "horizon": horizon,
                "view": view,
                "team": team,
                "shape_regret": s_r,
                "keeper_regret": k_r,
                "agree": set(s_pick) == set(k_pick),
            }
        )
    misses = len(resolution.unresolved) + len(resolution.ambiguous)
    print(
        f"     triples: {len(records)} decisions"
        f"{f', {len(dropped)} rosters dropped: {dropped}' if dropped else ''}"
        f"  (join: {misses} unresolved/ambiguous,"
        f" {len(resolution.unscoreable)} unscoreable)"
    )
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
    """Triple decisions pooled ACROSS base years -- where the 20 one-year decisions live.

    A per-base-year block can only ever show 10, since each draft year contributes ten
    teams. The spec's headline counts (10 multi-year, 20 one-year) are the pooled ones,
    so they have to be printed somewhere or they never appear.
    """
    if not records:
        return
    print(f"\n{'=' * 88}")
    print("POOLED KEEPER-TRIPLE DECISIONS (across base years)")
    print("=" * 88)
    for pool in sorted({r["pool"] for r in records}):
        for horizon in sorted({r["horizon"] for r in records}):
            for view in ("ALL", "INJURY-EXCLUDED"):
                subset = [
                    r
                    for r in records
                    if r["pool"] == pool and r["horizon"] == horizon and r["view"] == view
                ]
                if not subset:
                    continue
                years = sorted({r["base_year"] for r in subset})
                print(
                    f"\n  {pool.upper()} +{horizon} {view}: "
                    f"{len(subset)} decisions from base {years}"
                )
                _print_regret_block(subset, args, indent="     ")


DRAFT_FILE = PROJECT_ROOT / "data" / "historical_drafts_resolved.json"
FIELDING_CACHE = PROJECT_ROOT / "data" / "cache" / "keeper_skills"


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
) -> tuple[dict[int, dict[int, float]], int]:
    """`{mlbam_id: {horizon: mean}}` with the query player held out each time.

    Returns the low-support count alongside: those rows are KEPT in the headline,
    because dropping them would flatter shape by removing the predictions it is least
    sure of. They are reported, and one sensitivity line excludes them.
    """
    lookup = anchors.set_index("mlbam_id")
    out: dict[int, dict[int, float]] = {}
    low_support = 0
    for i, pid in enumerate(ids, start=1):
        if i % 50 == 0:
            print(f"    shape {i}/{len(ids)}...", flush=True)
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
            low_support += 1
        out[pid] = means
    return out, low_support


def run_historical(args: argparse.Namespace) -> int:
    """The #325 head-to-head: shape against the keeper-value chain, out of sample."""
    overrides = load_config(PROJECT_ROOT / "config" / "league.yaml").sgp_overrides
    levels = position_aware_replacement_levels(get_sgp_denominators(overrides))
    drafts = json.loads(DRAFT_FILE.read_text(encoding="utf-8"))
    pools = [args.pool] if args.pool else ["hitter", "pitcher"]
    pooled: list[dict] = []

    for kind in pools:
        raw_panel = load_scored_panel(kind, panel_dir=args.panel_dir, sgp_overrides=overrides)
        factors = era_factors(raw_panel, kind)
        full = era_normalize(raw_panel, kind, sgp_overrides=overrides)

        for base_year in args.base_year:
            horizons = horizons_for(base_year)
            if not horizons:
                print(f"\n{kind.upper()} base {base_year}: no scoreable horizon, skipped")
                continue
            print(
                f"\n{'=' * 88}\n{kind.upper()}S -- base {base_year}, horizons {horizons}\n{'=' * 88}"
            )

            mode = "causal" if args.causal_check else "loto"
            transitions = transitions_for(base_year, mode)
            future = sum(1 for _, end in transitions if end > base_year + 1)
            print(
                f"  persistence fit: {mode}, {len(transitions)} transitions, {future} of them FUTURE"
            )
            if not transitions:
                print("  -- no transitions to fit on; base year not scoreable in this mode")
                continue

            observed = load_rates(base_year, kind, source="actual", factors=factors)
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
                    factors=factors,
                )
                fallbacks.append(fallback)
                for pid, value in keeper_value_sgp(frame, kind, overrides).items():
                    keeper_sgp.setdefault(int(pid), {})[h] = float(value)
            worst = max(fallbacks, key=lambda f: (f.whole_pool, f.share))
            print(
                f"  gap-model fallback: {worst.per_player}/{worst.total} players"
                f" ({worst.share:.0%}), whole-pool={worst.whole_pool}"
            )
            if worst.exceeds_headline_threshold:
                print("  ** EXCLUDED FROM THE HEADLINE: mostly gap model, not the curve **")

            truncated = historical_panel(raw_panel, kind, base_year, overrides)
            anchors = build_history(truncated)
            anchors = anchors[anchors["season"] == base_year]
            common = intersect(list(anchors["mlbam_id"]), list(keeper_sgp))
            print(
                f"  coverage: shape {len(anchors)}, keeper-value {len(keeper_sgp)},"
                f" intersection {len(common)}"
            )
            if not common:
                print("  -- empty intersection; nothing to compare")
                continue

            shape_sgp, low_support = _shape_forecasts(truncated, anchors, kind, common, horizons)
            common = intersect(common, list(shape_sgp))
            print(f"  scored by both: {len(common)}   low-support shape rows: {low_support}")

            pooled += report_base_year(
                kind,
                base_year,
                horizons,
                common,
                shape_sgp,
                keeper_sgp,
                raw_panel,
                full,
                levels,
                drafts,
                args,
            )
    report_pooled(pooled, args)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=("hitter", "pitcher"), default="hitter")
    parser.add_argument("--horizon", type=int, default=1, help="years ahead to predict")
    parser.add_argument(
        "--elite-floor",
        type=float,
        default=14.0,
        help="prior-season SGP at or above which a query counts as elite",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="score a RANDOM sample of this size instead of every elite season",
    )
    parser.add_argument("--min-age", type=int, default=24)
    parser.add_argument("--max-age", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--out", type=Path, help="write the scored queries to this CSV")
    # -- #325: shape against the keeper-value chain, out of sample ------------------
    parser.add_argument(
        "--historical",
        action="store_true",
        help="run the #325 head-to-head instead of the matcher bake-off",
    )
    parser.add_argument(
        "--base-year",
        type=int,
        action="append",
        help="base year for --historical (repeatable; defaults to 2022 2023 2024)",
    )
    parser.add_argument(
        "--causal-check",
        action="store_true",
        help="fit the persistence share strictly causally; only informative for base 2023",
    )
    parser.add_argument("--censor-threshold", type=float, default=CENSOR_THRESHOLD)
    parser.add_argument("--draws", type=int, default=10_000, help="bootstrap resamples")
    # `forecast_pool` reads these off the namespace; they must match keeper_forecast's
    # own defaults or the historical run scores a different population than the live one.
    parser.add_argument("--min-pa", type=float, default=300)
    parser.add_argument("--min-ip", type=float, default=50)
    parser.add_argument("--min-next-pa", type=float, default=250)
    parser.add_argument("--min-next-ip", type=float, default=50)
    parser.add_argument("--no-aging", action="store_true")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if args.base_year and not args.historical:
        parser.error("--base-year applies to --historical")

    if not args.panel_dir.is_absolute():
        args.panel_dir = PROJECT_ROOT / args.panel_dir

    if args.historical:
        args.base_year = args.base_year or [2022, 2023, 2024]
        return run_historical(args)

    overrides = load_config(PROJECT_ROOT / "config" / "league.yaml").sgp_overrides
    # Kept separately: the estimators want the era-normalized frame, but `roles` needs
    # the raw one -- see its docstring. Everything below reads `panel` except that one
    # call.
    raw_panel = load_scored_panel(args.pool, panel_dir=args.panel_dir, sgp_overrides=overrides)
    panel = era_normalize(raw_panel, args.pool, sgp_overrides=overrides)
    last = int(panel["season"].max())

    # `build_history` supplies both anchors and censors seasons whose prior predates
    # the panel -- the same rows a real query would have.
    pool = build_history(panel)
    pool = pool[
        pool["age"].between(args.min_age, args.max_age) & (pool["season"] + args.horizon <= last)
    ]
    if args.sample:
        queries = pool.sample(min(args.sample, len(pool)), random_state=args.seed)
        header = f"random sample of {len(queries)}"
    else:
        queries = pool[pool["prior"] >= args.elite_floor]
        header = f"every season with a prior >= {args.elite_floor:g} SGP"

    print(
        f"{args.pool.upper()}S, +{args.horizon}: {header}, ages "
        f"{args.min_age}-{args.max_age}, {len(queries)} queries\n"
    )
    role_by_season = roles(raw_panel) if args.pool == "pitcher" else None
    df = score(panel, queries, args.pool, args.horizon, role_by_season)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    print(f"\nscored {len(df)} (query player held out of the panel each time)")
    print(f"{'':32s}       current -> shape")
    report(df, "ALL")
    elite = df[df["prior"] >= args.elite_floor]
    report(elite, f"elite (prior >= {args.elite_floor:g})")
    report(elite[elite["now"] < elite["prior"] * 0.8], "elite down year (<80% of prior)")
    report(elite[elite["now"] < elite["prior"] * 0.7], "elite big drop (<70% of prior)")
    report(elite[elite["now"] >= elite["prior"] * 0.8], "elite holding steady")
    report(df[df["now"] > df["prior"] * 1.25], "breakout (up >25%)")

    # The two-mode table above races shape against LEVEL matching only. `track` uses the
    # same two anchors shape does, so it is the closer competitor -- and retiring it
    # (#325) without ever racing it would be retiring an unmeasured alternative.
    print("\n  -- three-way, including track (hard prior band) --")
    report_track(df, "ALL")
    report_track(elite, f"elite (prior >= {args.elite_floor:g})")
    report_track(elite[elite["now"] < elite["prior"] * 0.7], "elite big drop (<70% of prior)")
    report_track(elite[elite["now"] >= elite["prior"] * 0.8], "elite holding steady")

    if args.pool == "pitcher":
        # #313: a pooled pitcher number can average a starter effect and a closer effect
        # into a null, so the roles are reported separately rather than trusted to agree.
        print("\n  -- by role of the query season --")
        for role in ("SP", "RP", "closer"):
            report(df[df["role"] == role], f"{role}")
            report(df[(df["role"] == role) & (df["prior"] >= args.elite_floor)], f"{role} elite")
        # 15% of pitcher-seasons score below replacement against 7.7% for hitters, and
        # the linear form was never checked against a negative anchor.
        print("\n  -- negative anchors --")
        report(df[(df["now"] < 0) | (df["prior"] < 0)], "either anchor negative")
        report(df[df["now"] < 0], "current season negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
