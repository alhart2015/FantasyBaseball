# Decompose `luck` into playing-time + batted-ball Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-30-decompose-luck-design.md`

**Goal:** Split the keeper `luck` family into a playing-time family and a batted-ball
component, run a holdout bake-off (baseline / keep-luck+pt / direct-batted-ball+pt),
and ship the per-pool winner.

**Architecture:** Generalize the composite from a fixed 4-family tuple to an explicit
per-pool ordered family set drawn from a `KNOWN_FAMILIES` universe. Add a `pt`
percentile family and a `batted_ball` overperformance family. The `--backtest` grid
evaluates all three candidates per pool and prints holdout rho, a fit-season noise
floor, best-fit weights, and watchlist rank moves; a human applies the margin/
tie-break/guard from that printed table and edits the shipped `FAMILIES`/
`FITTED_WEIGHTS` by hand, then `--fit` regenerates `projection.py`.

**Tech Stack:** Python, pandas, numpy, pytest. Branch `feat/277-decompose-luck`
stacked on `feat/273-league-keeper-board`.

## Global Constraints

- **ASCII-only** in all source, log, and format strings (CLAUDE.md). The one
  exception (`keeper_rankings.py` reconfigures stdout to UTF-8 for player names) is
  already in place; do not add new non-ASCII.
- **No `x or default` for numeric defaults.** `pt = 0` and `AVG - xBA = 0` are real
  values; use `is None` guards / `.get(k, default)` (CLAUDE.md).
- **Player IDs / index are `mlbam_id`** on every keeper frame; never reset it
  upstream of `_dedupe_two_way`.
- **Tests are the guardrail.** Do not loosen `test_projection.py` monotonicity/skew
  assertions; if a regenerated constant violates one, investigate the fit.
- **End-of-effort checks (repo root):** `pytest -q -n auto`, `ruff check .`,
  `ruff format --check .`, `mypy`, `vulture` (no new findings). Show outputs.
- **Selection is executor-applied, not shipped code.** The backtest prints numbers;
  a human edits `FAMILIES`/`FITTED_WEIGHTS`. No heuristic-selection code ships.

---

## Task 0: Build the keeper-skills cache (prerequisite)

**Files:** none (produces `data/cache/keeper_skills/` locally, gitignored).

The bake-off reads a per-season skills cache that is not committed. Build it for
2022-2026 (transitions need year and year+1; the live board needs 2026).

- [ ] **Step 1: Build the cache for each season**

Run:
```bash
for y in 2022 2023 2024 2025 2026; do python scripts/fetch_keeper_skills.py --year $y; done
```
Expected: for each year, `savant expected=... barrels=... pitch_mix=... bref
batting=... pitching=...` then two CSVs written. First run is ~1 min/year (Statcast).

- [ ] **Step 2: Verify the cache exists**

Run:
```bash
ls data/cache/keeper_skills/*_skills_20{22,23,24,25,26}.csv
```
Expected: ten files (hitter + pitcher for each of the five years).

- [ ] **Step 3: If the fetch is blocked, STOP and surface**

If Savant/BBRef return errors (Cloudflare/403/timeout) and the CSVs are not
produced, do NOT fabricate results. Report to the user that the bake-off cannot run
without the cache and wait. Do not proceed to Task 4+.

No commit (cache is gitignored).

---

## Task 1: Generalize the composite to a per-pool family set

**Files:**
- Modify: `src/fantasy_baseball/keepers/composite.py`
- Test: `tests/test_keepers/test_composite.py`

**Interfaces:**
- Produces: `FAMILIES: dict[str, tuple[str, ...]]` (per pool), `KNOWN_FAMILIES:
  frozenset[str]`, and `composite(families, kind, weights=None, *, family_order=None)`
  where `family_order` defaults to `FAMILIES[kind]`.

- [ ] **Step 1: Update the failing tests for the per-pool structure**

In `tests/test_keepers/test_composite.py`, the three tests that treat `FAMILIES` as a
flat tuple must index by pool. Replace them with:

```python
def test_a_missing_family_drops_out_of_the_denominator():
    supplied = {family: pd.Series([0.8]) for family in FAMILIES["hitter"]}
    full = composite(supplied, "hitter").iloc[0]
    partial = composite({k: v for k, v in supplied.items() if k != "future"}, "hitter").iloc[0]
    assert full == pytest.approx(0.8)
    assert partial == pytest.approx(0.8)


def test_luck_carries_a_positive_weight_for_every_position():
    for kind, weights in FITTED_WEIGHTS.items():
        assert weights[FAMILIES[kind].index("luck")] > 0, kind


def test_fitted_weights_have_one_entry_per_family_and_lead_with_skill():
    for kind, weights in FITTED_WEIGHTS.items():
        assert len(weights) == len(FAMILIES[kind]), kind
        assert weights[FAMILIES[kind].index("skill")] == max(weights), kind
        assert weights[FAMILIES[kind].index("future")] > 0, kind
```

Add two new tests for the generalization:

```python
def test_composite_honors_an_explicit_family_order():
    fams = {"skill": pd.Series([1.0, 0.0]), "pt": pd.Series([0.0, 1.0])}
    out = composite(fams, "hitter", weights=(1.0, 1.0), family_order=("skill", "pt"))
    assert out.tolist() == pytest.approx([0.5, 0.5])


def test_composite_rejects_a_family_outside_the_known_universe():
    with pytest.raises(KeyError, match="peripherals"):
        composite({"peripherals": pd.Series([0.5])}, "hitter", family_order=("skill",))
```

Note `test_composite_rejects_an_unknown_family` (existing) still passes with the new
`KNOWN_FAMILIES` message; leave it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_keepers/test_composite.py -q`
Expected: FAIL (FAMILIES is a tuple, not a dict; `family_order` kwarg unknown).

- [ ] **Step 3: Generalize `composite.py`**

Replace the `FITTED_WEIGHTS`/`FAMILIES` block:

```python
FITTED_WEIGHTS: dict[str, tuple[float, ...]] = {
    "hitter": (1.0, 0.8, 0.4, 0.3),
    "pitcher": (1.0, 0.6, 0.4, 0.15),
}
# The shipped family set per pool, aligned to FITTED_WEIGHTS. A dict, not one
# global tuple, because the pools already carry separate weights and fits and a
# split verdict from the bake-off (e.g. keep `luck` for hitters, `batted_ball` for
# pitchers) is a legitimate outcome. Until the bake-off ships a winner both pools
# hold the current four families, so behaviour is unchanged.
FAMILIES: dict[str, tuple[str, ...]] = {
    "hitter": ("skill", "luck", "future", "age"),
    "pitcher": ("skill", "luck", "future", "age"),
}
# Every family the model knows how to blend. `family_order` selects a subset per
# pool; a name outside this set is a typo, not a silent no-op.
KNOWN_FAMILIES: frozenset[str] = frozenset(
    {"skill", "luck", "pt", "batted_ball", "future", "age"}
)
```

Replace `composite()` with the `family_order`-driven version:

```python
def composite(
    families: dict[str, pd.Series],
    kind: str,
    weights: tuple[float, ...] | None = None,
    *,
    family_order: tuple[str, ...] | None = None,
) -> pd.Series:
    """Weighted blend of the pool's families, back on a 0-1 scale.

    `family_order` names the families to blend, in weight order; it defaults to the
    shipped `FAMILIES[kind]`. Missing families are treated as absent rather than as
    zero: their weight is dropped from the denominator, so a pool with no out-year
    projection still produces a comparable composite instead of one silently scaled
    down.
    """
    order = family_order if family_order is not None else FAMILIES[kind]
    chosen = weights if weights is not None else FITTED_WEIGHTS[kind]
    unknown = set(families) - KNOWN_FAMILIES
    if unknown:
        raise KeyError(f"unknown families {sorted(unknown)}; expected {sorted(KNOWN_FAMILIES)}")

    total = 0.0
    blended: pd.Series | None = None
    for name, weight in zip(order, chosen, strict=True):
        series = families.get(name)
        if series is None or weight == 0:
            continue
        term = weight * series.fillna(series.mean())
        blended = term if blended is None else blended + term
        total += weight
    if blended is None or total <= 0:
        raise ValueError("no weighted family supplied; cannot form a composite")
    return blended / total
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_keepers/test_composite.py -q`
Expected: PASS. (`test_composite_rejects_an_unknown_family` matches "peripherals" in
the new KeyError message.)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/keepers/composite.py tests/test_keepers/test_composite.py
git commit -m "refactor(keepers): per-pool family set + family_order in composite"
```

---

## Task 2: Add the `batted_ball` overperformance family

**Files:**
- Modify: `src/fantasy_baseball/keepers/composite.py`
- Test: `tests/test_keepers/test_composite.py`

**Interfaces:**
- Produces: `batted_ball(frame: pd.DataFrame, kind: str) -> pd.Series` returning
  rate overperformance (higher = luckier): `avg - xba` for hitters, `fip - era` for
  pitchers. The `pt` family needs no new function -- it is `percentile(frame["pt"])`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_keepers/test_composite.py`:

```python
from fantasy_baseball.keepers.composite import batted_ball


def test_batted_ball_is_avg_over_xba_for_hitters():
    frame = pd.DataFrame({"avg": [0.278, 0.240], "xba": [0.242, 0.250]})
    out = batted_ball(frame, "hitter")
    assert out.tolist() == pytest.approx([0.036, -0.010])


def test_batted_ball_is_fip_minus_era_for_pitchers():
    """ERA below FIP means the pitcher outran his peripherals -- luckier, higher."""
    frame = pd.DataFrame({"fip": [4.20, 3.50], "era": [3.10, 3.60]})
    out = batted_ball(frame, "pitcher")
    assert out.tolist() == pytest.approx([1.10, -0.10])


def test_batted_ball_keeps_nan_when_an_input_is_missing():
    frame = pd.DataFrame({"avg": [0.278, float("nan")], "xba": [0.242, 0.250]})
    out = batted_ball(frame, "hitter")
    assert out.iloc[0] == pytest.approx(0.036)
    assert math.isnan(out.iloc[1])
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_keepers/test_composite.py -k batted_ball -q`
Expected: FAIL with "cannot import name 'batted_ball'".

- [ ] **Step 3: Implement `batted_ball`**

Add to `src/fantasy_baseball/keepers/composite.py` (near `luck`):

```python
# Columns the batted-ball overperformance is measured from, per pool. Both sides
# of each difference are park-unadjusted here (only wrc_plus/era_minus are
# park-adjusted upstream), so the measure carries no park artifact.
BATTED_BALL_INPUTS: dict[str, tuple[str, str]] = {
    "hitter": ("avg", "xba"),
    "pitcher": ("fip", "era"),
}


def batted_ball(frame: pd.DataFrame, kind: str) -> pd.Series:
    """Rate overperformance the peripherals do not support (higher = luckier).

    `avg - xba` for hitters, `fip - era` for pitchers -- ERA below FIP means the
    pitcher outran his peripherals. This is the half of `luck` that regresses and
    should not be rewarded; parameterization B measures it directly instead of
    letting the `luck` catch-all proxy it. NaN in either input propagates.
    """
    produced, expected = BATTED_BALL_INPUTS[kind]
    missing = [c for c in (produced, expected) if c not in frame.columns]
    if missing:
        raise KeyError(f"{kind} batted_ball missing {missing}; got {sorted(frame.columns)}")
    return frame[produced] - frame[expected]
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_keepers/test_composite.py -k batted_ball -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/keepers/composite.py tests/test_keepers/test_composite.py
git commit -m "feat(keepers): batted_ball overperformance family"
```

---

## Task 3: Wire `pt`/`batted_ball` columns and per-pool family_order into the script

**Files:**
- Modify: `scripts/keeper_rankings.py` (`season_value`, `_qualified_families`,
  `_family_columns`, `composite_pct`, `build`, the `composite`/`batted_ball` import)

**Interfaces:**
- Consumes: `composite`, `batted_ball`, `FAMILIES` (per-pool), `percentile`.
- Produces: every qualified frame now carries `pt_pct` and `batted_ball_pct`;
  `season_value` frames carry `avg` (hitter) / `era` (pitcher); `build()` accepts
  optional `family_order`/`weights` overrides threaded to `composite_pct`.

- [ ] **Step 1: Add `batted_ball` to the composite import**

In `scripts/keeper_rankings.py`, add `batted_ball,` to the
`from fantasy_baseball.keepers.composite import (...)` block.

- [ ] **Step 2: Emit the raw rate stat from `season_value`**

In `season_value`, add the pool's rate stat to the returned frame under its real
name -- `avg` for hitters, `era` for pitchers -- so `batted_ball()` (Task 2) finds
the columns its `BATTED_BALL_INPUTS` names. Build `out` then attach the pool column:

```python
    out = pd.DataFrame(
        {
            "age": pd.to_numeric(frame["Age"], errors="coerce"),
            "pt": pt,
            "sgp": _sgp(lines, denoms),
        },
        index=frame.index,
    )
    out["avg" if kind == "hitter" else "era"] = lines["avg" if kind == "hitter" else "era"]
    return out
```

`_observed` joins this with the skills frame; neither `avg` nor `era` collides with a
skills column (hitter skills carry `xba`/`xwoba`, pitcher skills carry
`era_minus`/`fip`), so no `_sk` suffix appears.

- [ ] **Step 3: Build the new family columns in `_qualified_families`**

Replace `_qualified_families` body to add `pt_pct` and `batted_ball_pct`, reusing
`batted_ball()` (single source of truth for the sign, already unit-tested). It
returns `avg - xba` / `fip - era`, both already signed higher = luckier, so no
per-pool flip:

```python
def _qualified_families(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Apply the playing-time floor and build the same-season families.

    Computes ALL candidate families (`skill`, `luck`, `pt`, `batted_ball`, `age`);
    `future` is left to the caller. The active pool's `FAMILIES[kind]` selects which
    ones the composite actually blends, so the ranking and the backtest cannot drift
    into validating different feature definitions.
    """
    qualified = frame[frame["pt"] >= MIN_PT[kind]].copy()
    qualified["value_pct"] = percentile(qualified["sgp"])
    qualified["skill_pct"] = skill_percentile(qualified, kind)
    qualified["luck_pct"] = luck(qualified["value_pct"], qualified["skill_pct"])
    qualified["pt_pct"] = percentile(qualified["pt"])
    qualified["batted_ball_pct"] = percentile(batted_ball(qualified, kind))
    qualified["age_pct"] = percentile(qualified["age"], higher_is_better=False)
    return qualified
```

- [ ] **Step 4: Use the per-pool family_order in `_family_columns` and `composite_pct`**

```python
def composite_pct(
    frame: pd.DataFrame,
    kind: str,
    weights: tuple[float, ...] | None = None,
    *,
    family_order: tuple[str, ...] | None = None,
) -> pd.Series:
    """The composite, re-ranked to 0-1 -- the x-axis everything downstream uses."""
    order = family_order if family_order is not None else FAMILIES[kind]
    return percentile(composite(_family_columns(frame, order), kind, weights=weights, family_order=order))


def _family_columns(frame: pd.DataFrame, family_order: tuple[str, ...]) -> dict[str, pd.Series]:
    """The `{family: series}` mapping `composite` expects, for the active families."""
    return {family: frame[f"{family}_pct"] for family in family_order}
```

- [ ] **Step 5: Thread overrides through `build`**

Change `build`'s signature to accept overrides and pass them to `composite_pct`:

```python
def build(
    year: int,
    kind: str,
    denoms,
    keepers: dict[str, str],
    pricing: tuple[dict[str, list[str]], dict[str, float]] | None = None,
    *,
    family_order: tuple[str, ...] | None = None,
    weights: tuple[float, ...] | None = None,
) -> pd.DataFrame:
```

and change the composite line to:

```python
    qualified["composite"] = composite_pct(qualified, kind, weights=weights, family_order=family_order)
```

- [ ] **Step 6: No-behavior-change gate (value-identity on load-bearing columns)**

The shipped `FAMILIES` still holds the four current families, so `build` output must
be value-identical on the load-bearing columns (the added `pt_pct`/`batted_ball_pct`/
`avg`/`era` columns are expected). Capture the base-branch board, then compare. Use
the scratchpad dir `$SCRATCH` (the session scratchpad, not `/tmp`):

```bash
SCRATCH="C:/Users/HARTAL~1/AppData/Local/Temp/claude/C--Users-HartAlden-FantasyBaseball/d2b8a652-4156-4dec-96fb-f08689a6dbcf/scratchpad"
# 1) base capture (worktree keeps the current branch checked out)
git worktree add "$SCRATCH/base-273" feat/273-league-keeper-board
( cd "$SCRATCH/base-273" && python scripts/keeper_rankings.py --year 2026 >/dev/null )
cp "$SCRATCH/base-273/data/cache/keeper_skills/keeper_rankings_hitter_2026.csv" "$SCRATCH/base_hitter.csv"
cp "$SCRATCH/base-273/data/cache/keeper_skills/keeper_rankings_pitcher_2026.csv" "$SCRATCH/base_pitcher.csv"
git worktree remove "$SCRATCH/base-273" --force
# 2) new build on this branch
python scripts/keeper_rankings.py --year 2026 >/dev/null
```

Note the base worktree lacks the local skills cache and out-year projections; if the
worktree build cannot read them, instead capture the base board by `git switch
feat/273-league-keeper-board`, run the build, copy the two CSVs to `$SCRATCH`, then
`git switch feat/277-decompose-luck`. Compare the load-bearing columns:

```bash
python - <<'PY'
import pandas as pd, sys
SCRATCH = r"C:/Users/HARTAL~1/AppData/Local/Temp/claude/C--Users-HartAlden-FantasyBaseball/d2b8a652-4156-4dec-96fb-f08689a6dbcf/scratchpad"
cols = ["rank", "composite", "proj_sgp", "sd", "proj_var"]
for kind in ("hitter", "pitcher"):
    base = pd.read_csv(f"{SCRATCH}/base_{kind}.csv").set_index("mlbam_id")[cols].round(6)
    new = pd.read_csv(f"data/cache/keeper_skills/keeper_rankings_{kind}_2026.csv").set_index("mlbam_id")[cols].round(6)
    aligned = new.reindex(base.index)
    ok = base.equals(aligned)
    print(kind, "IDENTICAL" if ok else "DIFFERS")
    if not ok:
        diff = (base != aligned).any(axis=1)
        print(base[diff].join(aligned[diff], lsuffix="_base", rsuffix="_new").head())
        sys.exit(1)
PY
```
Expected: both pools print `IDENTICAL`. If either DIFFERS, the generalization is not
inert -- fix before Task 4.

- [ ] **Step 7: Run the existing suite**

Run: `pytest tests/test_keepers/ tests/test_scripts/test_keeper_rankings.py -q`
Expected: PASS (these do not exercise the new columns yet).

- [ ] **Step 8: Commit**

```bash
git add scripts/keeper_rankings.py
git commit -m "feat(keepers): pt/batted_ball columns + family_order plumbing (inert)"
```

---

## Task 4: Generalize `--backtest` into the bake-off

**Files:**
- Modify: `scripts/keeper_rankings.py` (`run_backtest`, add grid/candidate helpers,
  add `_watchlist_moves`)

**Interfaces:**
- Consumes: `_transition`, `composite_pct`, `build`, `_weighted_rho`.
- Produces: `--backtest` prints per pool a candidate table (baseline / A / B) with
  best-fit weights, fit rho, holdout rho, and the fit-season noise floor, plus a
  hitter watchlist table.

- [ ] **Step 1: Add grid + candidate constants and helpers**

Above `run_backtest`, add:

```python
# Weight grids, coarse on purpose -- two fit seasons cannot resolve a finer step.
# The `mid` grid (luck / batted_ball) spans below zero so a shrunk-to-zero or
# negative weight is observable; the shipped 0.4 floor would hide it.
_GRID_MID = (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
_GRID_PT = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
_GRID_FUTURE = (0.0, 0.2, 0.4, 0.6, 0.8)
_GRID_AGE = (0.0, 0.15, 0.3, 0.45)
_FAMILY_GRID = {
    "skill": (1.0,),  # pinned; every other family is measured against it
    "pt": _GRID_PT,
    "luck": _GRID_MID,
    "batted_ball": _GRID_MID,
    "future": _GRID_FUTURE,
    "age": _GRID_AGE,
}
# The three parameterizations the bake-off compares, per pool.
CANDIDATES: dict[str, tuple[str, ...]] = {
    "baseline": ("skill", "luck", "future", "age"),
    "A: pt+luck": ("skill", "pt", "luck", "future", "age"),
    "B: pt+batted_ball": ("skill", "pt", "batted_ball", "future", "age"),
}


def _best_weights(
    fit: list[pd.DataFrame], family_order: tuple[str, ...], kind: str
) -> tuple[tuple[float, ...], list[float]]:
    """Grid-search the family weights maximizing mean fit rho; skill pinned at 1.0.

    Returns the best weight tuple and the per-fit-season rho at those weights (used
    for the noise floor).
    """
    axes = [_FAMILY_GRID[f] for f in family_order]
    best_weights, best_mean = None, -2.0
    for weights in product(*axes):
        rhos = [_weighted_rho(f, weights, kind, family_order=family_order) for f in fit]
        mean = sum(rhos) / len(rhos)
        if mean > best_mean:
            best_weights, best_mean = weights, mean
    per_season = [_weighted_rho(f, best_weights, kind, family_order=family_order) for f in fit]
    return best_weights, per_season
```

Extend `_weighted_rho` to take a `family_order`:

```python
def _weighted_rho(
    frame: pd.DataFrame,
    weights: tuple[float, ...],
    kind: str,
    *,
    family_order: tuple[str, ...] | None = None,
) -> float:
    blended = composite_pct(frame, kind, weights=weights, family_order=family_order)
    return float(blended.corr(frame["target"], method="spearman"))
```

- [ ] **Step 2: Add the watchlist helper**

```python
# Hitters the bake-off must move the right way: the lucky everyday bats should fall,
# the genuinely skilled everyday bat (Alvarez) should not. Names are normalized on
# lookup, so accents do not matter here.
WATCHLIST = ("Ceddanne Rafaela", "Otto Lopez", "Yordan Alvarez")


def _watchlist_moves(
    year: int, denoms, best: dict[str, tuple[float, ...]]
) -> None:
    """Print each watchlist hitter's board rank under baseline vs each candidate."""
    pricing = pricing_table(denoms)
    boards = {
        label: build(
            year, "hitter", denoms, {}, pricing=pricing,
            family_order=CANDIDATES[label], weights=best[label],
        ).reset_index()
        for label in CANDIDATES
    }
    norm = {name: normalize_name(name) for name in WATCHLIST}
    print(f"\n  hitter watchlist ranks on the {year} board (lower = better):")
    print("    " + f"{'player':<18}" + "".join(f"{label:>20}" for label in CANDIDATES))
    for name in WATCHLIST:
        cells = []
        for label in CANDIDATES:
            b = boards[label]
            hit = b[b["name"].map(lambda n: normalize_name(str(n))) == norm[name]]
            cells.append(int(hit["rank"].iloc[0]) if len(hit) else -1)
        print(f"    {name:<18}" + "".join(f"{c:>20}" for c in cells))
```

- [ ] **Step 3: Rewrite `run_backtest` to the bake-off**

```python
def run_backtest(denoms) -> None:
    best_by_pool: dict[str, dict[str, tuple[float, ...]]] = {}
    for kind in POOLS:
        fit = [_transition(y, kind, denoms) for y in BACKTEST_FIT_YEARS]
        hold = _transition(BACKTEST_HOLDOUT, kind, denoms)
        print(f"\n{'=' * 70}\n{kind.upper()}  fit={list(BACKTEST_FIT_YEARS)} holdout={BACKTEST_HOLDOUT}")
        print(f"  {'candidate':<20}{'holdout':>9}{'fit':>9}{'noise':>9}  best weights")
        best_by_pool[kind] = {}
        for label, family_order in CANDIDATES.items():
            weights, per_season = _best_weights(fit, family_order, kind)
            best_by_pool[kind][label] = weights
            holdout = _weighted_rho(hold, weights, kind, family_order=family_order)
            fit_rho = sum(per_season) / len(per_season)
            noise = abs(per_season[0] - per_season[1])  # in-sample rho spread
            shown = " ".join(f"{f}={w:+.2f}" for f, w in zip(family_order, weights, strict=True))
            print(f"  {label:<20}{holdout:>9.4f}{fit_rho:>9.4f}{noise:>9.4f}  {shown}")
    # The watchlist is a hitter question; build 2026 boards under each candidate's
    # best hitter weights.
    _watchlist_moves(2026, denoms, best_by_pool["hitter"])
    print(
        "\n  Ship A or B over baseline only if it beats baseline holdout by MORE than"
        "\n  that pool's noise. Within noise, prefer the sniff-test passer, then A"
        "\n  (keeps SB/saves), then baseline. Apply by hand; see the spec."
    )
```

- [ ] **Step 4: Run the bake-off**

Run: `python scripts/keeper_rankings.py --backtest`
Expected: two per-pool tables (baseline / A / B) with holdout, fit, noise, and best
weights, then the hitter watchlist table. No crash. Save the full output to
`docs/superpowers/keeper-277-bakeoff-<date>.txt` for the verdict (Task 8).

- [ ] **Step 5: Commit**

```bash
git add scripts/keeper_rankings.py
git commit -m "feat(keepers): --backtest bake-off (baseline/pt+luck/pt+batted_ball)"
```

---

## Task 5: Add the mechanism rows to `--study`

**Files:**
- Modify: `scripts/keeper_rankings.py` (`run_study`)

- [ ] **Step 1: Add `pt` and `batted_ball` to the volume/rate table**

In `run_study`, extend the predictor loop so it reports whether each new family
predicts next-year PT vs next-year RATE:

```python
        for label, column in (
            ("last-yr value", "value_pct"),
            ("skills", "skill_pct"),
            ("luck", "luck_pct"),
            ("playing time", "pt_pct"),
            ("batted-ball", "batted_ball_pct"),
            ("age (younger)", "age_pct"),
            ("future (stale)", "future_pct"),
        ):
```

- [ ] **Step 2: Run `--study`**

Run: `python scripts/keeper_rankings.py --study`
Expected: the "what predicts next season" table now has `playing time` and
`batted-ball` rows. Confirm the mechanism: `playing time -> PT` should be strongly
positive; `batted-ball -> RATE` should be near zero or negative. Capture the output
into the bake-off notes.

- [ ] **Step 3: Commit**

```bash
git add scripts/keeper_rankings.py
git commit -m "feat(keepers): --study reports pt/batted_ball volume-vs-rate rows"
```

---

## Task 6: Apply the selection and set the shipped family set

**Files:**
- Modify: `src/fantasy_baseball/keepers/composite.py` (`FAMILIES`, `FITTED_WEIGHTS`)

This task is a decision, applied by hand from Task 4/5 output. No new code.

- [ ] **Step 1: Apply the margin/tie-break per pool**

For each pool, from the Task 4 table: a candidate (A or B) ships only if its holdout
rho exceeds baseline's by MORE than that pool's `noise`. Within the noise band,
prefer in order: (1) the candidate that passes the Task-4 watchlist sniff test,
(2) A over B (A keeps the SB/saves signal), (3) baseline. Record the chosen family
set and weights per pool.

- [ ] **Step 2: Apply the guard -- STOP if it trips**

On the hitter pool, check the watchlist table: the guard TRIPS if Rafaela or Otto
Lopez does not move to a strictly worse (higher) rank vs baseline, OR Alvarez moves
worse by more than 3 ranks. If it trips, do NOT edit the shipped config -- surface
the full bake-off table to the user and wait for a human decision.

- [ ] **Step 3: Write the winner into `composite.py`**

Set `FAMILIES[kind]` and `FITTED_WEIGHTS[kind]` per pool to the chosen family set and
best weights (round weights to the grid values). If baseline wins a pool, leave that
pool unchanged. Example (illustrative only -- use the ACTUAL winning weights):

```python
FITTED_WEIGHTS: dict[str, tuple[float, ...]] = {
    "hitter": (1.0, 0.6, 0.2, 0.4, 0.3),   # skill, pt, luck, future, age
    "pitcher": (1.0, 0.6, 0.4, 0.15),      # baseline unchanged, e.g.
}
FAMILIES: dict[str, tuple[str, ...]] = {
    "hitter": ("skill", "pt", "luck", "future", "age"),
    "pitcher": ("skill", "luck", "future", "age"),
}
```

- [ ] **Step 4: Re-run the backtest to confirm the shipped weights reproduce**

Run: `python scripts/keeper_rankings.py --backtest`
Expected: the chosen candidate's row still shows the weights you shipped. Sanity
only; no commit yet (docstrings/tests follow in Task 8).

---

## Task 7: Regenerate `projection.py` constants against the new composite

**Files:**
- Modify: `src/fantasy_baseball/keepers/projection.py` (`SGP_FIT`, `SGP_SD_FIT`,
  `STD_RESIDUAL_QUANTILES`)

- [ ] **Step 1: Run `--fit`**

Run: `python scripts/keeper_rankings.py --fit`
Expected: paste-ready `SGP_FIT = {...}`, `SGP_SD_FIT = {...}`,
`STD_RESIDUAL_QUANTILES = {...}` and per-pool `n`/`R2` lines.

- [ ] **Step 2: Paste the regenerated constants into `projection.py`**

Replace the three constant blocks with the printed values verbatim.

- [ ] **Step 3: Run the projection guardrails**

Run: `pytest tests/test_keepers/test_projection.py -q`
Expected: PASS. The monotonicity, hitter>pitcher, spread-dwarfs-gap, and
sorted-residual-quantile assertions must hold on the new constants. If any FAILS,
STOP and investigate the fit -- do not edit the test (CLAUDE.md).

- [ ] **Step 4: Commit (with Task 6's config)**

```bash
git add src/fantasy_baseball/keepers/composite.py src/fantasy_baseball/keepers/projection.py
git commit -m "feat(keepers): ship #277 luck decomposition winner + refit projection"
```

---

## Task 8: Update display, docstrings, downstream consumers, and the verdict

**Files:**
- Modify: `scripts/keeper_rankings.py` (`SHOWN`), `src/fantasy_baseball/keepers/composite.py`
  (module docstring), any downstream consumer found by grep
- Create: `docs/superpowers/keeper-277-verdict-<date>.md`

- [ ] **Step 1: Update `SHOWN` to the winning families**

Set `SHOWN` to display the shipped families, e.g. include `pt_pct` and whichever of
`luck_pct`/`batted_ball_pct` shipped (show both informationally if a pool kept
`luck` and the other took `batted_ball`):

```python
SHOWN = [
    "rank", "name", "age", "pt", "pos",
    "skill_pct", "pt_pct", "luck_pct", "batted_ball_pct", "future_pct",
    "composite", "proj_sgp", "sd", "proj_var", "keeper_of",
]
```
Drop any column not present on both pools' frames, or keep all (every `*_pct` column
is computed in `_qualified_families` regardless of the active family set, so all are
present). Verify the default run prints without KeyError:

Run: `python scripts/keeper_rankings.py --year 2026`
Expected: both pool tables print; CSVs written.

- [ ] **Step 2: Rewrite the `composite.py` module docstring**

Update the module docstring: the model now blends a per-pool family set; document
the `pt` family (playing time, real signal), the `batted_ball` family (rate
overperformance that regresses), and which parameterization shipped per pool and
why (cite the `--backtest` flag as the regenerable evidence, not numbers here). If B
shipped for a pool, document the SB/saves signal it drops and the speed-skill
follow-up (Task 9). Remove the now-false "four families" framing.

- [ ] **Step 3: Grep for downstream consumers of the output schema**

Run:
```bash
grep -rn "luck_pct\|FAMILIES\|FITTED_WEIGHTS\|keeper_rankings_" \
  src/ scripts/ tests/ docs/ --include=*.py --include=*.md | grep -v test_composite
```
For each hit outside this change: if it reads a column the winning set dropped,
update it. In particular check `--league`/`--roster` display code in
`keeper_rankings.py` for hard references to `luck_pct`. A clean grep (or all hits
handled) satisfies R11.

- [ ] **Step 4: Update the remaining tests deliberately**

Any assertion pinning the old 4-family shape (e.g. in `test_composite.py`,
`test_scripts/`) that the shipped change invalidates: update it to the new shape with
a one-line justification comment (the requirement changed; the assertion did not
silently loosen). Add a test that `FAMILIES[kind]` and `FITTED_WEIGHTS[kind]` have
equal length for both pools (already covered by
`test_fitted_weights_have_one_entry_per_family_and_lead_with_skill` -- confirm it
still passes with the new lengths).

- [ ] **Step 5: Write the verdict**

Create `docs/superpowers/keeper-277-verdict-<date>.md`: which family set shipped per
pool, the fitted weights, the holdout rho table (baseline/A/B), the mechanism-row
readouts, the watchlist moves, and the exact command that regenerates it
(`python scripts/keeper_rankings.py --backtest`). State plainly whether the shipped
weights changed and, if a pool stayed on baseline, why (#277 acceptance / R12).

- [ ] **Step 6: Full verification**

Run and show output:
```bash
pytest -q -n auto
ruff check .
ruff format --check .
mypy
vulture
```
Expected: suite green; ruff clean; format clean; mypy clean for touched files under
`[tool.mypy].files`; no NEW vulture findings.

- [ ] **Step 7: Live board sanity**

Run: `python scripts/keeper_rankings.py --league` (needs live KV; if unavailable, run
`--year 2026` default build) and confirm the watchlist demotion holds on the real
board and nothing regressed.

- [ ] **Step 8: Commit**

```bash
git add scripts/keeper_rankings.py src/fantasy_baseball/keepers/composite.py \
  tests/ docs/superpowers/keeper-277-verdict-*.md
git commit -m "docs(keepers): #277 SHOWN/docstring/verdict + downstream cleanup"
```

---

## Task 9: Document the SB/saves tradeoff (only if B won a pool)

**Files:**
- Modify: `src/fantasy_baseball/keepers/composite.py` (module docstring),
  `docs/superpowers/keeper-277-verdict-<date>.md`

- [ ] **Step 1: Record the tradeoff**

If parameterization B shipped for either pool, document that `batted_ball` excludes
the SB (hitters) / saves (pitchers) value `luck` used to carry, so a pure speedster
or a low-skill closer is no longer credited for it, and name the follow-up: add an
SB-rate (hitters) / save-role (pitchers) term to the skill family. Note it as a
candidate future issue, not work done here.

- [ ] **Step 2: Commit**

```bash
git add src/fantasy_baseball/keepers/composite.py docs/superpowers/keeper-277-verdict-*.md
git commit -m "docs(keepers): note SB/saves signal dropped by batted_ball parameterization"
```

---

## Self-review notes

- **Spec coverage:** R1 pt family (Task 3 Step 3); R2 batted_ball family (Task 2 +
  Task 3 Step 3); R3 season_value rate (Task 3 Step 2); R4 family machinery (Task 1);
  R5 grid bake-off (Task 4); R6 mechanism/shrinkage/sniff readouts (Task 4 watchlist
  + Task 5 rows); R7 winner + guard (Task 6); R8 projection refit (Task 7); R9
  SHOWN/docstring/grep (Task 8); R10 SB/saves doc (Task 9); R11 downstream grep
  (Task 8 Step 3); R12 verdict (Task 8 Step 5).
- **Shrinkage readout:** delivered by the Task 4 table -- baseline's best `luck`
  weight vs A's best `luck` weight are both printed, so the shrinkage is read
  directly off the two rows.
- **Grid size:** A/B grids are 7*9*5*4 = 1260 weight tuples per pool, each scored on
  2 fit seasons; expect the bake-off to take up to ~1-2 min. Acceptable.
- **Rank-equivalence edge case (spec):** needs no task. The spec noted that A ships
  as raw `(skill, pt, luck)` WITHOUT a residualization step, precisely so there is no
  second spelling to keep in sync -- the fitted `luck` weight is the shrinkage
  readout. Since no residualization is implemented, there is nothing to verify
  equivalent.
- **Edge-case coverage:** NaN handling (Task 2 tests + `percentile`/`composite`
  contract preserved); numeric-default trap (Global Constraints -- `pt_pct`/
  `batted_ball_pct` use `percentile`, never `x or default`); park basis (Task 2
  docstring + `BATTED_BALL_INPUTS` comment); baseline reproduction (Task 3 Step 6);
  pitcher ERA basis (Task 3 Step 2 -- `era` and `fip` are both BBRef-derived);
  mechanical `sgp_sd` refit, not #278 (Task 7 regenerates only).
- **Data-dependent weights:** the exact shipped `FAMILIES`/`FITTED_WEIGHTS` (Task 6)
  and `projection.py` constants (Task 7) are outputs of the bake-off, not knowable in
  advance; the plan gives the exact command whose output supplies them. These are
  not placeholders.
