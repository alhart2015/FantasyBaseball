# Draft Engine Unification -- Design

Date: 2026-06-09
Status: Approved (pending spec review)

## Problem

The draft pipeline has two parallel recommendation engines that nothing reconciles:

- **VAR/VONA engine** -- `draft/recommender.py` (`get_recommendations`) plus the
  `pick_*` functions in `draft/strategy.py` (the `STRATEGIES` registry,
  `strategy.py:887`). Driven by `config.scoring_mode` (`var`/`vona`) and
  `config.strategy`. Consumed by `scripts/simulate_draft.py`,
  `scripts/compare_strategies.py`, the static board (`draft/board.py`,
  `/api/board`), and the legacy `scripts/run_draft.py`.
- **deltaRoto engine** -- `draft/eroto_recs.py` (`rank_candidates`) plus
  `draft/recs_integration.py` glue. Consumed by the live dashboard `/api/recs`
  and `scripts/sim_deltaroto.py`. Ignores `config.scoring_mode` entirely.

Each engine emits its own row type:

- `Recommendation` (`recommender.py:33`): `name, var, score, best_position,
  positions, player_type, need_flag, note`.
- `RecRow` (`eroto_recs.py:88`): `player_id, name, positions, immediate_delta,
  value_of_picking_now, per_category`.

The strategies weld the VAR/VONA ranker *inside* each `pick_*` (the
`get_recommendations` call in `_get_recs`, `strategy.py:468`), so a strategy
cannot run on top of the deltaRoto ranker. The only already-shared seam is
`select_from_ranked` (`strategy.py:83`), which slot-gates a ranked list by
duck-typing on `.positions`.

Consequences:

1. The config knob (`scoring_mode`, `strategy`) drives the VAR/VONA engine,
   which the PR #127 bake-off verdict found is the *losing* path
   (deltaRoto-immediate wins; var ~= vona ~= closers are a mid-pack wash --
   see `project_strategy_verdict_2026_06_07`).
2. The live dashboard and the simulators can silently diverge on scoring logic.
3. "Match the verdict" is not expressible in `config/league.yaml` today, because
   `deltaroto_immediate` is not a valid `scoring_mode`
   (`VALID_SCORING_MODES = {"var", "vona"}`, `config.py:38`).

## Goal

One shared recommendation seam that the dashboard, the simulator, and the
strategy comparison all route through, with `scoring_mode` extended to include
the deltaRoto variants, and `strategy` decoupled into an orthogonal overlay so
any strategy composes with any scoring mode. Set the league config to the
verdict winner. Remove the duplicated engine and simulator code so the paths
cannot conflict.

Non-goals (YAGNI): no new scoring math (the deltaRoto math in
`lineup.delta_roto` is unchanged); no change to the projection/SGP/replacement
pipeline; no UI redesign; no multi-user/productionization work.

## Design

### 1. Uniform row -- `RankedPick`

A single dataclass that both rankers adapt into. Lives in the new
`draft/recommend.py` (or `draft/ranked_pick.py` if that reads cleaner during
implementation).

```python
@dataclass
class RankedPick:
    player_id: str
    name: str
    positions: list[Position]
    player_type: PlayerType
    score: float                       # active mode's primary metric
    per_category: dict[str, float] = field(default_factory=dict)
    note: str = ""
    need_flag: bool = False
```

`score` carries the active mode's primary metric: `var`, `vona`,
`immediate_delta`, or `value_of_picking_now`. Two adapters:
`RankedPick.from_recommendation(rec)` and `RankedPick.from_recrow(row, *,
metric)` where `metric` selects `immediate_delta` vs `value_of_picking_now`.

The single overlay read of `rec.var` (`strategy.py:833`) becomes `rec.score`.

### 2. The seam -- `recommend()`

One entry point in `draft/recommend.py` that every caller uses:

```python
def recommend(
    *,
    state,
    candidates,
    scoring_mode: str,
    strategy: str,
    config,
    picks_until_next: int | None,
    **kwargs,
) -> list[RankedPick]:
    ...
```

Responsibilities, in order:

1. **Rank** by dispatching on `scoring_mode`:
   - `var` / `vona` -> `recommender.get_recommendations(scoring_mode=...)`,
     adapted via `RankedPick.from_recommendation`.
   - `deltaroto_immediate` / `deltaroto_vopn` -> `eroto_recs.rank_candidates`,
     sorted by the matching metric, adapted via `RankedPick.from_recrow`.
2. **Apply the strategy overlay** (section 3).
3. **Slot-gate** via the existing `select_from_ranked`.

The deltaRoto branch reuses the input-assembly already in
`recs_integration.compute_rec_inputs` rather than duplicating it.

### 3. Strategies as orthogonal overlays

Hoist the ranker out of `pick_*`. `STRATEGIES` becomes a registry of overlay
functions with a uniform signature:

```python
overlay(
    ranked: list[RankedPick],
    roster_state,
    config,
    **kwargs,
) -> RankedPick | None
```

An overlay applies its constraints (closer timing, AVG floors, no-punt
category protection) to an already-ranked `list[RankedPick]` and returns the
chosen pick, or `None` to defer to plain slot-gated selection. `default` is the
identity overlay = plain greedy = the verdict winner. `select_from_ranked`
stays the shared slot-gating primitive.

This is the most invasive piece: every `pick_*` that currently reads pandas
board columns or leverage weights (`pick_no_punt_opp`, `pick_avg_hedge`,
`pick_avg_anchor`, the closer family) is reworked to read `RankedPick` fields
and `per_category` instead of the board frame.

**Risk and fallback:** if porting a leverage-weighted or AVG-floor strategy off
the pandas board proves materially harder than expected, the fallback is to
leave that specific strategy operating in "overlay-where-cheap" mode (it still
consumes `RankedPick`, but only the strategies that compose cleanly are ported
in this pass) without abandoning the seam. This is flagged at the moment of
friction, not pre-emptively.

### 4. One simulator

Merge `scripts/simulate_draft.py` (1033 LOC) and `scripts/sim_deltaroto.py`
(325 LOC) into a single `--scoring-mode`-driven simulator that calls
`recommend()` for every pick (user and field). Decompose the monolith into
three units:

- **harness** -- the draft loop, pick order, snake advance.
- **field model** -- how opponents pick (strategy assignment, pick_rank
  variance, ADP noise).
- **reporting** -- standings, roto points, keeper/strategy summaries.

`scripts/compare_strategies.py` iterates the `scoring_mode x strategy` grid
against the one simulator. `scripts/sim_deltaroto.py` is deleted once its two
adapters (`deltaroto_immediate`, `deltaroto_vopn`) exist as scoring modes and
its selection gate (already `select_from_ranked`) is covered.
`scripts/replay_picks.py` is repointed at the seam.

### 5. Config and dashboard

- `config.py:38`: `VALID_SCORING_MODES = {"var", "vona", "deltaroto_immediate",
  "deltaroto_vopn"}`.
- `web/app.py` `/api/recs`: call `recommend()` instead of reaching into
  `recs_integration` directly. **Keep `immediate_delta` as a JSON field alias**
  (populated from `RankedPick.score` when the mode is a deltaRoto mode) so
  `web/static/draft.js` is untouched in this change. Decision: alias over
  rename, to keep the frontend out of the blast radius.
- `config/league.yaml`: `scoring_mode: deltaroto_immediate`, `strategy:
  default`. This is the **final** commit -- the YAML cannot legally name
  `deltaroto_immediate` until the seam and config validation support it, so the
  "update YAML to match the verdict" request lands at the tail of this work,
  not as a standalone edit.

### 6. Parity guard (load-bearing)

Per the standing meta-lesson that small sim bugs repeatedly flipped the strategy
verdict (`project_draft_sim_scoring_bugs_2026_06_07`,
`project_strategy_verdict_2026_06_07`), capture a golden-master BEFORE any
change:

- current `/api/recs` output for a fixed draft state, and
- a fixed-seed simulator result run WITH `team_sds` (variance-aware scoring).

Assert equivalence after each phase. The deltaRoto path through `recommend()`
must reproduce the pre-refactor `/api/recs` rows exactly; the var/vona path
through `recommend()` must reproduce the pre-refactor simulator picks exactly.
No silent scoring regressions.

## Phasing

Per `CLAUDE.md` (phased execution, <=5 files per phase, approval between
phases). Each phase ends green on the relevant test subset.

- **P0 -- Step-0 cleanup.** Dead-code sweep (`ruff check --select F,I`,
  `vulture`) on the modules about to be touched. Own commit, no behavior change.
- **P1 -- `RankedPick` + adapters.** New row type and the two adapters; unit
  tests for each adapter. No call site switched yet. No behavior change.
- **P2 -- `recommend()` seam.** Build the dispatch + overlay + slot-gate seam;
  wire `/api/recs` to it behind the `immediate_delta` alias. Golden-master on
  `/api/recs` must match. Files: `draft/recommend.py`, `recs_integration.py`,
  `web/app.py`, tests.
- **P3 -- strategies as overlays.** Convert `pick_*` to overlay signature;
  `STRATEGIES` becomes the overlay registry; the `rec.var` read becomes
  `rec.score`. Files: `strategy.py` (+ helpers), tests.
- **P4 -- consolidate simulators.** Merge into one `--scoring-mode` sim;
  decompose the monolith; rewire `compare_strategies.py` and `replay_picks.py`;
  delete `sim_deltaroto.py`. Files: `scripts/simulate_draft.py`,
  `scripts/sim_deltaroto.py`, `scripts/compare_strategies.py`,
  `scripts/replay_picks.py`, tests.
- **P5 -- config + docs.** `config.py` validation; `config/league.yaml` flip to
  `deltaroto_immediate` + `default`; refresh `draft/CLAUDE.md` to describe the
  single seam. Files: `config.py`, `config/league.yaml`,
  `src/fantasy_baseball/draft/CLAUDE.md`.

## Testing

- **Adapter unit tests** -- `Recommendation -> RankedPick`, `RecRow ->
  RankedPick` (both metrics), including `per_category` and `note`/`need_flag`
  passthrough.
- **Overlay unit tests** -- each strategy overlay runs against a synthetic
  `list[RankedPick]`, decoupled from the ranker (this is a coverage win: today
  the strategies can only be tested through the full board pipeline).
- **Seam dispatch tests** -- each `scoring_mode` routes to the right ranker and
  produces `RankedPick` rows.
- **Golden-master parity tests** (section 6) -- gate every phase.
- **Config validation tests** -- the four valid modes accepted; an invalid mode
  rejected with the listing message.
- Full `pytest`, `ruff check`, `ruff format --check`, `vulture`, and `mypy`
  (for any file under `[tool.mypy].files`) green per the `CLAUDE.md` end-of-effort
  checklist.

## Open questions

None blocking. The P3 strategy-port risk has an explicit in-flight fallback
(section 3); it does not need to be resolved before starting.

## References

- Verdict and meta-lesson: `project_strategy_verdict_2026_06_07`,
  `project_draft_sim_scoring_bugs_2026_06_07` (memory).
- Draft subsystem rules: `src/fantasy_baseball/draft/CLAUDE.md`.
- Key code: `draft/recommender.py:33` (`Recommendation`), `draft/eroto_recs.py:88`
  (`RecRow`), `draft/strategy.py:83` (`select_from_ranked`), `:468`
  (`get_recommendations` inside `_get_recs`), `:887` (`STRATEGIES`),
  `config.py:38` (`VALID_SCORING_MODES`).
