# Stash Board v3 -- Retire Cost, Rank by P(helps)

Date: 2026-05-26
Status: approved (design), pre-implementation
Builds on: `docs/superpowers/specs/2026-05-26-stash-value-rate-redesign-design.md` (v2 gain metric)

## Problem

The shipped board (`src/fantasy_baseball/lineup/stash_value.py`) ranks candidates by
`stash_value = gain - cost`. The `cost` is the IL-slot allocation cost:

- Owned IL players: `cost = 0` always (they already hold a slot).
- Injured FAs: `cost = 0` if a slot is open; else the floored gain of the user's
  single weakest owned IL stash, charged identically to every FA.

Two problems with `cost`:

1. **It double-counts slot scarcity the cutline already expresses.** The board's job is
   "who deserves my N IL slots," and the cutline (top `IL`-capacity candidates) already
   answers that: if a FA ranks above the line and an owned stash ranks below it, that *is*
   the "grab the FA, drop the stash" recommendation. Subtracting the dropped stash's value
   again into the FA's `cost` charges for the slot twice.
2. **It is a uniform per-FA artifact.** Because it is always the weakest owned stash's
   gain, every injured FA shows the same `cost`. It tells you nothing about that specific
   player. This is the same class of artifact the v2 redesign already killed once (the old
   uniform `+0.12`-on-every-FA bug), reintroduced just floored non-negative.

"Found money" framing: you never give up *active production* to stash someone -- the
IL-for-IL rule guarantees the thing you drop is also a non-producing injured player. So a
candidate's worth is just his eventual marginal value (the v2 `gain`, already computed for
owned and FA alike against the healthy-lineup baseline). The only real scarcity is the slot
count, and the cutline is the right place to express it.

## Decision

Retire `cost`. Rank the board by **P(helps)** -- the probability the candidate's best swap
improves the user's roto total -- and show **Value** (expected gain) alongside it.

### Why P(helps) and not Value as the sort key

`P(helps) = Phi(mean / sd)` (`delta_roto.py:438-447`). It is monotonic in the z-score
`mean/sd`, not in `mean` alone. Because each candidate's swap has a different `sd`, sorting
by P(helps) is NOT the same order as sorting by Value:

| Candidate | Value (mean) | sd  | P(helps) |
|-----------|-------------:|----:|---------:|
| A         | 2.0          | 4.0 | 69%      |
| B         | 1.0          | 0.5 | 98%      |

Sort by Value -> A above B. Sort by P(helps) -> B above A. The two orderings agree on the
coarse split (every positive-Value candidate has p > 50%; every zero-Value candidate sits
at exactly 50%, the floor), so positives always rank above the dead weight either way. They
diverge only in the order *among the positive candidates* -- which is the set competing for
the IL slots at the cutline, so it can change who is above the line.

Accepted trade-offs of sorting by P(helps) (decided by the user):

- **It rides on the less-trustworthy number.** Per the v2 spec, the stash band's `sd` is an
  approximation (the synthetic swap line bundles the kept-incumbent share with the
  candidate), while `mean` (Value) is exact. P(helps) inherits that approximation.
- **A certain-but-trivial edge can rank #1.** When `sd ~= 0` and `mean > 0`, P(helps) pins
  to 100% (`delta_roto.py:441-442`), so a +0.1 lock can top a +5.0 likely-but-uncertain
  upgrade. P(helps) rewards certainty regardless of magnitude.

The user wants the risk-averse "most likely to actually help" ranking, which is a legitimate
objective for a stash board (betting on guys who pan out, not max expected value).

## What changes

### `StashScore` (data model)

Remove `gain` and `cost`. Final fields:

```
name, player_type, status, owned, stash_value, band, recommended_drop
```

- `stash_value` -- expected roto-point gain (band mean, floored at 0). The **Value** column.
- `band` -- `{mean, sd, p_positive, verdict}` (unchanged). **P(helps)** = `band["p_positive"]`.

`to_dict()` stays `asdict(self)`, so removing the fields removes them from the cache payload
automatically.

### Ranking

Sort descending by `(band["p_positive"], stash_value)` -- primary P(helps), deterministic
tie-break on Value. `cutline_rank = IL capacity` is unchanged; the top-N by P(helps) deserve
a slot.

### `recommended_drop`

Rename `_cost_and_drop` -> `_recommended_drop`; return `str | None` only (no cost). For a FA
when the IL is full: the owned stash at the **bottom of the same ranking** (lowest
`p_positive`, tie-broken by `stash_value`), so the drop suggestion is consistent with how the
board sorts. Return `None` for owned players and when an IL slot is open.

### UI (`web/templates/season/stash.html`)

Columns become: `#, Player, Status, Owned, Value, P(helps), Drop to add`. Remove the **Cost**
column and the now-duplicate **Gain** column (Value == old Gain once cost is gone). Keep the
above/below-cutline row styling. P(helps) keeps its existing `band.p_positive * 100` render;
Value renders `stash_value`.

### Docstrings

Update the module header (lines 16-18), the `StashScore` field comments, and
`score_stash_candidates`: no cost; ranked by P(helps); Value shown as the expected-gain read;
slot scarcity lives in the cutline + drop hint, not a per-row cost.

## Tests (`tests/test_lineup/test_stash_value.py`)

Requirement changed (user retired `cost` and switched the sort key), so cost/gain-based
assertions are rewritten to the new contract -- not loosened. Each touched test is called out
in the implementation plan with its reason.

- Rewrite assertions that read `cost`, `gain`, or `stash_value = gain - cost` to: no `cost`
  field present; `stash_value == band["mean"]` (>= 0); `recommended_drop` present for a
  full-IL FA and `None` otherwise.
- **Add** a test pinning the sort key to P(helps): a lower-Value / higher-P(helps) candidate
  ranks ABOVE a higher-Value / lower-P(helps) one (the A/B divergence above), so a future
  regression to value-sorting fails.
- Keep `test_owned_and_fa_player_get_equal_gain` (the equal-scale guard): the same player
  scored as owned IL vs FA still gets equal Value (and equal P(helps), since equal mean and
  equal sd give equal p).

## Blast radius

`stash_value.py`, `stash.html`, `tests/test_lineup/test_stash_value.py`. `refresh_pipeline`
and `season_routes` pass `to_dict()` through untouched. `stash_value.py` is in
`[tool.mypy].files`, so mypy must pass.

## Out of scope

- Any bench-displacement / deferred-roster-cost modeling (explicitly deferred: "retiring the
  cost is enough").
- Exact variance modeling of the synthetic swap line (still approximate, per v2).
- Changing the gain/swap metric itself -- only the cost, sort key, and presentation change.
