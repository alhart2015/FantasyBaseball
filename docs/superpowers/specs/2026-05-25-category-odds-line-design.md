# Category Odds line: per-category finish probabilities

Date: 2026-05-25
Status: Approved (design); pending implementation plan
Builds on: the Category Bars tab (PR #99) + its refinements (PR #100)

## Problem

The Category Bars tab shows, for a selected roto category, every team's
projected total with a +/-1 SD band. It shows the spread but does not
quantify the manager's odds: "given these projections and their
uncertainty, how likely am I to win this category, and how many teams am I
clearly ahead of?" The bands imply this but the eye can't read a probability
off overlapping error bars.

## Goal

Add a single line of three numbers about the USER's team for the selected
(category, projection), placed between the category selector and the chart:

- `1st: N%` -- probability the user's team finishes 1st in this category.
- `top 3: N%` -- probability the user's team finishes top-3 in this category.
- `wins: x/9` -- count of opponents whose band the user's band clearly
  clears (non-overlapping in the user's favor), out of the opponent count.

All three are per-category (this one category, e.g. most Runs / lowest ERA),
not overall-standings quantities.

## Non-goals

- No Monte Carlo. Probabilities are analytic, consistent with the
  analytic bands the chart already draws (decision recorded below).
- No change to the chart itself, the bound lines, or the category/projection
  toggles. This adds a stats line and the math behind it.
- Not the same as the Monte Carlo tab's `category_risk` "top3" (see
  Consistency note).

## Decision: analytic, not Monte Carlo

The chart's dots and error bars come from each team's analytic mean
(`projected_standings` / `preseason_standings`) and SD (`team_sds` /
`preseason_team_sds`). The odds are computed from the SAME means and SDs, so
they agree with what's on screen, are deterministic, and are instant. Monte
Carlo would add sampling noise and diverge from the bands; the dashboard has
deliberately moved toward analytic bands (the deltaRoto band redesign).

## The math (new function in `scoring.py`)

For the selected category, let team `i` have projected mean `mu_i` and SD
`sd_i`. The user is team `u`; the other 9 are opponents. Unify higher-better
and inverse (ERA/WHIP, lower-better) by working in a "bigger is better"
score: for inverse categories negate the means (SD is unchanged under
negation). After this flip every formula below is "higher wins."

Opponents are treated as independent of each other and of the user given the
user's draw -- valid because each fantasy team's total is built from a
disjoint set of players. This is the standard "probability of being the
maximum" model and a strict refinement of the pairwise model `score_roto`
already uses.

Let `Phi` be the standard-normal CDF (`0.5 * (1 + erf(z / sqrt(2)))`, the
repo's existing convention; see `scoring._prob_beats` and
`delta_roto._normal_cdf`).

**Conditioning on the user's value `x`:**

- An opponent `j` is beaten by the user when its draw is below `x`:
  `P(opp_j < x) = Phi((x - mu_j) / sd_j)`.
- So the probability opponent `j` BEATS the user at `x` is
  `q_j(x) = 1 - Phi((x - mu_j) / sd_j)`.

**1st place** = no opponent beats the user:

```
P(first) = E_x[ product_j Phi((x - mu_j) / sd_j) ]      x ~ N(mu_u, sd_u)
```

**Top 3** = at most 2 opponents beat the user. The number of opponents that
beat the user at `x` is a sum of independent Bernoulli(`q_j(x)`) -- a
Poisson-binomial. Compute `P(k <= 2 | x)` with a small DP over the 9
opponents (robust when some `q_j` is 0 or 1), then integrate:

```
P(top3) = E_x[ PoissonBinomialCDF(k <= 2; {q_j(x)}) ]
```

Note `P(first | x)` is exactly the `k = 0` term of the same Poisson-binomial,
so both integrals share one quadrature pass.

**Integration:** Gauss-Hermite over the user's normal. Nodes/weights from
`numpy.polynomial.hermite_e.hermegauss(N)` (probabilists', weight
`exp(-x^2/2)`); `E[g(X)] = (1/sqrt(2*pi)) * sum_k w_k * g(mu_u + sd_u *
node_k)`. Use `N = 24` (the integrands -- a product of CDFs and a
Poisson-binomial CDF -- are smooth; 24 nodes is ample and still trivial).
numpy is already a dependency; generating nodes avoids hardcoded magic
numbers. (`delta_roto.py` hardcodes a 9-node set for its own purpose; that is
pre-existing and out of scope -- not shared here to avoid a cross-module
refactor.)

**wins (clear, non-overlapping):** deterministic count of opponents the
user's band clearly clears:

- higher-better: `(mu_u - sd_u) > (mu_j + sd_j)`
- inverse:       `(mu_u + sd_u) < (mu_j - sd_j)`

(Equivalently, in the flipped "bigger is better" score: `lower_u > upper_j`.)
Denominator is the opponent count (9 in a 10-team league).

**Edge cases (fall out naturally):**

- `sd_u = 0`: every Gauss-Hermite node maps to `x = mu_u`, and the weights sum
  to 1, so the integral collapses to a point evaluation at `mu_u`. No special
  case needed.
- `sd_j = 0`: `q_j(x)` becomes a step (1 if `mu_j > x`, else 0; 0.5 at
  equality via the CDF limit) -- handle the `sd_j = 0` division explicitly in
  the per-opponent probability helper, mirroring `_prob_beats`'s
  `combined == 0` guard.
- No user row in the data (should not happen -- the user team is always in the
  league): return `None` odds; the display line hides.

**Function shape:** a pure function taking the per-category team data and
returning the three numbers, independently unit-testable. Suggested:

```python
@dataclass
class CategoryOdds:
    first_pct: float       # 0-100
    top3_pct: float        # 0-100
    clear_wins: int        # 0..opponents
    opponents: int         # league size - 1

def category_finish_odds(
    means: Sequence[float],
    sds: Sequence[float],
    user_index: int,
    *,
    higher_is_better: bool,
) -> CategoryOdds: ...
```

Placed in `scoring.py` near `_prob_beats`. Percent rounding is a display
concern -- the function returns full-precision floats; the formatter/JS
rounds to whole numbers.

## Data flow & payload change

`format_category_bars_for_display` / `_category_bars_one_flavor` (in
`season_data.py`) already builds, per flavor per category, a list of rows
`{team, value, sd, is_user}` sorted best-on-top. Extend it: after building a
category's rows, call `category_finish_odds` (deriving `means`/`sds`/
`user_index`/`higher_is_better` from the rows + `INVERSE_STATS`) and attach
the result.

This changes each category entry from a bare list to:

```
flavor[CAT] = {
  "rows": [ {team, value, sd, is_user}, ... ],   # unchanged contents
  "odds": { "first_pct": int, "top3_pct": int, "wins": int, "opponents": int }
           # or null when there is no user row
}
```

`first_pct`/`top3_pct` are rounded to whole numbers in the payload (display
spec); `wins`/`opponents` are integers. Empty flavor (pre-refresh) stays `{}`
(no categories), so the line hides.

Cost: one ~24-node integral per category per flavor = ~20 integrals per page
load, each trivial. Computed where the rows are already built; no extra cache
reads.

## Display (template + JS)

Template (`standings.html`): a new element between `#catbars-cat-toggle` and
`.catbars-wrapper`:

```html
<div id="catbars-odds" class="catbars-odds">
    <span><strong>1st:</strong> <span id="catbars-first">--</span></span>
    <span><strong>top 3:</strong> <span id="catbars-top3">--</span></span>
    <span><strong>wins:</strong> <span id="catbars-wins">--</span></span>
</div>
```

A light CSS rule lays the three out on one line with spacing (co-located with
the other `#catbars-*` rules in the inline `<style>` block).

JS (`season_category_bars.js`):
- `rowsFor()` now reads `flavor[category].rows` (the shape changed).
- A new `oddsFor()` returns `flavor[category].odds` (or null).
- In `render()`, after drawing, populate the odds line:
  `1st: N%`, `top 3: N%`, `wins: x/9` (x = wins, 9 = opponents). If odds is
  null or there are no rows, hide `#catbars-odds`.

## Consistency note

These are true per-category RANK probabilities from the chart's bands. They
intentionally differ from the Monte Carlo tab's `category_risk` "top3_pct",
which is `P(user scores >= 8 roto POINTS in the category)`, MC-sampled --
a different quantity. The new line is consistent with THIS chart.

## Testing

`scoring.py` (`tests/test_scoring.py` or a new test module):
- Symmetric case: all 10 teams identical mean + SD -> `first_pct ~= 10`,
  `top3_pct ~= 30`, `clear_wins == 0`.
- Dominant case: user mean far above all, tiny SDs -> `first_pct == 100`,
  `top3_pct == 100`, `clear_wins == 9`.
- Inverse category (ERA, `higher_is_better=False`): user lowest ERA wins;
  a user with the highest ERA has `first_pct ~= 0`.
- Determinism: all `sd = 0` -> `first_pct` is 100 if user strictly best else
  0; `top3_pct` 100 if at most 2 strictly better.
- `clear_wins` honors band overlap in both directions.

`season_data.py` (`tests/test_web/test_season_data.py`):
- Category entries now have `{"rows": [...], "odds": {...}}`; `odds` has keys
  `first_pct`, `top3_pct`, `wins`, `opponents`; percents are whole numbers.
- A clear-cut fixture (user dominant) yields `first_pct == 100`,
  `wins == opponents`.

`season_routes.py` (`tests/test_web/test_season_routes.py`):
- Update the embedded-JSON assertions for the new shape
  (`bars["current"]["R"]["rows"][0]["team"]`, and assert
  `bars["current"]["R"]["odds"]` has the expected keys).

JS/template: `node --check`; `pytest -k standings` (page renders); manual
browser check (the line shows `1st`/`top 3`/`wins` and updates with the
category and projection toggles, between the selector and the chart).

## Verification

- `pytest tests/test_scoring.py tests/test_web/ -q` green.
- `ruff check` / `ruff format --check` / `vulture` clean for touched files;
  `mypy` for any touched file in `[tool.mypy].files` (scoring.py, season_data.py,
  season_routes.py are covered).
- `node --check` on the JS module; manual dashboard smoke of the new line.
