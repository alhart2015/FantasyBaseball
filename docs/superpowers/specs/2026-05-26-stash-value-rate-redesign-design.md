# Stash Board v2 -- Rate-Upgrade-Over-Return-Window Design

Date: 2026-05-26
Status: approved, implementing on `fix/stash-values`
Supersedes the gain metric in: `docs/superpowers/specs/2026-05-25-il-stash-value-design.md`

## Problem

The shipped stash board (`src/fantasy_baseball/lineup/stash_value.py`) ranks injured
players by their **marginal active value if slotted into the current lineup today**,
comparing **total ROS counting stats**. Counting categories (R/HR/RBI/SB, W/K/SV) scale
with playing time, so an injured player with little remaining volume is structurally
incapable of beating a healthy active starter -- regardless of how good a hitter/pitcher
he is. In production this collapsed the board: 23 of 25 candidates showed
`gain 0, sd 0, p 0.50`, and the only differentiation was a presentation artifact
(every FA shared `cost = -0.12`, the weakest owned IL stash's value, so all showed
`stash_value = 0.12`).

Evidence (live board, 2026-05-26): only owned pitchers Logan Webb (0.89) and Josh
Hader (0.64) showed value; every injured FA hitter was exactly 0. A synthetic elite
injured FA hitter reproduced `gain 0` against a roster that already won every hitting
category (legit zero), and `gain 5.00` once the roster was contested -- proving the
slotting path works and the zeros are a *metric* problem, not a computation bug.

## What we actually want

> Once an injured player is healthy, would swapping his playing time in for the weakest
> guy he'd displace improve a category I'm contesting?

A 100-AB injured hitter should never out-*total* a 300-AB healthy starter -- but if his
**rate** (over the ~100 ABs he'll actually play once back) beats the guy he'd replace,
that's a real upgrade. Saves included: Hader at 18 SV / 40 IP (0.45 SV/IP) beats an
incumbent at 18 SV / 70 IP (0.257 SV/IP) -- over Hader's 40 IP, +7.7 saves.

## The metric

This rate model applies to the **injured-FA (add) path** -- the source of the bug. The
**owned-IL path keeps its existing drop-cost** computation: a unified swap reintroduces
the PR #101 double-count, because `compute_delta_roto_band` applies the swap delta to the
`from_rosters` anchor, which already prices an owned IL player's ROS (see the edge-case
table). The owned drop-cost is already volume-aware via that displacement, so Webb/Hader
keep scoring sensibly.

For an injured FA candidate **C**, gain is the best
rate-normalized swap C could make against the current active lineup:

```
W            = C's remaining ROS volume        (AB for hitters, IP for pitchers)
rate_X(S)    = X.ros[S] / X.ros[volume]         (per-AB / per-IP, from ROS)
scale        = max(0, vol_I - W) / vol_I        (incumbent's share kept; clamp >=0)

For an eligible active incumbent I, build a SYNTHETIC replacement line:
    synth.ros[S] = scale * I.ros[S]  +  C.ros[S]                 (counting stats)
    synth components (H/AB, ER/IP, BB+H/IP) likewise               (rate stats)

This makes the team delta exactly:
    after[S] - before[S] = C.ros[S] - rate_I(S) * W
i.e. "trade W of the incumbent's playing time for W of C's, at ROS rates."

swap_band(C, I) = compute_delta_roto_band(before=active, after=active with I->synth)
gain(C)         = max over eligible incumbents I of swap_band(C, I).mean, floored at 0
band(C)         = the band of the argmax swap (mean/sd/p_positive/verdict)
```

- **Rates come from ROS** -- our projections already encode injury status and return
  likelihood, which is exactly the signal we want (a long-injured player has small W
  and/or a depressed rate).
- **Leverage is preserved** via `compute_delta_roto_band`: a rate upgrade only scores if
  it moves a category the user is contesting; an upgrade in an already-won (or hopeless)
  category correctly shows ~0.
- **Incumbent = the weakest position-eligible active player C would displace** =
  the eligible incumbent that maximizes the swap (displacing the weakest yields the
  biggest gain). Eligible = C can fill a slot I occupies (hitters: shared position;
  pitchers: any active pitcher, generic P slot).

## Worked examples (-> tests)

1. **Hitter upgrade (contested):** injured 1B, ~100 ROS AB at a better HR/AB and AVG
   than the weakest eligible active corner bat, user contesting HR -> `gain > 0`,
   even though his season totals are tiny. (Currently 0 -- the core regression.)
2. **Should-be-zero:** injured backup catcher, worse rate than every eligible active
   hitter -> every swap delta negative -> `gain == 0`, honestly shown (not 0.12).
3. **Closer:** Hader 18 SV / 40 IP vs weakest eligible active P at an 18 SV / 70 IP
   rate, SV contested -> `gain > 0` from the +7.7 SV over his window.
4. **Uncontested stays zero:** roster already wins a category by a wide margin -> a
   better hitter there adds no roto points -> `gain == 0` (leverage still gates).

## Edge cases

| Case | Resolution |
|------|-----------|
| Owned IL vs injured FA | **NOT unified** (decided during impl). The FA path gets the rate swap; the owned path keeps drop-cost. Verified: `compute_delta_roto_band` builds the EV mean by applying the swap's IN/OUT ROS delta to the user's `from_rosters` anchor row -- which already prices an owned IL player's ROS via displacement. A unified "add C" swap would therefore double-count owned C (PR #101). Guarded by the existing `test_owned_strong_il_arm_gain_is_drop_cost_not_double_count`, which still passes. |
| Multi-position eligibility | Incumbent chosen by max-gain over all eligible active players (the one C would actually displace), not a naive single-position match. |
| SP vs RP share P slot | Compare over C's IP window; rate stats (ERA/WHIP/K9 + SV) decide. A reliever won't out-rank a strong starter because the starter's rates hold over the same innings. |
| `W >= vol_I` | `scale` clamped to 0 -> synthetic = C's full ROS (full replacement); the swap degrades to the existing whole-player swap, which is correct when C will also out-volume the incumbent. |
| `W == 0` or missing ROS volume | Cannot rate-normalize -> `gain 0` and skip (guard the divide-by-zero; do not silently sink). |
| Cost artifact | Floor the IL-slot cost at 0 -- displacing a net-negative owned stash never *credits* a candidate. Removes the uniform `+0.12`. |
| Roster Audit page | Untouched -- it keeps the "is he an upgrade today" framing. This rate/return framing is stash-specific. |

## Implementation notes

- Reuse `compute_delta_roto_band(before, after, ...)` with the synthetic-incumbent swap;
  the band `mean` is the exact EV delta. The synthetic line's variance is approximate
  (it bundles the kept-incumbent share with C) -- acceptable for v1; the mean is exact.
- Build the synthetic line with `dataclasses.replace` on the incumbent's
  `rest_of_season`, scaling counting + rate components, then summing C's ROS.
- Keep `_open_il_slots`, `_owned_il_stashes`, and the cost/drop allocation; only the
  gain definition and the cost floor change.
- `stash_value.py` is in `[tool.mypy].files` -- mypy must pass.

## Out of scope (v1)

- Exact variance modeling of the synthetic line (mean is exact; sd is a reasonable
  approximation).
- Full-season-rate option (we use ROS rates, per decision).
- Re-optimizing the whole lineup around C (we do best pairwise swap, which matches the
  "replace player X" framing and avoids the volume-based optimizer trap).
