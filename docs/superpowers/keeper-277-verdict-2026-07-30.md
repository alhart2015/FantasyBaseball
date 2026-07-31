# #277 verdict: decompose `luck` into playing-time and batted-ball

**Shipped:** parameterization **C** for BOTH pools -- keep `luck`, add a `batted_ball`
family at a NEGATIVE weight.

```
FAMILIES        = {"hitter": (skill, luck, batted_ball, future, age),
                   "pitcher": (skill, luck, batted_ball, future, age)}
FITTED_WEIGHTS  = {"hitter":  (1.0, 0.8, -0.2, 0.2, 0.30),
                   "pitcher": (1.0, 0.8, -0.2, 0.2, 0.15)}
```

Regenerate all of the below with `python scripts/keeper_rankings.py --backtest`
(weights) and `--study` (mechanism); `--fit` regenerates `projection.py`. Raw run in
`keeper-277-bakeoff-2026-07-30.txt`.

## The question

`luck = value_pct - skill_pct` carried a single positive weight (0.8/0.6). It bundles
two things: real playing-time/role signal, and batted-ball rate luck (AVG over xBA,
ERA under FIP) that regresses. The single term oversold everyday-but-lucky bats
(Ceddanne Rafaela .278 AVG vs .242 xBA; Otto Lopez). #277 asked to split it.

## The bake-off (holdout = 2024, fit = 2022-23)

| candidate | hitter holdout | pitcher holdout | note |
|---|---|---|---|
| baseline (pre-change) | **0.7085** | 0.4962 | reproduced exactly -> generalization is inert |
| A: pt + luck | 0.6878 | 0.4905 | adding a raw playing-time family; overfit, lost holdout |
| B: pt + batted_ball (luck dropped) | 0.6812 | 0.4932 | batted_ball weight = **0.00**; drops luck's SB/role signal |
| **C: luck + batted_ball claw-back** | **0.7002** | **0.5094** | **shipped** |

hitter noise floor 0.0172, pitcher 0.14 (all pitcher gaps are statistical ties).

## Why C, and what it settles

- **Playing time is real signal, batted-ball luck is noise -- both confirmed.**
  `--study`: playing time -> next-year PT **0.607** (hitters); batted-ball ->
  next-year SGP/PT/RATE **0.03-0.05** (hitters), **~0.00** (pitchers).
- **The original #277 hypotheses were wrong.** Adding a playing-time family did NOT
  shrink `luck` (it stayed 0.80) and it overfit (A < baseline). Measuring batted-ball
  directly and dropping luck (B) put batted_ball at exactly zero and lost the holdout,
  because luck also carries the real SB/saves/role signal B threw away.
- **C is the fix the evidence pointed to** (added beyond the original A/B plan). Keep
  `luck`; add `batted_ball` free to go negative. The grid independently chose **-0.20
  in both pools** -- it wants to claw the batted-ball half back out of luck. batted_ball
  predicts nothing on its own but correlates with luck, so the negative weight subtracts
  only the non-repeating part.
- **Cost: none.** Hitters 0.7002 vs 0.7085 is within the 0.0172 noise floor; pitchers
  0.5094 > 0.4962 (nominally better, within noise).
- **Decision check passes.** On the live 2026 hitter board, C moves Rafaela 31 -> 50
  and Otto Lopez 34 -> 38 (down, as intended) while Yordan Alvarez 22 -> 16 (up) --
  the skilled everyday control is not demoted.

## Did the shipped weights change?

Yes. Hitters and pitchers both moved from four families to five: `luck` retained
(pitcher luck 0.60 -> 0.80), `batted_ball` added at -0.20, `future` 0.4 -> 0.2. This
is a real change to keeper valuations, decided by the holdout, not by intuition.

## Deliberately not done

- `pt` did not ship as a family (A lost the holdout). No raw playing-time term.
- No speed/SB skill (C keeps luck, so the SB signal is retained; nothing was dropped).
- `sgp_sd` refit is mechanical (new composite), NOT #278's skill-term fix.
- `luck`x`future` collinearity left alone (#277: not a demonstrated bias).
