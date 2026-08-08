# Handoff: the churn detector did not fire on a run that churned

**Date:** 2026-08-08
**Branch that produced the evidence:** `feat/350-trajectory-player-search`
**Ledger (still on disk):** `.git/loop-review/feat-350-trajectory-player-search/`
**Code under discussion:** `~/.claude/skills/loop-review/scripts/ledger.py`, `detect_churn` (line 541)
**Related:** #274 (where the discriminator was validated), the PR #280 case study,
`docs/superpowers/specs/2026-08-08-loop-review-rebuild-design.md` R7.1

---

## Read this first: an earlier diagnosis in the ledger is WRONG

A `pass --note` on pass 7 of that branch claims the detector is "blind to fix-only
passes" because the loop interleaves observing passes (1, 4, 7) with fix-only passes
(2, 3, 5, 6). **That is false and should not be built on.** It was asserted without
being checked, and checking took one command.

`pass_file_sets` and `pass_counts` are both built only from `observe` ops. A pass that
records no observations never enters either dict, so `passes = sorted(files)` on that
branch is `[1, 4, 7]` -- the detector compared the two consecutive *observing* passes,
exactly as intended. Verify with:

```python
import ledger
d = ledger.ledger_dir()          # from inside the repo, on that branch
print(sorted(ledger.pass_file_sets(d)))   # [1, 4, 7]
print(ledger.detect_churn(d))
```

Delete or correct that note when you touch this.

## What actually happened

```
counts per observing pass:  {1: 28, 4: 14, 7: 9}
detect_churn:               churn=False, count_dropped=True, jaccard=0.4
                            sticky_files = trajectory_suggest.js, trajectory_view.py
```

The detector did what it was built to do. Findings decayed 28 -> 14 -> 9, and the
#274 rule is "overlap AND the count did not drop." The count dropped, so it stayed
silent, correctly by its own definition.

**The run still churned.** Passes 4 and 7 were verification passes over the *fixes*,
and each found defects the previous batch of fixes had created:

- pass 4 found 3 regressions introduced by the pass 2-3 fixes, plus 2 fixes that closed
  only half of what they claimed
- pass 7 found 1 regression introduced by the pass 5 fix (moving the suggestion list
  onto the form broke the `focusout` guard's premise, so Tab destroyed the list), plus
  the *same shape* of incomplete fix recurring: `_board_horizons` was made the single
  owner of `max_horizon` while `base_season`, read identically one line above it, was
  left unguarded

So the real gap is not decay detection. It is:

> **The discriminator measures decay, not provenance.** A run whose finding counts fall
> steadily can still be generating a constant fraction of new defects out of its own
> fixes, and nothing in the ledger's signals distinguishes "we are finding the original
> diff's bugs" from "we are finding the bugs we just wrote."

#280's shape was *count does not drop*. This run's shape is *count drops on schedule
while every pass cleans up after the last one*. Both are churn; only the first is
detected.

## The obvious fix does not work -- measured, not assumed

The ledger already records `pass_meta[N]["fix_files"]`, so the tempting signal is
"a finding first seen at pass N in a file that pass N-1's fixes modified." That is the
exact inverse of the escaped-findings computation (R9.1), which excludes those files.

Measured on this branch:

```
pass 4: 14 new findings, 14 in files the previous fixes touched (100%)
pass 7:  9 new findings,  9 in files the previous fixes touched (100%)
```

**File granularity is useless here.** The branch touches 7 files; remediation edits the
same files review is reading, so every finding trivially lands in a "touched" file. A
signal that fires 100% of the time on a healthy branch is worse than no signal, because
it will be ignored within two runs.

Do not ship the file-overlap version. If you want to pursue inference rather than
provenance, it has to be at hunk granularity -- a finding whose `line` falls inside a
range the previous pass's diff actually changed -- and you should measure the false-fire
rate on this branch's ledger before trusting it. `git diff <prev_sha>..<sha> -U0` gives
the changed line ranges, and `pass_meta` already stores both shas.

## What to build instead: record provenance, do not infer it

Both verification passes *stated the provenance in plain language*, unprompted:

> "This handler did not exist before this batch, so it is a regression introduced by
> the fix."
> "Introduced by this batch."

The reviewer knows. Nothing captures it. That is the cheapest correct fix:

1. **`ledger.py add --regression`** -- a boolean recorded on the observe op and folded
   onto the record. The finder prompt asks for it when a finding is in code the pass is
   reviewing *as a fix* (the verification-pass shape), not on a pass-1 feature review.
2. **`detect_churn` gains a second, independent trigger.** Keep the existing
   overlap-and-no-decay rule (it is validated and it catches #280's shape). Add:
   fire when a pass's regression fraction is at or above a threshold, regardless of
   whether counts dropped. On this branch that is 3/14 and 1/9 -- pick the threshold
   from more than one run, and state in the docstring which runs it was fitted on, the
   way the existing 0.60/0.25 match thresholds do.
3. **`status` reports it**, and `reference/arbitration.md` gets the escalation: when the
   regression trigger fires, stop fixing and hand back. This run needed that rule and
   the skill does not have it -- I kept fixing through two verification passes that each
   proved my fixes were the problem.

## The rule the skill is actually missing

Separately from the signal: **there is no stop rule for "the operator is the churn
source."** The escalations are drift, churn, the pass cap, and the budget. None fires
when remediation keeps generating findings. On this branch the honest action after
pass 4 was to hand back, and nothing told me so.

Suggested wording for `reference/arbitration.md`:

> Two consecutive verification passes that each find defects introduced by the previous
> batch of fixes is an escalation, not a reason to fix again. Hand back the open list
> and say which fixes caused which findings. A third batch from the same hand will
> produce a fourth pass.

## Verification plan

`selftest.py` is the harness (currently 168 checks; it lives beside `ledger.py` and the
repo's pytest does not cover any of this).

1. **Regression on the existing behaviour.** A fixture reproducing #280's shape --
   overlapping files, counts flat -- must still fire on the original rule. This is the
   test that stops the new trigger from replacing a validated one.
2. **Decay-with-regressions.** Counts 28 -> 14 -> 9 with 3 and 1 findings flagged
   `regression=True`. Asserts the new trigger fires where the old one is silent. Build
   the fixture from this branch's real ledger rather than inventing numbers.
3. **Healthy decay.** Counts falling with zero regressions must stay silent, or the
   signal is the 100%-fire failure above wearing a different hat.
4. **Provenance round-trip.** `add --regression` survives `fold`, and a finding
   re-observed without the flag does not lose it.
5. **`status` and `subsystems` unchanged** for a ledger with no regression flags, so
   every existing branch's output is stable.

## Reproducing the evidence

The branch is not merged and not pushed. The ledger is complete: 51 findings, 28 fixed,
23 open, six passes recorded, per-agent costs attributed, and verification notes on the
findings whose premises were checked.

```bash
git checkout feat/350-trajectory-player-search
L="python C:/Users/HartAlden/.claude/skills/loop-review/scripts/ledger.py"
$L status
$L list --full
python C:/Users/HartAlden/.claude/skills/loop-review/scripts/cost.py report
```

Findings tagged `regression` do not exist yet -- that is the point. The four that would
carry it are, in the ledger's own words:

- the `focusout` handler emptying the list before a mouse click can navigate (pass 4)
- moving the list into the `<label>`, which broke tab order and label semantics (pass 4)
- the `<ul>` inside a `<label>` being invalid content (pass 4)
- the list moving back onto the form breaking the `focusout` guard's premise (pass 7)
