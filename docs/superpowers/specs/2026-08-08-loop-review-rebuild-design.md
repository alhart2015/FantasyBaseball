# loop-review rebuild: tiered dimension agents + cost instrumentation

**Date:** 2026-08-08
**Status:** approved design, pending implementation plan
**Artifact under change:** `~/.claude/skills/loop-review` (a global skill, outside this repo)
**Related:** issues #274 (loop-review latency), #316 (code-review 10-finding cap),
`docs/superpowers/loop-review-case-study-pr280-2026-07-31.md`

---

## Problem

`/loop-review` drives changed code to a clean state by alternating quality and
correctness review, arbitrating findings against a persistent ledger, and fixing
them one at a time. It finds real bugs. It is also expensive and does not reliably
converge, and both failures are measured rather than suspected.

**Cost.** On PR #280 a single pass consumed roughly 1.2M subagent tokens for the
correctness review alone, plus 250-350k for the four `/simplify` agents. Individual
review agents on PR #272 ran 14-17 minutes and 110-118k tokens each.

**Non-convergence.** The #280 run hit the 10-pass cap without ever reaching a clean
pass. At least five of the ten passes were reacting to the previous pass's own fix.
The case study's diagnosis: independent bugs decay to zero, but a cross-cutting
invariant gets discovered one facet per pass, because each narrow fix shifts the
problem to the adjacent facet.

**Truncated reporting.** `/code-review` displays a fixed number of findings (measured:
10 at high effort, every run) regardless of how many it verified. Across 14 consecutive
runs: 451 candidates, 140 reported, 72 refuted, 239 verified-and-unrefuted findings
never displayed. A quiet pass is not evidence of a clean branch.

Three causes are documented by Anthropic's own Opus 5 guidance and are addressable
without trading away finding quality:

1. Explicit verification instructions cause over-verification on Opus 5; removing
   them reduces tokens with no loss in quality. The current reviewers are prompted
   exactly that way (this is #274's hypothesis 2).
2. Review prompts that say "only report high-severity issues" or "be conservative"
   are followed literally and under-report. Reporting everything and filtering in a
   separate pass is the documented fix.
3. `low` and `medium` effort hold accuracy on Opus 5 and are the primary cost lever.

## Goals

1. **Converge on cross-cutting invariants**, not just independent bugs -- escalate
   from facet-fixing to state-space enumeration when the churn signal fires.
2. **Cut per-pass cost substantially** without downgrading finder capability, by
   removing work that produces no findings (redundant gate runs, re-review of stable
   code, up-front verification of candidates that will never be fixed).
3. **Cover the review dimensions explicitly**, including one that does not exist
   today: whether the diff achieves the goal stated in the issue.
4. **Make cost attributable**, so future tiering decisions come from a measured
   cost-per-confirmed-finding table rather than judgment.
5. **Preserve persistence** across passes, sessions, and compaction.

## Non-goals

- **Not downgrading model tiers now.** Phase 1 keeps every finder on Opus and varies
  only effort. Model tiering is Phase 2, gated on data this rebuild collects.
- **Not replacing `/code-review`** as a standalone tool. `harvest.py` is retained so
  a `/code-review` run can still be ingested when the user chooses to run one.
- **Not a persistent moderator subagent.** See "Moderator persistence" below.
- **Not automating `/code-review` invocation.** It is `disable-model-invocation`, a
  safety classifier blocks the workflow-level equivalent, and this was settled on
  2026-08-04. The rebuild sidesteps it by not invoking `/code-review` at all.
- **Not repo-specific.** The skill is global and must work in any repository; only
  the gate command list is project-derived.

## Chosen approach

Rebuild the engine in place. Keep the parts validated against real runs: the ledger,
the arbitration rules, the churn and drift discriminators, and the one-fix-at-a-time
protocol. Replace the finding-production layer and add cost instrumentation.

### Component layout

```
~/.claude/skills/loop-review/
  SKILL.md                     rewritten: new pass shape
  reference/ledger.md          unchanged
  reference/arbitration.md     + refute-on-demand, + second-finding rule
  reference/fix-protocol.md    unchanged
  scripts/ledger.py            + found_by, + goal, + per-pass agent cost records
  scripts/cost.py              NEW
  scripts/harvest.py           retained for optional /code-review ingest
  scripts/selftest.py          + cost.py coverage

~/.claude/agents/
  lr-correctness.md            model: opus,   effort: high
  lr-quality.md                model: opus,   effort: medium
  lr-intent.md                 model: opus,   effort: medium
```

Agent frontmatter supports both `model:` and `effort:`, verified against existing
agent definitions on this machine. That makes model and reasoning effort controllable
per dimension through the Agent tool's `subagent_type`, with no Workflow involvement.

### The three dimensions

The six dimensions originally requested do not partition cleanly: correctness, bugs,
and "other pitfalls commonly caught in review" are one search over one body of code.
Running them as separate agents produces duplicate findings, and every duplicate costs
the arbitrator a merge decision. Three axes that do not overlap:

| agent | searches for | cost of a miss |
|---|---|---|
| `lr-correctness` | bugs, edge cases, silent failures, guards that do not guard, invariant violations | high -- ships |
| `lr-quality` | reuse, duplication, simplification opportunities, dead code, altitude | low -- a missed cleanup costs nothing |
| `lr-intent` | whether the diff achieves the issue's stated goal; what it changed that the issue did not ask for; blast radius | high -- wrong feature, silently |

### Moderator persistence

The moderator is the **main loop agent**, with the ledger as its durable memory,
re-read at the top of every pass. It is not a persistent subagent.

The moderator must edit files, run gates, and commit, so it has to be the main agent
regardless. Beyond that, the #280 run is evidence against agent-context persistence
as a churn remedy: the orchestrator was continuously in context for all ten passes
and still band-aided facet after facet. What prevents churn is state that can be
*queried* -- `ledger.py status` returns the churn signal as a computed fact. A
subagent's context cannot be queried, audited, or recovered after session death, and
degrades under compaction at exactly the ten-pass horizon where it matters most.

---

## Requirements

Each requirement is numbered for traceability from the implementation plan.

### R1 -- Preflight: resolve the goal (once per loop)

R1.1 Resolve the branch's stated goal in this order and record the first hit:
  1. The GitHub issue referenced by the branch name or the PR body
     (`gh issue view <n> --json title,body`).
  2. The PR description (`gh pr view --json title,body`).
  3. Ask the user for a one-line statement.

R1.2 Write the resolved goal and its source to the ledger as a branch-level field, so
the intent reviewer and every later session read the same statement.

R1.3 If `gh` is unavailable or unauthenticated, fall through to (3) rather than failing
the loop.

### R2 -- Gates computed once per pass, in the parent

R2.1 The moderator runs the project's gates once per pass and writes combined output
to `<ledger-dir>/gates-pass-<N>.txt`.

R2.2 The gate command list is derived from the repository, not hardcoded in the skill.
For this repo that is `pytest -n auto`, `ruff check .`, `ruff format --check .`,
`vulture`, and `mypy`. When no gate list can be derived, record that fact in the
gates file rather than writing an empty file.

R2.3 Every finder prompt names that file path and instructs the agent **not** to run
the gates itself.

R2.4 Gates run after the pass's fixes land, and once before pass 1 to establish a
baseline.

### R3 -- Scope per pass

R3.1 Pass 1 reviews the full `main...HEAD` diff plus uncommitted changes.

R3.2 Passes after the first review `git diff <last-reviewed-sha>..HEAD` plus the blast
radius: files importing symbols the fixes touched.

R3.3 When the churn signal fires, the normal finders are replaced for that pass by a
single theme-enumeration prompt scoped to the sticky files (see R7).

R3.4 The sha reviewed by each pass is recorded in the ledger so R3.2 is computable in
a later session.

### R4 -- Finder prompts

R4.1 Each finder prompt carries a marker string of the exact form
`LR-PASS-<N>-<dimension>` (e.g. `LR-PASS-3-correctness`). This is the join key
`cost.py` uses to attribute a transcript to a pass and dimension.

R4.2 Each finder prompt carries: the scoped diff, the gate-output path, and the
resolved goal statement.

R4.3 Finder prompts contain **no verification instruction** -- no "verify before
reporting", no "reproduce it first", no "use a subagent to check". Removing these is
the documented Opus 5 fix for over-verification and is the single largest expected
latency win.

R4.4 Finder prompts contain **no severity filter**. They instruct the agent to report
everything including uncertain findings, and to mark its confidence. Filtering happens
at arbitration.

R4.5 Finder prompts instruct the agent not to run the test suite or gates.

### R5 -- Dimension gating

R5.1 `lr-correctness` runs every pass.

R5.2 `lr-quality` and `lr-intent` run on pass 1, and on a later pass only if that
dimension has produced at least one confirmed (fixed, not refuted) finding earlier on
this branch.

R5.3 When a dimension is skipped, that fact is recorded in the pass record, so a
report never implies coverage it did not have.

### R6 -- Refutation on demand

R6.1 Findings are ingested without up-front verification.

R6.2 The arbitrator checks the premise only of findings it is about to fix. A finding
whose premise does not reproduce is resolved as `refuted` with a note, which is durable
and does not reopen on re-observation.

R6.3 Findings below the fix threshold are never premise-checked; they remain open,
below threshold, and are named in the final report.

### R7 -- Cross-cutting escalation

R7.1 The existing churn signal is retained: the last two passes overlap in flagged
files and the finding count did not drop.

R7.2 A new, earlier trigger: on the **second confirmed finding in the same subsystem**
(same file, or same `--theme`), the arbitrator stops fixing facets and issues a
theme-enumeration prompt over that subsystem -- enumerate every state the invariant can
be in (absent, empty, all-NaN, partially joined, half-populated), state the intended
behavior of each and where it is enforced, then fix them together with tests.

R7.3 All findings from a theme-enumeration pass are tagged with one `--theme` and
fixed as one coherent change. A large diff from such a pass is a success, not a
violation of the one-fix-at-a-time protocol.

### R8 -- Cost instrumentation

R8.1 `cost.py collect --pass <N>` scans `<claude-project>/<session>/subagents/agent-*.jsonl`,
locates each finder by its R4.1 marker in the transcript's first user message, sums the
per-message `usage` blocks, and writes a per-agent record to `passes.jsonl`.

R8.2 Each record holds: `agent`, `model`, `effort`, `input_tokens`, `output_tokens`,
`cache_creation_input_tokens`, `cache_read_input_tokens`, `cost_usd`, `wall_ms`.

R8.3 `cost_usd` is computed from a weighted price table, not raw token totals. Cache
reads cost roughly 0.1x input and cache writes 1.25x (5-minute TTL), so raw totals
misattribute cost by more than an order of magnitude -- a sampled subagent carried
345,145 cache-read tokens against 281 output tokens.

R8.4 The price table is a separate, dated data file, listing per-million input and
output rates. Verified 2026-08-08: Opus 5 $5.00/$25.00; Sonnet 5 $3.00/$15.00 with
introductory $2.00/$10.00 in effect through 2026-08-31; Haiku 4.5 $1.00/$5.00. Cache
reads are priced at 0.1x the model's input rate and cache writes at 1.25x. The file
carries its verification date, and a rate older than 90 days is reported as stale
rather than silently trusted.

R8.5 Every finding records `found_by` (the dimension that observed it) at observation
time. When two dimensions observe the same finding, all observing dimensions are
retained, so neither gets sole credit.

R8.6 `cost.py report` emits, per dimension across the branch: total weighted cost,
confirmed count, refuted count, and **cost per confirmed finding**. Refuted findings
count against the dimension.

R8.7 `cost.py bakeoff --tier <spec>` re-runs one pass's finders at an alternate
model/effort against the same tree and reports the set difference between the finding
sets. It does not apply fixes.

### R9 -- Under-spend detection

R9.1 The ledger computes **escaped findings**: a finding first observed at pass N in a
file that was inside pass N-1's recorded scope. The earlier pass looked at that code
and missed it.

R9.2 `cost.py report` includes escaped-rate and refuted-rate per dimension. A tier
downgrade that raises either is under-powered, and that is visible in the same table
that motivates downgrades.

### R10 -- Budget

R10.1 The loop accumulates weighted cost across passes and compares it to a threshold.

R10.2 On crossing the threshold the loop **warns the user and asks whether to continue**.
It does not abort automatically: each fix is committed individually, so stopping is
always safe, but an automatic abort would hand back a half-hardened branch with no
decision from the user.

R10.3 The threshold is an option with a default; the user can raise, lower, or disable
it per run.

### R11 -- Convergence and escalation (unchanged)

R11.1 Converged means `ledger.py open --threshold <t>` exits 0: nothing open at or
above the threshold, and nothing untriaged.

R11.2 Escalations hand control back with the open list intact: drift (a finding
resurfaced twice), churn that survives a theme-enumeration pass, the pass cap, or the
budget threshold.

---

## Edge cases and failure modes

**Marker collision.** Two agents in one pass could carry the same marker if a dimension
were dispatched twice. `cost.py` treats multiple transcripts matching one marker as
multiple agents for that dimension and sums them, rather than picking one arbitrarily.

**Transcript not yet flushed.** A subagent transcript may not be complete when the
parent resumes. `cost.py collect` is idempotent and re-runnable; a pass whose collection
found no transcript is recorded as `cost: unknown`, never as zero. A zero would corrupt
the cost-per-finding table in the direction that makes a tier look cheap.

**Session directory changes mid-loop.** A resumed or compacted session may write to a
different session directory. `cost.py` scans all session directories under the project,
not only the current one.

**Goal is absent.** No issue, no PR, and the user declines to supply one: `lr-intent`
is skipped for the run and the final report says so explicitly. It does not fall back
to guessing intent from the diff, which would make it a second correctness reviewer.

**Goal is stale.** The issue may describe a superseded plan. `lr-intent` reports
divergence from the stated goal as a finding; the arbitrator may resolve it as
`wontfix` with a note, which is durable.

**Gates fail before pass 1.** A branch whose gates are already red is reported and the
loop asks whether to proceed, since every later gate run will be red for a reason the
loop did not cause.

**Gates cannot be derived.** R2.2 records the absence; finders are told gates were not
run, rather than being pointed at an empty file that implies a clean suite.

**A dimension is gated off and the code changes underneath it.** R5.2 gates on
historical yield, not on relevance. A pass that introduces substantial new code
re-enables all dimensions regardless of prior yield.

**Price table drift.** R8.4's staleness check surfaces it. A wrong price table
misranks dimensions, which is exactly the decision the table exists to inform.

**Escaped-finding false positives.** A finding can be "escaped" because the earlier
pass genuinely missed it, or because the fix in between *created* it. The ledger
already distinguishes these: a finding in code touched by the previous pass's fix is
fix-induced, not escaped. R9.1 excludes files the previous pass's fix modified.

---

## Testing

The skill lives outside this repository, so the repo's pytest suite does not cover it.
`selftest.py` is the test harness and must be extended alongside the implementation.

1. **`cost.py` arithmetic.** Synthetic subagent transcripts with known `usage` blocks,
   asserting the weighted `cost_usd` including the cache-read discount and cache-write
   premium. A test with a cache-read-dominated transcript pins R8.3 specifically.
2. **Marker matching.** Transcripts with matching, non-matching, and duplicate markers,
   asserting correct attribution and the multiple-match behavior.
3. **Missing transcript.** Asserts `cost: unknown`, never zero.
4. **Escaped-finding computation.** A fixture ledger with a finding first observed at
   pass 3 in a file inside pass 2's scope, asserting it counts as escaped -- and a
   second fixture where that file was modified by pass 2's fix, asserting it does not.
5. **Dimension gating.** A fixture ledger asserting that a dimension with only refuted
   findings is gated off, and one with a confirmed finding is not.
6. **Second-finding escalation.** A fixture ledger with two confirmed findings in one
   file, asserting the escalation trigger fires.
7. **Price-table staleness.** A table dated more than 90 days back is reported stale.
8. **Existing ledger self-tests** continue to pass unchanged.

Beyond the self-test, the rebuild is validated on a real branch: one run, with
`cost.py report` compared against the #280 and #272 measurements recorded in the case
study and issue #274.

---

## Phasing

**Phase 1 -- engine (this spec).** The three agent definitions, the rewritten pass
shape, gates-once, fix-delta scoping, dimension gating, goal resolution, refute-on-demand,
the second-finding escalation, `cost.py`, and the extended self-test. All finders stay
on Opus; only effort varies.

**Phase 2 -- tiering (not yet specified).** After three to five real branches have
produced a cost-per-confirmed-finding table, tier down where the table and a `bakeoff`
run agree it is safe. Phase 2 is a separate spec and is explicitly out of scope here;
the acceptance criterion for Phase 1 is that the table exists and is trustworthy, not
that any tier has changed.
