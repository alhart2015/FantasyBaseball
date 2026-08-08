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
  1. **The GitHub issue.** Take the issue number from the first run of 1-5 digits in
     the branch name that is delimited by `/`, `-`, or `_` (so `feat/277-decompose-luck`
     yields 277, and `feat/loop-review-rebuild` yields nothing). If the branch yields
     no number, take the first `#<digits>` reference in the PR body. Fetch with
     `gh issue view <n> --json title,body`.
  2. The PR description (`gh pr view --json title,body`).
  3. Ask the user for a one-line statement.

R1.2 Write the resolved goal, its source, and the resolved issue number (when there is
one) to the ledger as branch-level fields, so the intent reviewer and every later
session read the same statement.

R1.3 If `gh` is unavailable or unauthenticated, or the resolved number does not
correspond to an existing issue, fall through to the next source rather than failing
the loop. A number that resolves to an unrelated issue is the failure mode this order
guards against: the moderator confirms the fetched issue title is plausibly about the
branch before accepting it, and falls through if it is not.

### R2 -- Gates computed once per pass, in the parent

R2.1 The moderator runs the project's gates **once per pass, at the end of the pass
after its fixes have landed**, and writes combined output to
`<ledger-dir>/gates-pass-<N>.txt`. A baseline run before pass 1 writes
`gates-pass-0.txt`.

R2.1.1 Pass N's finders therefore read `gates-pass-<N-1>.txt` -- the most recent gate
output, describing exactly the tree they are reviewing. Pass 1 reads the baseline. This
is the whole point of computing gates in the parent: the finders get a current, shared
result without any of them paying to produce it.

R2.2 The gate command list is derived from the repository, not hardcoded in the skill.
Resolution order, first hit wins:
  1. A `gates` list in `<ledger-dir>/config.json`, if the user has set one for this
     repository. This is also where a user overrides a wrong derivation.
  2. Commands appearing in a fenced block under a heading matching
     `/verification|gates|checks|end-of-effort/i` in the nearest `CLAUDE.md`. In this
     repository that yields `pytest -v`, `ruff check .`, `ruff format --check .`,
     `vulture`, and `mypy`.
  3. Ecosystem defaults, chosen by manifest: `pyproject.toml` or `setup.py` ->
     `pytest -q`; `package.json` -> its `test` and `lint` scripts when present;
     `go.mod` -> `go test ./...` and `go vet ./...`; `Cargo.toml` -> `cargo test` and
     `cargo clippy`.
  4. No gates.

R2.2.1 The resolved list and the rule that produced it are written to the top of the
gates file. When resolution reaches (4), the gates file records "no gates derived" and
the finder prompts say gates were not run -- never an empty file, which would read as
a clean suite.

R2.2.2 The derived list is recorded in `config.json` on first resolution, so a later
pass does not silently re-derive a different list mid-loop.

R2.3 Every finder prompt names its pass's gate file path (per R2.1.1) and instructs the
agent **not** to run the gates itself.

R2.4 A pass that applies no fixes does not re-run the gates; its finders continue to
read the most recent gate file, and the pass record notes which one.

### R3 -- Scope per pass

R3.1 Pass 1 reviews the full `main...HEAD` diff plus uncommitted changes.

R3.2 Passes after the first review `git diff <last-reviewed-sha>..HEAD` plus the blast
radius. Blast radius is computed language-agnostically: for each symbol added, renamed,
or removed by the **preceding** pass's fixes, a repo-wide literal search for that name
(`rg --fixed-strings`), and every file with a hit joins the scope. This over-includes
compared with a real import graph and deliberately so -- the failure it prevents is a
caller left unreviewed, and the cost of an extra file in scope is far below the cost of
a missed one. A language-aware refinement is possible later but is not required here.

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

R5.2 `lr-quality` and `lr-intent` run on pass 1, and on a later pass if **either**
condition holds:
  (a) the dimension has produced at least one confirmed (fixed, not refuted) finding
      earlier on this branch; or
  (b) the pass's scope (R3.2) contains a file that is new to the branch, or the scope's
      diff adds 50 or more lines. New code has no yield history, so gating it on prior
      yield would let a dimension miss the only code it was ever going to have an
      opinion about.

R5.3 When a dimension is skipped, that fact and the reason are recorded in the pass
record, so a report never implies coverage it did not have.

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

R7.2 A new, earlier trigger: when the ledger holds **two or more triaged, unrefuted
findings in the same subsystem** (same file, or same `--theme`) that are at or above the
fix threshold, the arbitrator stops fixing facets and issues a theme-enumeration prompt
over that subsystem -- enumerate every state the invariant can be in (absent, empty,
all-NaN, partially joined, half-populated), state the intended behavior of each and
where it is enforced, then fix them together with tests.

R7.2.1 The trigger is evaluated on **observation and triage, before any of the group is
fixed** -- not on confirmed-as-fixed. Firing after two fixes would mean band-aiding two
facets first, which is the behavior R7 exists to prevent. The two findings need not
come from the same pass: two facets surfaced in one pass are exactly the case the
trigger should catch soonest.

R7.2.2 Because R6.2 defers premise-checking to fix time, "unrefuted" would otherwise be
vacuous at trigger time -- nothing has been checked yet. So the trigger premise-checks
the candidate group first: when two or more same-subsystem findings reach the threshold,
the arbitrator checks each one's premise, and fires only if at least two survive.
Refuted ones are resolved as refuted and do not count. This is the one place premise
checking happens before a fix is chosen, and it is worth the cost: a
theme-enumeration pass is the most expensive action the loop can take, and firing it on
two false premises spends that cost on an invariant that does not exist.

R7.3 All findings from a theme-enumeration pass are tagged with one `--theme` and
fixed as one coherent change. A large diff from such a pass is a success, not a
violation of the one-fix-at-a-time protocol.

### R8 -- Cost instrumentation

R8.1 `cost.py collect --pass <N>` scans `<claude-project>/<session>/subagents/agent-*.jsonl`,
locates each finder by its R4.1 marker in the transcript's first user message, sums the
per-message `usage` blocks, and writes a per-agent record to `passes.jsonl`.

R8.2 Each record holds: `agent`, `model`, `effort`, `input_tokens`, `output_tokens`,
`cache_creation_input_tokens`, `cache_read_input_tokens`, `cost_usd`, and `wall_ms`.
`wall_ms` is the span between the first and last message timestamps in that agent's
transcript, not a field of `usage`. When a transcript carries no usable timestamps,
`wall_ms` is recorded as `null` -- never 0, which would read as an instant agent.

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

R8.5.1 A finding observed by K dimensions counts **1/K toward each** in R8.6's
confirmed and refuted counts. Whole credit to each would make the per-dimension counts
sum to more than the number of real findings, and cost-per-finding would understate
every dimension's cost by exactly the overlap rate -- the error grows with duplication,
which is the thing the three-dimension split was designed to minimize and therefore the
thing the table most needs to measure honestly. `report` also prints the raw
whole-credit count beside the fractional one, since "how many findings did this
dimension touch" and "how much of the yield is attributable to it" are different
questions.

R8.6 `cost.py report` emits, per dimension across the branch: total weighted cost,
confirmed count, refuted count, and **cost per confirmed finding**. Refuted findings
count against the dimension.

R8.7 `cost.py bakeoff --pass <N> --dimension <name> --tier <model>/<effort>` (e.g.
`--pass 2 --dimension quality --tier sonnet/low`) re-runs that one dimension against the
tree pass N reviewed, at the alternate model and effort, and reports three sets: findings
only the baseline tier found, findings only the alternate found, and findings both
found. It does not apply fixes and does not write to the ledger's finding table; its
output is a standalone report.

### R9 -- Under-spend detection

R9.1 The ledger computes **escaped findings**: a finding first observed at pass N in a
file that was inside pass N-1's recorded scope **and was not modified by pass N-1's
fixes**. The earlier pass looked at that code, it did not change afterwards, and the
finding was still missed. Excluding fix-modified files is what separates a genuine miss
from a fix-induced defect; without the exclusion, churn inflates the escaped-rate and
makes a capable tier look under-powered.

R9.1.1 An escape is charged only to dimensions that **actually ran over that scope in
pass N-1**. A dimension gated off by R5.2 had no opportunity to see the code and is not
charged. Without this, gating and the under-spend signal fight each other: the cheaper
a dimension runs, the worse its escaped-rate looks for the passes it sat out, and the
table would recommend re-enabling a dimension precisely because it was economical.

R9.1.2 An escape charged to a dimension is charged **whole to each dimension that ran**,
not fractionally as in R8.5.1. The questions differ: R8.5.1 apportions credit for one
finding among its finders, whereas every dimension that looked at the code and missed it
missed it in full.

R9.2 `cost.py report` includes escaped-rate and refuted-rate per dimension, each with
the denominator it is a rate over -- escaped-rate over the passes that dimension ran,
refuted-rate over the findings it reported. A tier downgrade that raises either is
under-powered, and that is visible in the same table that motivates downgrades.

### R10 -- Budget

R10.1 The loop accumulates weighted cost across passes and compares it to a threshold.

R10.1.1 Agents whose cost is `unknown` (a transcript that could not be read, per the
edge case below) are counted at the branch's mean per-agent cost so far, and the budget
report states how many agents were estimated. Counting them as zero would let a loop
with unreadable transcripts run past the threshold without the check ever firing --
the failure mode where the safeguard is quietest is exactly the one where it matters.
When no agent has a known cost yet, the check does not fire and says so.

R10.2 On crossing the threshold the loop **warns the user and asks whether to continue**.
It does not abort automatically: each fix is committed individually, so stopping is
always safe, but an automatic abort would hand back a half-hardened branch with no
decision from the user.

R10.3 The threshold defaults to **USD 15.00** per loop. `--budget <usd>` raises or
lowers it; `--budget none` disables the check. This default is a deliberate placeholder,
not a derived figure: the pre-rebuild measurements in #274 and the #280 case study
record token counts, not weighted dollar costs, so no honest pass-equivalence can be
stated until `cost.py` has produced one. Retuning it from the first real runs is part of
Phase 1's output, and the first report should state what the default actually bought.

### R11 -- Convergence and escalation (unchanged)

R11.1 Converged means `ledger.py open --threshold <t>` exits 0: nothing open at or
above the threshold, and nothing untriaged.

R11.2 Escalations hand control back with the open list intact: drift (a finding
resurfaced twice), churn that survives a theme-enumeration pass, the pass cap, or the
budget threshold.

### R12 -- harvest.py

R12.1 `harvest.py` is unchanged and is **not** invoked by the loop. The rebuilt pass
never runs `/code-review`, so there is no telemetry to ingest by default.

R12.2 It remains available for the case where the user runs `/code-review` themselves
and wants its full verified finding set (not its truncated report) folded into the
ledger: `harvest.py ingest --pass <N>`. Findings ingested this way are recorded with
`found_by: code-review` so they do not distort the per-dimension cost table, which has
no cost record for them.

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

**Gates cannot be derived.** R2.2.1 records the absence; finders are told gates were not
run, rather than being pointed at an empty file that implies a clean suite.

**A dimension is gated off and the code changes underneath it.** Handled by R5.2(b):
a scope containing a file new to the branch, or adding 50 or more lines, re-enables
every dimension regardless of prior yield.

**Price table drift.** R8.4's staleness check surfaces it. A wrong price table
misranks dimensions, which is exactly the decision the table exists to inform.

**Escaped-finding false positives.** A finding can be "escaped" because the earlier
pass genuinely missed it, or because the fix in between *created* it. R9.1's exclusion
of fix-modified files separates the two.

**Gate derivation picks the wrong commands.** R2.2(2) parses prose, and prose changes.
A wrong list makes every gates file misleading. R2.2.2 pins the list on first
resolution so it cannot drift mid-loop, and R2.2(1) lets the user override it durably
per repository.

**A budget warning arrives mid-fix.** The check runs at pass boundaries, not inside the
fix protocol, so a fix is never left half-applied by the prompt.

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
   pass 3 in a file inside pass 2's scope, asserting it counts as escaped; a second
   fixture where that file was modified by pass 2's fix, asserting it does not; and a
   third where the dimension was gated off in pass 2, asserting that dimension is not
   charged (R9.1.1) while a dimension that did run is charged in full (R9.1.2).
5. **Credit apportionment.** A fixture where one finding is observed by two dimensions,
   asserting each is credited 0.5 in the fractional count and 1 in the raw count, and
   that the fractional counts across dimensions sum to the number of distinct findings.
6. **Dimension gating.** A fixture ledger asserting that a dimension with only refuted
   findings is gated off; one with a confirmed finding is not; and one with only
   refuted findings is nonetheless re-enabled when the scope contains a new file or
   adds 50+ lines (R5.2b).
7. **Second-finding escalation.** A fixture ledger with two triaged, unrefuted,
   at-or-above-threshold findings in one file, asserting the trigger fires **before**
   either is fixed. A companion fixture where one of the two is refuted by the R7.2.2
   premise check asserts it does not fire.
8. **Budget with unknown costs.** A fixture where one agent's cost is unknown asserts
   it is estimated at the running mean and reported as estimated, not counted as zero;
   and that with no known costs the check reports that it cannot evaluate.
9. **Gate derivation.** Four fixtures exercising R2.2's resolution order: an explicit
   `config.json` list wins over CLAUDE.md; a CLAUDE.md verification block is parsed;
   a bare `pyproject.toml` yields the ecosystem default; a repo matching none records
   "no gates derived" rather than an empty list.
10. **Gate file selection.** A fixture asserting pass N's finder prompts name
    `gates-pass-<N-1>.txt`, that pass 1 names the baseline, and that a pass applying no
    fixes leaves the pointer on the most recent file (R2.1.1, R2.4).
11. **Price-table staleness.** A table dated more than 90 days back is reported stale.
12. **Existing ledger self-tests** continue to pass unchanged.

Beyond the self-test, the rebuild is validated on one real branch, reported per A5
below. The pre-rebuild figures on record are token counts and wall clock, not weighted
dollars, so the comparison is stated in those units.

---

## Phasing

**Phase 1 -- engine (this spec).** The three agent definitions, the rewritten pass
shape, gates-once, fix-delta scoping, dimension gating, goal resolution, refute-on-demand,
the second-finding escalation, `cost.py`, and the extended self-test. All finders stay
on Opus; only effort varies.

**Phase 1 acceptance.** All of the following are observable:

  A1. `selftest.py` passes, including every new test listed above.
  A2. A full loop runs to convergence or a named escalation on one real branch, with
      `ledger.py open --threshold medium` exiting 0 or the escalation reported.
  A3. `cost.py report` emits a per-dimension table with a non-null weighted cost, a
      confirmed count, a refuted count, an escaped count, and cost-per-confirmed for
      every dimension that ran.
  A4. Every agent that ran in that loop appears in the cost table -- no pass records
      `cost: unknown` for an agent that completed.
  A5. That loop's total weighted cost is reported alongside the pre-rebuild
      measurements (#272: ~110-118k tokens per review agent; #280: ~1.2M subagent
      tokens per pass), so the change in cost is stated as a number rather than
      claimed.

A5 reports the comparison; it does not assert a target. A run that costs more than
expected is a valid Phase 1 outcome provided the number is measured and reported --
the point of Phase 1 is the instrument, not a predetermined reading.

**Phase 2 -- tiering (not yet specified).** After three to five real branches have
produced a cost-per-confirmed-finding table, tier down where the table and a `bakeoff`
run agree it is safe. Phase 2 is a separate spec and is explicitly out of scope here.
