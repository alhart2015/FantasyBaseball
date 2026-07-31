# /loop-review measured findings + recommendation (issue #274)

**Deliverable for issue #274** ("Investigate /loop-review latency -- 15 min per review
agent, and whether the loop is still the right shape for Opus 5").

This is a measurement-backed recommendation built from the *full population* of recent
loop-review runs on this machine, not just the two hand-written case studies already in
#274 (the #272 run inline in the issue, and the #280 run in
`loop-review-case-study-pr280-2026-07-31.md`). It supersedes the leading hypothesis in
#274 on one important point (H1) and reprioritizes the rest.

- **Population measured:** every `/code-review` workflow this repo's Claude Code sessions
  recorded, 2026-07-14 through 2026-07-31.
- **N = 40 `/code-review` passes across 620 subagents, ~44.1M code-review tokens.**
- **Model under measurement: `claude-opus-4-8[1m]` (39/40 passes; 1 recorded as
  `claude-opus-4-8`). None were a distinct "Opus 5."** See "On the Opus 5 premise" below.

---

## TL;DR

1. **Verdict: RESTRUCTURE, not replace -- but not the restructure #274 assumed.** The lever
   is **pass count / churn**, not per-pass gate compute.

2. **Kill H1 (shared gate cache). It is a non-remedy for this workload.** The issue assumed
   "every agent independently re-runs the full gate suite (`pytest -n auto` = 153s, plus
   mypy/ruff/vulture and 5 script modes)." Measured reality across **620 agents**:
   `pytest -n auto` was run **0 times**; `pytest` of any (scoped) kind **0.31x per agent**;
   `mypy` 0.11x; `ruff` 0.12x; `vulture` 0.03x. You cannot cache away spend that is not
   happening. The per-pass cost is agents *reading the diff and reasoning*, and it is mostly
   cache-served.

3. **The real high-leverage fix is H6 (cross-cutting-invariant churn), and it is
   empirically detectable.** Three multi-pass runs churned instead of decaying; a simple
   location-overlap trigger separates them cleanly from the clean-decay runs. Collapsing the
   worst run's churn tail into one enumerate-the-invariant pass would have saved an estimated
   ~7-8M tokens on that single run.

4. **Secondary levers, in order:** H4 (fix-diff scoping after pass 1, supported by the same
   overlap data), then H3 (decay early-exit) and agent-count taper -- both of which change
   the skill's explicit "cost is never a stop condition" stance and so are deliberate policy
   calls, not silent tweaks. And **`/simplify` (Finding 7) is the most cuttable component** --
   73% apply rate but 2.7x lower yield than the code-review that follows it, ~20% of the
   machinery's tokens, with one documented regression to its name.

5. **This is the Opus 4.8[1m] baseline.** Opus 5 will need re-measurement before its loop is
   tuned; the *mechanisms* below should transfer, the *magnitudes* will not. The "What to
   re-measure on Opus 5" section is the checklist for that.

---

## On the "Opus 5" premise

The issue title asks whether the loop is "still the right shape for Opus 5." Every workflow
record in the telemetry carries `defaultModel = claude-opus-4-8[1m]`. As of this writing the
machine has not been upgraded to Opus 5, so **all measured data is the Opus 4.8[1m]
baseline.** Treat this document as: (a) the definitive 4.8 baseline, and (b) the mechanism
model to carry into building the loop for Opus 5. Do not assume the token/latency magnitudes
here hold on Opus 5 -- re-run the extraction (appendix) against the first real Opus 5 runs.

---

## Method (reproducible)

Each `/code-review` invocation is workflow-backed and leaves two artifacts per run:

- `~/.claude/projects/<PROJECT>/<session>/workflows/wf_<runId>.json` -- the workflow record:
  `workflowName`, `args` (branch + effort), `agentCount`, `totalTokens`, `totalToolCalls`,
  `durationMs`, `defaultModel`, and `result.stats` (`finders`/`verifierAgents`/`reported`/
  `refuted`) plus `result.findings[]` (each with `file`, `line`, `category`, `verdict`).
- `~/.claude/projects/<PROJECT>/<session>/subagents/workflows/wf_<runId>/agent-*.jsonl` --
  the per-subagent transcript (every tool_use, including the exact Bash command lines).

`/simplify` is **not** workflow-backed (it is a plain parallel fan-out of Agent calls), so it
leaves no `wf_*.json`; this census therefore covers the `/code-review` half of each pass --
the heavyweight, ~75%+ of pass tokens. See "Measurement gap" below.

Extraction scripts are in the appendix. Grouping into "runs": consecutive `/code-review`
workflows on the same branch within a session (or, for uncommitted-diff reviews, within a
~90-minute gap).

---

## Finding 1 -- Per-pass cost, and where it goes

Aggregate over 40 passes:

| metric | min | median | mean | max | sum |
|---|---|---|---|---|---|
| tokens / pass (`totalTokens`) | 488k | **1,119k** | 1,101k | 1,972k | **44.1M** |
| subagents / pass | 9 | 14 | 15.5 | 32 | 620 |
| findings reported / pass | 0 | 4 | 5.0 | 15 | -- |
| wall-clock min / pass | 8.5 | **23.5** | -- | ~30 | -- |

(One pass recorded `durationMs` ~= 343 min / 5.7h. That is a background-queueing measurement
artifact, not agent compute -- flagged already in the #280 case study. `durationMs` is
unreliable as a compute proxy; use `totalTokens` and inter-pass timestamps instead.)

**Where the tokens go (one representative 15-agent xhigh pass, `wf_6a85cdc6`):**

- Tool mix: `Bash` 94 (of which `git` 56, `grep` 23, `sed` 8, `python` 4, other 3),
  `Read` 87, `Grep` 20, `StructuredOutput` 15. **446 assistant turns.**
- Token shape: raw input incl. cache ~= **33.6M**, output ~= **0.31M**, workflow-reported
  `totalTokens` ~= 1.30M. Output is tiny; the spend is *input/context* -- agents reading the
  diff and source -- and most of it is **cache-served**.

So a pass is "N independent agents each re-read the same diff and the same changed files and
reason over them." The redundancy that exists is diff/file re-reading, not gate execution --
and because it is cache-served, deduping it yields only modest *billed* savings.

## Finding 2 -- H1 (shared gate cache) is a non-remedy. REJECT.

The issue's single biggest proposed win was: run the gate suite once per iteration in the
parent and pass results down, because "with 4 simplify agents + 1 review agent per iteration,
one iteration can pay [the 153s pytest + mypy + ruff + vulture + 5 script modes] cost 5+
times."

Measured across **all 620 code-review agents**:

| gate | total actual invocations | per agent |
|---|---|---|
| `pytest -n auto` (full parallel suite) | **0** | 0.00 |
| `pytest` (any, scoped subsets) | 190 | 0.31 |
| `mypy` | 71 | 0.11 |
| `ruff` | 75 | 0.12 |
| `vulture` | 20 | 0.03 |

When agents did run tests, they ran narrowly scoped subsets (e.g.
`pytest tests/test_keepers/test_composite.py tests/test_scripts/test_keeper_rankings.py -q`),
never `-n auto`. Agents verified findings by **reading** code and git history and running the
*specific* code path in question -- not by re-running the project's gate suite. The
"multiply the 153s suite across 5 agents" cost model did not occur in a single one of 620
agents.

The one place the full suite *is* paid is the loop's own **Final verification** step, run
**once per loop by the orchestrator** at the end -- a single run, not an N-way multiply, and
not something a per-agent cache would touch.

Conclusion: **do not build the shared-gate cache.** Its expected saving on this workload is
near zero. (A minor variant -- have the parent compute `git diff` + a changed-file snapshot
once and inject it, so agents skip re-deriving the diff -- is defensible, but the tokens are
cache-served, so prioritize it below everything in Findings 3-5.)

## Finding 3 -- H6 (cross-cutting churn) is the biggest lever, and it is detectable

Of 8 multi-pass runs, the shape splits cleanly in two, and the split is visible in the
**finding locations**, not just the counts:

**Churn runs** -- consecutive passes flag the *same files* and the finding count does not
decay (it oscillates or rebounds):

| run | passes | reported sequence | sticky file(s) | tokens |
|---|---|---|---|---|
| `feat/277-decompose-luck` (#280) | 10 | 8,2,7,2,6,1,2,2,2,1 | `keeper_rankings.py` 9/10, `composite.py` 5/10 | **12.1M** |
| `fix/282-bare-name-keying` | 3 | **7,2,6** (rebound) | `keeper_rankings.py` 3/3 (Jaccard 1.00) | 3.5M |
| `(diff)@431a3ffb` (keeper_value) | 2 | 12,5 | `keeper_value.py` 2/2 (Jaccard 1.00) | 2.5M |

**Decay runs** -- files move pass to pass and the count falls to zero:

| run | passes | reported sequence | consecutive file-Jaccard |
|---|---|---|---|
| `(diff)@f778fd19` (breakout) | 3 | 1,1,0 | 0.00, 0.00 |
| `fix/273-per-season-positions` | 2 | 7,1 | 0.33 |
| `(diff)@b8f1c550` (keeper_trades) | 6 | 2,7,4,3,4,0 | mostly 0.00-0.50, ends 0 |

**The discriminator:** a trigger of the form *">= 2 consecutive passes report findings whose
file sets overlap AND the finding count did not drop"* fires on exactly the three churn runs
and stays silent on the decay runs. It is cheap to compute (Jaccard over the `result.findings`
file:line sets, which are already in the workflow record) and it is the single highest-leverage
change for the worst-case runs.

**Mechanism (this is the part that should transfer to Opus 5).** The loop does *local,
salience-driven* review: each pass finds the single most salient issue, a narrow local fix is
applied, and the fix shifts the problem to the adjacent facet, which the next 25-30 min pass
then "discovers." A cross-cutting invariant -- e.g. "what happens when a keeper pool / family
/ board is empty, absent, all-NaN, or half-joined?", which on `feat/277` spanned `composite`,
`_qualified_families`, `_require_mandatory_families`, `projected`, `build`, three live
commands, and four diagnostics -- is exactly the shape that gets discovered one facet per
pass. **The diff shape predicts the failure mode: independent bugs decay and terminate;
cross-cutting invariants churn.** `keeper_rankings.py` was the sticky epicenter across two
separate runs, which is itself a signal that the keeper subsystem carries an
under-specified invariant (tracked separately: keeper roster id join, #284/#230/#269).

**Proposed remedy (H6):** when the cluster trigger fires, the loop should stop asking "what
is the next issue" and issue one pass of the form *"enumerate every state this invariant can
be in across {sticky files}, state the intended behavior of each, and fix them together."*
This encodes into the skill the discipline that currently lives only in operator judgment and
in the `feedback_loop_review_cross_cutting_churn` memory -- which explicitly notes the
operator did not self-correct into "step back and design the whole invariant" on `feat/277`.

## Finding 4 -- H4 (fix-diff scoping) is supported by the same data

In the churn runs the finding source had migrated from the original diff to the *last fix* by
the middle passes (on `feat/277`, `keeper_rankings.py` -- the file the fixes kept touching --
was flagged in 9 of 10 passes). Reviewing the whole `main...HEAD` diff every pass re-reads
code that has been stable for many passes. **Remedy:** after pass 1, scope the review to
`git diff <last-reviewed-sha>..HEAD` plus that delta's blast radius. This narrows the read
surface (Finding 1's cost driver) and naturally concentrates review where churn actually
lives. Complementary to H6, cheaper, lower risk.

## Finding 5 -- H3 (decay early-exit) is real but a policy change

16 of 40 passes reported <= 2 findings yet still cost a median of **920k tokens** (range
488k-1366k) each -- the flat-cost-vs-decaying-yield problem, quantified. The decay-shaped
runs pay full price for their zero/near-zero confirming tail.

**But:** the current skill makes "cost is never a stop condition" a load-bearing principle,
precisely because a zero-finding pass is what *proves* convergence and a pass that found a bug
is evidence there are more. A decay early-exit weakens that guarantee. It is worth doing --
but as a deliberate, user-owned decision, e.g. "after a pass whose only findings were
low-severity and non-overlapping with the prior pass, one cheap confirming pass (lower effort,
fewer finders) may substitute for a full one." Note it would **not** have helped the churn
runs at all -- they never decayed -- which is exactly why H6, not H3, is the headline.

## Finding 6 -- Agent count is an unnamed lever

xhigh spins ~6 finders + ~6 verifiers regardless of how mature the run is; tail passes with
0-2 findings still ran 10-15 agents. Tapering finder/verifier count (or dropping to `high`)
on late, low-yield passes is a direct token saving orthogonal to H3/H4/H6. loop-review
currently forwards a fixed effort for the whole run; letting it *decay the effort* as a run
matures is a small change with a real payoff. (The verify stage is doing genuine work --
passes routinely refuted 1-4 plausible-but-wrong candidates -- so cut finder breadth before
cutting verification depth.)

## Finding 7 -- /simplify (the first half of every pass) earns little of its keep

`/simplify` is the other half of each pass and was the largest unmeasured corner; it is now
measured. It fans out into **4 read-only reviewer agents** (reuse / simplification /
efficiency / altitude). They do **not** edit -- across **140 simplify agents, 139 applied
zero edits**; they report recommendations and the loop orchestrator applies them in the main
session. So the right layer to judge them is the orchestrator's edits per invocation, not the
sub-agent transcripts.

Measured across all main-session transcripts:

| signal | `/simplify` | `/code-review` (runs right after, same pass) |
|---|---|---|
| invocations applying >= 1 **code** edit | **73%** (19/26) | 96% (27/28) |
| pure no-ops (0 code edits) | **27%** | 4% |
| mean code-edits / invocation | 3.3 | 8.8 |
| total code-edits driven | 86 | 246 |
| reviewer agents surfacing a substantive rec | 62% of 140 | -- |
| cost | ~11M totalTokens-equiv (~20% of machinery) | ~44M |

(Cost method: simplify's billed-proxy `cache_creation + input + output` = 27.5M against 196M
`cache_read` -- as cache-served as code-review. For the one code-review workflow where both
numbers exist, billed-proxy 3.24M vs the workflow's own `totalTokens` 1.30M, a 0.40 ratio;
applying it to simplify's 27.5M gives ~11M `totalTokens`-equivalent, which also matches the
#280 case study's ~250-350k/pass estimate.)

So they are **not** dead weight -- 86 real cleanups, a recommendation ~62% of the time, cheap
in output. But the case against them, on the same data:

1. **Lower yield than the tool that immediately follows.** `/code-review` runs right after
   `/simplify` every pass and does correctness *and* cleanups, out-producing it 2.7x per
   invocation. #274's own observation: simplify's late findings were "mostly absorbed into the
   review agent's."
2. **A quarter of invocations (27%) are pure no-ops** (vs 4% for code-review) -- paying the
   4-agent fan-out to find nothing.
3. **Documented harm, not just absence of help.** On `feat/277` pass 7, a simplify **altitude**
   recommendation relocated an empty-pool guard into `_qualified_families` and crashed
   `--study`/`--backtest` early-season, reverted the next pass. Simplify *contributed* to the
   churn there.
4. Runs the full 4-dimension fan-out every pass regardless of run maturity.

**Remedy (for the Opus 5 loop), in confidence order:** cut `altitude` first (the one dimension
with a documented regression and the most architectural-opinion surface); run `/simplify` once
at pass 1 (freshest diff, cleanups most valuable) rather than every pass, which kills the
no-op tail and the late-pass overlap with code-review; or drop `/simplify` entirely and let
code-review's cleanup dimension carry quality (it already absorbs most of it) for a real ~20%
token saving at small yield loss. Keep `reuse` if you keep any dimension -- it is the one most
orthogonal to code-review's correctness focus (samples show it catching genuine
drop-in-existing-helper opportunities).

**Confidence caveat:** the "harm" is one well-documented case plus a structural overlap
argument, **not** a measured harm *rate*. Quantifying reverted-simplify-edits across all runs
would need per-edit diff tracking not done here.

---

## Recommendation, prioritized

Restructure the loop; keep it (do not replace with a single-shot `/code-review ultra`, and do
not weaken the verified-only prompting in Finding 2/H2 -- that is where the findings come
from). In priority order:

1. **H6 cluster-detection -> theme-enumeration escalation.** Highest leverage; attacks the
   12.1M / 3.5M churn runs; empirically detectable from data already in the workflow record.
2. **H4 fix-diff scoping after pass 1.** Cheap, complementary, reduces the Finding-1 read cost.
3. **Do NOT build H1 (shared gate cache).** Measured near-zero saving. Document it as
   rejected so it is not re-proposed.
4. **H3 decay early-exit + Finding-6 effort taper.** Real savings on decay-shaped runs;
   changes the "cost never stops the loop" semantics -- make it an explicit user decision.
5. **Trim `/simplify` (Finding 7).** Cut `altitude`, or run simplify once at pass 1, or drop
   it entirely. ~20% token saving at small yield loss, since code-review absorbs most of its
   territory. Also a user-facing behavior change, so pair it with the decision in item 4.

Expected effect: the two churn runs measured here (15.6M tokens combined) are the ones H6+H4
target directly; the decay-run tails (H3/F6) are a broad but smaller trim; and dropping or
front-loading `/simplify` (F7) removes ~20% of the machinery with little yield cost.

---

## Measurement gap (be honest about it)

The original gap -- `/simplify` was not workflow-backed and so not in the code-review census --
is now closed in **Finding 7** (measured from the simplify sub-agent transcripts and the
orchestrator's per-invocation edits). The residual gap is narrower: the **harm rate** of
applied simplify edits is not quantified (only one reverted case is documented); measuring it
would require per-edit diff tracking against the final commit, not done here.

---

## What to re-measure on Opus 5 (the upgrade checklist)

When this machine moves to Opus 5, re-run the appendix extraction against the first handful of
real Opus 5 loop-review runs and check whether the *mechanism* still holds and the
*magnitudes* have moved:

1. **Cost shape:** is output still tiny and input/context still cache-dominated? Re-run the
   tool census. If Opus 5 reasons more per turn, tokens/pass may rise even with fewer agents.
2. **H1 sanity re-check:** do Opus 5 agents still avoid the full gate suite (still ~0
   `pytest -n auto`)? If Opus 5 verifies harder by *executing* (the #274 H2 hypothesis said
   4.x already trends this way), gate/script execution could become a real cost -- and H1
   would move from "rejected" to "worth a shared-artifact cache." This is the one finding most
   likely to flip on Opus 5. Watch it.
3. **Churn detectability:** does the file-overlap discriminator still separate churn from
   decay? The threshold (">=2 consecutive overlapping passes, count not dropping") may need
   retuning if Opus 5's per-pass finding counts change.
4. **Pass count to convergence:** does Opus 5 converge in fewer passes on the same diff shape?
   If so, the flat-cost tail (H3) shrinks and the priority order may shift.

---

## Appendix -- extraction scripts

Run against `BASE = ~/.claude/projects/C--Users-HartAlden-FantasyBaseball`.

**A. Per-workflow cost/finding table** (walk `*/workflows/wf_*.json`):

```python
import json, glob, re
BASE = "<...>/C--Users-HartAlden-FantasyBaseball"
for wf in glob.glob(BASE + "/*/workflows/wf_*.json"):
    d = json.load(open(wf, encoding="utf-8", errors="replace"))
    res = d.get("result") or {}; stats = res.get("stats") or {}
    args = d.get("args") or ""
    branch = (re.search(r'branch (\S+)', args) or [None, "(diff)"])[1] if "branch " in args else "(diff)"
    files = [(f.get("file"), f.get("line")) for f in (res.get("findings") or []) if isinstance(f, dict)]
    print(d["timestamp"], d.get("workflowName"), branch, d.get("agentCount"),
          d.get("totalTokens"), stats.get("reported"), stats.get("refuted"),
          d.get("defaultModel"), [f[0] for f in files])
```

**B. Actual gate invocations per agent** (walk `*/subagents/workflows/wf_*/agent-*.jsonl`,
find `tool_use` blocks with `name in {Bash, PowerShell}`, and regex the `input.command`
against `pytest -n`, `\bpytest\b`, `\bmypy\b`, `\bruff\b`, `\bvulture\b`). This is what
produced the Finding-2 table; the file-mention shortcut (`grep -l pytest agent-*.jsonl`)
overcounts badly because the injected CLAUDE.md names every gate -- parse the Bash command,
do not grep the transcript.

**C. Pass-to-pass churn** (group consecutive same-branch/near-in-time workflows into runs;
per run, take each pass's `result.findings` file set; report the consecutive-pass Jaccard and
the set of files appearing in >= 2 passes). Churn = high overlap + non-dropping count; decay =
low overlap + count -> 0.

**D. /simplify measurement** (Finding 7). `/simplify` is NOT workflow-backed, so it leaves
plain Agent transcripts, not `wf_*.json`:

- Identify simplify sub-agents by their `subagents/*.meta.json` `description` matching
  `(reuse|simplif|efficien|altitude)` (the 4 dimension reviewers). Count `Edit`/`Write`/
  `MultiEdit` `tool_use` in each transcript to confirm they apply ~nothing themselves (they
  report; the orchestrator applies).
- For real application, parse each MAIN session transcript (`BASE/*.jsonl`) in order: track
  `Skill` `tool_use` calls (`input.skill` in {`simplify`, `code-review`}) as phase markers and
  bucket the main agent's own `Edit`/`Write`/`MultiEdit` calls (`.py`, non-`.md`) to the most
  recent phase. That yields the per-invocation code-edit distribution (73% vs 96% apply rate).
- For cost, sum `usage.{cache_creation_input_tokens, cache_read_input_tokens, input_tokens,
  output_tokens}` over the simplify transcripts; billed-proxy = cc + input + output. Convert to
  `totalTokens`-equivalent with the ~0.40 ratio from any code-review workflow where both the
  proxy and the reported `totalTokens` exist.

---

## References

- GitHub issue **#274** (this document's home).
- `docs/superpowers/loop-review-case-study-pr280-2026-07-31.md` -- the #280 narrative
  (data point that this census generalizes).
- Skill under study: `~/.claude/skills/loop-review/SKILL.md` (user-global; not in this repo).
- Memory: `feedback_loop_review_cross_cutting_churn` -- the operator-discipline note H6 would
  move into the skill.
