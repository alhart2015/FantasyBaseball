---
name: group-review
user_invocable: true
description: Use when you want a comprehensive codebase review from four specialist perspectives - baseball expertise, data pipeline quality, software engineering, and code maintainability. Spawns baseball-scout, data-scientist, software-engineer, and code-maintainability agents in parallel, then consolidates findings into a severity-ranked report.
---

# Group Review

Dispatch four specialist agents to review the codebase in parallel, then consolidate their findings into a single severity-ranked report.

## Agents

| Agent | Focus | What they catch |
|-------|-------|-----------------|
| **baseball-scout** | Baseball reality | Unrealistic projections, bad assumptions about player performance, category scarcity errors, ADP/value mismatches |
| **data-scientist** | Pipeline correctness | Silent data loss, NaN propagation, rate stat blending errors, pool contamination, methodological flaws |
| **software-engineer** | Code quality | Performance bottlenecks, correctness bugs, test coverage gaps, race conditions, reliability risks |
| **code-maintainability** | Structural health | Dead indirection, stringly-typed data, leaky abstractions, coupling, missing normalization boundaries, bug factories |

## Execution

1. **Spawn all three agents in parallel** using the Agent tool with `subagent_type` set to each agent name. Each agent gets a prompt telling it to do a full review of its domain area. Include any user-provided focus areas in each prompt.

2. **Wait for all three to complete.** Do not synthesize partial results.

3. **Consolidate findings** into a single report with four severity tiers:

### Severity Tiers

**CRITICAL** — Will affect correctness of projections or cause crashes on draft day.
- Bugs that produce wrong player valuations or rankings
- Crashes or unhandled exceptions in draft-day code paths
- Data pipeline errors that silently corrupt results
- Projection issues that would lead to clearly bad draft picks

**MEDIUM** — Opportunities to improve projection quality or performance.
- Statistical methodology improvements that would produce better valuations
- Performance bottlenecks that slow interactive draft or simulation
- Test coverage gaps for important code paths
- Projection assumptions that are defensible but suboptimal

**LOW** — Cosmetic, cleanup, or nice-to-haves.
- Code style or organization improvements
- Minor edge cases unlikely to trigger in practice
- Dead code or unused variables
- Documentation gaps

**SUGGESTED FEATURES** — Ideas from any expert that would improve the project.
- New capabilities any of the three experts think would add value
- Enhancements to existing features
- Data sources or methodologies worth incorporating

## Report Format

```
## Group Review — [date]

### CRITICAL
1. [Agent: baseball-scout] Finding title
   Details...

2. [Agent: data-scientist] Finding title
   Details...

### MEDIUM
...

### LOW
...

### SUGGESTED FEATURES
...

### Stats
- baseball-scout: N findings (X critical, Y medium, Z low, W features)
- data-scientist: N findings (...)
- software-engineer: N findings (...)
- code-maintainability: N findings (...)
```

## Deduplication

When multiple agents flag the same issue from different angles, merge into a single finding and credit all agents. Example: if the data-scientist flags "VONA pool truncation biases scores" and the software-engineer flags "VONA computed on subset instead of full pool" — that's one finding, not two.

## Mediator Role

After presenting the consolidated report, briefly note any findings where agents would likely disagree (e.g., the baseball-scout thinks a projection is fine but the data-scientist flags the methodology that produced it). Call out the tension and give your assessment of who's right.
