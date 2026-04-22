# Projected standings: active-slot face-value rule

**Status:** Spec (2026-04-22)
**Scope:** `src/fantasy_baseball/scoring.py::_apply_displacement` and its tests.

## Problem

Projected standings (`ProjectedStandings.from_rosters` → `project_team_stats(roster, displacement=True)`) currently classify a rostered player as "IL" whenever **either** the Yahoo `status` is in `IL_STATUSES` **or** the `selected_position` is in `IL_SLOTS`. IL-classified players then run through the displacement algorithm: their full ROS stats are added, and the worst SGP-matched active player has their stats scaled by `max(0, active_pt − il_pt) / active_pt`.

This produces a surprising failure mode when a rostered player picks up an IL status while remaining in an active slot. Concretely: the user moved Juan Soto (status=`IL10`) into their OF slot. Current behavior, verified against live data:

- Soto is routed to the IL branch (status-based) even though he's in the active OF slot.
- Soto's ROS PA (570) is essentially equal to Byron Buxton's (569) — Buxton is the worst-SGP OF-eligible active hitter.
- `factor = max(0, 569 − 570) / 569 = 0`.
- **Buxton is zeroed out of the projected standings** — he contributes 0 HR, 0 R, 0 RBI, 0 SB, 0 AVG components despite being a starter.
- Net effect vs. the pre-IL state: the team's projected HR dropped ~28, which matches the user-reported "hitting stats went way down" after Soto's status flipped.

The root cause is that the IL classifier overrides the user's explicit signal. When a player is in an active slot, the manager is stating "I'm starting this player" — the standings model should honor that rather than re-interpret the slot based on injury status.

## Rule

Replace status-OR-slot classification with **slot-first** classification:

| `selected_position` | Yahoo `status` | Classification | Contribution |
|---|---|---|---|
| Active slot (C, 1B, 2B, 3B, SS, OF, IF, UTIL, SP, RP, P) | any | active | full ROS; may be a displacement target |
| IL / IL+ / DL / DL+ slot | any | IL | full ROS + triggers displacement on worst active match |
| BN slot | on-IL status (IL10/IL15/IL60/DTD/NA/O) | IL | full ROS + triggers displacement *(unchanged)* |
| BN slot | healthy | excluded | 0 *(unchanged)* |

Only the first row changes. Today, an active-slotted player with an IL status is treated as IL; under the new rule, that player is treated as active and counted at face value, with no displacement side effect.

Applies to **all teams** (user and opponents) — single code path, single rule.

## Rationale

- The active slot is the manager's explicit assertion about who is starting. Respect it.
- Opponents who leave a hurt star in an active slot are signaling they expect the player to produce; the model should reflect that, not discount it.
- Eliminates the "invisible collateral damage" where an active starter (e.g., Buxton) is zeroed out because an IL-statused teammate happens to share positional eligibility.
- Preserves the existing ROS-signal benefits of displacement for the cases where it's unambiguously correct: a player in an IL slot *will* return and *will* take playing time from someone; a BN player with an IL status is not currently producing.

## Non-goals

- No changes to `_is_il` / `_is_bench` helpers. They remain correct for their other callers.
- No changes to `roster_audit.py`'s separate `_is_il` (different concern — lineup-feasibility check).
- No changes to opponent-roster hydration, Yahoo status parsing, or projection blending.
- No UI/routing/template changes. The standings page reads the recomputed totals transparently.
- No historical snapshotting of projected standings (separate feature if wanted later).

## Implementation sketch

```python
# scoring.py::_apply_displacement, replacing lines 261–271
for p in roster:
    if not isinstance(p, Player):
        active.append(p)
        continue
    slot = p.selected_position
    if slot == Position.BN:
        if _is_il(p):
            il_players.append(p)    # BN + IL status → displace (unchanged)
        # else: healthy bench → excluded (unchanged)
        continue
    if slot in IL_SLOTS:
        il_players.append(p)        # IL slot → displace (unchanged)
        continue
    active.append(p)                # active slot → face value (CHANGED)
```

Rest of the function (displacement loop, output assembly) is unchanged.

Docstring updated to match the new rule.

## Test impact

`tests/test_scoring.py` has ~15 tests under `project_team_stats(..., displacement=True)`. Expected breakdown:

- **Unaffected:** tests that construct IL players with `selected_position=Position.IL` (or another IL slot). This is the dominant pattern.
- **Needs updating:** any test that constructs an IL player with an active slot + IL status and asserts displacement behavior. Those tests need to either:
  - Switch the construction to `selected_position=Position.IL` to retain the "IL displaces active" assertion, or
  - Be reframed as asserting the new face-value behavior.
- **New coverage to add:**
  - Active-slotted player with `status="IL15"` is NOT displaced and NOT a displacement trigger — contributes full ROS.
  - BN + IL status still triggers displacement (regression guard).
  - IL slot still triggers displacement (regression guard).

Full test audit happens during implementation; each affected case gets a per-test decision.

## Verification

- `pytest -v tests/test_scoring.py` — all tests pass.
- `pytest -v` — full suite passes.
- `ruff check .` and `ruff format --check .` — clean.
- `mypy` — clean for files in `[tool.mypy].files`.
- Manual: run `scripts/run_season_dashboard.py` locally, confirm user team's projected HR recovers by ~28 (Buxton unzeroed), confirm opponents' totals shift in the expected direction for teams with active-slotted IL-status players.

## Rollout

Single commit on a dedicated feature branch. No feature flag, no migration — the change only affects per-refresh recomputation, which overwrites the projections cache on the next run.
