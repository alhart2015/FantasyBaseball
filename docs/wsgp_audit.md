# wSGP Audit: Can We Remove It?

**Date:** 2026-04-16
**Goal:** Remove wSGP, replace display with deltaroto where appropriate.

## TL;DR

wSGP is deeply embedded as a **computation engine** (lineup optimization, waiver filtering, trade ranking). It cannot be fully removed — deltaroto needs a swap context (drop X, add Y, recompute standings) and can't serve as a per-player objective function. But wSGP can be **hidden from the UI** and replaced with deltaroto in display contexts.

---

## wSGP as Computation Engine (cannot remove)

These use wSGP as the **objective function or filter**, not just display:

| Use | File | Role |
|-----|------|------|
| Lineup optimization | `lineup/optimizer.py`, `lineup/team_optimizer.py` | Hungarian algorithm objective — maximizes total wSGP |
| Waiver scanning | `lineup/waivers.py` | Filters FAs by wSGP > 0, sorts swaps by wSGP gain |
| Trade evaluation | `trades/evaluate.py` | Filters trades by positive wSGP gain, ranks by magnitude |
| Player classification | `lineup/player_classification.py` | Median wSGP splits "core" vs "trade_candidate" vs "droppable" |
| Draft recommender | `draft/recommender.py` | Scores candidates by wSGP when leverage available |
| Transaction scoring | `analysis/transactions.py` | Net wSGP = add_wsgp - drop_wsgp |

**Why deltaroto can't replace these:** DeltaRoto is an O(n) team-level operation per candidate (needs full roster + projected standings + swap context). wSGP is a fast per-player score that works as an objective function for Hungarian algorithm optimization and for sorting/filtering large candidate pools. They solve different problems: wSGP scores individual players, deltaroto scores roster decisions.

---

## wSGP as Display (replaceable)

Places wSGP is **shown in the UI**:

### 1. Lineup page (`templates/season/lineup.html`)
- wSGP column for your hitters and pitchers
- Dual wSGP columns (them/you) for opponent lineup
- **Question:** Replace with deltaroto? Unclear baseline — deltaroto needs a swap context (bench them? drop them?). Could just remove.

### 2. Players browse (`templates/season/players.html`)
- wSGP column, sortable
- wSGP in comparison panel
- **Question:** Replace with deltaroto? Would need "swap this FA for worst at position" computation per player. Expensive but doable.

### 3. Roster audit (`templates/season/roster_audit.html`)
- Player wSGP, FA wSGP, wSGP gap columns
- DeltaRoto is **already shown alongside**
- **Question:** Remove wSGP columns, keep only deltaroto? Seems cleanest.

### 4. Transactions (`templates/season/transactions.html`)
- "Net wSGP" column, add/drop wSGP values
- **Question:** Replace with deltaroto? Would need projected standings at time of each transaction — may not be available historically.

### 5. Waivers/Trades (`templates/season/waivers_trades.html`)
- "X.XX wSGP" gain label on trade candidates
- **Question:** Replace with deltaroto gain? Expensive but most accurate.

### 6. API responses (`web/season_routes.py`)
- `api_players_ros()` returns wSGP in JSON (line ~614)
- `api_player_browse()` returns wSGP in JSON (line ~695)
- **Question:** Keep in API for internal use but stop displaying? Or remove from API too?

---

## Decisions Needed

For each display context, choose one:
- **Replace with deltaroto** — compute and show deltaroto instead
- **Just remove** — hide wSGP, don't replace with anything
- **Keep** — leave wSGP visible (if it turns out to be useful here)

| Context | Replace w/ deltaroto | Just remove | Keep | Status |
|---------|---------------------|-------------|------|--------|
| 1. Lineup page — your roster | ✓ |  |  | shipped — `delta_roto` col |
| 2. Lineup page — opponent |  | ✓ |  | shipped — dropped wSGP them/you, show single SGP col |
| 3. Players browse |  | ✓ |  | shipped (template clean; API still returns `wsgp` — see #7) |
| 4. Roster audit |  | ✓ |  | shipped — deltaroto was already alongside, wSGP cols removed |
| 5. Transactions | ✓ |  |  | shipped — ΔRoto per txn, net ΔRoto per team |
| 6. Waivers/Trades | ✓ |  |  | shipped (commits a553fc9, b0e0c7e) |
| 7. API responses | ? | ? | ? | **open** |
