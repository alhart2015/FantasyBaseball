## Group Review — 2026-03-25

### CRITICAL

**1. [baseball-scout] Starting Pitcher Systematic Overvaluation**
The unified P replacement level (SGP=7.97) is 2.7-3.8 SGP below hitter position replacement levels. Result: 16 of the top 20 available players post-keepers are pitchers. Bryan Woo ranked #1 overall (VAR=6.15) ahead of every available hitter. Following the raw VAR board would produce 3+ SPs in the first 5 rounds — a losing roto strategy since pitching is more replaceable via streaming. VONA partially corrects this (Woo's VONA drops to 0.25) but the additive formula still lets high-raw-SGP pitchers outscore moderate hitters.

**Files:** `sgp/replacement.py:16-28`, `sgp/var.py`
**Mitigation in place:** VONA mode + strategy constraints. But VAR board still drives many strategy functions.

---

### MEDIUM

**2. [data-scientist, software-engineer] `anti_fragile` strategy defined but not registered in STRATEGIES dict**
`pick_anti_fragile` is fully implemented (line 765) and documented, but missing from the `STRATEGIES` dict (line 820). `--strategy anti_fragile` would crash with `KeyError`. Existing sim results in `data/sim_results/` must have been generated with an older codebase version.

**Fix:** Add `"anti_fragile": pick_anti_fragile,` to `STRATEGIES` dict. One line.

---

**3. [software-engineer] `pick_avg_hedge` ignores `scoring_mode` parameter**
Calls `get_recommendations()` directly instead of `_get_recs()`, never passing `scoring_mode`. When configured with VONA, silently falls back to VAR scoring.

**Fix:** Add `scoring_mode=kwargs.get("scoring_mode", "var")` to the call, or refactor to use `_get_recs()`.

---

**4. [data-scientist, software-engineer] NaN propagation risk in scoring pipeline**
`scoring.py:project_team_stats` does not guard against NaN inputs — one NaN poisons the entire team total. `sgp/player_value.py:calculate_player_sgp` also propagates NaN silently. The blending pipeline uses `fillna(0)` so projections are clean, but Yahoo API data (used by lineup optimizer) could introduce NaN.

**Fix:** Coerce None/NaN to 0 in `project_team_stats` and `calculate_player_sgp`.

---

**5. [data-scientist] Two different `score_roto` implementations with inconsistent tie-breaking**
`scoring.py` uses fractional tie-breaking (ties for 3rd both get 3.5). `draft/projections.py` uses simple sequential ranking. Draft simulation uses the former; Monte Carlo uses the latter.

**Fix:** Consolidate to single implementation.

---

**6. [data-scientist, baseball-scout] "Util" position not in HITTER_POSITIONS**
Yahoo returns "Util" for DH-eligible players, but `positions.py:HITTER_POSITIONS` only has "DH". `is_hitter(["Util"])` returns False. Currently only affects Ohtani (a keeper), so impact is limited. Would break for any future DH-only non-keeper.

**Fix:** Add `"Util"` to `HITTER_POSITIONS` or map "Util" → "DH" on load.

---

**7. [baseball-scout] No injury/volatility risk adjustment**
Projections treat IP/PA as deterministic. deGrom (161 IP projected after missing 2 seasons), Sale (154 IP at age 37), and other health risks get the same per-IP valuation as durable pitchers. Both Steamer and ZiPS partially discount via playing time, but the model applies no additional variance penalty.

---

**8. [baseball-scout] Oopsy projection system inflates blend**
Oopsy projects 12.8% more HR and 16.1% more RBI than Steamer across all qualified hitters. With equal 20% weight, it pulls every player's blend upward. If Oopsy is ceiling-oriented, it should get lower weight (suggested: 10%, redistribute to ATC).

---

**9. [software-engineer] `blend_projections()` startup bottleneck — 6.3s**
Per-group pandas loop creates ~49,000 intermediate Series objects. Vectorized `groupby.sum()` with pre-multiplied weights would reduce to <0.5s.

---

**10. [software-engineer] Simulation opponent picks use `iterrows()` — O(n*d)**
Each opponent pick scans from the top of a 3,669-row DataFrame. 0.54s per simulation; pre-sorted list with pointer measured 6,154x faster. Saves ~54s per 100-sim batch.

---

**11. [software-engineer] `simulate_draft.py` writes to live `draft_state.json`**
Uses non-atomic `json.dump` directly to `data/draft_state.json`. If run during a live draft, it would overwrite live state, corrupt version counter, and break the dashboard.

**Fix:** Write to `data/sim_state.json` instead.

---

**12. [software-engineer] No config validation**
`load_config()` doesn't validate YAML values. `num_teams: 0` → division by zero. `sgp_denominators` with zero → `ZeroDivisionError`. Empty `roster_slots` → all replacement levels become 0. Missing keeper keys → `KeyError`.

---

### LOW

**13. [software-engineer] Empty player pool crashes `calculate_replacement_levels`**
When all players are drafted, `.apply()` on an empty DataFrame drops columns, causing `KeyError: 'total_sgp'`. Only triggers at absolute end of draft.

---

**14. [baseball-scout] Rate stat leverage fixed at 1.0 in draft balance**
AVG/ERA/WHIP get constant leverage weight regardless of how far the team is from target. A team batting .220 gets no extra urgency toward high-AVG hitters. Strategy-level AVG floors partially compensate.

---

**15. [software-engineer] AVG/ERA warning thresholds are useless**
AVG warning fires at `.260 * 0.6 = .156`. ERA warning fires at `3.60 * 1.67 = 6.0`. Neither will ever trigger during a real draft.

---

**16. [data-scientist] Monte Carlo rate stats invariant to performance noise**
When all counting stats are multiplied by the same `perf` factor, rate stats cancel out (ERA = ER*perf*9 / IP*perf). Quality variance only enters through the injury model.

---

**17. [baseball-scout, data-scientist] Pitcher Ohtani gets wrong positions**
Position cache maps "Shohei Ohtani" → ["Util"] (batter), but pitcher projection uses the same name. Pitcher Ohtani (VAR=-0.74) would never be drafted, so zero practical impact.

---

**18. [software-engineer] `_board_written` flag is dead code**
Declared in `state.py` but never set to `True`. Suggests incomplete feature.

---

**19. [baseball-scout] IF slots excluded from UTIL replacement calculation**
`replacement.py` excludes IF from positional count. Shifts UTIL replacement by 0.167 SGP. Negligible impact.

---

**20. [software-engineer] `read_state` has no retry on Windows `PermissionError`**
`_atomic_write` retries 5 times, but `read_state` returns `{}` immediately. Dashboard briefly shows empty state if read collides with write. Auto-recovers on next 2s poll.

---

### SUGGESTED FEATURES

**21. [baseball-scout] Track projection system disagreements**
When Steamer and ZiPS disagree by >20% on a counting stat, flag in the dashboard. Large disagreements often signal stale projections or trajectory information.

---

**22. [baseball-scout] Apply hitter/pitcher balance correction to VAR**
Either a `PITCHER_VAR_DISCOUNT` factor (~0.85) or separate SP/RP replacement levels to fix the systematic pitcher overvaluation.

---

**23. [baseball-scout] Responsive rate stat leverage**
Compare team rate stats to targets and scale leverage weight proportionally, instead of fixed 1.0.

---

**24. [software-engineer] Add tests for delta protocol critical path**
`compute_delta`, `serialize_board`, `read_board`, `write_board` are untested despite being draft-day critical.

---

**25. [baseball-scout] Leverage cap in in-season optimizer**
The defined `MAX_MEANINGFUL_GAP_MULTIPLIER = 3.0` is never used. Near-tied categories produce leverage values approaching 1000, dominating all lineup decisions.

---

**26. [baseball-scout] Minimum IP enforcement in lineup optimizer**
No check for league IP minimums. Extreme leverage scenarios could recommend benching all pitchers to protect ratios.

---

### Stats

| Agent | Total | Critical | Medium | Low | Features |
|-------|-------|----------|--------|-----|----------|
| baseball-scout | 14 | 1 | 5 | 4 | 4 |
| data-scientist | 10 | 0 | 4 | 4 | 2 |
| software-engineer | 14 | 0 | 5 | 7 | 2 |
| **Consolidated** | **26** | **1** | **11** | **8** | **6** |

*(After deduplication — 38 raw findings merged to 26)*

---

### Mediator Notes

**SP Overvaluation — scouts vs. engineers:**
The baseball-scout rates this CRITICAL; neither the data-scientist nor software-engineer flagged it. The data-scientist verified the math is correct (replacement levels compute properly), so there's no *bug*. The scout's concern is that correct math applied to a unified pitcher pool produces strategically bad rankings. **I side with the scout** — the model's purpose is to guide draft picks, and recommending 3 SPs in 5 rounds would produce a losing roto team regardless of mathematical correctness. However, VONA + strategy constraints already largely compensate, making this more of a MEDIUM in practice when using the recommended `two_closers+vona` configuration.

**"Util" position — severity disagreement:**
The data-scientist rated this CRITICAL; I've placed it at MEDIUM. Current impact is limited to Ohtani (a keeper). The fix is important for future-proofing but won't affect this year's draft. **The data-scientist is right that it should be fixed**, but the severity depends on timeline — it's a pre-next-season fix, not a draft-day emergency.

**Performance findings — relevance:**
The software-engineer found real bottlenecks (6.3s startup, 0.54s/sim). These matter for simulation batches but won't affect the interactive draft experience (40ms per pick is already fast). Prioritize the simulation speedups only if you plan to run large sweeps.
