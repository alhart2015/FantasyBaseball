## Group Review — 2026-03-26

### CRITICAL

**1. [baseball-scout, data-scientist, software-engineer] Leverage weight explosion when no pitchers are drafted**

When a team has zero pitchers (common after keeper rounds — Hart has 3 hitter keepers), ERA and WHIP get `raw = 1.0 / epsilon = 1000.0` while counting stats at zero are capped at `10.0`. This creates a **642:1 pitching-to-hitting leverage ratio**, causing the recommender to draft mid-tier SPs over elite hitters after keeper rounds. In VONA mode, hitter urgency is effectively zeroed out, making the bug worse. The existing test (`test_none_pitching_totals_get_high_weight`) only asserts pitching > hitting, missing the severity entirely.

**Fix:** `balance.py:121` — cap `None` raw weight to `10.0` (same as counting stats). Strengthen test to assert ratio < 20.

**2. [baseball-scout] Config strategy/scoring_mode contradicts validated analysis**

`league.yaml` says `strategy: three_closers` + `scoring_mode: var`, but the team's own adversarial review (2026-03-24) validated `two_closers` + `vona` after finding 4 bugs that invalidated prior results. The config was either never updated or reverted. Running the wrong strategy on draft day would be costly.

**Fix:** `config/league.yaml:9-10` — change to `strategy: two_closers` and `scoring_mode: vona`.

**3. [baseball-scout] "Util" position not recognized for UTIL roster slot**

Yahoo caches Ohtani (batter) with positions `["Util"]`, but `HITTER_POSITIONS` doesn't include `"Util"`, so `can_fill_slot(["Util"], "UTIL")` returns `False`. Ohtani can't be assigned to any active roster slot. Currently he's kept by another team so it doesn't wreck Hart's draft, but it affects opponent roster modeling in simulations.

**Fix:** `positions.py:3` — add `"Util"` to `HITTER_POSITIONS`.

---

### MEDIUM

**4. [baseball-scout] Backfill mechanism inflates pitcher value and double-counts injury risk**

Backfill adds waiver-quality IP to any starter below 178 IP (1,437 pitchers affected) and waiver-quality AB to hitters below 600 AB (1,832 hitters affected). This double-counts injury risk since Monte Carlo already models playing time loss. Worse, rate stat dilution punishes high-AVG hitters: Yordan Alvarez drops 11 points of AVG from .295 to .284. The 15 IP / 50 AB thresholds are too tight.

**Recommendation:** Raise thresholds (SP from 15→30, hitter from 50→100), cap backfill at 25% of projected stats, or apply only to counting stats and leave AVG/ERA/WHIP at original values.

**5. [baseball-scout] "Oopsy" projection system runs hot, may distort blend**

Oopsy consistently projects higher HR and AVG than all four established systems (Judge: +10 HR, +21 AVG pts vs Steamer). At equal 20% weight, it pulls the blend upward. ATC already double-counts Steamer/ZiPS since it's a blend of those systems. Adding a fifth hot system compounds this.

**Recommendation:** Reduce Oopsy weight to 10% or remove, unless it has a verified accuracy track record.

**6. [software-engineer] Simulation `iterrows()` bottleneck — 62% of runtime**

`simulate_draft.py:453-467` — opponent pick loop uses `adp_board.iterrows()` for ~180 picks per simulation, consuming 4.0 of 6.5 seconds. Converting to `adp_board.to_dict('records')` gives ~2.4x overall speedup. For `compare_strategies.py` (1120 simulations): saves ~25 minutes.

**7. [software-engineer] `_lookup_pid` searches by name, not player_id**

`strategy.py:728-734` — when two players share a name (e.g., Max Muncy LAD vs ATH), returns first DataFrame match. Should use `player_id` from recommendation dict when available.

**8. [baseball-scout] three_closers deadlines force negative-VAR closers**

Deadlines at rounds 5/9/13 force drafting closers at ADP 80-120+ where most have negative VAR (-0.54 to -2.32). Only matters if Finding #2 isn't fixed, since the validated strategy is two_closers.

---

### LOW

**9. [data-scientist] PA not adjusted during hitter backfill** — PA stays at pre-backfill value while AB is increased, creating inverted PA<AB. Cosmetic only — PA is never used in calculations.

**10. [baseball-scout] Pitcher correlation matrix has suspiciously uniform values** — ER-BB-H_allowed pairwise correlations are all exactly 0.729, which is unlikely from real calibration data. Modest simulation impact.

**11. [software-engineer] `_scarcity_cache` uses `id(board)` as cache key** — `recommender.py:428`. Memory address reuse could return stale data after GC. Safe in practice since board lives for entire session.

**12. [data-scientist] SV threshold cliff at 20 SV** — Binary closer classification creates sharp cliff (19.1 SV = SP/RP, 20.6 SV = closer). Design tradeoff, not a bug.

**13. [data-scientist] ADP infinite values for uncached players** — Handled correctly downstream (appear last in ADP-sorted lists).

**14. [software-engineer] Module-level state in `state.py` leaks across tests** — `_current_version` persists between tests. `reset_version_state()` exists but must be called manually.

**15. [software-engineer] Dependencies not pinned to specific versions** — Uses `>=` rather than `~=`. Appropriate for single-user app but fragile.

**16. [data-scientist] No config validation against schema** — Missing/malformed YAML fields silently use defaults.

---

### SUGGESTED FEATURES

**17. [baseball-scout] Cap backfill as percentage of projected stats** — Even with higher thresholds, a 100 IP pitcher shouldn't receive 78 IP of replacement pitching. Cap at 25%.

**18. [data-scientist] Log position cache miss rate at board build time** — Count default-position fallbacks and warn the user.

**19. [software-engineer] Extract `balance._hitters` access into a method** — Strategies directly access internal `_hitters` list. Add `get_projected_avg()` to decouple.

**20. [data-scientist] Update Yahoo position cache before draft** — Current cache has 938 entries vs 3699 board players. Emmanuel Clase defaults to `[SP]` instead of `[RP]`.

---

### Stats
- baseball-scout: 10 findings (2 critical, 4 medium, 2 low, 2 features)
- data-scientist: 8 findings (0 critical, 1 medium, 5 low, 2 features)
- software-engineer: 10 findings (2 critical, 3 medium, 4 low, 1 feature)

### Tensions

**Backfill value vs. double-counting:** The baseball scout flags backfill as distorting player values (penalizing high-AVG hitters, inflating SP). The data scientist verified the backfill math is *correct* — rate stats are properly recomputed from components. Both are right: the math is correct but the methodology is flawed. The Monte Carlo simulation already models injury-driven playing time loss, so backfill pre-adjusting projected stats before SGP is genuinely double-counting. **Scout wins this one.**

**SV variance (0.900) and closer valuation:** The scout notes the high SV variance may *systematically undervalue closers* in simulation, which could explain why simulations favor fewer closers. Meanwhile the scout also validates two_closers as the right strategy. There's tension here — if SV variance is too high, the simulation that validated two_closers may itself be biased against closers. Worth investigating but not blocking.

**Oopsy projections:** Only the scout flagged this. The data scientist verified the blending math is correct regardless of input quality. The scout's concern is about *input quality*, not pipeline correctness — a properly blended bad projection still produces a bad result. **This deserves attention** — if Oopsy doesn't have a verified track record, it shouldn't get equal weight with Steamer/ZiPS.
