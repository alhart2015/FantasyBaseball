## Group Review — 2026-04-05

### CRITICAL

**1. [baseball-scout] Config strategy/scoring_mode contradicts prior validated analysis**
`config/league.yaml` has `strategy: three_closers` and `scoring_mode: var`. Prior review determined `two_closers` + `vona` was correct after 4 bugs were found that invalidated earlier results. The `three_closers` strategy forces drafting closers at rounds 5/9/13 where most have negative VAR. This was flagged as Critical in the prior review and remains unfixed. *(Note: if the 2026 draft is already complete, this only affects future simulations — but it should still be corrected.)*

**2. [code-maintainability, data-scientist, software-engineer] Duplicated `simulate_season` — two diverging copies**
`simulation.py` (canonical, batch RNG draws, injury tracking) and `draft/projections.py` (per-player RNG, no injuries, no batched draws) implement the same logic independently. They've already diverged. A bug fix or calibration change in one won't propagate to the other. This directly risks inconsistent simulation results between the draft predictor and the post-draft Monte Carlo. *(Already noted in TODO.md under "Simplify suggested fixes".)*

**3. [software-engineer] No tests for the delta protocol (draft-day dashboard)**
`compute_delta`, `read_delta`, and the Flask `/api/state?since=<version>` endpoint have zero test coverage. This is the real-time update mechanism for the draft dashboard. A bug here silently serves stale data during a live draft. Missing tests: delta with None old_state, changed vs unchanged field detection, version inclusion, multi-version-behind fallback.

---

### MEDIUM

**4. [baseball-scout] SGP denominators wrong for this league**
The custom analysis (`data/analysis/custom_sgp_denominators.md`) shows AVG denominator is 0.005 vs league-specific 0.0025 (−50%), SB is 8 vs 11.9 (+49%), HR is 9 vs 7.3 (−19%). High-AVG hitters are systematically undervalued, speed is overvalued. The `sgp_denominators` override mechanism exists in `league.yaml` but isn't being used. Even a 50/50 blend with league-specific values would help.

**5. [baseball-scout] Equal-weight 5-system blend dilutes signal**
ATC is already a consensus of Steamer, ZiPS, and THE BAT — including it at equal weight double-counts those inputs. "Oopsy" has no established accuracy track record and runs systematically hot (+10 HR, +21 AVG points vs Steamer for Judge). A baseball-savvy blend: Steamer 30%, ZiPS 30%, ATC 25%, THE BAT X 15%, drop or heavily discount Oopsy.

**6. [baseball-scout, code-maintainability] "Util" missing from HITTER_POSITIONS**
`utils/positions.py` defines `HITTER_POSITIONS = {"C", "1B", "2B", "3B", "SS", "OF", "DH", "IF"}` — no `"Util"`. Yahoo returns `"Util"` for players like Ohtani. `is_hitter(["Util"])` returns `False`, misclassifying those players. Single-line fix.

**7. [baseball-scout, data-scientist] Backfill mechanism double-counts injury risk**
Backfill adds waiver-quality stats to players below healthy baselines (600 AB / 178 IP), but Monte Carlo already models playing time loss. This drops Yordan Alvarez's AVG by 11 points. The 600 AB threshold also penalizes high-walk hitters like Soto (536 AB = full healthy season for him). Data-scientist also notes the binary cliff: 549 AB gets no backfill, 550 AB gets full treatment. *(Already in TODO.md as a postseason item.)*

**8. [code-maintainability] Duplicated roto scoring implementation**
`trades/evaluate.py:compute_roto_points_by_cat` and `scoring.score_roto` both implement fractional tie-breaking roto scoring with different input formats. `draft/projections.py` adds a thin wrapper with a misleading `num_teams` parameter that's silently ignored.

**9. [code-maintainability] Pervasive stringly-typed `player_type`**
`"hitter"` appears in 36 comparison sites across 19 files. No enum, no exhaustiveness check. `calculate_player_sgp` silently returns `total_sgp = 0.0` if neither `"hitter"` nor `"pitcher"` matches. A `PlayerType` enum or `Literal["hitter", "pitcher"]` would turn silent no-ops into immediate errors.

**10. [code-maintainability] Rate stat formulas duplicated 16+ times across 7 files**
ERA (`er*9/ip`), WHIP (`(bb+h_allowed)/ip`), AVG (`h/ab`) computed inline in `simulation.py`, `balance.py`, `projections.py`, `replacement.py`, `scoring.py`, `pace.py`, `evaluate.py`. Classic bug factory. *(Already in TODO.md as "Extract rate stat utility functions".)*

**11. [code-maintainability, data-scientist, software-engineer] `_scarcity_cache` keyed by `id(board)` — fragile**
Python can reuse memory addresses after GC, serving stale cache data. Cache also doesn't vary by `roster_slots`. Three agents independently flagged this. Fix: content-based hash or pass scarcity order explicitly.

**12. [data-scientist] Management adjustment treats AVG/ERA/WHIP as counting stats**
`simulation.py:apply_management_adjustment` multiplies AVG by `factor` directly instead of adjusting H and AB components. Same for ERA/WHIP. Error is small (~1%) for typical factor ranges, but technically imprecise.

**13. [software-engineer] `build_player_lookup` and `_lookup_pid` performance**
`build_player_lookup` uses `iterrows()` on ~5700 rows (~30-50ms each call). `_lookup_pid` does O(n) DataFrame scan per call without passing `name_to_pid`. Several strategy helper functions rebuild lookups unnecessarily instead of passing the existing one through.

**14. [software-engineer] Missing tests: `serialize_board`, `get_roster_by_position`, `_filter_rosterable`**
Three untested functions in the draft-day code path. `_filter_rosterable` is critical for late-draft correctness (nearly-full roster). `get_roster_by_position` drives dashboard display. `serialize_board` is the one-time full board payload.

**15. [software-engineer] No tests for interactive draft flow**
`_handle_user_pick`, `_handle_other_pick`, `_get_player_input` in `run_draft.py` have no unit tests. Edge cases like out-of-range numbers, "skip", name collisions, and the "mine" keyword for traded picks are untested.

**16. [baseball-scout] Closer pool thinner than strategy assumes**
Only ~10 closers meet the 20 SV threshold in Steamer. `three_closers` needs 30 league-wide. Even `two_closers` needs 20, which doesn't exist. The `no_punt_cap3` strategy with dynamic SV monitoring is better suited.

**17. [code-maintainability] `calculate_player_sgp` and `calculate_weighted_sgp` accept 3 unrelated input types**
Both accept `HitterStats | PitcherStats | pd.Series` with 4 code paths each (8 total). Adding a new stat requires updating all branches. Standardize on the dataclass path.

**18. [code-maintainability] PITCHER_POSITIONS defined in two places**
`utils/positions.py` and `sgp/rankings.py` both define `{"P", "SP", "RP"}`. The latter should import from the former.

---

### LOW

**19. [code-maintainability] Inconsistent name normalization** — `trades/evaluate.py` uses `.lower()` instead of `normalize_name()`, `draft/search.py` reimplements its own `_norm()`. Accented names would fail to match.

**20. [code-maintainability] Strategies access `balance._hitters` directly** — Couples strategy layer to `CategoryBalance` internals. Add a `get_avg_components()` method.

**21. [code-maintainability] Hardcoded season date defaults in config** — `season_start: "2026-03-27"` will be stale next year. Either remove defaults or validate against `season_year`.

**22. [code-maintainability] No validation of `scoring_mode` or `strategy` config values** — Typos silently fall through to defaults.

**23. [code-maintainability] Ad-hoc Yahoo position normalization in `waivers.py`** — Should use centralized utility.

**24. [software-engineer] `CategoryBalance.get_totals()` inconsistency** — Returns 0 for W/K/SV but None for ERA/WHIP when no pitchers drafted, creating 100:1 weighting asymmetry in leverage calc.

**25. [software-engineer] `_pick_with_avg_floor` looks up by name, not player_id** — Name collisions possible. Recs should include `player_id`.

**26. [baseball-scout] Team IP constant inconsistency** — `player_value.py` uses 1400, `simulation.py` uses 1450. ~3.5% discrepancy in pitcher rate stat SGP.

**27. [baseball-scout] Pitcher correlation matrix uniform at 0.729** — ER-BB-H_allowed pairwise correlations are all exactly 0.729, which is unlikely from real calibration data. Modest simulation impact.

**28. [baseball-scout] Emmanuel Clase ghost across 3/5 projection systems** — Steamer and Oopsy project ~1 IP, 0 SV. Blend produces near-zero projection.

**29. [code-maintainability] Hardcoded user name `hart_*` in trade evaluation** — Should be `user_*`.

**30. [code-maintainability, software-engineer] Module-level mutable state** — `draft/state.py` globals, `_scarcity_cache` — fine for single-process but no reset for cache in tests.

**31. [data-scientist] No pipeline stage logging** — Row counts at each stage (blend → filter → backfill → SGP → VAR) would help debugging.

---

### SUGGESTED FEATURES

**32. [code-maintainability] Type-safe `TeamStats` container** — Centralize the 10 roto category totals with computed properties for rate stats. Eliminates inline formulas and provides validation.

**33. [code-maintainability] `DraftContext` dataclass** — Strategy functions all take `(board, full_board, tracker, balance, config, team_filled, **kwargs)`. A context object cleans up ~15 signatures. *(Already in TODO.md.)*

**34. [data-scientist] Auto-compute SGP denominators from player pool** — Standard deviation of projected team totals from Monte Carlo. Self-calibrating when league size or stat environment changes.

**35. [baseball-scout] "Stealth closer" alert for relievers projected 10-19 SV** — Robert Suarez (8-11 SV) falls below the threshold but could be a primary closer. Flag high-leverage relievers near the cutoff.

**36. [baseball-scout] Adjust projection weights based on track record** — Weight systems by historical accuracy (Steamer/ZiPS have published Brier scores, Oopsy doesn't).

---

### Stats
- baseball-scout: 11 findings (1 critical, 5 medium, 3 low, 2 features)
- data-scientist: 10 findings (0 critical, 3 medium, 5 low, 2 features)
- software-engineer: 18 findings (1 critical, 8 medium, 7 low, 2 features)
- code-maintainability: 17 findings (1 critical, 6 medium, 8 low, 2 features)
- Deduplicated from: 56 raw findings → 36 consolidated

### Mediator Notes — Agent Tensions

**Duplicate `simulate_season` severity**: Code-maintainability rated CRITICAL, data-scientist rated MEDIUM, software-engineer rated LOW (code health). Sided with code-maintainability — the CLAUDE.md explicitly says "a wrong answer that looks plausible is worse than no answer" and two diverging simulation engines risk exactly that.

**Backfill mechanism**: Baseball-scout wants it removed or heavily reworked (double-counts injury risk, distorts AVG for high-walk hitters). Data-scientist notes the binary cliff but considers the methodology defensible. Lean toward the scout — the Monte Carlo already handles injury variance, so backfill is redundant at best and distortionary at worst. Already acknowledged in TODO.md.

**SGP denominators**: Baseball-scout says the defaults are clearly wrong for this league. Data-scientist frames it as an architectural concern (static vs dynamic). Both are right — the immediate fix is to use the override mechanism that already exists, the long-term fix is auto-computation.

**Closer SV threshold**: Baseball-scout sees the shallow closer pool as a strategic problem. Data-scientist notes the binary cutoff as a methodological concern. These are two sides of the same coin — a softer threshold would partially address the strategic issue by giving partial closer credit to 15-19 SV pitchers.
