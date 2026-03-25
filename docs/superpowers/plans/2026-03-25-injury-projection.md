# Injury-Aware Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive replacement rates from the actual player pool and blend injury-prone players' stats with waiver-quality backfill before SGP calculation, so draft VAR reflects the true team-level cost of fragile pitchers and hitters.

**Architecture:** New `calculate_replacement_rates` function derives ERA/WHIP/AVG from the pool. New `apply_backfill_blending` function in `board.py` modifies the pool DataFrame's counting stats before SGP is calculated. Existing `calculate_player_sgp` receives pool-derived rates as kwargs, keeping its interface backward-compatible.

**Tech Stack:** Python, pandas, pytest

**Spec:** `docs/superpowers/specs/2026-03-25-injury-projection-design.md`

**Branch:** `injury-projection` (already created)

**Note:** Config overrides via `league.yaml` are deferred. For now, edit constants in `constants.py` directly. The spec mentions `league.yaml` configurability as a future enhancement.

---

### Task 1: Add waiver-quality constants to `constants.py`

**Files:**
- Modify: `src/fantasy_baseball/utils/constants.py:79-80`

- [ ] **Step 1: Add constants after existing `REPLACEMENT_RP`**

Add after line 79 in `constants.py`:

```python
# Waiver-quality stats for injury backfill blending (10-team league).
# Separate from REPLACEMENT_* (used by Monte Carlo) — these model what
# you'd actually stream from waivers, which is slightly better quality.
WAIVER_SP: dict[str, float] = {
    "w": 7, "k": 120, "sv": 0, "ip": 140, "er": 65, "bb": 48, "h_allowed": 133,
}
WAIVER_RP: dict[str, float] = {
    "w": 2, "k": 55, "sv": 5, "ip": 60, "er": 30, "bb": 21, "h_allowed": 60,
}
WAIVER_HITTER: dict[str, float] = {
    "r": 55, "hr": 12, "rbi": 50, "sb": 5, "h": 150, "ab": 600,
}

# Healthy baselines for backfill blending
HEALTHY_SP_IP: float = 178.0
HEALTHY_CLOSER_IP: float = 60.0
HEALTHY_HITTER_AB: float = 600.0

# Gap thresholds — backfill only applies when gap exceeds these
BACKFILL_SP_THRESHOLD: float = 15.0
BACKFILL_CLOSER_THRESHOLD: float = 10.0
BACKFILL_HITTER_THRESHOLD: float = 50.0

# IP threshold to distinguish starters from middle relievers
STARTER_IP_THRESHOLD: float = 100.0
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from fantasy_baseball.utils.constants import WAIVER_SP, HEALTHY_SP_IP; print('OK', WAIVER_SP)"`
Expected: prints OK with the dict

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/utils/constants.py
git commit -m "feat: add waiver-quality and backfill baseline constants"
```

---

### Task 2: Add `calculate_replacement_rates` to `replacement.py`

**Files:**
- Modify: `src/fantasy_baseball/sgp/replacement.py`
- Test: `tests/test_sgp/test_replacement.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sgp/test_replacement.py`:

```python
from fantasy_baseball.sgp.replacement import calculate_replacement_rates


def _make_pool_with_stats():
    """Pool with realistic rate stats for replacement rate testing."""
    hitters = []
    for i in range(130):
        ab = 550 - i * 2
        h = int(ab * (0.280 - i * 0.0004))
        hitters.append({
            "name": f"Hitter_{i}", "positions": ["OF"], "player_type": "hitter",
            "total_sgp": 20.0 - i * 0.15,
            "ab": ab, "h": h, "avg": h / ab if ab > 0 else 0,
        })
    pitchers = []
    for i in range(100):
        ip = 180 - i * 0.5
        er = int(ip * (3.50 + i * 0.015) / 9)
        bb = int(ip * 0.20) + i
        ha = int(ip * 0.85) + i
        pitchers.append({
            "name": f"Pitcher_{i}", "positions": ["SP"], "player_type": "pitcher",
            "total_sgp": 25.0 - i * 0.2,
            "ip": ip, "er": er, "bb": bb, "h_allowed": ha,
            "era": er * 9 / ip, "whip": (bb + ha) / ip,
        })
    return pd.DataFrame(hitters + pitchers)


class TestReplacementRates:
    def test_returns_era_whip_avg(self):
        pool = _make_pool_with_stats()
        rates = calculate_replacement_rates(pool)
        assert "era" in rates
        assert "whip" in rates
        assert "avg" in rates

    def test_era_between_reasonable_bounds(self):
        pool = _make_pool_with_stats()
        rates = calculate_replacement_rates(pool)
        assert 3.0 < rates["era"] < 6.0

    def test_avg_between_reasonable_bounds(self):
        pool = _make_pool_with_stats()
        rates = calculate_replacement_rates(pool)
        assert 0.200 < rates["avg"] < 0.300

    def test_empty_pitcher_pool_uses_defaults(self):
        hitters = [{"name": "H", "positions": ["OF"], "player_type": "hitter",
                     "total_sgp": 10, "ab": 550, "h": 150, "avg": 0.273}]
        pool = pd.DataFrame(hitters)
        rates = calculate_replacement_rates(pool)
        assert rates["era"] == 4.50  # fallback default
        assert rates["whip"] == 1.35

    def test_empty_hitter_pool_uses_defaults(self):
        pitchers = [{"name": "P", "positions": ["SP"], "player_type": "pitcher",
                      "total_sgp": 10, "ip": 180, "er": 70, "bb": 50,
                      "h_allowed": 155, "era": 3.50, "whip": 1.14}]
        pool = pd.DataFrame(pitchers)
        rates = calculate_replacement_rates(pool)
        assert rates["avg"] == 0.250  # fallback default

    def test_zero_ip_pitchers_excluded_from_band(self):
        pitchers = []
        for i in range(100):
            ip = 180 - i * 0.5 if i < 95 else 0  # last 5 have 0 IP
            er = int(ip * 3.80 / 9) if ip > 0 else 0
            pitchers.append({
                "name": f"P_{i}", "positions": ["SP"], "player_type": "pitcher",
                "total_sgp": 25.0 - i * 0.2, "ip": ip, "er": er,
                "bb": int(ip * 0.2), "h_allowed": int(ip * 0.85),
                "era": er * 9 / ip if ip > 0 else 0,
                "whip": (int(ip * 0.2) + int(ip * 0.85)) / ip if ip > 0 else 0,
            })
        pool = pd.DataFrame(pitchers)
        rates = calculate_replacement_rates(pool)
        assert rates["era"] > 0  # should not be poisoned by 0 IP pitchers

    def test_closers_only_pool(self):
        """Pool with only closers still produces valid pitcher rates."""
        pitchers = []
        for i in range(100):
            ip = 60 - i * 0.2
            er = int(ip * (3.20 + i * 0.02) / 9)
            pitchers.append({
                "name": f"RP_{i}", "positions": ["RP"], "player_type": "pitcher",
                "total_sgp": 15.0 - i * 0.1, "ip": ip, "er": er,
                "bb": int(ip * 0.18), "h_allowed": int(ip * 0.80),
                "era": er * 9 / ip if ip > 0 else 0,
                "whip": (int(ip * 0.18) + int(ip * 0.80)) / ip if ip > 0 else 0,
            })
        pool = pd.DataFrame(pitchers)
        rates = calculate_replacement_rates(pool)
        assert 2.0 < rates["era"] < 8.0
        assert 0.8 < rates["whip"] < 2.0

    def test_sp_only_pool(self):
        """Pool with only starters still produces valid pitcher rates."""
        pool = _make_pool_with_stats()  # already all SP
        rates = calculate_replacement_rates(pool)
        assert 3.0 < rates["era"] < 6.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sgp/test_replacement.py::TestReplacementRates -v`
Expected: FAIL — `calculate_replacement_rates` not found

- [ ] **Step 3: Implement `calculate_replacement_rates`**

Add import at the top of `src/fantasy_baseball/sgp/replacement.py` (with existing imports):

```python
from fantasy_baseball.sgp.player_value import REPLACEMENT_ERA, REPLACEMENT_WHIP, REPLACEMENT_AVG
```

Then add the function after `_get_eligible_players`:

```python
def calculate_replacement_rates(
    player_pool: pd.DataFrame,
    starters_per_position: dict[str, int] | None = None,
) -> dict[str, float]:
    """Derive replacement-level rate stats from the player pool.

    Averages ERA/WHIP/AVG across a band of ±5 players around the
    replacement threshold to smooth noise from any single player.
    Falls back to hardcoded defaults when the pool is empty.
    """
    if starters_per_position is None:
        starters_per_position = dict(STARTERS_PER_POSITION)

    rates: dict[str, float] = {}

    # Pitcher replacement rates
    num_p_starters = starters_per_position.get("P", 90)
    pitchers = _get_eligible_players(player_pool, "P")
    pitchers = pitchers.sort_values("total_sgp", ascending=False).reset_index(drop=True)

    if len(pitchers) > num_p_starters:
        lo = max(0, num_p_starters - 5)
        hi = min(len(pitchers), num_p_starters + 6)  # exclusive
        band = pitchers.iloc[lo:hi]
        band = band[band["ip"] > 0]  # exclude 0 IP
        if not band.empty:
            total_er = band["er"].sum()
            total_ip = band["ip"].sum()
            total_bb = band["bb"].sum()
            total_ha = band["h_allowed"].sum()
            rates["era"] = total_er * 9 / total_ip
            rates["whip"] = (total_bb + total_ha) / total_ip
        else:
            rates["era"] = REPLACEMENT_ERA
            rates["whip"] = REPLACEMENT_WHIP
    else:
        rates["era"] = REPLACEMENT_ERA
        rates["whip"] = REPLACEMENT_WHIP

    # Hitter replacement rates
    all_hitters = player_pool[
        player_pool["positions"].apply(is_hitter)
    ].sort_values("total_sgp", ascending=False).reset_index(drop=True)

    positional_hitter_slots = sum(
        n for pos, n in starters_per_position.items()
        if pos not in ("P", "IF", "UTIL")
    )
    util_slots = starters_per_position.get("UTIL", 0)
    total_hitter_starters = positional_hitter_slots + util_slots

    if len(all_hitters) > total_hitter_starters:
        lo = max(0, total_hitter_starters - 5)
        hi = min(len(all_hitters), total_hitter_starters + 6)
        band = all_hitters.iloc[lo:hi]
        band = band[band["ab"] > 0]
        if not band.empty:
            rates["avg"] = band["h"].sum() / band["ab"].sum()
        else:
            rates["avg"] = REPLACEMENT_AVG
    else:
        rates["avg"] = REPLACEMENT_AVG

    return rates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sgp/test_replacement.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/sgp/replacement.py tests/test_sgp/test_replacement.py
git commit -m "feat: add calculate_replacement_rates — derive ERA/WHIP/AVG from pool"
```

---

### Task 3: Add `apply_backfill_blending` to `board.py`

**Files:**
- Modify: `src/fantasy_baseball/draft/board.py`
- Test: `tests/test_draft/test_board.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_draft/test_board.py`:

```python
from fantasy_baseball.draft.board import apply_backfill_blending


class TestBackfillBlending:
    def _make_pitcher(self, name, ip, era, sv=0, positions=None):
        er = era * ip / 9
        bb = int(ip * 0.20)
        ha = int(ip * 0.85)
        return {
            "name": name, "player_type": "pitcher",
            "positions": positions or ["SP"],
            "ip": ip, "er": er, "bb": bb, "h_allowed": ha,
            "w": int(ip / 15), "k": int(ip * 0.9), "sv": sv,
            "era": era, "whip": (bb + ha) / ip if ip > 0 else 0,
        }

    def _make_hitter(self, name, ab, avg):
        h = int(ab * avg)
        return {
            "name": name, "player_type": "hitter",
            "positions": ["OF"],
            "ab": ab, "h": h, "r": int(ab * 0.16), "hr": int(ab * 0.05),
            "rbi": int(ab * 0.15), "sb": int(ab * 0.02),
            "avg": avg,
        }

    def test_sp_below_threshold_gets_blended(self):
        pool = pd.DataFrame([self._make_pitcher("Fragile Ace", 145, 3.20)])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["ip"] == pytest.approx(178.0)
        assert result.iloc[0]["er"] > 145 * 3.20 / 9  # more ER from backfill

    def test_sp_above_threshold_unchanged(self):
        pool = pd.DataFrame([self._make_pitcher("Durable SP", 170, 3.60)])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["ip"] == pytest.approx(170.0)  # gap=8, below 15 threshold

    def test_closer_uses_closer_baseline(self):
        pool = pd.DataFrame([self._make_pitcher("Hurt Closer", 45, 3.00, sv=25, positions=["RP"])])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["ip"] == pytest.approx(60.0)  # closer baseline

    def test_middle_reliever_unchanged(self):
        pool = pd.DataFrame([self._make_pitcher("Setup Man", 55, 3.50, sv=5, positions=["RP"])])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["ip"] == pytest.approx(55.0)  # no backfill

    def test_hitter_below_threshold_gets_blended(self):
        pool = pd.DataFrame([self._make_hitter("Fragile Slugger", 520, 0.280)])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["ab"] == pytest.approx(600.0)
        # AVG should be dragged toward .250 by replacement ABs
        assert result.iloc[0]["h"] / result.iloc[0]["ab"] < 0.280

    def test_hitter_above_threshold_unchanged(self):
        pool = pd.DataFrame([self._make_hitter("Healthy Hitter", 570, 0.280)])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["ab"] == pytest.approx(570.0)  # gap=30, below 50

    def test_original_stats_preserved(self):
        pool = pd.DataFrame([self._make_pitcher("Fragile Ace", 145, 3.20)])
        result = apply_backfill_blending(pool)
        assert result.iloc[0]["orig_ip"] == pytest.approx(145.0)
        assert result.iloc[0]["orig_era"] == pytest.approx(3.20)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_board.py::TestBackfillBlending -v`
Expected: FAIL — `apply_backfill_blending` not found

- [ ] **Step 3: Implement `apply_backfill_blending`**

Add to `src/fantasy_baseball/draft/board.py` before `build_draft_board`:

```python
from fantasy_baseball.utils.constants import (
    CLOSER_SV_THRESHOLD,
    WAIVER_SP, WAIVER_RP, WAIVER_HITTER,
    HEALTHY_SP_IP, HEALTHY_CLOSER_IP, HEALTHY_HITTER_AB,
    BACKFILL_SP_THRESHOLD, BACKFILL_CLOSER_THRESHOLD, BACKFILL_HITTER_THRESHOLD,
    STARTER_IP_THRESHOLD,
    safe_float as _safe,
)


def apply_backfill_blending(pool: pd.DataFrame) -> pd.DataFrame:
    """Blend injury-prone players' stats with waiver-quality backfill.

    Players projected below a healthy baseline have their counting stats
    augmented with replacement-level stats for the gap innings/ABs.  This
    produces effective stats that reflect the true team-level cost.

    Original stats are preserved in ``orig_*`` columns for display.
    """
    pool = pool.copy()

    for idx, row in pool.iterrows():
        if row["player_type"] == "pitcher":
            sv = _safe(row.get("sv", 0))
            ip = _safe(row.get("ip", 0))
            positions = row.get("positions", [])

            # Classify pitcher tier
            if sv >= CLOSER_SV_THRESHOLD:
                baseline = HEALTHY_CLOSER_IP
                threshold = BACKFILL_CLOSER_THRESHOLD
                waiver = WAIVER_RP
            elif "SP" in positions or ip >= STARTER_IP_THRESHOLD:
                baseline = HEALTHY_SP_IP
                threshold = BACKFILL_SP_THRESHOLD
                waiver = WAIVER_SP
            else:
                continue  # middle reliever — no backfill

            gap = baseline - ip
            if gap <= threshold:
                continue

            # Preserve originals
            pool.at[idx, "orig_ip"] = ip
            pool.at[idx, "orig_era"] = row.get("era", 0)
            pool.at[idx, "orig_whip"] = row.get("whip", 0)

            scale = gap / waiver["ip"]
            for col in ("w", "k", "sv", "ip", "er", "bb", "h_allowed"):
                pool.at[idx, col] = row.get(col, 0) + waiver[col] * scale

            # Recompute rate stats from blended components
            new_ip = pool.at[idx, "ip"]
            if new_ip > 0:
                pool.at[idx, "era"] = pool.at[idx, "er"] * 9 / new_ip
                pool.at[idx, "whip"] = (pool.at[idx, "bb"] + pool.at[idx, "h_allowed"]) / new_ip

        elif row["player_type"] == "hitter":
            ab = _safe(row.get("ab", 0))
            gap = HEALTHY_HITTER_AB - ab
            if gap <= BACKFILL_HITTER_THRESHOLD:
                continue

            pool.at[idx, "orig_ab"] = ab
            pool.at[idx, "orig_avg"] = row.get("avg", 0)

            scale = gap / WAIVER_HITTER["ab"]
            for col in ("r", "hr", "rbi", "sb", "h", "ab"):
                pool.at[idx, col] = row.get(col, 0) + WAIVER_HITTER[col] * scale

            new_ab = pool.at[idx, "ab"]
            if new_ab > 0:
                pool.at[idx, "avg"] = pool.at[idx, "h"] / new_ab

    return pool
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_board.py::TestBackfillBlending -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/board.py tests/test_draft/test_board.py
git commit -m "feat: add apply_backfill_blending for injury-prone players"
```

---

### Task 4: Wire both changes into `build_draft_board`

**Files:**
- Modify: `src/fantasy_baseball/draft/board.py:49-53`

- [ ] **Step 1: Write integration test**

Add to `tests/test_draft/test_board.py`:

```python
class TestBoardBackfillIntegration:
    def test_fragile_sp_has_lower_sgp_than_durable(self):
        """Backfill should penalize a 145 IP ace relative to a 185 IP workhorse."""
        fragile = {"name": "Fragile Ace", "player_type": "pitcher", "positions": ["SP"],
                   "ip": 145, "er": 52, "bb": 40, "h_allowed": 120,
                   "w": 10, "k": 160, "sv": 0, "era": 3.23, "whip": 1.10}
        durable = {"name": "Durable SP", "player_type": "pitcher", "positions": ["SP"],
                   "ip": 185, "er": 66, "bb": 51, "h_allowed": 153,
                   "w": 13, "k": 185, "sv": 0, "era": 3.21, "whip": 1.10}
        pool = pd.DataFrame([fragile, durable])
        blended = apply_backfill_blending(pool)

        from fantasy_baseball.sgp.player_value import calculate_player_sgp
        fragile_sgp = calculate_player_sgp(blended.iloc[0])
        durable_sgp = calculate_player_sgp(blended.iloc[1])
        # Durable SP should have higher SGP despite similar ERA — no backfill drag
        assert durable_sgp > fragile_sgp
```

- [ ] **Step 2: Update `build_draft_board` to use both changes**

In `src/fantasy_baseball/draft/board.py`, replace lines 49-56 (from `denoms = ...` through `replacement_levels = ...`) with the complete updated block. Add the `calculate_replacement_rates` import at the top of the file with the other replacement imports.

Add to imports at top of file:

```python
from fantasy_baseball.sgp.replacement import calculate_replacement_levels, calculate_replacement_rates
```

(Remove the existing `from fantasy_baseball.sgp.replacement import calculate_replacement_levels` line.)

Replace lines 49-56:

```python
    denoms = get_sgp_denominators(sgp_overrides)
    pool = pd.concat([hitters, pitchers], ignore_index=True)

    # Apply injury backfill blending before SGP calculation
    pool = apply_backfill_blending(pool)

    # Derive replacement rates from the pool for more accurate SGP
    starters = compute_starters_per_position(roster_slots, num_teams)
    repl_rates = calculate_replacement_rates(pool, starters)

    pool["total_sgp"] = pool.apply(
        lambda row: calculate_player_sgp(
            row, denoms=denoms,
            replacement_era=repl_rates["era"],
            replacement_whip=repl_rates["whip"],
            replacement_avg=repl_rates["avg"],
        ),
        axis=1,
    )

    replacement_levels = calculate_replacement_levels(pool, starters)
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: all 418+ tests pass

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/draft/board.py tests/test_draft/test_board.py
git commit -m "feat: wire backfill blending and dynamic replacement rates into board build"
```

---

### Task 5: Verify impact on real data

**Files:** None (verification only)

- [ ] **Step 1: Compare before/after VAR for key players**

```bash
python -c "
from pathlib import Path
from fantasy_baseball.draft.board import build_draft_board

board = build_draft_board(
    Path('data/projections'), Path('data/player_positions.json'),
    systems=['steamer', 'zips', 'atc', 'the-bat-x', 'oopsy'],
)

# Show pitchers most affected by backfill (have orig_ip column)
backfilled = board[board.get('orig_ip', default=float('nan')).notna()]
if not backfilled.empty:
    print('=== Pitchers with backfill applied ===')
    for _, p in backfilled.head(15).iterrows():
        print(f'{p[\"name\"]:25s}  orig_ip={p[\"orig_ip\"]:.0f}  eff_ip={p[\"ip\"]:.0f}  '
              f'orig_era={p[\"orig_era\"]:.2f}  eff_era={p[\"era\"]:.2f}  VAR={p[\"var\"]:.2f}')

# Show top 20 by VAR
print()
print('=== Top 20 by VAR ===')
for _, p in board.head(20).iterrows():
    print(f'{p[\"name\"]:25s}  {p[\"player_type\"]:8s}  VAR={p[\"var\"]:.2f}  IP/AB={p.get(\"ip\", p.get(\"ab\", 0)):.0f}')
"
```

- [ ] **Step 2: Verify fragile pitchers dropped in rankings**

Check that deGrom, Sale, and other injury-prone pitchers have lower VAR than before. The top 20 should have fewer SPs and more balanced hitter/pitcher mix.

- [ ] **Step 3: Run full test suite one more time**

Run: `pytest tests/ -v`
Expected: all pass

- [ ] **Step 4: Commit verification notes**

Update `review_notes.md` to mark item #3 (injury/volatility) as addressed.

```bash
git add review_notes.md
git commit -m "docs: mark injury projection finding as addressed in review notes"
```

---

### Task 6: Push branch

- [ ] **Step 1: Push**

```bash
git push -u origin injury-projection
```
