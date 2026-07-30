# Per-season position eligibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-30-per-season-positions-design.md`

**Goal:** Measure the keeper positional credits against each historical season's own
eligibility (derived from MLB Stats fielding, 10-game rule) instead of the 2026 Yahoo
map, then regenerate `NATIVE_CREDITS`; fold in findings 5 (UTIL floor) and 6 (prose).

**Architecture:** A pure `season_eligibility(fielding_frame) -> {mlbam: {slot}}` on the
existing cached MLB Stats fielding pull; `run_scarcity` joins it to each season's board
by MLBAM index (hitter fallback `{DH}`, pitcher `{P}`); `marginal_starter_floors` gets a
correct UTIL floor; `--scarcity` prints coverage + centred per-season credits + a
derived-2025-vs-Yahoo agreement check; then `NATIVE_CREDITS` is regenerated to delta 0.00.

**Tech Stack:** Python, pandas, requests, pytest. Branch `fix/273-per-season-positions`,
PRs into `feat/273-league-keeper-board`.

## Global Constraints

- **ASCII-only** in source, logs, and printed strings (CLAUDE.md). The script already
  reconfigures stdout to UTF-8 for player names (the one documented exception).
- **No `x or default` for numeric defaults** (CLAUDE.md). `games` is read with explicit
  numeric coercion, not `games or 0`.
- **Player/board index is MLBAM (`int`).** The fielding `player.id` is int64; coerce both
  sides to `int` before joining — a str/int mismatch makes every lookup silently miss.
- **The live board keeps the Yahoo map.** `build`/`projected` must not change position
  source; only `run_scarcity` and `marginal_starter_floors` change, and `NATIVE_CREDITS`
  only at Task 5.
- **`--scarcity` must regenerate `NATIVE_CREDITS` to delta 0.00** — the subsystem's
  self-consistency check.
- **Do not loosen a test to pass** — fix code or justify the change (requirement changed).
- **End-of-effort:** `pytest -q -n auto`, `ruff check .`, `ruff format --check .`, `mypy`,
  `vulture` clean. Show outputs.

---

## Task 0: Pre-warm the fielding cache (prerequisite)

**Files:** none (writes `data/cache/keeper_skills/mlb_fielding_{year}.csv`, gitignored).

`fetch_mlb_season(cache_dir, year, "fielding")` already fetches + caches. Pre-warm
2022-2025 so the measurement runs offline and repeatably.

- [ ] **Step 1: Fetch and cache fielding for 2022-2025**

Run:
```bash
python -c "import sys; sys.path.insert(0,'src'); from pathlib import Path; from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season; [print(y, len(fetch_mlb_season(Path('data/cache/keeper_skills'), y, 'fielding'))) for y in range(2022,2026)]"
```
Expected: four lines like `2022 <N>` with N in the low thousands (one row per
player-position). Four CSVs written.

- [ ] **Step 2: Verify the cache**

Run: `ls data/cache/keeper_skills/mlb_fielding_20{22,23,24,25}.csv`
Expected: four files.

- [ ] **Step 3: If the API is unreachable, STOP and surface**

If the fetch errors (timeout/HTTP), do not fabricate — report and wait. No commit
(cache is gitignored).

- [ ] **Step 4: Capture the live-board baseline (for the Task 2 gate)**

The branch is at `feat/273`'s tip with no source changes yet, so build the 2026 board
now and save it — Task 2 diffs against this to prove the measurement change doesn't
move the live board.
```bash
SCRATCH="C:/Users/HARTAL~1/AppData/Local/Temp/claude/C--Users-HartAlden-FantasyBaseball/d2b8a652-4156-4dec-96fb-f08689a6dbcf/scratchpad"
python scripts/keeper_rankings.py --year 2026 >/dev/null
cp data/cache/keeper_skills/keeper_rankings_hitter_2026.csv "$SCRATCH/baseline_hitter_2026.csv"
cp data/cache/keeper_skills/keeper_rankings_pitcher_2026.csv "$SCRATCH/baseline_pitcher_2026.csv"
```

---

## Task 1: `season_eligibility` derivation (pure)

**Files:**
- Create: `src/fantasy_baseball/keepers/appearances.py`
- Test: `tests/test_keepers/test_appearances.py`

**Interfaces:**
- Produces: `season_eligibility(fielding: pd.DataFrame) -> dict[int, set[str]]` — MLBAM id
  to the base slots (`C/1B/2B/3B/SS/OF/P`) at which he has >= 10 games that season.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_keepers/test_appearances.py`:

```python
import pandas as pd

from fantasy_baseball.keepers.appearances import season_eligibility


def _fielding(rows):
    """rows: list of (player_id, position_abbrev, games)."""
    return pd.DataFrame(
        [{"player.id": pid, "position.abbreviation": pos, "stat.games": g} for pid, pos, g in rows]
    )


def test_ten_games_is_eligible_nine_is_not():
    out = season_eligibility(_fielding([(1, "C", 10), (2, "C", 9)]))
    assert out == {1: {"C"}}  # player 2 has no >=10-game slot at all


def test_outfield_is_combined_across_corners():
    # 6 in LF + 5 in CF = 11 combined -> OF, even though neither corner reached 10.
    out = season_eligibility(_fielding([(1, "LF", 6), (1, "CF", 5)]))
    assert out == {1: {"OF"}}


def test_outfield_under_ten_combined_is_not_eligible():
    out = season_eligibility(_fielding([(1, "LF", 6), (1, "CF", 3)]))
    assert out == {}


def test_a_multi_position_player_gets_every_qualifying_slot():
    out = season_eligibility(_fielding([(1, "C", 12), (1, "1B", 15), (1, "SS", 4)]))
    assert out == {1: {"C", "1B"}}  # SS at 4 games drops out


def test_dh_and_unknown_tokens_map_to_no_slot():
    # A pure DH has a DH fielding row (or none); either way no base slot.
    out = season_eligibility(_fielding([(1, "DH", 100)]))
    assert out == {}


def test_pitchers_map_to_p():
    out = season_eligibility(_fielding([(1, "P", 30)]))
    assert out == {1: {"P"}}


def test_player_id_key_is_int():
    out = season_eligibility(_fielding([(660271, "1B", 20)]))
    assert list(out) == [660271] and isinstance(next(iter(out)), int)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_keepers/test_appearances.py -q`
Expected: FAIL, "No module named ... appearances".

- [ ] **Step 3: Implement `appearances.py`**

```python
"""Per-season position eligibility from the MLB Stats fielding leaderboard.

Pure and I/O-free like the rest of the normalization layer: it takes the raw
fielding frame (`keepers.mlb_stats.fetch_mlb_season(cache_dir, year, "fielding")`)
and returns, per MLBAM id, the base slots at which the player has >= 10 games that
season. That is the contemporaneous eligibility the scarcity measurement needs;
`keepers.positions` (the Yahoo map) has only the current season and is used for the
LIVE board, not for measuring history. See `docs/superpowers/specs/2026-07-30-
per-season-positions-design.md`.
"""

from __future__ import annotations

import pandas as pd

# Yahoo eligibility rule: >= 10 games at a position in the season. Outfield is
# combined -- the league rosters a single OF slot -- so LF/CF/RF all fold to OF and
# their games sum before the threshold. DH and any non-fielding token fold to no base
# slot, so a pure DH is absent here and the caller prices him at UTIL.
GAMES_THRESHOLD = 10
POSITION_TO_SLOT: dict[str, str] = {
    "C": "C",
    "1B": "1B",
    "2B": "2B",
    "3B": "3B",
    "SS": "SS",
    "LF": "OF",
    "CF": "OF",
    "RF": "OF",
    "P": "P",
}
_REQUIRED = ("player.id", "position.abbreviation", "stat.games")


def season_eligibility(fielding: pd.DataFrame) -> dict[int, set[str]]:
    """MLBAM id -> base slots with >= 10 games this season."""
    missing = [c for c in _REQUIRED if c not in fielding.columns]
    if missing:
        raise KeyError(f"fielding frame missing {missing}; got {sorted(fielding.columns)}")
    frame = pd.DataFrame(
        {
            "pid": pd.to_numeric(fielding["player.id"], errors="coerce").astype("Int64"),
            "slot": fielding["position.abbreviation"].map(POSITION_TO_SLOT),
            "games": pd.to_numeric(fielding["stat.games"], errors="coerce"),
        }
    ).dropna(subset=["pid", "slot", "games"])
    grouped = frame.groupby(["pid", "slot"], sort=False)["games"].sum()
    out: dict[int, set[str]] = {}
    for (pid, slot), games in grouped.items():
        if games >= GAMES_THRESHOLD:
            out.setdefault(int(pid), set()).add(str(slot))
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_keepers/test_appearances.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/keepers/appearances.py tests/test_keepers/test_appearances.py
git commit -m "feat(keepers): season_eligibility from MLB Stats fielding (10-game rule)"
```

---

## Task 2: Wire per-season eligibility into `run_scarcity` + coverage

**Files:**
- Modify: `scripts/keeper_rankings.py` (`run_scarcity`, add an eligibility helper + import)

**Interfaces:**
- Consumes: `season_eligibility`, `fetch_mlb_season`, `HITTER_ELIGIBLE`, `SKILLS_DIR`.
- Produces: `run_scarcity` scores each season against derived eligibility; prints per-season
  coverage. `NATIVE_CREDITS` unchanged (deltas will move; do not reship yet).

- [ ] **Step 1: Add imports**

In `scripts/keeper_rankings.py`, add:
```python
from fantasy_baseball.keepers.appearances import season_eligibility
from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season
```
`HITTER_ELIGIBLE` is already imported.

- [ ] **Step 2: Add a per-season eligibility helper**

Above `run_scarcity`:
```python
# Hitter base slots the derived source can emit; anything else (a stray "P" on a
# two-way player's line) is dropped from the hitter pool.
_HITTER_BASE_SLOTS = frozenset(str(p) for p in HITTER_ELIGIBLE) & frozenset(
    {"C", "1B", "2B", "3B", "SS", "OF"}
)


def _season_eligibility(year: int, board_index, kind: str) -> tuple[dict[object, set[str]], int]:
    """Per-season eligibility keyed by the board's MLBAM index, and the count of
    hitter rows that fell to the DH fallback (for the coverage print).

    Pitchers are `{P}` (the pool has one slot). A hitter with no >= 10-game hitter
    slot is priced at `{DH}` -- UTIL-only via `can_fill_slot`, matching the old
    `FALLBACK_POS` intent but derived per season instead of from the 2026 Yahoo map.
    """
    if kind == PlayerType.PITCHER:
        return {idx: {"P"} for idx in board_index}, 0
    derived = season_eligibility(fetch_mlb_season(SKILLS_DIR, year, "fielding"))
    eligible: dict[object, set[str]] = {}
    dh_fallback = 0
    for idx in board_index:
        slots = {s for s in derived.get(int(idx), set()) if s in _HITTER_BASE_SLOTS}
        if not slots:
            slots = {"DH"}
            dh_fallback += 1
        eligible[idx] = slots
    return eligible, dh_fallback
```

- [ ] **Step 3: Use it in `run_scarcity`, replacing the Yahoo-map lookup**

Replace the per-season loop body. The old block builds `eligible` from
`_slots_for(positions, name, kind)`; replace with `_season_eligibility`, and collect
coverage for printing:

```python
    per_season = []
    coverage: list[tuple[int, int, int]] = []  # (year, hitter_rows, dh_fallback)
    for year in SCARCITY_YEARS:
        floors: dict[str, float] = {}
        for kind in POOLS:
            board = projected(year, kind, denoms)
            eligible, dh_fallback = _season_eligibility(year, board.index, kind)
            floors.update(marginal_starter_floors(board["proj_sgp"], eligible, wanted[kind]))
            if kind == PlayerType.HITTER:
                coverage.append((year, len(board), dh_fallback))
        per_season.append((year, floors))
```

Remove the now-unused `positions, _ = pricing_table()` line at the top of
`run_scarcity` (the derived source replaces it). Keep the `capacities`/`wanted` setup.

- [ ] **Step 4: Print coverage**

After the per-season floor table print, add:
```python
    print("\n  hitter map coverage per season (rows priced at a real position vs DH):")
    print("    year   rows   real    DH")
    for year, rows, dh in coverage:
        print(f"    {year}  {rows:>5}  {rows - dh:>5}  {dh:>4}")
```

- [ ] **Step 5: Run `--scarcity` and sanity-check coverage**

Run: `python scripts/keeper_rankings.py --scarcity 2>&1 | grep -vE "FutureWarning|to_datetime"`
Expected: runs to completion; the coverage table shows most hitter rows at a real
position (DH fallback should be a small minority, unlike the ~140/320 the Yahoo map
routed to UTIL). The `floor/credit/shipped/delta` block will now show non-zero deltas
against the still-shipped `NATIVE_CREDITS` — that is expected; do NOT reship here.

- [ ] **Step 6: Confirm the live board is unchanged (gate)**

`run_scarcity` and `marginal_starter_floors` are not on the `build` path, so the live
board must be value-identical to the Task 0 baseline. Rebuild and diff the
load-bearing columns:
```bash
python scripts/keeper_rankings.py --year 2026 >/dev/null
python - <<'PY'
import pandas as pd, sys
SCRATCH = r"C:/Users/HARTAL~1/AppData/Local/Temp/claude/C--Users-HartAlden-FantasyBaseball/d2b8a652-4156-4dec-96fb-f08689a6dbcf/scratchpad"
cols = ["rank", "composite", "proj_sgp", "sd", "proj_var"]
bad = False
for kind in ("hitter", "pitcher"):
    base = pd.read_csv(f"{SCRATCH}/baseline_{kind}_2026.csv").set_index("mlbam_id")[cols].round(6)
    new = pd.read_csv(f"data/cache/keeper_skills/keeper_rankings_{kind}_2026.csv").set_index("mlbam_id")[cols].round(6)
    ok = base.equals(new.reindex(base.index))
    print(kind, "IDENTICAL" if ok else "DIFFERS"); bad |= not ok
sys.exit(1 if bad else 0)
PY
```
Expected: both `IDENTICAL`. If either DIFFERS, the measurement change leaked into the
live board — fix before continuing (only `NATIVE_CREDITS`, at Task 5, may move it).

- [ ] **Step 7: Commit**

```bash
git add scripts/keeper_rankings.py
git commit -m "feat(keepers): run_scarcity uses per-season derived eligibility + coverage"
```

---

## Task 3: Fix the UTIL floor (finding 5)

**Files:**
- Modify: `src/fantasy_baseball/keepers/scarcity.py` (`marginal_starter_floors`)
- Test: `tests/test_keepers/test_scarcity.py`

- [ ] **Step 1: Write the failing test (the case the old logic gets wrong)**

The distinguishing case: a UTIL-only (DH) bat is the best LEFTOVER after the fill, so
the correct UTIL floor is that DH's value — which the old `max(dedicated floors)` can
never produce. Add to `tests/test_keepers/test_scarcity.py`:
```python
def test_util_floor_can_be_set_by_a_util_only_leftover():
    """A UTIL-only bat (a DH) who is the best leftover must SET the UTIL floor. The
    old code maxed over dedicated floors only, so a player eligible for no dedicated
    slot could never set it, understating UTIL."""
    values = {0: 10.0, 1: 4.0, 2: 10.0, 3: 9.5, 4: 9.0, 5: 3.0}
    elig = {0: {"C"}, 1: {"C"}, 2: {"OF"}, 3: {"DH"}, 4: {"DH"}, 5: {"OF"}}
    floors = _floors(values, elig, {"C": 1, "OF": 1, "UTIL": 1})
    # Fill: C<-10@0, OF<-10@2, UTIL(flex)<-best remaining hitter = 9.5 DH@3.
    # Leftovers: 4.0 C@1, 9.0 DH@4, 3.0 OF@5.
    #   old (max dedicated floors): max(C 4.0, OF 3.0) = 4.0  -- WRONG
    #   new (best UTIL-eligible leftover): the 9.0 DH@4        -- RIGHT
    assert floors["UTIL"] == pytest.approx(9.0)


def test_util_floor_falls_back_to_max_dedicated_when_no_leftover_qualifies():
    """With no leftover at all, UTIL is omitted (nobody to price it), same as any
    slot with nobody left over -- exercising the fallback branch."""
    values = {0: 10.0, 1: 4.0}
    elig = {0: {"C"}, 1: {"C"}}
    floors = _floors(values, elig, {"C": 1, "UTIL": 1})
    # C<-10@0, UTIL(flex)<-4@1. No player is left over, so UTIL has no floor.
    assert "UTIL" not in floors
```
The existing `test_util_is_priced_at_the_deepest_hitter_floor` still passes: there the
best UTIL-eligible leftover equals the deepest dedicated floor, so the new logic
returns the same value.

- [ ] **Step 2: Run to verify the first test fails**

Run: `pytest tests/test_keepers/test_scarcity.py -k util_floor -q`
Expected: `test_util_floor_can_be_set_by_a_util_only_leftover` FAILS (old code returns
`max(dedicated) = 4.0`, not `9.0`). The fallback test passes already.

- [ ] **Step 3: Fix `marginal_starter_floors`**

Replace the UTIL block (currently `hitters = {...}; floors["UTIL"] = max(hitters.values())`):
```python
    # UTIL is a flex slot with no dedicated marginal starter, but every DH-only bat
    # is priced against it, so its floor is the best leftover who can fill UTIL --
    # found the same way as every dedicated slot. Only when no such leftover exists
    # (a shallow pool) does it fall back to the deepest dedicated floor.
    if capacities.get("UTIL", 0) > 0:
        util_floor = next(
            (
                float(ranked.loc[player])
                for player in ranked.index
                if player not in started and can_fill_slot(eligibility.get(player, ()), "UTIL")
            ),
            None,
        )
        if util_floor is None:
            hitters = [level for slot, level in floors.items() if slot != "P"]
            util_floor = max(hitters) if hitters else None
        if util_floor is not None:
            floors["UTIL"] = util_floor
```

- [ ] **Step 4: Run the full scarcity suite**

Run: `pytest tests/test_keepers/test_scarcity.py -q`
Expected: PASS, including the two new tests and the existing
`test_util_is_priced_at_the_deepest_hitter_floor` (its best leftover equals the
deepest dedicated floor, so the new logic returns the same value).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/keepers/scarcity.py tests/test_keepers/test_scarcity.py
git commit -m "fix(keepers): UTIL floor is the best UTIL-eligible leftover (finding 5)"
```

---

## Task 4: `--scarcity` centred credits + 2025-vs-Yahoo agreement; delete prose (finding 6)

**Files:**
- Modify: `scripts/keeper_rankings.py` (`run_scarcity`), `src/fantasy_baseball/keepers/scarcity.py` (comment)

- [ ] **Step 1: Print centred per-season credits beside each season's floors**

In `run_scarcity`, after the per-season floor table, add a centred-credit table so the
per-season spread is a flag output, not prose:
```python
    print("\n  centred credit per season (what makes the '0.50 to 2.18' range a flag):")
    print("    year  " + "".join(f"{slot:>8}" for slot in slots))
    for year, floors in per_season:
        credits = centred_credits(floors)
        print(f"    {year}  " + "".join(f"{credits.get(s, float('nan')):>8.2f}" for s in slots))
```

- [ ] **Step 2: Add the derived-2025-vs-Yahoo agreement check**

Add a helper and call it once from `run_scarcity`:
```python
def _validate_against_yahoo(denoms) -> None:
    """Sanity-check the 10-game rule against the real Yahoo map, on 2025 (a COMPLETE
    season whose eligibility is what Yahoo's 2026 map is built from). Bridges the
    derived map (MLBAM) to Yahoo (name) through the 2025 board's own name column."""
    yahoo, _ = pricing_table()  # normalized-name -> Yahoo slots
    derived = season_eligibility(fetch_mlb_season(SKILLS_DIR, 2025, "fielding"))
    board = projected(2025, PlayerType.HITTER, denoms)
    slots = ("C", "1B", "2B", "3B", "SS", "OF")
    hit = {s: [0, 0] for s in slots}  # [derived-agrees, yahoo-has]
    for idx, name in zip(board.index, board["name"], strict=True):
        y = set(yahoo.get(normalize_name(str(name)), []))
        d = derived.get(int(idx), set())
        for s in slots:
            if s in y:
                hit[s][1] += 1
                if s in d:
                    hit[s][0] += 1
    print("\n  derived-2025 vs Yahoo-2026 agreement (share of Yahoo's players recovered):")
    for s in slots:
        agree, total = hit[s]
        pct = 100.0 * agree / total if total else float("nan")
        print(f"    {s:<4} {agree:>4}/{total:<4} = {pct:5.1f}%")
```
Call `_validate_against_yahoo(denoms)` near the end of `run_scarcity`, before the
paste-ready block. If any dedicated slot is grossly low (well under ~80%), STOP and
investigate the mapping before Task 5 — do not reship on a broken rule.

- [ ] **Step 3: Delete the unregenerable prose (finding 6)**

In `src/fantasy_baseball/keepers/scarcity.py`, delete the sentence "the catcher figure
ranges 0.50 to 2.18 year to year, so the average is the number to use and no single
season is." Replace with a pointer: "`--scarcity` prints the per-season centred credit,
which is where the year-to-year spread is read; it is deliberately not restated here."

- [ ] **Step 4: Run `--scarcity`**

Run: `python scripts/keeper_rankings.py --scarcity 2>&1 | grep -vE "FutureWarning|to_datetime"`
Expected: floors, centred per-season credits, coverage, and the agreement table all
print; agreement on C/OF/infield is high (the rule reproduces Yahoo). Save the full
output to `docs/superpowers/keeper-scarcity-per-season-2026-07-30.txt`.

- [ ] **Step 5: Commit**

```bash
git add scripts/keeper_rankings.py src/fantasy_baseball/keepers/scarcity.py
git commit -m "feat(keepers): --scarcity prints per-season credits + Yahoo agreement; drop prose (finding 6)"
```

---

## Task 5: Regenerate `NATIVE_CREDITS`, propagate, verify

**Files:**
- Modify: `src/fantasy_baseball/keepers/scarcity.py` (`NATIVE_CREDITS`), `tests/` as needed
- Update: PR #279 description; `docs/superpowers/keeper-scarcity-per-season-2026-07-30.txt`

- [ ] **Step 1: Grep for tests/consumers pinning the old credits**

Run:
```bash
grep -rn "NATIVE_CREDITS\|1\.176\|1\.446\|proj_var" tests/ src/ | grep -viE "scarcity.py:|keeper_rankings.py:"
```
List anything pinning a `NATIVE_CREDITS` literal or a credit-dependent board value.
`test_scarcity.py::test_the_shipped_table_prices_catcher_as_the_scarcest_hitter_slot`
asserts C is the scarcest positive hitter credit — it must still hold on the new
numbers (catcher is genuinely scarce); if it does NOT, STOP and investigate rather
than editing the test.

- [ ] **Step 2: Re-confirm the board still matches baseline, THEN paste**

Before reshipping, prove Tasks 3-4 left the live board untouched (only the reship may
move it): re-run the Task 2 Step 6 gate; expect both pools `IDENTICAL`. Then, from the
Task 4 `--scarcity` output's paste-ready block, replace `NATIVE_CREDITS` in
`scarcity.py` with the regenerated dict. (After this paste the board is EXPECTED to
move — that is finding 1's fix.)

- [ ] **Step 3: Confirm delta 0.00**

Run: `python scripts/keeper_rankings.py --scarcity 2>&1 | grep -A12 "against what is shipped"`
Expected: the `delta` column is `0.00` (or `-0.00`) on every slot.

- [ ] **Step 4: Run the scarcity suite and the catcher-conclusion guard**

Run: `pytest tests/test_keepers/test_scarcity.py -q`
Expected: PASS. If `test_the_shipped_table_prices_catcher...` fails, the corrected
measurement no longer makes catcher the scarcest slot — investigate and report; do
not loosen the test.

- [ ] **Step 5: Re-run the boards**

Run:
```bash
python scripts/keeper_rankings.py --year 2026 2>&1 | grep -E "==="
python scripts/keeper_rankings.py --league --top 50 2>&1 | tail -40   # needs live KV; else note it
```
Record the new top board into the scarcity notes file. The catcher-led top order
should change (that is the intended outcome of finding 1).

- [ ] **Step 6: Update PR #279's description**

Update the PR body's scarcity table and the "catcher premium was roughly right (+1.38
vs +1.18)" claim to the corrected numbers (the old measurement was contaminated on
both sides). Use `gh pr edit 279 --body-file <updated>` or note the exact edits for
the author. State that the credits now come from per-season derived eligibility.

- [ ] **Step 7: Full verification**

Run and show output:
```bash
pytest -q -n auto
ruff check .
ruff format --check .
mypy
vulture
```
Expected: suite green (pre-existing unrelated failures noted — the `resend`
ModuleNotFoundError and the base-branch `test_send_daily_summary` failure are not from
this change); ruff/format/mypy clean; no NEW vulture findings.

- [ ] **Step 8: Commit**

```bash
git add src/fantasy_baseball/keepers/scarcity.py tests/ docs/superpowers/keeper-scarcity-per-season-2026-07-30.txt
git commit -m "feat(keepers): reship NATIVE_CREDITS from per-season eligibility (finding 1)"
```

---

## Self-review notes

- **Spec coverage:** R1 fetch (Task 0 uses existing `fetch_mlb_season`); R2 `season_eligibility`
  (Task 1); R3 wiring + fallback (Task 2); R4 UTIL floor (Task 3); R5 coverage + centred
  credits + agreement (Tasks 2, 4); R6 prose deleted (Task 4); R7 reship + delta 0.00
  (Task 5); R8 live board gate (Task 2 Step 6); R9 boards + PR (Task 5 Steps 5-6); R10
  full checks (Task 5 Step 7).
- **`stat.games` vs `stat.gamesPlayed`:** the fielding split carries both; `stat.games` is
  appearances at that position (what the probe showed per corner), which is the 10-game
  rule's basis. Do not switch to `gamesStarted` (a starter-only count).
- **Two-way (Ohtani):** hitter pool reads his hitter slots (mostly DH -> `{DH}` -> UTIL);
  pitcher pool is `{P}`. `_HITTER_BASE_SLOTS` drops any stray `P` from the hitter side.
- **No-live-board-change:** `marginal_starter_floors` and `run_scarcity` are not on the
  `build` path; the only thing that moves the board is `NATIVE_CREDITS`, and only at Task 5.
- **Data-dependent output:** the regenerated `NATIVE_CREDITS` (Task 5) and the new top
  board are outputs of the corrected measurement, not knowable in advance; the plan gives
  the exact `--scarcity` command that produces them. Not placeholders.
