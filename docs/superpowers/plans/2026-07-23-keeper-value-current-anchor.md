# Keeper Value: Current-Season Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-23-keeper-value-current-anchor-design.md`

**Goal:** Anchor the keeper-value metric on a current-season talent line (YTD + ROS) instead of the preseason blend, so in-season breakouts flow through, behind an `--anchor current|preseason` toggle defaulting to `current`.

**Architecture:** Two pure functions in `analysis/keeper_value.py` (`overlay_current_anchors`, `mark_preseason_fallback`) do the per-player anchor overlay and fallback flagging; a `parse_full_season_lines` extraction in `analysis/draft_value.py` lets the script parse the existing `cache:full_season_projections` blob read fresh from Upstash; `scripts/keeper_value.py` wires them into `build_results` behind the toggle. The out-year ratio math, VAR, discounting, and report are untouched.

**Tech Stack:** Python 3.11, pandas, pytest, existing SGP/VAR pipeline, Upstash KV.

## Global Constraints

- **Player IDs are `name::player_type`**; never key on bare names. Cross-source join key here is normalized `name::player_type` via `rank_key(name, player_type)`, VAR/volume tie-break for namesakes (repo convention).
- **ASCII-only** in all source, log, and report strings (Windows cp1252 stdout).
- **Do not use `x or default` for numeric defaults** (0/0.0 are falsy). Use `safe_float` / explicit `is None` checks.
- **`current` mode must fail loud** if `cache:full_season_projections` is missing/empty; never serve preseason under a `current` label.
- Downstream math (`per_year_var`, `_scale_line`, `discounted_total`, VAR) and the report are **unchanged**.
- `PlayerType.HITTER == "hitter"`, `PlayerType.PITCHER == "pitcher"` (StrEnum).
- Field sets: `HITTER_FIELDS = ("r","hr","rbi","sb","ab","avg")`, `PITCHER_FIELDS = ("w","k","sv","ip","era","whip")`. Floors: `DEFAULT_MIN_AB = 100.0`, `DEFAULT_MIN_IP = 20.0`.

---

### Task 1: Anchor overlay + fallback flag (pure)

**Files:**
- Modify: `src/fantasy_baseball/analysis/keeper_value.py`
- Test: `tests/test_analysis/test_keeper_value.py`

**Interfaces:**
- Consumes: `HITTER_FIELDS`, `PITCHER_FIELDS`, `DEFAULT_MIN_AB`, `DEFAULT_MIN_IP`, `safe_float`, `KeeperValueResult` (all already in this module); `rank_key` from `fantasy_baseball.sgp.rankings`; `replace` from `dataclasses`.
- Produces:
  - `overlay_current_anchors(hitters: pd.DataFrame, pitchers: pd.DataFrame, current_by_name: Mapping[str, Mapping[str, Any]], *, min_ab: float = DEFAULT_MIN_AB, min_ip: float = DEFAULT_MIN_IP) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]`
  - `mark_preseason_fallback(results: list[KeeperValueResult], current_keys: set[str]) -> list[KeeperValueResult]`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_analysis/test_keeper_value.py`:

```python
import pandas as pd
from fantasy_baseball.sgp.rankings import rank_key


def _frame(name, ptype_fields):
    # ptype_fields: dict of stat->value for one player row
    return pd.DataFrame([{"name": name, "fg_id": "1", **ptype_fields}])


def test_overlay_uses_current_line_above_floor():
    pre = _frame("Al Star", {"r": 60, "hr": 20, "rbi": 60, "sb": 5, "ab": 500, "avg": 0.250})
    empty_p = pd.DataFrame(columns=["name", "fg_id", "w", "k", "sv", "ip", "era", "whip"])
    current = {rank_key("Al Star", "hitter"): {"r": 90, "hr": 40, "rbi": 100, "sb": 8, "ab": 550, "avg": 0.300}}
    h, _p, keys = kv.overlay_current_anchors(pre, empty_p, current)
    assert h.iloc[0]["hr"] == 40 and h.iloc[0]["avg"] == 0.300  # current stats win
    assert rank_key("Al Star", "hitter") in keys


def test_overlay_keeps_preseason_when_no_current_line():
    pre = _frame("No Data", {"r": 60, "hr": 20, "rbi": 60, "sb": 5, "ab": 500, "avg": 0.250})
    empty_p = pd.DataFrame(columns=["name", "fg_id", "w", "k", "sv", "ip", "era", "whip"])
    h, _p, keys = kv.overlay_current_anchors(pre, empty_p, {})
    assert h.iloc[0]["hr"] == 20  # preseason unchanged
    assert keys == set()


def test_overlay_keeps_preseason_when_current_below_floor():
    pre = _frame("Hurt Guy", {"r": 60, "hr": 20, "rbi": 60, "sb": 5, "ab": 500, "avg": 0.250})
    empty_p = pd.DataFrame(columns=["name", "fg_id", "w", "k", "sv", "ip", "era", "whip"])
    # 40 AB is below DEFAULT_MIN_AB (100) -> keep preseason
    current = {rank_key("Hurt Guy", "hitter"): {"r": 8, "hr": 3, "rbi": 9, "sb": 0, "ab": 40, "avg": 0.300}}
    h, _p, keys = kv.overlay_current_anchors(pre, empty_p, current)
    assert h.iloc[0]["hr"] == 20  # kept preseason
    assert keys == set()


def test_mark_preseason_fallback_flags_only_non_current():
    r_cur = kv.KeeperValueResult("aaa::hitter", "Al Star", {2026: 1.0}, 1.0, [], 0.5, None)
    r_pre = kv.KeeperValueResult("bbb::hitter", "No Data", {2026: 1.0}, 1.0, [], 0.5, None)
    out = kv.mark_preseason_fallback([r_cur, r_pre], {rank_key("Al Star", "hitter")})
    flags = {r.name: r.flags for r in out}
    assert "anchor_preseason_fallback" not in flags["Al Star"]
    assert "anchor_preseason_fallback" in flags["No Data"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analysis/test_keeper_value.py -k "overlay or fallback" -q`
Expected: FAIL with `AttributeError: module ... has no attribute 'overlay_current_anchors'`.

- [ ] **Step 3: Implement the two functions**

Add imports at the top of `src/fantasy_baseball/analysis/keeper_value.py` (merge into existing import blocks):

```python
from dataclasses import dataclass, replace  # replace is new
from fantasy_baseball.sgp.rankings import rank_key
```

Add near the other module functions:

```python
def overlay_current_anchors(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
    current_by_name: Mapping[str, Mapping[str, Any]],
    *,
    min_ab: float = DEFAULT_MIN_AB,
    min_ip: float = DEFAULT_MIN_IP,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Replace each board frame's stat line with the current-talent line when one
    exists for that player (keyed name::player_type) AND clears the min-PT floor.

    Returns ``(merged_hitters, merged_pitchers, current_keys)`` where ``current_keys``
    are the ``rank_key(name, player_type)`` values that received the current anchor;
    every other player keeps its preseason line and is flagged by the caller.
    """
    current_keys: set[str] = set()
    out = []
    for df, ptype, fields, vol_field, floor in (
        (hitters, "hitter", HITTER_FIELDS, "ab", min_ab),
        (pitchers, "pitcher", PITCHER_FIELDS, "ip", min_ip),
    ):
        merged = df.copy()
        for idx, name in merged["name"].items():
            key = rank_key(str(name), ptype)
            line = current_by_name.get(key)
            if line is None or safe_float(line.get(vol_field, 0)) < floor:
                continue
            for f in fields:
                merged.at[idx, f] = line.get(f)
            current_keys.add(key)
        out.append(merged)
    return out[0], out[1], current_keys


def mark_preseason_fallback(
    results: list[KeeperValueResult], current_keys: set[str]
) -> list[KeeperValueResult]:
    """Append ``anchor_preseason_fallback`` to every result NOT scored off a current
    anchor (i.e. whose ``name::player_type`` is not in ``current_keys``)."""
    marked = []
    for r in results:
        ptype = r.player_id.rsplit("::", 1)[-1]
        if rank_key(r.name, ptype) in current_keys:
            marked.append(r)
        else:
            marked.append(replace(r, flags=[*r.flags, "anchor_preseason_fallback"]))
    return marked
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_keeper_value.py -k "overlay or fallback" -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/keeper_value.py tests/test_analysis/test_keeper_value.py
git commit -m "feat(keeper-value): pure anchor overlay + preseason-fallback flag"
```

---

### Task 2: Extract `parse_full_season_lines`

**Files:**
- Modify: `src/fantasy_baseball/analysis/draft_value.py:597-629`
- Test: `tests/test_analysis/test_draft_value.py` (create if absent)

**Interfaces:**
- Consumes: existing `_hit_line_from`, `_pit_line_from`, `_row_mlbam`, `_insert_by_name`, `rank_key` in `draft_value.py`.
- Produces: `parse_full_season_lines(payload: dict) -> tuple[dict[tuple[int, str], dict], dict[str, Any]]`. `load_full_season_lines()` keeps its signature and now delegates parsing to it.

- [ ] **Step 1: Write the failing test**

Create/append `tests/test_analysis/test_draft_value.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fantasy_baseball.analysis.draft_value import parse_full_season_lines
from fantasy_baseball.sgp.rankings import rank_key


def test_parse_full_season_lines_keys_by_name_and_mlbam():
    payload = {
        "hitters": [{"name": "Al Star", "mlbam_id": 111, "r": 90, "hr": 40, "rbi": 100, "sb": 8, "ab": 550, "h": 165}],
        "pitchers": [{"name": "Ace One", "mlbam_id": 222, "w": 15, "k": 200, "sv": 0, "ip": 190, "er": 60, "bb": 40, "h_allowed": 150}],
    }
    by_mlbam, by_name = parse_full_season_lines(payload)
    assert by_name[rank_key("Al Star", "hitter")]["hr"] == 40
    assert by_name[rank_key("Ace One", "pitcher")]["k"] == 200
    assert (111, "hitter") in by_mlbam


def test_parse_full_season_lines_empty_payload():
    assert parse_full_season_lines({}) == ({}, {})


def test_parse_full_season_lines_namesake_keeps_higher_volume():
    # two "Mason Miller" pitchers, distinct mlbam; by_name must keep the higher-IP one
    payload = {
        "hitters": [],
        "pitchers": [
            {"name": "Mason Miller", "mlbam_id": 1, "ip": 12, "k": 20, "w": 0, "sv": 5, "er": 4, "bb": 3, "h_allowed": 8},
            {"name": "Mason Miller", "mlbam_id": 2, "ip": 180, "k": 210, "w": 14, "sv": 0, "er": 60, "bb": 40, "h_allowed": 150},
        ],
    }
    _by_mlbam, by_name = parse_full_season_lines(payload)
    assert by_name[rank_key("Mason Miller", "pitcher")]["ip"] == 180  # higher-volume wins
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analysis/test_draft_value.py -q`
Expected: FAIL with `ImportError: cannot import name 'parse_full_season_lines'`.

- [ ] **Step 3: Refactor `load_full_season_lines` into read + parse**

In `src/fantasy_baseball/analysis/draft_value.py`, replace the body of `load_full_season_lines` (lines ~597-629). Move the loop into a new pure `parse_full_season_lines(payload)`; `load_full_season_lines` becomes a thin reader:

```python
def parse_full_season_lines(
    payload: dict[str, Any],
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    """Pure parse of a full_season_projections payload into ``(by_mlbam, by_name)``.

    Extracted from :func:`load_full_season_lines` so a caller with its own KV client
    (the keeper script reading fresh Upstash) reuses the identical keying/tie-break.
    """
    by_mlbam: dict[tuple[int, str], dict[str, Any]] = {}
    by_name: dict[str, Any] = {}
    name_mlbam: dict[str, int | None] = {}
    for ptype, recs, builder, vol in (
        ("hitter", payload.get("hitters", []), _hit_line_from, "ab"),
        ("pitcher", payload.get("pitchers", []), _pit_line_from, "ip"),
    ):
        for rec in recs:
            line = builder(rec)
            mlbam = _row_mlbam(rec)
            if mlbam is not None:
                by_mlbam[(mlbam, ptype)] = line
            name = rec.get("name") or ""
            if not name:
                continue
            key = rank_key(name, ptype)
            _insert_by_name(by_name, name_mlbam, key, line, vol, mlbam)
    return by_mlbam, by_name


def load_full_season_lines() -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    """Full-season projection lines, keyed by mlbam id AND by ``name::player_type``.

    Reads ``CacheKey.FULL_SEASON_PROJECTIONS`` from the KV store and parses it via
    :func:`parse_full_season_lines`. Returns ``({}, {})`` when the blob is absent.
    """
    payload = read_cache_dict(CacheKey.FULL_SEASON_PROJECTIONS) or {}
    return parse_full_season_lines(payload)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_draft_value.py -q`
Expected: PASS (2 tests). Also run any existing draft_value tests: `python -m pytest tests/test_analysis/ -k draft_value -q`.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py
git commit -m "refactor(draft-value): split parse_full_season_lines out of the reader"
```

---

### Task 3: Fresh-Upstash current-lines loader (fail-loud)

**Files:**
- Modify: `scripts/keeper_value.py`
- Test: `tests/test_scripts/test_keeper_value_script.py`

**Interfaces:**
- Consumes: `build_explicit_upstash_kv` (`data.kv_store`), `redis_key` + `CacheKey` (`data.cache_keys`), `unwrap_cache_envelope` (`web.season_data`), `parse_full_season_lines` (`analysis.draft_value`), `json`.
- Produces: `load_current_full_season_lines() -> dict[str, Any]` (the `by_name` map). Raises `SystemExit` with an ASCII "run a refresh" message when the blob is missing or empty.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_scripts/test_keeper_value_script.py`:

```python
def _fake_kv(raw):
    class _KV:
        def get(self, _key):
            return raw
    return _KV()


def test_load_current_lines_fails_loud_when_missing(monkeypatch):
    monkeypatch.setattr(script, "build_explicit_upstash_kv", lambda: _fake_kv(None))
    with pytest.raises(SystemExit):
        script.load_current_full_season_lines()


def test_load_current_lines_fails_loud_when_empty(monkeypatch):
    envelope = {"_meta": {}, "_data": {"hitters": [], "pitchers": []}}
    monkeypatch.setattr(script, "build_explicit_upstash_kv", lambda: _fake_kv(envelope))
    with pytest.raises(SystemExit):
        script.load_current_full_season_lines()


def test_load_current_lines_parses_present_blob(monkeypatch):
    from fantasy_baseball.sgp.rankings import rank_key

    envelope = {"_meta": {}, "_data": {"hitters": [{"name": "Al Star", "mlbam_id": 111, "hr": 40, "ab": 550, "h": 165}], "pitchers": []}}
    monkeypatch.setattr(script, "build_explicit_upstash_kv", lambda: _fake_kv(envelope))
    by_name = script.load_current_full_season_lines()
    assert by_name[rank_key("Al Star", "hitter")]["hr"] == 40
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scripts/test_keeper_value_script.py -k load_current -q`
Expected: FAIL with `AttributeError: module 'keeper_value' has no attribute 'load_current_full_season_lines'`.

- [ ] **Step 3: Implement the loader**

Add imports to `scripts/keeper_value.py` (merge into existing blocks):

```python
import json
from fantasy_baseball.analysis.draft_value import parse_full_season_lines
from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
from fantasy_baseball.web.season_data import unwrap_cache_envelope
```

Add the function:

```python
def load_current_full_season_lines() -> dict:
    """Fresh Upstash read of cache:full_season_projections (YTD+ROS blend), parsed to
    the by-name map. Fails loud if the blob is missing/empty -- never silently serve
    preseason under a `current` label."""
    kv = build_explicit_upstash_kv()
    raw = kv.get(redis_key(CacheKey.FULL_SEASON_PROJECTIONS))
    if raw is None:
        raise SystemExit("cache:full_season_projections missing in Upstash; run a refresh first.")
    payload = unwrap_cache_envelope(json.loads(raw) if isinstance(raw, str) else raw)
    if not isinstance(payload, dict) or not (payload.get("hitters") or payload.get("pitchers")):
        raise SystemExit("cache:full_season_projections is empty; run a refresh first.")
    _by_mlbam, by_name = parse_full_season_lines(payload)
    return by_name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scripts/test_keeper_value_script.py -k load_current -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/keeper_value.py tests/test_scripts/test_keeper_value_script.py
git commit -m "feat(keeper-value): fresh-Upstash current full-season loader (fail-loud)"
```

---

### Task 4: Wire the `--anchor` toggle into `build_results`

**Files:**
- Modify: `scripts/keeper_value.py` (`_parse_args`, `build_results`, `main`)
- Test: `tests/test_scripts/test_keeper_value_script.py`, `tests/test_analysis/test_keeper_value.py`

**Interfaces:**
- Consumes: `overlay_current_anchors`, `mark_preseason_fallback` (Task 1), `load_current_full_season_lines` (Task 3).
- Produces: `build_results(base_year, horizon, *, anchor="current")`; `--anchor` CLI arg (choices `current`/`preseason`, default `current`).

- [ ] **Step 1: Write the failing tests**

Add the CLI test to `tests/test_scripts/test_keeper_value_script.py`:

```python
def test_parse_args_anchor_default_is_current():
    assert script._parse_args([]).anchor == "current"
    assert script._parse_args(["--anchor", "preseason"]).anchor == "preseason"


def test_parse_args_anchor_rejects_invalid():
    with pytest.raises(SystemExit):
        script._parse_args(["--anchor", "bogus"])
```

Add the I/O-free value regression to `tests/test_analysis/test_keeper_value.py` (reuses the existing `_tiny_scale_and_board` fixture; proves a breakout anchor scores strictly higher than a modest one through the real pipeline):

```python
def test_breakout_anchor_scores_higher_than_modest():
    board, scale = _tiny_scale_and_board()
    row = board.iloc[0]
    zips_by_year = {
        2026: {"hr": 25, "r": 80, "rbi": 80, "sb": 5, "ab": 550, "avg": 0.260},
        2027: {"hr": 26, "r": 82, "rbi": 82, "sb": 5, "ab": 550, "avg": 0.262},
        2028: {"hr": 27, "r": 84, "rbi": 84, "sb": 5, "ab": 550, "avg": 0.264},
    }
    modest = {**row.to_dict(), "hr": 20, "r": 70, "rbi": 70, "sb": 4, "ab": 540, "avg": 0.255}
    breakout = {**row.to_dict(), "hr": 40, "r": 100, "rbi": 105, "sb": 9, "ab": 560, "avg": 0.300}
    common = dict(positions=list(row["positions"]), player_type=row["player_type"], zips_by_year=zips_by_year, scale=scale)
    lo = kv.discounted_total(kv.per_year_var(modest, **common)[0], 2026, 0.8, 3)
    hi = kv.discounted_total(kv.per_year_var(breakout, **common)[0], 2026, 0.8, 3)
    assert hi > lo
```

`per_year_var(anchor_line, positions, player_type, zips_by_year, scale, *, base_year, horizon, ...)` returns `(pyv, flags)`, so `[0]` is the per-year dict; `positions`/`player_type`/`zips_by_year`/`scale` are positional-or-keyword and pass cleanly as `**common`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scripts/test_keeper_value_script.py -k anchor tests/test_analysis/test_keeper_value.py -k breakout -q`
Expected: FAIL -- `_parse_args` has no `anchor` attribute; regression may already pass (it exercises existing code) -- if it passes, keep it as a guard.

- [ ] **Step 3: Add the `--anchor` arg**

In `_parse_args`, add:

```python
    ap.add_argument(
        "--anchor",
        choices=["current", "preseason"],
        default="current",
        help="anchor the 2026 base on current-season talent (YTD+ROS, default) or the "
        "preseason blend. current requires a synced cache:full_season_projections.",
    )
```

- [ ] **Step 4: Thread `anchor` through `build_results` and `main`**

Change `build_results` to accept `anchor` and apply the overlay in `current` mode:

```python
def build_results(base_year: int, horizon: int, *, anchor: str = "current"):
    conn = get_connection()
    try:
        hitters, pitchers = get_blended_projections(conn)
        positions = get_positions(conn)
    finally:
        conn.close()
    config = load_config(CONFIG_PATH)
    current_keys: set[str] = set()
    if anchor == "current":
        by_name = load_current_full_season_lines()
        hitters, pitchers, current_keys = overlay_current_anchors(hitters, pitchers, by_name)
        board_keys = {
            rank_key(str(n), pt)
            for df, pt in ((hitters, "hitter"), (pitchers, "pitcher"))
            for n in df["name"]
        }
        skipped = sum(1 for k in by_name if k not in board_keys)
        if skipped:
            print(
                f"[keeper-value] {skipped} current-blob players absent from the "
                f"preseason board (skipped; see spec follow-up)",
                file=sys.stderr,
            )
    board, scale = build_board_from_frames(
        hitters,
        pitchers,
        positions,
        roster_slots=config.roster_slots or None,
        num_teams=config.num_teams,
        sgp_overrides=config.sgp_overrides,
    )
    indices = {
        year: zips_index(*load_zips_year(PROJECTIONS_ROOT, year))
        for year in range(base_year, base_year + horizon)
    }
    candidate_ids = resolve_candidate_ids(board, CANDIDATES)
    results = []
    for _, row in board.iterrows():
        results.append(
            keeper_value(
                row["player_id"],
                row["name"],
                row.to_dict(),
                list(row["positions"]),
                str(row["player_type"]),
                _zips_by_year(_fg_id(row), row["name"], row["player_type"], indices),
                scale,
                base_year=base_year,
                horizon=horizon,
            )
        )
    if anchor == "current":
        results = mark_preseason_fallback(results, current_keys)
    return results, candidate_ids
```

Import the Task-1 functions at the top of `scripts/keeper_value.py`. (`rank_key` for
the skip count and `sys` are already imported -- lines 34 and 10 -- so no new
sgp.rankings/sys import is needed.)

```python
from fantasy_baseball.analysis.keeper_value import (
    discounted_total,          # existing
    mark_preseason_fallback,   # new
    out_year_share,            # existing
    overlay_current_anchors,   # new
)
```

**Cross-caller note (verify, do not skip):** `scripts/keeper_trades.py` also calls
`build_results(base_year=..., horizon=...)`. The new keyword-only `anchor="current"`
default means the keeper-**trade** generator now uses current-season keeper values
and requires a synced `cache:full_season_projections` (fail-loud) -- this is the
intended "trade generator inherits the fix" behavior from the spec. It is
signature-compatible (both existing call sites omit `anchor`). Task 5 verification
re-runs the keeper_trades suites to confirm nothing breaks.

**Characterization (preseason unchanged):** `preseason` mode is behavior-preserving
**by construction** -- every new line in `build_results` is guarded by
`if anchor == "current":`, so `--anchor preseason` executes the exact pre-change
path. The automated no-op is Task 1's `test_overlay_keeps_preseason_when_no_current_line`
(empty overlay = identity); the manual baseline (Step 6a) confirms end-to-end.

In `main`, pass the flag:

```python
    results, candidate_ids = build_results(base_year=BASE_YEAR, horizon=args.horizon, anchor=args.anchor)
```

- [ ] **Step 5: Run the automated tests**

Run: `python -m pytest tests/test_scripts/test_keeper_value_script.py tests/test_analysis/test_keeper_value.py -q`
Expected: PASS (all, including the new anchor + breakout tests).

- [ ] **Step 6: Manual end-to-end verification**

`build_results` is I/O-bound (local SQLite blend + ZiPS CSVs + Upstash), so its integration is verified by running the tool against live data:

```bash
python scripts/keeper_value.py --anchor preseason --limit 40   # baseline (old behavior)
python scripts/keeper_value.py --anchor current  --limit 40    # new default
```

Confirm: (a) `current` runs without error; (b) a known 2026 breakout (James Wood) ranks materially higher under `current` than `preseason`; (c) players lacking sufficient 2026 data show the `anchor_preseason_fallback` flag; (d) `--anchor current` with an unsynced blob fails loud with the "run a refresh" message (temporarily rename/clear the blob or run off a cold KV to check).

- [ ] **Step 7: Commit**

```bash
git add scripts/keeper_value.py tests/
git commit -m "feat(keeper-value): --anchor current|preseason (current-season anchor default)"
```

---

## End-of-effort verification

Run at repo root and fix every failure before declaring done:

```bash
# includes the keeper_trades suites: build_results' new default anchor flows into
# the trade generator, so confirm it still passes.
python -m pytest tests/test_analysis/test_keeper_value.py tests/test_analysis/test_draft_value.py tests/test_scripts/test_keeper_value_script.py tests/test_analysis/test_keeper_trades.py tests/test_scripts/test_keeper_trades_script.py -q
python -m ruff check src/fantasy_baseball/analysis/keeper_value.py src/fantasy_baseball/analysis/draft_value.py scripts/keeper_value.py tests/
python -m ruff format --check src/fantasy_baseball/analysis/keeper_value.py src/fantasy_baseball/analysis/draft_value.py scripts/keeper_value.py tests/
python -m mypy src/fantasy_baseball/analysis/keeper_value.py src/fantasy_baseball/analysis/draft_value.py
```

(`analysis/` is under mypy; `scripts/` is not. `draft_value.py` and `keeper_value.py` are both in the mypy set -- keep them clean.)

## Self-review notes (author)

- **Spec coverage:** anchor overlay (Task 1), min-PT floor gate (Task 1, `overlay_current_anchors`), name::player_type join (Task 1 via `rank_key`), fresh-Upstash read + fail-loud (Task 3), mode toggle + preseason no-op (Task 4), fallback flag (Task 1 + wired Task 4), report surfacing (render already emits `flags`), regression + characterization + fail-loud tests (Tasks 1/3/4). Injury known-limitation is accepted (no task, by design). Call-ups absent from the preseason board are out of scope (spec non-goal).
- **Type consistency:** `overlay_current_anchors`/`mark_preseason_fallback` names and signatures are identical across Tasks 1 and 4; `parse_full_season_lines` identical across Tasks 2 and 3.
- **No placeholders:** all steps carry real code; the one adjustable point (matching `per_year_var`'s exact keyword call shape in the breakout test) is flagged to the existing in-file example rather than left blank.
