# Keeper-Asset-Value Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-22-keeper-value-design.md`

**Goal:** Add a reusable, general keeper-asset-value metric that scores any player on discounted multi-year VAR (2026 blend anchor + ZiPS out-year trajectory), plus a ranked report script that sweeps the discount rate.

**Architecture:** A pure-math module `analysis/keeper_value.py` composes the existing full-season SGP->VAR path (`calculate_player_sgp` + `calculate_var`) over three years: year 2026 uses the player's blended anchor line; out-years scale that anchor per-stat by ZiPS's own year-over-year ratios (clamped), then run through the same scoring. A thin I/O script `scripts/keeper_value.py` builds the shared league context via `build_board_from_frames` (returns `ScaleInputs`), loads the manual ZiPS CSV exports, scores every board player, and renders an ASCII table swept across discount rates. No network I/O; risk modeling is out of scope for v1.

**Tech Stack:** Python 3, pandas, pytest. Reuses `fantasy_baseball.sgp.player_value`, `fantasy_baseball.sgp.var`, `fantasy_baseball.sgp.replacement`, `fantasy_baseball.draft.board`, `fantasy_baseball.data.fangraphs`, `fantasy_baseball.data.db`.

## Global Constraints

- **ASCII-only** in all source, log messages, format strings, and report output (Windows cp1252 stdout). Use `sigma`, `--`, `->`, straight quotes; never Unicode glyphs.
- **Never key on bare names.** Use `normalize_name` / `name::player_type` ids; tie-break namesake collisions by VAR (`draft/keepers.py::find_keeper_match`).
- **No `x or default` for numeric defaults.** Use `v if v is not None else default`. `V == 0.0` is a real value, never falsy-sunk. Use `utils.constants.safe_float` for stat reads.
- **Reuse, do not fork, the scoring math.** SGP comes from `sgp/player_value.py::calculate_player_sgp`; VAR from `sgp/var.py::calculate_var`. Do NOT use `analysis/draft_value.py::_sgp` (in-season to-date seam).
- **Stat field names (lowercase):** hitters `r, hr, rbi, sb, ab, avg` (also `h`, unused by SGP); pitchers `w, k, sv, ip, era, whip`. `calculate_player_sgp` reads `avg`/`era`/`whip` as rates directly.
- **Constants:** `DEFAULT_TEAM_AB = 5500`, `DEFAULT_TEAM_IP = 1300`, `PlayerType.HITTER = "hitter"`, `PlayerType.PITCHER = "pitcher"`, `Category` is a plain `Enum` (compare members, e.g. `Category.SV`).

---

## File Structure

- **Create** `src/fantasy_baseball/analysis/keeper_value.py` — pure metric math (dataclass, constants, scaling, per-year VAR, discounting, transparency). No file/DB/network I/O.
- **Create** `tests/test_analysis/test_keeper_value.py` — unit tests for the module.
- **Create** `scripts/keeper_value.py` — I/O + orchestration + ASCII report. Importable (guarded `main()`).
- **Create** `tests/test_scripts/test_keeper_value_script.py` — tests for the loader + candidate-highlight helpers.

---

### Task 1: Module skeleton -- dataclass, constants, and the scaling engine

**Files:**
- Create: `src/fantasy_baseball/analysis/keeper_value.py`
- Test: `tests/test_analysis/test_keeper_value.py`

**Interfaces:**
- Consumes: `utils.constants.safe_float`.
- Produces:
  - `KeeperValueResult` dataclass (frozen) with fields: `player_id: str`, `name: str`, `per_year_var: dict[int, float]`, `total: float`, `used_fallback: bool`, `flags: list[str]`, `pct_from_out_years: float | None`, `pct_from_saves: float | None`.
  - Module constants: `DEFAULT_DISCOUNT = 0.80`, `DEFAULT_HORIZON = 3`, `DEFAULT_RATIO_BAND = (0.25, 2.5)`, `DEFAULT_MIN_AB = 100.0`, `DEFAULT_MIN_IP = 20.0`, `EPS = 1e-6`, `DEFAULT_EPS_SHARE = 1.0`, `HITTER_FIELDS = ("r", "hr", "rbi", "sb", "ab", "avg")`, `PITCHER_FIELDS = ("w", "k", "sv", "ip", "era", "whip")`.
  - `_clamp_ratio(numer: float, denom: float, band: tuple[float, float], eps: float) -> float | None` — returns the clamped ratio, or `None` when `abs(denom) < eps` (ratio undefined).
  - `_scale_line(anchor: Mapping[str, Any], zips_base: Mapping[str, Any], zips_y: Mapping[str, Any], player_type: str, band: tuple[float, float], eps: float) -> dict[str, Any]` — returns a copy of `anchor` with each scored field multiplied by its clamped ZiPS ratio; a field whose ratio is `None` is left at the anchor value.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis/test_keeper_value.py
from fantasy_baseball.analysis import keeper_value as kv


def test_clamp_ratio_clamps_to_band():
    band = (0.25, 2.5)
    assert kv._clamp_ratio(10.0, 2.0, band, kv.EPS) == 2.5   # 5.0 -> clamp hi
    assert kv._clamp_ratio(1.0, 10.0, band, kv.EPS) == 0.25  # 0.1 -> clamp lo
    assert kv._clamp_ratio(3.0, 4.0, band, kv.EPS) == 0.75   # in-band


def test_clamp_ratio_none_on_tiny_denominator():
    assert kv._clamp_ratio(5.0, 0.0, (0.25, 2.5), kv.EPS) is None


def test_scale_line_scales_scored_fields_and_keeps_flat_on_none():
    anchor = {"r": 100.0, "hr": 30.0, "rbi": 90.0, "sb": 10.0, "ab": 500.0, "avg": 0.280}
    zips_base = {"r": 90.0, "hr": 25.0, "rbi": 80.0, "sb": 0.0, "ab": 450.0, "avg": 0.270}
    zips_y = {"r": 99.0, "hr": 20.0, "rbi": 88.0, "sb": 5.0, "ab": 441.0, "avg": 0.2565}
    out = kv._scale_line(anchor, zips_base, zips_y, "hitter", (0.25, 2.5), kv.EPS)
    assert out["r"] == 100.0 * (99.0 / 90.0)         # 1.10
    assert out["hr"] == 30.0 * (20.0 / 25.0)          # 0.80
    assert round(out["avg"], 4) == round(0.280 * (0.2565 / 0.270), 4)  # rate scaled directly
    assert out["sb"] == 10.0                          # zips_base sb == 0 -> ratio None -> flat
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_keeper_value.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError: module ... has no attribute '_clamp_ratio'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/analysis/keeper_value.py
"""General keeper-asset-value metric: discounted multi-year VAR.

Year 2026 uses a player's blended anchor line; out-years scale that anchor
per-stat by ZiPS's own year-over-year ratios (clamped), then run through the
same full-season SGP -> VAR path the draft board uses. Pure math, no I/O.
See docs/superpowers/specs/2026-07-22-keeper-value-design.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fantasy_baseball.utils.constants import safe_float

DEFAULT_DISCOUNT = 0.80
DEFAULT_HORIZON = 3
DEFAULT_RATIO_BAND = (0.25, 2.5)
DEFAULT_MIN_AB = 100.0
DEFAULT_MIN_IP = 20.0
EPS = 1e-6
DEFAULT_EPS_SHARE = 1.0

HITTER_FIELDS = ("r", "hr", "rbi", "sb", "ab", "avg")
PITCHER_FIELDS = ("w", "k", "sv", "ip", "era", "whip")


@dataclass(frozen=True)
class KeeperValueResult:
    player_id: str
    name: str
    per_year_var: dict[int, float]
    total: float
    used_fallback: bool
    flags: list[str]
    pct_from_out_years: float | None
    pct_from_saves: float | None


def _fields_for(player_type: str) -> tuple[str, ...]:
    return HITTER_FIELDS if player_type == "hitter" else PITCHER_FIELDS


def _clamp_ratio(
    numer: float, denom: float, band: tuple[float, float], eps: float
) -> float | None:
    if abs(denom) < eps:
        return None
    lo, hi = band
    return max(lo, min(hi, numer / denom))


def _scale_line(
    anchor: Mapping[str, Any],
    zips_base: Mapping[str, Any],
    zips_y: Mapping[str, Any],
    player_type: str,
    band: tuple[float, float],
    eps: float,
) -> dict[str, Any]:
    out = dict(anchor)
    for field in _fields_for(player_type):
        ratio = _clamp_ratio(
            safe_float(zips_y.get(field, 0)), safe_float(zips_base.get(field, 0)), band, eps
        )
        if ratio is None:
            continue  # undefined ratio -> hold the anchor value flat for this field
        out[field] = safe_float(anchor.get(field, 0)) * ratio
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_keeper_value.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/keeper_value.py tests/test_analysis/test_keeper_value.py
git commit -m "feat(keeper-value): module skeleton + ZiPS trajectory scaling engine"
```

---

### Task 2: `_value_of_line` -- full-season SGP -> VAR for one line

**Files:**
- Modify: `src/fantasy_baseball/analysis/keeper_value.py`
- Test: `tests/test_analysis/test_keeper_value.py`

**Interfaces:**
- Consumes: `draft.board.ScaleInputs` (fields `denoms`, `repl_rates`, `replacement_levels`, `team_ab`, `team_ip`); `sgp.player_value.calculate_player_sgp`; `sgp.var.calculate_var`; `models.player.PlayerType`.
- Produces: `_value_of_line(line: Mapping[str, Any], positions: list[str], player_type: str, scale: "ScaleInputs") -> float` — the VAR of a full-season line on the board's scale. `_line_sgp(line, player_type, scale) -> float` — the same line's total SGP (reused by the saves-share column).

- [ ] **Step 1: Write the failing test**

This test pins that the module reproduces the board's own scoring for a line, using a real `ScaleInputs`. It builds a tiny two-player pool so `build_board_from_frames` yields a scale and a board `var`, then asserts `_value_of_line` on the same line matches that `var`.

```python
# tests/test_analysis/test_keeper_value.py (add)
import pandas as pd
from fantasy_baseball.draft.board import build_board_from_frames


def _tiny_scale_and_board():
    hitters = pd.DataFrame([
        {"name": "Star Bat", "r": 100, "hr": 35, "rbi": 100, "sb": 15, "ab": 550, "h": 165, "avg": 0.300},
        {"name": "Meh Bat", "r": 60, "hr": 12, "rbi": 55, "sb": 5, "ab": 480, "h": 120, "avg": 0.250},
    ])
    pitchers = pd.DataFrame([
        {"name": "Ace Arm", "w": 15, "k": 220, "sv": 0, "ip": 190, "era": 3.10, "whip": 1.05},
        {"name": "Closer Guy", "w": 4, "k": 90, "sv": 35, "ip": 65, "era": 2.70, "whip": 1.00},
    ])
    positions = {"Star Bat": ["OF"], "Meh Bat": ["2B"], "Ace Arm": ["SP"], "Closer Guy": ["RP"]}
    board, scale = build_board_from_frames(hitters, pitchers, positions)
    return board, scale


def test_value_of_line_matches_board_var():
    from fantasy_baseball.analysis import keeper_value as kv
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    line = row.to_dict()
    v = kv._value_of_line(line, list(row["positions"]), row["player_type"], scale)
    assert abs(v - float(row["var"])) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_keeper_value.py::test_value_of_line_matches_board_var -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_value_of_line'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/analysis/keeper_value.py (add imports at top)
import pandas as pd

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.var import calculate_var
```

```python
# src/fantasy_baseball/analysis/keeper_value.py (add functions)
def _line_sgp(line: Mapping[str, Any], player_type: str, scale) -> float:
    series = pd.Series({**dict(line), "player_type": PlayerType(player_type)})
    return calculate_player_sgp(
        series,
        denoms=scale.denoms,
        replacement_avg=scale.repl_rates["avg"],
        replacement_era=scale.repl_rates["era"],
        replacement_whip=scale.repl_rates["whip"],
        team_ab=scale.team_ab,
        team_ip=scale.team_ip,
    )


def _value_of_line(
    line: Mapping[str, Any], positions: list[str], player_type: str, scale
) -> float:
    total_sgp = _line_sgp(line, player_type, scale)
    series = pd.Series(
        {
            **dict(line),
            "player_type": PlayerType(player_type),
            "positions": list(positions),
            "total_sgp": total_sgp,
        }
    )
    return float(calculate_var(series, scale.replacement_levels))
```

Note: full-season lines route the SP/RP floor correctly via `player["ip"]`, so no `role_ip` override is needed (spec residual note confirmed).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_keeper_value.py::test_value_of_line_matches_board_var -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/keeper_value.py tests/test_analysis/test_keeper_value.py
git commit -m "feat(keeper-value): _value_of_line reuses board SGP->VAR (parity test)"
```

---

### Task 3: `per_year_var` -- year loop with approach B / A fallbacks and flags

**Files:**
- Modify: `src/fantasy_baseball/analysis/keeper_value.py`
- Test: `tests/test_analysis/test_keeper_value.py`

**Interfaces:**
- Consumes: `_scale_line`, `_value_of_line`, `_fields_for`.
- Produces: `per_year_var(anchor_line, positions, player_type, zips_by_year, scale, *, base_year=2026, horizon=DEFAULT_HORIZON, ratio_band=DEFAULT_RATIO_BAND, min_pt=None, eps=EPS) -> tuple[dict[int, float], list[str], bool]`. Returns `(per_year_var_map, flags, used_fallback)`.
  - `zips_by_year: Mapping[int, Mapping[str, Any] | None]` maps each year to that player's ZiPS line (or `None` if absent that year).
  - Year `base_year`: uses `anchor_line` directly (or ZiPS base line via approach A if `anchor_line` is falsy).
  - Out-years: approach B (scale anchor by ZiPS ratios) normally; approach A (ZiPS line straight) when `anchor_line` is falsy, the ZiPS base line is missing, or the ZiPS base line is below the min-PT threshold; `V = 0.0` with flag `no_zips_<year>` when that year's ZiPS line is missing.
  - `min_pt`: threshold on the ZiPS base line's `ab` (hitter) / `ip` (pitcher); default `DEFAULT_MIN_AB` / `DEFAULT_MIN_IP`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis/test_keeper_value.py (add)
def test_per_year_var_missing_out_year_is_zero_and_flagged():
    from fantasy_baseball.analysis import keeper_value as kv
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    anchor = row.to_dict()
    # ZiPS base present, 2027 present, 2028 missing.
    zips_by_year = {
        2026: anchor,
        2027: {**anchor, "hr": anchor["hr"] * 0.9},
        2028: None,
    }
    pyv, flags, used_fallback = kv.per_year_var(
        anchor, list(row["positions"]), row["player_type"], zips_by_year, scale
    )
    assert set(pyv) == {2026, 2027, 2028}
    assert pyv[2028] == 0.0
    assert "no_zips_2028" in flags
    assert abs(pyv[2026] - float(row["var"])) < 1e-9  # base year == board var


def test_per_year_var_low_pt_base_falls_back_to_approach_a():
    from fantasy_baseball.analysis import keeper_value as kv
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    anchor = row.to_dict()
    # ZiPS base line has AB below the 100 default -> out-years use approach A.
    tiny_base = {**anchor, "ab": 40}
    zips_2027 = {**anchor, "hr": 20}
    zips_by_year = {2026: tiny_base, 2027: zips_2027, 2028: zips_2027}
    pyv, flags, used_fallback = kv.per_year_var(
        anchor, list(row["positions"]), row["player_type"], zips_by_year, scale
    )
    assert used_fallback is True
    assert "fallback_A" in flags
    # Approach A: out-year V equals scoring the raw ZiPS 2027 line directly.
    expected = kv._value_of_line(zips_2027, list(row["positions"]), row["player_type"], scale)
    assert abs(pyv[2027] - expected) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_keeper_value.py -k per_year_var -v`
Expected: FAIL with `AttributeError: ... 'per_year_var'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/analysis/keeper_value.py (add)
def _below_min_pt(zips_line: Mapping[str, Any], player_type: str, min_pt: float | None) -> bool:
    if player_type == "hitter":
        thresh = min_pt if min_pt is not None else DEFAULT_MIN_AB
        return safe_float(zips_line.get("ab", 0)) < thresh
    thresh = min_pt if min_pt is not None else DEFAULT_MIN_IP
    return safe_float(zips_line.get("ip", 0)) < thresh


def per_year_var(
    anchor_line: Mapping[str, Any],
    positions: list[str],
    player_type: str,
    zips_by_year: Mapping[int, Mapping[str, Any] | None],
    scale,
    *,
    base_year: int = 2026,
    horizon: int = DEFAULT_HORIZON,
    ratio_band: tuple[float, float] = DEFAULT_RATIO_BAND,
    min_pt: float | None = None,
    eps: float = EPS,
) -> tuple[dict[int, float], list[str], bool]:
    pyv: dict[int, float] = {}
    flags: list[str] = []
    used_fallback = False
    zips_base = zips_by_year.get(base_year)

    for k in range(horizon):
        year = base_year + k
        if k == 0:
            if anchor_line:
                pyv[year] = _value_of_line(anchor_line, positions, player_type, scale)
            elif zips_base:
                used_fallback = True
                if "fallback_A" not in flags:
                    flags.append("fallback_A")
                pyv[year] = _value_of_line(zips_base, positions, player_type, scale)
            else:
                pyv[year] = 0.0
                flags.append(f"no_zips_{year}")
            continue

        zips_y = zips_by_year.get(year)
        if not zips_y:
            pyv[year] = 0.0
            flags.append(f"no_zips_{year}")
            continue

        approach_a = (
            (not anchor_line)
            or (not zips_base)
            or _below_min_pt(zips_base, player_type, min_pt)
        )
        if approach_a:
            used_fallback = True
            if "fallback_A" not in flags:
                flags.append("fallback_A")
            line = zips_y
        else:
            line = _scale_line(anchor_line, zips_base, zips_y, player_type, ratio_band, eps)
        pyv[year] = _value_of_line(line, positions, player_type, scale)

    return pyv, flags, used_fallback
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_keeper_value.py -k per_year_var -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/keeper_value.py tests/test_analysis/test_keeper_value.py
git commit -m "feat(keeper-value): per_year_var with approach-A fallback + missing-year flags"
```

---

### Task 4: `discounted_total` + `keeper_value` -- assembly, parity, youth premium

**Files:**
- Modify: `src/fantasy_baseball/analysis/keeper_value.py`
- Test: `tests/test_analysis/test_keeper_value.py`

**Interfaces:**
- Consumes: `per_year_var`, `KeeperValueResult`.
- Produces:
  - `discounted_total(pyv: Mapping[int, float], base_year: int, discount: float, horizon: int) -> float` = `sum(discount**k * pyv[base_year + k] for k in range(horizon))`.
  - `keeper_value(player_id, name, anchor_line, positions, player_type, zips_by_year, scale, *, base_year=2026, discount=DEFAULT_DISCOUNT, horizon=DEFAULT_HORIZON, ratio_band=DEFAULT_RATIO_BAND, min_pt=None, eps=EPS, eps_share=DEFAULT_EPS_SHARE) -> KeeperValueResult`. Computes `per_year_var`, the discounted `total`, and `pct_from_out_years` (guarded: `None` when `total <= eps_share`). `pct_from_saves` is set in Task 5; for now pass `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis/test_keeper_value.py (add)
def test_discounted_total_weights_by_year():
    from fantasy_baseball.analysis import keeper_value as kv
    pyv = {2026: 10.0, 2027: 10.0, 2028: 10.0}
    assert kv.discounted_total(pyv, 2026, 0.8, 3) == 10.0 + 8.0 + 6.4


def test_keeper_value_horizon_1_equals_board_var():
    from fantasy_baseball.analysis import keeper_value as kv
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    anchor = row.to_dict()
    res = kv.keeper_value(
        row["player_id"], row["name"], anchor, list(row["positions"]), row["player_type"],
        {2026: anchor}, scale, horizon=1,
    )
    assert abs(res.total - float(row["var"])) < 1e-9  # currency parity


def test_youth_premium_emerges_and_widens_as_discount_shallows():
    """Two players, identical 2026 VAR, different ZiPS decline curves.
    The flatter (younger) curve ranks higher, and the gap widens as discount rises."""
    from fantasy_baseball.analysis import keeper_value as kv
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    anchor = row.to_dict()
    pt, positions = row["player_type"], list(row["positions"])

    # Young: holds. Old: decays ~15%/yr (ZiPS ratio applied to identical anchor).
    young = {2026: anchor, 2027: anchor, 2028: anchor}
    decayed_27 = {**anchor, "r": anchor["r"] * 0.85, "hr": anchor["hr"] * 0.85,
                  "rbi": anchor["rbi"] * 0.85, "sb": anchor["sb"] * 0.85}
    decayed_28 = {**anchor, "r": anchor["r"] * 0.70, "hr": anchor["hr"] * 0.70,
                  "rbi": anchor["rbi"] * 0.70, "sb": anchor["sb"] * 0.70}
    old = {2026: anchor, 2027: decayed_27, 2028: decayed_28}

    def total(zbys, discount):
        return kv.keeper_value("y", "y", anchor, positions, pt, zbys, scale, discount=discount).total

    gap_steep = total(young, 0.60) - total(old, 0.60)
    gap_shallow = total(young, 0.90) - total(old, 0.90)
    assert gap_steep > 0                      # young always wins
    assert gap_shallow > gap_steep            # advantage grows as out-years count more


def test_keeper_value_none_share_when_total_below_eps():
    from fantasy_baseball.analysis import keeper_value as kv
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Meh Bat"].iloc[0]  # low/near-replacement value
    anchor = row.to_dict()
    res = kv.keeper_value(
        row["player_id"], row["name"], anchor, list(row["positions"]), row["player_type"],
        {2026: anchor, 2027: anchor, 2028: anchor}, scale, eps_share=1e9,  # force the guard
    )
    assert res.pct_from_out_years is None


def test_keeper_value_zero_year_is_kept_not_dropped():
    """A year whose V is exactly 0.0 is a real value: it stays in per_year_var
    and participates in the discounted sum (numeric-default guard)."""
    from fantasy_baseball.analysis import keeper_value as kv
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    anchor = row.to_dict()
    res = kv.keeper_value(
        row["player_id"], row["name"], anchor, list(row["positions"]), row["player_type"],
        {2026: anchor, 2027: anchor, 2028: None}, scale, discount=1.0,  # 2028 missing -> 0.0
    )
    assert res.per_year_var[2028] == 0.0
    assert 2028 in res.per_year_var  # not dropped
    # discount=1.0 -> total is the plain sum incl. the 0.0 term
    assert abs(res.total - (res.per_year_var[2026] + res.per_year_var[2027] + 0.0)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_keeper_value.py -k "discounted_total or keeper_value or youth" -v`
Expected: FAIL with `AttributeError: ... 'discounted_total'` / `'keeper_value'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/analysis/keeper_value.py (add)
def discounted_total(
    pyv: Mapping[int, float], base_year: int, discount: float, horizon: int
) -> float:
    return sum(discount**k * pyv.get(base_year + k, 0.0) for k in range(horizon))


def keeper_value(
    player_id: str,
    name: str,
    anchor_line: Mapping[str, Any],
    positions: list[str],
    player_type: str,
    zips_by_year: Mapping[int, Mapping[str, Any] | None],
    scale,
    *,
    base_year: int = 2026,
    discount: float = DEFAULT_DISCOUNT,
    horizon: int = DEFAULT_HORIZON,
    ratio_band: tuple[float, float] = DEFAULT_RATIO_BAND,
    min_pt: float | None = None,
    eps: float = EPS,
    eps_share: float = DEFAULT_EPS_SHARE,
) -> KeeperValueResult:
    pyv, flags, used_fallback = per_year_var(
        anchor_line, positions, player_type, zips_by_year, scale,
        base_year=base_year, horizon=horizon, ratio_band=ratio_band, min_pt=min_pt, eps=eps,
    )
    total = discounted_total(pyv, base_year, discount, horizon)
    if total <= eps_share:
        pct_out = None
    else:
        out_years = sum(
            discount**k * pyv.get(base_year + k, 0.0) for k in range(1, horizon)
        )
        pct_out = out_years / total
    return KeeperValueResult(
        player_id=player_id,
        name=name,
        per_year_var=pyv,
        total=total,
        used_fallback=used_fallback,
        flags=flags,
        pct_from_out_years=pct_out,
        pct_from_saves=None,  # set in Task 5
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_keeper_value.py -k "discounted_total or keeper_value or youth" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/keeper_value.py tests/test_analysis/test_keeper_value.py
git commit -m "feat(keeper-value): keeper_value assembly + discount + youth-premium test"
```

---

### Task 5: `pct_from_saves` transparency column

**Files:**
- Modify: `src/fantasy_baseball/analysis/keeper_value.py`
- Test: `tests/test_analysis/test_keeper_value.py`

**Interfaces:**
- Consumes: `_line_sgp`; `sgp.player_value.calculate_counting_sgp`; `utils.constants.Category`.
- Produces: `pct_from_saves(anchor_line, player_type, scale, *, eps_share=DEFAULT_EPS_SHARE) -> float | None` — SV's share of the 2026-anchor SGP: `0.0` for hitters; `None` when `abs(anchor SGP) <= eps_share`; else `calculate_counting_sgp(sv, denoms[Category.SV]) / anchor_sgp`. Wired into `keeper_value` so the result's `pct_from_saves` is populated.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis/test_keeper_value.py (add)
def test_pct_from_saves_zero_for_hitter():
    from fantasy_baseball.analysis import keeper_value as kv
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    assert kv.pct_from_saves(row.to_dict(), "hitter", scale) == 0.0


def test_pct_from_saves_positive_for_closer_and_beats_starter():
    # Robust relational assertion (avoids a magic threshold that depends on the
    # tiny fixture's pool-derived denominators): a closer's saves share is
    # positive and strictly greater than a save-less starter's (which is 0.0).
    from fantasy_baseball.analysis import keeper_value as kv
    board, scale = _tiny_scale_and_board()
    closer = board[board["name"] == "Closer Guy"].iloc[0]
    ace = board[board["name"] == "Ace Arm"].iloc[0]
    closer_share = kv.pct_from_saves(closer.to_dict(), "pitcher", scale)
    ace_share = kv.pct_from_saves(ace.to_dict(), "pitcher", scale)
    assert ace_share == 0.0                    # no saves -> 0 share
    assert closer_share is not None and closer_share > 0.0
    assert closer_share > ace_share


def test_pct_from_saves_none_when_sgp_below_eps():
    from fantasy_baseball.analysis import keeper_value as kv
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Closer Guy"].iloc[0]
    assert kv.pct_from_saves(row.to_dict(), "pitcher", scale, eps_share=1e9) is None


def test_keeper_value_populates_pct_from_saves():
    from fantasy_baseball.analysis import keeper_value as kv
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Closer Guy"].iloc[0]
    anchor = row.to_dict()
    res = kv.keeper_value(
        row["player_id"], row["name"], anchor, list(row["positions"]), "pitcher",
        {2026: anchor, 2027: anchor, 2028: anchor}, scale,
    )
    assert res.pct_from_saves is not None and res.pct_from_saves > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_keeper_value.py -k pct_from_saves -v`
Expected: FAIL with `AttributeError: ... 'pct_from_saves'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/analysis/keeper_value.py (extend imports)
from fantasy_baseball.sgp.player_value import calculate_counting_sgp, calculate_player_sgp
from fantasy_baseball.utils.constants import Category, safe_float
```

```python
# src/fantasy_baseball/analysis/keeper_value.py (add)
def pct_from_saves(
    anchor_line: Mapping[str, Any],
    player_type: str,
    scale,
    *,
    eps_share: float = DEFAULT_EPS_SHARE,
) -> float | None:
    if player_type != "pitcher":
        return 0.0
    sgp = _line_sgp(anchor_line, player_type, scale)
    if abs(sgp) <= eps_share:
        return None
    sv_sgp = calculate_counting_sgp(safe_float(anchor_line.get("sv", 0)), scale.denoms[Category.SV])
    return sv_sgp / sgp
```

Then wire it into `keeper_value` -- replace `pct_from_saves=None` in the returned `KeeperValueResult` with:

```python
        pct_from_saves=pct_from_saves(anchor_line, player_type, scale, eps_share=eps_share),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_keeper_value.py -v`
Expected: PASS (all module tests, including the four new ones).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/keeper_value.py tests/test_analysis/test_keeper_value.py
git commit -m "feat(keeper-value): pct_from_saves transparency column (SGP-based, guarded)"
```

---

### Task 6: Report script -- ZiPS loader, orchestration, discount sweep, ASCII table

**Files:**
- Create: `scripts/keeper_value.py`
- Test: `tests/test_scripts/test_keeper_value_script.py`

**Interfaces:**
- Consumes: `analysis.keeper_value` (`keeper_value`, `KeeperValueResult`); `draft.board.build_board_from_frames`; `data.db.get_connection/get_blended_projections/get_positions`; `data.fangraphs.load_projection_set`; `config.load_config`; `utils.name_utils.normalize_name`; `draft.keepers.index_by_normalized_name/find_keeper_match`.
- Produces (module-level, importable functions -- `main()` guarded by `if __name__ == "__main__"`):
  - `load_zips_year(projections_root: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]` — loads `data/projections/<year>/zips-{hitters,pitchers}.csv` via `load_projection_set`; raises `FileNotFoundError` with the exact FanGraphs ZiPS URL when either frame is empty.
  - `zips_index(hitters: pd.DataFrame, pitchers: pd.DataFrame) -> dict[str, dict[str, Any]]` — maps `normalize_name(name)::player_type` -> line dict.
  - `CANDIDATES: list[str]` — the manager's highlight set.
  - `is_candidate(name: str, candidate_norms: set[str]) -> bool` — normalized-name membership.

**Global rule reminder:** first line of `main()` region does `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`; script injects `src/` on `sys.path` per repo convention.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scripts/test_keeper_value_script.py
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import keeper_value as script  # noqa: E402


def test_load_zips_year_missing_raises_with_url(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        script.load_zips_year(tmp_path, 2027)
    assert "fangraphs.com" in str(exc.value)
    assert "2027" in str(exc.value)


def test_load_zips_year_loads_present(tmp_path):
    d = tmp_path / "2027"
    d.mkdir()
    pd.DataFrame([{"Name": "A B", "AB": 500, "H": 150, "HR": 30, "R": 90, "RBI": 95, "SB": 10, "AVG": 0.300}]).to_csv(
        d / "zips-hitters.csv", index=False
    )
    # FanGraphs (and ZiPS) exports use "SO" for strikeouts; PITCHING_COLUMN_MAP
    # normalizes SO -> k. A "K" header would NOT be recognized.
    pd.DataFrame([{"Name": "C D", "IP": 180, "W": 14, "SO": 200, "ERA": 3.2, "WHIP": 1.05, "SV": 0}]).to_csv(
        d / "zips-pitchers.csv", index=False
    )
    hitters, pitchers = script.load_zips_year(tmp_path, 2027)
    assert not hitters.empty and not pitchers.empty


def test_is_candidate_matches_by_normalized_name():
    from fantasy_baseball.utils.name_utils import normalize_name
    norms = {normalize_name(n) for n in ["Julio Rodriguez"]}
    assert script.is_candidate("Julio Rodriguez", norms) is True
    assert script.is_candidate("Julio Rodriguez", norms) == script.is_candidate("Julio Rodriguez", norms)
    assert script.is_candidate("Some Other", norms) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_keeper_value_script.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'keeper_value'` (script not created yet).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/keeper_value.py
"""Rank players by keeper-asset value (discounted multi-year VAR), swept across
discount rates. Reads projections/positions from SQLite and manual ZiPS out-year
CSV exports; does no network I/O. See
docs/superpowers/specs/2026-07-22-keeper-value-design.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fantasy_baseball.analysis.keeper_value import keeper_value  # noqa: E402
from fantasy_baseball.config import load_config  # noqa: E402
from fantasy_baseball.data.db import (  # noqa: E402
    get_blended_projections,
    get_connection,
    get_positions,
)
from fantasy_baseball.data.fangraphs import load_projection_set  # noqa: E402
from fantasy_baseball.draft.board import build_board_from_frames  # noqa: E402
from fantasy_baseball.models.player import PlayerType  # noqa: E402
from fantasy_baseball.utils.name_utils import normalize_name  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTIONS_ROOT = REPO_ROOT / "data" / "projections"
CONFIG_PATH = REPO_ROOT / "config" / "league.yaml"
DISCOUNTS = [0.60, 0.70, 0.80, 0.90]
CANDIDATES = [
    "Juan Soto", "Julio Rodriguez", "Junior Caminero",
    "CJ Abrams", "Mason Miller", "Kyle Tucker",
]
_ZIPS_URL = "https://www.fangraphs.com/projections?type=zips&stats={t}&pos=all"


def load_zips_year(projections_root: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    hitters, pitchers = load_projection_set(projections_root / str(year), "zips")
    if hitters.empty or pitchers.empty:
        raise FileNotFoundError(
            f"Missing ZiPS {year} export. Download the {year} ZiPS projections from "
            f"{_ZIPS_URL.format(t='bat')} (hitters) and {_ZIPS_URL.format(t='pit')} "
            f"(pitchers), save as data/projections/{year}/zips-hitters.csv and "
            f"data/projections/{year}/zips-pitchers.csv."
        )
    return hitters, pitchers


def zips_index(hitters: pd.DataFrame, pitchers: pd.DataFrame) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for df, ptype in [(hitters, PlayerType.HITTER), (pitchers, PlayerType.PITCHER)]:
        for _, row in df.iterrows():
            idx[f"{normalize_name(row['name'])}::{ptype}"] = row.to_dict()
    return idx


def is_candidate(name: str, candidate_norms: set[str]) -> bool:
    return normalize_name(name) in candidate_norms


def _zips_by_year(player_key: str, indices: dict[int, dict[str, dict[str, Any]]]) -> dict:
    return {year: idx.get(player_key) for year, idx in indices.items()}


def build_results(base_year: int, horizon: int):
    conn = get_connection()
    try:
        hitters, pitchers = get_blended_projections(conn)
        positions = get_positions(conn)
    finally:
        conn.close()
    config = load_config(CONFIG_PATH)
    board, scale = build_board_from_frames(
        hitters, pitchers, positions,
        roster_slots=config.roster_slots or None,
        num_teams=config.num_teams,
        sgp_overrides=config.sgp_overrides,
    )
    indices = {
        year: zips_index(*load_zips_year(PROJECTIONS_ROOT, year))
        for year in range(base_year, base_year + horizon)
    }
    results = []
    for _, row in board.iterrows():
        key = f"{row['name_normalized']}::{row['player_type']}"
        results.append(
            keeper_value(
                row["player_id"], row["name"], row.to_dict(), list(row["positions"]),
                str(row["player_type"]), _zips_by_year(key, indices), scale,
                base_year=base_year, horizon=horizon,
            )
        )
    return results


def render(results, discounts: list[float]) -> str:
    from fantasy_baseball.analysis.keeper_value import discounted_total
    candidate_norms = {normalize_name(c) for c in CANDIDATES}
    if not results:
        return "No players scored."
    base_year = min(next(iter(results)).per_year_var)
    horizon = len(next(iter(results)).per_year_var)

    # Total for every (player, discount), then rank per discount so the ranking
    # slide across discounts is explicit (spec: "total and rank at each discount").
    totals = {
        d: {r.player_id: discounted_total(r.per_year_var, base_year, d, horizon) for r in results}
        for d in discounts
    }
    ranks: dict[float, dict[str, int]] = {}
    for d in discounts:
        order = sorted(results, key=lambda r: totals[d][r.player_id], reverse=True)
        ranks[d] = {r.player_id: i + 1 for i, r in enumerate(order)}

    primary = discounts[-1]  # order rows by the most dynasty-weighted discount
    ranked = sorted(results, key=lambda r: totals[primary][r.player_id], reverse=True)

    lines = [
        "Keeper-asset value (discounted multi-year VAR); cell = total(#rank at that discount)",
        "",
    ]
    header = f"{'':1} {'Player':22} " + " ".join(f"{'d=' + format(d, '.2f'):>13}" for d in discounts)
    header += "  perYr(" + "/".join(str(base_year + k) for k in range(horizon)) + ")  %out  %sv  flags"
    lines.append(header)
    for r in ranked:
        mark = "*" if is_candidate(r.name, candidate_norms) else " "
        cells = " ".join(
            f"{totals[d][r.player_id]:7.1f}(#{ranks[d][r.player_id]:>3})" for d in discounts
        )
        per = "/".join(f"{r.per_year_var[base_year + k]:.0f}" for k in range(horizon))
        pout = "N/A " if r.pct_from_out_years is None else f"{r.pct_from_out_years * 100:3.0f}%"
        psv = "N/A " if r.pct_from_saves is None else f"{r.pct_from_saves * 100:3.0f}%"
        lines.append(f"{mark} {r.name[:22]:22} {cells}  {per:>10}  {pout} {psv}  {','.join(r.flags)}")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    results = build_results(base_year=2026, horizon=3)
    print(render(results, DISCOUNTS))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scripts/test_keeper_value_script.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/keeper_value.py tests/test_scripts/test_keeper_value_script.py
git commit -m "feat(keeper-value): ranked report script with ZiPS loader + discount sweep"
```

---

### Task 7: End-of-effort verification

**Files:** none (verification only).

- [ ] **Step 1: Full test suite for touched areas**

Run: `pytest tests/test_analysis/test_keeper_value.py tests/test_scripts/test_keeper_value_script.py -v`
Expected: all PASS.

- [ ] **Step 2: Broader regression check**

Run: `pytest tests/test_analysis tests/test_draft tests/test_sgp -q`
Expected: no new failures (the module reuses board/SGP/VAR; confirm nothing regressed).

- [ ] **Step 3: Lint + format + dead-code**

Run: `ruff check . && ruff format --check . && vulture src/fantasy_baseball/analysis/keeper_value.py scripts/keeper_value.py`
Expected: zero ruff violations; no formatting drift; no NEW vulture findings (report pre-existing unrelated ones).

- [ ] **Step 4: mypy (only if covered)**

Run: `python -c "import tomllib,pathlib; cfg=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print(cfg['tool']['mypy'].get('files'))"` then, if `analysis/keeper_value.py` or `scripts/keeper_value.py` falls under the listed `files`, run `mypy` and fix findings. If not listed, state that it is out of mypy coverage.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore(keeper-value): lint/format/type fixes from end-of-effort verification"
```

---

## Notes for the implementer

- **You cannot run `scripts/keeper_value.py` end-to-end without the manual ZiPS 2027/2028 CSVs** (`data/projections/2027/zips-*.csv`, `2028/`). That is expected -- `load_zips_year` raises an actionable error naming the FanGraphs URL. The unit tests do not need those files; they build synthetic frames. Do not fabricate ZiPS data to make `main()` run.
- **The 2026 anchor uses the blended board line; ZiPS 2026 is only the trajectory denominator.** Two different 2026 sources by design.
- **`build_board_from_frames` filters to AB>=50 / IP>=10** and holds 2026 positions/floors -- consistent with the spec's held-constant assumptions.
- If `config.sgp_overrides` / `config.roster_slots` / `config.num_teams` attribute names differ from what `build_draft_board` passes, mirror `draft/board.py::rebuild_board` exactly (it is the reference caller).
