# deltaRoto Confidence Indicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a `mean +/- SD` confidence band on every deltaRoto (colored by whether the band crosses zero) across the roster audit, trade builder, player comparison, and lineup pages, so a coin-flip swap is visually distinct from a real edge.

**Architecture:** A new core function `compute_delta_roto_band(before_players, after_players, field_stats, ...)` Monte-Carlos a roster change: it samples each side's realized stats with the existing, calibrated `simulation._apply_variance` (ROS-remaining, correlated, scaled by `fraction_remaining`) using common random numbers across before/after, scores each draw against a fixed field with `scoring.score_roto_dict`, and returns `(mean, sd, p_positive)`. Every surface builds a before/after player pair and calls this one function. A small `band_class()` helper maps a band to a CSS verdict (`real` / `coin-flip` / `downgrade`). The point-estimate `compute_delta_roto` stays; the band's `mean` must track it (the linchpin test).

**Tech Stack:** Python 3.11, numpy (existing MC), Flask + Jinja templates, pytest, ruff. Spec: `docs/superpowers/specs/2026-05-21-deltaroto-confidence-design.md`.

---

## File Structure

- `src/fantasy_baseball/lineup/delta_roto.py` (MODIFY) -- add `DeltaRotoBand` dataclass + `compute_delta_roto_band(...)` core function and a one-for-one convenience wrapper. Owns the band metric.
- `src/fantasy_baseball/lineup/band_format.py` (CREATE) -- `band_class(mean, sd)` -> `"real" | "coin-flip" | "downgrade"` and `band_label(mean, sd)` -> `"+1.9 +/- 2.3"`. Pure formatting/verdict, no deps. Shared by every surface so the rule lives in one place.
- `src/fantasy_baseball/lineup/roster_audit.py` (MODIFY) -- attach a `band` to each candidate dict.
- `src/fantasy_baseball/trades/multi_trade.py` (MODIFY) -- add a band on the trade total.
- `src/fantasy_baseball/lineup/optimizer.py` (MODIFY) -- attach a band to each `HitterAssignment` / `PitcherStarter`.
- `src/fantasy_baseball/web/season_data.py` (MODIFY) -- `compute_comparison_standings` returns a band.
- `src/fantasy_baseball/web/season_routes.py` (MODIFY) -- serialize band fields in `/api/evaluate-trade`.
- Templates (MODIFY): `roster_audit.html`, `waivers_trades.html` (JS), `players.html` (JS), `lineup.html` + `_lineup_hitters_tbody.html` + `_lineup_pitchers_tbody.html`.
- `src/fantasy_baseball/web/static/season.css` (MODIFY) -- band verdict classes.
- Tests: `tests/test_lineup/test_delta_roto_band.py` (CREATE), `tests/test_lineup/test_band_format.py` (CREATE), plus assertions added to existing audit / trade / comparison / lineup tests.

Implementation order: **Task 2 (band_format) is built first** so Task 1's `DeltaRotoBand.to_dict()` can import `band_class` and attach a `verdict` string computed ONCE in Python. Every surface (Jinja + JS) reads `band.verdict` and maps it to a CSS class -- the crosses-zero rule is never re-derived in a template or in JS. Tasks 3-6 (surfaces) depend only on Tasks 1-2 and are independent of each other.

---

## Task 1: Core band metric

**Files:**
- Modify: `src/fantasy_baseball/lineup/delta_roto.py`
- Test: `tests/test_lineup/test_delta_roto_band.py` (create)

Reference current code:
- `simulation._apply_variance(players, player_type, rng, injuries_out, fraction_remaining)` returns `list[dict]` of realized counting stats; `players` are FLAT dicts (use `simulation._flatten_full_season(p)` to flatten a `Player`). It internally draws playing-time scales + a correlated multivariate-normal once per call.
- `simulate_season` (simulation.py:148-167) shows the canonical sum of realized hitters/pitchers into team totals + rate recombination via `calculate_avg/era/whip`.
- `scoring.score_roto_dict(all_team_stats, team_sds=None)` returns `{team: {"R_pts":..., "total":...}}`; pass `team_sds=None` for rank-based scoring of a single realized draw.

Algorithm (player-level MC, common random numbers, ROS-remaining, field fixed):
1. `union = after_players + [p for p in before_players if p not in after_players]` (every distinct player across both sides).
2. For each of `n_draws` iterations, sample realized ROS stats for the whole `union` ONCE (so shared players are identical in before and after -- CRN). Flatten each player, split by type, call `_apply_variance` for hitters and pitchers with `fraction_remaining`, key the results by name.
3. Sum the `before_players` subset and the `after_players` subset of that one realization into two `CategoryStats` (counting sums + rate recombination, exactly like `simulate_season`).
4. Score each against the FIXED `field_stats` (other teams' point-estimate `CategoryStats`): `score_roto_dict({team_name: before, **field}, team_sds=None)` and likewise for after. `dRoto_i = after_total - before_total`.
5. Aggregate: `mean = float(np.mean(deltas))`, `sd = float(np.std(deltas))`, `p_positive = float(np.mean(np.asarray(deltas) > 0))`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lineup/test_delta_roto_band.py
import numpy as np
import pytest

from fantasy_baseball.lineup.delta_roto import (
    DeltaRotoBand,
    compute_delta_roto_band,
)
from fantasy_baseball.models.player import HitterStats, Player, PlayerType
from fantasy_baseball.models.standings import CategoryStats


def _hitter(name, **ros):
    base = dict(pa=600, ab=540, h=150, r=85, hr=25, rbi=85, sb=10)
    base.update(ros)
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=["OF"],
        rest_of_season=HitterStats(**base),
        full_season_projection=HitterStats(**base),
    )


def _field():
    # Eight rival teams at varied strengths so categories aren't degenerate.
    field = {}
    for i in range(8):
        field[f"Team{i}"] = CategoryStats(
            r=800 + i * 15, hr=220 + i * 5, rbi=780 + i * 12, sb=110 + i * 4,
            avg=0.255 + i * 0.002, w=85, k=1300, sv=70, era=3.8, whip=1.20,
        )
    return field


def test_band_returns_mean_sd_ppos():
    before = [_hitter(f"H{i}") for i in range(13)]
    after = before[:-1] + [_hitter("BigBat", hr=45, r=105, rbi=110)]
    band = compute_delta_roto_band(
        before, after, _field(), "Me", fraction_remaining=0.6, n_draws=300, seed=1
    )
    assert isinstance(band, DeltaRotoBand)
    assert band.sd > 0
    assert 0.0 <= band.p_positive <= 1.0


def test_band_is_deterministic_for_fixed_seed():
    before = [_hitter(f"H{i}") for i in range(13)]
    after = before[:-1] + [_hitter("BigBat", hr=45)]
    a = compute_delta_roto_band(before, after, _field(), "Me", fraction_remaining=0.6, n_draws=200, seed=7)
    b = compute_delta_roto_band(before, after, _field(), "Me", fraction_remaining=0.6, n_draws=200, seed=7)
    assert a.mean == b.mean and a.sd == b.sd and a.p_positive == b.p_positive


def test_identity_swap_has_near_zero_mean():
    before = [_hitter(f"H{i}") for i in range(13)]
    after = list(before)  # no change
    band = compute_delta_roto_band(before, after, _field(), "Me", fraction_remaining=0.6, n_draws=300, seed=3)
    assert abs(band.mean) < 0.05


def test_mean_tracks_point_estimate(real_point_estimate_fixture=None):
    # The band mean must approximate the existing point deltaRoto. Build the
    # same swap as a ProjectedStandings + compute_delta_roto, and assert the
    # band mean is within tolerance. (Implementer: wire compute_delta_roto on
    # the same before/after; tolerance ~0.5 roto pts at n_draws>=500.)
    pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lineup/test_delta_roto_band.py -v`
Expected: FAIL with `ImportError: cannot import name 'DeltaRotoBand'`.

- [ ] **Step 3: Implement `DeltaRotoBand` + `compute_delta_roto_band`**

Add to `src/fantasy_baseball/lineup/delta_roto.py`:

```python
@dataclass
class DeltaRotoBand:
    mean: float
    sd: float
    p_positive: float

    def to_dict(self) -> dict[str, float | str]:
        # verdict computed once here so no template/JS re-derives the rule.
        from fantasy_baseball.lineup.band_format import band_class

        return {
            "mean": round(self.mean, 2),
            "sd": round(self.sd, 2),
            "p_positive": round(self.p_positive, 3),
            "verdict": band_class(self.mean, self.sd),
        }


def _sum_realized(realized_rows: list[dict]) -> CategoryStats:
    """Sum realized per-player dicts (from _apply_variance) into CategoryStats."""
    from fantasy_baseball.models.player import PlayerType
    from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

    h = [r for r in realized_rows if r["player_type"] == PlayerType.HITTER]
    p = [r for r in realized_rows if r["player_type"] == PlayerType.PITCHER]
    ab = sum(x["ab"] for x in h)
    hits = sum(x["h"] for x in h)
    ip = sum(x["ip"] for x in p)
    er = sum(x["er"] for x in p)
    bb = sum(x["bb"] for x in p)
    ha = sum(x["h_allowed"] for x in p)
    return CategoryStats(
        r=sum(x["r"] for x in h), hr=sum(x["hr"] for x in h),
        rbi=sum(x["rbi"] for x in h), sb=sum(x["sb"] for x in h),
        avg=calculate_avg(hits, ab),
        w=sum(x["w"] for x in p), k=sum(x["k"] for x in p),
        sv=sum(x.get("sv", 0) for x in p),
        era=calculate_era(er, ip), whip=calculate_whip(bb, ha, ip),
    )


def compute_delta_roto_band(
    before_players: list,
    after_players: list,
    field_stats: Mapping[str, "CategoryStats"],
    team_name: str,
    fraction_remaining: float,
    *,
    n_draws: int = 400,
    seed: int = 0,
) -> DeltaRotoBand:
    """Monte-Carlo the deltaRoto of a before->after roster change.

    Samples each side's realized ROS stats with the calibrated variance
    model (common random numbers across before/after so shared players are
    identical), scores each draw against the fixed ``field_stats``, and
    returns mean/sd/P(>0) of the dRoto distribution. ``mean`` tracks the
    point estimate from ``compute_delta_roto``.
    """
    import numpy as np

    from fantasy_baseball.models.player import PlayerType
    from fantasy_baseball.scoring import score_roto_dict
    from fantasy_baseball.simulation import _apply_variance, _flatten_full_season

    before_names = [p.name for p in before_players]
    after_names = [p.name for p in after_players]
    union = list({p.name: p for p in (after_players + before_players)}.values())
    union_h = [_flatten_full_season(p) for p in union if p.player_type == PlayerType.HITTER]
    union_p = [_flatten_full_season(p) for p in union if p.player_type == PlayerType.PITCHER]

    rng = np.random.default_rng(seed)
    field_table = {name: cs for name, cs in field_stats.items()}
    deltas = np.empty(n_draws)
    for i in range(n_draws):
        inj: list = []
        rows = _apply_variance(union_h, PlayerType.HITTER, rng, inj, fraction_remaining)
        rows += _apply_variance(union_p, PlayerType.PITCHER, rng, inj, fraction_remaining)
        by_name = {r["name"]: r for r in rows}
        before_cs = _sum_realized([by_name[n] for n in before_names])
        after_cs = _sum_realized([by_name[n] for n in after_names])
        b = score_roto_dict({team_name: before_cs, **field_table}, team_sds=None)
        a = score_roto_dict({team_name: after_cs, **field_table}, team_sds=None)
        deltas[i] = a[team_name]["total"] - b[team_name]["total"]

    return DeltaRotoBand(
        mean=float(np.mean(deltas)),
        sd=float(np.std(deltas)),
        p_positive=float(np.mean(deltas > 0)),
    )
```

Add `from fantasy_baseball.models.standings import CategoryStats` under the existing `TYPE_CHECKING` import is not enough -- `_sum_realized` uses it at runtime, so add a top-level `from fantasy_baseball.models.standings import CategoryStats`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_lineup/test_delta_roto_band.py -v`
Expected: PASS for the three concrete tests. Then implement `test_mean_tracks_point_estimate` (build a `ProjectedStandings` whose user-team entry equals the summed `before_players` and whose other entries equal `field_stats`; call `compute_delta_roto` for the one-for-one swap; assert `abs(band.mean - point.total) < 0.5` at `n_draws=600`). Tune nothing in the source to pass it -- if it fails, the sampling seam (ROS vs full-season) is wrong; fix the seam, not the test.

- [ ] **Step 5: Add the one-for-one convenience wrapper**

```python
def compute_one_for_one_band(
    drop_name: str,
    add_player,
    active_players: list,
    field_stats: Mapping[str, "CategoryStats"],
    team_name: str,
    fraction_remaining: float,
    *,
    n_draws: int = 400,
    seed: int = 0,
) -> DeltaRotoBand:
    """Band for dropping ``drop_name`` from ``active_players`` and adding ``add_player``."""
    before = list(active_players)
    after = [p for p in active_players if p.name != drop_name] + [add_player]
    return compute_delta_roto_band(
        before, after, field_stats, team_name, fraction_remaining,
        n_draws=n_draws, seed=seed,
    )
```

- [ ] **Step 6: Verify + commit**

Run: `python -m pytest tests/test_lineup/test_delta_roto_band.py -v && python -m ruff check src/fantasy_baseball/lineup/delta_roto.py`
Expected: PASS, clean.

```bash
git add src/fantasy_baseball/lineup/delta_roto.py tests/test_lineup/test_delta_roto_band.py
git commit -m "feat: deltaRoto confidence band (MC core)"
```

---

## Task 2: Band verdict + label formatting

**Files:**
- Create: `src/fantasy_baseball/lineup/band_format.py`
- Test: `tests/test_lineup/test_band_format.py` (create)

The single source of truth for the crosses-zero rule and the display string.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lineup/test_band_format.py
from fantasy_baseball.lineup.band_format import band_class, band_label


def test_band_class_clears_zero_is_real():
    assert band_class(3.4, 1.1) == "real"        # 3.4 - 1.1 = 2.3 > 0


def test_band_class_straddles_zero_is_coin_flip():
    assert band_class(1.9, 2.3) == "coin-flip"   # 1.9 - 2.3 < 0 < 1.9 + 2.3


def test_band_class_below_zero_is_downgrade():
    assert band_class(-2.0, 1.0) == "downgrade"  # -2.0 + 1.0 = -1.0 < 0


def test_band_label_format():
    assert band_label(1.9, 2.3) == "+1.9 +/- 2.3"
    assert band_label(-0.6, 1.8) == "-0.6 +/- 1.8"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lineup/test_band_format.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/fantasy_baseball/lineup/band_format.py
"""Display helpers for the deltaRoto confidence band.

Single source of truth for the crosses-zero verdict and the +/- label,
so every surface (roster audit, trade, compare, lineup) colors and
formats bands identically.
"""

from __future__ import annotations


def band_class(mean: float, sd: float) -> str:
    """Verdict from a deltaRoto band, keyed on whether +/-1 SD crosses zero.

    real      -- band entirely above zero (mean - sd > 0, ~P(help) >= 84%)
    downgrade -- band entirely below zero (mean + sd < 0)
    coin-flip -- band straddles zero
    """
    if mean - sd > 0:
        return "real"
    if mean + sd < 0:
        return "downgrade"
    return "coin-flip"


def band_label(mean: float, sd: float) -> str:
    """ASCII band string, e.g. ``+1.9 +/- 2.3``."""
    return f"{mean:+.1f} +/- {sd:.1f}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lineup/test_band_format.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/band_format.py tests/test_lineup/test_band_format.py
git commit -m "feat: band verdict + label formatting helper"
```

---

## Task 3: Roster audit surface

**Files:**
- Modify: `src/fantasy_baseball/lineup/roster_audit.py:306-356` (candidate loop)
- Modify: `src/fantasy_baseball/web/templates/season/roster_audit.html:67-76` and `99-120`
- Modify: `src/fantasy_baseball/web/static/season.css:3024-3056`
- Test: `tests/test_lineup/test_roster_audit.py` (add a band assertion)

`audit_roster` already has `active_roster`, `team_name`, `projected_standings`, `team_sds`, and computes `dr = score_swap(...)`. Add a band per scored candidate using `compute_one_for_one_band`, with the field = projected standings minus the user team. `fraction_remaining` must be threaded into `audit_roster` (it is available in the refresh pipeline as `self.fraction_remaining` -- the caller `_audit_roster` at `refresh_pipeline.py:925` passes `team_sds`; add `fraction_remaining=self.fraction_remaining` there and a matching kwarg on `audit_roster`).

- [ ] **Step 1: Write the failing test** (add to `tests/test_lineup/test_roster_audit.py`)

```python
def test_audit_candidate_has_band(...):  # reuse the existing audit fixture
    # After audit_roster(..., fraction_remaining=0.6), each scored candidate
    # dict carries a "band" key with mean/sd/p_positive.
    entries = audit_roster(..., fraction_remaining=0.6)
    cand = next(e for e in entries if e.candidates)["candidates"][0]
    assert set(cand["band"]) == {"mean", "sd", "p_positive"}
    assert cand["band"]["sd"] >= 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_lineup/test_roster_audit.py -k band -v`
Expected: FAIL (`KeyError: 'band'` or TypeError on the new kwarg).

- [ ] **Step 3: Implement** -- add the `fraction_remaining` param to `audit_roster`, build the field once, and attach the band in the candidate dict (roster_audit.py, in the loop at lines ~329-356):

```python
from fantasy_baseball.lineup.delta_roto import compute_one_for_one_band, score_swap
# ...
field_stats = {e.team_name: e.stats for e in projected_standings.entries if e.team_name != team_name}
# inside the per-fa loop, after computing dr:
band = compute_one_for_one_band(
    player.name, fa, active_roster, field_stats, team_name, fraction_remaining,
    n_draws=300, seed=abs(hash((player.name, fa.name))) % (2**32),
)
scored.append({
    "name": fa.name,
    "player_type": fa.player_type.value,
    "positions": list(fa.positions),
    "sgp": round(fa_sgp.get(fa.name, 0.0), 2),
    "gap": sgp_gap,
    "delta_roto": dr.to_dict(),
    "band": band.to_dict(),
    "player_id": fa.yahoo_id,
})
```

- [ ] **Step 4: Update the template** (`roster_audit.html`) -- replace the magnitude-threshold badge with the band + crosses-zero color. Header-row badge (lines 67-76):

```html
        <td>
            {% if entry.candidates and entry.candidates[0].band %}
            {% set b = entry.candidates[0].band %}
            {% set gap = {'real':'gap-positive','coin-flip':'gap-marginal','downgrade':'gap-negative'}[b.verdict] %}
            <span class="gap-badge {{ gap }}">
                {{ "%+.1f"|format(b.mean) }} +/- {{ "%.1f"|format(b.sd) }}
            </span>
            {% else %}
            <span class="gap-none">&mdash;</span>
            {% endif %}
        </td>
```

The verdict comes from Python (`band.verdict`); the template only maps it to the existing filled-badge class name (it does NOT re-derive the crosses-zero rule). Apply the same `band.verdict` -> `gap-*` map to the detail-row deltaRoto cell (lines 99-120), replacing the `c.delta_roto.total` span with `c.band` mean/sd.

- [ ] **Step 5: Update CSS comment** in `season.css:3024` so the traffic-light meaning is documented as "band clears zero / straddles / below" rather than the old magnitude threshold. (Class names `gap-positive/marginal/negative` are reused; no new classes needed.)

- [ ] **Step 6: Verify + commit**

Run: `python -m pytest tests/test_lineup/test_roster_audit.py -v && python -m ruff check src/fantasy_baseball/lineup/roster_audit.py`

```bash
git add src/fantasy_baseball/lineup/roster_audit.py src/fantasy_baseball/web/templates/season/roster_audit.html src/fantasy_baseball/web/static/season.css src/fantasy_baseball/web/refresh_pipeline.py tests/test_lineup/test_roster_audit.py
git commit -m "feat: confidence band on roster audit"
```

---

## Task 4: Trade builder surface

**Files:**
- Modify: `src/fantasy_baseball/trades/multi_trade.py` (`evaluate_multi_trade`, add a total band)
- Modify: `src/fantasy_baseball/web/season_routes.py:711-796` (serialize the band)
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html` (`renderResult`, show the band)
- Test: `tests/test_trades/test_multi_trade.py` + `tests/test_web/test_evaluate_trade_route.py` (assert band present)

`evaluate_multi_trade` already builds `mine_leaving` / `mine_entering` player lists and the fixed projected standings. The band's before/after rosters are: before = current active mine; after = mine with leaving removed + entering added. Field = projected standings minus the user team. Compute one band on the trade total.

- [ ] **Step 1: Write failing tests** -- in `test_multi_trade.py`, assert `result.band` has `mean/sd/p_positive`; in `test_evaluate_trade_route.py`, assert the JSON response includes a `"band"` object.

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_trades/test_multi_trade.py tests/test_web/test_evaluate_trade_route.py -k band -v`
Expected: FAIL.

- [ ] **Step 3: Implement in `multi_trade.py`** -- add a `band: dict | None = None` field to `MultiTradeResult`; after the existing scoring, build before/after active rosters (reuse `mine_leaving`/`mine_entering`, applied to `_current_active_set` resolved to Player objects) and call `compute_delta_roto_band`, then set `band=band.to_dict()` on the result. `fraction_remaining` is threaded from the route (read `proj_cache`/season_progress; pass through `evaluate_multi_trade`).

- [ ] **Step 4: Serialize in the route** (`season_routes.py:711-796`) -- add `"band": result.band` to the `jsonify({...})` payload.

- [ ] **Step 5: Render in JS** (`waivers_trades.html` `renderResult`, ~line 717) -- prepend a band line to `bt-result-deltas`:

```javascript
    if (data.band) {
      const m = data.band.mean, s = data.band.sd;
      // band.verdict is computed once in Python; JS only maps it to a class.
      target.insertAdjacentHTML('afterbegin',
        '<div class="band-line band-' + data.band.verdict + '">' +
        (m >= 0 ? '+' : '') + m.toFixed(1) + ' +/- ' + s.toFixed(1) + ' roto</div>');
    }
```

- [ ] **Step 6: Verify + commit**

Run: `python -m pytest tests/test_trades/ tests/test_web/test_evaluate_trade_route.py -v`

```bash
git add src/fantasy_baseball/trades/multi_trade.py src/fantasy_baseball/web/season_routes.py src/fantasy_baseball/web/templates/season/waivers_trades.html tests/test_trades/test_multi_trade.py tests/test_web/test_evaluate_trade_route.py
git commit -m "feat: confidence band on trade builder"
```

---

## Task 5: Player comparison surface

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py` (`compute_comparison_standings`, add a band)
- Modify: `src/fantasy_baseball/web/templates/season/players.html:645-676` (`renderDeltaRoto`)
- Test: `tests/test_web/` comparison test (assert band in response)

`compute_comparison_standings` already computes the one-for-one delta. Add a band via `compute_one_for_one_band(roster_player_name, other_player, user_roster_active, field_stats, team_name, fraction_remaining)` and include `"band": band.to_dict()` in its returned dict.

- [ ] **Step 1: Write failing test** -- assert `/api/players/compare` JSON includes `band` with mean/sd/p_positive when comparing a roster player to a non-roster player.

- [ ] **Step 2: Run to verify fail.** Expected: FAIL (`KeyError`).

- [ ] **Step 3: Implement** in `season_data.compute_comparison_standings` (build field from the projected standings minus the user team; thread `fraction_remaining`).

- [ ] **Step 4: Render** -- extend `renderDeltaRoto(dr, band)` in `players.html` to prepend the band line (same `band-real/coin-flip/downgrade` class + `mean +/- sd` text as Task 4).

- [ ] **Step 5: Verify + commit**

Run: `python -m pytest tests/test_web/ -k compar -v`

```bash
git add src/fantasy_baseball/web/season_data.py src/fantasy_baseball/web/templates/season/players.html tests/test_web/
git commit -m "feat: confidence band on player comparison"
```

---

## Task 6: Lineup surface

**Files:**
- Modify: `src/fantasy_baseball/lineup/optimizer.py:18-40, 210-289` (attach band to assignments)
- Modify: `src/fantasy_baseball/web/templates/season/lineup.html:69-86` and the `_lineup_*_tbody.html` cells
- Test: `tests/test_lineup/test_optimizer.py` (assert assignment band)

Each `HitterAssignment` / `PitcherStarter` already carries a `roto_delta` (the point estimate vs its best counterfactual). Add `band: dict | None`. For the move chips and per-player cells, the band's before/after are: before = optimal active lineup; after = optimal lineup with that player swapped for their best counterfactual (the same comparison `roto_delta` already encodes). Reuse `compute_delta_roto_band` with those two player lists. Compute in the optimizer (offline, in the refresh) so `cache:lineup_optimal` carries the band.

- [ ] **Step 1: Write failing test** -- assert `optimize_hitter_lineup(...)[0].to_dict()` includes a `band` with mean/sd/p_positive.

- [ ] **Step 2: Run to verify fail.** Expected: FAIL.

- [ ] **Step 3: Implement** -- add `band: dict | None = None` to both dataclasses; include it in `to_dict()`; in the `roto_deltas` loops (lines 210-232 / 268-284) also compute a band for each starter via `compute_delta_roto_band(active_subset, alt_subset_for_starter, field_stats, team_name, fraction_remaining)` where `alt_subset_for_starter` is the best counterfactual subset already found. Thread `fraction_remaining` + `field_stats` into the optimizer (the refresh passes `team_sds` already; add the two args at `refresh_pipeline.py:846-859`).

- [ ] **Step 4: Render** -- in `lineup.html` move chips (lines 69-86) and the `_lineup_hitters_tbody.html` / `_lineup_pitchers_tbody.html` cells (lines 79 / 70), show `mean +/- sd` with the `band-real/coin-flip/downgrade` class instead of (or alongside) the bare `roto_delta`.

- [ ] **Step 5: Verify + commit**

Run: `python -m pytest tests/test_lineup/test_optimizer.py -v`

```bash
git add src/fantasy_baseball/lineup/optimizer.py src/fantasy_baseball/web/templates/season/lineup.html src/fantasy_baseball/web/templates/season/_lineup_hitters_tbody.html src/fantasy_baseball/web/templates/season/_lineup_pitchers_tbody.html src/fantasy_baseball/web/refresh_pipeline.py tests/test_lineup/test_optimizer.py
git commit -m "feat: confidence band on lineup moves"
```

---

## Task 7: Shared CSS + final verification

**Files:**
- Modify: `src/fantasy_baseball/web/static/season.css` (add `.band-real`, `.band-coin-flip`, `.band-downgrade` for the JS-rendered surfaces)

- [ ] **Step 1: Add band classes** mirroring the traffic-light palette:

```css
.band-real      { color: var(--lead-deep); }
.band-coin-flip { color: var(--gold-deep); }
.band-downgrade { color: var(--trail-deep); }
.band-line { font-weight: 700; font-size: 13px; margin-bottom: 6px; }
```

- [ ] **Step 2: Full verification**

Run: `python -m pytest tests/test_lineup/ tests/test_trades/ tests/test_web/ -q && python -m ruff check . && python -m ruff format --check .`
Expected: all pass, clean. (Note: run the FULL `pytest tests/` only after PR #86's streaks-mock fix is on this branch's base; otherwise scope to the dirs above.)

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/static/season.css
git commit -m "style: band verdict color classes"
```

---

## Notes for the implementer

- The linchpin is Task 1's `test_mean_tracks_point_estimate`. If the band mean does not track the point `compute_delta_roto`, the sampling seam is wrong (most likely sampling full-season instead of ROS-remaining, or re-selecting the active roster per draw). Fix the seam; never relax the test.
- `_apply_variance` expects FLAT dicts; always pass `_flatten_full_season(player)`.
- Use `safe_float` for any stat read you add (avoid the `x or 0` NaN trap).
- `n_draws=300-400` is the default; the audit precomputes offline so it can afford it. If the refresh gets slow, lower draws before changing the algorithm.
- Field is held at point estimate by design (see spec). Do not sample the other teams.
