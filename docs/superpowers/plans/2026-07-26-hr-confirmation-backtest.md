# HR-confirmation Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-26-hr-confirmation-backtest-design.md`

**Goal:** Backtest whether barrel rate or park-adjusted xHR confirms next-year HR better than the SLG-vs-xSLG proxy used in `w_for_stat`, producing a go/no-go verdict; wire nothing into `breakout.py` (that is a verdict-gated follow-up).

**Architecture:** Three layers. (1) Data: add a Baseball Savant expected-HR fetcher to `skill_luck.py` and two optional fields to `SkillLuckRow` so the existing corpus carries xHR + barrels-per-PA. (2) Pure logic: a new `analysis/hr_confirm.py` holds candidate confirm math, barrel calibration, scale tuning, scoring, level-control, and the verdict rule -- all I/O-free and unit-tested. (3) Orchestration: a thin `scripts/backtest_hr_confirm.py` builds the corpus (reusing `backtest_breakout.build_corpus`) and calls `hr_confirm.run`, prints, writes a CSV.

**Tech Stack:** Python 3.12, pandas, stdlib `random`/`statistics`. Reuses `breakout.py` (`SkillLuckRow`, `line_rates`, `_confirm_gap`, `_reliability`) and `breakout_backtest.py` (`marcel_prior`, `_league_mean`, `_rates_to_line`, `_spearman`, `_bootstrap_diff`, `rate_mae`, `Corpus`/`Line` types). No new dependencies. No FanGraphs.

## Global Constraints

- **ASCII-only** in all source, log strings, and report renderers (no true minus, sigma, arrows, smart quotes). Entry-point script may `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` for player names, but default to ASCII.
- **No `x or default` for numeric defaults** -- use `v if v is not None else default`. `0.0` HR-rates are real.
- **Player IDs / joins keyed by MLBAM** (int). Never key on bare names.
- **Isolate the variable under test:** only the HR *confirmation source* varies. `stat_stabilize["hr"]=120`, `confirm_weight=0.5`, reliability shape stay at shipped values.
- **Determinism:** the bootstrap CI must be reproducible. `_bootstrap_diff(a, b, actual, *, iters=2000, seed=0)` already seeds; always pass an explicit `seed=SEED` constant.
- **No behavior change to `breakout.py`'s `w_for_stat`.** Adding fields to `SkillLuckRow` is allowed; changing the HR branch is not (deferred).
- **Named constants, not magic numbers:** `PA_FLOOR=150`, `HR_MOVE_MIN=0.005`, `SEED=12345`, the scale grids, and `MAE_EPS=0.0005` are module-level constants.
- **End-of-effort checks (repo rule):** `pytest -v`, `ruff check .`, `ruff format --check .`, `vulture`, and `mypy` (only if a touched file is under `[tool.mypy].files`). Show outputs.

---

### Task 1: xHR fetcher + `SkillLuckRow` fields (data layer)

**Files:**
- Modify: `src/fantasy_baseball/analysis/breakout.py` (`SkillLuckRow` dataclass, ~line 16-40)
- Modify: `src/fantasy_baseball/data/skill_luck.py` (`fetch_or_cache`, barrel rename, new HR fetcher, `build_hitter_skill_luck`, `build_pitcher_skill_luck`)
- Test: `tests/test_data/test_skill_luck.py` (extend existing joins test + new fetcher tests)

**Interfaces:**
- Consumes: existing `fetch_or_cache`, `_rename_strict`, `_sf`, `load_statcast_hitters`, `_join_and_count`, `build_hitter_skill_luck(cache_dir, year, *, fetchers=None)`.
- Produces:
  - `SkillLuckRow.xhr: float | None` (season xHR count) and `SkillLuckRow.brl_pa: float | None` (barrels-per-PA share, 0-1).
  - `skill_luck.load_statcast_hr(cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None) -> pd.DataFrame` with columns `["mlbam", "xhr"]` (0 rows for pre-2016 empty leaderboards).
  - `fetch_or_cache(path, fetcher, *, tolerate_empty: bool = False)` -- when `tolerate_empty` and the fetch is empty, returns the empty (typed) frame without writing a cache, instead of raising.
  - `build_hitter_skill_luck` now also accepts a `"sc_hr"` fetcher key and sets `xhr`/`brl_pa` on each row.

- [ ] **Step 1: Write failing tests for the new data plumbing**

Add to `tests/test_data/test_skill_luck.py`:

```python
def test_fetch_or_cache_tolerate_empty_returns_typed_frame_without_writing(tmp_path: Path):
    q = tmp_path / "hr_empty.csv"
    empty = pd.DataFrame({"player_id": [], "xhr": []})  # header-only (pre-2016)
    out = skill_luck.fetch_or_cache(q, lambda: empty, tolerate_empty=True)
    assert list(out.columns) == ["player_id", "xhr"] and len(out) == 0
    assert not q.exists()  # empty leaderboard is not cached


def test_load_statcast_hr_renames_and_keys_by_mlbam(tmp_path: Path):
    raw = pd.DataFrame(
        {"player_id": [621566, 665742], "xhr": [48.2, 40.1], "hr_total": [54, 41]}
    )
    out = skill_luck.load_statcast_hr(tmp_path, 2023, fetcher=lambda: raw)
    assert list(out.columns) == ["mlbam", "xhr"]
    assert out.set_index("mlbam").loc[621566, "xhr"] == pytest.approx(48.2)


def test_load_statcast_hr_tolerates_empty_pre_2016(tmp_path: Path):
    empty = pd.DataFrame({"player_id": [], "xhr": [], "hr_total": []})
    out = skill_luck.load_statcast_hr(tmp_path, 2015, fetcher=lambda: empty)
    assert list(out.columns) == ["mlbam", "xhr"] and len(out) == 0
```

Then EXTEND the existing `test_build_hitter_skill_luck_joins_and_reports_coverage`:
- add `"brl_pa": [4.9]` to the `sc_brl` fake frame,
- add a `sc_hr` fake and fetcher,
- assert the new fields land.

```python
    sc_brl = pd.DataFrame({"player_id": [665742], "brl_percent": [14.0], "brl_pa": [4.9]})
    sc_hr = pd.DataFrame({"player_id": [665742], "xhr": [28.5], "hr_total": [30]})

    rows, cov = skill_luck.build_hitter_skill_luck(
        tmp_path, 2024,
        fetchers={"mlb": lambda: mlb, "sc_x": lambda: sc_x,
                  "sc_brl": lambda: sc_brl, "sc_hr": lambda: sc_hr},
    )
    soto = rows[665742]
    assert soto.barrel_pct == 0.14 and soto.brl_pa == pytest.approx(0.049)
    assert soto.xhr == pytest.approx(28.5)
    # unmatched player: new fields None
    assert rows[800].xhr is None and rows[800].brl_pa is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_skill_luck.py -v`
Expected: FAIL -- `AttributeError: 'SkillLuckRow' object has no attribute 'xhr'` / `load_statcast_hr` not defined / `TypeError: tolerate_empty`.

- [ ] **Step 3: Add the two fields to `SkillLuckRow`**

In `src/fantasy_baseball/analysis/breakout.py`, at the END of the `SkillLuckRow` field list (after `bb_pct`), add:

```python
    # HR-confirmation extras (issue #262). Optional: None for pitchers, pre-2016
    # (xHR leaderboard starts 2016), or unmatched players. Defaults keep existing
    # keyword constructors valid.
    brl_pa: float | None = None  # barrels per PA (share, e.g. 0.049)
    xhr: float | None = None  # park-adjusted expected HR (season count)
```

- [ ] **Step 4: Add `tolerate_empty` to `fetch_or_cache`**

In `src/fantasy_baseball/data/skill_luck.py`, replace `fetch_or_cache`:

```python
def fetch_or_cache(
    path: Path, fetcher: Callable[[], pd.DataFrame], *, tolerate_empty: bool = False
) -> pd.DataFrame:
    cached = _read_cached(path)
    if cached is not None:
        return cached
    df = fetcher()
    if df is None or df.empty:
        if tolerate_empty and df is not None:
            return df  # expected-empty (e.g. pre-2016 xHR); do not cache emptiness
        raise RuntimeError(
            f"fetch for {path.name} returned empty; refusing to overwrite/write cache"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
```

- [ ] **Step 5: Add `brl_pa` to the barrel rename + normalize**

In `skill_luck.py`, change `_STATCAST_BARREL_RENAME` and the normalization line in `load_statcast_hitters`:

```python
_STATCAST_BARREL_RENAME = {"player_id": "mlbam", "brl_percent": "barrel_pct", "brl_pa": "brl_pa"}
```

In `load_statcast_hitters`, after the existing `barrel_pct` normalization, also normalize `brl_pa` (Savant serves both as 0-100 percents):

```python
    # barrel_pct and brl_pa both arrive as percents (0-100) on Savant; -> shares.
    b = b.assign(
        barrel_pct=b["barrel_pct"].astype(float) / 100.0,
        brl_pa=b["brl_pa"].astype(float) / 100.0,
    )
```

- [ ] **Step 6: Add the xHR fetcher**

In `skill_luck.py`, in the Statcast section, add the rename map and loader. The default fetcher hits Savant directly (pybaseball has no wrapper) with a browser User-Agent and BOM-safe decode:

```python
_STATCAST_HR_RENAME = {"player_id": "mlbam", "xhr": "xhr"}
_SAVANT_HR_URL = (
    "https://baseballsavant.mlb.com/leaderboard/home-runs"
    "?type=batter&year={year}&min=1&csv=true"
)


def _fetch_savant_hr(year: int) -> pd.DataFrame:
    """Baseball Savant expected-HR leaderboard CSV. pybaseball has no wrapper, so
    fetch the CSV directly (browser UA + utf-8-sig for the BOM). Pre-2016 returns a
    header-only body (empty frame)."""
    import io
    import urllib.request

    req = urllib.request.Request(
        _SAVANT_HR_URL.format(year=year),
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # fixed Savant https host
        body = resp.read().decode("utf-8-sig", "replace")
    return pd.read_csv(io.StringIO(body))
```

(Do not pre-add a `# noqa`. If `ruff check .` in Task 4 flags `S310` on the
`urlopen` line, add `# noqa: S310` then; adding it speculatively can itself trip
`RUF100` unused-noqa when the `S` rules are not enabled.)

```python
# (end of _fetch_savant_hr; the loader below is unchanged)


def load_statcast_hr(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    """MLBAM -> park-adjusted expected HR (`xhr`, season count). Empty (0-row) frame
    for pre-2016 years, where the leaderboard has no data."""
    raw = fetch_or_cache(
        cache_dir / f"sc_hr_h_{year}.csv",
        fetcher or (lambda: _fetch_savant_hr(year)),
        tolerate_empty=True,
    )
    return _rename_strict(raw, _STATCAST_HR_RENAME)
```

- [ ] **Step 7: Thread xHR + brl_pa through `build_hitter_skill_luck`**

In `build_hitter_skill_luck`, after building `sc = load_statcast_hitters(...)`, merge in xHR and set the new row fields:

```python
    sc = load_statcast_hitters(
        cache_dir, year,
        xstats_fetcher=fetchers.get("sc_x"), barrels_fetcher=fetchers.get("sc_brl"),
    )
    hr = load_statcast_hr(cache_dir, year, fetcher=fetchers.get("sc_hr"))
    sc = sc.merge(hr, on="mlbam", how="left")  # xhr NaN when a player has no xHR row
```

And in the `build_row` lambda add the two fields:

```python
            brl_pa=_sf(s, "brl_pa"),
            xhr=_sf(s, "xhr"),
```

In `build_pitcher_skill_luck`'s `build_row`, add `brl_pa=None, xhr=None,` (pitchers have neither).

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_data/test_skill_luck.py -v`
Expected: PASS (all, including the extended join test).

- [ ] **Step 9: Commit**

```bash
git add src/fantasy_baseball/analysis/breakout.py src/fantasy_baseball/data/skill_luck.py tests/test_data/test_skill_luck.py
git commit -m "feat(breakout): Savant xHR + brl_pa on SkillLuckRow (#262 data layer)"
```

---

### Task 2: HR-confirm candidate math (pure module)

**Files:**
- Create: `src/fantasy_baseball/analysis/hr_confirm.py`
- Test: `tests/test_analysis/test_hr_confirm.py`

**Interfaces:**
- Consumes: `SkillLuckRow`, `line_rates`, `_confirm_gap`, `_reliability` from `breakout`; `Corpus`, `Line`, `marcel_prior`, `_league_mean`, `_rates_to_line`, `_spearman`, `_bootstrap_diff`, `rate_mae` from `breakout_backtest`. Uses `SkillLuckRow.brl_pa`, `.xhr`, `.pa`, `.slg`, `.xslg`, `.age` from Task 1.
- Produces:
  - `HrRecord = dict[str, float]` with keys `mlbam, prior_hr, surface_hr, actual_hr, pa, slg, xslg, brl_pa, xhr_rate`.
  - `build_hr_records(corpus, years, *, pa_floor=PA_FLOOR, hr_move_min=HR_MOVE_MIN) -> list[HrRecord]`
  - `BarrelCalib = tuple[float, float]` (slope, intercept); `fit_barrel_calibration(records) -> BarrelCalib`
  - `expected_hr_rate(candidate, rec, calib) -> float` and `confirm(candidate, rec, calib, scale) -> float`
  - `forward_hr(rec, confirm_value, *, cw=0.5, hr_stabilize=120.0) -> float`
  - `tune_scale(fit_records, candidate, calib) -> float`
  - `signed_gap(candidate, rec, calib) -> float`
  - `level_control(records, candidate, calib) -> list[float]` (per-tier Spearman, low->high prior-HR tier)
  - `run(corpus, *, fit_years, report_years, seed=SEED) -> dict` (the full result: per-candidate Spearman/MAE/CI/tuned scale/level-control + verdict strings)
  - Constants: `CANDIDATES=("xslg","barrel","xhr")`, `PA_FLOOR=150.0`, `HR_MOVE_MIN=0.005`, `SEED=12345`, `MAE_EPS=0.0005`, `HRPA_SCALE_GRID`, `SLG_SCALE_GRID`, `SHIPPED_XSLG_SCALE=0.150`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_analysis/test_hr_confirm.py`:

```python
import math

from fantasy_baseball.analysis import breakout, hr_confirm


def _rec(**kw):
    base = dict(
        mlbam=1, prior_hr=0.04, surface_hr=0.06, actual_hr=0.05, pa=600.0,
        slg=0.520, xslg=0.470, brl_pa=0.08, xhr_rate=0.045,
    )
    base.update(kw)
    return base


def test_barrel_calibration_recovers_known_line():
    # y = 0.5*x + 0.01 exactly -> slope 0.5, intercept 0.01
    recs = [_rec(brl_pa=x / 100.0, surface_hr=0.5 * (x / 100.0) + 0.01) for x in range(2, 20)]
    slope, intercept = hr_confirm.fit_barrel_calibration(recs)
    assert math.isclose(slope, 0.5, rel_tol=1e-6)
    assert math.isclose(intercept, 0.01, abs_tol=1e-6)


def test_confirm_is_monotonic_in_overperformance():
    calib = (0.5, 0.01)
    # bigger surface-vs-expected gap -> lower confirm, for every candidate
    for cand in hr_confirm.CANDIDATES:
        small = hr_confirm.confirm(cand, _rec(surface_hr=0.045, slg=0.475), calib, scale=0.05)
        big = hr_confirm.confirm(cand, _rec(surface_hr=0.090, slg=0.600), calib, scale=0.05)
        assert big < small


def test_forward_hr_matches_breakout_w_for_stat_for_xslg_shipped():
    # The xslg candidate at the shipped 0.150 scale must equal w_for_stat's HR blend.
    rec = _rec()
    row = breakout.SkillLuckRow(
        mlbam=1, player_type="hitter", pa=rec["pa"], ip=0.0, age=None,
        barrel_pct=None, xslg=rec["xslg"], slg=rec["slg"], xba=None, ba=None,
        babip=None, xwoba=None, woba=None, k_pct=None, bb_pct=None,
    )
    w = breakout.w_for_stat("hr", row, "hitter", breakout.DEFAULT_WMAP)
    cv = hr_confirm.confirm("xslg", rec, (0.0, 0.0), scale=hr_confirm.SHIPPED_XSLG_SCALE)
    forward = hr_confirm.forward_hr(rec, cv)
    # forward = prior + w*(surface-prior); recover the effective w and compare.
    eff_w = (forward - rec["prior_hr"]) / (rec["surface_hr"] - rec["prior_hr"])
    assert math.isclose(eff_w, w, rel_tol=1e-9)


def test_tune_scale_picks_grid_argmax_on_fit():
    recs = [_rec(mlbam=i, surface_hr=0.04 + 0.001 * i, actual_hr=0.04 + 0.001 * i)
            for i in range(30)]
    s = hr_confirm.tune_scale(recs, "xhr", (0.5, 0.01))
    assert s in hr_confirm.HRPA_SCALE_GRID


def test_verdict_rule_all_branches():
    V = hr_confirm._verdict_for
    # CI includes 0 -> no, regardless of tiers/MAE
    assert V(ci=(-0.01, 0.05), mae_delta=0.0, tier_signs=[-1, -1, -1]) == "no (CI includes 0)"
    # CI-positive but expected sign in <2 tiers -> level-confounded
    assert V(ci=(0.02, 0.08), mae_delta=0.0, tier_signs=[+1, +1, -1]) == (
        "level-confounded -- do not wire in"
    )
    # CI-positive, survives level-control, but MAE worse by > MAE_EPS -> inconsistent
    assert V(ci=(0.02, 0.08), mae_delta=0.001, tier_signs=[-1, -1, -1]) == (
        "CI-positive but MAE-inconsistent"
    )
    # CI-positive, survives level-control, MAE fine -> wire-in eligible
    assert V(ci=(0.02, 0.08), mae_delta=0.0, tier_signs=[-1, -1, -1]) == "wire-in eligible"
    # Fewer than 3 tiers (thin sample) -> inconclusive, not a false confound verdict
    assert V(ci=(0.02, 0.08), mae_delta=0.0, tier_signs=[]) == (
        "inconclusive (thin sample for level-control)"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis/test_hr_confirm.py -v`
Expected: FAIL -- `ModuleNotFoundError: hr_confirm`.

- [ ] **Step 3: Implement `hr_confirm.py`**

Create `src/fantasy_baseball/analysis/hr_confirm.py`:

```python
"""HR-confirmation backtest logic (issue #262): does barrel rate or park-adjusted
xHR confirm next-year HR better than the SLG-vs-xSLG proxy w_for_stat uses? Pure
(no I/O); the corpus is built by scripts/backtest_hr_confirm.py and passed in.

See docs/superpowers/specs/2026-07-26-hr-confirmation-backtest-design.md.
"""

from __future__ import annotations

from collections.abc import Collection
from statistics import fmean

from fantasy_baseball.analysis.breakout import (
    _confirm_gap,  # reused so the xslg candidate is byte-identical to w_for_stat's HR branch
    _reliability,
    line_rates,
)
from fantasy_baseball.analysis.breakout_backtest import (
    Corpus,
    Line,
    _bootstrap_diff,
    _league_mean,
    _rates_to_line,
    marcel_prior,
    _spearman,
    rate_mae,
)

HrRecord = dict[str, float]
BarrelCalib = tuple[float, float]  # (slope, intercept) for HR/PA ~ brl_pa

CANDIDATES: tuple[str, ...] = ("xslg", "barrel", "xhr")
PA_FLOOR = 150.0
HR_MOVE_MIN = 0.005  # HR/PA move to count as a candidate (~3 HR / 600 PA)
SEED = 12345
MAE_EPS = 0.0005  # HR/PA tolerance for the MAE consistency flag (~0.3 HR / 600 PA)
SHIPPED_XSLG_SCALE = 0.150
HR_STABILIZE = 120.0  # shipped stat_stabilize["hr"]
CONFIRM_WEIGHT = 0.5  # shipped confirm_weight
HRPA_SCALE_GRID = [0.010 + 0.005 * i for i in range(11)]  # 0.010 .. 0.060
SLG_SCALE_GRID = [0.075 + 0.025 * i for i in range(8)]  # 0.075 .. 0.250


def build_hr_records(
    corpus: Corpus,
    years: Collection[int],
    *,
    pa_floor: float = PA_FLOOR,
    hr_move_min: float = HR_MOVE_MIN,
) -> list[HrRecord]:
    """Per-hitter-season HR records on the common support (all four confirm
    ingredients present), filtered to the moved-HR candidate population."""
    recs: list[HrRecord] = []
    for year in years:
        year_data = corpus[year]
        lg = _league_mean(year_data)
        for surface, sl, actual_next, hist, _zips in year_data.values():
            if sl.slg is None or sl.xslg is None or sl.brl_pa is None or sl.xhr is None:
                continue  # off common support
            pa = float(sl.pa)
            if pa < pa_floor:
                continue
            proj_line = {**surface, **_rates_to_line(marcel_prior(hist, lg, sl.age), surface)}
            prior_hr = line_rates(proj_line, "hitter")["hr"]
            surface_hr = line_rates(surface, "hitter")["hr"]
            if abs(surface_hr - prior_hr) < hr_move_min:
                continue  # HR did not move -- not a mirage/breakout candidate
            recs.append(
                {
                    "mlbam": float(sl.mlbam),
                    "prior_hr": prior_hr,
                    "surface_hr": surface_hr,
                    "actual_hr": actual_next["hr"],
                    "pa": pa,
                    "slg": sl.slg,
                    "xslg": sl.xslg,
                    "brl_pa": sl.brl_pa,
                    "xhr_rate": sl.xhr / pa if pa > 0 else 0.0,
                }
            )
    return recs


def fit_barrel_calibration(records: Collection[HrRecord]) -> BarrelCalib:
    """OLS slope/intercept of HR/PA ~ brl_pa (barrels are a rate skill; this maps
    them to an expected HR/PA). Scale-invariant: the slope absorbs brl_pa's units."""
    xs = [r["brl_pa"] for r in records]
    ys = [r["surface_hr"] for r in records]
    n = len(xs)
    if n < 2:
        return 0.0, fmean(ys) if ys else 0.0
    mx, my = fmean(xs), fmean(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0.0:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / var
    return slope, my - slope * mx


def expected_hr_rate(candidate: str, rec: HrRecord, calib: BarrelCalib) -> float:
    """The candidate's expected HR/PA (for barrel/xhr). Unused for xslg (its gap is
    in SLG units); callers guard on candidate."""
    if candidate == "barrel":
        slope, intercept = calib
        return intercept + slope * rec["brl_pa"]
    if candidate == "xhr":
        return rec["xhr_rate"]
    raise ValueError(candidate)


def confirm(candidate: str, rec: HrRecord, calib: BarrelCalib, *, scale: float) -> float:
    """confirm in [0,1], reusing breakout._confirm_gap so xslg == w_for_stat's HR branch."""
    if candidate == "xslg":
        return _confirm_gap(rec["slg"], rec["xslg"], scale)
    return _confirm_gap(rec["surface_hr"], expected_hr_rate(candidate, rec, calib), scale)


def forward_hr(
    rec: HrRecord,
    confirm_value: float,
    *,
    cw: float = CONFIRM_WEIGHT,
    hr_stabilize: float = HR_STABILIZE,
) -> float:
    """prior + w*(surface-prior), w = reliability*((1-cw)+cw*confirm) -- the exact
    w_for_stat blend, HR branch."""
    reliability = _reliability(rec["pa"], hr_stabilize)
    w = reliability * ((1.0 - cw) + cw * confirm_value)
    return rec["prior_hr"] + w * (rec["surface_hr"] - rec["prior_hr"])


def _scale_grid(candidate: str) -> list[float]:
    return SLG_SCALE_GRID if candidate == "xslg" else HRPA_SCALE_GRID


def _forwards(records: Collection[HrRecord], candidate: str, calib: BarrelCalib, scale: float):
    return [forward_hr(r, confirm(candidate, r, calib, scale=scale)) for r in records]


def tune_scale(fit_records: list[HrRecord], candidate: str, calib: BarrelCalib) -> float:
    """Grid-search the confirm-gap scale on FIT records, maximizing forward-line
    Spearman (the same metric the verdict gates on)."""
    actual = [r["actual_hr"] for r in fit_records]
    best_scale, best_rho = _scale_grid(candidate)[0], -2.0
    for scale in _scale_grid(candidate):
        rho = _spearman(_forwards(fit_records, candidate, calib, scale), actual)
        if rho > best_rho:
            best_scale, best_rho = scale, rho
    return best_scale


def signed_gap(candidate: str, rec: HrRecord, calib: BarrelCalib) -> float:
    """The candidate's own over/under-performance signal (positive = overperformed)."""
    if candidate == "xslg":
        return rec["slg"] - rec["xslg"]
    return rec["surface_hr"] - expected_hr_rate(candidate, rec, calib)


def level_control(records: list[HrRecord], candidate: str, calib: BarrelCalib) -> list[float]:
    """Within prior-HR/PA terciles, Spearman(signed_gap, next-year change). A real
    luck signal stays negative in each tier (overperformance -> next-year decline)."""
    ordered = sorted(records, key=lambda r: r["prior_hr"])
    k = len(ordered) // 3
    if k < 2:
        return []
    tiers = [ordered[:k], ordered[k : 2 * k], ordered[2 * k :]]
    out: list[float] = []
    for tier in tiers:
        gaps = [signed_gap(candidate, r, calib) for r in tier]
        change = [r["actual_hr"] - r["surface_hr"] for r in tier]
        out.append(_spearman(gaps, change))
    return out


def _mean_mae(forwards: list[float], actual: list[float]) -> float:
    return fmean([rate_mae({"hr": f}, {"hr": a}) for f, a in zip(forwards, actual, strict=True)])


def _verdict_for(*, ci: tuple[float, float], mae_delta: float, tier_signs: list[int]) -> str:
    """mae_delta = MAE(candidate) - MAE(xslg). tier_signs = sign of each tier's
    level-control Spearman (expected: negative)."""
    if ci[0] <= 0:
        return "no (CI includes 0)"
    if len(tier_signs) < 3:
        return "inconclusive (thin sample for level-control)"
    if sum(1 for s in tier_signs if s < 0) < 2:
        return "level-confounded -- do not wire in"
    if mae_delta > MAE_EPS:
        return "CI-positive but MAE-inconsistent"
    return "wire-in eligible"


def run(
    corpus: Corpus, *, fit_years: Collection[int], report_years: Collection[int], seed: int = SEED
) -> dict:
    """Full backtest: calibrate + tune scales on fit_years, score all candidates on
    the held-out report_years, level-control, verdict per challenger vs xslg."""
    # Barrel HR/PA ~ brl_pa is calibrated on the FULL fit-year skill range (PA floor +
    # common support, NO mover filter) per the spec, so the slope is unbiased; scale
    # tuning uses the mover population `fit` (matching what report scoring evaluates).
    calib = fit_barrel_calibration(build_hr_records(corpus, fit_years, hr_move_min=0.0))
    fit = build_hr_records(corpus, fit_years)
    report = build_hr_records(corpus, report_years)
    actual = [r["actual_hr"] for r in report]

    scales = {c: tune_scale(fit, c, calib) for c in CANDIDATES}
    scales["xslg_shipped"] = SHIPPED_XSLG_SCALE
    forwards = {
        "xslg": _forwards(report, "xslg", calib, scales["xslg"]),
        "xslg_shipped": _forwards(report, "xslg", calib, SHIPPED_XSLG_SCALE),
        "barrel": _forwards(report, "barrel", calib, scales["barrel"]),
        "xhr": _forwards(report, "xhr", calib, scales["xhr"]),
        "surface": [r["surface_hr"] for r in report],
        "prior": [r["prior_hr"] for r in report],
    }
    spearman = {k: _spearman(v, actual) for k, v in forwards.items()}
    mae = {k: _mean_mae(v, actual) for k, v in forwards.items()}

    verdicts: dict[str, dict] = {}
    for cand in ("barrel", "xhr"):
        ci = _bootstrap_diff(forwards[cand], forwards["xslg"], actual, seed=seed)
        tiers = level_control(report, cand, calib)
        tier_signs = [1 if t >= 0 else -1 for t in tiers]
        verdicts[cand] = {
            "ci_vs_xslg": ci,
            "tier_spearman": tiers,
            "verdict": _verdict_for(ci=ci, mae_delta=mae[cand] - mae["xslg"], tier_signs=tier_signs),
        }
    return {
        "n_fit": len(fit),
        "n_report": len(report),
        "barrel_calib": calib,
        "scales": scales,
        "spearman": spearman,
        "mae": mae,
        "verdicts": verdicts,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_hr_confirm.py -v`
Expected: PASS. (If `test_barrel_calibration_recovers_known_line` shows the stray `pytest_approx` line, delete that line before running -- keep only the `math.isclose` asserts.)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/hr_confirm.py tests/test_analysis/test_hr_confirm.py
git commit -m "feat(breakout): HR-confirm candidate math + verdict rule (#262)"
```

---

### Task 3: The backtest script + integration test

**Files:**
- Create: `scripts/backtest_hr_confirm.py`
- Test: `tests/test_scripts/test_backtest_hr_confirm.py`

**Interfaces:**
- Consumes: `hr_confirm.run` (Task 2), `backtest_breakout.build_corpus`, `hr_confirm` constants.
- Produces: a CLI that prints the per-candidate table + verdict and writes `data/stats/hr_confirm_backtest_results.csv`. No importable API beyond `main()`.

- [ ] **Step 1: Write a failing integration test on a synthetic corpus**

Create `tests/test_scripts/test_backtest_hr_confirm.py`. Build a tiny two-year corpus by hand (mirroring `tests/test_scripts/test_backtest_breakout.py`), so `run` executes end-to-end with no network:

```python
from fantasy_baseball.analysis import breakout, hr_confirm


def _row(mlbam, pa, slg, xslg, brl_pa, xhr):
    return breakout.SkillLuckRow(
        mlbam=mlbam, player_type="hitter", pa=pa, ip=0.0, age=27.0,
        barrel_pct=None, xslg=xslg, slg=slg, xba=None, ba=None, babip=None,
        xwoba=None, woba=None, k_pct=None, bb_pct=None, brl_pa=brl_pa, xhr=xhr,
    )


_PA = 600.0
# Low-HR prior history so every current surface is a clear mover
# (|surface-prior| >> HR_MOVE_MIN); avoids any tuning-to-pass on the filter.
_PRIOR_HR = 8


def _entry(hr, next_hr, brl_pa, xhr, mlbam):
    surface = {"pa": _PA, "hr": hr, "r": 80, "rbi": 80, "sb": 5, "avg": 0.270}
    actual_next = breakout.line_rates(
        {"pa": _PA, "hr": next_hr, "r": 80, "rbi": 80, "sb": 5, "avg": 0.270}, "hitter"
    )
    prior_line = {"pa": _PA, "hr": _PRIOR_HR, "r": 60, "rbi": 60, "sb": 5, "avg": 0.250}
    hist = [
        (2018, breakout.line_rates(prior_line, "hitter")),
        (2019, breakout.line_rates(prior_line, "hitter")),
    ]
    return (surface, _row(mlbam, _PA, 0.520, 0.470, brl_pa, xhr), actual_next, hist, None)


def test_run_produces_wellformed_verdicts():
    # 12 hitters/year, HR spread 26..37 (all clear movers vs the low prior); brl_pa
    # varies so the barrel calibration is non-degenerate; overperformers regress
    # next year (next_hr = hr - 4), and xhr tracks the regressed level.
    def year():
        data = {}
        for i in range(12):
            hr = 26 + i
            brl_pa = 0.04 + 0.004 * i
            data[1000 + i] = _entry(hr, hr - 4, brl_pa, (hr - 4), 1000 + i)
        return data

    corpus = {2020: year(), 2021: year()}
    res = hr_confirm.run(corpus, fit_years=[2020], report_years=[2021])
    assert res["n_report"] == 12  # all 12 clear the PA floor + mover filter
    assert set(res["verdicts"]) == {"barrel", "xhr"}
    for cand in ("barrel", "xhr"):
        assert res["verdicts"][cand]["verdict"] in {
            "no (CI includes 0)",
            "level-confounded -- do not wire in",
            "CI-positive but MAE-inconsistent",
            "wire-in eligible",
            "inconclusive (thin sample for level-control)",
        }
    # deterministic: identical CI bounds on a second run (seeded bootstrap)
    res2 = hr_confirm.run(corpus, fit_years=[2020], report_years=[2021])
    assert res["verdicts"]["xhr"]["ci_vs_xslg"] == res2["verdicts"]["xhr"]["ci_vs_xslg"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_backtest_hr_confirm.py -v`
Expected: FAIL -- collection/`ImportError` or an assertion, because Task 2's `hr_confirm.run` is what it exercises (not the script's I/O). Once Task 2 is implemented it PASSES: the fixture is a mover by construction (surface HR 26..37 vs a prior of 8), so `n_report == 12` deterministically -- no fixture tuning required.

- [ ] **Step 3: Write the script**

Create `scripts/backtest_hr_confirm.py`:

```python
"""Backtest: does barrel rate or park-adjusted xHR confirm next-year HR better than
the SLG-vs-xSLG proxy in w_for_stat? Hitters, common support 2016-2024 (xHR starts
2016). Go/no-go = challenger's bootstrap CI vs xslg excludes 0, with a level-control
veto. Wires nothing in -- that is a verdict-gated follow-up.

See docs/superpowers/specs/2026-07-26-hr-confirmation-backtest-design.md (#262).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

import backtest_breakout as bt  # build_corpus + cache/projection paths
from fantasy_baseball.analysis import hr_confirm

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "data" / "stats" / "hr_confirm_backtest_results.csv"

SOURCE_YEARS = list(range(2016, 2025))  # 2016..2024 predicting 2017..2025 (common support)
FIT_YEARS = list(range(2016, 2021))  # 2016..2020: tuning/calibration only, never scored
REPORT_YEARS = list(range(2021, 2025))  # 2021..2024: held-out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = bt.build_corpus(bt.SKILL_LUCK_CACHE_DIR, bt.PROJECTIONS_ROOT, SOURCE_YEARS)
    res = hr_confirm.run(corpus, fit_years=FIT_YEARS, report_years=REPORT_YEARS)

    print("HR-confirmation backtest -- barrels/xHR vs the SLG-vs-xSLG proxy")
    print(f"  fit {FIT_YEARS}  report {REPORT_YEARS}")
    print(f"  n fit {res['n_fit']}  n report {res['n_report']}")
    slope, intercept = res["barrel_calib"]
    print(f"  barrel calib: HR/PA = {slope:+.4f}*brl_pa {intercept:+.4f}")
    print("  tuned scales:", {k: round(v, 3) for k, v in res["scales"].items()})
    print("  Spearman(forward HR/PA, next-year HR/PA)  (higher = ranks the future better):")
    for k, v in res["spearman"].items():
        print(f"    {k:14s} {v:+.3f}")
    print("  rate MAE (lower = better forward line):")
    for k, v in res["mae"].items():
        print(f"    {k:14s} {v:.4f}")
    for cand, d in res["verdicts"].items():
        lo, hi = d["ci_vs_xslg"]
        tiers = ", ".join(f"{t:+.2f}" for t in d["tier_spearman"])
        print(f"  {cand.upper()} vs xslg: CI [{lo:+.3f}, {hi:+.3f}]  "
              f"level-control(low->high tiers) [{tiers}]  -> {d['verdict']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"metric": "n_report", "value": res["n_report"]}]
    for k, v in res["spearman"].items():
        rows.append({"metric": f"spearman_{k}", "value": v})
    for k, v in res["mae"].items():
        rows.append({"metric": f"mae_{k}", "value": v})
    for cand, d in res["verdicts"].items():
        rows.append({"metric": f"{cand}_ci_low", "value": d["ci_vs_xslg"][0]})
        rows.append({"metric": f"{cand}_ci_high", "value": d["ci_vs_xslg"][1]})
        rows.append({"metric": f"{cand}_verdict", "value": d["verdict"]})
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the integration test**

Run: `pytest tests/test_scripts/test_backtest_hr_confirm.py -v`
Expected: PASS.

- [ ] **Step 5: Run the real backtest (fetches xHR 2016-2024 once, then cached)**

Run: `python scripts/backtest_hr_confirm.py`
Expected: prints the table + a verdict per challenger; writes `data/stats/hr_confirm_backtest_results.csv`. First run fetches 9 xHR CSVs to `data/skill_luck/sc_hr_h_*.csv` (network). If Savant is unreachable, the run errors on the first fetch -- rerun when reachable; the cache persists.

- [ ] **Step 6: Commit (script + cached xHR CSVs + results)**

```bash
git add scripts/backtest_hr_confirm.py tests/test_scripts/test_backtest_hr_confirm.py data/skill_luck/sc_hr_h_*.csv data/stats/hr_confirm_backtest_results.csv
git commit -m "feat(breakout): HR-confirmation backtest script + result (#262)"
```

---

### Task 4: Report the verdict + end-of-effort checks

**Files:** none new (verification + reporting).

- [ ] **Step 1: Interpret the result against the guardrail**

Read the printed verdict and `data/stats/hr_confirm_backtest_results.csv`. Map to the spec's decision:
- Any challenger `"wire-in eligible"` -> report it as a candidate to wire into `w_for_stat`'s HR branch in a follow-up (NOT in this branch).
- `"level-confounded"` / `"CI-positive but MAE-inconsistent"` / `"no (CI includes 0)"` -> proxy stays; report the negative result.
State the numbers (Spearman deltas, CI bounds, per-tier level-control) plainly; do not over-read absolute Spearman (survivorship + non-independence caveat from the spec).

- [ ] **Step 2: Full end-of-effort checks (repo rule)**

Run and fix every failure; show outputs:

```bash
pytest -v
ruff check .
ruff format --check .
vulture
```

Run `mypy` only if a touched file (`breakout.py`, `skill_luck.py`, `hr_confirm.py`) is under `[tool.mypy].files` in `pyproject.toml` -- check first.

- [ ] **Step 3: Note the wiring follow-up**

If a challenger cleared the gate, open/refresh the follow-up note (issue #262 comment or a new issue): the wire-in is a separate small diff -- replace `breakout.py:167` HR branch with the winning candidate's confirm, add a pinned equivalence/behavior test, and add the xHR field to the live report path. Do it only after this backtest result is reviewed.

## Self-Review

**Spec coverage:**
- xHR fetcher (endpoint, coverage, cache `sc_hr_h_{year}.csv`) -> Task 1 Steps 6-7. [covered]
- `SkillLuckRow.xhr` + `brl_pa` rename -> Task 1 Steps 3, 5. [covered]
- Three candidates + barrel `HR/PA ~ brl_pa` calibration -> Task 2 (`fit_barrel_calibration`, `confirm`, `expected_hr_rate`). [covered]
- Scale tuning: grid, same metric (fit Spearman), ordinal `_spearman`, shipped-0.150 reported -> Task 2 `tune_scale`, `run` (`xslg_shipped`). [covered]
- Primary test: forward-line formula, Spearman + MAE + seeded bootstrap CI -> Task 2 `forward_hr`, `run`, `_bootstrap_diff(seed=SEED)`. [covered]
- Verdict: CI-sole-gate + MAE-consistency epsilon + level-control veto (>=2 of 3 tiers) -> Task 2 `_verdict_for`, pinned in tests. [covered]
- Level-control: per-candidate own signal, prior-HR/PA terciles -> Task 2 `signed_gap`, `level_control`. [covered]
- Sample/splits: common support 2016-2024, fit 2016-2020, report 2021-2024, `PA_FLOOR`/`HR_MOVE_MIN` constants -> Task 3 constants + Task 2 `build_hr_records`. [covered]
- No `w_for_stat` change; wiring deferred -> Task 4 Step 3. [covered]
- Testing: fetcher rename/empty-tolerance, calibration, confirm monotonicity, verdict-logic pins, xslg==w_for_stat equivalence -> Tasks 1-2 tests. [covered]
- Bootstrap determinism: `_bootstrap_diff` already seeds (Global Constraints); the spec's "seed 12345" is `SEED`. Pinned as a REQUIRED assertion in Task 3's `test_run_produces_wellformed_verdicts` -- two `run(...)` calls on the same corpus return identical `ci_vs_xslg`. [covered]
- Barrel calibration population: fit on the FULL fit-year skill range (`hr_move_min=0.0`), not the mover subset, per the spec -> Task 2 `run` (`fit_barrel_calibration(build_hr_records(..., hr_move_min=0.0))`). [covered]

**Placeholder scan:** No TODO/TBD. The one stray `pytest_approx` reference in Task 2 Step 1 is explicitly flagged to delete. Empty-tolerance, network-failure, and thin-tercile paths all have concrete handling.

**Type consistency:** `HrRecord` keys, `BarrelCalib`, `confirm(...scale=)`, `forward_hr`, `tune_scale`, `signed_gap`, `level_control`, `_verdict_for`, and `run` signatures are used identically across Tasks 2 and 3. `SkillLuckRow` field names (`brl_pa`, `xhr`) match between Task 1 (definition) and Tasks 2-3 (use). Robustness note (barrel/xSLG on 2015-2024) is optional and NOT a task -- the spec marks it "not the gate"; it can be run by widening `SOURCE_YEARS` after the gate, relying on Task 1's empty-2015 tolerance.
