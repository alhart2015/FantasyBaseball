# EV-Based Roto Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the discrete tie-band-based `score_roto` and its accompanying defensive-comfort / erosion heuristics with a continuous expected-value formulation that integrates projection uncertainty directly into roto-point scoring.

**Architecture:** For each team in each category, compute `pts = 1 + Σ_{j≠me} P(me > j)` where `P(A > B) = Φ((μ_A − μ_B) / √(σ_A² + σ_B²))` under Gaussian independence of team totals. This single continuous formula subsumes the tie-band threshold, `compute_defense_comfort`, the erosion penalty, and the gain-discount logic in `score_swap`. When `team_sds=None` or all σ=0, the formula reproduces the current rank-based scoring exactly (including the averaged-ranks convention for ties), so the `None` path stays behaviorally identical to `main`.

**Tech Stack:** Python 3.11+, stdlib `math.erf` / `math.sqrt`, pytest, numpy not required. Existing codebase patterns: `from __future__ import annotations`, `CategoryStats` for team totals, `STAT_VARIANCE` dict of CVs.

---

## Pre-work context (read before starting)

**Branch state going in:** `feat/delta-roto-tie-band` is 4 commits ahead of `main` (`ba82ab5` → `b89f7ee` → `4af4f4e` → `780c9f4`). All four commits implement the tie-band approach this plan replaces. Task 1 resets the branch hard to `main`, wiping all four. The `project_team_sds` function (added in `ba82ab5`) is re-introduced fresh in Task 3 — it is mathematically sound and we want to keep its logic, but we rebuild it as part of the EV flow rather than cherry-pick.

**What is re-introduced from the old work (re-implemented, not cherry-picked):**
- `project_team_sds(roster)` in `scoring.py` — analytical SD propagation using `STAT_VARIANCE` CVs.
- `team_sds` caching in `season_data.py::run_full_refresh` — scaled by `√fraction_remaining`.
- `team_sds` threading through `audit_roster`, `compute_delta_roto`, `compute_comparison_standings`.

**What is permanently deleted:**
- `compute_defense_comfort` function (`src/fantasy_baseball/lineup/delta_roto.py`).
- `fragile_threshold`, `tie_floor`, `erosion_weight`, `erosion_cap` constants and kwargs.
- `defense_before` / `defense_after` fields on `CategoryDelta`.
- `tie_band_sd_factor` parameter (replaced by raw σ).
- `delta_roto:` config section in `config/league.yaml` (all four knobs above).
- `docs/tie_band_branch_state.md` (describes the obsolete approach).

**Frontend blast radius:** Zero. Grep of `src/fantasy_baseball/web/templates`, `*.html`, `*.js` for `defense_before`, `defense_after`, `erosion`, `compute_defense_comfort` returns no matches. The fields are serialized into `delta_roto.to_dict()` but no UI reads them.

**Commit message convention:** This repo uses conventional commits (`<type>(<scope>): message`). Types seen recently: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`. Keep commit messages imperative, lowercase after the scope, no trailing period. Every commit in this plan ends with:

```
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Task 1: Reset branch and delete obsolete state doc

**Files:**
- Delete: `docs/tie_band_branch_state.md`
- Create (staged for commit): `docs/superpowers/plans/2026-04-15-ev-roto-scoring.md` (this file)

- [ ] **Step 1: Confirm current branch and commit position**

Run:
```bash
git status
git log --oneline main..HEAD
```

Expected: branch is `feat/delta-roto-tie-band`, four commits ahead of main (`780c9f4`, `4af4f4e`, `b89f7ee`, `ba82ab5`). Working tree has `docs/tie_band_branch_state.md` tracked with modifications.

- [ ] **Step 2: Stash plan file so reset doesn't wipe it**

The plan file is currently untracked. `git reset --hard` preserves untracked files by default, but verify:

```bash
ls docs/superpowers/plans/2026-04-15-ev-roto-scoring.md
```

Expected: file exists. If missing, stop and recreate from this document before proceeding.

- [ ] **Step 3: Hard-reset branch to main**

This is destructive and wipes four commits. User has approved.

```bash
git reset --hard main
```

Expected output: `HEAD is now at <hash from main>` with `main`'s commit subject.

- [ ] **Step 4: Verify reset cleanly wiped tie-band code**

```bash
git log --oneline main..HEAD
grep -rn "tie_band_sd_factor\|compute_defense_comfort\|project_team_sds\|fragile_threshold" src/ tests/ | head -20
```

Expected: `git log` produces zero output (branch == main). The grep finds no matches in `src/` or `tests/` (all tie-band-era code is gone). The `config/league.yaml` may still contain `fragile_threshold` etc. if that config was introduced earlier — verify:

```bash
grep -n "fragile_threshold\|tie_floor\|erosion_weight\|erosion_cap" config/league.yaml
```

If present, those lines exist from before this branch and will be removed in Task 10. If absent, skip Task 10.

- [ ] **Step 5: Verify plan file survived reset**

```bash
ls docs/superpowers/plans/
cat docs/superpowers/plans/2026-04-15-ev-roto-scoring.md | head -5
```

Expected: the plan file is still present and readable.

- [ ] **Step 6: Verify tie_band_branch_state.md is gone**

```bash
ls docs/tie_band_branch_state.md 2>&1
```

Expected: `ls: cannot access ...: No such file or directory`. The file was introduced in `ba82ab5`, so resetting to `main` removes it automatically. No manual deletion needed.

- [ ] **Step 7: Run baseline tests to confirm starting state**

```bash
pytest -q 2>&1 | tail -5
```

Expected: all tests pass (should match `main` test count — roughly 1050-ish, exact number depends on `main`).

- [ ] **Step 8: Commit the plan document**

```bash
git add docs/superpowers/plans/2026-04-15-ev-roto-scoring.md
git commit -m "$(cat <<'EOF'
docs(plan): EV-based roto scoring implementation plan

Plan to replace the tie-band approach with expected-value scoring
using pairwise Gaussian win-probabilities. Wipes four tie-band
commits from the branch (reset to main) and specifies a 12-task
TDD rebuild centered on project_team_sds + pairwise EV in
score_roto.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds, `git log --oneline main..HEAD` shows one commit.

---

## Task 2: Add `_prob_beats` helper with direct tests

**Files:**
- Modify: `src/fantasy_baseball/scoring.py` (add private helper)
- Test: `tests/test_scoring.py` (add `TestProbBeats` class)

- [ ] **Step 1: Write the failing tests**

Append this to `tests/test_scoring.py` (before the existing `TestScoreRoto` class if one exists, otherwise at end of file — pick whichever location groups with other math unit tests):

```python
import math
from fantasy_baseball.scoring import _prob_beats


class TestProbBeats:
    """Unit tests for the pairwise Gaussian win-probability helper."""

    def test_equal_means_zero_sd_returns_half(self):
        assert _prob_beats(100, 100, 0, 0, higher_is_better=True) == 0.5

    def test_deterministic_win_when_ahead_with_zero_sd(self):
        assert _prob_beats(110, 100, 0, 0, higher_is_better=True) == 1.0

    def test_deterministic_loss_when_behind_with_zero_sd(self):
        assert _prob_beats(90, 100, 0, 0, higher_is_better=True) == 0.0

    def test_equal_means_positive_sd_returns_half(self):
        assert _prob_beats(100, 100, 10, 10, higher_is_better=True) == pytest.approx(0.5)

    def test_one_sd_ahead_equal_variance(self):
        # μ_a - μ_b = 14.14, combined sd = sqrt(100 + 100) = 14.14
        # z = 14.14 / 14.14 = 1.0 → Φ(1.0) ≈ 0.8413
        assert _prob_beats(114.14, 100, 10, 10, higher_is_better=True) == pytest.approx(
            0.8413, abs=1e-3
        )

    def test_inverse_flips_direction(self):
        # Lower is better: A has smaller μ, so A "beats" B.
        assert _prob_beats(3.50, 4.00, 0, 0, higher_is_better=False) == 1.0
        assert _prob_beats(4.00, 3.50, 0, 0, higher_is_better=False) == 0.0

    def test_inverse_equal_means_still_half(self):
        assert _prob_beats(3.75, 3.75, 0.2, 0.2, higher_is_better=False) == pytest.approx(0.5)

    def test_zero_sd_with_negative_diff_returns_zero(self):
        # Degenerate case: combined sd == 0 and μ_a < μ_b.
        assert _prob_beats(50, 100, 0, 0, higher_is_better=True) == 0.0

    def test_complementary_probabilities_sum_to_one(self):
        # P(A > B) + P(B > A) == 1 when means differ.
        p_ab = _prob_beats(110, 100, 5, 5, higher_is_better=True)
        p_ba = _prob_beats(100, 110, 5, 5, higher_is_better=True)
        assert p_ab + p_ba == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scoring.py::TestProbBeats -v 2>&1 | tail -20
```

Expected: ImportError — `_prob_beats` is not defined. All 9 tests fail.

- [ ] **Step 3: Implement `_prob_beats` in `scoring.py`**

Add these imports near the top of `src/fantasy_baseball/scoring.py` if not already present:

```python
from math import erf, sqrt
```

Add this function below `_stat` and above the `_GENERIC_SLOTS` constant (so it lives near the other private helpers):

```python
def _prob_beats(
    mu_a: float,
    mu_b: float,
    sd_a: float,
    sd_b: float,
    *,
    higher_is_better: bool,
) -> float:
    """P(team A's category total exceeds team B's) under Gaussian independence.

    When combined SD is zero, this is a step function: 1.0 if A is ahead,
    0.0 if behind, 0.5 on exact equality. Positive combined SD smooths the
    step into a continuous sigmoid. The ``higher_is_better`` flag flips
    the direction for inverse categories (ERA, WHIP).

    This is the pairwise primitive the EV-based ``score_roto`` sums over.
    """
    diff = (mu_a - mu_b) if higher_is_better else (mu_b - mu_a)
    combined = sqrt(sd_a * sd_a + sd_b * sd_b)
    if combined == 0.0:
        if diff > 0:
            return 1.0
        if diff < 0:
            return 0.0
        return 0.5
    return 0.5 * (1.0 + erf(diff / (combined * sqrt(2.0))))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scoring.py::TestProbBeats -v 2>&1 | tail -15
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/scoring.py tests/test_scoring.py
git commit -m "$(cat <<'EOF'
feat(scoring): add _prob_beats pairwise Gaussian primitive

Base operation for the EV-based roto scoring that replaces the
tie-band approach. Zero combined SD recovers the step-function
semantics of rank-based scoring; positive SD smooths into a
continuous sigmoid.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `project_team_sds` with full test coverage

**Files:**
- Modify: `src/fantasy_baseball/scoring.py` (add public function)
- Test: `tests/test_scoring.py` (add `TestProjectTeamSDs` class)

- [ ] **Step 1: Write the failing tests**

Append this after `TestProbBeats` in `tests/test_scoring.py`:

```python
from fantasy_baseball.scoring import project_team_sds
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.utils.constants import STAT_VARIANCE


def _make_hitter(name, **stats):
    """Build a Player with HitterStats for unit tests."""
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=["OF"],
        selected_position="OF",
        status="",
        rest_of_season=HitterStats(**stats),
    )


def _make_pitcher(name, **stats):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=["SP"],
        selected_position="SP",
        status="",
        rest_of_season=PitcherStats(**stats),
    )


class TestProjectTeamSDs:
    """Per-team-per-category SD from analytical variance propagation."""

    def test_empty_roster_returns_zeros(self):
        sds = project_team_sds([])
        for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]:
            assert sds[cat] == 0.0

    def test_single_hitter_counting_stat(self):
        p = _make_hitter("A", r=80, hr=20, rbi=70, sb=10, h=150, ab=500)
        sds = project_team_sds([p])
        # SD_R = CV_r * sqrt(r^2) = CV_r * r  (single player case)
        assert sds["R"] == pytest.approx(STAT_VARIANCE["r"] * 80)
        assert sds["HR"] == pytest.approx(STAT_VARIANCE["hr"] * 20)

    def test_independence_aggregates_in_quadrature(self):
        a = _make_hitter("A", r=100, hr=0, rbi=0, sb=0, h=0, ab=0)
        b = _make_hitter("B", r=60, hr=0, rbi=0, sb=0, h=0, ab=0)
        sds = project_team_sds([a, b])
        expected = STAT_VARIANCE["r"] * math.sqrt(100**2 + 60**2)
        assert sds["R"] == pytest.approx(expected)

    def test_avg_uses_hits_variance_over_total_ab(self):
        a = _make_hitter("A", r=0, hr=0, rbi=0, sb=0, h=150, ab=500)
        b = _make_hitter("B", r=0, hr=0, rbi=0, sb=0, h=100, ab=400)
        sds = project_team_sds([a, b])
        expected = STAT_VARIANCE["h"] * math.sqrt(150**2 + 100**2) / (500 + 400)
        assert sds["AVG"] == pytest.approx(expected)

    def test_era_scales_by_nine_over_ip(self):
        a = _make_pitcher("A", w=10, k=180, sv=0, ip=180, er=60, bb=40, h_allowed=140)
        b = _make_pitcher("B", w=8, k=140, sv=0, ip=150, er=55, bb=35, h_allowed=130)
        sds = project_team_sds([a, b])
        expected = 9.0 * STAT_VARIANCE["er"] * math.sqrt(60**2 + 55**2) / (180 + 150)
        assert sds["ERA"] == pytest.approx(expected)

    def test_whip_combines_bb_and_h_allowed_variance(self):
        a = _make_pitcher("A", w=0, k=0, sv=0, ip=100, er=0, bb=30, h_allowed=90)
        sds = project_team_sds([a])
        expected = math.sqrt(
            STAT_VARIANCE["bb"]**2 * 30**2
            + STAT_VARIANCE["h_allowed"]**2 * 90**2
        ) / 100
        assert sds["WHIP"] == pytest.approx(expected)

    def test_all_ten_categories_present(self):
        p = _make_hitter("A", r=50, hr=10, rbi=40, sb=5, h=100, ab=400)
        sds = project_team_sds([p])
        assert set(sds.keys()) == {"R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"}

    def test_displacement_kwarg_defaults_true(self):
        # Bench players excluded by default. A bench-slot hitter should
        # not contribute to SDs when displacement=True (default).
        active = _make_hitter("A", r=80, hr=20, rbi=70, sb=10, h=150, ab=500)
        bench = Player(
            name="B",
            player_type=PlayerType.HITTER,
            positions=["OF"],
            selected_position="BN",
            status="",
            rest_of_season=HitterStats(r=80, hr=20, rbi=70, sb=10, h=150, ab=500),
        )
        sds_with_bench = project_team_sds([active, bench])
        sds_active_only = project_team_sds([active])
        assert sds_with_bench["R"] == pytest.approx(sds_active_only["R"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scoring.py::TestProjectTeamSDs -v 2>&1 | tail -20
```

Expected: ImportError or AttributeError — `project_team_sds` is not defined.

- [ ] **Step 3: Implement `project_team_sds`**

Append this function at the end of `src/fantasy_baseball/scoring.py` (after `score_roto` — or wherever `project_team_stats` lives, since they are siblings):

```python
def project_team_sds(
    roster,
    *,
    displacement: bool = True,
) -> dict[str, float]:
    """Aggregate per-player projection variance into team-level SDs.

    Uses ``STAT_VARIANCE`` (per-stat CV calibrated from 2022-2024
    Steamer+ZiPS vs actuals) under a player-independence assumption:

        SD_cat_team = CV_cat * sqrt(sum_over_players(stat_i^2))

    Rate stats propagate through their component totals:

        SD_AVG  = CV_h * sqrt(sum h_i^2) / sum_AB
        SD_ERA  = 9 * CV_er * sqrt(sum er_i^2) / sum_IP
        SD_WHIP = sqrt(CV_bb^2 * sum bb_i^2 + CV_ha^2 * sum ha_i^2) / sum_IP

    ``displacement`` matches :func:`project_team_stats` — bench excluded,
    IL players displace their worst active positional match.

    Returns ``{cat: sd}`` for every category in ``ALL_CATS``. Empty
    roster returns zeros.
    """
    if displacement:
        roster = _apply_displacement(roster)

    h_sum_sq: dict[str, float] = {k: 0.0 for k in HITTING_COUNTING}
    p_sum_sq: dict[str, float] = {k: 0.0 for k in PITCHING_COUNTING}
    total_ab = 0.0
    total_ip = 0.0

    for p in roster:
        ptype = _get(p, "player_type")
        if ptype == PlayerType.HITTER:
            for k in HITTING_COUNTING:
                v = _stat(p, k)
                h_sum_sq[k] += v * v
            total_ab += _stat(p, "ab")
        elif ptype == PlayerType.PITCHER:
            for k in PITCHING_COUNTING:
                v = _stat(p, k)
                p_sum_sq[k] += v * v
            total_ip += _stat(p, "ip")

    sds: dict[str, float] = {c: 0.0 for c in ALL_CATS}
    for stat_key, cat in [("r", "R"), ("hr", "HR"), ("rbi", "RBI"), ("sb", "SB")]:
        sds[cat] = STAT_VARIANCE[stat_key] * sqrt(h_sum_sq[stat_key])
    for stat_key, cat in [("w", "W"), ("k", "K"), ("sv", "SV")]:
        sds[cat] = STAT_VARIANCE[stat_key] * sqrt(p_sum_sq[stat_key])
    if total_ab > 0:
        sds["AVG"] = STAT_VARIANCE["h"] * sqrt(h_sum_sq["h"]) / total_ab
    if total_ip > 0:
        sds["ERA"] = 9.0 * STAT_VARIANCE["er"] * sqrt(p_sum_sq["er"]) / total_ip
        whip_var = (
            (STAT_VARIANCE["bb"] ** 2) * p_sum_sq["bb"]
            + (STAT_VARIANCE["h_allowed"] ** 2) * p_sum_sq["h_allowed"]
        )
        sds["WHIP"] = sqrt(whip_var) / total_ip
    return sds
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_scoring.py::TestProjectTeamSDs -v 2>&1 | tail -15
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/scoring.py tests/test_scoring.py
git commit -m "$(cat <<'EOF'
feat(scoring): add project_team_sds for analytical team-level variance

Propagates per-stat CVs from STAT_VARIANCE through team totals
assuming player independence. Same displacement semantics as
project_team_stats so the SD matches the projection it's paired
with. Will feed the pairwise EV in score_roto.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Rewrite `score_roto` with pairwise EV

**Files:**
- Modify: `src/fantasy_baseball/scoring.py:300-388` (replace `score_roto` body, delete `_values_tied`)
- Test: `tests/test_scoring.py` (add `TestScoreRotoEV` class; existing `TestScoreRoto` stays as the `team_sds=None` backwards-compat suite)

- [ ] **Step 1: Write the failing tests**

Append this to `tests/test_scoring.py`:

```python
from fantasy_baseball.scoring import score_roto


def _twelve_team_stats(r_values):
    """Build ``{team: {R: value, other cats: 0}}`` for 12 teams."""
    teams = {}
    for i, r in enumerate(r_values):
        teams[f"T{i+1}"] = {
            "R": r, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0.0,
            "W": 0, "K": 0, "SV": 0, "ERA": 0.0, "WHIP": 0.0,
        }
    return teams


class TestScoreRotoEV:
    """Expected-value roto scoring with projection uncertainty."""

    def test_no_sds_matches_rank_scoring_distinct(self):
        # 12 distinct values → integer points 1..12.
        stats = _twelve_team_stats([100 + i for i in range(12)])
        roto = score_roto(stats)
        # T12 has highest R (111), gets 12 pts.
        assert roto["T12"]["R_pts"] == pytest.approx(12.0)
        assert roto["T1"]["R_pts"] == pytest.approx(1.0)

    def test_no_sds_exact_tie_averages_ranks(self):
        # Two teams tied at top: both get avg of 12 and 11 → 11.5.
        vals = [111, 111] + [100 + i for i in range(10)]
        stats = _twelve_team_stats(vals)
        roto = score_roto(stats)
        assert roto["T1"]["R_pts"] == pytest.approx(11.5)
        assert roto["T2"]["R_pts"] == pytest.approx(11.5)

    def test_no_sds_three_way_tie_averages(self):
        # Three teams tied at top: avg of 12+11+10 = 11.
        vals = [111, 111, 111] + [100 + i for i in range(9)]
        stats = _twelve_team_stats(vals)
        roto = score_roto(stats)
        for t in ["T1", "T2", "T3"]:
            assert roto[t]["R_pts"] == pytest.approx(11.0)

    def test_zero_sds_matches_none_path(self):
        stats = _twelve_team_stats([100 + i for i in range(12)])
        roto_none = score_roto(stats)
        zero_sds = {t: {c: 0.0 for c in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}
                    for t in stats}
        roto_zero = score_roto(stats, team_sds=zero_sds)
        for t in stats:
            for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]:
                assert roto_zero[t][f"{cat}_pts"] == pytest.approx(
                    roto_none[t][f"{cat}_pts"]
                )

    def test_large_sds_collapse_toward_middle(self):
        # Huge σ >> any μ gap → every team's pairwise P ≈ 0.5 → pts ≈ (N+1)/2 = 6.5.
        stats = _twelve_team_stats([100 + i for i in range(12)])
        huge_sds = {t: {c: 1_000_000 for c in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}
                    for t in stats}
        roto = score_roto(stats, team_sds=huge_sds)
        for t in stats:
            assert roto[t]["R_pts"] == pytest.approx(6.5, abs=0.01)

    def test_monotone_in_own_stat(self):
        # Increasing team i's stat never decreases its EV points.
        stats = _twelve_team_stats([100 + i for i in range(12)])
        sds = {t: {c: 5.0 for c in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}
               for t in stats}
        before = score_roto(stats, team_sds=sds)["T5"]["R_pts"]
        stats["T5"]["R"] = 108  # was 104, now 108
        after = score_roto(stats, team_sds=sds)["T5"]["R_pts"]
        assert after > before

    def test_total_pts_per_category_invariant(self):
        # Σ pts across teams in a category = N*(N+1)/2 = 78 for N=12.
        stats = _twelve_team_stats([100 + i for i in range(12)])
        sds = {t: {c: 5.0 for c in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}
               for t in stats}
        roto = score_roto(stats, team_sds=sds)
        total_r = sum(roto[t]["R_pts"] for t in stats)
        assert total_r == pytest.approx(78.0, abs=1e-6)

    def test_inverse_category_direction(self):
        # ERA: lower is better. Team with lowest ERA gets highest pts.
        stats = _twelve_team_stats([0] * 12)
        for i, t in enumerate(stats):
            stats[t]["ERA"] = 3.0 + i * 0.1
        roto = score_roto(stats)
        assert roto["T1"]["ERA_pts"] == pytest.approx(12.0)
        assert roto["T12"]["ERA_pts"] == pytest.approx(1.0)

    def test_small_swap_within_uncertainty_produces_small_delta(self):
        # Two teams tied at 100 R with σ=10 each. Moving 1 R changes
        # pts by only ~0.03, not the full 1.0 of a rank flip.
        stats = _twelve_team_stats([100, 100, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50])
        sds = {t: {c: 10.0 if c == "R" else 1.0
                   for c in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}
               for t in stats}
        before = score_roto(stats, team_sds=sds)["T1"]["R_pts"]
        stats["T1"]["R"] = 101  # tiny edge
        after = score_roto(stats, team_sds=sds)["T1"]["R_pts"]
        delta = after - before
        assert 0 < delta < 0.1  # smooth, not a rank flip

    def test_total_includes_all_categories(self):
        stats = _twelve_team_stats([100 + i for i in range(12)])
        roto = score_roto(stats)
        for t in stats:
            assert roto[t]["total"] == pytest.approx(
                sum(roto[t][f"{cat}_pts"] for cat in
                    ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"])
            )
```

- [ ] **Step 2: Run the new tests to verify they fail**

The existing `score_roto` uses rank-based logic; the `team_sds` kwarg does not exist yet. Most tests should fail.

```bash
pytest tests/test_scoring.py::TestScoreRotoEV -v 2>&1 | tail -20
```

Expected: multiple TypeErrors (`team_sds` is an unexpected kwarg) plus failures on the tests that use positive σ.

- [ ] **Step 3: Rewrite `score_roto` body**

Replace the current `score_roto` function in `src/fantasy_baseball/scoring.py` (lines ≈300–357) and delete the `_values_tied` helper below it. Full replacement (keep the docstring updated):

```python
def score_roto(
    all_team_stats: dict,
    *,
    team_sds: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Assign expected-value roto points per team per category.

    For each team in each category, points equal

        pts = 1 + Σ_{j≠me} P(me > j)

    where ``P(A > B) = Φ((μ_A - μ_B) / √(σ_A² + σ_B²))`` under Gaussian
    independence of team totals (Φ is the standard-normal CDF). When
    ``team_sds`` is ``None`` or every σ is zero, this reduces to the
    step function that recovers the standard rank-based scoring,
    including the averaged-ranks convention on exact ties.

    Args:
        all_team_stats: ``{team: stats}``. Values can be ``dict`` or
            ``CategoryStats`` — both support ``[cat]`` indexing.
        team_sds: optional ``{team: {cat: sd}}``. ``None`` disables
            uncertainty (exact-rank behavior).

    Returns:
        ``{team: {R_pts, HR_pts, ..., total}}``. All values are floats.
        Points range from 1 (last) to N (first) for N teams.
    """
    teams = list(all_team_stats.keys())
    results: dict[str, dict[str, float]] = {t: {} for t in teams}

    for cat in ALL_CATS:
        higher_is_better = cat not in INVERSE_CATS
        for me in teams:
            mu_me = all_team_stats[me][cat]
            sd_me = team_sds.get(me, {}).get(cat, 0.0) if team_sds else 0.0
            pts = 1.0
            for other in teams:
                if other is me:
                    continue
                mu_o = all_team_stats[other][cat]
                sd_o = team_sds.get(other, {}).get(cat, 0.0) if team_sds else 0.0
                pts += _prob_beats(
                    mu_me, mu_o, sd_me, sd_o,
                    higher_is_better=higher_is_better,
                )
            results[me][f"{cat}_pts"] = pts

    for t in results:
        results[t]["total"] = sum(results[t].get(f"{c}_pts", 0.0) for c in ALL_CATS)

    return results
```

Delete the `_values_tied` function entirely (lines ≈360–388 in the old file).

- [ ] **Step 4: Run the new EV tests**

```bash
pytest tests/test_scoring.py::TestScoreRotoEV -v 2>&1 | tail -20
```

Expected: all 10 tests pass.

- [ ] **Step 5: Run the pre-existing score_roto tests to verify backwards compatibility**

The existing `TestScoreRoto` class on `main` tests things like `test_two_teams_simple`, `test_fractional_tiebreaker`, `test_inverse_stats_lower_is_better`, `test_all_categories_present`. These all call `score_roto` without `team_sds`, so they should still pass.

```bash
pytest tests/test_scoring.py::TestScoreRoto -v 2>&1 | tail -15
```

Expected: all pre-existing tests pass unchanged. If any fail, the new implementation broke backwards compatibility — stop and diagnose.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/scoring.py tests/test_scoring.py
git commit -m "$(cat <<'EOF'
refactor(scoring): replace rank-based score_roto with pairwise EV

Per-team per-category points now equal 1 + Σ P(me > j) under
Gaussian independence, using _prob_beats as the primitive. With
team_sds=None or all-zero σ, the formula exactly reproduces the
prior rank-based output including averaged ties. Removes the
_values_tied helper; removes the tie_band_sd_factor kwarg.

Eliminates the step-function behavior that caused day-to-day
audit recommendation flips (e.g., Adolis Garcia ↔ Taylor Ward).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Simplify `score_swap`, delete `compute_defense_comfort`

**Files:**
- Modify: `src/fantasy_baseball/lineup/delta_roto.py` (collapse score_swap, delete compute_defense_comfort, delete tuning constants)
- Test: `tests/test_lineup/test_delta_roto.py` (replace `TestComputeDefenseComfort` and `TestScoreSwap` with smaller EV-delta suite)

**Rationale for deleted tests (per CLAUDE.md — tests are the guardrail):** The deleted tests in this task asserted specific behaviors of the erosion penalty and defensive-comfort heuristic. Those mechanisms existed to patch the step-function failure mode of rank-based `score_roto`. The EV formulation makes that failure mode impossible, so the tests describe obsolete requirements. Each deleted test is listed with its reason below.

- [ ] **Step 1: Delete obsolete test classes, write new EV-delta tests**

Open `tests/test_lineup/test_delta_roto.py`.

**Delete:**
- `TestComputeDefenseComfort` class entirely (7 tests: `test_counting_stat_defense`, `test_inverse_stat_defense`, `test_first_place_has_infinite_attack_finite_defense`, `test_last_place_defense_is_infinite`, `test_team_sds_skips_band_tied_threat`, `test_team_sds_none_preserves_raw_behavior`). Reason: `compute_defense_comfort` is being deleted. Vulnerability is now priced into EV directly.
- `TestScoreSwap` class entirely (8 tests: `test_loss_counted_at_full_value`, `test_gain_discounted_at_exact_tie`, `test_gain_at_full_credit_when_comfortable`, `test_gain_partially_discounted_below_threshold`, `test_comfort_erosion_penalty`, `test_erosion_capped`, `test_no_double_penalty_on_loss_plus_erosion`, `test_total_is_sum_of_categories`). Reason: `score_swap` collapses to a pure subtraction; its discount/erosion logic no longer exists. A single `test_total_is_subtraction` replaces all 8.
- `test_team_sds_suppresses_spurious_erosion_on_gain` inside `TestComputeDeltaRoto`. Reason: tests the exact erosion-penalty path being removed. **But keep the fixture data and re-assert the EV-continuity outcome in the new test below.**
- `test_team_sds_collapses_swap_within_tie_band` inside `TestComputeDeltaRoto`. Reason: tests tie-band collapse. Replaced with EV-continuity check.

**Add** to `TestComputeDeltaRoto` (keep the class, keep `test_end_to_end_swap` and `test_drop_not_found_raises`):

```python
def test_team_sds_produces_small_delta_for_within_uncertainty_swap(self):
    """With wide σ, a 10-unit SB swap across two tied teams produces |ΔRoto| < 0.5,
    not the full 1.0 of a rank flip. Replaces the prior tie-band collapse test."""
    # Build 12 teams: user at 100 SB, rival at 99 SB, others spread away.
    projected_standings = [
        {"name": "User", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 100, "AVG": 0,
                                    "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
        {"name": "Rival", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 99, "AVG": 0,
                                     "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
    ] + [
        {"name": f"T{i}", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 10 + i, "AVG": 0,
                                     "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}}
        for i in range(10)
    ]
    # Roster: a SB-heavy hitter we drop and a SB-poor hitter we add → -10 SB.
    drop_hitter = _make_hitter("Drop", r=0, hr=0, rbi=0, sb=20, h=0, ab=100)
    add_hitter = _make_hitter("Add", r=0, hr=0, rbi=0, sb=10, h=0, ab=100)
    roster = [drop_hitter]
    # Wide SDs for SB (10 each) → combined ≈ 14 >> the 10-unit gap.
    team_sds = {t["name"]: {c: 0.0 for c in ["R", "HR", "RBI", "SB", "AVG",
                                              "W", "K", "SV", "ERA", "WHIP"]}
                for t in projected_standings}
    team_sds["User"]["SB"] = 10.0
    team_sds["Rival"]["SB"] = 10.0
    result = compute_delta_roto(
        drop_name="Drop", add_player=add_hitter, user_roster=roster,
        projected_standings=projected_standings, team_name="User",
        team_sds=team_sds,
    )
    # Without team_sds the swap collapses rank from 1 to 2 → ΔRoto_SB = -1.0.
    # With team_sds the combined-σ 14 >> gap 10 → |ΔRoto_SB| < 0.5.
    assert abs(result.categories["SB"].roto_delta) < 0.5

def test_team_sds_none_matches_pre_uncertainty_behavior(self):
    """team_sds=None reproduces exact-rank deltaRoto (backwards compat)."""
    # Same fixture: 10-unit SB drop moves user from rank 1 to rank 2.
    projected_standings = [
        {"name": "User", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 100, "AVG": 0,
                                    "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
        {"name": "Rival", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 99, "AVG": 0,
                                     "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
    ] + [
        {"name": f"T{i}", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 10 + i, "AVG": 0,
                                     "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}}
        for i in range(10)
    ]
    drop_hitter = _make_hitter("Drop", r=0, hr=0, rbi=0, sb=20, h=0, ab=100)
    add_hitter = _make_hitter("Add", r=0, hr=0, rbi=0, sb=10, h=0, ab=100)
    result = compute_delta_roto(
        drop_name="Drop", add_player=add_hitter, user_roster=[drop_hitter],
        projected_standings=projected_standings, team_name="User",
        team_sds=None,
    )
    assert result.categories["SB"].roto_delta == pytest.approx(-1.0)

def test_total_is_simple_subtraction_with_team_sds_none(self):
    """score_swap total == roto_after[team].total - roto_before[team].total."""
    # Minimal swap, no uncertainty.
    projected_standings = [
        {"name": "User", "stats": {"R": 100, "HR": 20, "RBI": 80, "SB": 30, "AVG": 0.270,
                                    "W": 20, "K": 200, "SV": 40, "ERA": 3.80, "WHIP": 1.20}},
        {"name": "Rival", "stats": {"R": 95, "HR": 18, "RBI": 75, "SB": 25, "AVG": 0.265,
                                     "W": 18, "K": 190, "SV": 35, "ERA": 3.90, "WHIP": 1.22}},
    ]
    drop_hitter = _make_hitter("Drop", r=10, hr=5, rbi=15, sb=5, h=20, ab=80)
    add_hitter = _make_hitter("Add", r=20, hr=8, rbi=25, sb=3, h=25, ab=80)
    result = compute_delta_roto(
        drop_name="Drop", add_player=add_hitter, user_roster=[drop_hitter],
        projected_standings=projected_standings, team_name="User",
        team_sds=None,
    )
    assert result.total == pytest.approx(result.after_total - result.before_total)
```

You will need `_make_hitter` and `_make_pitcher` helpers here — they may already exist in this test file; if not, copy them from the Task 3 test additions.

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_lineup/test_delta_roto.py -v 2>&1 | tail -25
```

Expected: the new tests fail (functions still take `comfort_before`/`comfort_after`), plus ImportErrors on any tests that still reference deleted `TestComputeDefenseComfort` or `TestScoreSwap`.

- [ ] **Step 3: Simplify `score_swap` and delete `compute_defense_comfort`**

Replace the contents of `src/fantasy_baseball/lineup/delta_roto.py` with this full file:

```python
"""deltaRoto — roto-point impact metric for player swaps.

Uses EV-based score_roto, so deltaRoto.total is simply the change in
total expected roto points across all categories. No tuning knobs,
no tie bands, no defensive-comfort heuristic — the Gaussian pairwise
win-probabilities price projection uncertainty and vulnerability
directly into the score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball.utils.constants import ALL_CATEGORIES

if TYPE_CHECKING:
    from fantasy_baseball.models.player import Player


@dataclass
class CategoryDelta:
    roto_delta: float

    @property
    def score(self) -> float:
        """Deprecated alias for ``roto_delta``. Kept for any consumers
        reading ``.score`` directly; remove in a follow-up."""
        return self.roto_delta


@dataclass
class DeltaRotoResult:
    total: float
    categories: dict[str, CategoryDelta]
    before_total: float
    after_total: float

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 2),
            "before_total": round(self.before_total, 2),
            "after_total": round(self.after_total, 2),
            "categories": {
                cat: {"roto_delta": round(cd.roto_delta, 2)}
                for cat, cd in self.categories.items()
            },
        }


def score_swap(
    roto_before: dict[str, dict],
    roto_after: dict[str, dict],
    team_name: str,
) -> DeltaRotoResult:
    """Per-category deltaRoto from before/after ``score_roto`` outputs.

    Total is the change in team total EV roto points. Each category's
    ``roto_delta`` is the change in that category's EV points for the
    user's team. No discounts, no penalties — the EV already reflects
    projection uncertainty, defensive vulnerability, and boundary
    proximity via the sigmoid on pairwise win probabilities.
    """
    categories: dict[str, CategoryDelta] = {}
    for cat in ALL_CATEGORIES:
        rd = roto_after[team_name][f"{cat}_pts"] - roto_before[team_name][f"{cat}_pts"]
        categories[cat] = CategoryDelta(roto_delta=rd)

    return DeltaRotoResult(
        total=roto_after[team_name]["total"] - roto_before[team_name]["total"],
        categories=categories,
        before_total=roto_before[team_name]["total"],
        after_total=roto_after[team_name]["total"],
    )


def compute_delta_roto(
    drop_name: str,
    add_player: "Player",
    user_roster: "list[Player]",
    projected_standings: list[dict],
    team_name: str,
    *,
    team_sds: dict[str, dict[str, float]] | None = None,
) -> DeltaRotoResult:
    """Compute deltaRoto for dropping one player and adding another.

    When ``team_sds`` is provided, ``score_roto`` uses pairwise Gaussian
    win-probabilities so a swap's impact reflects projection
    uncertainty. ``None`` preserves exact-rank semantics.

    Args:
        drop_name: roster player to drop.
        add_player: Player to add.
        user_roster: current roster (used to resolve the dropped player's ROS).
        projected_standings: end-of-season stats for all teams.
        team_name: user's team name.
        team_sds: optional ``{team: {cat: sd}}`` for EV scoring.

    Raises:
        ValueError: if drop_name is not found on the roster.
    """
    from fantasy_baseball.scoring import score_roto
    from fantasy_baseball.trades.evaluate import (
        apply_swap_delta, find_player_by_name, player_rest_of_season_stats,
    )

    dropped = find_player_by_name(drop_name, user_roster)
    if dropped is None:
        raise ValueError(f"Player '{drop_name}' not found on roster")

    loses_ros = player_rest_of_season_stats(dropped)
    gains_ros = player_rest_of_season_stats(add_player)

    all_before = {t["name"]: dict(t["stats"]) for t in projected_standings}
    all_after = dict(all_before)
    all_after[team_name] = apply_swap_delta(
        all_before[team_name], loses_ros, gains_ros,
    )

    roto_before = score_roto(all_before, team_sds=team_sds)
    roto_after = score_roto(all_after, team_sds=team_sds)

    return score_swap(roto_before, roto_after, team_name)
```

This delete the `compute_defense_comfort` function, the `FRAGILE_THRESHOLD`/`EROSION_WEIGHT`/`TIE_FLOOR`/`EROSION_CAP` constants, the `fragile_threshold`/`erosion_weight`/`tie_floor`/`erosion_cap` kwargs, the `tie_band_sd_factor` kwarg, and the `defense_before`/`defense_after`/`reason` fields on `CategoryDelta`. `INVERSE_STATS` import is also removed — no longer referenced.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_lineup/test_delta_roto.py -v 2>&1 | tail -20
```

Expected: all remaining tests pass. `test_end_to_end_swap`, `test_drop_not_found_raises`, the three new tests, and `_make_hitter`/`_make_pitcher` helper tests all pass.

If `test_end_to_end_swap` fails, it may reference `defense_before`/`defense_after` or the removed `score` assertion pattern — update it to use `roto_delta` directly.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/delta_roto.py tests/test_lineup/test_delta_roto.py
git commit -m "$(cat <<'EOF'
refactor(delta_roto): collapse score_swap to subtraction, delete comfort

score_roto now returns EV points, so score_swap reduces to after -
before. Deletes compute_defense_comfort (vulnerability is priced
into EV via P(me > team_below)), the erosion penalty, the
fragile/tie_floor/erosion_weight/erosion_cap knobs, and the
defense_before/defense_after fields on CategoryDelta. Frontend
grep confirms no consumer reads the removed JSON fields.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update `audit_roster` signature and callers

**Files:**
- Modify: `src/fantasy_baseball/lineup/roster_audit.py:151-178` (drop `tie_band_sd_factor` kwarg)
- Test: `tests/test_lineup/test_roster_audit.py` (update test class that used the old kwarg)

- [ ] **Step 1: Find and update the test that referenced `tie_band_sd_factor`**

```bash
grep -n "tie_band_sd_factor\|tie_band" tests/test_lineup/test_roster_audit.py
```

Expected: two references inside `test_team_sds_kwarg_suppresses_within_band_rank_flips` (lines around 298 and 308).

Rename the test and update it to assert EV-continuity behavior. Replace the test body with:

```python
def test_team_sds_produces_fractional_delta_within_uncertainty(
    self, leverage, roster_slots
):
    """End-to-end: audit_roster's top-candidate deltaRoto reflects EV,
    not discrete rank flips, when team_sds is provided."""
    # (Reuse the existing fixture builders from above in this test file.)
    projected_standings = [
        {"name": "User", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 100, "AVG": 0,
                                    "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
        {"name": "Rival", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 99, "AVG": 0,
                                     "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
    ] + [
        {"name": f"T{i}", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 10 + i, "AVG": 0,
                                     "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}}
        for i in range(10)
    ]
    team_sds = {t["name"]: {c: 0.0 for c in
                            ["R", "HR", "RBI", "SB", "AVG",
                             "W", "K", "SV", "ERA", "WHIP"]}
                for t in projected_standings}
    team_sds["User"]["SB"] = 10.0
    team_sds["Rival"]["SB"] = 10.0

    # Build a roster with a SB-heavy hitter; FAs include a SB-poor hitter.
    roster = [_make_hitter("Drop", r=0, hr=0, rbi=0, sb=20, h=0, ab=100)]
    fas = [_make_hitter("Add", r=0, hr=0, rbi=0, sb=10, h=0, ab=100)]

    # Without team_sds: rank flip gives the swap a -1.0 SB delta.
    entries_without = audit_roster(
        roster, fas, leverage, roster_slots,
        projected_standings=projected_standings, team_name="User",
        team_sds=None,
    )
    drop_entry_without = next(e for e in entries_without if e.player == "Drop")
    assert drop_entry_without.candidates[0]["delta_roto"]["categories"]["SB"]["roto_delta"] == pytest.approx(-1.0)

    # With wide SDs: EV produces a smaller magnitude delta.
    entries_with = audit_roster(
        roster, fas, leverage, roster_slots,
        projected_standings=projected_standings, team_name="User",
        team_sds=team_sds,
    )
    drop_entry_with = next(e for e in entries_with if e.player == "Drop")
    assert abs(drop_entry_with.candidates[0]["delta_roto"]["categories"]["SB"]["roto_delta"]) < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_lineup/test_roster_audit.py::TestAuditRosterTeamSDs -v 2>&1 | tail -15
```

Expected: TypeError because `audit_roster` still requires `tie_band_sd_factor`. (If the class name differs, use the actual class from the existing test file.)

- [ ] **Step 3: Remove `tie_band_sd_factor` from `audit_roster`**

Open `src/fantasy_baseball/lineup/roster_audit.py`. In the `audit_roster` signature (~line 151):

Change:
```python
def audit_roster(
    roster: list[Player],
    free_agents: list[Player],
    leverage: dict[str, float],
    roster_slots: dict[str, int],
    *,
    projected_standings: list[dict],
    team_name: str,
    team_sds: dict[str, dict[str, float]] | None = None,
    tie_band_sd_factor: float = 1.0,
) -> list[AuditEntry]:
```

To:
```python
def audit_roster(
    roster: list[Player],
    free_agents: list[Player],
    leverage: dict[str, float],
    roster_slots: dict[str, int],
    *,
    projected_standings: list[dict],
    team_name: str,
    team_sds: dict[str, dict[str, float]] | None = None,
) -> list[AuditEntry]:
```

Inside the function body, update the `compute_delta_roto` call (around line 271) to drop the `tie_band_sd_factor` kwarg:

Change:
```python
dr = compute_delta_roto(
    drop_name=player.name,
    add_player=fa,
    user_roster=roster,
    projected_standings=projected_standings,
    team_name=team_name,
    team_sds=team_sds,
    tie_band_sd_factor=tie_band_sd_factor,
)
```

To:
```python
dr = compute_delta_roto(
    drop_name=player.name,
    add_player=fa,
    user_roster=roster,
    projected_standings=projected_standings,
    team_name=team_name,
    team_sds=team_sds,
)
```

Update the docstring to remove references to `tie_band_sd_factor`.

- [ ] **Step 4: Run roster_audit tests**

```bash
pytest tests/test_lineup/test_roster_audit.py -v 2>&1 | tail -15
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/roster_audit.py tests/test_lineup/test_roster_audit.py
git commit -m "$(cat <<'EOF'
refactor(roster_audit): drop tie_band_sd_factor kwarg

score_roto's EV formulation no longer needs a tie-band factor.
Callers pass team_sds directly; raw σ scaled once at the refresh
site is the only knob we still expose.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update `compute_comparison_standings` and cache plumbing

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:682-769` (simplify compute_comparison_standings, drop comfort)
- Modify: `src/fantasy_baseball/web/season_data.py:~1180-1195` (re-introduce team_sds caching without tie_band_sd_factor)
- Modify: `src/fantasy_baseball/web/season_routes.py:~755-770` (drop tie_band_sd_factor read)
- Test: `tests/test_web/test_season_data.py` (update comparison-standings test)

Note: the cache-writing block at ~line 1180 currently doesn't exist on `main` (it was added in `ba82ab5`). This task re-introduces it. Confirm by grep:

```bash
grep -n "project_team_sds\|team_sds" src/fantasy_baseball/web/season_data.py
```

Expected: zero matches. If matches exist, the reset in Task 1 was incomplete — stop and investigate.

- [ ] **Step 1: Write the failing test for `compute_comparison_standings`**

Open `tests/test_web/test_season_data.py`. Delete any existing `TestComparisonTieBand` class. Add a new `TestComparisonEV` class:

```python
class TestComparisonEV:
    """compute_comparison_standings with EV-based scoring."""

    def test_team_sds_none_matches_rank_based(self):
        from fantasy_baseball.web.season_data import compute_comparison_standings
        # Minimal fixture: 2-team league, user drops a SB hitter for a
        # worse one → -10 SB, rank flip → ΔRoto_SB = -1.0.
        projected_standings = [
            {"name": "User", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 100, "AVG": 0,
                                        "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
            {"name": "Rival", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 99, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
        ]
        drop_hitter = _make_hitter("Drop", r=0, hr=0, rbi=0, sb=20, h=0, ab=100)
        add_hitter = _make_hitter("Add", r=0, hr=0, rbi=0, sb=10, h=0, ab=100)
        result = compute_comparison_standings(
            roster_player_name="Drop",
            other_player=add_hitter,
            user_roster=[drop_hitter],
            projected_standings=projected_standings,
            user_team_name="User",
            team_sds=None,
        )
        assert result["delta_roto"]["categories"]["SB"]["roto_delta"] == pytest.approx(-1.0)

    def test_team_sds_produces_fractional_delta_under_uncertainty(self):
        from fantasy_baseball.web.season_data import compute_comparison_standings
        projected_standings = [
            {"name": "User", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 100, "AVG": 0,
                                        "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
            {"name": "Rival", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 99, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
        ]
        team_sds = {"User": {"SB": 10.0, "R": 0, "HR": 0, "RBI": 0, "AVG": 0,
                             "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0},
                    "Rival": {"SB": 10.0, "R": 0, "HR": 0, "RBI": 0, "AVG": 0,
                              "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}}
        drop_hitter = _make_hitter("Drop", r=0, hr=0, rbi=0, sb=20, h=0, ab=100)
        add_hitter = _make_hitter("Add", r=0, hr=0, rbi=0, sb=10, h=0, ab=100)
        result = compute_comparison_standings(
            roster_player_name="Drop",
            other_player=add_hitter,
            user_roster=[drop_hitter],
            projected_standings=projected_standings,
            user_team_name="User",
            team_sds=team_sds,
        )
        assert abs(result["delta_roto"]["categories"]["SB"]["roto_delta"]) < 0.5
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_web/test_season_data.py::TestComparisonEV -v 2>&1 | tail -15
```

Expected: fails because `compute_comparison_standings` still takes `tie_band_sd_factor` and calls the deleted `compute_defense_comfort`.

- [ ] **Step 3: Simplify `compute_comparison_standings`**

Replace the function body in `src/fantasy_baseball/web/season_data.py` (around lines 682-769). New body:

```python
def compute_comparison_standings(
    roster_player_name: str,
    other_player: "Player",
    user_roster: "list[Player]",
    projected_standings: list[dict],
    user_team_name: str,
    *,
    roster_player_projection: "Player | None" = None,
    team_sds: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Compute before/after roto standings for a player swap.

    Uses cached ``projected_standings`` as the single source of truth
    and applies the swap via :func:`apply_swap_delta`, which guarantees
    the "before" totals match the standings page exactly.

    When ``roster_player_projection`` is provided, its ROS stats are
    used for the dropped player's contribution instead of the roster
    cache entry — keeps the delta consistent with the browse page.

    When ``team_sds`` is provided, ``score_roto`` uses the EV-based
    pairwise Gaussian formulation so the comparison matches what the
    roster audit shows for the same swap.

    Returns dict with before/after stats and roto, or ``{"error": ...}``.
    """
    from fantasy_baseball.scoring import score_roto
    from fantasy_baseball.trades.evaluate import (
        apply_swap_delta, find_player_by_name, player_rest_of_season_stats,
    )
    from fantasy_baseball.lineup.delta_roto import score_swap

    dropped = find_player_by_name(roster_player_name, user_roster)
    if dropped is None:
        return {"error": f"Player '{roster_player_name}' not found on roster"}

    loses_ros = player_rest_of_season_stats(roster_player_projection or dropped)
    gains_ros = player_rest_of_season_stats(other_player)

    all_stats_before = {t["name"]: dict(t["stats"]) for t in projected_standings}
    all_stats_after = dict(all_stats_before)
    all_stats_after[user_team_name] = apply_swap_delta(
        all_stats_before[user_team_name], loses_ros, gains_ros,
    )

    roto_before = score_roto(all_stats_before, team_sds=team_sds)
    roto_after = score_roto(all_stats_after, team_sds=team_sds)
    delta_roto = score_swap(roto_before, roto_after, user_team_name)

    return {
        "before": {"stats": all_stats_before, "roto": roto_before},
        "after": {"stats": all_stats_after, "roto": roto_after},
        "delta_roto": delta_roto.to_dict(),
        "categories": ALL_CATEGORIES,
        "user_team": user_team_name,
    }
```

Gone: `compute_defense_comfort` imports and calls, `get_sgp_denominators` import and call (if unused elsewhere in this function), `tie_band_sd_factor` parameter, `comfort_before`/`comfort_after` locals.

- [ ] **Step 4: Re-introduce team_sds caching in `run_full_refresh`**

Find the line in `src/fantasy_baseball/web/season_data.py` where the `projections` cache is written (~line 1187 on the old branch, somewhere analogous on main). Locate where `projected_standings` is computed (after `projected_standings = compute_projected_standings(...)`):

```bash
grep -n "projected_standings\|write_cache.*projections" src/fantasy_baseball/web/season_data.py | head -20
```

In the same area, add the following block immediately before the `write_cache("projections", ...)` call (and update the cache payload). Minimal diff:

Add imports near top of file:
```python
from fantasy_baseball.scoring import project_team_sds, score_roto  # score_roto already imported — only add project_team_sds
```

Before the cache write, compute team_sds:
```python
import math
from datetime import date

_season_start = date.fromisoformat(config.season_start)
_season_end = date.fromisoformat(config.season_end)
_total_days = (_season_end - _season_start).days
_remaining_days = max(0, (_season_end - local_today()).days)
fraction_remaining = (_remaining_days / _total_days) if _total_days > 0 else 0.0
_sd_scale = math.sqrt(fraction_remaining)

team_sds: dict[str, dict[str, float]] = {}
for _tname, _troster in all_team_rosters.items():
    _raw_sds = project_team_sds(_troster, displacement=True)
    team_sds[_tname] = {c: sd * _sd_scale for c, sd in _raw_sds.items()}
```

Then update the cache write to include team_sds and fraction_remaining:
```python
write_cache(
    "projections",
    {
        "projected_standings": projected_standings,
        "team_sds": team_sds,
        "fraction_remaining": fraction_remaining,
    },
    cache_dir,
)
```

Note: `tie_band_sd_factor` is NOT in the payload. It is gone.

Finally, pass `team_sds` to `audit_roster` wherever it's called downstream. Grep:
```bash
grep -n "audit_roster\|team_sds" src/fantasy_baseball/web/season_data.py
```

Update the `audit_roster(...)` call site to include `team_sds=team_sds` (and remove `tie_band_sd_factor` if present).

- [ ] **Step 5: Update `season_routes.py`**

In `src/fantasy_baseball/web/season_routes.py`, find the `compute_comparison_standings` call (~line 760). Remove the `tie_band_sd_factor=...` argument. Keep `team_sds=proj_cache.get("team_sds")`.

```bash
grep -n "tie_band_sd_factor" src/fantasy_baseball/web/season_routes.py
```

Replace every match. Final call should look like:

```python
result = compute_comparison_standings(
    roster_player_name=...,
    other_player=...,
    user_roster=...,
    projected_standings=...,
    user_team_name=...,
    team_sds=proj_cache.get("team_sds"),
)
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_web/test_season_data.py -v 2>&1 | tail -20
```

Expected: all pass, including the two new `TestComparisonEV` tests.

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py src/fantasy_baseball/web/season_routes.py tests/test_web/test_season_data.py
git commit -m "$(cat <<'EOF'
refactor(web): thread team_sds through comparison and audit

Re-introduces per-team SD cache alongside projected_standings,
scaled by sqrt(fraction_remaining). Both the roster audit page
and the player-compare endpoint read from the same cache so they
can no longer disagree on the sign of a swap. Removes the
tie_band_sd_factor cache field and argument — raw σ is the only
knob.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Remove config knobs

**Files:**
- Modify: `config/league.yaml` (delete the four tuning fields, if present)

- [ ] **Step 1: Check whether the knobs are present on the branch**

```bash
grep -n "fragile_threshold\|tie_floor\|erosion_weight\|erosion_cap\|tie_band_sd_factor\|delta_roto:" config/league.yaml
```

If zero matches: skip this task entirely, commit nothing.

If matches: open the file and delete the full `delta_roto:` section (or the individual keys). Example current content:

```yaml
delta_roto:
  fragile_threshold: 1.0
  erosion_weight: 0.3
  tie_floor: 0.5
  erosion_cap: 0.5
```

Delete all five lines.

- [ ] **Step 2: Verify nothing reads these keys**

```bash
grep -rn "fragile_threshold\|tie_floor\|erosion_weight\|erosion_cap" src/ scripts/
```

Expected: zero matches. If anything still reads them, back up and fix it before committing.

- [ ] **Step 3: Run config-loading tests**

```bash
pytest tests/test_config.py -v 2>&1 | tail -10
```

(If this file doesn't exist, substitute the right path — e.g. `tests/test_league_config.py`. Or skip this step if no config-loading tests exist.)

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add config/league.yaml
git commit -m "$(cat <<'EOF'
chore(config): drop obsolete delta_roto tuning knobs

fragile_threshold, tie_floor, erosion_weight, erosion_cap were
tuning parameters for the discount and erosion logic removed in
the EV refactor. No consumers left.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Full test suite + smoke checks

- [ ] **Step 1: Run the full test suite**

```bash
pytest -q 2>&1 | tail -10
```

Expected: all tests pass. Record the exact count for the final summary.

- [ ] **Step 2: Run the fast smoke test**

```bash
python scripts/smoke_test.py 2>&1 | tail -20
```

Expected: no exceptions. If `smoke_test.py` imports `compute_defense_comfort`, update it to drop the call.

- [ ] **Step 3: Search for any lingering references**

```bash
grep -rn "tie_band_sd_factor\|compute_defense_comfort\|fragile_threshold\|defense_before\|defense_after\|EROSION\|TIE_FLOOR\|FRAGILE_THRESHOLD" src/ tests/ scripts/
```

Expected: zero matches. If anything is left, it's dead code — remove it.

- [ ] **Step 4: Commit any cleanup from Step 3**

If cleanups were needed:
```bash
git add .
git commit -m "$(cat <<'EOF'
chore(cleanup): remove lingering references to obsolete heuristics

Tidy-up pass after grep for tie_band_sd_factor,
compute_defense_comfort, fragile_threshold, defense_before/after
surfaced <specific file(s)>. These were dead code paths not
caught in the per-module refactors.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

If no cleanup was needed: no commit, move on.

---

## Task 10: Live A/B diff vs main

Per `feedback_run_refresh_before_merge.md`: exercise the real refresh path locally before any merge.

- [ ] **Step 1: Run the full season-dashboard refresh on the branch**

```bash
python scripts/run_season_dashboard.py 2>&1 | tail -30
```

Expected: refresh completes with no errors. If Yahoo auth is stale, follow the re-auth flow documented in the previous attempt (see `docs/tie_band_branch_state.md` history for the `yahoo_oauth` re-consent steps).

If Yahoo is unreachable, stop here and resume when it's back — do not continue to a merge attempt without running refresh successfully.

- [ ] **Step 2: Snapshot the branch's audit output**

```bash
cp data/cache/roster_audit.json data/cache/roster_audit.branch.json
```

- [ ] **Step 3: Switch to main and run refresh**

```bash
git stash push -u -m "preserve branch data during A/B"
git checkout main
python scripts/run_season_dashboard.py 2>&1 | tail -30
```

- [ ] **Step 4: Diff the two outputs**

```bash
diff data/cache/roster_audit.branch.json data/cache/roster_audit.json | head -60
```

Expected qualitative differences:
- Within-uncertainty swaps (like Garcia ↔ Ward) show fractional ΔRoto on the branch, full ±1.0 on main.
- Candidate ordering changes for slots where previously-flipped picks lose their edge; genuine upgrades rise.
- No per-category `defense_before`/`defense_after` fields on the branch side (expected — they were deleted).

Record the diff summary for the PR description. If the branch looks obviously wrong (e.g. every candidate shows 0 deltaRoto, or negative numbers everywhere), stop and diagnose.

- [ ] **Step 5: Return to branch**

```bash
git checkout feat/delta-roto-tie-band
git stash pop
```

- [ ] **Step 6: Sanity-check the dashboard UI**

```bash
python scripts/run_season_dashboard.py --serve 2>&1 &
# or whatever serving entry point the repo uses — if a separate
# command is needed, check README or AGENTS.md.
```

Open the dashboard in a browser. Verify:
- Audit page renders; deltaRoto column shows fractional values.
- Player-compare modal shows the same sign and approximately the same magnitude for the same swap as the audit page.
- No console errors about missing `defense_before`/`defense_after` fields (there shouldn't be; grep was clean).

- [ ] **Step 7: No commit needed** (A/B is verification only)

---

## Task 11: Final state check

- [ ] **Step 1: Confirm branch history**

```bash
git log --oneline main..HEAD
```

Expected: 8-10 commits, all EV-related, no tie-band legacy. First commit is the plan doc. Last commit is the final cleanup / full test run.

- [ ] **Step 2: Re-run tests one last time**

```bash
pytest -q 2>&1 | tail -5
```

Expected: 100% pass, count comparable to main's baseline (small decrease acceptable since ~15 obsolete tests were deleted, some new ones added; net ~wash).

- [ ] **Step 3: Merge readiness check**

Confirm:
- [ ] All tasks above marked complete.
- [ ] Full test suite passes.
- [ ] Live A/B diff sanity-checked.
- [ ] Dashboard UI renders without errors.
- [ ] `feat/delta-roto-tie-band` branch name is misleading (no longer tie-band) but we're not renaming — note in the PR description.

Branch is ready for PR to `main`. Per memory `feedback_no_merge_without_asking.md`: **do not merge without explicit confirmation from Alden**. Offer to open the PR using `gh pr create` once the A/B review is complete and the user gives the go-ahead.

---

## Self-review notes

- Every deleted test was listed with an explicit reason in Task 5 (per `feedback_dont_modify_failing_tests.md` / CLAUDE.md "tests are the guardrail").
- All call sites of removed APIs grepped in Task 9 Step 3 (per `feedback_fix_all_callsites.md`).
- Refresh path exercised locally before any merge suggestion in Task 10 (per `feedback_run_refresh_before_merge.md`).
- Reset approved by user before planning.
- Each task ends with a single commit; frequent commits per the skill's guidance.
- Types used throughout: `DeltaRotoResult`, `CategoryDelta` (both from `delta_roto.py`), `dict[str, dict[str, float]]` for `team_sds`, `dict[str, dict[str, float]]` for `all_team_stats` (same shape as `score_roto` return). `_prob_beats` signature matches between definition (Task 2) and callers (Task 4).
