# Refresh Pipeline Testing & Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the 847-line `run_full_refresh` into a `RefreshRun` class with one method per current step block, extract 9 pure helpers into their domain modules with unit tests, and add one fixture-driven integration test that asserts cache-file shape and cross-step invariants.

**Architecture:** Three layers of safety. (1) Each pure helper gets a unit test that catches arithmetic and shape bugs. (2) One integration test runs the whole pipeline against mocked Yahoo + `fakeredis`, asserting cache files exist with the expected top-level keys plus invariants like "every player in moves exists in roster". (3) Existing source-string regression guards in `test_season_data.py` are updated to walk all `RefreshRun` methods. Order of work: extract helpers first (independent), then write the integration test against the *current* `run_full_refresh` (uses it as a regression guard), then wire helpers in, then refactor to `RefreshRun`, finally update guards.

**Tech Stack:** Python 3.11, pytest, pandas, `fakeredis` (already in conftest), `unittest.mock` for Yahoo patches. No new dependencies.

**Branch:** `refresh-pipeline-testing` (already created).

**Note on spec deviation:** The spec said `scoring/team_projection.py (new)`, but `src/fantasy_baseball/scoring.py` is a flat module, not a package. The two scoring helpers (`build_projected_standings`, `build_team_sds`) go directly in `scoring.py` — no new package directory. Tests go in the existing `tests/test_scoring.py`.

---

## Phase 1: Extract pure helpers

Each task is a self-contained TDD cycle: write test → fail → implement → pass → commit. None of these tasks touch `run_full_refresh` yet — they only add new functions and tests.

### Task 1: `compute_effective_date` in `time_utils.py`

**Files:**
- Modify: `src/fantasy_baseball/utils/time_utils.py` (add new function near `next_tuesday`)
- Test: `tests/test_utils/test_time_utils.py` (add new test class)

**Context:** Currently inline in `refresh_pipeline.py:163`:
```python
effective_date = next_tuesday(date.fromisoformat(end_date))
```
Encodes the rule "next lineup-lock Tuesday strictly after the scoring period's Sunday end_date". `next_tuesday` already exists in `time_utils` and returns the next Tuesday strictly after the given date.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_utils/test_time_utils.py`:
```python
from datetime import date
from fantasy_baseball.utils.time_utils import compute_effective_date


class TestComputeEffectiveDate:
    def test_sunday_end_date_returns_following_tuesday(self):
        # Yahoo scoring period ends on a Sunday; effective date is the
        # following Tuesday (lineup lock day).
        assert compute_effective_date("2026-04-19") == date(2026, 4, 21)

    def test_accepts_iso_string(self):
        assert compute_effective_date("2026-05-03") == date(2026, 5, 5)

    def test_tuesday_input_returns_following_tuesday(self):
        # next_tuesday is strict — a Tuesday input still moves forward.
        assert compute_effective_date("2026-04-21") == date(2026, 4, 28)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils/test_time_utils.py::TestComputeEffectiveDate -v`
Expected: FAIL with `ImportError: cannot import name 'compute_effective_date'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/fantasy_baseball/utils/time_utils.py` after `next_tuesday`:
```python
def compute_effective_date(end_date: str) -> date:
    """Return the next lineup-lock Tuesday strictly after ``end_date``.

    Yahoo's scoring period ends on a Sunday (``end_date``). The user's
    league locks lineups on Tuesday morning, so the effective date for
    fetching post-lock rosters is the next Tuesday strictly after that
    Sunday — ``end_date + 1`` would land on Monday, one day too early.
    """
    return next_tuesday(date.fromisoformat(end_date))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_utils/test_time_utils.py::TestComputeEffectiveDate -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/utils/time_utils.py tests/test_utils/test_time_utils.py
git commit -m "feat(time_utils): add compute_effective_date helper"
```

---

### Task 2: `compute_fraction_remaining` in `time_utils.py`

**Files:**
- Modify: `src/fantasy_baseball/utils/time_utils.py`
- Test: `tests/test_utils/test_time_utils.py`

**Context:** Currently inline in two places (`refresh_pipeline.py:405-409` and `697-702`). Both compute the same value — this extraction enables the bonus dedup later. Existing semantics: lower-bound clamp to 0 (today past season_end → 0.0); divide-by-zero protection (season_end == season_start → 0.0). No upper-bound clamp (today before season_start could yield > 1.0, but doesn't happen in practice — preserve existing behavior, don't add clamps that weren't there).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_utils/test_time_utils.py`:
```python
from fantasy_baseball.utils.time_utils import compute_fraction_remaining


class TestComputeFractionRemaining:
    def test_mid_season(self):
        season_start = date(2026, 4, 1)
        season_end = date(2026, 9, 30)  # 182 days total
        today = date(2026, 7, 1)        # 91 days remaining
        result = compute_fraction_remaining(season_start, season_end, today)
        assert result == pytest.approx(91 / 182)

    def test_today_after_season_end_returns_zero(self):
        result = compute_fraction_remaining(
            date(2026, 4, 1), date(2026, 9, 30), date(2026, 10, 15),
        )
        assert result == 0.0

    def test_today_equals_season_start_returns_one(self):
        result = compute_fraction_remaining(
            date(2026, 4, 1), date(2026, 9, 30), date(2026, 4, 1),
        )
        assert result == 1.0

    def test_zero_total_days_returns_zero(self):
        # Defensive: avoid divide-by-zero if season_end == season_start
        result = compute_fraction_remaining(
            date(2026, 4, 1), date(2026, 4, 1), date(2026, 4, 1),
        )
        assert result == 0.0
```

Add `import pytest` at the top of the test file if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils/test_time_utils.py::TestComputeFractionRemaining -v`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

Add to `src/fantasy_baseball/utils/time_utils.py`:
```python
def compute_fraction_remaining(
    season_start: date, season_end: date, today: date
) -> float:
    """Return the fraction of the regular season still ahead of ``today``.

    Used for SD scaling on projected standings (``sqrt`` damps variance
    as the season progresses) and for ROS Monte Carlo weighting.

    Returns 0.0 if the season has not started (season_end == season_start)
    or if ``today`` is on/after ``season_end``. Lower bound only — does
    not clamp the upper bound, matching existing behavior.
    """
    total_days = (season_end - season_start).days
    if total_days <= 0:
        return 0.0
    remaining_days = max(0, (season_end - today).days)
    return remaining_days / total_days
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_utils/test_time_utils.py::TestComputeFractionRemaining -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/utils/time_utils.py tests/test_utils/test_time_utils.py
git commit -m "feat(time_utils): add compute_fraction_remaining helper"
```

---

### Task 3: `build_projected_standings` and `build_team_sds` in `scoring.py`

**Files:**
- Modify: `src/fantasy_baseball/scoring.py` (add two new functions near `project_team_stats`)
- Test: `tests/test_scoring.py` (add new test class)

**Context:** Currently inline in `refresh_pipeline.py:392-426`. Two related but distinct helpers that wrap the existing `project_team_stats` and `project_team_sds` functions to produce per-team aggregated outputs. Keeping them in the same task because they share fixture setup.

`build_projected_standings(team_rosters)` returns the list of dicts written to `cache:projections.projected_standings`:
```python
[{"name": tname, "team_key": "", "rank": 0, "stats": <dict from CategoryStats.to_dict()>}, ...]
```

`build_team_sds(team_rosters, sd_scale)` returns the dict written to `cache:projections.team_sds`:
```python
{tname: {category: scaled_sd, ...}, ...}
```
where `scaled_sd = raw_sd * sd_scale`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_scoring.py` (look at existing tests in this file to match the Player-construction pattern; below assumes Players with `.rest_of_season` populated):
```python
from fantasy_baseball.scoring import build_projected_standings, build_team_sds


def _make_hitter_player(name, r=80, hr=20, rbi=70, sb=10, h=140, ab=500):
    """Minimal Player with rest_of_season stats — match how scoring.py
    consumes player.rest_of_season inside project_team_stats."""
    from fantasy_baseball.models.player import Player, PlayerType, RosterStats
    p = Player(
        name=name,
        positions=["OF"],
        player_type=PlayerType.HITTER,
        selected_position="OF",
        player_id=f"{name}::hitter",
    )
    p.rest_of_season = RosterStats(
        r=r, hr=hr, rbi=rbi, sb=sb, avg=h/ab if ab else 0,
        h=h, ab=ab, pa=ab,
    )
    return p


def _make_pitcher_player(name, w=10, k=180, sv=0, ip=180, era=3.5, whip=1.2):
    from fantasy_baseball.models.player import Player, PlayerType, RosterStats
    p = Player(
        name=name,
        positions=["SP"],
        player_type=PlayerType.PITCHER,
        selected_position="P",
        player_id=f"{name}::pitcher",
    )
    p.rest_of_season = RosterStats(
        w=w, k=k, sv=sv, era=era, whip=whip, ip=ip,
        er=era*ip/9, bb=int(whip*ip*0.3), h_allowed=int(whip*ip*0.7),
    )
    return p


class TestBuildProjectedStandings:
    def test_returns_one_entry_per_team(self):
        rosters = {
            "Team A": [_make_hitter_player("Player1")],
            "Team B": [_make_hitter_player("Player2")],
        }
        result = build_projected_standings(rosters)
        assert len(result) == 2
        team_names = {entry["name"] for entry in result}
        assert team_names == {"Team A", "Team B"}

    def test_each_entry_has_expected_keys(self):
        rosters = {"Team A": [_make_hitter_player("Player1")]}
        result = build_projected_standings(rosters)
        entry = result[0]
        assert set(entry.keys()) == {"name", "team_key", "rank", "stats"}
        assert entry["team_key"] == ""
        assert entry["rank"] == 0
        assert isinstance(entry["stats"], dict)

    def test_stats_dict_has_all_categories(self):
        rosters = {
            "Team A": [
                _make_hitter_player("H1"),
                _make_pitcher_player("P1"),
            ],
        }
        result = build_projected_standings(rosters)
        stats = result[0]["stats"]
        # CategoryStats.to_dict yields all 10 5x5 categories
        for cat in ("r", "hr", "rbi", "sb", "avg", "w", "k", "sv", "era", "whip"):
            assert cat in stats


class TestBuildTeamSDs:
    def test_returns_one_dict_per_team(self):
        rosters = {
            "Team A": [_make_hitter_player("P1")],
            "Team B": [_make_hitter_player("P2")],
        }
        result = build_team_sds(rosters, sd_scale=1.0)
        assert set(result.keys()) == {"Team A", "Team B"}

    def test_sd_scale_multiplies_each_value(self):
        rosters = {"Team A": [_make_hitter_player("P1")]}
        unscaled = build_team_sds(rosters, sd_scale=1.0)
        scaled = build_team_sds(rosters, sd_scale=0.5)
        for cat, sd in unscaled["Team A"].items():
            assert scaled["Team A"][cat] == pytest.approx(sd * 0.5)

    def test_sd_scale_zero_yields_zero_sds(self):
        rosters = {"Team A": [_make_hitter_player("P1")]}
        result = build_team_sds(rosters, sd_scale=0.0)
        for sd in result["Team A"].values():
            assert sd == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py::TestBuildProjectedStandings tests/test_scoring.py::TestBuildTeamSDs -v`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

Add to `src/fantasy_baseball/scoring.py` after `project_team_sds`:
```python
def build_projected_standings(
    team_rosters: dict[str, list],
) -> list[dict]:
    """Build the projected_standings list written to the projections cache.

    Each entry has keys ``name``, ``team_key`` (always empty — the
    consumer fills it from standings if needed), ``rank`` (always 0 —
    ranking is computed downstream), and ``stats`` (the team's projected
    category totals from ``project_team_stats`` with ``displacement=True``).
    """
    return [
        {
            "name": tname,
            "team_key": "",
            "rank": 0,
            "stats": project_team_stats(roster, displacement=True).to_dict(),
        }
        for tname, roster in team_rosters.items()
    ]


def build_team_sds(
    team_rosters: dict[str, list], sd_scale: float,
) -> dict[str, dict[str, float]]:
    """Build the team_sds dict written to the projections cache.

    Each team's per-category SDs from ``project_team_sds`` are scaled by
    ``sd_scale`` (typically ``sqrt(fraction_remaining)`` so variance
    damps as the season progresses).
    """
    return {
        tname: {
            cat: sd * sd_scale
            for cat, sd in project_team_sds(roster, displacement=True).items()
        }
        for tname, roster in team_rosters.items()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring.py::TestBuildProjectedStandings tests/test_scoring.py::TestBuildTeamSDs -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/scoring.py tests/test_scoring.py
git commit -m "feat(scoring): add build_projected_standings and build_team_sds helpers"
```

---

### Task 4: `attach_pace_to_roster` in `analysis/pace.py`

**Files:**
- Modify: `src/fantasy_baseball/analysis/pace.py` (add new function alongside `compute_player_pace`)
- Test: `tests/test_analysis/test_pace.py` (add new test class)

**Context:** Currently inline in `refresh_pipeline.py:482-502`. Per-player loop that picks the right projection-key set based on player_type, builds the projected/actual stat bags, and calls `compute_player_pace`. Mutates each player by setting `player.pace`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_analysis/test_pace.py` (look at the existing tests for fixtures to reuse):
```python
from fantasy_baseball.analysis.pace import attach_pace_to_roster
from fantasy_baseball.models.player import Player, PlayerType, RosterStats


def _hitter_with_ros(name="Soto", r=80, hr=25, rbi=80, sb=5, avg=0.290, ab=500):
    p = Player(
        name=name, positions=["OF"], player_type=PlayerType.HITTER,
        selected_position="OF", player_id=f"{name}::hitter",
    )
    p.rest_of_season = RosterStats(
        r=r, hr=hr, rbi=rbi, sb=sb, avg=avg, h=int(avg*ab), ab=ab, pa=ab,
    )
    return p


def _pitcher_with_ros(name="Cole", w=12, k=200, sv=0, era=3.0, whip=1.10, ip=180):
    p = Player(
        name=name, positions=["SP"], player_type=PlayerType.PITCHER,
        selected_position="P", player_id=f"{name}::pitcher",
    )
    p.rest_of_season = RosterStats(
        w=w, k=k, sv=sv, era=era, whip=whip, ip=ip,
        er=era*ip/9, bb=int(whip*ip*0.3), h_allowed=int(whip*ip*0.7),
    )
    return p


class TestAttachPaceToRoster:
    def test_hitter_pace_uses_hitter_logs(self):
        h = _hitter_with_ros()
        hitter_logs = {"soto": {"r": 40, "hr": 15, "rbi": 50, "sb": 2, "avg": 0.310}}
        attach_pace_to_roster(
            [h], hitter_logs, pitcher_logs={},
            preseason_lookup={}, sgp_denoms={"r": 25, "hr": 8, "rbi": 25, "sb": 8, "avg": 0.012, "w": 5, "k": 80, "sv": 12, "era": 0.30, "whip": 0.05},
        )
        assert h.pace is not None

    def test_pitcher_pace_uses_pitcher_logs(self):
        p = _pitcher_with_ros()
        pitcher_logs = {"cole": {"w": 6, "k": 110, "sv": 0, "era": 2.80, "whip": 1.05}}
        attach_pace_to_roster(
            [p], hitter_logs={}, pitcher_logs=pitcher_logs,
            preseason_lookup={}, sgp_denoms={"r": 25, "hr": 8, "rbi": 25, "sb": 8, "avg": 0.012, "w": 5, "k": 80, "sv": 12, "era": 0.30, "whip": 0.05},
        )
        assert p.pace is not None

    def test_missing_actuals_uses_empty_dict(self):
        # Player with no game logs (e.g. just-called-up rookie) should
        # still get a pace attached — compute_player_pace handles empty.
        h = _hitter_with_ros(name="Newbie")
        attach_pace_to_roster(
            [h], hitter_logs={}, pitcher_logs={},
            preseason_lookup={}, sgp_denoms={"r": 25, "hr": 8, "rbi": 25, "sb": 8, "avg": 0.012, "w": 5, "k": 80, "sv": 12, "era": 0.30, "whip": 0.05},
        )
        assert h.pace is not None

    def test_preseason_projections_used_when_available(self):
        # If preseason_lookup has the player, projected stats come from
        # there rather than zero-filled.
        h = _hitter_with_ros(name="Soto")
        pre = _hitter_with_ros(name="Soto", r=100, hr=35)
        attach_pace_to_roster(
            [h], hitter_logs={"soto": {"r": 50, "hr": 18}},
            pitcher_logs={},
            preseason_lookup={"soto": pre},
            sgp_denoms={"r": 25, "hr": 8, "rbi": 25, "sb": 8, "avg": 0.012, "w": 5, "k": 80, "sv": 12, "era": 0.30, "whip": 0.05},
        )
        assert h.pace is not None

    def test_player_without_rest_of_season_still_processed(self):
        # Player.rest_of_season can be None for unmatched FAs etc.
        h = Player(
            name="NoProj", positions=["OF"], player_type=PlayerType.HITTER,
            selected_position="OF", player_id="NoProj::hitter",
        )
        # h.rest_of_season is None
        attach_pace_to_roster(
            [h], hitter_logs={}, pitcher_logs={},
            preseason_lookup={},
            sgp_denoms={"r": 25, "hr": 8, "rbi": 25, "sb": 8, "avg": 0.012, "w": 5, "k": 80, "sv": 12, "era": 0.30, "whip": 0.05},
        )
        # Should not raise; pace is set (compute_player_pace handles None)
        assert h.pace is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_pace.py::TestAttachPaceToRoster -v`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

Add to `src/fantasy_baseball/analysis/pace.py`:
```python
def attach_pace_to_roster(
    players: list,
    hitter_logs: dict,
    pitcher_logs: dict,
    preseason_lookup: dict,
    sgp_denoms: dict,
) -> None:
    """Attach a ``pace`` attribute to every player in ``players``.

    For each player, picks the right log dict (hitter_logs vs pitcher_logs)
    by player_type, builds projected stats from ``preseason_lookup`` (zero-
    filled if no preseason entry), pulls current ROS stats from the player
    if present, and calls ``compute_player_pace``. Mutates each player.
    """
    from fantasy_baseball.models.player import PlayerType
    from fantasy_baseball.utils.constants import HITTER_PROJ_KEYS, PITCHER_PROJ_KEYS
    from fantasy_baseball.utils.name_utils import normalize_name

    for player in players:
        norm = normalize_name(player.name)
        if player.player_type == PlayerType.HITTER:
            actuals = hitter_logs.get(norm, {})
            ros_keys = ["r", "hr", "rbi", "sb", "avg"]
            proj_keys = HITTER_PROJ_KEYS
        else:
            actuals = pitcher_logs.get(norm, {})
            ros_keys = ["w", "k", "sv", "era", "whip"]
            proj_keys = PITCHER_PROJ_KEYS
        pre_player = preseason_lookup.get(norm)
        if pre_player and pre_player.rest_of_season:
            projected = {k: getattr(pre_player.rest_of_season, k, 0) for k in proj_keys}
        else:
            projected = {k: 0 for k in proj_keys}
        ros_dict = (
            {k: getattr(player.rest_of_season, k, 0) for k in ros_keys}
            if player.rest_of_season else None
        )
        player.pace = compute_player_pace(
            actuals, projected, player.player_type,
            rest_of_season_stats=ros_dict, sgp_denoms=sgp_denoms,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_pace.py::TestAttachPaceToRoster -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/pace.py tests/test_analysis/test_pace.py
git commit -m "feat(pace): add attach_pace_to_roster helper"
```

---

### Task 5: `build_rankings_lookup` in `sgp/rankings.py`

**Files:**
- Modify: `src/fantasy_baseball/sgp/rankings.py`
- Test: `tests/test_sgp/test_rankings.py`

**Context:** Currently inline in `refresh_pipeline.py:526-533`. Three-way merge of ranking dicts keyed by `name::player_type`. Each value in the output is `{rest_of_season: ..., preseason: ..., current: ...}` with `None` where missing.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sgp/test_rankings.py`:
```python
from fantasy_baseball.sgp.rankings import build_rankings_lookup


class TestBuildRankingsLookup:
    def test_player_in_all_three(self):
        ros = {"Soto::hitter": {"overall": 5}}
        pre = {"Soto::hitter": {"overall": 3}}
        cur = {"Soto::hitter": {"overall": 7}}
        result = build_rankings_lookup(ros, pre, cur)
        assert result["Soto::hitter"] == {
            "rest_of_season": {"overall": 5},
            "preseason": {"overall": 3},
            "current": {"overall": 7},
        }

    def test_player_only_in_ros_has_none_for_others(self):
        result = build_rankings_lookup(
            ros={"Newbie::hitter": {"overall": 100}},
            preseason={},
            current={},
        )
        assert result["Newbie::hitter"] == {
            "rest_of_season": {"overall": 100},
            "preseason": None,
            "current": None,
        }

    def test_player_only_in_preseason_has_none_for_others(self):
        # E.g. preseason hype guy who didn't end up on the ROS list
        result = build_rankings_lookup(
            ros={}, preseason={"Bust::hitter": {"overall": 50}}, current={},
        )
        assert result["Bust::hitter"] == {
            "rest_of_season": None,
            "preseason": {"overall": 50},
            "current": None,
        }

    def test_player_only_in_current_has_none_for_others(self):
        # Surprise breakout with no projection on either side
        result = build_rankings_lookup(
            ros={}, preseason={}, current={"Surprise::hitter": {"overall": 25}},
        )
        assert result["Surprise::hitter"] == {
            "rest_of_season": None,
            "preseason": None,
            "current": {"overall": 25},
        }

    def test_union_includes_keys_from_all_three(self):
        result = build_rankings_lookup(
            ros={"A::hitter": {"o": 1}},
            preseason={"B::hitter": {"o": 2}},
            current={"C::hitter": {"o": 3}},
        )
        assert set(result.keys()) == {"A::hitter", "B::hitter", "C::hitter"}

    def test_empty_inputs_yield_empty_dict(self):
        assert build_rankings_lookup({}, {}, {}) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sgp/test_rankings.py::TestBuildRankingsLookup -v`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

Add to `src/fantasy_baseball/sgp/rankings.py`:
```python
def build_rankings_lookup(
    ros: dict, preseason: dict, current: dict,
) -> dict[str, dict]:
    """Three-way merge of player ranking dicts keyed by ``name::player_type``.

    The output is a dict mapping each player key to a dict with three
    keys (``rest_of_season``, ``preseason``, ``current``); missing
    entries are ``None``. The union of keys from all three inputs is
    represented.
    """
    all_keys = set(ros) | set(preseason) | set(current)
    return {
        key: {
            "rest_of_season": ros.get(key),
            "preseason": preseason.get(key),
            "current": current.get(key),
        }
        for key in all_keys
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sgp/test_rankings.py::TestBuildRankingsLookup -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/sgp/rankings.py tests/test_sgp/test_rankings.py
git commit -m "feat(sgp): add build_rankings_lookup helper"
```

---

### Task 6: `merge_matched_and_raw_roster` in new `web/refresh_steps.py`

**Files:**
- Create: `src/fantasy_baseball/web/refresh_steps.py`
- Create: `tests/test_web/test_refresh_steps.py`

**Context:** Currently inline in `refresh_pipeline.py:449-469`. Takes the matched-to-projections player list plus the raw roster, and produces a unified list with: (a) every matched player gets `player.preseason` set if a preseason entry exists, (b) any raw player not in the matched set is added as a Player built from the raw dict (player_type inferred from positions). Mutates matched players. Returns the combined list.

- [ ] **Step 1: Write the failing test**

Create `tests/test_web/test_refresh_steps.py`:
```python
"""Tests for refresh_steps.py — pure helpers extracted from
run_full_refresh that are specific to the refresh orchestration
(not general enough to push into a domain module)."""
import pytest
from fantasy_baseball.models.player import Player, PlayerType, RosterStats
from fantasy_baseball.web.refresh_steps import (
    build_positions_map,
    compute_lineup_moves,
    merge_matched_and_raw_roster,
)


def _player(name, player_type=PlayerType.HITTER, positions=None,
            selected_position=None, wsgp=0.0, ros=None):
    positions = positions or (["OF"] if player_type == PlayerType.HITTER else ["SP"])
    selected_position = selected_position or positions[0]
    p = Player(
        name=name, positions=positions, player_type=player_type,
        selected_position=selected_position,
        player_id=f"{name}::{player_type.value}",
    )
    p.rest_of_season = ros
    p.wsgp = wsgp
    return p


class TestMergeMatchedAndRawRoster:
    def test_matched_players_get_preseason_attached(self):
        soto = _player("Soto", ros=RosterStats(r=80))
        soto_pre = _player("Soto", ros=RosterStats(r=100, hr=35))
        result = merge_matched_and_raw_roster(
            matched=[soto],
            roster_raw=[{"name": "Soto", "positions": ["OF"], "selected_position": "OF", "player_id": "1", "status": ""}],
            preseason_lookup={"soto": soto_pre},
        )
        assert len(result) == 1
        assert result[0].preseason is soto_pre.rest_of_season

    def test_matched_player_without_preseason_entry(self):
        soto = _player("Soto", ros=RosterStats(r=80))
        result = merge_matched_and_raw_roster(
            matched=[soto], roster_raw=[
                {"name": "Soto", "positions": ["OF"], "selected_position": "OF", "player_id": "1", "status": ""}
            ],
            preseason_lookup={},  # no preseason match
        )
        assert len(result) == 1
        # No preseason attached (attribute not set or stays as default)

    def test_unmatched_raw_player_added_as_hitter(self):
        # Raw player not in matched list — should be added with
        # player_type inferred from positions (OF → HITTER).
        result = merge_matched_and_raw_roster(
            matched=[],
            roster_raw=[{"name": "Newbie", "positions": ["OF"], "selected_position": "OF", "player_id": "99", "status": ""}],
            preseason_lookup={},
        )
        assert len(result) == 1
        assert result[0].name == "Newbie"
        assert result[0].player_type == PlayerType.HITTER

    def test_unmatched_raw_player_added_as_pitcher(self):
        # SP positions → PITCHER
        result = merge_matched_and_raw_roster(
            matched=[],
            roster_raw=[{"name": "RookiePitcher", "positions": ["SP"], "selected_position": "P", "player_id": "100", "status": ""}],
            preseason_lookup={},
        )
        assert len(result) == 1
        assert result[0].player_type == PlayerType.PITCHER

    def test_matched_player_skipped_in_raw_iteration(self):
        # When a player is in BOTH matched and raw, only one entry should
        # appear in the result (the matched one).
        soto = _player("Soto")
        result = merge_matched_and_raw_roster(
            matched=[soto],
            roster_raw=[
                {"name": "Soto", "positions": ["OF"], "selected_position": "OF", "player_id": "1", "status": ""},
                {"name": "Newbie", "positions": ["OF"], "selected_position": "BN", "player_id": "99", "status": ""},
            ],
            preseason_lookup={},
        )
        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"Soto", "Newbie"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_refresh_steps.py::TestMergeMatchedAndRawRoster -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Write minimal implementation**

Create `src/fantasy_baseball/web/refresh_steps.py`:
```python
"""Pure helpers extracted from run_full_refresh.

These pieces are refresh-specific orchestration glue — they don't
belong in a domain module like ``scoring`` or ``analysis.pace``
because they only exist to compose those domains' outputs into the
shape the cache files need.
"""
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import PITCHER_POSITIONS


def merge_matched_and_raw_roster(
    matched: list[Player],
    roster_raw: list[dict],
    preseason_lookup: dict[str, Player],
) -> list[Player]:
    """Combine projection-matched players with any unmatched raw entries.

    For every matched player, attaches ``player.preseason`` from the
    corresponding entry in ``preseason_lookup`` (keyed by normalized
    name) if one exists. Then appends a Player built from each raw
    roster entry that wasn't matched, inferring ``player_type`` from
    positions (any pitcher position → PITCHER, otherwise HITTER).

    Mutates each matched Player. Returns the combined list.
    """
    matched_names = set()
    out: list[Player] = []
    for player in matched:
        norm = normalize_name(player.name)
        matched_names.add(norm)
        pre_entry = preseason_lookup.get(norm)
        if pre_entry and pre_entry.rest_of_season:
            player.preseason = pre_entry.rest_of_season
        out.append(player)

    for raw_player in roster_raw:
        if normalize_name(raw_player["name"]) not in matched_names:
            inferred_type = (
                PlayerType.PITCHER
                if set(raw_player.get("positions", [])) & PITCHER_POSITIONS
                else PlayerType.HITTER
            )
            out.append(Player.from_dict({**raw_player, "player_type": inferred_type}))

    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_web/test_refresh_steps.py::TestMergeMatchedAndRawRoster -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/refresh_steps.py tests/test_web/test_refresh_steps.py
git commit -m "feat(web): add merge_matched_and_raw_roster helper in refresh_steps"
```

---

### Task 7: `compute_lineup_moves` in `web/refresh_steps.py`

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_steps.py`
- Modify: `tests/test_web/test_refresh_steps.py`

**Context:** Currently inline in `refresh_pipeline.py:578-597`. Compares the optimizer's output (`{slot_key: player_name}`) to each player's currently selected position. Emits a START move when the player is moving from a bench-like slot to a non-bench-like slot (or vice versa). Bench-like slots are `BN`, `IL`, `DL`. Slot keys may have suffixes like `OF_1`, `OF_2` — only the prefix before `_` matters.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_web/test_refresh_steps.py`:
```python
class TestComputeLineupMoves:
    def test_bench_to_starter_emits_start_move(self):
        # Player on BN; optimizer wants them at OF
        p = _player("Soto", selected_position="BN", wsgp=12.5)
        optimal = {"OF_1": "Soto"}
        moves = compute_lineup_moves(optimal, [p])
        assert len(moves) == 1
        assert moves[0]["action"] == "START"
        assert moves[0]["player"] == "Soto"
        assert moves[0]["slot"] == "OF"
        assert "12.5" in moves[0]["reason"]

    def test_starter_to_starter_emits_no_move(self):
        # Player already at OF; optimizer keeps them at OF — no move
        p = _player("Soto", selected_position="OF", wsgp=12.5)
        optimal = {"OF_1": "Soto"}
        assert compute_lineup_moves(optimal, [p]) == []

    def test_il_to_starter_emits_start_move(self):
        # IL counts as bench-like
        p = _player("Soto", selected_position="IL", wsgp=10.0)
        optimal = {"OF_1": "Soto"}
        moves = compute_lineup_moves(optimal, [p])
        assert len(moves) == 1
        assert moves[0]["action"] == "START"

    def test_starter_to_bench_emits_start_move(self):
        # Optimizer demoting a starter to bench also counts
        # (loop only iterates optimal slots, so this case fires when
        # the same player appears in optimal under a BN_x slot).
        p = _player("Soto", selected_position="OF", wsgp=12.5)
        optimal = {"BN_1": "Soto"}
        moves = compute_lineup_moves(optimal, [p])
        assert len(moves) == 1
        assert moves[0]["slot"] == "BN"

    def test_player_not_on_roster_skipped(self):
        # Defensive: optimizer references a name not in roster_players
        p = _player("Other", selected_position="OF")
        optimal = {"OF_1": "Ghost"}
        assert compute_lineup_moves(optimal, [p]) == []

    def test_player_with_no_selected_position_treated_as_bench(self):
        # selected_position is None → falls back to "BN"
        p = _player("Soto", selected_position=None, wsgp=12.5)
        # With no current slot and optimizer wanting OF, it's bench→starter
        optimal = {"OF_1": "Soto"}
        moves = compute_lineup_moves(optimal, [p])
        assert len(moves) == 1

    def test_slot_suffix_stripped(self):
        # OF_1 vs OF_2 — both should be treated as OF
        p = _player("Soto", selected_position="OF", wsgp=12.5)
        optimal = {"OF_2": "Soto"}
        # Current is OF, target is OF (after stripping _2) → no move
        assert compute_lineup_moves(optimal, [p]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_refresh_steps.py::TestComputeLineupMoves -v`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

Append to `src/fantasy_baseball/web/refresh_steps.py`:
```python
def compute_lineup_moves(
    optimal_hitters: dict[str, str],
    roster_players: list[Player],
) -> list[dict]:
    """Compare optimizer output to current slots; emit START moves.

    Only emits a move when the player is crossing the bench/active
    boundary. Bench-like slots: BN, IL, DL. Slot keys may have suffixes
    like ``OF_1`` — only the prefix before ``_`` matters for comparison.
    """
    bench_slots = {"BN", "IL", "DL"}
    moves: list[dict] = []
    for slot, player_name in optimal_hitters.items():
        for player in roster_players:
            if player.name != player_name:
                continue
            current_slot = player.selected_position or "BN"
            base_slot = slot.split("_")[0]
            if current_slot != base_slot and (
                current_slot in bench_slots or base_slot in bench_slots
            ):
                moves.append({
                    "action": "START",
                    "player": player_name,
                    "slot": base_slot,
                    "reason": f"wSGP: {player.wsgp:.1f}",
                })
            break
    return moves
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_web/test_refresh_steps.py::TestComputeLineupMoves -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/refresh_steps.py tests/test_web/test_refresh_steps.py
git commit -m "feat(web): add compute_lineup_moves helper"
```

---

### Task 8: `build_positions_map` in `web/refresh_steps.py`

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_steps.py`
- Modify: `tests/test_web/test_refresh_steps.py`

**Context:** Currently inline in `refresh_pipeline.py:636-648`. Builds a single normalized-name → positions-list map drawing from three sources: the user's roster, all opponent rosters, and free agents. Free agents with empty positions are skipped (their positions field can be empty when Yahoo doesn't return position data). When the same player appears in multiple sources, later sources overwrite earlier ones (FAs > opponents > user roster) — this matches the existing iteration order.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_web/test_refresh_steps.py`:
```python
class TestBuildPositionsMap:
    def test_includes_roster_players(self):
        roster = [_player("Soto", positions=["OF"])]
        result = build_positions_map(roster, opp_rosters={}, fa_players=[])
        assert result["soto"] == ["OF"]

    def test_includes_opponent_players(self):
        opp = {"OtherTeam": [_player("Trout", positions=["OF", "Util"])]}
        result = build_positions_map([], opp_rosters=opp, fa_players=[])
        assert result["trout"] == ["OF", "Util"]

    def test_includes_free_agents(self):
        fas = [_player("Acuna", positions=["OF"])]
        result = build_positions_map([], opp_rosters={}, fa_players=fas)
        assert result["acuna"] == ["OF"]

    def test_free_agent_with_empty_positions_skipped(self):
        # FAs with no positions data shouldn't pollute the map
        fa = _player("Mystery", positions=[])
        result = build_positions_map([], opp_rosters={}, fa_players=[fa])
        assert "mystery" not in result

    def test_normalizes_keys(self):
        # Accents and case should be normalized
        roster = [_player("José Ramírez", positions=["3B"])]
        result = build_positions_map(roster, opp_rosters={}, fa_players=[])
        # normalize_name strips accents and lowercases
        assert "jose ramirez" in result

    def test_combines_all_three_sources(self):
        roster = [_player("A", positions=["OF"])]
        opp = {"T2": [_player("B", positions=["1B"])]}
        fas = [_player("C", positions=["SS"])]
        result = build_positions_map(roster, opp, fas)
        assert set(result.keys()) == {"a", "b", "c"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_refresh_steps.py::TestBuildPositionsMap -v`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

Append to `src/fantasy_baseball/web/refresh_steps.py`:
```python
def build_positions_map(
    roster_players: list[Player],
    opp_rosters: dict[str, list[Player]],
    fa_players: list[Player],
) -> dict[str, list[str]]:
    """Build a normalized-name → positions-list map from three sources.

    Iteration order is roster → opponents → FAs, so a player appearing
    in multiple sources gets the FA positions if present, then opponent,
    then user roster. FAs with empty positions are skipped (Yahoo
    sometimes returns no position data for them).
    """
    out: dict[str, list[str]] = {}
    for p in roster_players:
        out[normalize_name(p.name)] = list(p.positions)
    for opp_roster in opp_rosters.values():
        for p in opp_roster:
            out[normalize_name(p.name)] = list(p.positions)
    for fa in fa_players:
        if fa.positions:
            out[normalize_name(fa.name)] = list(fa.positions)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_web/test_refresh_steps.py::TestBuildPositionsMap -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run full refresh_steps test file to ensure all 18 tests pass**

Run: `pytest tests/test_web/test_refresh_steps.py -v`
Expected: PASS (18 tests total: 5 merge + 7 moves + 6 positions)

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/refresh_steps.py tests/test_web/test_refresh_steps.py
git commit -m "feat(web): add build_positions_map helper"
```

---

## Phase 2: Build the integration test (against current run_full_refresh)

This phase writes the integration test *before* refactoring `run_full_refresh`. The test then doubles as a regression guard for Phases 3 and 4. Splitting fixture work, Yahoo mocks, and assertions into separate tasks keeps each commit reviewable.

### Task 9: Build minimal League fixture data

**Files:**
- Create: `tests/test_web/_refresh_fixture.py` (helper module — not a test file, leading underscore avoids pytest collection)

**Context:** The integration test needs realistic-shaped but tiny test data: 12 teams, ~10 players each, plus standings, projections, game logs, schedule, and team batting stats. Putting the fixture builders in a helper module keeps the eventual test file readable.

- [ ] **Step 1: Create fixture helper module**

Create `tests/test_web/_refresh_fixture.py`:
```python
"""Minimal-but-realistic fixture data for the run_full_refresh
integration test. Returns plain dicts/lists matching the shapes
produced by Yahoo (post-parse) and the projection CSVs (post-blend).
"""
from datetime import date
from typing import Any


TEAM_NAMES = [f"Team {i:02d}" for i in range(1, 13)]  # 12 teams
USER_TEAM_NAME = "Team 01"


def _hitter_proj_row(name: str, fg_id: str, **stats) -> dict:
    """One row of a blended hitter projection table."""
    base = {
        "name": name, "fg_id": fg_id, "team": "TBD",
        "positions": "OF", "ab": 500, "pa": 580,
        "r": 80, "hr": 25, "rbi": 80, "sb": 8, "h": 145, "avg": 0.290,
    }
    base.update(stats)
    return base


def _pitcher_proj_row(name: str, fg_id: str, **stats) -> dict:
    base = {
        "name": name, "fg_id": fg_id, "team": "TBD",
        "positions": "SP", "ip": 180.0,
        "w": 12, "k": 200, "sv": 0, "era": 3.50, "whip": 1.15,
        "er": 70, "bb": 50, "h_allowed": 160,
    }
    base.update(stats)
    return base


def hitter_projections() -> list[dict]:
    """120 hitters — enough to cover 12 rosters x ~6 hitters + spares."""
    rows = []
    for i in range(120):
        rows.append(_hitter_proj_row(
            name=f"Hitter{i:03d}", fg_id=f"fg_h_{i:03d}",
            r=70 + (i % 30), hr=15 + (i % 20), rbi=60 + (i % 30),
            sb=2 + (i % 15), avg=0.250 + (i % 50) / 1000,
        ))
    return rows


def pitcher_projections() -> list[dict]:
    """80 pitchers — covers 12 rosters x ~5 pitchers + spares + closers."""
    rows = []
    for i in range(80):
        is_closer = i < 12  # First 12 are closers
        rows.append(_pitcher_proj_row(
            name=f"Pitcher{i:03d}", fg_id=f"fg_p_{i:03d}",
            positions="RP" if is_closer else "SP",
            ip=70.0 if is_closer else 180.0,
            sv=25 if is_closer else 0,
            w=4 if is_closer else 10 + (i % 8),
            k=80 if is_closer else 180 + (i % 40),
            era=3.0 + (i % 10) / 10, whip=1.10 + (i % 10) / 100,
        ))
    return rows


def standings() -> list[dict]:
    """12 teams with all 10 categories populated."""
    out = []
    for i, name in enumerate(TEAM_NAMES, start=1):
        out.append({
            "name": name,
            "team_key": f"458.l.123.t.{i}",
            "rank": i,
            "stats": {
                "R": 100 + i * 10, "HR": 20 + i, "RBI": 100 + i * 8,
                "SB": 10 + i, "AVG": 0.250 + i / 1000,
                "W": 10 + i, "K": 150 + i * 5, "SV": 5 + i,
                "ERA": 3.50 + i / 100, "WHIP": 1.15 + i / 1000,
            },
        })
    return out


def roster_for_team(team_index: int) -> list[dict]:
    """One team's roster: 6 hitters, 5 pitchers (1 closer + 4 starters)."""
    base_h = team_index * 6
    base_p = team_index * 5
    out = []
    # Hitters in OF / Util / BN slots
    slot_cycle = ["OF", "OF", "OF", "Util", "BN", "BN"]
    for i in range(6):
        idx = base_h + i
        out.append({
            "name": f"Hitter{idx:03d}",
            "positions": ["OF", "Util"] if i < 4 else ["OF"],
            "selected_position": slot_cycle[i],
            "player_id": f"yh_h_{idx:03d}",
            "status": "",
        })
    # Pitchers — first one is a closer (RP), rest are SP
    for i in range(5):
        idx = base_p + i
        is_closer = i == 0
        out.append({
            "name": f"Pitcher{idx:03d}",
            "positions": ["RP", "P"] if is_closer else ["SP", "P"],
            "selected_position": "P",
            "player_id": f"yh_p_{idx:03d}",
            "status": "",
        })
    return out


def all_rosters() -> dict[str, list[dict]]:
    """All 12 teams' rosters by team name."""
    return {name: roster_for_team(i) for i, name in enumerate(TEAM_NAMES)}


def hitter_game_logs() -> dict[str, dict]:
    """Mid-season actuals keyed by normalized name. ~half the league."""
    out = {}
    for i in range(60):
        name = f"hitter{i:03d}"
        out[name] = {
            "r": 40 + (i % 20), "hr": 10 + (i % 12), "rbi": 40 + (i % 20),
            "sb": 1 + (i % 8), "avg": 0.260 + (i % 30) / 1000,
        }
    return out


def pitcher_game_logs() -> dict[str, dict]:
    out = {}
    for i in range(40):
        name = f"pitcher{i:03d}"
        out[name] = {
            "w": 5 + (i % 6), "k": 90 + (i % 40),
            "sv": 12 if i < 12 else 0,
            "era": 3.20 + (i % 10) / 10, "whip": 1.10 + (i % 10) / 100,
        }
    return out


def free_agents() -> list[dict]:
    """20 free agents — players NOT on any roster."""
    out = []
    # Hitters 72..91 (past the 72 used by 12 teams x 6 hitters)
    for i in range(72, 92):
        out.append({
            "name": f"Hitter{i:03d}",
            "positions": ["OF"],
            "selected_position": "BN",
            "player_id": f"yh_h_{i:03d}",
            "status": "",
        })
    return out


def transactions() -> list[dict]:
    """Empty by default — transaction analyzer handles this."""
    return []


def schedule_payload(start_date: date, end_date: date) -> dict:
    """Empty schedule — get_probable_starters tolerates this."""
    return {}


def team_batting_stats() -> dict[str, Any]:
    """Empty team batting stats — matchup factors fall back to defaults."""
    return {}


def scoring_period() -> tuple[str, str]:
    """Sunday-ending scoring week."""
    return ("2026-04-13", "2026-04-19")  # Mon-Sun
```

- [ ] **Step 2: Verify the helper module imports cleanly**

Run: `python -c "from tests.test_web._refresh_fixture import all_rosters, standings, hitter_projections, pitcher_projections; print(len(all_rosters()), len(standings()), len(hitter_projections()), len(pitcher_projections()))"`
Expected: `12 12 120 80`

- [ ] **Step 3: Commit**

```bash
git add tests/test_web/_refresh_fixture.py
git commit -m "test(web): add minimal League fixture data for refresh integration test"
```

---

### Task 10: Build Yahoo + Redis mock setup module

**Files:**
- Modify: `tests/test_web/_refresh_fixture.py` (add mocks/setup helpers)

**Context:** `run_full_refresh` calls many Yahoo helpers and indirectly reads/writes Redis. We need a single helper that patches all of them and seeds Redis. Doing this once in a helper means the integration test stays focused on assertions.

- [ ] **Step 1: Add mock setup helpers**

Append to `tests/test_web/_refresh_fixture.py`:
```python
import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


def _mock_league(team_keys_to_names: dict[str, str]) -> MagicMock:
    """Yahoo league mock that returns canned teams() and supports the
    attribute access patterns used in run_full_refresh."""
    mock = MagicMock(name="MockLeague")
    mock.teams.return_value = {
        team_key: {"name": tname, "team_key": team_key}
        for team_key, tname in team_keys_to_names.items()
    }
    return mock


def _team_keys_to_names() -> dict[str, str]:
    return {f"458.l.123.t.{i}": name for i, name in enumerate(TEAM_NAMES, start=1)}


def seed_redis(client) -> None:
    """Write the projection blobs into fake Redis so
    redis_get_blended() reads them back. Uses the keys that
    fantasy_baseball.data.redis_store expects."""
    # Match the keys used by data.redis_store.get_blended_projections.
    # The reader inspects keys like "blended_projections:hitters" and
    # "blended_projections:pitchers". Encode as JSON list of dicts.
    client.set("blended_projections:hitters", json.dumps(hitter_projections()))
    client.set("blended_projections:pitchers", json.dumps(pitcher_projections()))


@contextmanager
def patched_refresh_environment(
    fake_redis,
    *,
    has_rest_of_season: bool = True,
    cache_dir,
):
    """Patch every external dependency of run_full_refresh and yield.

    - Yahoo session/league: MagicMock returning canned teams()
    - fetch_roster, fetch_standings, fetch_scoring_period: canned data
    - fetch_all_transactions: returns transactions() (empty by default)
    - fetch_and_match_free_agents: returns ([Player...], None)
    - fetch_game_log_totals: writes nothing (Redis seeded already)
    - get_week_schedule, get_team_batting_stats: return canned/empty
    - run_monte_carlo, run_ros_monte_carlo: 10 iters instead of 1000
    - get_default_client (data.redis_store): returns fake_redis
    - read_cache("ros_projections"): returns ROS proj rows or None
    """
    from fantasy_baseball.models.player import Player, PlayerType
    rosters = all_rosters()
    team_keys = _team_keys_to_names()

    league_mock = _mock_league(team_keys)

    # Seed Redis with projections + League data
    seed_redis(fake_redis)

    # Build League dataclass from rosters by writing snapshot keys
    from fantasy_baseball.data.redis_store import (
        write_roster_snapshot, write_standings_snapshot,
    )
    snapshot_date = "2026-04-21"  # next_tuesday after 2026-04-19
    for tname, team_roster in rosters.items():
        entries = [
            {
                "slot": r["selected_position"],
                "player_name": r["name"],
                "positions": ", ".join(r.get("positions", [])),
                "status": r.get("status") or "",
                "yahoo_id": r.get("player_id") or "",
            }
            for r in team_roster
        ]
        write_roster_snapshot(fake_redis, snapshot_date, tname, entries)
    write_standings_snapshot(
        fake_redis, snapshot_date,
        {"teams": [
            {
                "team": s["name"],
                "team_key": s["team_key"],
                "rank": s["rank"],
                **{k.lower(): v for k, v in s["stats"].items()},
            }
            for s in standings()
        ]},
    )

    # FA players
    fa_player_objs = [
        Player(
            name=fa["name"],
            positions=fa["positions"],
            player_type=PlayerType.HITTER,
            selected_position=fa.get("selected_position", "BN"),
            player_id=fa.get("player_id", ""),
        )
        for fa in free_agents()
    ]

    def _fetch_roster(league, team_key, day=None):
        tname = team_keys.get(team_key)
        return rosters.get(tname, [])

    def _fetch_standings(league):
        return standings()

    def _fetch_scoring_period(league):
        return scoring_period()

    def _fetch_all_transactions(league):
        return transactions()

    def _fetch_and_match_fa(league, hitters_proj, pitchers_proj):
        return (fa_player_objs, None)

    def _fetch_game_logs(season_year, progress_cb=None):
        # Seed game logs into Redis using the data.redis_store API
        from fantasy_baseball.data.redis_store import set_game_log_totals
        try:
            set_game_log_totals(fake_redis, season_year, "hitters", hitter_game_logs())
            set_game_log_totals(fake_redis, season_year, "pitchers", pitcher_game_logs())
        except Exception:
            # Function name may vary — this is a best-effort seed
            pass

    def _ros_pipeline_blend(*args, **kwargs):
        if has_rest_of_season:
            # Write ROS projections into the cache so read_cache picks them up
            from fantasy_baseball.web.season_data import write_cache
            write_cache(
                "ros_projections",
                {"hitters": hitter_projections(), "pitchers": pitcher_projections()},
                cache_dir,
            )

    def _scaled_mc(team_rosters, h_slots, p_slots, user_team_name,
                   n_iterations=1000, use_management=False, progress_cb=None):
        from fantasy_baseball.simulation import run_monte_carlo as real_mc
        return real_mc(
            team_rosters, h_slots, p_slots, user_team_name,
            n_iterations=10, use_management=use_management,
            progress_cb=progress_cb,
        )

    def _scaled_ros_mc(*, team_rosters, actual_standings, fraction_remaining,
                       h_slots, p_slots, user_team_name,
                       n_iterations=1000, use_management=False, progress_cb=None):
        from fantasy_baseball.simulation import run_ros_monte_carlo as real_ros_mc
        return real_ros_mc(
            team_rosters=team_rosters, actual_standings=actual_standings,
            fraction_remaining=fraction_remaining,
            h_slots=h_slots, p_slots=p_slots, user_team_name=user_team_name,
            n_iterations=10, use_management=use_management,
            progress_cb=progress_cb,
        )

    patches = [
        patch("fantasy_baseball.web.refresh_pipeline.get_yahoo_session", return_value=MagicMock()),
        patch("fantasy_baseball.web.refresh_pipeline.get_league", return_value=league_mock),
        patch("fantasy_baseball.web.refresh_pipeline.fetch_standings", side_effect=_fetch_standings),
        patch("fantasy_baseball.web.refresh_pipeline.fetch_scoring_period", side_effect=_fetch_scoring_period),
        patch("fantasy_baseball.web.refresh_pipeline.fetch_roster", side_effect=_fetch_roster),
        patch("fantasy_baseball.web.refresh_pipeline.fetch_all_transactions", side_effect=_fetch_all_transactions),
        patch("fantasy_baseball.web.refresh_pipeline.fetch_and_match_free_agents", side_effect=_fetch_and_match_fa),
        patch("fantasy_baseball.web.refresh_pipeline.fetch_game_log_totals", side_effect=_fetch_game_logs),
        patch("fantasy_baseball.web.refresh_pipeline.get_week_schedule", return_value={}),
        patch("fantasy_baseball.web.refresh_pipeline.get_team_batting_stats", return_value={}),
        patch("fantasy_baseball.web.refresh_pipeline.blend_and_cache_ros", side_effect=_ros_pipeline_blend),
        patch("fantasy_baseball.web.refresh_pipeline.run_monte_carlo", side_effect=_scaled_mc),
        patch("fantasy_baseball.web.refresh_pipeline.run_ros_monte_carlo", side_effect=_scaled_ros_mc),
        patch("fantasy_baseball.data.redis_store.get_default_client", return_value=fake_redis),
        patch("fantasy_baseball.web.season_data._get_redis", return_value=fake_redis),
    ]

    started = []
    try:
        for p in patches:
            started.append(p.start())
        yield
    finally:
        for p in patches:
            p.stop()
```

**Important context note for the implementer:** The patch targets above (`fantasy_baseball.web.refresh_pipeline.get_league`, etc.) assume these names are imported at the module level. They are CURRENTLY imported lazily inside `run_full_refresh`. Two options:

1. **Patch where they live** instead of where they're used: e.g. `patch("fantasy_baseball.auth.yahoo_auth.get_league", ...)`. This works regardless of where the import happens.
2. **Pre-import** the symbols at the top of `refresh_pipeline.py`. Cleaner long-term but a structural change.

Use option 1 (patch at source) for this task. Update the patch targets accordingly. This is verified empirically in Step 3 of Task 11.

- [ ] **Step 2: Verify the module still imports**

Run: `python -c "from tests.test_web._refresh_fixture import patched_refresh_environment; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add tests/test_web/_refresh_fixture.py
git commit -m "test(web): add Yahoo/Redis mock setup for refresh integration test"
```

---

### Task 11: Write integration test with shape assertions

**Files:**
- Create: `tests/test_web/test_refresh_pipeline.py`

**Context:** First end-to-end integration test for `run_full_refresh`. Asserts that every expected cache file is written and has the expected top-level structure. Does NOT yet check invariants (next task) or parametrize ROS branches (task after).

- [ ] **Step 1: Write the test file**

Create `tests/test_web/test_refresh_pipeline.py`:
```python
"""Integration test for run_full_refresh.

Mocks Yahoo and uses fakeredis. Asserts shape of every cache artifact
plus cross-step invariants (in test_invariants below). Does NOT lock
down values — Monte Carlo has randomness and projections change weekly.
"""
import json
from pathlib import Path

import pytest

from fantasy_baseball.web import refresh_pipeline
from tests.test_web._refresh_fixture import patched_refresh_environment


def _read(cache_dir: Path, name: str):
    """Read a cache JSON file."""
    return json.loads((cache_dir / f"{name}.json").read_text())


@pytest.fixture
def configured_test_env(monkeypatch, fake_redis, tmp_path):
    """Set environment variables expected by load_config."""
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "http://fake")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake-token")
    return tmp_path


class TestRefreshShape:
    """Shape assertions: every expected cache file is written with the
    expected top-level keys and types."""

    def test_all_expected_cache_files_written(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)

        expected_files = [
            "standings", "pending_moves", "projections", "roster",
            "rankings", "lineup_optimal", "probable_starters", "positions",
            "roster_audit", "leverage", "monte_carlo", "spoe",
            "transaction_analyzer", "meta", "opp_rosters",
        ]
        for name in expected_files:
            path = cache_dir / f"{name}.json"
            assert path.exists(), f"Missing cache file: {name}.json"

    def test_standings_shape(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "standings")
        assert isinstance(data, list)
        assert len(data) == 12
        for entry in data:
            assert {"name", "team_key", "rank", "stats"}.issubset(entry.keys())
            assert {"R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"}.issubset(
                entry["stats"].keys()
            )

    def test_projections_shape(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "projections")
        assert {"projected_standings", "team_sds", "fraction_remaining"} <= data.keys()
        assert isinstance(data["projected_standings"], list)
        assert isinstance(data["team_sds"], dict)
        assert isinstance(data["fraction_remaining"], (int, float))

    def test_lineup_optimal_shape(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "lineup_optimal")
        assert {"hitter_lineup", "pitcher_starters", "pitcher_bench", "moves"} <= data.keys()
        assert isinstance(data["moves"], list)

    def test_monte_carlo_shape(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "monte_carlo")
        assert "base" in data
        assert "with_management" in data
        # ROS keys may be None when has_rest_of_season=False (next task)

    def test_meta_shape(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "meta")
        assert {"last_refresh", "start_date", "end_date", "team_name"} <= data.keys()
        assert data["team_name"] == "Team 01"
```

- [ ] **Step 2: Run the test to see what's broken**

Run: `pytest tests/test_web/test_refresh_pipeline.py::TestRefreshShape -v -x`
Expected: Tests will likely fail on first run — patches may need to be redirected to where the symbols actually live (e.g. `fantasy_baseball.auth.yahoo_auth.get_league` instead of `fantasy_baseball.web.refresh_pipeline.get_league`). Read the error output and fix patch targets in `_refresh_fixture.py`.

- [ ] **Step 3: Iterate on patch targets until tests pass**

Common fixes you'll need:
- For each patched symbol that's imported lazily inside `run_full_refresh`, change the patch target to the source module. Example: change `fantasy_baseball.web.refresh_pipeline.get_league` → `fantasy_baseball.auth.yahoo_auth.get_league`.
- If `redis_get_blended` is called via `from fantasy_baseball.data.redis_store import get_blended_projections as redis_get_blended`, patch `fantasy_baseball.data.redis_store.get_blended_projections`.
- If a function name in `redis_store` differs from what's used in the fixture (e.g. `set_game_log_totals` doesn't exist), check the actual API in `src/fantasy_baseball/data/redis_store.py` and use the right name.
- If `load_config` requires fields the test doesn't provide, set them via the test config file (`config/league.yaml`) or monkeypatch `load_config` to return a canned `LeagueConfig`.

Iterate: run, read error, fix, repeat. Each fix is a small edit to `_refresh_fixture.py` or the test.

- [ ] **Step 4: Verify all 6 shape tests pass**

Run: `pytest tests/test_web/test_refresh_pipeline.py::TestRefreshShape -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_web/_refresh_fixture.py tests/test_web/test_refresh_pipeline.py
git commit -m "test(web): add shape assertions for run_full_refresh integration"
```

---

### Task 12: Add invariant assertions and parametrize ROS branch

**Files:**
- Modify: `tests/test_web/test_refresh_pipeline.py`

**Context:** Cross-step invariants — assertions that encode the contracts between pipeline steps. These are what catch wiring bugs without locking down values. Then parametrize the ROS branch so both `has_rest_of_season=True` and `=False` paths are exercised.

- [ ] **Step 1: Add invariant test class**

Append to `tests/test_web/test_refresh_pipeline.py`:
```python
class TestRefreshInvariants:
    """Cross-step contracts — these catch wiring regressions."""

    @pytest.fixture(autouse=True)
    def _run_refresh(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        self.cache_dir = cache_dir

    def test_every_team_in_standings_appears_in_projected_standings(self):
        standings = _read(self.cache_dir, "standings")
        projections = _read(self.cache_dir, "projections")
        standings_names = {t["name"] for t in standings}
        projected_names = {t["name"] for t in projections["projected_standings"]}
        assert standings_names == projected_names

    def test_every_roster_player_has_pace(self):
        roster = _read(self.cache_dir, "roster")
        for player in roster:
            assert "pace" in player, f"{player.get('name')} missing pace"
            assert player["pace"] is not None

    def test_lineup_moves_only_reference_roster_players(self):
        roster = _read(self.cache_dir, "roster")
        optimal = _read(self.cache_dir, "lineup_optimal")
        roster_names = {p["name"] for p in roster}
        for move in optimal["moves"]:
            assert move["player"] in roster_names, (
                f"Move references {move['player']!r} not on roster"
            )

    def test_positions_map_covers_roster_and_opponents_and_fas(self):
        positions = _read(self.cache_dir, "positions")
        roster = _read(self.cache_dir, "roster")
        opp_rosters = _read(self.cache_dir, "opp_rosters")
        from fantasy_baseball.utils.name_utils import normalize_name
        # Roster players
        for p in roster:
            assert normalize_name(p["name"]) in positions
        # Opponent players
        for opp_name, opp_roster in opp_rosters.items():
            for p in opp_roster:
                assert normalize_name(p["name"]) in positions

    def test_meta_last_refresh_is_set(self):
        meta = _read(self.cache_dir, "meta")
        assert meta["last_refresh"]  # truthy

    def test_meta_team_name_matches_config(self):
        meta = _read(self.cache_dir, "meta")
        assert meta["team_name"] == "Team 01"
```

- [ ] **Step 2: Run invariants and fix any that fail**

Run: `pytest tests/test_web/test_refresh_pipeline.py::TestRefreshInvariants -v`
Expected: PASS (6 tests). Failures here indicate either a real bug in the current pipeline or a fixture issue — investigate before adjusting the test.

- [ ] **Step 3: Add parametrized ROS test class**

Append to `tests/test_web/test_refresh_pipeline.py`:
```python
class TestMonteCarloROSBranch:
    @pytest.mark.parametrize("has_ros", [True, False])
    def test_monte_carlo_keys_match_ros_availability(
        self, configured_test_env, fake_redis, has_ros,
    ):
        cache_dir = configured_test_env
        with patched_refresh_environment(
            fake_redis, has_rest_of_season=has_ros, cache_dir=cache_dir,
        ):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "monte_carlo")
        assert data["base"] is not None
        assert data["with_management"] is not None
        if has_ros:
            assert data["rest_of_season"] is not None
            assert data["rest_of_season_with_management"] is not None
        else:
            assert data["rest_of_season"] is None
            assert data["rest_of_season_with_management"] is None
```

- [ ] **Step 4: Run the parametrized tests**

Run: `pytest tests/test_web/test_refresh_pipeline.py::TestMonteCarloROSBranch -v`
Expected: PASS (2 tests — one for each parametrize value)

- [ ] **Step 5: Run the full integration test file**

Run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: PASS (14 tests total: 6 shape + 6 invariants + 2 parametrize)

- [ ] **Step 6: Commit**

```bash
git add tests/test_web/test_refresh_pipeline.py
git commit -m "test(web): add invariant assertions and ROS parametrize for refresh"
```

---

## Phase 3: Wire pure helpers into existing run_full_refresh

Each task in this phase replaces inline logic in `run_full_refresh` with a call to the new helper. The integration test from Phase 2 acts as the safety net — it must pass after every change.

### Task 13: Wire `compute_effective_date` and `compute_fraction_remaining`

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py`

**Context:** Replaces inline computations at three sites: line ~163 (effective_date), lines ~405-409 (fraction_remaining for SD scaling), lines ~697-702 (fraction_remaining for ROS MC). This is the bonus dedup mentioned in the spec — `fraction_remaining` is computed from the same inputs in two places.

- [ ] **Step 1: Modify `refresh_pipeline.py`**

Find and replace at line ~163 in `run_full_refresh`:
```python
        # OLD:
        start_date, end_date = fetch_scoring_period(league)
        effective_date = next_tuesday(date.fromisoformat(end_date))

        # NEW:
        from fantasy_baseball.utils.time_utils import compute_effective_date
        start_date, end_date = fetch_scoring_period(league)
        effective_date = compute_effective_date(end_date)
```

Find and replace at lines ~404-410 in `run_full_refresh`:
```python
        # OLD:
        import math
        _season_start = date.fromisoformat(config.season_start)
        _season_end = date.fromisoformat(config.season_end)
        _total_days = (_season_end - _season_start).days
        _remaining_days = max(0, (_season_end - local_today()).days)
        fraction_remaining = (_remaining_days / _total_days) if _total_days > 0 else 0.0
        _sd_scale = math.sqrt(fraction_remaining)

        # NEW:
        from fantasy_baseball.utils.time_utils import compute_fraction_remaining
        fraction_remaining = compute_fraction_remaining(
            date.fromisoformat(config.season_start),
            date.fromisoformat(config.season_end),
            local_today(),
        )
        _sd_scale = math.sqrt(fraction_remaining)
```

Find and replace at lines ~697-702 in `run_full_refresh` (Step 13b):
```python
        # OLD:
        if has_rest_of_season:
            from fantasy_baseball.simulation import run_ros_monte_carlo
            season_start = date.fromisoformat(config.season_start)
            season_end = date.fromisoformat(config.season_end)
            total_days = (season_end - season_start).days
            remaining_days = max(0, (season_end - local_today()).days)
            fraction_remaining = remaining_days / total_days if total_days > 0 else 0

        # NEW (re-uses fraction_remaining computed earlier):
        if has_rest_of_season:
            from fantasy_baseball.simulation import run_ros_monte_carlo
            # fraction_remaining was computed in Step 4e and is reused here
```

- [ ] **Step 2: Run integration test to verify no regression**

Run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: PASS (14 tests)

- [ ] **Step 3: Run the existing source-string regression guards**

Run: `pytest tests/test_web/test_season_data.py -v`
Expected: PASS (still relies on inspect.getsource(run_full_refresh) — not yet broken)

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py
git commit -m "refactor(refresh): use time_utils helpers and dedup fraction_remaining"
```

---

### Task 14: Wire `build_projected_standings` and `build_team_sds`

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py`

**Context:** Replaces inline loops at lines ~389-416 with calls to the new scoring helpers.

- [ ] **Step 1: Modify `refresh_pipeline.py`**

Find and replace in `run_full_refresh` (Step 4e block, lines ~387-417):
```python
        # OLD:
        from fantasy_baseball.scoring import project_team_stats

        all_team_rosters = {config.team_name: matched}
        all_team_rosters.update(opp_rosters)

        projected_standings = []
        for tname, roster in all_team_rosters.items():
            proj_stats = project_team_stats(roster, displacement=True)
            projected_standings.append({
                "name": tname,
                "team_key": "",
                "rank": 0,
                "stats": proj_stats.to_dict(),
            })

        import math
        # ... fraction_remaining lines (already replaced in Task 13) ...

        from fantasy_baseball.scoring import project_team_sds
        team_sds: dict[str, dict[str, float]] = {}
        for _tname, _troster in all_team_rosters.items():
            _raw_sds = project_team_sds(_troster, displacement=True)
            team_sds[_tname] = {c: sd * _sd_scale for c, sd in _raw_sds.items()}

        # NEW:
        from fantasy_baseball.scoring import build_projected_standings, build_team_sds

        all_team_rosters = {config.team_name: matched}
        all_team_rosters.update(opp_rosters)

        projected_standings = build_projected_standings(all_team_rosters)

        # _sd_scale and fraction_remaining computed earlier (Task 13)
        team_sds = build_team_sds(all_team_rosters, _sd_scale)
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: PASS (14 tests)

- [ ] **Step 3: Run unit tests for scoring**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py
git commit -m "refactor(refresh): use build_projected_standings and build_team_sds"
```

---

### Task 15: Wire `attach_pace_to_roster`

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py`

**Context:** Replaces inline pace-loop at lines ~482-502.

- [ ] **Step 1: Modify `refresh_pipeline.py`**

Find and replace in `run_full_refresh` (Step 6c block):
```python
        # OLD:
        from fantasy_baseball.sgp.denominators import get_sgp_denominators
        sgp_denoms = get_sgp_denominators(config.sgp_overrides)
        for player in roster_players:
            norm = normalize_name(player.name)
            if player.player_type == PlayerType.HITTER:
                actuals = hitter_logs.get(norm, {})
                rest_of_season_keys = ["r", "hr", "rbi", "sb", "avg"]
            else:
                actuals = pitcher_logs.get(norm, {})
                rest_of_season_keys = ["w", "k", "sv", "era", "whip"]
            proj_keys = HITTER_PROJ_KEYS if player.player_type == PlayerType.HITTER else PITCHER_PROJ_KEYS
            pre_player = preseason_lookup.get(norm)
            if pre_player and pre_player.rest_of_season:
                projected = {k: getattr(pre_player.rest_of_season, k, 0) for k in proj_keys}
            else:
                projected = {k: 0 for k in proj_keys}
            rest_of_season_dict = {k: getattr(player.rest_of_season, k, 0) for k in rest_of_season_keys} if player.rest_of_season else None
            player.pace = compute_player_pace(
                actuals, projected, player.player_type,
                rest_of_season_stats=rest_of_season_dict, sgp_denoms=sgp_denoms,
            )

        # NEW:
        from fantasy_baseball.sgp.denominators import get_sgp_denominators
        from fantasy_baseball.analysis.pace import attach_pace_to_roster
        sgp_denoms = get_sgp_denominators(config.sgp_overrides)
        attach_pace_to_roster(
            roster_players, hitter_logs, pitcher_logs,
            preseason_lookup, sgp_denoms,
        )
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: PASS (14 tests)

- [ ] **Step 3: Run pace tests**

Run: `pytest tests/test_analysis/test_pace.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py
git commit -m "refactor(refresh): use attach_pace_to_roster helper"
```

---

### Task 16: Wire `build_rankings_lookup`

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py`

**Context:** Replaces inline three-way merge at lines ~526-533.

- [ ] **Step 1: Modify `refresh_pipeline.py`**

Find and replace in `run_full_refresh` (Step 6d block):
```python
        # OLD:
        all_keys = set(rest_of_season_ranks) | set(preseason_ranks) | set(current_ranks)
        rankings_lookup = {}
        for key in all_keys:
            rankings_lookup[key] = {
                "rest_of_season": rest_of_season_ranks.get(key),
                "preseason": preseason_ranks.get(key),
                "current": current_ranks.get(key),
            }

        # NEW:
        from fantasy_baseball.sgp.rankings import build_rankings_lookup
        rankings_lookup = build_rankings_lookup(
            rest_of_season_ranks, preseason_ranks, current_ranks,
        )
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: PASS (14 tests)

- [ ] **Step 3: Run rankings tests**

Run: `pytest tests/test_sgp/test_rankings.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py
git commit -m "refactor(refresh): use build_rankings_lookup helper"
```

---

### Task 17: Wire `merge_matched_and_raw_roster`, `compute_lineup_moves`, `build_positions_map`

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py`

**Context:** Replaces three separate inline blocks: roster merge (~449-469), moves computation (~578-597), positions map (~636-648). Doing them in one task because they share the new `refresh_steps` import and produce reviewable-sized diffs.

- [ ] **Step 1: Add the import once at top of `run_full_refresh`**

After the existing imports inside `run_full_refresh` (after line ~123), add:
```python
        from fantasy_baseball.web.refresh_steps import (
            build_positions_map,
            compute_lineup_moves,
            merge_matched_and_raw_roster,
        )
```

- [ ] **Step 2: Replace the roster-merge block (Step 6, ~lines 448-470)**

```python
        # OLD:
        # Build Player objects from matched entries
        matched_names = set()
        roster_players: list[Player] = []
        for player in matched:
            norm = normalize_name(player.name)
            matched_names.add(norm)

            # Attach preseason stat bag
            pre_entry = preseason_lookup.get(norm)
            if pre_entry and pre_entry.rest_of_season:
                player.preseason = pre_entry.rest_of_season

            roster_players.append(player)

        # Include unmatched players
        for raw_player in roster_raw:
            if normalize_name(raw_player["name"]) not in matched_names:
                player = Player.from_dict({
                    **raw_player,
                    "player_type": PlayerType.PITCHER if set(raw_player.get("positions", [])) & PITCHER_POSITIONS else PlayerType.HITTER,
                })
                roster_players.append(player)

        # NEW:
        roster_players = merge_matched_and_raw_roster(
            matched, roster_raw, preseason_lookup,
        )
```

- [ ] **Step 3: Replace the moves-computation block (Step 8, ~lines 577-598)**

```python
        # OLD:
        moves = []
        for slot, player_name in optimal_hitters.items():
            for player in roster_players:
                if player.name == player_name:
                    current_slot = player.selected_position or "BN"
                    base_slot = slot.split("_")[0]
                    bench_slots = {"BN", "IL", "DL"}
                    if current_slot != base_slot and (
                        current_slot in bench_slots or base_slot in bench_slots
                    ):
                        moves.append({
                            "action": "START",
                            "player": player_name,
                            "slot": base_slot,
                            "reason": f"wSGP: {player.wsgp:.1f}",
                        })
                    break

        # NEW:
        moves = compute_lineup_moves(optimal_hitters, roster_players)
```

- [ ] **Step 4: Replace the positions-map block (Step 10, ~lines 635-648)**

```python
        # OLD:
        from fantasy_baseball.utils.name_utils import normalize_name as _norm
        positions_map: dict[str, list[str]] = {}
        for p in roster_players:
            positions_map[_norm(p.name)] = list(p.positions)
        for _opp_roster in opp_rosters.values():
            for p in _opp_roster:
                positions_map[_norm(p.name)] = list(p.positions)
        for fa in fa_players:
            if fa.positions:
                positions_map[_norm(fa.name)] = list(fa.positions)

        # NEW:
        positions_map = build_positions_map(roster_players, opp_rosters, fa_players)
```

- [ ] **Step 5: Run the integration test**

Run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: PASS (14 tests)

- [ ] **Step 6: Run the refresh_steps unit tests**

Run: `pytest tests/test_web/test_refresh_steps.py -v`
Expected: PASS (18 tests)

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py
git commit -m "refactor(refresh): use refresh_steps helpers (merge, moves, positions)"
```

---

## Phase 4: Refactor to RefreshRun class

This phase mechanically extracts each `# --- Step N` block into a method on a new `RefreshRun` class. The integration test must still pass after the refactor — it's the primary safety net.

### Task 18: Create RefreshRun class skeleton with state attributes

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py`

**Context:** Add the `RefreshRun` class to the module without yet replacing `run_full_refresh`. This isolates the addition from the migration so each step is reviewable.

- [ ] **Step 1: Add the class skeleton near the top of `refresh_pipeline.py`**

Add after the existing module-level functions (after `_write_spoe_snapshot`, before `def run_full_refresh`):
```python
class RefreshRun:
    """Encapsulates one execution of the season dashboard refresh.

    Each step from the original ``run_full_refresh`` is now a private
    method. The class holds shared state as instance attributes so
    methods don't need 10-arg signatures. Methods are NOT individually
    unit-tested; the integration test in
    ``tests/test_web/test_refresh_pipeline.py`` covers them collectively.

    Module-level state (``_refresh_lock``, ``_refresh_status``) is shared
    across threads and stays at module scope.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        from fantasy_baseball.web.job_logger import JobLogger
        self.cache_dir = cache_dir
        self.logger = JobLogger("refresh")

        # Shared state — populated as steps run. All initialized to None
        # so attribute access errors surface as clear AttributeErrors
        # rather than silent fall-through to the wrong type.
        self.config = None
        self.league = None              # Yahoo session-bound league
        self.league_model = None        # League dataclass loaded from Redis
        self.user_team_key = None
        self.standings = None
        self.standings_snap = None
        self.projected_standings = None
        self.projected_standings_snap = None
        self.team_sds = None
        self.fraction_remaining = None
        self.sd_scale = None
        self.effective_date = None
        self.start_date = None
        self.end_date = None
        self.roster_raw = None
        self.raw_rosters_by_team = None
        self.opp_rosters = None
        self.matched = None
        self.roster_players = None
        self.preseason_lookup = None
        self.preseason_hitters = None
        self.preseason_pitchers = None
        self.hitters_proj = None
        self.pitchers_proj = None
        self.has_rest_of_season = False
        self.hitter_logs = None
        self.pitcher_logs = None
        self.leverage = None
        self.rankings_lookup = None
        self.optimal_hitters = None
        self.optimal_pitchers_starters = None
        self.optimal_pitchers_bench = None
        self.fa_players = None

    def _progress(self, msg: str) -> None:
        _set_refresh_progress(msg)
        self.logger.log(msg)
        log.info(msg)

    def run(self) -> None:
        """Run the full refresh pipeline.

        Same try/except/finally protocol as the legacy ``run_full_refresh``:
        sets ``_refresh_status`` throughout, captures errors into
        ``_refresh_status['error']`` while still raising, and clears
        ``running`` in the ``finally`` block.
        """
        with _refresh_lock:
            _refresh_status["running"] = True
            _refresh_status["progress"] = "Starting..."
            _refresh_status["error"] = None

        try:
            self._authenticate()
            self._find_user_team()
            self._fetch_standings_and_roster()
            self._load_projections()
            self._fetch_opponent_rosters()
            self._write_snapshots_and_load_league()
            self._hydrate_rosters()
            self._build_projected_standings()
            self._compute_leverage()
            self._match_roster_to_projections()
            self._fetch_game_logs()
            self._compute_pace()
            self._compute_wsgp()
            self._compute_rankings()
            self._optimize_lineup()
            self._compute_moves()
            self._fetch_probable_starters()
            self._audit_roster()
            self._compute_per_team_leverage()
            self._run_monte_carlo()
            self._run_ros_monte_carlo()
            self._compute_spoe()
            self._analyze_transactions()
            self._write_meta()

            self.logger.finish("ok")
            self._progress("Done")
            from fantasy_baseball.web.season_data import clear_opponent_cache
            clear_opponent_cache()
        except Exception as exc:
            with _refresh_lock:
                _refresh_status["error"] = str(exc)
            self.logger.finish("error", str(exc))
            raise
        finally:
            with _refresh_lock:
                _refresh_status["running"] = False

    # Step methods — populated in subsequent tasks
    def _authenticate(self): raise NotImplementedError
    def _find_user_team(self): raise NotImplementedError
    def _fetch_standings_and_roster(self): raise NotImplementedError
    def _load_projections(self): raise NotImplementedError
    def _fetch_opponent_rosters(self): raise NotImplementedError
    def _write_snapshots_and_load_league(self): raise NotImplementedError
    def _hydrate_rosters(self): raise NotImplementedError
    def _build_projected_standings(self): raise NotImplementedError
    def _compute_leverage(self): raise NotImplementedError
    def _match_roster_to_projections(self): raise NotImplementedError
    def _fetch_game_logs(self): raise NotImplementedError
    def _compute_pace(self): raise NotImplementedError
    def _compute_wsgp(self): raise NotImplementedError
    def _compute_rankings(self): raise NotImplementedError
    def _optimize_lineup(self): raise NotImplementedError
    def _compute_moves(self): raise NotImplementedError
    def _fetch_probable_starters(self): raise NotImplementedError
    def _audit_roster(self): raise NotImplementedError
    def _compute_per_team_leverage(self): raise NotImplementedError
    def _run_monte_carlo(self): raise NotImplementedError
    def _run_ros_monte_carlo(self): raise NotImplementedError
    def _compute_spoe(self): raise NotImplementedError
    def _analyze_transactions(self): raise NotImplementedError
    def _write_meta(self): raise NotImplementedError
```

- [ ] **Step 2: Confirm class imports cleanly (run_full_refresh untouched)**

Run: `python -c "from fantasy_baseball.web.refresh_pipeline import RefreshRun, run_full_refresh; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Confirm integration test still passes (run_full_refresh path is unchanged)**

Run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: PASS (14 tests)

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py
git commit -m "feat(refresh): add RefreshRun class skeleton"
```

---

### Task 19: Migrate step bodies into RefreshRun methods

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py`

**Context:** This is the core refactor. Each `# --- Step N` block in `run_full_refresh` becomes the body of the corresponding `_method()` on `RefreshRun`. Local variables become `self.X` references. The full mechanical mapping:

| Source step | Target method | Reads from self | Writes to self |
|---|---|---|---|
| Step 1 | `_authenticate` | (none) | `config`, `league` |
| Step 2 | `_find_user_team` | `config`, `league` | `user_team_key` |
| Step 3 | `_fetch_standings_and_roster` | `league`, `user_team_key`, `config`, `cache_dir` | `standings`, `start_date`, `end_date`, `effective_date`, `standings_snap`, `roster_raw` |
| Step 4 | `_load_projections` | `cache_dir`, `config` | `hitters_proj`, `pitchers_proj`, `preseason_hitters`, `preseason_pitchers`, `has_rest_of_season` |
| Step 4b | `_fetch_opponent_rosters` | `league`, `config`, `effective_date`, `roster_raw` | `raw_rosters_by_team` |
| Step 4c | `_write_snapshots_and_load_league` | `effective_date`, `raw_rosters_by_team`, `standings`, `config` | `league_model` |
| Step 4d | `_hydrate_rosters` | `league_model`, `config`, `hitters_proj`, `pitchers_proj`, `cache_dir` | `matched`, `opp_rosters` |
| Step 4e | `_build_projected_standings` | `config`, `matched`, `opp_rosters`, `cache_dir` | `projected_standings`, `team_sds`, `fraction_remaining`, `sd_scale`, `projected_standings_snap` |
| Step 5 | `_compute_leverage` | `standings_snap`, `config`, `projected_standings_snap` | `leverage` |
| Step 6 | `_match_roster_to_projections` | `roster_raw`, `preseason_hitters`, `preseason_pitchers`, `matched` | `roster_players`, `preseason_lookup` |
| Step 6b | `_fetch_game_logs` | `config` | (none — writes to Redis via fetch_game_log_totals) |
| Step 6c | `_compute_pace` | `config`, `roster_players`, `preseason_lookup` | `hitter_logs`, `pitcher_logs` |
| Step 6e | `_compute_wsgp` | `roster_players`, `leverage` | (mutates roster_players) |
| Step 6d | `_compute_rankings` | `hitters_proj`, `pitchers_proj`, `preseason_hitters`, `preseason_pitchers`, `hitter_logs`, `pitcher_logs`, `roster_players`, `cache_dir` | `rankings_lookup` |
| Step 7 | `_optimize_lineup` | `roster_players`, `leverage`, `config` | `optimal_hitters`, `optimal_pitchers_starters`, `optimal_pitchers_bench` |
| Step 8 | `_compute_moves` | `optimal_hitters`, `optimal_pitchers_starters`, `optimal_pitchers_bench`, `roster_players`, `cache_dir` | (none — writes lineup_optimal to cache) |
| Step 9 | `_fetch_probable_starters` | `start_date`, `end_date`, `roster_players`, `cache_dir` | (writes probable_starters cache) |
| Step 10 | `_audit_roster` | `league`, `hitters_proj`, `pitchers_proj`, `roster_players`, `opp_rosters`, `leverage`, `config`, `projected_standings`, `team_sds`, `cache_dir` | `fa_players` |
| Step 11 | `_compute_per_team_leverage` | `standings_snap`, `projected_standings_snap`, `cache_dir` | (writes leverage cache) |
| Step 12 | `_run_monte_carlo` | `config`, `matched`, `opp_rosters` | `base_mc`, `mgmt_mc` (or write straight through to cache in the dict at end) |
| Step 13b | `_run_ros_monte_carlo` | `has_rest_of_season`, `config`, `matched`, `opp_rosters`, `standings`, `fraction_remaining`, `cache_dir` | (writes monte_carlo cache) |
| Step 14 | `_compute_spoe` | `preseason_hitters`, `preseason_pitchers`, `league_model`, `standings`, `config`, `cache_dir` | (writes spoe cache) |
| Step 15 | `_analyze_transactions` | `league`, `league_model`, `config`, `cache_dir` | (writes transactions and transaction_analyzer caches) |
| Step 16 | `_write_meta` | `start_date`, `end_date`, `config`, `cache_dir` | (writes meta cache) |

The mechanical pattern for each step:
1. Copy the `# --- Step N` block from `run_full_refresh`.
2. Paste it into the matching method body.
3. Rename every local that's listed in the "Reads from self" or "Writes to self" column to use `self.X`.
4. Replace `_progress(msg)` calls with `self._progress(msg)`.

Two places to be careful:
- Step 12 produces `base_mc` and `mgmt_mc` as locals consumed by Step 13b. Choose: either store them on `self` (cleaner) or merge Step 12 and 13b's cache write into the same method. The plan above stores on `self.base_mc`, `self.mgmt_mc` (added to `__init__`).
- Step 13b's `rest_of_season_mc` and `rest_of_season_mgmt_mc` are also written to cache alongside `base_mc`/`mgmt_mc` in a single `write_cache("monte_carlo", ...)` call. Either Step 13b does the cache write (after consuming `self.base_mc`/`self.mgmt_mc`) or the write moves into Step 12 (and Step 13b just sets the ROS values on `self`).

Recommended: do the `write_cache("monte_carlo", ...)` call inside `_run_ros_monte_carlo` (the last MC step), reading the four MC results from `self`. Add `self.base_mc`, `self.mgmt_mc`, `self.rest_of_season_mc`, `self.rest_of_season_mgmt_mc` to `__init__`.

- [ ] **Step 1: Add the four MC attributes to `RefreshRun.__init__`**

Add to `__init__`:
```python
        self.base_mc = None
        self.mgmt_mc = None
        self.rest_of_season_mc = None
        self.rest_of_season_mgmt_mc = None
```

- [ ] **Step 2: Migrate step bodies — work top-down, one method at a time**

This is the largest single task in the plan. Consider committing in batches of 4-6 methods (e.g., Steps 1-4, 4b-4e, 5-6e, 6d-9, 10-13b, 14-16) so each commit is reviewable and the integration test can be re-run between batches. The integration test only fully passes once ALL methods are migrated and `run_full_refresh` is switched over (Step 4 below) — but module-level imports and class instantiation should keep working after each batch.

For each method in the table above, copy the corresponding block from `run_full_refresh` and rename locals to `self.X`. Examples for the simplest ones to ground the pattern:

`_authenticate`:
```python
    def _authenticate(self):
        from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
        from fantasy_baseball.config import load_config

        self._progress("Authenticating with Yahoo...")
        sc = get_yahoo_session()
        project_root = Path(__file__).resolve().parents[3]
        self.config = load_config(project_root / "config" / "league.yaml")
        self.league = get_league(sc, self.config.league_id, self.config.game_code)
```

`_find_user_team`:
```python
    def _find_user_team(self):
        self._progress("Finding team...")
        teams = self.league.teams()
        for key, team_info in teams.items():
            if team_info.get("name") == self.config.team_name:
                self.user_team_key = key
                break
        if self.user_team_key is None:
            self.user_team_key = next(iter(teams))
```

`_write_meta`:
```python
    def _write_meta(self):
        self._progress("Finalizing...")
        meta = {
            "last_refresh": local_now().strftime("%Y-%m-%d %H:%M"),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "team_name": self.config.team_name,
        }
        write_cache("meta", meta, self.cache_dir)
```

Continue for all 24 methods. Use the table above as the contract for what each method reads from / writes to `self`. Keep all comments from the original blocks (they document the *why* — which is the reason the comments survive in the codebase).

- [ ] **Step 3: Make the migration runnable by checking imports**

Once all methods have bodies (no more `raise NotImplementedError`):
```bash
python -c "from fantasy_baseball.web.refresh_pipeline import RefreshRun; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Run the integration test against the new class**

Temporarily change `run_full_refresh` to delegate so we can test the class:
```python
def run_full_refresh(cache_dir: Path = CACHE_DIR) -> None:
    RefreshRun(cache_dir).run()
```

Then run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: PASS (14 tests). Failures here mean the migration introduced a bug — investigate the specific assertion that failed and check the corresponding method's `self.X` references against the original code.

- [ ] **Step 5: Delete the old `run_full_refresh` body**

After tests pass, the old body is dead code — remove everything that used to be inside `run_full_refresh` (lines from `with _refresh_lock:` through to the end of the original `finally:` block). The function should now be only:
```python
def run_full_refresh(cache_dir: Path = CACHE_DIR) -> None:
    """Connect to Yahoo, fetch all data, run computations, and write cache files.

    Thin wrapper around RefreshRun for backward compatibility with
    existing callers (scripts/run_lineup.py, season_routes.py).
    """
    RefreshRun(cache_dir).run()
```

- [ ] **Step 6: Re-run integration test**

Run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: PASS (14 tests)

- [ ] **Step 7: Run the full project test suite to catch any unrelated regressions**

Run: `pytest -v`
Expected: PASS overall, except potentially the source-string regression guards in `test_season_data.py` (those are fixed in Task 20).

- [ ] **Step 8: Commit**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py
git commit -m "refactor(refresh): migrate run_full_refresh body into RefreshRun methods"
```

---

## Phase 5: Update existing regression guards

### Task 20: Update `test_season_data.py` source-string guards

**Files:**
- Modify: `tests/test_web/test_season_data.py`

**Context:** The existing source-string regression guards use `inspect.getsource(refresh_pipeline.run_full_refresh)`. After Task 19, that function is just `RefreshRun(cache_dir).run()` — the source string no longer contains the step logic. The guards need to walk every method on `RefreshRun` instead.

- [ ] **Step 1: Read the existing guards to understand what they check**

Open `tests/test_web/test_season_data.py` and find the two tests around `inspect.getsource(refresh_pipeline.run_full_refresh)`:
- One walks `config.<attr>` references and confirms each exists on `LeagueConfig`
- One asserts no `from datetime import date` inside `run_full_refresh` (the date import must be at module level)

- [ ] **Step 2: Add a helper to the test file**

At the top of the relevant test class (or as a module-level function), add:
```python
def _refresh_run_source() -> str:
    """Concatenate the source of every RefreshRun method.

    The pre-class regression guards inspected ``run_full_refresh``
    directly. After the RefreshRun refactor, the same logic lives in
    methods on the class — this helper rebuilds the equivalent source
    blob so the guards keep working.
    """
    cls = refresh_pipeline.RefreshRun
    return "\n".join(
        inspect.getsource(getattr(cls, name))
        for name in dir(cls)
        if callable(getattr(cls, name))
        and not name.startswith("__")
    )
```

- [ ] **Step 3: Replace both `inspect.getsource(refresh_pipeline.run_full_refresh)` calls**

Find every call to `inspect.getsource(refresh_pipeline.run_full_refresh)` in `tests/test_web/test_season_data.py` and replace with `_refresh_run_source()`.

- [ ] **Step 4: Update assertion error messages**

Where the test error messages mention "run_full_refresh", change to "RefreshRun" so the failure is debuggable. Example:
```python
# OLD:
f"run_full_refresh references config attributes that don't exist on LeagueConfig: {missing}"
# NEW:
f"RefreshRun references config attributes that don't exist on LeagueConfig: {missing}"
```

- [ ] **Step 5: Run the regression guards**

Run: `pytest tests/test_web/test_season_data.py -v -k "refresh or run_full"`
Expected: PASS

- [ ] **Step 6: Run the full test suite as a final check**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_web/test_season_data.py
git commit -m "test(web): walk RefreshRun methods in source-string regression guards"
```

---

## Final verification

- [ ] **Run full test suite one more time**

Run: `pytest -v`
Expected: All tests pass, no skips beyond what was already in the codebase.

- [ ] **Verify `run_lineup.py` still works locally** (per `feedback_local_testing` and `feedback_run_refresh_before_merge` memories)

Run: `python scripts/run_lineup.py` (requires Yahoo OAuth — only run if you have local credentials).
Expected: Refresh completes without error, cache files written.

- [ ] **Confirm branch is ready for merge**

Run: `git log --oneline main..refresh-pipeline-testing`
Expected: ~20 commits matching the tasks above. None of them touch unrelated files.
