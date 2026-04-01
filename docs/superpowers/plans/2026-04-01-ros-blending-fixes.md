# ROS Blending Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix ROS projection blending so remaining-games systems are normalized to full-season, and compute leverage from blended current+projected standings.

**Architecture:** Two independent fixes: (1) add player-level actual stats to remaining-games ROS projections before cross-system blend, (2) new `blend_standings()` utility that `calculate_leverage` uses when projected standings are available. Pipeline reorder in `season_data.py` moves opponent roster fetching before leverage computation.

**Tech Stack:** Python, pandas, SQLite, pytest

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/data/fangraphs.py` | Modify | Add `MLBAMID` to column maps |
| `src/fantasy_baseball/data/projections.py` | Modify | Add `FULL_SEASON_ROS_SYSTEMS` constant, `normalize_ros_to_full_season()`, normalizer param on `blend_projections()` |
| `src/fantasy_baseball/data/db.py` | Modify | Add `get_season_totals()`, update `load_ros_projections()` |
| `src/fantasy_baseball/lineup/leverage.py` | Modify | Add `blend_standings()`, update `calculate_leverage()` |
| `src/fantasy_baseball/web/season_data.py` | Modify | Reorder pipeline, wire projected standings |
| `scripts/summary.py` | Modify | Pass projected standings to leverage |
| `scripts/run_lineup.py` | Modify | Read cached projected standings, pass to leverage |
| `tests/fixtures/steamer_hitters.csv` | Modify | Add MLBAMID column |
| `tests/fixtures/steamer_pitchers.csv` | Modify | Add MLBAMID column |
| `tests/fixtures/zips_hitters.csv` | Modify | Add MLBAMID column |
| `tests/fixtures/zips_pitchers.csv` | Modify | Add MLBAMID column |
| `tests/test_data/test_projections.py` | Modify | Add normalization tests |
| `tests/test_data/test_db.py` | Modify | Add get_season_totals test |
| `tests/test_lineup/test_leverage.py` | Modify | Add blend_standings + projected leverage tests |

---

### Task 1: Add mlbam_id to fangraphs column maps

**Files:**
- Modify: `src/fantasy_baseball/data/fangraphs.py:4-18` (HITTING_COLUMN_MAP)
- Modify: `src/fantasy_baseball/data/fangraphs.py:20-35` (PITCHING_COLUMN_MAP)
- Modify: `tests/fixtures/steamer_hitters.csv`, `tests/fixtures/zips_hitters.csv`, `tests/fixtures/steamer_pitchers.csv`, `tests/fixtures/zips_pitchers.csv`
- Test: `tests/test_data/test_projections.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_data/test_projections.py`:

```python
def test_blend_preserves_mlbam_id_in_metadata(self, fixtures_dir):
    """mlbam_id should be available in loaded system DataFrames."""
    from fantasy_baseball.data.fangraphs import load_projection_set
    hitters, pitchers = load_projection_set(fixtures_dir, "steamer")
    assert "mlbam_id" in hitters.columns, "mlbam_id missing from hitter columns"
    judge = hitters[hitters["name"] == "Aaron Judge"].iloc[0]
    assert judge["mlbam_id"] == 592450
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_projections.py::TestBlendProjections::test_blend_preserves_mlbam_id_in_metadata -v`
Expected: FAIL — `mlbam_id` not in columns (MLBAMID not mapped)

- [ ] **Step 3: Add MLBAMID to fixture CSVs**

Update `tests/fixtures/steamer_hitters.csv`:
```csv
Name,Team,G,PA,AB,H,2B,3B,HR,R,RBI,BB,SO,HBP,SB,CS,AVG,OBP,SLG,OPS,playerid,MLBAMID
Aaron Judge,NYY,155,650,550,160,30,1,45,110,120,90,170,5,5,1,.291,.400,.580,.980,15640,592450
Mookie Betts,LAD,145,620,540,155,35,3,28,105,85,70,100,4,15,3,.287,.370,.510,.880,13611,605141
Adley Rutschman,BAL,140,600,520,140,30,1,22,80,90,75,110,2,2,1,.269,.360,.460,.820,28442,668939
Marcus Semien,TEX,155,680,610,160,32,2,24,100,80,55,130,6,12,4,.262,.325,.440,.765,12532,543760
```

Update `tests/fixtures/zips_hitters.csv`:
```csv
Name,Team,G,PA,AB,H,2B,3B,HR,R,RBI,BB,SO,HBP,SB,CS,AVG,OBP,SLG,OPS,playerid,MLBAMID
Aaron Judge,NYY,150,640,545,155,28,1,42,105,115,85,175,5,4,1,.284,.395,.570,.965,15640,592450
Mookie Betts,LAD,148,625,545,158,33,2,30,108,88,72,98,3,14,3,.290,.372,.520,.892,13611,605141
Adley Rutschman,BAL,138,595,515,135,28,1,20,78,88,73,108,2,3,1,.262,.355,.450,.805,28442,668939
Marcus Semien,TEX,152,675,605,155,30,2,22,95,78,56,128,5,10,3,.256,.320,.430,.750,12532,543760
```

Update `tests/fixtures/steamer_pitchers.csv` — add `MLBAMID` column with values: Gerrit Cole=543037, Corbin Burnes=669203, Emmanuel Clase=661403.

Update `tests/fixtures/zips_pitchers.csv` — same MLBAMID values.

- [ ] **Step 4: Add MLBAMID to fangraphs.py column maps**

In `src/fantasy_baseball/data/fangraphs.py`, add to `HITTING_COLUMN_MAP`:
```python
"MLBAMID": "mlbam_id",
```

Add to `PITCHING_COLUMN_MAP`:
```python
"MLBAMID": "mlbam_id",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_data/test_projections.py::TestBlendProjections::test_blend_preserves_mlbam_id_in_metadata -v`
Expected: PASS

- [ ] **Step 6: Run full projection test suite for regressions**

Run: `pytest tests/test_data/test_projections.py -v`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/data/fangraphs.py tests/fixtures/*.csv tests/test_data/test_projections.py
git commit -m "feat: map MLBAMID column in fangraphs CSV parsing"
```

---

### Task 2: Add get_season_totals() to db.py

**Files:**
- Modify: `src/fantasy_baseball/data/db.py`
- Test: `tests/test_data/test_db.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_data/test_db.py`:

```python
class TestGetSeasonTotals:
    def test_returns_hitter_totals_by_mlbam_id(self):
        from fantasy_baseball.data.db import get_connection, create_tables, get_season_totals
        conn = get_connection(":memory:")
        create_tables(conn)
        # Insert two games for the same hitter
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, team, player_type, date, "
            "pa, ab, h, r, hr, rbi, sb) VALUES (2026, 592450, 'Aaron Judge', 'NYY', "
            "'hitter', '2026-03-27', 5, 4, 2, 1, 1, 3, 0)"
        )
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, team, player_type, date, "
            "pa, ab, h, r, hr, rbi, sb) VALUES (2026, 592450, 'Aaron Judge', 'NYY', "
            "'hitter', '2026-03-28', 4, 3, 1, 2, 0, 1, 1)"
        )
        conn.commit()
        hitter_totals, pitcher_totals = get_season_totals(conn, 2026)
        assert 592450 in hitter_totals
        t = hitter_totals[592450]
        assert t["pa"] == 9
        assert t["ab"] == 7
        assert t["h"] == 3
        assert t["r"] == 3
        assert t["hr"] == 1
        assert t["rbi"] == 4
        assert t["sb"] == 1
        assert len(pitcher_totals) == 0

    def test_returns_pitcher_totals_by_mlbam_id(self):
        from fantasy_baseball.data.db import get_connection, create_tables, get_season_totals
        conn = get_connection(":memory:")
        create_tables(conn)
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, team, player_type, date, "
            "ip, k, er, bb, h_allowed, w, sv, gs) VALUES (2026, 543037, 'Gerrit Cole', "
            "'NYY', 'pitcher', '2026-03-27', 7.0, 9, 2, 1, 5, 1, 0, 1)"
        )
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, team, player_type, date, "
            "ip, k, er, bb, h_allowed, w, sv, gs) VALUES (2026, 543037, 'Gerrit Cole', "
            "'NYY', 'pitcher', '2026-03-31', 6.0, 7, 3, 2, 4, 0, 0, 1)"
        )
        conn.commit()
        hitter_totals, pitcher_totals = get_season_totals(conn, 2026)
        assert len(hitter_totals) == 0
        assert 543037 in pitcher_totals
        t = pitcher_totals[543037]
        assert t["ip"] == 13.0
        assert t["k"] == 16
        assert t["er"] == 5
        assert t["bb"] == 3
        assert t["h_allowed"] == 9
        assert t["w"] == 1
        assert t["sv"] == 0

    def test_empty_when_no_data(self):
        from fantasy_baseball.data.db import get_connection, create_tables, get_season_totals
        conn = get_connection(":memory:")
        create_tables(conn)
        hitter_totals, pitcher_totals = get_season_totals(conn, 2026)
        assert hitter_totals == {}
        assert pitcher_totals == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_db.py::TestGetSeasonTotals -v`
Expected: FAIL — `get_season_totals` not importable

- [ ] **Step 3: Implement get_season_totals**

Add to `src/fantasy_baseball/data/db.py`:

```python
def get_season_totals(
    conn, season: int,
) -> tuple[dict[int, dict], dict[int, dict]]:
    """Get accumulated season stats from game_logs, keyed by mlbam_id.

    Returns (hitter_totals, pitcher_totals) where each is
    {mlbam_id: {stat: value}}.
    """
    hitter_totals = {}
    rows = conn.execute(
        "SELECT mlbam_id, SUM(pa) as pa, SUM(ab) as ab, SUM(h) as h, "
        "SUM(r) as r, SUM(hr) as hr, SUM(rbi) as rbi, SUM(sb) as sb "
        "FROM game_logs WHERE season = ? AND player_type = 'hitter' "
        "GROUP BY mlbam_id", (season,)
    ).fetchall()
    for row in rows:
        hitter_totals[row["mlbam_id"]] = {
            "pa": row["pa"] or 0, "ab": row["ab"] or 0, "h": row["h"] or 0,
            "r": row["r"] or 0, "hr": row["hr"] or 0, "rbi": row["rbi"] or 0,
            "sb": row["sb"] or 0,
        }

    pitcher_totals = {}
    rows = conn.execute(
        "SELECT mlbam_id, SUM(ip) as ip, SUM(k) as k, SUM(w) as w, SUM(sv) as sv, "
        "SUM(er) as er, SUM(bb) as bb, SUM(h_allowed) as h_allowed "
        "FROM game_logs WHERE season = ? AND player_type = 'pitcher' "
        "GROUP BY mlbam_id", (season,)
    ).fetchall()
    for row in rows:
        pitcher_totals[row["mlbam_id"]] = {
            "ip": row["ip"] or 0, "k": row["k"] or 0, "w": row["w"] or 0,
            "sv": row["sv"] or 0, "er": row["er"] or 0, "bb": row["bb"] or 0,
            "h_allowed": row["h_allowed"] or 0,
        }

    return hitter_totals, pitcher_totals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_db.py::TestGetSeasonTotals -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/db.py tests/test_data/test_db.py
git commit -m "feat: add get_season_totals() for game log aggregation by mlbam_id"
```

---

### Task 3: Add normalize_ros_to_full_season()

**Files:**
- Modify: `src/fantasy_baseball/data/projections.py`
- Test: `tests/test_data/test_projections.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_data/test_projections.py`:

```python
import numpy as np


class TestNormalizeRosToFullSeason:
    def test_adds_hitter_actuals_to_remaining_games(self):
        from fantasy_baseball.data.projections import (
            normalize_ros_to_full_season, HITTING_COUNTING_COLS,
        )
        df = pd.DataFrame([{
            "name": "Aaron Judge", "mlbam_id": 592450, "player_type": "hitter",
            "pa": 400, "ab": 300, "h": 90, "r": 60, "hr": 25, "rbi": 65, "sb": 4,
        }])
        game_log_totals = {
            592450: {"pa": 100, "ab": 80, "h": 25, "r": 15, "hr": 5, "rbi": 15, "sb": 1},
        }
        result = normalize_ros_to_full_season(df, game_log_totals, "hitter")
        judge = result.iloc[0]
        assert judge["pa"] == 500
        assert judge["ab"] == 380
        assert judge["h"] == 115
        assert judge["r"] == 75
        assert judge["hr"] == 30
        assert judge["rbi"] == 80
        assert judge["sb"] == 5

    def test_adds_pitcher_actuals_to_remaining_games(self):
        from fantasy_baseball.data.projections import normalize_ros_to_full_season
        df = pd.DataFrame([{
            "name": "Gerrit Cole", "mlbam_id": 543037, "player_type": "pitcher",
            "ip": 170, "k": 190, "w": 12, "sv": 0, "er": 60, "bb": 40, "h_allowed": 130,
        }])
        game_log_totals = {
            543037: {"ip": 13, "k": 16, "w": 1, "sv": 0, "er": 5, "bb": 3, "h_allowed": 9},
        }
        result = normalize_ros_to_full_season(df, game_log_totals, "pitcher")
        cole = result.iloc[0]
        assert cole["ip"] == 183
        assert cole["k"] == 206
        assert cole["w"] == 13
        assert cole["er"] == 65
        assert cole["bb"] == 43
        assert cole["h_allowed"] == 139

    def test_no_game_log_leaves_player_unchanged(self):
        from fantasy_baseball.data.projections import normalize_ros_to_full_season
        df = pd.DataFrame([{
            "name": "Rookie Player", "mlbam_id": 999999, "player_type": "hitter",
            "pa": 400, "ab": 300, "h": 90, "r": 60, "hr": 25, "rbi": 65, "sb": 4,
        }])
        result = normalize_ros_to_full_season(df, {}, "hitter")
        assert result.iloc[0]["pa"] == 400
        assert result.iloc[0]["hr"] == 25

    def test_missing_mlbam_id_leaves_player_unchanged(self):
        from fantasy_baseball.data.projections import normalize_ros_to_full_season
        df = pd.DataFrame([{
            "name": "Aaron Judge", "player_type": "hitter",
            "pa": 400, "ab": 300, "h": 90, "r": 60, "hr": 25, "rbi": 65, "sb": 4,
        }])
        game_log_totals = {
            592450: {"pa": 100, "ab": 80, "h": 25, "r": 15, "hr": 5, "rbi": 15, "sb": 1},
        }
        result = normalize_ros_to_full_season(df, game_log_totals, "hitter")
        # No mlbam_id column → can't match → unchanged
        assert result.iloc[0]["pa"] == 400

    def test_does_not_mutate_input_dataframe(self):
        from fantasy_baseball.data.projections import normalize_ros_to_full_season
        df = pd.DataFrame([{
            "name": "Aaron Judge", "mlbam_id": 592450, "player_type": "hitter",
            "pa": 400, "ab": 300, "h": 90, "r": 60, "hr": 25, "rbi": 65, "sb": 4,
        }])
        game_log_totals = {
            592450: {"pa": 100, "ab": 80, "h": 25, "r": 15, "hr": 5, "rbi": 15, "sb": 1},
        }
        normalize_ros_to_full_season(df, game_log_totals, "hitter")
        assert df.iloc[0]["pa"] == 400  # original unchanged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_projections.py::TestNormalizeRosToFullSeason -v`
Expected: FAIL — `normalize_ros_to_full_season` not importable

- [ ] **Step 3: Add constant and implement normalize_ros_to_full_season**

Add to `src/fantasy_baseball/data/projections.py` (after the existing constants):

```python
# ROS projection systems that produce full-season-updated projections.
# Systems NOT in this set produce remaining-games-only projections and
# need actual accumulated stats added before cross-system blending.
FULL_SEASON_ROS_SYSTEMS: set[str] = {"steamer", "the-bat-x"}


def normalize_ros_to_full_season(
    df: pd.DataFrame,
    game_log_totals: dict[int, dict],
    player_type: str,
) -> pd.DataFrame:
    """Add actual accumulated stats to remaining-games ROS projections.

    For each player with a matching mlbam_id in game_log_totals, adds the
    actual counting stats to the ROS counting stats so the result represents
    a full-season projection. Players without a match are left unchanged.

    Returns a new DataFrame (does not mutate the input).
    """
    if not game_log_totals or "mlbam_id" not in df.columns:
        return df.copy()

    result = df.copy()
    counting_cols = HITTING_COUNTING_COLS if player_type == "hitter" else PITCHING_COUNTING_COLS

    for idx, row in result.iterrows():
        mid = row.get("mlbam_id")
        if pd.isna(mid):
            continue
        mid = int(mid)
        actuals = game_log_totals.get(mid)
        if actuals is None:
            continue
        for col in counting_cols:
            if col in result.columns and col in actuals:
                result.at[idx, col] = row[col] + actuals[col]

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_projections.py::TestNormalizeRosToFullSeason -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/projections.py tests/test_data/test_projections.py
git commit -m "feat: add normalize_ros_to_full_season() for remaining-games ROS projections"
```

---

### Task 4: Wire normalizer into blend_projections and load_ros_projections

**Files:**
- Modify: `src/fantasy_baseball/data/projections.py:80-157` (blend_projections)
- Modify: `src/fantasy_baseball/data/db.py:660-710` (load_ros_projections)
- Test: `tests/test_data/test_projections.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_data/test_projections.py`:

```python
class TestBlendWithNormalizer:
    def test_normalizer_called_for_each_system(self, fixtures_dir):
        """Normalizer callback is invoked per system with correct args."""
        calls = []

        def track_normalizer(system_name, hitters_df, pitchers_df):
            calls.append(system_name)
            return hitters_df, pitchers_df

        blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
            normalizer=track_normalizer,
        )
        assert "steamer" in calls
        assert "zips" in calls

    def test_normalizer_modifies_counting_stats_before_blend(self, fixtures_dir):
        """When normalizer bumps a system's stats, the blend reflects it."""
        def bump_zips_hr(system_name, hitters_df, pitchers_df):
            if system_name == "zips":
                hitters_df = hitters_df.copy()
                hitters_df["hr"] = hitters_df["hr"] + 10
            return hitters_df, pitchers_df

        baseline, _, _ = blend_projections(fixtures_dir, systems=["steamer", "zips"])
        bumped, _, _ = blend_projections(
            fixtures_dir, systems=["steamer", "zips"], normalizer=bump_zips_hr,
        )

        judge_base = baseline[baseline["name"] == "Aaron Judge"].iloc[0]["hr"]
        judge_bump = bumped[bumped["name"] == "Aaron Judge"].iloc[0]["hr"]
        # ZiPS HR went up by 10, weight is 0.5 each → blend goes up by 5
        assert judge_bump == pytest.approx(judge_base + 5, abs=0.1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_projections.py::TestBlendWithNormalizer -v`
Expected: FAIL — `blend_projections()` doesn't accept `normalizer` kwarg

- [ ] **Step 3: Add normalizer parameter to blend_projections**

In `src/fantasy_baseball/data/projections.py`, modify the `blend_projections` signature:

```python
def blend_projections(
    projections_dir: Path,
    systems: list[str],
    weights: dict[str, float] | None = None,
    roster_names: set[str] | None = None,
    progress_cb=None,
    normalizer=None,
) -> tuple[pd.DataFrame, pd.DataFrame, "QualityReport | None"]:
```

Add normalizer call inside the system loading loop, after `load_projection_set` and before weight/system tagging. Change:

```python
    for system in systems:
        hitters, pitchers = load_projection_set(projections_dir, system)
        system_dfs[system] = (hitters, pitchers)
```

to:

```python
    for system in systems:
        hitters, pitchers = load_projection_set(projections_dir, system)
        if normalizer is not None:
            hitters, pitchers = normalizer(system, hitters, pitchers)
        system_dfs[system] = (hitters, pitchers)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_projections.py::TestBlendWithNormalizer -v`
Expected: PASS

- [ ] **Step 5: Run full projection test suite for regressions**

Run: `pytest tests/test_data/test_projections.py -v`
Expected: All tests pass (normalizer=None preserves existing behavior)

- [ ] **Step 6: Update load_ros_projections to pass normalizer**

In `src/fantasy_baseball/data/db.py`, modify `load_ros_projections` signature:

```python
def load_ros_projections(
    conn,
    projections_dir,
    systems: list[str],
    weights: dict[str, float] | None = None,
    roster_names: set[str] | None = None,
    progress_cb=None,
) -> None:
```

Add game log loading and normalizer construction at the top of the function body, after `projections_dir = Path(projections_dir)`:

```python
    from fantasy_baseball.data.projections import (
        FULL_SEASON_ROS_SYSTEMS, normalize_ros_to_full_season,
    )
    from datetime import date

    # Load actual accumulated stats for normalizing remaining-games systems
    hitter_totals, pitcher_totals = get_season_totals(conn, date.today().year)

    def _normalizer(system_name, hitters_df, pitchers_df):
        if system_name.lower() in FULL_SEASON_ROS_SYSTEMS:
            return hitters_df, pitchers_df
        if progress_cb:
            progress_cb(f"Normalizing {system_name} ROS → full-season")
        h = normalize_ros_to_full_season(hitters_df, hitter_totals, "hitter")
        p = normalize_ros_to_full_season(pitchers_df, pitcher_totals, "pitcher")
        return h, p
```

Then pass `normalizer=_normalizer` to the `blend_projections` call inside the loop:

```python
                hitters_df, pitchers_df, _ = blend_projections(
                    date_dir, systems, weights,
                    roster_names=roster_names, progress_cb=progress_cb,
                    normalizer=_normalizer,
                )
```

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/data/projections.py src/fantasy_baseball/data/db.py tests/test_data/test_projections.py
git commit -m "feat: wire ROS normalization into blend pipeline via normalizer callback"
```

---

### Task 5: Add blend_standings() to leverage.py

**Files:**
- Modify: `src/fantasy_baseball/lineup/leverage.py`
- Test: `tests/test_lineup/test_leverage.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_lineup/test_leverage.py`:

```python
from fantasy_baseball.lineup.leverage import blend_standings


class TestBlendStandings:
    def _make_current(self):
        return [
            {"name": "Team A", "stats": {"R": 200, "HR": 50, "RBI": 180, "SB": 30,
             "AVG": 0.260, "W": 20, "K": 300, "SV": 15, "ERA": 4.00, "WHIP": 1.30}},
            {"name": "Team B", "stats": {"R": 180, "HR": 45, "RBI": 170, "SB": 40,
             "AVG": 0.270, "W": 18, "K": 280, "SV": 12, "ERA": 3.80, "WHIP": 1.25}},
        ]

    def _make_projected(self):
        return [
            {"name": "Team A", "stats": {"R": 800, "HR": 200, "RBI": 720, "SB": 100,
             "AVG": 0.265, "W": 80, "K": 1200, "SV": 60, "ERA": 3.80, "WHIP": 1.22}},
            {"name": "Team B", "stats": {"R": 780, "HR": 210, "RBI": 700, "SB": 120,
             "AVG": 0.272, "W": 75, "K": 1150, "SV": 55, "ERA": 3.60, "WHIP": 1.20}},
        ]

    def test_progress_zero_returns_projected(self):
        blended = blend_standings(self._make_current(), self._make_projected(), 0.0)
        team_a = next(t for t in blended if t["name"] == "Team A")
        assert team_a["stats"]["R"] == pytest.approx(800)
        assert team_a["stats"]["AVG"] == pytest.approx(0.265)

    def test_progress_one_returns_current(self):
        blended = blend_standings(self._make_current(), self._make_projected(), 1.0)
        team_a = next(t for t in blended if t["name"] == "Team A")
        assert team_a["stats"]["R"] == pytest.approx(200)
        assert team_a["stats"]["AVG"] == pytest.approx(0.260)

    def test_progress_half_interpolates(self):
        blended = blend_standings(self._make_current(), self._make_projected(), 0.5)
        team_a = next(t for t in blended if t["name"] == "Team A")
        assert team_a["stats"]["R"] == pytest.approx(500)  # (200+800)/2
        assert team_a["stats"]["AVG"] == pytest.approx(0.2625)  # (0.260+0.265)/2

    def test_teams_matched_by_name(self):
        current = self._make_current()
        projected = list(reversed(self._make_projected()))  # reverse order
        blended = blend_standings(current, projected, 0.0)
        team_a = next(t for t in blended if t["name"] == "Team A")
        assert team_a["stats"]["R"] == pytest.approx(800)  # matched correctly

    def test_team_only_in_current_included_as_is(self):
        current = self._make_current() + [
            {"name": "Team C", "stats": {"R": 100, "HR": 20, "RBI": 90, "SB": 10,
             "AVG": 0.240, "W": 10, "K": 150, "SV": 5, "ERA": 4.50, "WHIP": 1.40}},
        ]
        blended = blend_standings(current, self._make_projected(), 0.5)
        team_c = next(t for t in blended if t["name"] == "Team C")
        assert team_c["stats"]["R"] == 100  # no projected match, kept as-is
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lineup/test_leverage.py::TestBlendStandings -v`
Expected: FAIL — `blend_standings` not importable

- [ ] **Step 3: Implement blend_standings**

Add to `src/fantasy_baseball/lineup/leverage.py`:

```python
def blend_standings(
    current: list[dict],
    projected: list[dict],
    progress: float,
) -> list[dict]:
    """Blend current and projected standings based on season progress.

    For each stat: blended = progress * current + (1 - progress) * projected.
    At progress=0.0, result is fully projected. At progress=1.0, fully current.

    Teams matched by name. Teams appearing in only one list are included as-is.
    """
    proj_by_name = {t["name"]: t for t in projected}
    seen_names = set()
    blended = []

    for team in current:
        name = team["name"]
        seen_names.add(name)
        proj_team = proj_by_name.get(name)
        if proj_team is None:
            blended.append(team)
            continue

        blended_stats = {}
        for cat in team["stats"]:
            cur_val = team["stats"].get(cat, 0)
            proj_val = proj_team["stats"].get(cat, 0)
            blended_stats[cat] = progress * cur_val + (1.0 - progress) * proj_val

        blended.append({
            **team,
            "stats": blended_stats,
        })

    # Include projected-only teams
    for team in projected:
        if team["name"] not in seen_names:
            blended.append(team)

    return blended
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lineup/test_leverage.py::TestBlendStandings -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/leverage.py tests/test_lineup/test_leverage.py
git commit -m "feat: add blend_standings() for interpolating current and projected standings"
```

---

### Task 6: Update calculate_leverage for projected standings

**Files:**
- Modify: `src/fantasy_baseball/lineup/leverage.py:51-161` (calculate_leverage)
- Test: `tests/test_lineup/test_leverage.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_lineup/test_leverage.py`:

```python
class TestCalculateLeverageWithProjected:
    def _make_projected(self):
        """Projected standings where SB gaps are large but HR gaps are tiny."""
        return [
            {"name": "Team 4", "rank": 4, "stats": {"R": 780, "HR": 201, "RBI": 720, "SB": 200, "AVG": 0.268, "W": 78, "K": 1200, "SV": 72, "ERA": 3.65, "WHIP": 1.21}},
            {"name": "User Team", "rank": 5, "stats": {"R": 760, "HR": 200, "RBI": 700, "SB": 100, "AVG": 0.265, "W": 75, "K": 1180, "SV": 65, "ERA": 3.75, "WHIP": 1.24}},
            {"name": "Team 6", "rank": 6, "stats": {"R": 720, "HR": 199, "RBI": 680, "SB": 80, "AVG": 0.260, "W": 70, "K": 1150, "SV": 60, "ERA": 3.90, "WHIP": 1.27}},
        ]

    def test_projected_standings_override_uniform_ramp(self):
        """At season_progress=0 with projected standings, leverage is NOT uniform."""
        standings = _make_standings()
        projected = self._make_projected()
        leverage = calculate_leverage(
            standings, "User Team",
            season_progress=0.0, projected_standings=projected,
        )
        # Should NOT be uniform — projected gaps matter
        values = list(leverage.values())
        assert max(values) - min(values) > 0.01

    def test_projected_tiny_hr_gap_gets_high_leverage(self):
        """HR gap is 1 in projected standings → high HR leverage."""
        standings = _make_standings()
        projected = self._make_projected()
        leverage = calculate_leverage(
            standings, "User Team",
            season_progress=0.0, projected_standings=projected,
        )
        # HR gaps are tiny (201 vs 200 vs 199) so HR should be high leverage
        # SB gaps are huge (200 vs 100 vs 80) so SB should be low
        assert leverage["HR"] > leverage["SB"]

    def test_no_projected_preserves_existing_behavior(self):
        """Without projected_standings, behavior is unchanged (uniform ramp)."""
        standings = _make_standings()
        leverage_old = calculate_leverage(
            standings, "User Team", season_progress=0.0,
        )
        leverage_new = calculate_leverage(
            standings, "User Team", season_progress=0.0, projected_standings=None,
        )
        for cat in leverage_old:
            assert leverage_old[cat] == pytest.approx(leverage_new[cat])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lineup/test_leverage.py::TestCalculateLeverageWithProjected -v`
Expected: FAIL — `calculate_leverage()` doesn't accept `projected_standings` kwarg

- [ ] **Step 3: Update calculate_leverage**

In `src/fantasy_baseball/lineup/leverage.py`, modify `calculate_leverage` signature:

```python
def calculate_leverage(
    standings: list[dict],
    user_team_name: str,
    *,
    attack_weight: float = 0.6,
    defense_weight: float = 0.4,
    season_progress: float | None = None,
    projected_standings: list[dict] | None = None,
) -> dict[str, float]:
```

At the end of the function, replace the existing blend logic (lines 154-161):

```python
    # Blend standings-based leverage with uniform weights based on season progress.
    # Early season: mostly uniform (standings are noise).
    # Late season: fully standings-driven.
    uniform = 1.0 / len(ALL_CATEGORIES)
    return {
        cat: season_progress * standings_leverage[cat] + (1.0 - season_progress) * uniform
        for cat in ALL_CATEGORIES
    }
```

with:

```python
    if projected_standings is not None:
        # Blend current standings with projected, then use full standings-based leverage.
        # The blend itself handles the early/late season weighting — no uniform ramp needed.
        blended = blend_standings(standings, projected_standings, season_progress)

        # Recompute leverage from blended standings (reuse same logic above)
        blended_sorted = sorted(blended, key=lambda t: t.get("rank", 99))
        blended_user = None
        blended_idx = None
        for i, team in enumerate(blended_sorted):
            if team["name"] == user_team_name:
                blended_user = team
                blended_idx = i
                break

        if blended_user is None:
            return {cat: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}

        blended_stats = blended_user.get("stats", {})
        b_above = blended_sorted[blended_idx - 1] if blended_idx > 0 else None
        b_below = (
            blended_sorted[blended_idx + 1]
            if blended_idx < len(blended_sorted) - 1
            else None
        )

        if b_above is not None and b_below is not None:
            bw_attack, bw_defense = attack_weight, defense_weight
        elif b_above is not None:
            bw_attack, bw_defense = 1.0, 0.0
        elif b_below is not None:
            bw_attack, bw_defense = 0.0, 1.0
        else:
            return {cat: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}

        b_above_stats = b_above.get("stats", {}) if b_above else {}
        b_below_stats = b_below.get("stats", {}) if b_below else {}

        blended_raw: dict[str, float] = {}
        for cat in ALL_CATEGORIES:
            bval = blended_stats.get(cat, 0)
            lev = 0.0
            if b_above is not None:
                gap = _gap_for_category(cat, bval, b_above_stats.get(cat, 0))
                lev += bw_attack * (1.0 / (gap + epsilon))
            if b_below is not None:
                gap = _gap_for_category(cat, bval, b_below_stats.get(cat, 0))
                lev += bw_defense * (1.0 / (gap + epsilon))
            blended_raw[cat] = lev

        if blended_raw:
            med = statistics.median(blended_raw.values())
            cap = med * MAX_MEANINGFUL_GAP_MULTIPLIER
            if cap > 0:
                blended_raw = {cat: min(val, cap) for cat, val in blended_raw.items()}

        total = sum(blended_raw.values())
        if total > 0:
            return {cat: val / total for cat, val in blended_raw.items()}
        return {cat: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}

    # Fallback: blend standings-based leverage with uniform weights.
    uniform = 1.0 / len(ALL_CATEGORIES)
    return {
        cat: season_progress * standings_leverage[cat] + (1.0 - season_progress) * uniform
        for cat in ALL_CATEGORIES
    }
```

- [ ] **Step 4: Run all leverage tests**

Run: `pytest tests/test_lineup/test_leverage.py -v`
Expected: All tests pass (existing + new)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/leverage.py tests/test_lineup/test_leverage.py
git commit -m "feat: calculate_leverage accepts projected_standings for blended leverage"
```

---

### Task 7: Reorganize season_data.py pipeline

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`

This task reorganizes the pipeline so opponent rosters are fetched before leverage, and projected standings feed into leverage. No new tests — this wires together already-tested components. Verified by running the full test suite.

- [ ] **Step 1: Move opponent roster fetching to before leverage (new Step 4b)**

In `src/fantasy_baseball/web/season_data.py`, after Step 4 (Load projections, ~line 736) and before Step 5 (Leverage), insert the opponent roster fetching code currently in Step 11 (lines 897-930). The new section:

```python
        # --- Step 4b: Fetch opponent rosters ---
        _progress("Fetching opponent rosters...")
        from fantasy_baseball.data.projections import match_roster_to_projections

        opp_rosters: dict[str, list[dict]] = {}
        all_raw_rosters = {config.team_name: roster_raw}

        def _fetch_opp(key_and_info):
            key, team_info = key_and_info
            tname = team_info.get("name", "")
            try:
                opp_raw = fetch_roster(league, key)
                opp_proj_list = match_roster_to_projections(
                    opp_raw, hitters_proj, pitchers_proj
                )
                return (tname, opp_raw, opp_proj_list)
            except Exception:
                return None

        opp_items = [
            (key, info) for key, info in teams.items()
            if info.get("name", "") != config.team_name and key != user_team_key
        ]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for result in pool.map(_fetch_opp, opp_items):
                if result is None:
                    continue
                tname, opp_raw, opp_proj_list = result
                all_raw_rosters[tname] = opp_raw
                if opp_proj_list:
                    opp_rosters[tname] = opp_proj_list
        _progress(f"Fetched rosters for {len(opp_rosters)} opponents")
```

Also move the `match_roster_to_projections` import from Step 6 to Step 4b (or the top of the function) since it's now used earlier. Remove the duplicate import from Step 6 if present.

- [ ] **Step 2: Build projected standings and match user roster (new Step 4c)**

After Step 4b, add:

```python
        # --- Step 4c: Build projected standings ---
        _progress("Projecting end-of-season standings...")
        from fantasy_baseball.scoring import project_team_stats

        # Match user roster to projections (wSGP added later after leverage)
        matched = match_roster_to_projections(roster_raw, hitters_proj, pitchers_proj)

        all_team_rosters = {config.team_name: matched}
        all_team_rosters.update(opp_rosters)

        projected_standings = []
        for tname, roster in all_team_rosters.items():
            proj_stats = project_team_stats(roster)
            projected_standings.append({
                "name": tname,
                "team_key": "",
                "rank": 0,
                "stats": proj_stats,
            })
        _progress(f"Projected standings for {len(projected_standings)} teams")
```

- [ ] **Step 3: Update Step 5 to pass projected standings to leverage**

Change:

```python
        leverage = calculate_leverage(standings, config.team_name)
```

to:

```python
        leverage = calculate_leverage(
            standings, config.team_name,
            projected_standings=projected_standings,
        )
```

- [ ] **Step 4: Update Step 6 to reuse matched roster from Step 4c**

Step 6 currently does `matched = match_roster_to_projections(...)`. Since we already did this in Step 4c, remove the duplicate call. The rest of Step 6 (ROS lookup, wSGP computation, unmatched players) stays the same but uses the `matched` variable from Step 4c.

Remove from Step 6:
```python
        from fantasy_baseball.data.projections import match_roster_to_projections
        matched = match_roster_to_projections(roster_raw, hitters_proj, pitchers_proj)
```

(These are now in Step 4b/4c.)

- [ ] **Step 5: Update Step 11 to reuse already-fetched opponent rosters**

Remove the opponent roster fetching code from Step 11 (the `_fetch_opp` function, the `opp_items` list, the `ThreadPoolExecutor` block, and `opp_rosters`/`all_raw_rosters` initialization). These now live in Step 4b.

Keep the trade evaluation code that uses `opp_rosters` and `all_raw_rosters` — these variables are now defined earlier.

Update `leverage_by_team` computation in Step 11 to also use projected standings:

```python
        leverage_by_team: dict[str, dict] = {}
        for team in standings:
            tname = team["name"]
            leverage_by_team[tname] = calculate_leverage(
                standings, tname, projected_standings=projected_standings,
            )
```

- [ ] **Step 6: Remove old Step 12 (projected standings) — now in Step 4c**

The old Step 12 code that built `projected_standings` is now in Step 4c. Remove it. Keep the `write_cache("projections", ...)` call — move it right after Step 4c:

```python
        write_cache("projections", {"projected_standings": projected_standings}, cache_dir)
```

- [ ] **Step 7: Run existing tests**

Run: `pytest tests/test_web/ -v`
Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "refactor: reorder season_data pipeline — fetch opponents before leverage"
```

---

### Task 8: Update summary.py and run_lineup.py

**Files:**
- Modify: `scripts/summary.py`
- Modify: `scripts/run_lineup.py`

- [ ] **Step 1: Update summary.py**

In `scripts/summary.py`, the leverage calculation at ~line 207 currently uses current standings:

```python
    if standings:
        leverage = calculate_leverage(standings, team_name)
```

Replace with:

```python
    # Build projected standings for leverage
    projected_standings = [
        {"name": name, "stats": all_stats[name]} for name in all_stats
    ]

    if standings:
        leverage = calculate_leverage(
            standings, team_name, projected_standings=projected_standings,
        )
```

Also update the trade `leverage_by_team` loop at ~line 399:

```python
        leverage_by_team = {}
        for team in trade_standings:
            leverage_by_team[team["name"]] = calculate_leverage(
                trade_standings, team["name"],
                projected_standings=projected_standings,
            )
```

- [ ] **Step 2: Update run_lineup.py**

In `scripts/run_lineup.py`, after loading standings (~line 204), add a cache read:

```python
    # Read cached projected standings from dashboard (if available)
    projected_standings = None
    projections_cache = Path(PROJECT_ROOT / "data" / "cache" / "projections.json")
    if projections_cache.exists():
        try:
            import json
            cached = json.loads(projections_cache.read_text(encoding="utf-8"))
            projected_standings = cached.get("projected_standings")
            if projected_standings:
                print(f"Loaded cached projected standings ({len(projected_standings)} teams)")
        except Exception:
            pass
```

Then update the leverage call at ~line 244:

```python
    leverage = calculate_leverage(
        standings, config.team_name,
        projected_standings=projected_standings,
    )
```

- [ ] **Step 3: Run existing tests**

Run: `pytest -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add scripts/summary.py scripts/run_lineup.py
git commit -m "feat: pass projected standings to leverage in summary.py and run_lineup.py"
```
