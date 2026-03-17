# Shared Foundation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared foundation — auth, data ingestion, projection blending, and SGP engine — that the draft assistant and lineup optimizer both depend on.

**Architecture:** Python package `fantasy_baseball` under `src/` with four modules: `utils` (pure constants/helpers), `auth` (Yahoo OAuth2), `data` (FanGraphs CSV parsing + projection blending), and `sgp` (SGP denominators, replacement levels, player values). Each module has clear boundaries and can be tested independently.

**Tech Stack:** Python 3.11+, pandas, numpy, yahoo-fantasy-api, yahoo-oauth, pytest

---

## Chunk 1: Project Setup + Utils + Data Layer

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/fantasy_baseball/__init__.py`
- Create: `config/league.yaml.example`
- Create: `data/projections/.gitkeep`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fantasy-baseball"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "yahoo-fantasy-api>=2.12",
    "yahoo-oauth>=2.0",
    "pandas>=2.0",
    "numpy>=1.24",
    "scipy>=1.11",
    "requests>=2.31",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/

# Config secrets
config/oauth.json

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 3: Create directory structure and __init__.py files**

Create these empty files:
- `src/fantasy_baseball/__init__.py`
- `src/fantasy_baseball/auth/__init__.py`
- `src/fantasy_baseball/data/__init__.py`
- `src/fantasy_baseball/sgp/__init__.py`
- `src/fantasy_baseball/utils/__init__.py`
- `tests/__init__.py`
- `tests/test_utils/__init__.py`
- `tests/test_data/__init__.py`
- `tests/test_sgp/__init__.py`
- `tests/fixtures/.gitkeep`
- `data/projections/.gitkeep`

- [ ] **Step 4: Create config template**

Create `config/league.yaml.example`:
```yaml
league:
  id: 5652
  num_teams: 10
  game_code: mlb

draft:
  position: 8  # Snake draft pick position (1-indexed)

keepers: []
  # - name: "Player Name"
  #   team: "Fantasy Team Name"

roster_slots:
  C: 1
  1B: 1
  2B: 1
  3B: 1
  SS: 1
  IF: 1
  OF: 4
  UTIL: 2
  P: 9
  BN: 2
  IL: 2

projections:
  systems:
    - steamer
    - zips
  weights:  # Optional, defaults to equal weighting
    steamer: 0.5
    zips: 0.5

sgp_denominators:  # Override defaults if desired
  R: 20
  HR: 9
  RBI: 20
  SB: 8
  AVG: 0.005
  W: 3
  K: 30
  ERA: 0.15
  WHIP: 0.015
  SV: 7
```

- [ ] **Step 5: Create tests/conftest.py**

```python
import pytest
from pathlib import Path


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"
```

- [ ] **Step 6: Install the project in dev mode and verify pytest runs**

Run: `pip install -e ".[dev]"`
Then: `pytest --co`
Expected: "no tests ran" (collected 0 items), no import errors

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with pyproject.toml, directory structure, and config template"
```

---

### Task 2: Utils — Constants

**Files:**
- Create: `src/fantasy_baseball/utils/constants.py`
- Create: `tests/test_utils/test_constants.py`

- [ ] **Step 1: Write the test**

Create `tests/test_utils/test_constants.py`:
```python
from fantasy_baseball.utils.constants import (
    HITTING_CATEGORIES,
    PITCHING_CATEGORIES,
    ALL_CATEGORIES,
    RATE_STATS,
    INVERSE_STATS,
    ROSTER_SLOTS,
    STARTERS_PER_POSITION,
    DEFAULT_SGP_DENOMINATORS,
    IF_ELIGIBLE,
    NUM_TEAMS,
)


def test_hitting_categories():
    assert HITTING_CATEGORIES == ["R", "HR", "RBI", "SB", "AVG"]


def test_pitching_categories():
    assert PITCHING_CATEGORIES == ["W", "K", "ERA", "WHIP", "SV"]


def test_all_categories_is_union():
    assert ALL_CATEGORIES == HITTING_CATEGORIES + PITCHING_CATEGORIES
    assert len(ALL_CATEGORIES) == 10


def test_rate_stats():
    assert RATE_STATS == {"AVG", "ERA", "WHIP"}


def test_inverse_stats_subset_of_rate():
    assert INVERSE_STATS.issubset(RATE_STATS)
    assert INVERSE_STATS == {"ERA", "WHIP"}


def test_roster_slots_total():
    assert sum(ROSTER_SLOTS.values()) == 25


def test_starters_per_position_total():
    # C+1B+2B+3B+SS+IF = 60 hitter slots, OF=40, UTIL=20 -> 120 hitter, P=90 -> 210
    hitter_slots = sum(v for k, v in STARTERS_PER_POSITION.items() if k != "P")
    assert hitter_slots == 120
    assert STARTERS_PER_POSITION["P"] == 90


def test_sgp_denominators_cover_all_categories():
    assert set(DEFAULT_SGP_DENOMINATORS.keys()) == set(ALL_CATEGORIES)


def test_if_eligible_positions():
    assert IF_ELIGIBLE == {"1B", "2B", "3B", "SS"}


def test_num_teams():
    assert NUM_TEAMS == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils/test_constants.py -v`
Expected: FAIL (ImportError — module doesn't exist yet)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/utils/constants.py`:
```python
HITTING_CATEGORIES: list[str] = ["R", "HR", "RBI", "SB", "AVG"]
PITCHING_CATEGORIES: list[str] = ["W", "K", "ERA", "WHIP", "SV"]
ALL_CATEGORIES: list[str] = HITTING_CATEGORIES + PITCHING_CATEGORIES

RATE_STATS: set[str] = {"AVG", "ERA", "WHIP"}
INVERSE_STATS: set[str] = {"ERA", "WHIP"}  # Lower is better

ROSTER_SLOTS: dict[str, int] = {
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "IF": 1,
    "OF": 4,
    "UTIL": 2,
    "P": 9,
    "BN": 2,
    "IL": 2,
}

STARTERS_PER_POSITION: dict[str, int] = {
    "C": 10,
    "1B": 10,
    "2B": 10,
    "3B": 10,
    "SS": 10,
    "IF": 10,
    "OF": 40,
    "UTIL": 20,
    "P": 90,
}

IF_ELIGIBLE: set[str] = {"1B", "2B", "3B", "SS"}

DEFAULT_SGP_DENOMINATORS: dict[str, float] = {
    "R": 20.0,
    "HR": 9.0,
    "RBI": 20.0,
    "SB": 8.0,
    "AVG": 0.005,
    "W": 3.0,
    "K": 30.0,
    "ERA": 0.15,
    "WHIP": 0.015,
    "SV": 7.0,
}

NUM_TEAMS: int = 10
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_utils/test_constants.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/utils/constants.py tests/test_utils/test_constants.py
git commit -m "feat: add league constants, stat categories, and SGP denominators"
```

---

### Task 3: Utils — Position Helpers

**Files:**
- Create: `src/fantasy_baseball/utils/positions.py`
- Create: `tests/test_utils/test_positions.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_utils/test_positions.py`:
```python
import pytest
from fantasy_baseball.utils.positions import can_fill_slot, is_hitter, is_pitcher


class TestCanFillSlot:
    def test_catcher_fills_c(self):
        assert can_fill_slot(["C"], "C") is True

    def test_catcher_cannot_fill_1b(self):
        assert can_fill_slot(["C"], "1B") is False

    def test_shortstop_fills_if(self):
        assert can_fill_slot(["SS"], "IF") is True

    def test_catcher_cannot_fill_if(self):
        assert can_fill_slot(["C"], "IF") is False

    def test_outfielder_fills_of(self):
        assert can_fill_slot(["OF"], "OF") is True

    def test_any_hitter_fills_util(self):
        assert can_fill_slot(["C"], "UTIL") is True
        assert can_fill_slot(["OF"], "UTIL") is True
        assert can_fill_slot(["1B", "OF"], "UTIL") is True

    def test_pitcher_cannot_fill_util(self):
        assert can_fill_slot(["SP"], "UTIL") is False
        assert can_fill_slot(["RP"], "UTIL") is False

    def test_pitcher_fills_p(self):
        assert can_fill_slot(["SP"], "P") is True
        assert can_fill_slot(["RP"], "P") is True
        assert can_fill_slot(["P"], "P") is True

    def test_multi_position_player(self):
        assert can_fill_slot(["SS", "2B"], "SS") is True
        assert can_fill_slot(["SS", "2B"], "2B") is True
        assert can_fill_slot(["SS", "2B"], "IF") is True
        assert can_fill_slot(["SS", "2B"], "UTIL") is True
        assert can_fill_slot(["SS", "2B"], "OF") is False

    def test_any_hitter_cannot_fill_p(self):
        assert can_fill_slot(["1B"], "P") is False

    def test_bench_and_il_accept_anyone(self):
        assert can_fill_slot(["C"], "BN") is True
        assert can_fill_slot(["SP"], "BN") is True
        assert can_fill_slot(["OF"], "IL") is True
        assert can_fill_slot(["RP"], "IL") is True


class TestIsHitter:
    def test_catcher_is_hitter(self):
        assert is_hitter(["C"]) is True

    def test_outfielder_is_hitter(self):
        assert is_hitter(["OF"]) is True

    def test_pitcher_is_not_hitter(self):
        assert is_hitter(["SP"]) is False
        assert is_hitter(["RP"]) is False

    def test_two_way_player(self):
        # Ohtani-type: has both hitting and pitching positions
        assert is_hitter(["DH", "SP"]) is True


class TestIsPitcher:
    def test_sp_is_pitcher(self):
        assert is_pitcher(["SP"]) is True

    def test_rp_is_pitcher(self):
        assert is_pitcher(["RP"]) is True

    def test_hitter_is_not_pitcher(self):
        assert is_pitcher(["1B"]) is False

    def test_two_way_player(self):
        assert is_pitcher(["DH", "SP"]) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_utils/test_positions.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/utils/positions.py`:
```python
from .constants import IF_ELIGIBLE

HITTER_POSITIONS: set[str] = {"C", "1B", "2B", "3B", "SS", "OF", "DH", "IF"}
PITCHER_POSITIONS: set[str] = {"P", "SP", "RP"}


def can_fill_slot(player_positions: list[str], slot: str) -> bool:
    """Check if a player with given eligible positions can fill a roster slot."""
    if slot in ("BN", "IL"):
        return True
    if slot == "UTIL":
        return any(pos in HITTER_POSITIONS for pos in player_positions)
    if slot == "IF":
        return any(pos in IF_ELIGIBLE for pos in player_positions)
    if slot == "OF":
        return "OF" in player_positions
    if slot == "P":
        return any(pos in PITCHER_POSITIONS for pos in player_positions)
    return slot in player_positions


def is_hitter(positions: list[str]) -> bool:
    """Check if a player is a hitter based on their eligible positions."""
    return any(pos in HITTER_POSITIONS for pos in positions)


def is_pitcher(positions: list[str]) -> bool:
    """Check if a player is a pitcher based on their eligible positions."""
    return any(pos in PITCHER_POSITIONS for pos in positions)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_utils/test_positions.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/utils/positions.py tests/test_utils/test_positions.py
git commit -m "feat: add position eligibility helpers (can_fill_slot, is_hitter, is_pitcher)"
```

---

### Task 4: Data — FanGraphs CSV Parsing

**Files:**
- Create: `src/fantasy_baseball/data/fangraphs.py`
- Create: `tests/test_data/test_fangraphs.py`
- Create: `tests/fixtures/steamer_hitters.csv`
- Create: `tests/fixtures/steamer_pitchers.csv`

- [ ] **Step 1: Create test fixture CSVs**

Create `tests/fixtures/steamer_hitters.csv`:
```csv
Name,Team,G,PA,AB,H,2B,3B,HR,R,RBI,BB,SO,HBP,SB,CS,AVG,OBP,SLG,OPS,playerid
Aaron Judge,NYY,155,650,550,160,30,1,45,110,120,90,170,5,5,1,.291,.400,.580,.980,15640
Mookie Betts,LAD,145,620,540,155,35,3,28,105,85,70,100,4,15,3,.287,.370,.510,.880,13611
Adley Rutschman,BAL,140,600,520,140,30,1,22,80,90,75,110,2,2,1,.269,.360,.460,.820,28442
Marcus Semien,TEX,155,680,610,160,32,2,24,100,80,55,130,6,12,4,.262,.325,.440,.765,12532
```

Create `tests/fixtures/steamer_pitchers.csv`:
```csv
Name,Team,G,GS,IP,W,L,SV,HLD,ERA,WHIP,K/9,BB/9,HR/9,SO,BB,HR,ER,H,FIP,WAR,playerid
Gerrit Cole,NYY,32,32,200,15,7,0,0,3.15,1.05,10.80,2.50,1.00,240,56,22,70,154,3.10,5.2,13125
Emmanuel Clase,CLE,70,0,70,4,3,40,0,1.80,0.90,9.00,1.80,0.40,70,14,3,14,49,2.20,2.8,18498
Corbin Burnes,BAL,33,33,210,14,8,0,0,3.40,1.15,9.00,2.30,0.90,210,54,21,79,163,3.50,4.5,19361
```

- [ ] **Step 2: Write the tests**

Create `tests/test_data/test_fangraphs.py`:
```python
import pytest
import pandas as pd
from pathlib import Path
from fantasy_baseball.data.fangraphs import (
    parse_hitting_csv,
    parse_pitching_csv,
    load_projection_set,
)


class TestParseHittingCsv:
    def test_parses_standard_columns(self, fixtures_dir):
        df = parse_hitting_csv(fixtures_dir / "steamer_hitters.csv")
        assert "name" in df.columns
        assert "hr" in df.columns
        assert "r" in df.columns
        assert "rbi" in df.columns
        assert "sb" in df.columns
        assert "avg" in df.columns
        assert "ab" in df.columns
        assert "h" in df.columns

    def test_correct_row_count(self, fixtures_dir):
        df = parse_hitting_csv(fixtures_dir / "steamer_hitters.csv")
        assert len(df) == 4

    def test_player_type_set_to_hitter(self, fixtures_dir):
        df = parse_hitting_csv(fixtures_dir / "steamer_hitters.csv")
        assert (df["player_type"] == "hitter").all()

    def test_stat_values_correct(self, fixtures_dir):
        df = parse_hitting_csv(fixtures_dir / "steamer_hitters.csv")
        judge = df[df["name"] == "Aaron Judge"].iloc[0]
        assert judge["hr"] == 45
        assert judge["r"] == 110
        assert judge["rbi"] == 120
        assert judge["sb"] == 5
        assert judge["avg"] == pytest.approx(0.291, abs=0.001)

    def test_raises_on_missing_columns(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("Name,Team,G\nFoo,BAR,100\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            parse_hitting_csv(bad_csv)


class TestParsePitchingCsv:
    def test_parses_standard_columns(self, fixtures_dir):
        df = parse_pitching_csv(fixtures_dir / "steamer_pitchers.csv")
        assert "name" in df.columns
        assert "ip" in df.columns
        assert "w" in df.columns
        assert "k" in df.columns
        assert "era" in df.columns
        assert "whip" in df.columns
        assert "sv" in df.columns

    def test_correct_row_count(self, fixtures_dir):
        df = parse_pitching_csv(fixtures_dir / "steamer_pitchers.csv")
        assert len(df) == 3

    def test_player_type_set_to_pitcher(self, fixtures_dir):
        df = parse_pitching_csv(fixtures_dir / "steamer_pitchers.csv")
        assert (df["player_type"] == "pitcher").all()

    def test_strikeouts_mapped_from_SO(self, fixtures_dir):
        df = parse_pitching_csv(fixtures_dir / "steamer_pitchers.csv")
        cole = df[df["name"] == "Gerrit Cole"].iloc[0]
        assert cole["k"] == 240

    def test_earned_runs_available(self, fixtures_dir):
        df = parse_pitching_csv(fixtures_dir / "steamer_pitchers.csv")
        cole = df[df["name"] == "Gerrit Cole"].iloc[0]
        assert cole["er"] == 70

    def test_hits_allowed_mapped(self, fixtures_dir):
        df = parse_pitching_csv(fixtures_dir / "steamer_pitchers.csv")
        cole = df[df["name"] == "Gerrit Cole"].iloc[0]
        assert cole["h_allowed"] == 154


class TestLoadProjectionSet:
    def test_loads_matching_files(self, fixtures_dir):
        hitters, pitchers = load_projection_set(fixtures_dir, "steamer")
        assert len(hitters) == 4
        assert len(pitchers) == 3

    def test_returns_empty_for_missing_system(self, fixtures_dir):
        hitters, pitchers = load_projection_set(fixtures_dir, "nonexistent")
        assert hitters.empty
        assert pitchers.empty
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_data/test_fangraphs.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 4: Write the implementation**

Create `src/fantasy_baseball/data/fangraphs.py`:
```python
import pandas as pd
from pathlib import Path

HITTING_COLUMN_MAP: dict[str, str] = {
    "Name": "name",
    "Team": "team",
    "PA": "pa",
    "AB": "ab",
    "H": "h",
    "HR": "hr",
    "R": "r",
    "RBI": "rbi",
    "SB": "sb",
    "AVG": "avg",
    "playerid": "fg_id",
}

PITCHING_COLUMN_MAP: dict[str, str] = {
    "Name": "name",
    "Team": "team",
    "IP": "ip",
    "W": "w",
    "SO": "k",
    "ERA": "era",
    "WHIP": "whip",
    "SV": "sv",
    "ER": "er",
    "BB": "bb",
    "H": "h_allowed",
    "playerid": "fg_id",
}

REQUIRED_HITTING_COLS: list[str] = ["name", "ab", "h", "hr", "r", "rbi", "sb", "avg"]
REQUIRED_PITCHING_COLS: list[str] = ["name", "ip", "w", "k", "era", "whip", "sv"]


def parse_hitting_csv(filepath: Path) -> pd.DataFrame:
    """Parse a FanGraphs hitting projections CSV into normalized columns."""
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    rename = {k: v for k, v in HITTING_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED_HITTING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["player_type"] = "hitter"
    return df


def parse_pitching_csv(filepath: Path) -> pd.DataFrame:
    """Parse a FanGraphs pitching projections CSV into normalized columns."""
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    rename = {k: v for k, v in PITCHING_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED_PITCHING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["player_type"] = "pitcher"
    return df


def load_projection_set(
    projections_dir: Path, system_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a named projection system from the projections directory.

    Expects files named like: steamer_hitters.csv, steamer_pitchers.csv
    """
    hitting_file = projections_dir / f"{system_name}_hitters.csv"
    pitching_file = projections_dir / f"{system_name}_pitchers.csv"
    hitters = parse_hitting_csv(hitting_file) if hitting_file.exists() else pd.DataFrame()
    pitchers = parse_pitching_csv(pitching_file) if pitching_file.exists() else pd.DataFrame()
    return hitters, pitchers
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_data/test_fangraphs.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/data/fangraphs.py tests/test_data/test_fangraphs.py tests/fixtures/
git commit -m "feat: add FanGraphs CSV parsing with column normalization"
```

---

### Task 5: Data — Projection Blending

**Files:**
- Create: `src/fantasy_baseball/data/projections.py`
- Create: `tests/test_data/test_projections.py`
- Create: `tests/fixtures/zips_hitters.csv`
- Create: `tests/fixtures/zips_pitchers.csv`

- [ ] **Step 1: Create second projection fixture CSVs (ZiPS)**

Create `tests/fixtures/zips_hitters.csv`:
```csv
Name,Team,G,PA,AB,H,2B,3B,HR,R,RBI,BB,SO,HBP,SB,CS,AVG,OBP,SLG,OPS,playerid
Aaron Judge,NYY,150,640,545,155,28,1,42,105,115,85,175,5,4,1,.284,.395,.570,.965,15640
Mookie Betts,LAD,148,625,545,158,33,2,30,108,88,72,98,3,14,3,.290,.372,.520,.892,13611
Adley Rutschman,BAL,138,595,515,135,28,1,20,78,88,73,108,2,3,1,.262,.355,.450,.805,28442
Marcus Semien,TEX,152,675,605,155,30,2,22,95,78,56,128,5,10,3,.256,.320,.430,.750,12532
```

Create `tests/fixtures/zips_pitchers.csv`:
```csv
Name,Team,G,GS,IP,W,L,SV,HLD,ERA,WHIP,K/9,BB/9,HR/9,SO,BB,HR,ER,H,FIP,WAR,playerid
Gerrit Cole,NYY,31,31,195,14,8,0,0,3.30,1.08,10.50,2.60,1.05,228,57,23,72,154,3.25,4.8,13125
Emmanuel Clase,CLE,68,0,68,3,3,38,0,2.00,0.95,8.70,2.00,0.50,66,15,4,15,50,2.50,2.4,18498
Corbin Burnes,BAL,32,32,205,13,8,0,0,3.55,1.18,8.80,2.40,0.95,200,55,22,81,167,3.60,4.0,19361
```

- [ ] **Step 2: Write the tests**

Create `tests/test_data/test_projections.py`:
```python
import pytest
import pandas as pd
from pathlib import Path
from fantasy_baseball.data.projections import blend_projections


class TestBlendProjections:
    def test_blend_two_systems_equal_weight(self, fixtures_dir):
        hitters, pitchers = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
        )
        assert len(hitters) == 4
        assert len(pitchers) == 3

    def test_blended_counting_stats_are_averaged(self, fixtures_dir):
        hitters, pitchers = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
        )
        judge = hitters[hitters["name"] == "Aaron Judge"].iloc[0]
        # Steamer: 45 HR, ZiPS: 42 HR -> avg = 43.5
        assert judge["hr"] == pytest.approx(43.5)
        # Steamer: 110 R, ZiPS: 105 R -> avg = 107.5
        assert judge["r"] == pytest.approx(107.5)

    def test_blended_avg_recomputed_from_components(self, fixtures_dir):
        hitters, pitchers = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
        )
        judge = hitters[hitters["name"] == "Aaron Judge"].iloc[0]
        # Steamer: 160 H / 550 AB, ZiPS: 155 H / 545 AB
        # Blended: 157.5 H / 547.5 AB = .2877
        expected_avg = 157.5 / 547.5
        assert judge["avg"] == pytest.approx(expected_avg, abs=0.001)

    def test_blended_era_recomputed_from_components(self, fixtures_dir):
        hitters, pitchers = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
        )
        cole = pitchers[pitchers["name"] == "Gerrit Cole"].iloc[0]
        # Steamer: 70 ER / 200 IP, ZiPS: 72 ER / 195 IP
        # Blended: 71 ER / 197.5 IP -> ERA = 71 * 9 / 197.5 = 3.234
        expected_era = 71.0 * 9 / 197.5
        assert cole["era"] == pytest.approx(expected_era, abs=0.01)

    def test_blended_whip_recomputed_from_components(self, fixtures_dir):
        hitters, pitchers = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
        )
        cole = pitchers[pitchers["name"] == "Gerrit Cole"].iloc[0]
        # Steamer: (56 BB + 154 H) / 200 IP = 1.05
        # ZiPS: (57 BB + 154 H) / 195 IP = 1.08
        # Blended: (56.5 BB + 154 H) / 197.5 IP
        expected_whip = (56.5 + 154.0) / 197.5
        assert cole["whip"] == pytest.approx(expected_whip, abs=0.01)

    def test_custom_weights(self, fixtures_dir):
        hitters, pitchers = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
            weights={"steamer": 0.75, "zips": 0.25},
        )
        judge = hitters[hitters["name"] == "Aaron Judge"].iloc[0]
        # Steamer: 45 HR * 0.75 + ZiPS: 42 HR * 0.25 = 33.75 + 10.5 = 44.25
        assert judge["hr"] == pytest.approx(44.25)

    def test_single_system(self, fixtures_dir):
        hitters, pitchers = blend_projections(
            fixtures_dir,
            systems=["steamer"],
        )
        judge = hitters[hitters["name"] == "Aaron Judge"].iloc[0]
        assert judge["hr"] == 45

    def test_missing_system_ignored(self, fixtures_dir):
        hitters, pitchers = blend_projections(
            fixtures_dir,
            systems=["steamer", "nonexistent"],
        )
        # Only steamer data, but weights still applied correctly
        assert len(hitters) == 4
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_data/test_projections.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 4: Write the implementation**

Create `src/fantasy_baseball/data/projections.py`:
```python
import pandas as pd
from pathlib import Path
from .fangraphs import load_projection_set

# Counting stats to blend directly (weighted average)
HITTING_COUNTING_COLS: list[str] = ["r", "hr", "rbi", "sb", "h", "ab", "pa"]
PITCHING_COUNTING_COLS: list[str] = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]


def blend_projections(
    projections_dir: Path,
    systems: list[str],
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend multiple projection systems into weighted averages.

    Counting stats are blended directly. Rate stats (AVG, ERA, WHIP)
    are recomputed from blended component stats.

    Args:
        projections_dir: Path to directory containing projection CSVs.
        systems: List of projection system names (e.g., ["steamer", "zips"]).
        weights: Optional dict of system -> weight. Defaults to equal weighting.

    Returns:
        Tuple of (blended_hitters, blended_pitchers) DataFrames.
    """
    if weights is None:
        weights = {s: 1.0 / len(systems) for s in systems}

    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    all_hitters: list[pd.DataFrame] = []
    all_pitchers: list[pd.DataFrame] = []

    for system in systems:
        hitters, pitchers = load_projection_set(projections_dir, system)
        w = weights.get(system, 0)
        if not hitters.empty:
            hitters = hitters.copy()
            hitters["_weight"] = w
            all_hitters.append(hitters)
        if not pitchers.empty:
            pitchers = pitchers.copy()
            pitchers["_weight"] = w
            all_pitchers.append(pitchers)

    blended_hitters = _blend_hitters(all_hitters)
    blended_pitchers = _blend_pitchers(all_pitchers)
    return blended_hitters, blended_pitchers


def _blend_hitters(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Blend hitter projections. Recomputes AVG from blended H and AB."""
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    results = []
    for name, group in combined.groupby("name"):
        w = group["_weight"].values
        row: dict = {"name": name, "player_type": "hitter"}
        for col in HITTING_COUNTING_COLS:
            if col in group.columns:
                row[col] = (group[col] * w).sum()
        # Recompute AVG from blended H and AB
        if row.get("ab", 0) > 0:
            row["avg"] = row["h"] / row["ab"]
        else:
            row["avg"] = 0.0
        if "team" in group.columns:
            row["team"] = group.loc[group["_weight"].idxmax(), "team"]
        if "fg_id" in group.columns:
            row["fg_id"] = group.iloc[0]["fg_id"]
        results.append(row)
    return pd.DataFrame(results)


def _blend_pitchers(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Blend pitcher projections. Recomputes ERA and WHIP from components."""
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    results = []
    for name, group in combined.groupby("name"):
        w = group["_weight"].values
        row: dict = {"name": name, "player_type": "pitcher"}
        for col in PITCHING_COUNTING_COLS:
            if col in group.columns:
                row[col] = (group[col] * w).sum()
        # Recompute ERA = ER * 9 / IP
        ip = row.get("ip", 0)
        if ip > 0:
            row["era"] = row.get("er", 0) * 9 / ip
            bb = row.get("bb", 0)
            h_allowed = row.get("h_allowed", 0)
            row["whip"] = (bb + h_allowed) / ip
        else:
            row["era"] = 0.0
            row["whip"] = 0.0
        if "team" in group.columns:
            row["team"] = group.loc[group["_weight"].idxmax(), "team"]
        if "fg_id" in group.columns:
            row["fg_id"] = group.iloc[0]["fg_id"]
        results.append(row)
    return pd.DataFrame(results)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_data/test_projections.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/data/projections.py tests/test_data/test_projections.py tests/fixtures/zips_*.csv
git commit -m "feat: add projection blending with rate stat recomputation"
```

---

### Task 6: Auth — Yahoo OAuth2 Wrapper

**Files:**
- Create: `src/fantasy_baseball/auth/yahoo_auth.py`
- Create: `tests/test_auth/__init__.py`
- Create: `tests/test_auth/test_yahoo_auth.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_auth/__init__.py` (empty) and `tests/test_auth/test_yahoo_auth.py`:
```python
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fantasy_baseball.auth.yahoo_auth import (
    get_yahoo_session,
    get_league,
    CONFIG_PATH,
)


def test_config_path_points_to_oauth_json():
    assert CONFIG_PATH.name == "oauth.json"
    assert "config" in CONFIG_PATH.parts


def test_get_yahoo_session_raises_if_no_config(tmp_path):
    with patch("fantasy_baseball.auth.yahoo_auth.CONFIG_PATH", tmp_path / "nope.json"):
        with pytest.raises(FileNotFoundError, match="oauth.json"):
            get_yahoo_session()


def test_get_league_returns_league_object():
    mock_session = MagicMock()
    mock_game = MagicMock()
    mock_league = MagicMock()
    mock_game.to_league.return_value = mock_league
    with patch("fantasy_baseball.auth.yahoo_auth.yfa.Game", return_value=mock_game):
        league = get_league(mock_session, league_id=5652, game_key="mlb")
    mock_game.to_league.assert_called_once()
    assert league == mock_league
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_auth/test_yahoo_auth.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/auth/yahoo_auth.py`:
```python
from pathlib import Path
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa

CONFIG_PATH: Path = Path(__file__).resolve().parents[3] / "config" / "oauth.json"


def get_yahoo_session(config_path: Path | None = None) -> OAuth2:
    """Create an authenticated Yahoo OAuth2 session.

    On first run, opens a browser for Yahoo login. Token is cached
    in the oauth.json file for subsequent runs.

    Args:
        config_path: Path to oauth.json. Defaults to config/oauth.json.

    Raises:
        FileNotFoundError: If oauth.json doesn't exist.
    """
    path = config_path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"oauth.json not found at {path}. "
            "Create it with your Yahoo app's consumer_key and consumer_secret. "
            "See config/league.yaml.example for setup instructions."
        )
    return OAuth2(None, None, from_file=str(path))


def get_league(session: OAuth2, league_id: int, game_key: str = "mlb"):
    """Get a Yahoo Fantasy league object.

    Args:
        session: Authenticated OAuth2 session.
        league_id: Yahoo league ID (e.g., 5652).
        game_key: Yahoo game key (default: "mlb" for current season).

    Returns:
        yahoo_fantasy_api League object.
    """
    game = yfa.Game(session, game_key)
    league_ids = game.league_ids()
    # Find the league key matching our league_id
    league_key = None
    for lid in league_ids:
        if str(league_id) in lid:
            league_key = lid
            break
    if league_key is None:
        league_key = f"{game.game_id()}.l.{league_id}"
    return game.to_league(league_key)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_auth/test_yahoo_auth.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/auth/yahoo_auth.py tests/test_auth/
git commit -m "feat: add Yahoo OAuth2 session and league wrapper"
```

---

## Chunk 2: SGP Engine + Integration

### Task 7: SGP — Denominators and Config Loading

**Files:**
- Create: `src/fantasy_baseball/sgp/denominators.py`
- Create: `tests/test_sgp/test_denominators.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_sgp/test_denominators.py`:
```python
import pytest
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.constants import DEFAULT_SGP_DENOMINATORS, ALL_CATEGORIES


def test_returns_defaults_with_no_overrides():
    denoms = get_sgp_denominators()
    assert denoms == DEFAULT_SGP_DENOMINATORS


def test_overrides_specific_categories():
    overrides = {"HR": 10.0, "SV": 8.0}
    denoms = get_sgp_denominators(overrides)
    assert denoms["HR"] == 10.0
    assert denoms["SV"] == 8.0
    # Others unchanged
    assert denoms["R"] == DEFAULT_SGP_DENOMINATORS["R"]


def test_all_categories_present():
    denoms = get_sgp_denominators()
    assert set(denoms.keys()) == set(ALL_CATEGORIES)


def test_all_denominators_positive():
    denoms = get_sgp_denominators()
    for cat, val in denoms.items():
        assert val > 0, f"SGP denominator for {cat} must be positive"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sgp/test_denominators.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/sgp/denominators.py`:
```python
from fantasy_baseball.utils.constants import DEFAULT_SGP_DENOMINATORS


def get_sgp_denominators(
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """Get SGP denominators, optionally overriding defaults.

    Args:
        overrides: Dict of category -> denominator to override defaults.

    Returns:
        Complete dict of category -> SGP denominator.
    """
    denoms = dict(DEFAULT_SGP_DENOMINATORS)
    if overrides:
        denoms.update(overrides)
    return denoms
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sgp/test_denominators.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/sgp/denominators.py tests/test_sgp/test_denominators.py
git commit -m "feat: add SGP denominator loading with configurable overrides"
```

---

### Task 8: SGP — Player SGP Calculation (Counting + Rate Stats)

**Files:**
- Create: `src/fantasy_baseball/sgp/player_value.py`
- Create: `tests/test_sgp/test_player_value.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_sgp/test_player_value.py`:
```python
import pytest
import pandas as pd
from fantasy_baseball.sgp.player_value import (
    calculate_counting_sgp,
    calculate_hitting_rate_sgp,
    calculate_pitching_rate_sgp,
    calculate_player_sgp,
)


class TestCountingSgp:
    def test_hr_sgp(self):
        # 45 HR / 9 SGP_denom = 5.0 SGP
        assert calculate_counting_sgp(45, 9.0) == pytest.approx(5.0)

    def test_zero_stat(self):
        assert calculate_counting_sgp(0, 9.0) == pytest.approx(0.0)

    def test_saves(self):
        # 40 SV / 7 SGP_denom = 5.714
        assert calculate_counting_sgp(40, 7.0) == pytest.approx(5.714, abs=0.001)


class TestHittingRateSgp:
    def test_avg_marginal_hits(self):
        # Player: .291 AVG, 550 AB -> 160.05 H
        # Replacement: .250 AVG
        # Marginal hits = (.291 - .250) * 550 = 22.55
        # SGP = 22.55 / sgp_denom_marginal
        # We use team_ab=5500 (10 hitters * 550 AB avg)
        # One SGP in AVG = 0.005 * 5500 = 27.5 marginal hits
        sgp = calculate_hitting_rate_sgp(
            player_avg=0.291,
            player_ab=550,
            replacement_avg=0.250,
            sgp_denominator=0.005,
            team_ab=5500,
        )
        expected = (0.291 - 0.250) * 550 / (0.005 * 5500)
        assert sgp == pytest.approx(expected)

    def test_below_replacement_avg(self):
        sgp = calculate_hitting_rate_sgp(
            player_avg=0.220,
            player_ab=400,
            replacement_avg=0.250,
            sgp_denominator=0.005,
            team_ab=5500,
        )
        assert sgp < 0  # Below replacement = negative SGP


class TestPitchingRateSgp:
    def test_era_marginal(self):
        # Player: 3.15 ERA, 200 IP -> 70 ER
        # Replacement: 4.50 ERA
        # Marginal ER saved = (4.50 - 3.15) * 200 / 9 = 30.0
        # Team IP = 1400 (rough), one SGP = 0.15 * 1400 / 9 = 23.33 marginal ER
        sgp = calculate_pitching_rate_sgp(
            player_rate=3.15,
            player_ip=200,
            replacement_rate=4.50,
            sgp_denominator=0.15,
            team_ip=1400,
            innings_divisor=9,
        )
        expected = (4.50 - 3.15) * 200 / 9 / (0.15 * 1400 / 9)
        assert sgp == pytest.approx(expected)

    def test_whip_marginal(self):
        # Player: 1.05 WHIP, 200 IP
        # Replacement: 1.35 WHIP
        # Marginal baserunners prevented = (1.35 - 1.05) * 200 = 60
        # Team IP = 1400, one SGP = 0.015 * 1400 = 21 marginal baserunners
        sgp = calculate_pitching_rate_sgp(
            player_rate=1.05,
            player_ip=200,
            replacement_rate=1.35,
            sgp_denominator=0.015,
            team_ip=1400,
            innings_divisor=1,
        )
        expected = (1.35 - 1.05) * 200 / (0.015 * 1400)
        assert sgp == pytest.approx(expected)

    def test_bad_era_is_negative(self):
        sgp = calculate_pitching_rate_sgp(
            player_rate=5.50,
            player_ip=150,
            replacement_rate=4.50,
            sgp_denominator=0.15,
            team_ip=1400,
            innings_divisor=9,
        )
        assert sgp < 0


class TestCalculatePlayerSgp:
    def test_hitter_total_sgp(self):
        player = pd.Series({
            "name": "Aaron Judge",
            "player_type": "hitter",
            "r": 110, "hr": 45, "rbi": 120, "sb": 5,
            "avg": 0.291, "ab": 550, "h": 160,
        })
        sgp = calculate_player_sgp(player, team_ab=5500, team_ip=1400)
        # Should have positive SGP from counting stats
        assert sgp > 0
        # HR alone: 45/9 = 5.0 SGP. Total should be well above that.
        assert sgp > 5.0

    def test_pitcher_total_sgp(self):
        player = pd.Series({
            "name": "Gerrit Cole",
            "player_type": "pitcher",
            "w": 15, "k": 240, "sv": 0,
            "era": 3.15, "whip": 1.05, "ip": 200,
            "er": 70, "bb": 56, "h_allowed": 154,
        })
        sgp = calculate_player_sgp(player, team_ab=5500, team_ip=1400)
        assert sgp > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sgp/test_player_value.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/sgp/player_value.py`:
```python
import pandas as pd
from fantasy_baseball.utils.constants import (
    DEFAULT_SGP_DENOMINATORS,
    INVERSE_STATS,
)
from .denominators import get_sgp_denominators

# Typical team totals for a 10-team league (used for rate stat conversion)
DEFAULT_TEAM_AB: int = 5500
DEFAULT_TEAM_IP: int = 1400

# Replacement-level rate stats (league-average baselines for year 1)
REPLACEMENT_AVG: float = 0.250
REPLACEMENT_ERA: float = 4.50
REPLACEMENT_WHIP: float = 1.35


def calculate_counting_sgp(stat_value: float, sgp_denominator: float) -> float:
    """Convert a counting stat to SGP.

    SGP = stat_value / sgp_denominator
    """
    return stat_value / sgp_denominator


def calculate_hitting_rate_sgp(
    player_avg: float,
    player_ab: int,
    replacement_avg: float,
    sgp_denominator: float,
    team_ab: int,
) -> float:
    """Calculate SGP for batting average using marginal hits.

    Marginal hits = (player_AVG - replacement_AVG) * player_AB
    One SGP of AVG = sgp_denominator * team_AB marginal hits
    SGP = marginal_hits / (sgp_denominator * team_AB)
    """
    marginal_hits = (player_avg - replacement_avg) * player_ab
    one_sgp_in_hits = sgp_denominator * team_ab
    return marginal_hits / one_sgp_in_hits


def calculate_pitching_rate_sgp(
    player_rate: float,
    player_ip: float,
    replacement_rate: float,
    sgp_denominator: float,
    team_ip: float,
    innings_divisor: float,
) -> float:
    """Calculate SGP for a pitching rate stat using marginal value.

    For ERA (innings_divisor=9):
        Marginal ER saved = (replacement_ERA - player_ERA) * IP / 9
        One SGP = sgp_denominator * team_IP / 9 marginal ER
    For WHIP (innings_divisor=1):
        Marginal baserunners prevented = (replacement_WHIP - player_WHIP) * IP
        One SGP = sgp_denominator * team_IP marginal baserunners

    Positive = better than replacement (lower ERA/WHIP).
    """
    marginal = (replacement_rate - player_rate) * player_ip / innings_divisor
    one_sgp = sgp_denominator * team_ip / innings_divisor
    return marginal / one_sgp


def calculate_player_sgp(
    player: pd.Series,
    denoms: dict[str, float] | None = None,
    team_ab: int = DEFAULT_TEAM_AB,
    team_ip: int = DEFAULT_TEAM_IP,
    replacement_avg: float = REPLACEMENT_AVG,
    replacement_era: float = REPLACEMENT_ERA,
    replacement_whip: float = REPLACEMENT_WHIP,
) -> float:
    """Calculate total SGP for a player across all relevant categories.

    Args:
        player: Series with player stats (must include 'player_type').
        denoms: SGP denominators. Defaults to published baselines.
        team_ab: Estimated total team AB for rate stat conversion.
        team_ip: Estimated total team IP for rate stat conversion.
        replacement_avg/era/whip: Replacement-level rate stats.

    Returns:
        Total SGP value (sum across all categories).
    """
    if denoms is None:
        denoms = get_sgp_denominators()

    total_sgp = 0.0

    if player.get("player_type") == "hitter":
        # Counting stats
        for stat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
            val = player.get(col, 0)
            total_sgp += calculate_counting_sgp(val, denoms[stat])
        # Rate stat: AVG
        total_sgp += calculate_hitting_rate_sgp(
            player_avg=player.get("avg", 0),
            player_ab=int(player.get("ab", 0)),
            replacement_avg=replacement_avg,
            sgp_denominator=denoms["AVG"],
            team_ab=team_ab,
        )

    elif player.get("player_type") == "pitcher":
        # Counting stats
        for stat, col in [("W", "w"), ("K", "k"), ("SV", "sv")]:
            val = player.get(col, 0)
            total_sgp += calculate_counting_sgp(val, denoms[stat])
        # Rate stats: ERA and WHIP
        ip = player.get("ip", 0)
        if ip > 0:
            total_sgp += calculate_pitching_rate_sgp(
                player_rate=player.get("era", 0),
                player_ip=ip,
                replacement_rate=replacement_era,
                sgp_denominator=denoms["ERA"],
                team_ip=team_ip,
                innings_divisor=9,
            )
            total_sgp += calculate_pitching_rate_sgp(
                player_rate=player.get("whip", 0),
                player_ip=ip,
                replacement_rate=replacement_whip,
                sgp_denominator=denoms["WHIP"],
                team_ip=team_ip,
                innings_divisor=1,
            )

    return total_sgp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sgp/test_player_value.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/sgp/player_value.py tests/test_sgp/test_player_value.py
git commit -m "feat: add SGP calculation for counting and rate stats"
```

---

### Task 9: SGP — Replacement Level Calculation

**Files:**
- Create: `src/fantasy_baseball/sgp/replacement.py`
- Create: `tests/test_sgp/test_replacement.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_sgp/test_replacement.py`:
```python
import pytest
import pandas as pd
from fantasy_baseball.sgp.replacement import calculate_replacement_levels


def _make_player_pool():
    """Create a small test player pool with known replacement levels."""
    hitters = []
    # 15 catchers with decreasing SGP (10 above replacement, 5 below)
    for i in range(15):
        hitters.append({
            "name": f"Catcher_{i}",
            "positions": ["C"],
            "total_sgp": 20.0 - i,
            "player_type": "hitter",
        })
    # 15 first basemen
    for i in range(15):
        hitters.append({
            "name": f"FirstBase_{i}",
            "positions": ["1B"],
            "total_sgp": 25.0 - i,
            "player_type": "hitter",
        })
    # 50 outfielders (40 above replacement)
    for i in range(50):
        hitters.append({
            "name": f"Outfielder_{i}",
            "positions": ["OF"],
            "total_sgp": 30.0 - i * 0.5,
            "player_type": "hitter",
        })

    pitchers = []
    # 100 pitchers (90 above replacement)
    for i in range(100):
        pitchers.append({
            "name": f"Pitcher_{i}",
            "positions": ["SP"] if i < 70 else ["RP"],
            "total_sgp": 25.0 - i * 0.2,
            "player_type": "pitcher",
        })

    all_players = hitters + pitchers
    return pd.DataFrame(all_players)


class TestReplacementLevels:
    def test_catcher_replacement_level(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        # 10 catchers above replacement -> replacement = Catcher_10 SGP = 20-10 = 10.0
        assert levels["C"] == pytest.approx(10.0)

    def test_first_base_replacement_level(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        # 10 first basemen above replacement -> replacement = FirstBase_10 SGP = 25-10 = 15.0
        assert levels["1B"] == pytest.approx(15.0)

    def test_of_replacement_level(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        # 40 OF above replacement -> replacement = Outfielder_40 SGP = 30-40*0.5 = 10.0
        assert levels["OF"] == pytest.approx(10.0)

    def test_pitcher_replacement_level(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        # 90 pitchers above replacement -> replacement = Pitcher_90 SGP = 25-90*0.2 = 7.0
        assert levels["P"] == pytest.approx(7.0)

    def test_all_starter_positions_have_levels(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        assert "C" in levels
        assert "OF" in levels
        assert "P" in levels
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sgp/test_replacement.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/sgp/replacement.py`:
```python
import pandas as pd
from fantasy_baseball.utils.constants import STARTERS_PER_POSITION
from fantasy_baseball.utils.positions import is_pitcher


def calculate_replacement_levels(
    player_pool: pd.DataFrame,
    starters_per_position: dict[str, int] | None = None,
) -> dict[str, float]:
    """Calculate replacement-level SGP for each position.

    Replacement level = the SGP of the (N+1)th best player at that position,
    where N = number of starters across all teams.

    Args:
        player_pool: DataFrame with columns: name, positions (list), total_sgp, player_type.
        starters_per_position: Override for number of starters per position.

    Returns:
        Dict of position -> replacement-level SGP.
    """
    if starters_per_position is None:
        starters_per_position = dict(STARTERS_PER_POSITION)

    replacement_levels: dict[str, float] = {}

    for position, num_starters in starters_per_position.items():
        if position in ("IF", "UTIL"):
            continue  # Handled as flex slots, not primary positions

        eligible = _get_eligible_players(player_pool, position)
        eligible = eligible.sort_values("total_sgp", ascending=False).reset_index(drop=True)

        if len(eligible) > num_starters:
            replacement_levels[position] = eligible.iloc[num_starters]["total_sgp"]
        elif len(eligible) > 0:
            replacement_levels[position] = eligible.iloc[-1]["total_sgp"]
        else:
            replacement_levels[position] = 0.0

    return replacement_levels


def _get_eligible_players(pool: pd.DataFrame, position: str) -> pd.DataFrame:
    """Filter player pool to those eligible for a position."""
    if position == "P":
        return pool[pool["positions"].apply(
            lambda pos: any(p in ("P", "SP", "RP") for p in pos)
        )]
    if position == "OF":
        return pool[pool["positions"].apply(lambda pos: "OF" in pos)]
    return pool[pool["positions"].apply(lambda pos: position in pos)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sgp/test_replacement.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/sgp/replacement.py tests/test_sgp/test_replacement.py
git commit -m "feat: add replacement level calculation per position"
```

---

### Task 10: SGP — Value Above Replacement (VAR)

**Files:**
- Create: `src/fantasy_baseball/sgp/var.py`
- Create: `tests/test_sgp/test_var.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_sgp/test_var.py`:
```python
import pytest
import pandas as pd
from fantasy_baseball.sgp.var import calculate_var


def test_var_simple_hitter():
    player = pd.Series({
        "name": "Test Hitter",
        "positions": ["1B"],
        "total_sgp": 20.0,
        "player_type": "hitter",
    })
    replacement_levels = {"1B": 12.0, "C": 8.0, "OF": 10.0, "P": 7.0}
    var = calculate_var(player, replacement_levels)
    # VAR = 20.0 - 12.0 = 8.0
    assert var == pytest.approx(8.0)


def test_var_multi_position_uses_most_valuable():
    player = pd.Series({
        "name": "Multi Pos",
        "positions": ["SS", "2B"],
        "total_sgp": 18.0,
        "player_type": "hitter",
    })
    # SS replacement is lower, so SS position is more valuable
    replacement_levels = {"SS": 8.0, "2B": 12.0, "C": 8.0, "OF": 10.0, "P": 7.0}
    var = calculate_var(player, replacement_levels)
    # VAR = 18.0 - 8.0 = 10.0 (uses SS, the more valuable position)
    assert var == pytest.approx(10.0)


def test_var_pitcher():
    player = pd.Series({
        "name": "Test Pitcher",
        "positions": ["SP"],
        "total_sgp": 15.0,
        "player_type": "pitcher",
    })
    replacement_levels = {"P": 7.0, "C": 8.0}
    var = calculate_var(player, replacement_levels)
    assert var == pytest.approx(8.0)


def test_var_below_replacement_is_negative():
    player = pd.Series({
        "name": "Bad Player",
        "positions": ["C"],
        "total_sgp": 5.0,
        "player_type": "hitter",
    })
    replacement_levels = {"C": 8.0, "P": 7.0}
    var = calculate_var(player, replacement_levels)
    assert var == pytest.approx(-3.0)


def test_var_assigns_best_position():
    player = pd.Series({
        "name": "Multi",
        "positions": ["1B", "OF"],
        "total_sgp": 20.0,
        "player_type": "hitter",
    })
    replacement_levels = {"1B": 15.0, "OF": 10.0, "C": 8.0, "P": 7.0}
    var, pos = calculate_var(player, replacement_levels, return_position=True)
    assert var == pytest.approx(10.0)
    assert pos == "OF"  # OF replacement is lower, so more VAR at OF
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sgp/test_var.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/sgp/var.py`:
```python
import pandas as pd
from fantasy_baseball.utils.positions import is_pitcher


def calculate_var(
    player: pd.Series,
    replacement_levels: dict[str, float],
    return_position: bool = False,
) -> float | tuple[float, str]:
    """Calculate Value Above Replacement for a player.

    For multi-position players, uses the position where they have
    the highest VAR (lowest replacement level).

    Args:
        player: Series with 'positions' (list[str]), 'total_sgp' (float).
        replacement_levels: Dict of position -> replacement SGP.
        return_position: If True, also return the best position.

    Returns:
        VAR value, or (VAR, best_position) if return_position=True.
    """
    total_sgp = player["total_sgp"]
    positions = player["positions"]

    best_var = float("-inf")
    best_pos = None

    for pos in positions:
        # Map SP/RP to the generic P slot
        lookup_pos = "P" if pos in ("P", "SP", "RP") else pos
        if lookup_pos in replacement_levels:
            var = total_sgp - replacement_levels[lookup_pos]
            if var > best_var:
                best_var = var
                best_pos = lookup_pos

    # If no matching position found, use 0 as replacement
    if best_pos is None:
        best_var = total_sgp
        best_pos = positions[0] if positions else "UTIL"

    if return_position:
        return best_var, best_pos
    return best_var
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sgp/test_var.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/sgp/var.py tests/test_sgp/test_var.py
git commit -m "feat: add Value Above Replacement (VAR) calculation"
```

---

### Task 11: Config Loading

**Files:**
- Create: `src/fantasy_baseball/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_config.py`:
```python
import pytest
from pathlib import Path
from fantasy_baseball.config import load_config, LeagueConfig


@pytest.fixture
def sample_config(tmp_path):
    config_file = tmp_path / "league.yaml"
    config_file.write_text("""
league:
  id: 5652
  num_teams: 10
  game_code: mlb

draft:
  position: 8

keepers:
  - name: "Player A"
    team: "Team 1"
  - name: "Player B"
    team: "Team 2"

roster_slots:
  C: 1
  1B: 1
  2B: 1
  3B: 1
  SS: 1
  IF: 1
  OF: 4
  UTIL: 2
  P: 9
  BN: 2
  IL: 2

projections:
  systems:
    - steamer
    - zips
  weights:
    steamer: 0.6
    zips: 0.4

sgp_denominators:
  HR: 10
""")
    return config_file


def test_load_config_basic(sample_config):
    config = load_config(sample_config)
    assert config.league_id == 5652
    assert config.num_teams == 10
    assert config.draft_position == 8


def test_load_config_keepers(sample_config):
    config = load_config(sample_config)
    assert len(config.keepers) == 2
    assert config.keepers[0]["name"] == "Player A"


def test_load_config_projection_weights(sample_config):
    config = load_config(sample_config)
    assert config.projection_systems == ["steamer", "zips"]
    assert config.projection_weights == {"steamer": 0.6, "zips": 0.4}


def test_load_config_sgp_overrides(sample_config):
    config = load_config(sample_config)
    assert config.sgp_overrides == {"HR": 10}


def test_load_config_roster_slots(sample_config):
    config = load_config(sample_config)
    assert config.roster_slots["C"] == 1
    assert config.roster_slots["OF"] == 4
    assert sum(config.roster_slots.values()) == 25


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/league.yaml"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/config.py`:
```python
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class LeagueConfig:
    league_id: int
    num_teams: int
    game_code: str
    draft_position: int
    keepers: list[dict]
    roster_slots: dict[str, int]
    projection_systems: list[str]
    projection_weights: dict[str, float]
    sgp_overrides: dict[str, float] = field(default_factory=dict)


def load_config(config_path: Path) -> LeagueConfig:
    """Load league configuration from a YAML file.

    Args:
        config_path: Path to league.yaml.

    Returns:
        LeagueConfig with all settings.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    league = raw.get("league", {})
    draft = raw.get("draft", {})
    projections = raw.get("projections", {})

    return LeagueConfig(
        league_id=league.get("id", 0),
        num_teams=league.get("num_teams", 10),
        game_code=league.get("game_code", "mlb"),
        draft_position=draft.get("position", 1),
        keepers=raw.get("keepers", []),
        roster_slots=raw.get("roster_slots", {}),
        projection_systems=projections.get("systems", []),
        projection_weights=projections.get("weights", {}),
        sgp_overrides=raw.get("sgp_denominators", {}),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/config.py tests/test_config.py
git commit -m "feat: add YAML config loading for league settings"
```

---

### Task 12: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration.py`:
```python
"""Integration test: load projections -> blend -> calculate SGP -> rank by VAR."""
import pytest
import pandas as pd
from pathlib import Path
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.replacement import calculate_replacement_levels
from fantasy_baseball.sgp.var import calculate_var


def test_full_pipeline(fixtures_dir):
    """End-to-end: blend projections, calculate SGP, compute VAR, rank players."""
    # Step 1: Blend projections
    hitters, pitchers = blend_projections(
        fixtures_dir,
        systems=["steamer", "zips"],
    )
    assert len(hitters) == 4
    assert len(pitchers) == 3

    # Step 2: Calculate SGP for each player
    # Add mock position data (in real use, this comes from Yahoo API)
    hitters["positions"] = [["OF"], ["OF"], ["C"], ["2B", "SS"]]
    pitchers["positions"] = [["SP"], ["RP"], ["SP"]]

    for idx, row in hitters.iterrows():
        hitters.loc[idx, "total_sgp"] = calculate_player_sgp(row)
    for idx, row in pitchers.iterrows():
        pitchers.loc[idx, "total_sgp"] = calculate_player_sgp(row)

    # Step 3: Build player pool and calculate replacement levels
    pool = pd.concat([hitters, pitchers], ignore_index=True)
    assert len(pool) == 7

    # With only 7 players, replacement levels will use the last player
    # Use smaller starters_per_position for this test
    small_starters = {"C": 1, "OF": 2, "SS": 1, "2B": 1, "P": 2}
    levels = calculate_replacement_levels(pool, small_starters)

    # Step 4: Calculate VAR for each player
    vars_list = []
    for _, player in pool.iterrows():
        var = calculate_var(player, levels)
        vars_list.append({"name": player["name"], "var": var})

    rankings = (
        pd.DataFrame(vars_list)
        .sort_values("var", ascending=False)
        .reset_index(drop=True)
    )

    # Step 5: Verify rankings make sense
    assert len(rankings) == 7
    # Top player should have positive VAR
    assert rankings.iloc[0]["var"] > 0
    # Rankings should be sorted descending
    assert rankings.iloc[0]["var"] >= rankings.iloc[-1]["var"]
    # Aaron Judge should be near the top (high HR, R, RBI)
    judge_rank = rankings[rankings["name"] == "Aaron Judge"].index[0]
    assert judge_rank <= 2  # Top 3


def test_pipeline_with_keepers(fixtures_dir):
    """Verify keepers can be removed from the player pool."""
    hitters, pitchers = blend_projections(
        fixtures_dir,
        systems=["steamer"],
    )
    hitters["positions"] = [["OF"], ["OF"], ["C"], ["2B", "SS"]]
    pitchers["positions"] = [["SP"], ["RP"], ["SP"]]

    pool = pd.concat([hitters, pitchers], ignore_index=True)
    assert len(pool) == 7

    # Remove keepers
    keepers = ["Aaron Judge", "Gerrit Cole"]
    pool = pool[~pool["name"].isin(keepers)]
    assert len(pool) == 5
    assert "Aaron Judge" not in pool["name"].values
```

- [ ] **Step 2: Run the integration test**

Run: `pytest tests/test_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Run the full test suite**

Run: `pytest -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add end-to-end integration test for projection -> SGP -> VAR pipeline"
```

---

## Summary

After completing all 12 tasks, the shared foundation provides:

1. **`utils`** — League constants, position eligibility helpers
2. **`auth`** — Yahoo OAuth2 session management
3. **`data`** — FanGraphs CSV parsing and multi-system projection blending
4. **`sgp`** — SGP denominators, player SGP calculation (counting + rate stats), replacement levels, VAR
5. **`config`** — YAML config loading for league settings

The draft assistant (Phase 2) and lineup optimizer (Phase 3) can build directly on these modules.
