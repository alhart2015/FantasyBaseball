# Preseason Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the "Opening Day" Monte Carlo pair (base + with_management) as a per-season Redis artifact so the refresh reads it instead of re-running the MCs every refresh. Relabel the standings view to "Opening Day baseline" and surface the freeze date.

**Architecture:** A one-off script (`scripts/freeze_preseason_baseline.py`) fetches every team's Opening-Day roster via `fetch_roster(..., day=config.season_start)`, matches against preseason projections from Redis, runs `run_monte_carlo` twice (base + mgmt) at 1000 iterations, and writes `preseason_baseline:{season_year}` to Redis. `RefreshRun._run_monte_carlo` is deleted; `_run_ros_monte_carlo`'s cache-write step reads the frozen baseline and drops it into the `monte_carlo` cache alongside the ROS MCs. The template keeps the same cache-key shape so `season_routes.py` needs only minor guards.

**Tech Stack:** Python 3.12, pytest, fakeredis, Upstash Redis, Flask/Jinja2, pandas, `yahoo_fantasy_api`, `run_monte_carlo`.

---

## File Structure

**Create:**
- `scripts/freeze_preseason_baseline.py` — one-off per-season script.
- `tests/test_scripts/__init__.py` — package marker (if not present).
- `tests/test_scripts/test_freeze_preseason_baseline.py` — end-to-end script test.

**Modify:**
- `src/fantasy_baseball/data/redis_store.py` — add `get_preseason_baseline`, `set_preseason_baseline`.
- `src/fantasy_baseball/web/refresh_pipeline.py` — delete `_run_monte_carlo`, update `_run_ros_monte_carlo` to read baseline from Redis.
- `src/fantasy_baseball/web/season_routes.py` — guard `mc_data` / `mc_mgmt_data` on truthy + pass `baseline_meta` to the template.
- `src/fantasy_baseball/web/templates/season/standings.html` — relabel "Preseason" → "Opening Day baseline" and add `baseline_meta.roster_date` tooltip.
- `tests/test_web/_refresh_fixture.py` — remove preseason-MC plumbing; seed a canned baseline in fake Redis.
- `tests/test_web/test_refresh_pipeline.py` — extend shape + ROS-branch assertions to the new semantics (None when baseline absent).
- `tests/test_data/test_redis_store_preseason_baseline.py` (new) — unit tests for the two Redis helpers.

**Touched but no code change:** `TODO.md` already has the `initialize_season()` postseason item (committed with the spec).

---

## Task 1: Redis helpers for preseason baseline

**Files:**
- Create: `tests/test_data/test_redis_store_preseason_baseline.py`
- Modify: `src/fantasy_baseball/data/redis_store.py` (add after `set_blended_projections`, before `ROS_PROJECTIONS_KEY`)

- [ ] **Step 1: Write the failing test file**

Create `tests/test_data/test_redis_store_preseason_baseline.py`:

```python
"""Tests for preseason_baseline:{year} helpers."""
import pytest

from fantasy_baseball.data import redis_store


BASELINE = {
    "base": {"team_results": {"Team 01": {"median_pts": 72.5}}, "category_risk": {}},
    "with_management": {"team_results": {"Team 01": {"median_pts": 75.0}}, "category_risk": {}},
    "meta": {
        "frozen_at": "2026-04-18T12:00:00Z",
        "season_year": 2026,
        "roster_date": "2026-03-27",
        "projections_source": "blended",
    },
}


def test_get_preseason_baseline_empty(fake_redis):
    assert redis_store.get_preseason_baseline(fake_redis, 2026) is None


def test_set_and_get_round_trip(fake_redis):
    redis_store.set_preseason_baseline(fake_redis, 2026, BASELINE)
    result = redis_store.get_preseason_baseline(fake_redis, 2026)
    assert result == BASELINE


def test_different_seasons_isolated(fake_redis):
    redis_store.set_preseason_baseline(fake_redis, 2026, BASELINE)
    assert redis_store.get_preseason_baseline(fake_redis, 2025) is None


def test_get_returns_none_on_corrupt_json(fake_redis):
    fake_redis.set("preseason_baseline:2026", "not valid json {{{")
    assert redis_store.get_preseason_baseline(fake_redis, 2026) is None


def test_get_returns_none_on_non_dict_payload(fake_redis):
    import json
    fake_redis.set("preseason_baseline:2026", json.dumps(["not", "a", "dict"]))
    assert redis_store.get_preseason_baseline(fake_redis, 2026) is None


def test_get_returns_none_when_client_none():
    assert redis_store.get_preseason_baseline(None, 2026) is None


def test_set_noop_when_client_none():
    # Should not raise
    redis_store.set_preseason_baseline(None, 2026, BASELINE)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_redis_store_preseason_baseline.py -v`
Expected: every test FAILs with `AttributeError: module 'fantasy_baseball.data.redis_store' has no attribute 'get_preseason_baseline'` (or similar).

- [ ] **Step 3: Implement the helpers**

In `src/fantasy_baseball/data/redis_store.py`, add directly after the `set_blended_projections` function (around line 135) and before the `ROS_PROJECTIONS_KEY` block:

```python
def _preseason_baseline_key(season_year: int) -> str:
    return f"preseason_baseline:{season_year}"


def get_preseason_baseline(client, season_year: int) -> dict | None:
    """Read the frozen preseason Monte Carlo baseline for ``season_year``.

    Returns ``None`` on missing key, corrupt JSON, non-dict payload, or
    ``client is None``. Shape on success::

        {"base": {...}, "with_management": {...}, "meta": {...}}

    where ``base`` / ``with_management`` are ``run_monte_carlo`` outputs
    captured once per season against Opening-Day rosters + preseason
    projections.
    """
    if client is None:
        return None
    raw = client.get(_preseason_baseline_key(season_year))
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "Corrupt JSON at Redis key %r; ignoring",
            _preseason_baseline_key(season_year),
        )
        return None
    if not isinstance(data, dict):
        return None
    return data


def set_preseason_baseline(
    client, season_year: int, payload: dict
) -> None:
    """Overwrite the frozen preseason baseline for ``season_year``.

    The caller is responsible for the payload shape; this helper just
    serializes and stores. No-op when ``client is None`` (e.g. in
    unconfigured environments).
    """
    if client is None:
        return
    client.set(
        _preseason_baseline_key(season_year), json.dumps(payload)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_redis_store_preseason_baseline.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/redis_store.py tests/test_data/test_redis_store_preseason_baseline.py
git commit -m "feat(redis): add preseason_baseline get/set helpers"
```

---

## Task 2: `freeze_preseason_baseline.py` script

**Files:**
- Create: `scripts/freeze_preseason_baseline.py`
- Create: `tests/test_scripts/__init__.py` (if missing)
- Create: `tests/test_scripts/test_freeze_preseason_baseline.py`

- [ ] **Step 1: Ensure `tests/test_scripts/` exists**

Run: `test -d tests/test_scripts || mkdir tests/test_scripts && touch tests/test_scripts/__init__.py`
(If the directory already exists with `__init__.py`, nothing to do.)

- [ ] **Step 2: Write the failing test**

Create `tests/test_scripts/test_freeze_preseason_baseline.py`:

```python
"""Tests for scripts/freeze_preseason_baseline.py."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _mk_config(monkeypatch):
    from fantasy_baseball.config import LeagueConfig
    return LeagueConfig(
        league_id=123,
        num_teams=2,
        game_code="mlb",
        team_name="Team 01",
        draft_position=1,
        keepers=[],
        roster_slots={"OF": 3, "P": 9, "BN": 3, "IL": 2, "Util": 1,
                      "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1},
        projection_systems=["atc"],
        projection_weights={"atc": 1.0},
        sgp_overrides={},
        teams={1: "Team 01", 2: "Team 02"},
        strategy="no_punt_opp",
        scoring_mode="var",
        season_year=2026,
        season_start="2026-03-27",
        season_end="2026-09-28",
    )


def _fake_hitter(name, pid):
    return {"name": name, "positions": ["OF"], "selected_position": "OF",
            "player_id": pid, "status": ""}


def _fake_pitcher(name, pid):
    return {"name": name, "positions": ["SP"], "selected_position": "P",
            "player_id": pid, "status": ""}


def _fake_projection_row(name, player_type):
    if player_type == "hitter":
        return {
            "name": name, "player_type": "hitter", "team": "NYY",
            "pa": 600, "ab": 540, "h": 145, "r": 85, "hr": 25,
            "rbi": 80, "sb": 10, "avg": 0.269,
        }
    return {
        "name": name, "player_type": "pitcher", "team": "NYY",
        "ip": 180, "er": 65, "bb": 50, "h_allowed": 160,
        "w": 12, "k": 190, "sv": 0, "era": 3.25, "whip": 1.17,
    }


@pytest.fixture
def patched_script_env(fake_redis, monkeypatch):
    from fantasy_baseball.data import redis_store
    # Seed preseason projections in Redis
    redis_store.set_blended_projections(fake_redis, "hitters",
        [_fake_projection_row("H1", "hitter"), _fake_projection_row("H2", "hitter")])
    redis_store.set_blended_projections(fake_redis, "pitchers",
        [_fake_projection_row("P1", "pitcher"), _fake_projection_row("P2", "pitcher")])

    league_mock = MagicMock()
    league_mock.teams.return_value = {
        "t.1": {"name": "Team 01"}, "t.2": {"name": "Team 02"},
    }

    def _fetch_roster(league, team_key, day=None):
        assert day == "2026-03-27"
        if team_key == "t.1":
            return [_fake_hitter("H1", "1"), _fake_pitcher("P1", "2")]
        return [_fake_hitter("H2", "3"), _fake_pitcher("P2", "4")]

    def _scaled_mc(team_rosters, h_slots, p_slots, user_team_name,
                   n_iterations=1000, use_management=False, progress_cb=None):
        return {
            "team_results": {t: {"median_pts": 70.0} for t in team_rosters},
            "category_risk": {},
            "_used_management": use_management,
        }

    patches = [
        patch("fantasy_baseball.config.load_config", return_value=_mk_config(monkeypatch)),
        patch("fantasy_baseball.auth.yahoo_auth.get_yahoo_session", return_value=MagicMock()),
        patch("fantasy_baseball.auth.yahoo_auth.get_league", return_value=league_mock),
        patch("fantasy_baseball.lineup.yahoo_roster.fetch_roster", side_effect=_fetch_roster),
        patch("fantasy_baseball.data.redis_store.get_default_client", return_value=fake_redis),
        patch("fantasy_baseball.simulation.run_monte_carlo", side_effect=_scaled_mc),
    ]
    for p in patches:
        p.start()
    yield fake_redis
    for p in patches:
        p.stop()


def test_script_writes_baseline_to_redis(patched_script_env):
    from freeze_preseason_baseline import main
    main([])

    from fantasy_baseball.data import redis_store
    baseline = redis_store.get_preseason_baseline(patched_script_env, 2026)
    assert baseline is not None
    assert "base" in baseline and "with_management" in baseline
    assert baseline["base"]["_used_management"] is False
    assert baseline["with_management"]["_used_management"] is True
    assert baseline["meta"]["season_year"] == 2026
    assert baseline["meta"]["roster_date"] == "2026-03-27"
    assert "frozen_at" in baseline["meta"]


def test_script_refuses_to_overwrite_without_force(patched_script_env, capsys):
    from freeze_preseason_baseline import main
    main([])  # first write
    with pytest.raises(SystemExit) as excinfo:
        main([])  # second without --force
    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "already frozen" in captured.out.lower() or "--force" in captured.out.lower()


def test_script_overwrites_with_force(patched_script_env):
    from freeze_preseason_baseline import main
    main([])
    main(["--force"])  # should not raise

    from fantasy_baseball.data import redis_store
    baseline = redis_store.get_preseason_baseline(patched_script_env, 2026)
    assert baseline is not None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_scripts/test_freeze_preseason_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'freeze_preseason_baseline'`.

- [ ] **Step 4: Implement the script**

Create `scripts/freeze_preseason_baseline.py`:

```python
"""Freeze the preseason Monte Carlo baseline for the current season.

Fetches every team's Opening-Day roster from Yahoo, matches against
preseason projections from Redis, runs run_monte_carlo twice
(base + with_management) at 1000 iterations each, and writes the
result to Redis under ``preseason_baseline:{season_year}``.

Run this once per season, after the draft completes. The refresh
pipeline reads this artifact on every refresh instead of re-running
the MCs.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct invocation.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season-year", type=int, default=None,
        help="Override season_year from config/league.yaml.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing baseline in Redis.",
    )
    args = parser.parse_args(argv)

    import pandas as pd
    from fantasy_baseball.config import load_config
    from fantasy_baseball.auth.yahoo_auth import (
        get_league, get_yahoo_session,
    )
    from fantasy_baseball.lineup.yahoo_roster import fetch_roster
    from fantasy_baseball.data.projections import match_roster_to_projections
    from fantasy_baseball.data.redis_store import (
        get_blended_projections, get_default_client,
        get_preseason_baseline, set_preseason_baseline,
    )
    from fantasy_baseball.simulation import run_monte_carlo
    from fantasy_baseball.utils.name_utils import normalize_name

    config = load_config(_PROJECT_ROOT / "config" / "league.yaml")
    season_year = args.season_year or config.season_year

    client = get_default_client()
    if client is None:
        raise RuntimeError(
            "Redis client not configured: set UPSTASH_REDIS_REST_URL / "
            "UPSTASH_REDIS_REST_TOKEN in the environment (or .env)."
        )

    existing = get_preseason_baseline(client, season_year)
    if existing and not args.force:
        frozen_at = existing.get("meta", {}).get("frozen_at", "?")
        print(
            f"Preseason baseline for {season_year} already frozen at "
            f"{frozen_at}. Re-run with --force to overwrite."
        )
        sys.exit(1)

    print(f"Authenticating with Yahoo...")
    sc = get_yahoo_session()
    league = get_league(sc, config.league_id, config.game_code)

    print(f"Fetching Opening-Day rosters (day={config.season_start})...")
    team_rosters_raw: dict[str, list[dict]] = {}
    for team_key, team_info in league.teams().items():
        tname = team_info.get("name", team_key)
        team_rosters_raw[tname] = fetch_roster(
            league, team_key, day=config.season_start
        )
        print(f"  {tname}: {len(team_rosters_raw[tname])} players")

    print("Loading preseason projections from Redis...")
    hitter_rows = get_blended_projections(client, "hitters") or []
    pitcher_rows = get_blended_projections(client, "pitchers") or []
    if not hitter_rows or not pitcher_rows:
        raise RuntimeError(
            "Preseason projections not found in Redis "
            "(blended_projections:hitters / blended_projections:pitchers). "
            "Run `python scripts/build_db.py` first."
        )
    hitters_proj = pd.DataFrame(hitter_rows)
    pitchers_proj = pd.DataFrame(pitcher_rows)
    hitters_proj["_name_norm"] = hitters_proj["name"].apply(normalize_name)
    pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)

    print("Matching rosters to projections...")
    team_rosters: dict[str, list] = {}
    for tname, raw in team_rosters_raw.items():
        team_rosters[tname] = match_roster_to_projections(
            raw, hitters_proj, pitchers_proj,
            context=f"preseason_baseline:{tname}",
        )

    h_slots = sum(
        v for k, v in config.roster_slots.items()
        if k not in ("P", "BN", "IL", "DL")
    )
    p_slots = config.roster_slots.get("P", 9)

    print("Running base Monte Carlo (1000 iterations)...")
    base = run_monte_carlo(
        team_rosters, h_slots, p_slots, config.team_name,
        n_iterations=1000, use_management=False,
    )
    print("Running with-management Monte Carlo (1000 iterations)...")
    with_mgmt = run_monte_carlo(
        team_rosters, h_slots, p_slots, config.team_name,
        n_iterations=1000, use_management=True,
    )

    payload = {
        "base": base,
        "with_management": with_mgmt,
        "meta": {
            "frozen_at": _dt.datetime.now(_dt.UTC).isoformat(),
            "season_year": season_year,
            "roster_date": config.season_start,
            "projections_source": "blended",
        },
    }
    set_preseason_baseline(client, season_year, payload)
    print(
        f"Wrote preseason_baseline:{season_year} "
        f"({len(team_rosters)} teams)."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_scripts/test_freeze_preseason_baseline.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/freeze_preseason_baseline.py tests/test_scripts/__init__.py tests/test_scripts/test_freeze_preseason_baseline.py
git commit -m "feat(scripts): add freeze_preseason_baseline.py"
```

---

## Task 3: Refresh pipeline — delete `_run_monte_carlo`, read baseline from Redis

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py` — delete method, update cache write.
- Modify: `tests/test_web/_refresh_fixture.py` — seed baseline in fake Redis; drop preseason-MC scaling.
- Modify: `tests/test_web/test_refresh_pipeline.py` — extend assertions to new semantics.

- [ ] **Step 1: Update the refresh fixture to seed a canned baseline**

In `tests/test_web/_refresh_fixture.py`, find the section around line 314-327 (`# Capture the real Monte Carlo functions BEFORE patching them`) and the `_scaled_mc` wrapper. Split preseason from ROS handling:

Replace the `_scaled_mc` definition block (currently lines ~314-327 up to the patches list) with:

```python
    # Capture the real Monte Carlo function BEFORE patching it,
    # otherwise the scaled wrapper calls itself recursively.
    from fantasy_baseball.simulation import (
        run_ros_monte_carlo as _real_ros_mc,
    )

    # Seed a canned preseason baseline so refresh reads it from Redis
    # instead of running the (now-deleted) preseason MC live.
    from fantasy_baseball.data.redis_store import set_preseason_baseline
    _canned_mc = {
        "team_results": {
            tname: {
                "median_pts": 70.0, "p10": 60.0, "p90": 80.0,
                "first_pct": 8.0, "top3_pct": 25.0,
            }
            for tname in rosters
        },
        "category_risk": {
            cat: {"median_pts": 7.0, "p10": 4.0, "p90": 10.0,
                  "top3_pct": 25.0, "bot3_pct": 20.0}
            for cat in ("R", "HR", "RBI", "SB", "AVG",
                        "W", "K", "SV", "ERA", "WHIP")
        },
    }
    set_preseason_baseline(fake_redis, 2026, {
        "base": _canned_mc,
        "with_management": _canned_mc,
        "meta": {
            "frozen_at": "2026-04-17T00:00:00Z",
            "season_year": 2026,
            "roster_date": "2026-03-27",
            "projections_source": "blended",
        },
    })

    def _scaled_ros_mc(*, team_rosters, actual_standings, fraction_remaining,
                       h_slots, p_slots, user_team_name,
                       n_iterations=1000, use_management=False, progress_cb=None):
        return _real_ros_mc(
            team_rosters=team_rosters, actual_standings=actual_standings,
            fraction_remaining=fraction_remaining,
            h_slots=h_slots, p_slots=p_slots, user_team_name=user_team_name,
            n_iterations=10, use_management=use_management,
            progress_cb=progress_cb,
        )
```

Then remove the `run_monte_carlo` patch from the `patches = [...]` list (the line `patch("fantasy_baseball.simulation.run_monte_carlo", side_effect=_scaled_mc)`).

Also update the docstring around line 221 from `run_monte_carlo, run_ros_monte_carlo: 10 iters instead of 1000` to `run_ros_monte_carlo: 10 iters instead of 1000; preseason baseline seeded in Redis`.

- [ ] **Step 2: Add a new assertion class in the integration test for the baseline-missing branch**

In `tests/test_web/test_refresh_pipeline.py`, extend `TestMonteCarloROSBranch` with a new parametrized class (append at end of file):

```python
class TestPreseasonBaseline:
    """The refresh reads preseason_baseline:{year} from Redis; if
    missing, the base/with_management cache fields are None but the
    refresh still completes."""

    def test_baseline_present_populates_cache(
        self, configured_test_env, fake_redis,
    ):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "monte_carlo")
        assert data["base"] is not None
        assert data["with_management"] is not None
        assert "team_results" in data["base"]
        assert data["baseline_meta"]["roster_date"] == "2026-03-27"

    def test_baseline_missing_leaves_none(
        self, configured_test_env, fake_redis,
    ):
        cache_dir = configured_test_env
        # Intentionally strip the baseline that the fixture seeds.
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            fake_redis.delete("preseason_baseline:2026")
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "monte_carlo")
        assert data["base"] is None
        assert data["with_management"] is None
        assert data["baseline_meta"] is None
```

- [ ] **Step 3: Run the two new integration tests to verify they fail**

Run: `pytest tests/test_web/test_refresh_pipeline.py::TestPreseasonBaseline -v`
Expected: both FAIL. `test_baseline_present_populates_cache` fails on `data["baseline_meta"]["roster_date"]` KeyError (no `baseline_meta` yet). `test_baseline_missing_leaves_none` fails because `base` is still populated by the live preseason MC.

- [ ] **Step 4: Delete the preseason MC codepath from the refresh pipeline**

In `src/fantasy_baseball/web/refresh_pipeline.py`:

(a) Delete `self.base_mc = None` and `self.mgmt_mc = None` (lines 149-150 in the current file — the `__init__` block that sets default attributes).

(b) Delete the entire `_run_monte_carlo` method block (currently lines 747-770 including the `# --- Step 12: Monte Carlo simulation ---` header comment).

(c) Delete the call `self._run_monte_carlo()` from the `run()` sequence (currently line 191).

(d) Rewrite the cache-write block at the bottom of `_run_ros_monte_carlo` (currently lines 829-834). Replace:

```python
        write_cache("monte_carlo", {
            "base": self.base_mc,
            "with_management": self.mgmt_mc,
            "rest_of_season": self.rest_of_season_mc,
            "rest_of_season_with_management": self.rest_of_season_mgmt_mc,
        }, self.cache_dir)
```

with:

```python
        from fantasy_baseball.data.redis_store import (
            get_default_client as _get_redis_client,
            get_preseason_baseline,
        )
        _redis_client = _get_redis_client()
        baseline = (
            get_preseason_baseline(_redis_client, self.config.season_year)
            if _redis_client is not None else None
        ) or {}
        if not baseline:
            self._progress(
                "Preseason baseline missing — "
                "run scripts/freeze_preseason_baseline.py"
            )

        write_cache("monte_carlo", {
            "base": baseline.get("base"),
            "with_management": baseline.get("with_management"),
            "baseline_meta": baseline.get("meta"),
            "rest_of_season": self.rest_of_season_mc,
            "rest_of_season_with_management": self.rest_of_season_mgmt_mc,
        }, self.cache_dir)
```

- [ ] **Step 5: Run the new integration tests to verify they pass**

Run: `pytest tests/test_web/test_refresh_pipeline.py::TestPreseasonBaseline -v`
Expected: 2 passed.

- [ ] **Step 6: Run the full refresh pipeline test suite**

Run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: all tests pass. If `TestMonteCarloROSBranch.test_monte_carlo_keys_match_ros_availability` still expects `base is not None` when the ROS branch is off, it will still pass because the fixture seeds the baseline (the `has_rest_of_season` flag only affects ROS, not the preseason baseline).

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py tests/test_web/_refresh_fixture.py tests/test_web/test_refresh_pipeline.py
git commit -m "refactor(refresh): read preseason baseline from Redis, drop live preseason MC"
```

---

## Task 4: Route + template — label update, `baseline_meta` passthrough

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py` (`/standings` handler around lines 248-278)
- Modify: `src/fantasy_baseball/web/templates/season/standings.html` (lines 127-211)
- Modify: `tests/test_web/test_season_routes.py` — add a test for `baseline_meta` in the rendered context.

- [ ] **Step 1: Inspect the existing test for `/standings`**

Run: `pytest tests/test_web/test_season_routes.py -v --collect-only`
Take note of the existing route-render tests so the new one follows the same fixture pattern.

- [ ] **Step 2: Write a failing route test**

In `tests/test_web/test_season_routes.py`, add:

```python
def test_standings_passes_baseline_meta_to_template(
    client, fake_redis, tmp_cache_dir,
):
    """When monte_carlo.baseline_meta is present in the cache, it is
    available to the template as `baseline_meta`."""
    from fantasy_baseball.web.season_data import write_cache

    # Seed the minimum cache state the /standings handler reads.
    # Include monte_carlo with a baseline_meta block.
    write_cache("monte_carlo", {
        "base": {"team_results": {}, "category_risk": {}},
        "with_management": {"team_results": {}, "category_risk": {}},
        "baseline_meta": {
            "frozen_at": "2026-04-17T00:00:00Z",
            "roster_date": "2026-03-27",
            "season_year": 2026,
        },
        "rest_of_season": None,
        "rest_of_season_with_management": None,
    }, tmp_cache_dir)
    # Other cache files the route reads — minimal stubs
    write_cache("standings", [], tmp_cache_dir)

    resp = client.get("/standings")
    assert resp.status_code == 200
    # The date string should appear in the rendered HTML (tooltip/subtext)
    assert b"2026-03-27" in resp.data
```

(If `tmp_cache_dir` / `client` fixtures differ in `test_season_routes.py`, match the existing pattern in that file — read the top of the file before writing.)

- [ ] **Step 3: Run the new test to verify it fails**

Run: `pytest tests/test_web/test_season_routes.py::test_standings_passes_baseline_meta_to_template -v`
Expected: FAIL — `2026-03-27` is not in the rendered HTML (no tooltip yet).

- [ ] **Step 4: Update the `/standings` route to guard + pass baseline_meta**

In `src/fantasy_baseball/web/season_routes.py`, inside the `standings()` handler, locate the block around lines 248-256:

```python
            raw_mc = read_cache("monte_carlo")
            if raw_mc:
                mc_data = format_monte_carlo_for_display(
                    raw_mc.get("base", raw_mc), config.team_name
                )
                if "with_management" in raw_mc:
                    mc_mgmt_data = format_monte_carlo_for_display(
                        raw_mc["with_management"], config.team_name
                    )
```

Replace with:

```python
            raw_mc = read_cache("monte_carlo")
            baseline_meta = None
            if raw_mc:
                baseline_meta = raw_mc.get("baseline_meta")
                if raw_mc.get("base"):
                    mc_data = format_monte_carlo_for_display(
                        raw_mc["base"], config.team_name
                    )
                if raw_mc.get("with_management"):
                    mc_mgmt_data = format_monte_carlo_for_display(
                        raw_mc["with_management"], config.team_name
                    )
```

Then in the `render_template` call at the bottom of the handler (currently around lines 266-278), add `baseline_meta=baseline_meta,` to the kwargs list:

```python
        return render_template(
            "season/standings.html",
            meta=meta,
            active_page="standings",
            standings=standings_data,
            preseason=preseason_data,
            current_projected=current_projected_data,
            mc=mc_data,
            mc_mgmt=mc_mgmt_data,
            baseline_meta=baseline_meta,
            rest_of_season_mc=rest_of_season_mc_data,
            rest_of_season_mgmt_mc=rest_of_season_mgmt_mc_data,
            categories=ALL_CATEGORIES,
        )
```

- [ ] **Step 5: Update the template to relabel and show the roster_date tooltip**

In `src/fantasy_baseball/web/templates/season/standings.html`:

(a) Change the MC tab labels at lines 130-133 from:

```html
        <button class="pill active" data-mctab="preseason" onclick="toggleMcTab(this)">Preseason</button>
        <button class="pill" data-mctab="preseason-mgmt" onclick="toggleMcTab(this)">Preseason + Mgmt</button>
        <button class="pill" data-mctab="current" onclick="toggleMcTab(this)">Current</button>
        <button class="pill" data-mctab="current-mgmt" onclick="toggleMcTab(this)">Current + Mgmt</button>
```

to:

```html
        <button class="pill active" data-mctab="preseason" onclick="toggleMcTab(this)">Opening Day</button>
        <button class="pill" data-mctab="preseason-mgmt" onclick="toggleMcTab(this)">Opening Day + Mgmt</button>
        <button class="pill" data-mctab="current" onclick="toggleMcTab(this)">Current</button>
        <button class="pill" data-mctab="current-mgmt" onclick="toggleMcTab(this)">Current + Mgmt</button>
```

(b) Inside `#mc-preseason` (lines 195-202), add a caption line above the `mc_team_table` call:

```html
    <div id="mc-preseason">
        {% if mc %}
        {% if baseline_meta and baseline_meta.roster_date %}
        <p class="baseline-caption" style="opacity: 0.7; margin-bottom: 0.5em;">
            Frozen at Opening Day rosters ({{ baseline_meta.roster_date }}).
        </p>
        {% endif %}
        {{ mc_team_table(mc) }}
        {{ cat_risk_table(mc) }}
        {% else %}
        <p class="placeholder-text">No Opening Day baseline available. Run <code>scripts/freeze_preseason_baseline.py</code>.</p>
        {% endif %}
    </div>
```

(c) Do the same inside `#mc-preseason-mgmt` (lines 204-211):

```html
    <div id="mc-preseason-mgmt" style="display: none;">
        {% if mc_mgmt %}
        {% if baseline_meta and baseline_meta.roster_date %}
        <p class="baseline-caption" style="opacity: 0.7; margin-bottom: 0.5em;">
            Frozen at Opening Day rosters ({{ baseline_meta.roster_date }}).
        </p>
        {% endif %}
        {{ mc_team_table(mc_mgmt) }}
        {{ cat_risk_table(mc_mgmt) }}
        {% else %}
        <p class="placeholder-text">No Opening Day baseline available. Run <code>scripts/freeze_preseason_baseline.py</code>.</p>
        {% endif %}
    </div>
```

- [ ] **Step 6: Run the route test to verify it passes**

Run: `pytest tests/test_web/test_season_routes.py::test_standings_passes_baseline_meta_to_template -v`
Expected: PASS.

- [ ] **Step 7: Visual sanity check (optional for agentic run — required for human run)**

If running locally with a live league, start the server and open `/standings`:

```bash
python scripts/run_season_dashboard.py
```

Click the Monte Carlo view. The tab labels should read "Opening Day" / "Opening Day + Mgmt" / "Current" / "Current + Mgmt". The two baseline tabs should show the `Frozen at Opening Day rosters (2026-03-27)` caption.

- [ ] **Step 8: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py src/fantasy_baseball/web/templates/season/standings.html tests/test_web/test_season_routes.py
git commit -m "feat(standings): relabel preseason MC as Opening Day + show freeze date"
```

---

## Task 5: Full verification checklist

**Files:** none modified — runs the repo-wide checks listed in CLAUDE.md ("FORCED VERIFICATION — END-OF-EFFORT CHECKLIST").

- [ ] **Step 1: Full test suite**

Run: `pytest -v`
Expected: all pass. If any test outside `test_web/` / `test_data/` / `test_scripts/` fails, it likely indicates a hidden import or stray reference to `_run_monte_carlo` / `base_mc` / `mgmt_mc` — grep for those tokens and clean up.

Grep for leftover references:

```bash
grep -rn "_run_monte_carlo\|\.base_mc\|\.mgmt_mc" src/ tests/ scripts/
```

Expected: the only remaining hits should be in the commit history (there should be NO source-code hits; the attribute and method are deleted).

- [ ] **Step 2: Lint**

Run: `ruff check .`
Expected: zero violations.

- [ ] **Step 3: Formatter**

Run: `ruff format --check .`
Expected: no drift. If it reports files, run `ruff format .` to fix and re-commit.

- [ ] **Step 4: Dead code**

Run: `vulture`
Expected: no NEW findings. Pre-existing findings unrelated to this change are acceptable — call them out if you see any.

- [ ] **Step 5: Type check the touched files**

Run: `mypy`
The changed files include `src/fantasy_baseball/data/redis_store.py` (under mypy coverage). Expected: pass with no new errors.

- [ ] **Step 6: Summary commit (if anything was fixed in steps 2-5)**

If the lint/format/type check surfaced small fixes, stage them and commit:

```bash
git add -u
git commit -m "chore: address lint/type findings from preseason baseline work"
```

If nothing to commit, skip.

---

## Post-plan manual step (not a code task)

After merging this work to `main`, run the script once to populate the 2026 baseline:

```bash
python scripts/freeze_preseason_baseline.py
```

Expected output ends with `Wrote preseason_baseline:2026 (12 teams).`. Subsequent refreshes read from this artifact and skip the live preseason MC.

---

## Self-Review Notes

Spec coverage check (cross-checked against `docs/superpowers/specs/2026-04-17-preseason-baseline-design.md`):

- **Storage** (spec §"Storage"): Task 1 implements `preseason_baseline:{season_year}` get/set with the documented payload shape. ✓
- **Generation script** (spec §"Generation script"): Task 2 implements all 8 flow steps, CLI args, and the `--force` guard. ✓
- **Redis store helpers** (spec §"Redis store helpers"): Task 1. ✓
- **Refresh pipeline changes** (spec §"Refresh pipeline changes"): Task 3 implements all four numbered items. ✓
- **UI changes** (spec §"UI changes"): Task 4 implements `baseline_meta` passthrough, tab relabel, tooltip/caption. ✓
- **Tests** (spec §"Tests"): Task 1 (Redis helpers), Task 2 (script), Task 3 (refresh pipeline baseline-present/missing), Task 4 (route). ✓
- **2026 bootstrap** (spec §"2026 bootstrap"): Covered in the post-plan manual step above. ✓
- **Risks / edge cases** (spec §"Risks / edge cases"): Yahoo historical fetch failure is handled by the script crashing loudly (Task 2's bootstrap run will surface it); traded players and unprojected waiver pickups require no code; rerun semantics enforced by the `--force` guard (Task 2 test `test_script_refuses_to_overwrite_without_force`). ✓

Type-consistency: `get_preseason_baseline` returns `dict | None`, `set_preseason_baseline` takes `dict`, Task 3 treats a falsy return as "missing baseline" via `or {}`. Consistent.

No placeholders — every task has concrete code, exact paths, and expected pytest output.
