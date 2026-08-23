"""Tests for ``scripts/run_manual_refresh.py``, the manual-pipeline entry point.

Two properties dominate this file, and both are safety properties rather than
features:

1. **The Yahoo baseline is unreachable.** A manual run writes hand-typed
   standings and rosters into ``cache:standings``, ``weekly_rosters_history``
   and ``standings_history``. If any of that landed in ``data/local.db`` the
   Yahoo baseline would be silently corrupted with data that LOOKS like a real
   refresh. So the driver refuses to start on a baseline path or under
   ``RENDER``, it refuses BEFORE setting any environment variable, and it
   verifies after the fact that the live ``get_kv()`` really did bind to the
   isolated store.

2. **No ``fantasy_baseball`` import happens at module scope.** ``get_kv()`` is a
   process-wide singleton that captures ``FANTASY_LOCAL_KV_PATH`` on its first
   call, so an import that reaches it before ``main()`` sets that variable
   would pin the whole process to ``data/local.db``. The invariant is enforced
   by walking this script's own AST -- a comment cannot survive a refactor, a
   test can.

Every test that calls ``main()`` points ``--kv-path`` at ``tmp_path``. The
``_restore_manual_env`` fixture is autouse because ``main()`` mutates
``os.environ`` by design, and a leaked ``FANTASY_LOCAL_KV_PATH`` would redirect
every later test in the session.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_manual_refresh as drv

DRIVER_SOURCE = Path(drv.__file__)

_STATS_A = {
    "R": 800,
    "HR": 250,
    "RBI": 750,
    "SB": 160,
    "AVG": 0.263,
    "W": 83,
    "SV": 76,
    "K": 1181,
    "ERA": 3.40,
    "WHIP": 1.12,
}
_STATS_B = {
    "R": 700,
    "HR": 200,
    "RBI": 650,
    "SB": 120,
    "AVG": 0.250,
    "W": 70,
    "SV": 60,
    "K": 1000,
    "ERA": 3.90,
    "WHIP": 1.25,
}


@pytest.fixture(autouse=True)
def _restore_manual_env():
    """Snapshot the two env vars ``main()`` sets, and reset the KV singleton.

    Restored by hand rather than through ``monkeypatch``, because the obvious
    monkeypatch spelling silently does nothing in the case that matters:

        monkeypatch.delenv(name, raising=False)   # name is NOT set

    ``delenv`` records an undo entry only when it actually removes something.
    On an already-absent name with ``raising=False`` it is a no-op that
    registers NOTHING -- so when ``main()`` later assigns the variable itself
    (``_activate_manual_environment`` sets both), there is no undo to run and
    the value escapes the test.

    That is not a hypothetical. It leaked ``FB_SKIP_YAHOO=1`` and a
    ``FANTASY_LOCAL_KV_PATH`` pointing at a torn-down ``tmp_path`` into every
    later test in the same process, and ``tests/test_web/test_refresh_pipeline.py``
    then died with "FB_SKIP_YAHOO is set but no cached standings exist". Under
    ``pytest -n auto`` (with ``pytest-randomly`` shuffling) that surfaced as a
    handful of unrelated files failing in whichever worker drew this module --
    green in isolation, red in the full suite.
    """
    from fantasy_baseball.data import kv_store

    names = ("FANTASY_LOCAL_KV_PATH", "FB_SKIP_YAHOO")
    saved = {name: drv.os.environ.get(name) for name in names}
    try:
        yield
    finally:
        for name, before in saved.items():
            if before is None:
                drv.os.environ.pop(name, None)
            else:
                drv.os.environ[name] = before
        kv_store._reset_singleton()


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _write_config(tmp_path: Path, *, team_name: str = "Alpha") -> Path:
    """A real ``league.yaml`` with a tiny roster and a two-team-friendly name.

    Derived from the committed config rather than hand-built so every field
    ``load_config`` requires stays present as that schema evolves.
    """
    raw = yaml.safe_load((PROJECT_ROOT / "config" / "league.yaml").read_text(encoding="utf-8"))
    raw["league"]["team_name"] = team_name
    raw["roster_slots"] = {"C": 1, "P": 1, "BN": 1, "IL": 1}
    path = tmp_path / "league.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _write_standings(tmp_path: Path, *, names: tuple[str, str] = ("Alpha", "Beta")) -> Path:
    payload = {
        "effective_date": "2026-08-22",
        "teams": [
            {"name": names[0], "rank": 1, "stats": _STATS_A, "points_for": 15},
            {"name": names[1], "rank": 2, "stats": _STATS_B, "points_for": 5},
        ],
    }
    path = tmp_path / "standings.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _write_rosters(tmp_path: Path, *, names: tuple[str, str] = ("Alpha", "Beta")) -> Path:
    def players(prefix: str) -> list[dict[str, str]]:
        return [
            {"name": f"{prefix} Catcher", "slot": "C", "positions": "C"},
            {"name": f"{prefix} Pitcher", "slot": "P", "positions": "P"},
            {"name": f"{prefix} Bench", "slot": "BN", "positions": "OF"},
        ]

    payload = {
        "snapshot_date": "2026-08-22",
        "teams": [{"name": n, "players": players(n)} for n in names],
    }
    path = tmp_path / "rosters.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


@pytest.fixture
def good_inputs(tmp_path, monkeypatch):
    """A validating pair of transcriptions plus a matching config."""
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(drv, "CONFIG_PATH", config_path)
    return {
        "config": config_path,
        "standings": _write_standings(tmp_path),
        "rosters": _write_rosters(tmp_path),
        "kv": tmp_path / "manual.db",
    }


def _dry_run_argv(good_inputs: dict, *extra: str) -> list[str]:
    return [
        "--dry-run",
        "--kv-path",
        str(good_inputs["kv"]),
        "--rosters",
        str(good_inputs["rosters"]),
        "--standings",
        str(good_inputs["standings"]),
        "--exclusions",
        str(good_inputs["rosters"].parent / "absent.yaml"),
        *extra,
    ]


# --------------------------------------------------------------------------
# Import-order invariant
# --------------------------------------------------------------------------


def test_driver_has_no_top_level_fantasy_baseball_import():
    """The whole isolation mechanism rests on this. See the module docstring."""
    source = DRIVER_SOURCE.read_text(encoding="utf-8")
    assert drv.module_level_fantasy_imports(source) == []


def test_detector_catches_a_plain_top_level_import():
    source = "import fantasy_baseball.data.kv_store\n"
    assert drv.module_level_fantasy_imports(source) == ["fantasy_baseball.data.kv_store"]


def test_detector_catches_a_top_level_from_import():
    source = "from fantasy_baseball.config import load_config\n"
    assert drv.module_level_fantasy_imports(source) == ["fantasy_baseball.config"]


def test_detector_catches_an_import_hidden_in_a_top_level_try():
    source = (
        "try:\n    from fantasy_baseball import config\nexcept ImportError:\n    config = None\n"
    )
    assert drv.module_level_fantasy_imports(source) == ["fantasy_baseball"]


def test_detector_ignores_imports_inside_a_function():
    source = (
        "def f():\n    from fantasy_baseball.config import load_config\n    return load_config\n"
    )
    assert drv.module_level_fantasy_imports(source) == []


def test_detector_ignores_the_type_checking_block():
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from fantasy_baseball.config import LeagueConfig\n"
    )
    assert drv.module_level_fantasy_imports(source) == []


def test_detector_ignores_unrelated_top_level_imports():
    source = "import os\nimport pandas as pd\nfrom pathlib import Path\n"
    assert drv.module_level_fantasy_imports(source) == []


# --------------------------------------------------------------------------
# Startup guards
# --------------------------------------------------------------------------


def test_banner_prints_the_resolved_absolute_kv_path_first(capsys, good_inputs):
    drv.main(_dry_run_argv(good_inputs))
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("====")
    assert str(good_inputs["kv"].resolve()) in lines[2]
    assert "DRY RUN" in lines[3]


def test_live_mode_is_labelled_differently_from_dry_run(capsys, good_inputs, monkeypatch):
    """The banner must never say the same thing for a read-only and a writing run."""
    drv.main(_dry_run_argv(good_inputs))
    dry = capsys.readouterr().out.splitlines()[3]
    # A live run refuses immediately (the store does not exist), which is
    # enough to see the banner it printed first.
    argv = [a for a in _dry_run_argv(good_inputs) if a != "--dry-run"]
    drv.main(argv)
    live = capsys.readouterr().out.splitlines()[3]
    assert dry != live
    assert "LIVE" in live


@pytest.mark.parametrize("value", ["true", "1", "false"])
def test_refuses_when_render_is_set(capsys, good_inputs, monkeypatch, value):
    """Stricter than ``kv_store.is_remote()``: any non-empty RENDER refuses."""
    monkeypatch.setenv("RENDER", value)
    assert drv.main(_dry_run_argv(good_inputs)) == drv.RC_REFUSED
    out = capsys.readouterr().out
    assert "REFUSING TO RUN" in out
    assert "RENDER" in out


def test_empty_render_is_not_a_refusal(good_inputs, monkeypatch):
    monkeypatch.setenv("RENDER", "")
    assert drv.main(_dry_run_argv(good_inputs)) == drv.RC_OK


def test_refuses_the_repo_baseline_store(capsys, good_inputs):
    argv = _dry_run_argv(good_inputs)
    argv[argv.index("--kv-path") + 1] = "data/local.db"
    assert drv.main(argv) == drv.RC_REFUSED
    assert "Yahoo baseline" in capsys.readouterr().out


def test_refuses_any_store_named_local_db(capsys, good_inputs, tmp_path):
    argv = _dry_run_argv(good_inputs)
    argv[argv.index("--kv-path") + 1] = str(tmp_path / "local.db")
    assert drv.main(argv) == drv.RC_REFUSED
    assert "Yahoo baseline" in capsys.readouterr().out


def test_refuses_local_db_case_insensitively(good_inputs, tmp_path):
    argv = _dry_run_argv(good_inputs)
    argv[argv.index("--kv-path") + 1] = str(tmp_path / "LOCAL.DB")
    assert drv.main(argv) == drv.RC_REFUSED


def test_a_refused_run_never_sets_the_isolation_env_vars(good_inputs, monkeypatch):
    """The guard runs BEFORE ``_activate_manual_environment``.

    If it did not, a refused run would still have pointed the process (and
    ``FB_SKIP_YAHOO``) somewhere, and a caller that ignored the return code
    would proceed against a half-configured environment.
    """
    monkeypatch.delenv("FANTASY_LOCAL_KV_PATH", raising=False)
    monkeypatch.delenv("FB_SKIP_YAHOO", raising=False)
    argv = _dry_run_argv(good_inputs)
    argv[argv.index("--kv-path") + 1] = "data/local.db"
    assert drv.main(argv) == drv.RC_REFUSED
    assert "FANTASY_LOCAL_KV_PATH" not in drv.os.environ
    assert "FB_SKIP_YAHOO" not in drv.os.environ


def test_relative_kv_paths_anchor_at_the_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert drv.resolve_path("data/manual.db") == (PROJECT_ROOT / "data" / "manual.db").resolve()


def test_absolute_kv_paths_are_left_alone(tmp_path):
    target = tmp_path / "elsewhere.db"
    assert drv.resolve_path(target) == target.resolve()


def test_activate_binds_get_kv_to_the_requested_store(tmp_path):
    from fantasy_baseball.data.kv_store import get_kv
    from fantasy_baseball.manual.seed import resolve_kv_path

    target = (tmp_path / "manual.db").resolve()
    drv._activate_manual_environment(target)
    assert drv.os.environ["FANTASY_LOCAL_KV_PATH"] == str(target)
    assert drv.os.environ["FB_SKIP_YAHOO"] == "1"
    assert resolve_kv_path(get_kv()) == target


def test_activate_discards_a_singleton_built_against_another_store(tmp_path, monkeypatch):
    """The exact failure the belt-and-braces reset exists for."""
    from fantasy_baseball.data.kv_store import get_kv
    from fantasy_baseball.manual.seed import resolve_kv_path

    stale = (tmp_path / "stale.db").resolve()
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(stale))
    assert resolve_kv_path(get_kv()) == stale

    target = (tmp_path / "manual.db").resolve()
    drv._activate_manual_environment(target)
    assert resolve_kv_path(get_kv()) == target


def test_verify_kv_target_accepts_the_bound_store(tmp_path):
    target = (tmp_path / "manual.db").resolve()
    drv._activate_manual_environment(target)
    assert drv._verify_kv_target(target) == drv.RC_OK


def test_verify_kv_target_refuses_a_mismatch(capsys, tmp_path):
    drv._activate_manual_environment((tmp_path / "manual.db").resolve())
    assert drv._verify_kv_target((tmp_path / "other.db").resolve()) == drv.RC_REFUSED
    assert "did not bind to the manual store" in capsys.readouterr().out


# --------------------------------------------------------------------------
# --dry-run
# --------------------------------------------------------------------------


def test_dry_run_succeeds_on_valid_transcriptions(capsys, good_inputs):
    assert drv.main(_dry_run_argv(good_inputs)) == drv.RC_OK
    out = capsys.readouterr().out
    assert "Transcriptions OK" in out
    assert "DRY RUN -- validation only" in out


def test_dry_run_opens_no_kv_store(good_inputs):
    """``--dry-run`` must not create the store, not even an empty one."""
    assert not good_inputs["kv"].exists()
    assert drv.main(_dry_run_argv(good_inputs)) == drv.RC_OK
    assert not good_inputs["kv"].exists()
    for suffix in ("-wal", "-shm", "-journal"):
        assert not Path(str(good_inputs["kv"]) + suffix).exists()


def test_dry_run_writes_no_report_file(good_inputs, tmp_path):
    out_file = tmp_path / "audit.txt"
    assert drv.main(_dry_run_argv(good_inputs, "--report-out", str(out_file))) == drv.RC_OK
    assert not out_file.exists()


def test_dry_run_flags_a_missing_kv_store(capsys, good_inputs):
    drv.main(_dry_run_argv(good_inputs))
    assert "MISSING; run scripts/bootstrap_manual_kv.py" in capsys.readouterr().out


def test_dry_run_does_not_flag_an_existing_kv_store(capsys, good_inputs):
    good_inputs["kv"].write_bytes(b"")
    drv.main(_dry_run_argv(good_inputs))
    assert "MISSING" not in capsys.readouterr().out


def test_dry_run_lists_active_and_il_counts_per_team(capsys, good_inputs):
    drv.main(_dry_run_argv(good_inputs))
    out = capsys.readouterr().out
    assert "* Alpha" in out  # the user's own team is marked
    assert "3 active + 0 IL" in out


def test_dry_run_flags_a_short_roster(capsys, tmp_path, monkeypatch):
    """A skipped screenshot row is the most likely transcription error."""
    monkeypatch.setattr(drv, "CONFIG_PATH", _write_config(tmp_path))
    rosters = yaml.safe_load(_write_rosters(tmp_path).read_text(encoding="utf-8"))
    rosters["teams"][1]["players"].pop()  # Beta loses its bench bat
    path = tmp_path / "short.yaml"
    path.write_text(yaml.safe_dump(rosters), encoding="utf-8")
    argv = [
        "--dry-run",
        "--kv-path",
        str(tmp_path / "manual.db"),
        "--rosters",
        str(path),
        "--standings",
        str(_write_standings(tmp_path)),
    ]
    assert drv.main(argv) == drv.RC_OK
    out = capsys.readouterr().out
    assert "2 active + 0 IL   <-- 3 active slots; re-check" in out


def test_dry_run_reports_ros_snapshot_staleness(capsys, good_inputs):
    """The 2-day ROS freshness guard is what actually blocks a live run today."""
    drv.main(_dry_run_argv(good_inputs, "--season", "1901"))
    out = capsys.readouterr().out
    assert "no dated snapshot under data/projections/1901/rest_of_season" in out
    assert "ACTION: stage a fresh FanGraphs export" in out


def test_dry_run_honours_skip_flags_in_the_plan(capsys, good_inputs):
    drv.main(_dry_run_argv(good_inputs, "--skip-blend", "--skip-game-logs"))
    out = capsys.readouterr().out
    assert "SKIP the MLB game-log sync" in out
    assert "SKIP the ROS blend" in out


def test_dry_run_previews_the_default_report_path(capsys, good_inputs):
    drv.main(_dry_run_argv(good_inputs))
    expected = drv.DEFAULT_REPORT_DIR / f"audit-{drv._STAMP_PLACEHOLDER}.txt"
    assert str(expected) in capsys.readouterr().out


def test_dry_run_warns_that_free_agents_have_no_injury_status(capsys, good_inputs):
    drv.main(_dry_run_argv(good_inputs))
    assert "no injury status" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Validation failures: explicit and actionable, never a stack trace
# --------------------------------------------------------------------------


def test_a_still_template_roster_file_gets_an_actionable_message(capsys, tmp_path, monkeypatch):
    """The shipped template fails on the team-set mismatch, which explains nothing."""
    monkeypatch.setattr(drv, "CONFIG_PATH", _write_config(tmp_path))
    template = tmp_path / "rosters.yaml"
    template.write_text(
        "# REPLACE-ME: transcribe the real rosters here\n"
        + yaml.safe_dump(
            {
                "snapshot_date": "2026-08-22",
                "teams": [
                    {
                        "name": "Example Team",
                        "players": [{"name": "Example Player", "slot": "C", "positions": "C"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    argv = [
        "--dry-run",
        "--kv-path",
        str(tmp_path / "manual.db"),
        "--rosters",
        str(template),
        "--standings",
        str(_write_standings(tmp_path)),
    ]
    assert drv.main(argv) == drv.RC_FAILED
    out = capsys.readouterr().out
    assert "What to do:" in out
    assert "REPLACE-ME" in out
    assert "has not been transcribed yet" in out
    assert "Traceback" not in out


def test_a_missing_roster_file_is_reported_not_raised(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "CONFIG_PATH", _write_config(tmp_path))
    argv = [
        "--dry-run",
        "--kv-path",
        str(tmp_path / "manual.db"),
        "--rosters",
        str(tmp_path / "nope.yaml"),
        "--standings",
        str(_write_standings(tmp_path)),
    ]
    assert drv.main(argv) == drv.RC_FAILED
    out = capsys.readouterr().out
    assert "does not exist" in out
    assert "Traceback" not in out


def test_every_validation_error_is_printed_not_just_the_first(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "CONFIG_PATH", _write_config(tmp_path))
    rosters = yaml.safe_load(_write_rosters(tmp_path).read_text(encoding="utf-8"))
    rosters["teams"][0]["name"] = "Alfa"  # typo: two team-set errors plus the config one
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(rosters), encoding="utf-8")
    argv = [
        "--dry-run",
        "--kv-path",
        str(tmp_path / "manual.db"),
        "--rosters",
        str(path),
        "--standings",
        str(_write_standings(tmp_path)),
    ]
    assert drv.main(argv) == drv.RC_FAILED
    out = capsys.readouterr().out
    assert out.count("  * ") >= 3
    assert "not in rosters.yaml" in out


def test_a_bad_slot_names_the_player_and_the_team(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "CONFIG_PATH", _write_config(tmp_path))
    rosters = yaml.safe_load(_write_rosters(tmp_path).read_text(encoding="utf-8"))
    rosters["teams"][0]["players"][0]["slot"] = ""
    path = tmp_path / "badslot.yaml"
    path.write_text(yaml.safe_dump(rosters), encoding="utf-8")
    argv = [
        "--dry-run",
        "--kv-path",
        str(tmp_path / "manual.db"),
        "--rosters",
        str(path),
        "--standings",
        str(_write_standings(tmp_path)),
    ]
    assert drv.main(argv) == drv.RC_FAILED
    out = capsys.readouterr().out
    assert "Alpha Catcher" in out
    assert "Traceback" not in out


def test_template_hints_flag_an_absent_file(tmp_path):
    hints = drv.template_hints({"rosters.yaml": tmp_path / "gone.yaml"})
    assert len(hints) == 1
    assert "does not exist" in hints[0]


def test_template_hints_stay_quiet_on_a_real_transcription(tmp_path):
    assert drv.template_hints({"rosters.yaml": _write_rosters(tmp_path)}) == []


# --------------------------------------------------------------------------
# Live-path wiring
# --------------------------------------------------------------------------


def test_live_run_refuses_when_the_manual_store_is_missing(capsys, good_inputs):
    argv = [a for a in _dry_run_argv(good_inputs) if a != "--dry-run"]
    assert drv.main(argv) == drv.RC_REFUSED
    out = capsys.readouterr().out
    assert "bootstrap_manual_kv.py" in out
    assert not good_inputs["kv"].exists()


def test_per_position_cap_of_zero_is_honoured_not_defaulted(capsys, monkeypatch):
    """``0`` is falsy; ``args.per_position_cap or DEFAULT`` would silently ignore it."""
    import fantasy_baseball.keepers.positions as positions_mod
    from fantasy_baseball.manual.free_agents import build_manual_free_agents

    monkeypatch.setattr(positions_mod, "load_positions", lambda *a, **k: {"a b": ["C"]})
    args = drv._build_parser().parse_args(["--per-position-cap", "0"])
    source = drv._build_free_agent_source(args, frozenset())
    assert source.func is build_manual_free_agents
    assert source.keywords["per_position_cap"] == 0


def test_free_agent_source_carries_positions_and_exclusions(monkeypatch):
    import fantasy_baseball.keepers.positions as positions_mod
    from fantasy_baseball.manual.free_agents import DEFAULT_PER_POSITION_CAP

    monkeypatch.setattr(positions_mod, "load_positions", lambda *a, **k: {"a b": ["C"]})
    args = drv._build_parser().parse_args([])
    source = drv._build_free_agent_source(args, frozenset({"juan soto"}))
    assert source.keywords["positions_by_name"] == {"a b": ["C"]}
    assert source.keywords["excluded_names"] == frozenset({"juan soto"})
    assert source.keywords["per_position_cap"] == DEFAULT_PER_POSITION_CAP


def test_run_pipeline_passes_the_manual_kwargs_to_refresh_run(monkeypatch):
    """skip_yahoo=True is what keeps ``_push_streak_scores_to_remote`` unreachable."""
    import fantasy_baseball.keepers.positions as positions_mod
    import fantasy_baseball.web.refresh_pipeline as pipeline_mod

    monkeypatch.setattr(positions_mod, "load_positions", lambda *a, **k: {"a b": ["C"]})
    seen: dict[str, object] = {}

    class _Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def run(self):
            seen["ran"] = True

    monkeypatch.setattr(pipeline_mod, "RefreshRun", _Recorder)
    args = drv._build_parser().parse_args([])
    assert drv._run_pipeline(args, frozenset()) == drv.RC_OK
    assert seen["skip_yahoo"] is True
    assert seen["job_label"] == "manual"
    assert seen["free_agent_source"] is not None
    assert seen["ran"] is True


def test_game_log_sync_refuses_an_empty_rollup(capsys, monkeypatch):
    """An empty rollup fails silently downstream -- full-season collapses to ROS."""
    import fantasy_baseball.data.mlb_game_logs as logs_mod

    monkeypatch.setattr(logs_mod, "fetch_game_log_totals", lambda *a, **k: ({}, {"1": {}}, 0))
    assert drv._sync_game_logs(2026) == drv.RC_FAILED
    assert "came back empty" in capsys.readouterr().out


def test_game_log_sync_accepts_populated_rollups(monkeypatch):
    import fantasy_baseball.data.mlb_game_logs as logs_mod

    monkeypatch.setattr(
        logs_mod, "fetch_game_log_totals", lambda *a, **k: ({"1": {}}, {"2": {}}, 9)
    )
    assert drv._sync_game_logs(2026) == drv.RC_OK


def test_blend_reports_a_stale_snapshot_instead_of_raising(capsys, monkeypatch, tmp_path):
    import fantasy_baseball.data.ros_pipeline as ros_mod
    from fantasy_baseball.config import load_config

    def _raise(*_args, **_kwargs):
        raise ros_mod.StaleROSSnapshotError("snapshot 2026-07-21 is 32 days stale")

    monkeypatch.setattr(ros_mod, "blend_and_cache_ros", _raise)
    config = load_config(_write_config(tmp_path))
    assert drv._blend_ros(2026, config) == drv.RC_FAILED
    out = capsys.readouterr().out
    assert "32 days stale" in out
    assert "--skip-blend" in out


def test_blend_passes_config_systems_and_no_roster_names(monkeypatch, tmp_path):
    """``roster_names=None`` is deliberate: it only feeds a coverage log line."""
    import pandas as pd

    import fantasy_baseball.data.ros_pipeline as ros_mod
    from fantasy_baseball.config import load_config

    captured: dict[str, object] = {}

    def _fake(projections_dir, systems, weights, roster_names, season_year, progress_cb=None):
        captured.update(
            projections_dir=projections_dir,
            systems=systems,
            weights=weights,
            roster_names=roster_names,
            season_year=season_year,
        )
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(ros_mod, "blend_and_cache_ros", _fake)
    config = load_config(_write_config(tmp_path))
    assert drv._blend_ros(2026, config) == drv.RC_OK
    assert captured["roster_names"] is None
    assert captured["season_year"] == 2026
    assert captured["systems"] == list(config.projection_systems)
    assert captured["projections_dir"] == PROJECT_ROOT / "data" / "projections"


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def _standings_for(tmp_path: Path):
    from fantasy_baseball.manual.transcripts import load_manual_standings

    return load_manual_standings(_write_standings(tmp_path), team_keys={})


def test_current_roto_standings_prefer_yahoos_own_totals(tmp_path):
    """Re-scoring locally splits Yahoo's display ties and disagrees with the page."""
    standings = _standings_for(tmp_path)
    assert drv._current_roto_standings(standings) == [("Alpha", 15.0), ("Beta", 5.0)]


def test_current_roto_standings_fall_back_to_score_roto(tmp_path):
    from fantasy_baseball.scoring import score_roto

    standings = _standings_for(tmp_path)
    for entry in standings.entries:
        entry.yahoo_points_for = None
    expected = score_roto(standings)
    assert drv._current_roto_standings(standings) == [
        ("Alpha", expected["Alpha"].total),
        ("Beta", expected["Beta"].total),
    ]


def test_current_roto_standings_follow_yahoos_rank_order(tmp_path):
    standings = _standings_for(tmp_path)
    standings.entries[0].rank = 2
    standings.entries[1].rank = 1
    assert [name for name, _ in drv._current_roto_standings(standings)] == ["Beta", "Alpha"]


def test_report_path_defaults_to_the_snapshot_date():
    args = drv._build_parser().parse_args([])
    assert drv._report_path(args, "2026-08-22") == drv.DEFAULT_REPORT_DIR / "audit-2026-08-22.txt"


def test_report_path_honours_the_override(tmp_path):
    args = drv._build_parser().parse_args(["--report-out", str(tmp_path / "x.txt")])
    assert drv._report_path(args, "2026-08-22") == (tmp_path / "x.txt").resolve()


def _audit_payload() -> list[dict]:
    return [
        {
            "player": "Alpha Catcher",
            "player_type": "hitter",
            "positions": ["C"],
            "slot": "C",
            "player_sgp": 3.5,
            "player_id": "Alpha Catcher::hitter",
        }
    ]


@pytest.fixture
def stub_caches(monkeypatch):
    """Stand in for the caches the pipeline would have written."""
    import fantasy_baseball.web.season_data as season_data
    from fantasy_baseball.data.cache_keys import CacheKey

    payloads = {
        CacheKey.ROSTER_AUDIT: _audit_payload(),
        CacheKey.PROJECTIONS: {"fraction_remaining": 0.11},
    }
    monkeypatch.setattr(season_data, "read_cache_list", lambda key: payloads.get(key))
    monkeypatch.setattr(season_data, "read_cache_dict", lambda key: payloads.get(key))
    monkeypatch.setattr(
        season_data,
        "read_cache_with_meta",
        lambda key: (None, {"_ros_snapshot_date": "2026-08-22"}),
    )
    return payloads


def test_render_report_writes_and_prints_the_report(capsys, tmp_path, stub_caches, monkeypatch):
    from fantasy_baseball.config import load_config
    from fantasy_baseball.manual.transcripts import load_manual_rosters

    config = load_config(_write_config(tmp_path))
    rosters = load_manual_rosters(_write_rosters(tmp_path))
    standings = _standings_for(tmp_path)
    out_file = tmp_path / "audit.txt"
    args = drv._build_parser().parse_args(["--report-out", str(out_file)])

    rc = drv._render_report(args, config, standings, rosters, tmp_path / "manual.db")

    assert rc == drv.RC_OK
    assert out_file.is_file()
    written = out_file.read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    assert "Alpha Catcher" in written
    assert written.rstrip("\n") in printed
    assert str(tmp_path / "manual.db") in written
    assert "11.0%" in written  # fraction_remaining came from the cache, not a default
    assert "2026-08-22" in written


def test_render_report_names_the_default_output_file_after_the_snapshot(
    tmp_path, stub_caches, monkeypatch
):
    from fantasy_baseball.config import load_config
    from fantasy_baseball.manual.transcripts import load_manual_rosters

    monkeypatch.setattr(drv, "DEFAULT_REPORT_DIR", tmp_path / "out")
    config = load_config(_write_config(tmp_path))
    rosters = load_manual_rosters(_write_rosters(tmp_path))
    args = drv._build_parser().parse_args([])

    drv._render_report(args, config, _standings_for(tmp_path), rosters, tmp_path / "manual.db")

    assert (tmp_path / "out" / "audit-2026-08-22.txt").is_file()


def test_render_report_flags_players_the_audit_dropped(capsys, tmp_path, stub_caches):
    """Three players transcribed, one audit entry back -- two names went missing."""
    from fantasy_baseball.config import load_config
    from fantasy_baseball.manual.transcripts import load_manual_rosters

    config = load_config(_write_config(tmp_path))
    rosters = load_manual_rosters(_write_rosters(tmp_path))
    args = drv._build_parser().parse_args(["--report-out", str(tmp_path / "audit.txt")])

    drv._render_report(args, config, _standings_for(tmp_path), rosters, tmp_path / "manual.db")

    out = capsys.readouterr().out
    assert "3 players transcribed for Alpha but the audit returned 1" in out


def test_render_report_survives_an_empty_audit(capsys, tmp_path, monkeypatch):
    """An empty audit means the roster came back empty -- NOT that it is optimal."""
    import fantasy_baseball.web.season_data as season_data
    from fantasy_baseball.config import load_config
    from fantasy_baseball.manual.transcripts import load_manual_rosters

    monkeypatch.setattr(season_data, "read_cache_list", lambda key: None)
    monkeypatch.setattr(season_data, "read_cache_dict", lambda key: None)
    monkeypatch.setattr(season_data, "read_cache_with_meta", lambda key: (None, {}))
    config = load_config(_write_config(tmp_path))
    rosters = load_manual_rosters(_write_rosters(tmp_path))
    out_file = tmp_path / "audit.txt"
    args = drv._build_parser().parse_args(["--report-out", str(out_file)])

    assert (
        drv._render_report(args, config, _standings_for(tmp_path), rosters, tmp_path / "manual.db")
        == drv.RC_OK
    )
    assert "NOT that the roster is optimal" in out_file.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------


def test_the_driver_is_pure_ascii():
    """cp1252 stdout on this dev box: one stray glyph kills the run."""
    raw = DRIVER_SOURCE.read_bytes()
    assert [i for i, byte in enumerate(raw) if byte > 127] == []


def test_defaults_point_at_the_manual_store_and_transcriptions():
    assert drv.DEFAULT_KV_PATH == PROJECT_ROOT / "data" / "manual.db"
    assert drv.DEFAULT_ROSTERS == PROJECT_ROOT / "data" / "manual" / "rosters.yaml"
    assert drv.DEFAULT_STANDINGS == PROJECT_ROOT / "data" / "manual" / "standings.yaml"
    assert drv.DEFAULT_EXCLUSIONS == PROJECT_ROOT / "data" / "manual" / "fa_exclusions.yaml"
    assert drv.DEFAULT_KV_PATH.name != drv.BASELINE_DB_NAME


def test_exit_codes_are_distinct():
    assert len({drv.RC_OK, drv.RC_FAILED, drv.RC_REFUSED}) == 3


def test_season_falls_back_to_config_but_an_explicit_value_wins(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(drv, "CONFIG_PATH", _write_config(tmp_path))
    argv = [
        "--dry-run",
        "--kv-path",
        str(tmp_path / "manual.db"),
        "--rosters",
        str(_write_rosters(tmp_path)),
        "--standings",
        str(_write_standings(tmp_path)),
    ]
    drv.main(argv)
    assert "game logs for 2026" in capsys.readouterr().out
    drv.main([*argv, "--season", "2019"])
    assert "game logs for 2019" in capsys.readouterr().out


def test_effective_date_comes_from_the_roster_snapshot(tmp_path, stub_caches):
    """The report is stamped with the transcription's vintage, not today's date."""
    from fantasy_baseball.config import load_config
    from fantasy_baseball.manual.transcripts import load_manual_rosters

    config = load_config(_write_config(tmp_path))
    rosters = load_manual_rosters(_write_rosters(tmp_path))
    assert rosters.snapshot_date == date(2026, 8, 22)
    out_file = tmp_path / "audit.txt"
    args = drv._build_parser().parse_args(["--report-out", str(out_file)])
    drv._render_report(args, config, _standings_for(tmp_path), rosters, tmp_path / "manual.db")
    assert "2026-08-22" in out_file.read_text(encoding="utf-8")
