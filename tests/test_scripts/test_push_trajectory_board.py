"""Where `push_trajectory_board` writes, and what provenance it stamps.

The script had no tests at all. These cover the destination decision, which is the
part that can quietly send an unverified board to production.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from fantasy_baseball.data.kv_store import SqliteKVStore

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location(
        "push_trajectory_board", PROJECT_ROOT / "scripts" / "push_trajectory_board.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_empty_pool_refuses_the_push(monkeypatch) -> None:
    """A pool that scores nobody must not overwrite a good board.

    `panel_path` picks the newest panel per kind INDEPENDENTLY and the base season comes
    off the HITTER panel, so a stale pitcher panel makes `board_inputs(season=2026)` hit
    `current.empty` and return [] -- the run then prints "0 players with a 2026 line" and
    pushes a hitters-only board over a complete one. Same shape as the 2026-06-04 ROS
    incident, and guarded the same way: refuse before touching the KV.
    """
    module = _script()
    with pytest.raises(module.EmptyPoolError) as exc:
        module._require_scored_pool("pitcher", [], 2026, "pitcher_pt_panel_2000_2025.csv")

    message = str(exc.value)
    assert "pitcher" in message and "2026" in message
    assert "pitcher_pt_panel_2000_2025.csv" in message, "name the panel that came up short"

    # A pool that scored is silent.
    module._require_scored_pool("hitter", [object()], 2026, "hitter_pt_panel_2000_2026.csv")


PANEL_DIR = PROJECT_ROOT / "data" / "trajectory"


@pytest.mark.skipif(
    not PANEL_DIR.exists() or not any(PANEL_DIR.glob("*_pt_panel_*.csv")),
    reason=(
        "drives the real build_payload, which loads data/trajectory/*_pt_panel_*.csv and "
        "data/cache/keeper_skills -- both gitignored, so this cannot run on a fresh clone"
    ),
)
def test_the_guard_checks_what_the_sweep_produced_not_what_it_was_given(monkeypatch) -> None:
    """Candidate rows are not scored rows, and the guard has to look at the latter.

    `sweep_pool` independently drops every player whose VAR path has no observable point.
    A panel whose COMPLETE seasons do not span horizon 1 -- a pitcher panel built only
    through the in-progress year, say -- yields hundreds of candidates, sails past a
    guard that inspects its INPUT, and then returns [] with no exception. The push
    proceeds and overwrites a complete board with a pool-less one that renders normally,
    which is verbatim what EmptyPoolError's docstring says it prevents.

    Drives `build_payload` rather than the guard directly: the defect was the guard's
    POSITION, so calling it with an empty list would pass either way.
    """
    module = _script()
    # build_payload imports sweep_pool inside the function, so patch it at the source.
    monkeypatch.setattr("fantasy_baseball.trajectory.sweep.sweep_pool", lambda *a, **k: [])

    with pytest.raises(module.EmptyPoolError) as exc:
        module.build_payload(max_horizon=1, panel_dir=None)

    assert "scored 0 players" in str(exc.value)


def test_local_stays_local_even_when_render_is_already_set(monkeypatch, tmp_path) -> None:
    """`--local` exists to keep an unverified board OFF Render.

    `get_kv()` resolves through `is_remote()`, which reads RENDER at call time. The
    script used to SET that variable on the prod path and never clear it, so running
    --local in a shell where RENDER was already "true" -- the state several repo scripts
    and the documented "read prod locally" workflow both create -- silently wrote the
    board to PRODUCTION while printing "wrote to local SQLite".
    """
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "local.db"))

    target = _script()._target_store(local=True)

    assert isinstance(target, SqliteKVStore), (
        "--local must pick the local store from the FLAG, never from the environment"
    )


def test_the_push_never_mutates_render(monkeypatch) -> None:
    """The prod path selects its client explicitly, so it has no reason to touch RENDER.

    Setting it was copied from refresh_remote.py, where it IS load-bearing because that
    script then runs code resolving through get_kv(). Here nothing after the flip reads
    get_kv(), and the flip inverted the provenance: with RENDER set, `_code_sha()` takes
    the is_remote() branch, skips the git fallback, finds no RENDER_GIT_COMMIT on a
    laptop and stamps `_sha: "unknown"` -- so the PROD blob recorded "unknown" while
    --local recorded the real commit.
    """
    monkeypatch.delenv("RENDER", raising=False)
    module = _script()

    with pytest.raises(Exception):  # noqa: B017 - refuses to build an Upstash client here
        module._target_store(local=False)

    assert "RENDER" not in __import__("os").environ, (
        "selecting the prod store must not leave RENDER set for whatever runs next"
    )


@pytest.mark.skipif(
    not PANEL_DIR.exists() or not any(PANEL_DIR.glob("*_pt_panel_*.csv")),
    reason=(
        "drives the real build_payload, which loads data/trajectory/*_pt_panel_*.csv and "
        "data/cache/keeper_skills -- both gitignored, so this cannot run on a fresh clone"
    ),
)
def test_the_payload_carries_career_history_and_comps(monkeypatch) -> None:
    """Both are computed where the panel lives and travel in the same blob.

    The dashboard has no panel and never will -- `data/trajectory/` is gitignored and
    absent on Render -- so anything the chart needs has to be baked here or it does not
    exist at request time.
    """
    module = _script()
    payload, scored = module.build_payload(
        max_horizon=5, panel_dir=PROJECT_ROOT / "data" / "trajectory"
    )

    assert scored > 0
    player = payload["players"][0]

    # NOT "every scored player has at least his current season" -- history excludes
    # the current season by construction (`complete = live[~live["partial_season"]]`
    # in build_payload), which is exactly what #324's F3 review caught: a debut rookie
    # with no prior complete season legitimately has `history == []`. This assertion
    # passes only because `players[0]` in the real panel happens to have prior
    # seasons; it says nothing about every player.
    assert player["history"], "players[0] has prior complete seasons in the real panel"
    ages = [row[0] for row in player["history"]]
    assert ages == sorted(ages), "history ascends by age so a line can be drawn from it"
    assert all(len(row) == 2 for row in player["history"])

    assert len(player["comps"]) <= module.MAX_COMPS
    if player["comps"]:
        first = player["comps"][0]
        assert set(first) == {"name", "season", "rmse", "path"}
        assert len(first["path"]) == 5
        rmses = [c["rmse"] for c in player["comps"]]
        assert rmses == sorted(rmses), "closest first"
