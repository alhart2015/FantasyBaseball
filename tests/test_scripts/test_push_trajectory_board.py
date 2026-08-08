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


def _swept(n_points: int):
    """One `SweptPlayer` whose observable path carries `n_points` horizons."""
    from fantasy_baseball.trajectory.sweep import SweptPlayer, YearPoint

    return SweptPlayer(
        mlbam_id=1,
        name="Short Path",
        pool="hitter",
        age=27,
        slot="OF",
        floor=4.0,
        now=12.0,
        prior=11.0,
        support=0.5,
        extrapolated=False,
        sgp=tuple(
            YearPoint(
                horizon=h,
                age=27 + h,
                mean=12.0,
                p10=9.0,
                p90=15.0,
                n_effective=50.0,
                band_fell_back=False,
            )
            for h in range(1, n_points + 1)
        ),
    )


def _prepared(horizons):
    from fantasy_baseball.trajectory.shape import prepare
    from tests._trajectory_panel import synthetic_panel

    return prepare(synthetic_panel(), kind="hitter", horizons=horizons)


def test_a_player_observable_at_fewer_horizons_than_the_sweep_loses_only_his_comps() -> None:
    """One short path must not discard the whole ~52s sweep.

    `player.sgp` is `traj.observable` -- points with `n > 0` only -- and the candidate
    mask `seasons + h <= last` shrinks as h grows, so a player can be observable at
    h=1..3 and not at h=4..5. `sweep_pool` keeps him (it drops only an ENTIRELY empty
    path), `closest_paths` raises when `len(predicted) != len(prepared.horizons)`, and
    nothing catches it, so the push writes nothing at all. Directly reachable via the
    documented `--max-horizon` flag.

    HIS COMPS ARE SKIPPED, not padded. `forward` stores a real 0.0 for "out of the
    league", so padding the path with zeros would match him against a cohort that
    stopped playing -- and the page already renders an empty comps list with an
    explanation of why.
    """
    module = _script()
    horizons = (1, 2, 3)
    prepared = _prepared(horizons)

    assert module.player_comps(prepared, _swept(2), horizons, {}) is None, "skipped"

    full = module.player_comps(prepared, _swept(3), horizons, {})
    assert full is not None and full, "a full-length path still gets its comps"
    assert set(full[0]) == {"name", "season", "rmse", "path"}


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


class _RecordingStore:
    """A KV that remembers the ORDER it was written in. Enough of the interface for
    `write_cache_to` (which only calls `set`) and the script's read-back (`get`)."""

    def __init__(self) -> None:
        self.writes: list[str] = []
        self.values: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self.writes.append(key)
        self.values[key] = value

    def get(self, key: str) -> str | None:
        return self.values.get(key)


def _fake_payloads(module, monkeypatch):
    """Patch `build_payload` to a two-player board and its paired chart data.

    Lets `main()` be driven end to end without the gitignored panel -- the ordering and
    the dry-run behaviour are properties of `main`, not of the sweep.
    """
    from fantasy_baseball.trajectory.sweep import to_chart_payload

    stamp = "2026-08-07T09:00:00-04:00"
    board = {
        "generated_at": stamp,
        "base_season": 2026,
        "panel_vintage": {"hitter": "h.csv", "pitcher": "p.csv"},
        "players": [{"id": 1, "pool": "hitter"}, {"id": 1, "pool": "pitcher"}],
    }
    chart = to_chart_payload(
        {
            (1, "hitter"): {"history": [[26, 14.0]], "comps": []},
            (1, "pitcher"): {"history": [[26, 9.0]], "comps": []},
        },
        generated_at=stamp,
    )
    monkeypatch.setattr(module, "build_payload", lambda *a, **k: (board, chart, 2))
    return board, chart


def _stub_the_sweep(monkeypatch):
    """Make `build_payload` runnable with no panel on disk, so what it STAMPS can be
    pinned on a fresh clone.

    Everything `build_payload` imports it imports inside the function, so each stub goes
    on the source module the way the empty-pool test already patches `sweep_pool`. The
    fit itself is left real -- it runs against the shared synthetic panel -- because the
    property under test is about the timestamp, and a stub that also faked the sweep
    could not tell a payload from a chart.
    """
    import pathlib
    import types

    import pandas as pd

    from fantasy_baseball.trajectory.board import BoardRow
    from tests._trajectory_panel import synthetic_panel

    panel = synthetic_panel()
    # The production panel carries this; the shared fixture does not, and `build_payload`
    # splits on it to keep the in-progress season out of the comp pool.
    panel["partial_season"] = False

    # One row per pool, keyed by kind: `build_payload` sweeps both, and a stub that
    # returned the same row twice would let a hitter into the pitcher pool.
    rows = {
        "hitter": [BoardRow(1, "Big Bat", "hitter", 27, 20.0, 19.0, "OF", 4.0)],
        "pitcher": [BoardRow(2, "Big Arm", "pitcher", 27, 18.0, 17.0, "SP", 3.0)],
    }
    stubs = {
        "fantasy_baseball.config.load_config": lambda _p: types.SimpleNamespace(sgp_overrides=None),
        "fantasy_baseball.sgp.denominators.get_sgp_denominators": lambda _o: {},
        "fantasy_baseball.sgp.replacement.position_aware_replacement_levels": lambda _d: {
            "OF": 4.0
        },
        "fantasy_baseball.trajectory.panel.load_scored_panel": lambda _k, **_kw: panel.copy(),
        "fantasy_baseball.trajectory.panel.panel_path": lambda k, _d: pathlib.Path(f"{k}.csv"),
        "fantasy_baseball.trajectory.panel.season_elapsed_fraction": lambda _df, _s: 0.7,
        "fantasy_baseball.trajectory.era.era_normalize": lambda df, _k, **_kw: df,
        "fantasy_baseball.trajectory.board.player_names": lambda _c: pd.Series(dtype=object),
        "fantasy_baseball.trajectory.board.season_slots": lambda _c, _s: {},
        "fantasy_baseball.trajectory.board.board_inputs": lambda *_a, **kw: rows[kw["kind"]],
    }
    for target, stub in stubs.items():
        monkeypatch.setattr(target, stub)


def test_both_payloads_carry_ONE_stamp_taken_ONCE(monkeypatch) -> None:
    """The pairing the player view checks is an equality test on `generated_at`.

    A second `local_now()` call is nearly invisible: the stamp is truncated to seconds,
    so two adjacent calls almost always produce the same string and every other test
    stays green. It would fail only when the two calls straddle a second boundary --
    an intermittent "every career line vanished on this push", which is far harder to
    diagnose than a permanent break. So the clock is made to return a DIFFERENT value on
    every call, and the two payloads still have to agree.
    """
    module = _script()
    _stub_the_sweep(monkeypatch)

    stamps = iter(["2026-08-07T09:00:00", "2026-08-07T09:00:01", "2026-08-07T09:00:02"])

    class _Clock:
        def isoformat(self, timespec: str = "seconds") -> str:
            return next(stamps)

    monkeypatch.setattr("fantasy_baseball.utils.time_utils.local_now", _Clock)

    payload, chart, scored = module.build_payload(max_horizon=1, panel_dir=None)

    assert scored == 2, "the stub swept one player per pool"
    assert payload["generated_at"] == chart["generated_at"]
    assert payload["generated_at"] == "2026-08-07T09:00:00", "the FIRST reading, taken once"


def test_the_chart_data_is_written_before_the_board(monkeypatch, capsys) -> None:
    """A successful board write must imply its extras are already stored.

    The two blobs are paired by `generated_at` and the player view refuses a pair that
    disagrees, so the write ORDER decides how a half-finished push degrades. Extras
    first: a crash between the two leaves the OLD board beside new extras, which the
    reader catches. Board first would publish a fresh board beside stale extras and
    silently drop every career line until the next run.
    """
    module = _script()
    _fake_payloads(module, monkeypatch)
    store = _RecordingStore()
    monkeypatch.setattr(module, "_target_store", lambda *, local: store)
    monkeypatch.setattr("sys.argv", ["push_trajectory_board.py", "--local"])

    assert module.main() == 0
    assert store.writes == ["cache:trajectory_chart_data", "cache:trajectory_board"]
    assert "chart data: 2 players" in capsys.readouterr().out, "the read-back names both"


def test_dry_run_reports_both_sizes_and_writes_nothing(monkeypatch, capsys) -> None:
    """The two keys have very different read profiles -- every view pays the board, only
    the player chart pays the extras -- so one combined figure hides the number #344 was
    opened about. And a --dry-run that touches the KV is the whole reason the flag
    exists.
    """
    module = _script()
    _fake_payloads(module, monkeypatch)
    store = _RecordingStore()
    monkeypatch.setattr(module, "_target_store", lambda *, local: store)
    monkeypatch.setattr("sys.argv", ["push_trajectory_board.py", "--dry-run"])

    assert module.main() == 0
    assert store.writes == [], "--dry-run writes nothing"

    out = capsys.readouterr().out
    assert "board" in out and "chart data" in out, "both keys are sized"
    assert len([line for line in out.splitlines() if "KB" in line]) == 2, "two sizes, not one"


@pytest.mark.skipif(
    not PANEL_DIR.exists() or not any(PANEL_DIR.glob("*_pt_panel_*.csv")),
    reason=(
        "drives the real build_payload, which loads data/trajectory/*_pt_panel_*.csv and "
        "data/cache/keeper_skills -- both gitignored, so this cannot run on a fresh clone"
    ),
)
def test_the_chart_data_carries_career_history_and_comps(monkeypatch) -> None:
    """Both are computed where the panel lives and travel in their OWN blob.

    The dashboard has no panel and never will -- `data/trajectory/` is gitignored and
    absent on Render -- so anything the chart needs has to be baked here or it does not
    exist at request time. It rides in `cache:trajectory_chart_data` rather than in the
    board (#344): only the player view reads it, and inline it more than doubled what
    the two default views had to fetch.
    """
    from fantasy_baseball.trajectory.sweep import chart_key

    module = _script()
    payload, chart, scored = module.build_payload(
        max_horizon=5, panel_dir=PROJECT_ROOT / "data" / "trajectory"
    )

    assert scored > 0
    # PAIRED: the player view compares these two stamps and refuses to draw a chart
    # whose extras do not match the board it is rendering.
    assert chart["generated_at"] == payload["generated_at"]

    row = payload["players"][0]
    assert "history" not in row and "comps" not in row, "the board carries neither"
    player = chart["players"][chart_key(row["id"], row["pool"])]

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


@pytest.mark.skipif(
    not PANEL_DIR.exists() or not any(PANEL_DIR.glob("*_pt_panel_*.csv")),
    reason=(
        "drives the real build_payload, which loads data/trajectory/*_pt_panel_*.csv and "
        "data/cache/keeper_skills -- both gitignored, so this cannot run on a fresh clone"
    ),
)
def test_the_push_stamps_whether_the_base_season_is_still_running() -> None:
    """Read off the PANEL, never off today's date. `_live_seasons` in build_pt_panel.py
    flags a season partial iff `year >= today.year`, so the reader has to follow the
    panel the board was actually built from."""
    module = _script()
    payload, _, _ = module.build_payload(max_horizon=3, panel_dir=PANEL_DIR)

    assert payload["base_season_partial"] is True, (
        "the shipped 2000-2026 panels were built during the 2026 season"
    )
