"""A player the anchor dropped for want of a rest-of-season row, on the single-player CLI.

THE CASE LIST IS THE POINT. Two earlier attempts at this rule each fixed one reading and
broke another: keying the exclusion on every id matching the name suppressed a player who
was perfectly scorable, and keying it on the resolved id printed the WRONG player's
trajectory under the typed name. Both rendered normally and both passed the tests written
alongside them, because each test pinned only the reading its author had in mind.

Every case below is one a review pass actually caught. They are kept together, in one
file, so the next person to touch this rule sees the whole state space rather than the
facet in front of them.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location(
        "player_trajectory", PROJECT_ROOT / "scripts" / "player_trajectory.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _people(monkeypatch, module, ids=(1,), name="Solo Player") -> None:
    """The name -> id cache `_resolve_player` resolves through, without the real files."""
    frame = pd.DataFrame(
        {"id": list(ids), "fullName": [name] * len(ids), "norm": [name.lower()] * len(ids)}
    )
    monkeypatch.setattr(module, "board_people", lambda _cache: frame)


def _rows(mlbam_id: int) -> pd.DataFrame:
    """Two settled seasons. What the panel holds for a dropped player AFTER the anchor
    has removed his in-progress row -- which is why the newest one looks ordinary."""
    return pd.DataFrame(
        [
            {"mlbam_id": mlbam_id, "season": 2024, "age": 25, "sgp": 9.0, "partial_season": False},
            {"mlbam_id": mlbam_id, "season": 2025, "age": 26, "sgp": 11.0, "partial_season": False},
        ]
    )


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["mlbam_id", "season", "age", "sgp", "partial_season"])


def test_a_dropped_player_is_not_backdated_to_his_prior_season(monkeypatch) -> None:
    """The anchor deletes his current-season row, so `idxmax` lands on a SETTLED prior
    year -- the in-progress notice never prints and the CLI renders a full trajectory
    headed with the wrong age and the wrong anchor, indistinguishable from a right one."""
    module = _script()
    _people(monkeypatch, module)
    with pytest.raises(SystemExit) as exc:
        module._resolve_player("Solo Player", {"hitter": _rows(1)}, no_ros={"hitter": [1]})
    assert "rest-of-season" in str(exc.value), "say WHY he is not priced"
    assert "2025" not in str(exc.value), "and do not offer the prior season as a substitute"


def test_a_dropped_player_with_no_earlier_season_still_gets_the_real_reason(monkeypatch) -> None:
    """A call-up whose ONLY panel row was the anchored one. It is deleted, so he has no
    rows anywhere -- and "no observed season in the panel" is FALSE of him. He has one;
    it cannot be anchored, which is a different and nameable thing."""
    module = _script()
    _people(monkeypatch, module)
    with pytest.raises(SystemExit) as exc:
        module._resolve_player("Solo Player", {"hitter": _empty()}, no_ros={"hitter": [1]})
    assert "rest-of-season" in str(exc.value)
    assert "no observed season" not in str(exc.value)


def test_a_player_who_really_is_absent_still_says_so(monkeypatch) -> None:
    """The generic message has to survive for the case it is actually true of."""
    module = _script()
    _people(monkeypatch, module)
    with pytest.raises(SystemExit, match="no observed season"):
        module._resolve_player("Solo Player", {"hitter": _empty()}, no_ros={"hitter": []})


def test_a_dropped_namesake_does_not_suppress_the_other(monkeypatch) -> None:
    """Keying the exclusion on every id matching the name skipped the pool for BOTH, so
    `found` came out empty and the disambiguation branch never ran -- pointing the user
    at a data problem instead of at `--mlbam-id`."""
    module = _script()
    _people(monkeypatch, module, ids=(1, 2), name="Angel Sanchez")
    panel = pd.concat([_rows(1), _rows(2)], ignore_index=True)
    with pytest.raises(SystemExit) as exc:
        module._resolve_player("Angel Sanchez", {"hitter": panel}, no_ros={"hitter": [2]})
    assert "pick one" in str(exc.value)


def test_a_dropped_namesake_with_no_rows_is_still_a_candidate(monkeypatch) -> None:
    """The reverse error, and the worse one. His only row was the anchored season, so he
    vanishes from the panel entirely -- and keying disambiguation on surviving rows alone
    resolved silently to the OTHER player and printed that man's trajectory under the
    typed name. A dropped player is still a real player matching the name."""
    module = _script()
    _people(monkeypatch, module, ids=(1, 2), name="Angel Sanchez")
    with pytest.raises(SystemExit) as exc:
        module._resolve_player("Angel Sanchez", {"hitter": _rows(2)}, no_ros={"hitter": [1]})
    assert "pick one" in str(exc.value), "he is ambiguous, not absent"


def test_the_menu_does_not_backdate_a_dropped_namesake(monkeypatch) -> None:
    """`through YEAR` is read off the panel, from which the anchor has already removed
    his current season -- so the menu advertised a player as finished a year before he
    actually played, and picking him then died with the exclusion anyway."""
    module = _script()
    _people(monkeypatch, module, ids=(1, 2), name="Angel Sanchez")
    panel = pd.concat([_rows(1), _rows(2)], ignore_index=True)
    with pytest.raises(SystemExit) as exc:
        module._resolve_player("Angel Sanchez", {"hitter": panel}, no_ros={"hitter": [2]})
    line = next(ln for ln in str(exc.value).splitlines() if "--mlbam-id 2" in ln)
    assert "through" not in line, "no stale year for a player the anchor dropped"
    assert "rest-of-season" in line, "say what happened to him instead"


def test_only_the_pools_actually_queried_are_named(monkeypatch) -> None:
    """`--pool hitter` restricts `panels`. Reading the dropped set off the whole `no_ros`
    dict made a pure pitcher fail a HITTER query with "no row in the rest-of-season
    snapshot (pitcher)", naming a pool nobody asked about."""
    module = _script()
    _people(monkeypatch, module)
    with pytest.raises(SystemExit) as exc:
        module._resolve_player(
            "Solo Player", {"hitter": _empty()}, no_ros={"hitter": [], "pitcher": [1]}
        )
    assert "pitcher" not in str(exc.value)


def test_a_two_way_player_keeps_the_half_that_has_a_projection(monkeypatch, capsys) -> None:
    """He is two assets in this league; a missing bat projection says nothing about his
    arm. And the dropped half must be ANNOUNCED -- a pitcher-only table for a two-way
    player looks exactly like a player who never hit."""
    module = _script()
    _people(monkeypatch, module)
    panels = {"hitter": _rows(1), "pitcher": _rows(1)}
    resolved = module._resolve_player(
        "Solo Player", panels, no_ros={"hitter": [1], "pitcher": []}, snapshot_date="2026-07-21"
    )
    assert [pool for pool, *_ in resolved] == ["pitcher"]
    said = capsys.readouterr().out
    assert "hitter" in said and "2026-07-21" in said, "name the pool and the vintage"


def test_the_refusal_names_an_escape_that_actually_works(monkeypatch) -> None:
    """The manual path is the `--player`-less branch, so an instruction to "supply the
    line by hand" that keeps `--player` re-hits this same refusal."""
    module = _script()
    _people(monkeypatch, module)
    with pytest.raises(SystemExit) as exc:
        module._resolve_player("Solo Player", {"hitter": _rows(1)}, no_ros={"hitter": [1]})
    assert "without --player" in str(exc.value)


def test_a_player_with_a_projection_is_untouched(monkeypatch) -> None:
    """The rule must key on the dropped set, not on "his newest season is settled" --
    which is every player's normal state in the offseason."""
    module = _script()
    _people(monkeypatch, module)
    resolved = module._resolve_player("Solo Player", {"hitter": _rows(1)}, no_ros={"hitter": []})
    assert resolved == [("hitter", 1, 26, 11.0)]


def test_no_ros_absent_entirely_changes_nothing(monkeypatch) -> None:
    """`no_ros=None` is the offseason path, where nothing was anchored and nothing
    dropped."""
    module = _script()
    _people(monkeypatch, module)
    assert module._resolve_player("Solo Player", {"hitter": _rows(1)}) == [("hitter", 1, 26, 11.0)]
