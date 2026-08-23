"""A transcribed roster player who fails to hydrate must stop a manual run.

`match_roster_to_projections` documents that unmatched entries are OMITTED. On
the Yahoo path that costs one player's projections and nothing more. On the
manual path the hydrated rosters are the ONLY thing subtracted from the
synthesized free-agent pool, so an omitted player leaves his own projection row
addable -- and `audit_roster`, which takes the maximum-DeltaRoto free agent per
slot, headlines "drop X, add Y" for a Y another manager already owns. Nothing
raises and the report reads normally.
"""

from __future__ import annotations

import pytest

from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.roster import Roster, RosterEntry
from fantasy_baseball.web.refresh_pipeline import ManualRosterUnmatched, RefreshRun


def _roster(*names: str) -> Roster:
    from datetime import date

    return Roster(
        effective_date=date(2026, 8, 22),
        entries=[
            RosterEntry(name=n, positions=[Position.OF], selected_position=Position.OF)
            for n in names
        ],
    )


class _Team:
    def __init__(self, name, roster):
        self.name = name
        self.rosters = [roster]

    def latest_roster(self):
        return self.rosters[0]


class _League:
    def __init__(self, teams):
        self.teams = teams

    def team_by_name(self, name):
        return next(t for t in self.teams if t.name == name)


def _run(monkeypatch, *, manual: bool, hydrate_returns):
    """A RefreshRun with hydration stubbed, so only the loss check is exercised."""
    from fantasy_baseball.data import projections

    run = RefreshRun(skip_yahoo=True, free_agent_source=(object() if manual else None))
    assert run.manual_mode is manual

    class _Cfg:
        team_name = "Mine"

    run.config = _Cfg()
    run.hitters_proj = object()
    run.pitchers_proj = object()
    run.full_hitters_proj = None
    run.full_pitchers_proj = None
    run.preseason_hitters = None
    run.preseason_pitchers = None
    run.league_model = _League([_Team("Mine", _roster("Kept Bat", "Lost Bat"))])
    monkeypatch.setattr(projections, "hydrate_roster_entries", hydrate_returns)
    monkeypatch.setattr(run, "_progress", lambda *a, **k: None)
    return run


class _P:
    def __init__(self, name):
        self.name = name


def test_manual_mode_refuses_when_a_transcribed_player_did_not_hydrate(monkeypatch):
    run = _run(monkeypatch, manual=True, hydrate_returns=lambda *a, **k: [_P("Kept Bat")])

    with pytest.raises(ManualRosterUnmatched) as exc:
        run._hydrate_rosters()

    msg = str(exc.value)
    assert "Lost Bat" in msg, "the operator can only act if the player is named"
    assert "Mine" in msg, "and only if the team is named"
    assert "rosters.yaml" in msg, "and only if told where to fix it"


def test_manual_mode_proceeds_when_every_player_hydrated(monkeypatch):
    run = _run(
        monkeypatch,
        manual=True,
        hydrate_returns=lambda *a, **k: [_P("Kept Bat"), _P("Lost Bat")],
    )
    run._hydrate_rosters()
    assert {p.name for p in run.matched} == {"Kept Bat", "Lost Bat"}


def test_the_yahoo_path_still_tolerates_an_unmatched_player(monkeypatch):
    """Not manual mode: the roster and the pool both come from Yahoo, so a drop
    cannot leak the player into the free-agent pool. Refusing here would turn a
    survivable Yahoo hiccup into a failed refresh."""
    run = _run(monkeypatch, manual=False, hydrate_returns=lambda *a, **k: [_P("Kept Bat")])
    run._hydrate_rosters()
    assert [p.name for p in run.matched] == ["Kept Bat"]
