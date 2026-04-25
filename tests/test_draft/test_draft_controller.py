from pathlib import Path

import pytest


def test_resume_or_init_returns_empty_when_file_missing(tmp_path: Path):
    from fantasy_baseball.draft.draft_controller import resume_or_init

    state_path = tmp_path / "draft_state.json"
    state = resume_or_init(state_path)
    assert state == {
        "version": 0,
        "keepers": [],
        "picks": [],
        "undo_stack": [],
        "on_the_clock": None,
        "projected_standings_cache": {},
    }


def test_resume_or_init_loads_existing_state(tmp_path: Path):
    import json

    from fantasy_baseball.draft.draft_controller import resume_or_init

    state_path = tmp_path / "draft_state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 5,
                "on_the_clock": "TeamA",
                "picks": [],
                "keepers": [],
                "undo_stack": [],
                "projected_standings_cache": {},
            }
        )
    )
    state = resume_or_init(state_path)
    assert state["version"] == 5
    assert state["on_the_clock"] == "TeamA"


def test_start_new_draft_seeds_keepers_and_sets_on_the_clock():
    from fantasy_baseball.draft.draft_controller import start_new_draft

    league_yaml = {
        "league": {"team_name": "Hart of the Order"},
        "draft": {
            "position": 8,
            "teams": {
                1: "Send in the Cavalli",
                2: "SkeleThor",
                3: "Work in Progress",
                4: "Jon's Underdogs",
                5: "Boston Estrellas",
                6: "Spacemen",
                7: "Springfield Isotopes",
                8: "Hart of the Order",
                9: "Tortured Baseball Department",
                10: "Hello Peanuts!",
            },
        },
        "keepers": [
            {"name": "Juan Soto", "team": "Hart of the Order"},
            {"name": "Elly De La Cruz", "team": "Hello Peanuts!"},
        ],
    }
    resolver = {
        ("Juan Soto", "Hart of the Order"): ("Juan Soto::hitter", "Juan Soto", "OF"),
        ("Elly De La Cruz", "Hello Peanuts!"): (
            "Elly De La Cruz::hitter",
            "Elly De La Cruz",
            "SS",
        ),
    }

    def resolve_keeper(name: str, team: str):
        return resolver[(name, team)]

    state = start_new_draft(league_yaml, resolve_keeper=resolve_keeper)

    assert len(state["keepers"]) == 2
    assert state["keepers"][0]["team"] == "Hart of the Order"
    assert state["keepers"][0]["pick_number"] is None
    assert state["picks"] == []
    assert state["on_the_clock"] == "Send in the Cavalli"


def test_start_new_draft_raises_on_unresolvable_keeper():
    from fantasy_baseball.draft.draft_controller import (
        UnresolvedKeeperError,
        start_new_draft,
    )

    league_yaml = {
        "league": {"team_name": "Hart of the Order"},
        "draft": {"position": 8, "teams": {1: "Send in the Cavalli"}},
        "keepers": [{"name": "Not A Real Player", "team": "Send in the Cavalli"}],
    }

    def resolver(name, team):
        raise KeyError(f"no match for {name}")

    with pytest.raises(UnresolvedKeeperError) as exc:
        start_new_draft(league_yaml, resolve_keeper=resolver)
    assert "Not A Real Player" in str(exc.value)


@pytest.fixture
def teams_by_position():
    return {i: f"Team{i}" for i in range(1, 11)}


@pytest.fixture
def starter_state(teams_by_position):
    return {
        "version": 0,
        "keepers": [],
        "picks": [],
        "undo_stack": [],
        "on_the_clock": teams_by_position[1],
        "projected_standings_cache": {},
    }


def test_apply_pick_advances_snake_order_round1(starter_state, teams_by_position):
    from fantasy_baseball.draft.draft_controller import apply_pick

    state = apply_pick(
        starter_state,
        player_id="P1::hitter",
        player_name="Player One",
        position="OF",
        team="Team1",
        teams_by_position=teams_by_position,
    )
    assert state["on_the_clock"] == "Team2"
    assert len(state["picks"]) == 1
    assert state["picks"][0]["pick_number"] == 1
    assert state["picks"][0]["round"] == 1


def test_apply_pick_reverses_at_round_boundary(starter_state, teams_by_position):
    from fantasy_baseball.draft.draft_controller import apply_pick

    state = starter_state
    for i in range(1, 11):
        state = apply_pick(
            state,
            player_id=f"P{i}::hitter",
            player_name=f"Player {i}",
            position="OF",
            team=f"Team{i}",
            teams_by_position=teams_by_position,
        )
    assert state["on_the_clock"] == "Team10"
    assert len(state["picks"]) == 10


def test_apply_pick_rejects_wrong_team(starter_state, teams_by_position):
    from fantasy_baseball.draft.draft_controller import WrongTeamError, apply_pick

    with pytest.raises(WrongTeamError):
        apply_pick(
            starter_state,
            player_id="P1::hitter",
            player_name="Player One",
            position="OF",
            team="Team5",
            teams_by_position=teams_by_position,
        )


def test_undo_pick_round_trips(starter_state, teams_by_position):
    from fantasy_baseball.draft.draft_controller import apply_pick, undo_pick

    after = apply_pick(
        starter_state,
        player_id="P1::hitter",
        player_name="Player One",
        position="OF",
        team="Team1",
        teams_by_position=teams_by_position,
    )
    undone = undo_pick(after, teams_by_position=teams_by_position)
    assert undone["picks"] == []
    assert undone["on_the_clock"] == "Team1"
    assert len(undone["undo_stack"]) == 1
    assert undone["undo_stack"][0]["player_id"] == "P1::hitter"


def test_undo_pick_chains(starter_state, teams_by_position):
    from fantasy_baseball.draft.draft_controller import apply_pick, undo_pick

    state = starter_state
    for i in range(1, 4):
        state = apply_pick(
            state,
            player_id=f"P{i}::hitter",
            player_name=f"Player {i}",
            position="OF",
            team=f"Team{i}",
            teams_by_position=teams_by_position,
        )
    state = undo_pick(state, teams_by_position=teams_by_position)
    state = undo_pick(state, teams_by_position=teams_by_position)
    state = undo_pick(state, teams_by_position=teams_by_position)
    assert state["picks"] == []
    assert state["on_the_clock"] == "Team1"
    assert len(state["undo_stack"]) == 3


def test_undo_with_empty_picks_is_noop(starter_state, teams_by_position):
    from fantasy_baseball.draft.draft_controller import undo_pick

    result = undo_pick(starter_state, teams_by_position=teams_by_position)
    assert result == starter_state


def test_start_new_draft_catches_keeper_not_found():
    """Resolvers may raise KeeperNotFound (the production-intended sentinel)
    instead of KeyError; both paths must aggregate into UnresolvedKeeperError."""
    from fantasy_baseball.draft.draft_controller import (
        KeeperNotFound,
        UnresolvedKeeperError,
        start_new_draft,
    )

    league_yaml = {
        "league": {"team_name": "Hart of the Order"},
        "draft": {"position": 8, "teams": {1: "Send in the Cavalli"}},
        "keepers": [{"name": "Nobody", "team": "Send in the Cavalli"}],
    }

    def resolver(name, team):
        raise KeeperNotFound(f"no match for {name}")

    with pytest.raises(UnresolvedKeeperError) as exc:
        start_new_draft(league_yaml, resolve_keeper=resolver)
    assert "Nobody" in str(exc.value)


def test_resume_or_init_returns_isolated_copies():
    """Mutating the returned state must not affect a second call to resume_or_init."""
    from fantasy_baseball.draft.draft_controller import resume_or_init

    state_a = resume_or_init(Path("does/not/exist/state.json"))
    state_a["picks"].append({"player_id": "X::hitter"})
    state_a["projected_standings_cache"]["Team1"] = {"HR": 99.0}

    state_b = resume_or_init(Path("also/does/not/exist.json"))
    assert state_b["picks"] == []
    assert state_b["projected_standings_cache"] == {}
