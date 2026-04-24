"""Pure-function state machine for the live draft dashboard.

The Flask writer endpoints in ``web/app.py`` call these functions and
write the returned state dict via ``draft.state.write_state``. No Flask
coupling here — tests drive the controller without starting a server.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

from fantasy_baseball.draft.state import Pick, StateKey, read_state

EMPTY_STATE: dict[str, Any] = {
    StateKey.VERSION: 0,
    StateKey.KEEPERS: [],
    StateKey.PICKS: [],
    StateKey.UNDO_STACK: [],
    StateKey.ON_THE_CLOCK: None,
    StateKey.PROJECTED_STANDINGS_CACHE: {},
}


def resume_or_init(state_path: Path) -> dict[str, Any]:
    """Load an existing draft-state file or return a blank state."""
    if not state_path.exists():
        return copy.deepcopy(EMPTY_STATE)
    return read_state(state_path)


class UnresolvedKeeperError(Exception):
    """Raised when a keeper in league.yaml cannot be matched to a projection row."""


class KeeperNotFound(Exception):
    """Raised by a resolver callable when a keeper cannot be found in the board.

    `start_new_draft` catches both this and built-in `KeyError` and rolls
    them into an `UnresolvedKeeperError` listing every missing keeper. Use
    `KeeperNotFound` in resolvers backed by non-dict sources (e.g. a board
    search) so the intent is clearer than reusing `KeyError`.
    """


def start_new_draft(
    league_yaml: dict[str, Any],
    *,
    resolve_keeper,
) -> dict[str, Any]:
    """Seed keepers from league.yaml and return the initial state dict.

    ``resolve_keeper`` is injected (not imported) so tests can stub it.
    Production wiring in web/app.py will pass a function backed by
    draft.search.find_player_by_name.
    """
    teams_by_position = league_yaml["draft"]["teams"]
    first_picker = (
        teams_by_position[1] if isinstance(teams_by_position, dict) else teams_by_position[0]
    )

    keepers: list[dict[str, Any]] = []
    unresolved: list[str] = []
    now = time.time()
    for entry in league_yaml.get("keepers", []):
        name, team = entry["name"], entry["team"]
        try:
            pid, pname, pos = resolve_keeper(name, team)
        except (KeyError, KeeperNotFound):
            unresolved.append(f"{name} ({team})")
            continue
        keepers.append(
            Pick(
                pick_number=None,
                round=0,
                team=team,
                player_id=pid,
                player_name=pname,
                position=pos,
                timestamp=now,
            ).to_dict()
        )
    if unresolved:
        raise UnresolvedKeeperError("Could not resolve keepers: " + "; ".join(unresolved))

    return {
        StateKey.VERSION: 0,
        StateKey.KEEPERS: keepers,
        StateKey.PICKS: [],
        StateKey.UNDO_STACK: [],
        StateKey.ON_THE_CLOCK: first_picker,
        StateKey.PROJECTED_STANDINGS_CACHE: {},
    }


class WrongTeamError(Exception):
    """Raised when apply_pick is called with a team that isn't on the clock."""


class AlreadyDraftedError(Exception):
    """Raised when a player_id is already in keepers or picks."""


def _snake_order(teams_by_position: dict[int, str], num_rounds: int) -> list[str]:
    """Return the full pick order as a flat list of team names."""
    positions = [teams_by_position[i] for i in sorted(teams_by_position)]
    order: list[str] = []
    for r in range(num_rounds):
        order.extend(positions if r % 2 == 0 else list(reversed(positions)))
    return order


def _compute_on_the_clock(
    teams_by_position: dict[int, str],
    picks_so_far: int,
) -> str | None:
    """Return the team name for the next live pick, or None if the draft is done."""
    order = _snake_order(teams_by_position, num_rounds=30)
    if picks_so_far >= len(order):
        return None
    return order[picks_so_far]


def apply_pick(
    state: dict[str, Any],
    *,
    player_id: str,
    player_name: str,
    position: str,
    team: str,
    teams_by_position: dict[int, str],
) -> dict[str, Any]:
    """Record a live pick, advance the snake order, return the new state."""
    if state[StateKey.ON_THE_CLOCK] != team:
        raise WrongTeamError(f"{team} is not on the clock — {state[StateKey.ON_THE_CLOCK]} is.")
    all_ids = {p["player_id"] for p in state[StateKey.KEEPERS]} | {
        p["player_id"] for p in state[StateKey.PICKS]
    }
    if player_id in all_ids:
        raise AlreadyDraftedError(f"{player_id} already drafted")

    pick_number = len(state[StateKey.PICKS]) + 1
    num_teams = len(teams_by_position)
    round_number = (pick_number - 1) // num_teams + 1

    new_pick = Pick(
        pick_number=pick_number,
        round=round_number,
        team=team,
        player_id=player_id,
        player_name=player_name,
        position=position,
        timestamp=time.time(),
    ).to_dict()

    new_state = {**state}
    new_state[StateKey.PICKS] = state[StateKey.PICKS] + [new_pick]
    new_state[StateKey.ON_THE_CLOCK] = _compute_on_the_clock(
        teams_by_position, len(new_state[StateKey.PICKS])
    )
    new_state[StateKey.UNDO_STACK] = []
    return new_state


UNDO_CAP = 20


def undo_pick(
    state: dict[str, Any],
    *,
    teams_by_position: dict[int, str],
) -> dict[str, Any]:
    """Pop the most recent live pick, advance the undo stack, roll on_the_clock back."""
    if not state[StateKey.PICKS]:
        return state
    new_state = {**state}
    popped = new_state[StateKey.PICKS][-1]
    new_state[StateKey.PICKS] = new_state[StateKey.PICKS][:-1]
    undo_stack = new_state.get(StateKey.UNDO_STACK, [])[:]
    undo_stack.append(popped)
    if len(undo_stack) > UNDO_CAP:
        undo_stack = undo_stack[-UNDO_CAP:]
    new_state[StateKey.UNDO_STACK] = undo_stack
    new_state[StateKey.ON_THE_CLOCK] = _compute_on_the_clock(
        teams_by_position, len(new_state[StateKey.PICKS])
    )
    return new_state
