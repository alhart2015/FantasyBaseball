"""Load the league's real draft pick order (with traded picks) for the live draft.

``config/draft_order.json`` carries the resolved per-round team order -- trades are
already baked into the ``rounds`` arrays; the ``trades`` list is annotation only.
The live draft begins after the keeper rounds, so the dashboard skips the first
``keeper_rounds`` rounds (it seeds keepers separately) and works through the rest.

This returns the flat, name-keyed shape the web controller needs. The CLI
(``scripts/run_draft._load_draft_order``) and simulator
(``scripts/simulate_draft._load_pick_order``) have their own historical loaders
returning richer / team-number shapes; they are not reused here to avoid a
scripts -> package import dependency.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_post_keeper_pick_order(path: Path, *, keeper_rounds: int) -> list[str] | None:
    """Flat post-keeper pick order as team names, or ``None`` if the file is absent.

    ``keeper_rounds`` leading rounds are dropped. Each remaining round contributes
    its team names in slot order (trades already resolved in ``rounds``). Returning
    ``None`` lets callers fall back to a pure snake order when no custom file
    exists (mock drafts, tests).
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    rounds = data["rounds"]
    order = [team for round_teams in rounds[keeper_rounds:] for team in round_teams]
    # Return None (not []) for an empty result so both consumers agree to snake.
    # An empty list otherwise splits them: start_new_draft (`if pick_order:`)
    # snakes, but _compute_on_the_clock (`is not None`) treats [] as an exhausted
    # order and seats no one -> on_the_clock None on pick 1.
    return order or None
