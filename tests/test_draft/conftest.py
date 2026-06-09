"""Shared fixtures for test_draft.

Provides the ``deltaroto_ctx`` fixture factory used by test_recommend.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from fantasy_baseball.draft import recs_integration
from fantasy_baseball.draft.recommend import RecommendContext
from fantasy_baseball.draft.recs_integration import RecInputs

_FIXTURES = Path(__file__).parent / "fixtures"
_STATE_PATH = _FIXTURES / "recs_golden_state.json"
_BOARD_PATH = _FIXTURES / "recs_golden_state_board.json"
_LEAGUE_YAML_PATH = Path(__file__).parents[2] / "config" / "league.yaml"

_TEAM_NAME = "Hart of the Order"


@pytest.fixture(scope="module")
def _rec_inputs() -> RecInputs:
    """Build RecInputs once per module from the Task 4 golden fixtures."""
    state: dict[str, Any] = json.loads(_STATE_PATH.read_text())
    with open(_LEAGUE_YAML_PATH) as fh:
        league_yaml: dict[str, Any] = yaml.safe_load(fh)
    return recs_integration.compute_rec_inputs(state, _BOARD_PATH, league_yaml)


@pytest.fixture(scope="module")
def deltaroto_ctx(_rec_inputs: RecInputs):
    """Factory: returns a callable that builds a RecommendContext for a given scoring_mode."""

    def _make(*, scoring_mode: str) -> RecommendContext:
        return RecommendContext(
            scoring_mode=scoring_mode,
            team_name=_TEAM_NAME,
            picks_until_next=8,
            inputs=_rec_inputs,
        )

    return _make
