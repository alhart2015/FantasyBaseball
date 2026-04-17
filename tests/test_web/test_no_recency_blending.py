"""Regression guardrails: recency blending is off-limits for roster decisions.

These tests fail if anyone re-introduces a call to the recency blending
module from production code paths. The `analysis/recency.py` module is
still permitted as a research tool — only the `lineup/blending.py` bridge
and direct `predict_*` calls from production are forbidden.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_FILES_TO_CHECK = [
    "src/fantasy_baseball/web/season_data.py",
    "src/fantasy_baseball/web/season_routes.py",
    "src/fantasy_baseball/trades/evaluate.py",
    "src/fantasy_baseball/lineup/waivers.py",
    "src/fantasy_baseball/lineup/optimizer.py",
    "src/fantasy_baseball/lineup/roster_audit.py",
    "src/fantasy_baseball/analysis/buy_low.py",
    "src/fantasy_baseball/simulation.py",
    "scripts/run_lineup.py",
]

FORBIDDEN_IMPORTS = [
    "from fantasy_baseball.lineup.blending",
    "import fantasy_baseball.lineup.blending",
    "from fantasy_baseball.analysis.recency",
    "import fantasy_baseball.analysis.recency",
]

FORBIDDEN_CALLS = [
    "blend_player_list",
    "blend_player_with_game_logs",
    "predict_reliability_blend",
    "predict_exponential_decay",
    "predict_fixed_blend",
    "apply_recency_blend",
]


def test_no_blending_imports_in_production_files():
    for rel_path in PRODUCTION_FILES_TO_CHECK:
        path = REPO_ROOT / rel_path
        assert path.is_file(), f"expected production file at {rel_path}"
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_IMPORTS:
            assert forbidden not in text, (
                f"{rel_path} imports recency blending ({forbidden!r}). "
                "Recency blending is off-limits for decision paths."
            )


def test_no_blending_calls_in_production_files():
    for rel_path in PRODUCTION_FILES_TO_CHECK:
        path = REPO_ROOT / rel_path
        assert path.is_file(), f"expected production file at {rel_path}"
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_CALLS:
            assert forbidden not in text, (
                f"{rel_path} calls {forbidden!r}. "
                "Recency blending is off-limits for decision paths."
            )
