"""Run the season dashboard refresh pipeline locally.

Same code path as the Render deploy, so local success means remote success.

Usage:
    python scripts/run_lineup.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.web.season_data import run_full_refresh

run_full_refresh()
