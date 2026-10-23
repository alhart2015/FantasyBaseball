"""Freeze the finished draft so draft-value can grade it.

Run once, right after the draft, before rebuilding the dashboard board or changing
team volumes. Copies the draft dashboard's state and board into
data/frozen_draft_<season_year>/ and records the team volumes the board was built with.

    python scripts/freeze_draft.py           # refuses if that season is already frozen
    python scripts/freeze_draft.py --force   # replace an existing freeze
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.analysis.draft_value import freeze_draft
from fantasy_baseball.config import load_config
from fantasy_baseball.web.app import DEFAULT_STATE_PATH

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
# create_app derives the board path from the state path the same way.
BOARD_PATH = DEFAULT_STATE_PATH.with_name(
    DEFAULT_STATE_PATH.stem + "_board" + DEFAULT_STATE_PATH.suffix
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace an existing freeze.")
    args = parser.parse_args()
    config = load_config(CONFIG_PATH)
    try:
        dest = freeze_draft(config, DEFAULT_STATE_PATH, BOARD_PATH, force=args.force)
    except (OSError, ValueError) as exc:
        print(f"Not frozen: {exc}")
        return 1
    print(f"Froze the {config.season_year} draft into {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
