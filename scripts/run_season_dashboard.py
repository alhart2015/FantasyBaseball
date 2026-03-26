#!/usr/bin/env python3
"""Launch the season dashboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fantasy_baseball.web.season_app import create_app

if __name__ == "__main__":
    app = create_app()
    print("Season dashboard: http://localhost:5001")
    app.run(port=5001, debug=True)
