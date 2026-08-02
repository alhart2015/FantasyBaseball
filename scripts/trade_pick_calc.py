#!/usr/bin/env python
"""CLI for the trade-for-future-pick calculator.

Usage:
    python scripts/trade_pick_calc.py --send "Julio Rodriguez" --to "SkeleThor" --pick-round 5

Reads stored (last-refresh) state from Upstash; run a dashboard refresh first if
the state is stale. See docs/superpowers/specs/2026-07-31-trade-pick-calculator-design.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Scripts inject src/ into sys.path (repo convention) rather than relying solely
# on the editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Player names may carry non-ASCII (e.g. accents); reconfigure stdout so a name
# from data does not crash the report on Windows cp1252.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fantasy_baseball.analysis.trade_pick import (
    compute_trade_pick,
    render_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show the two-sided impact of a trade: this year's win odds and next year's pick value."
    )
    parser.add_argument("--send", required=True, help="Player you trade away (on your roster).")
    parser.add_argument("--to", required=True, help="Trade partner's team name.")
    parser.add_argument(
        "--receive",
        default=None,
        help="Player coming back (on the partner's roster). Makes this a two-way swap: "
        "the real incoming player fills the slot instead of a replacement-level filler.",
    )
    parser.add_argument(
        "--send-pick-round",
        type=int,
        default=None,
        help="Nominal round of a pick you GIVE UP, if the trade swaps picks. The report "
        "then shows the NET pick value rather than the gross value of the one received.",
    )
    parser.add_argument(
        "--receive-player-type",
        choices=["hitter", "pitcher"],
        default=None,
        help="Disambiguate a same-named player on the partner's roster.",
    )
    parser.add_argument(
        "--pick-round",
        type=int,
        required=True,
        help="Nominal round of the pick you receive (keeper rounds are subtracted).",
    )
    parser.add_argument(
        "--player-type",
        choices=["hitter", "pitcher"],
        default=None,
        help="Disambiguate when two same-named players are on your roster.",
    )
    parser.add_argument(
        "--pick-slot",
        choices=["round", "early", "mid", "late"],
        default="round",
        help="Narrow the pick's value within the drafted round (default: round average).",
    )
    parser.add_argument(
        "--iterations", type=int, default=2000, help="MC iterations (default 2000)."
    )
    parser.add_argument("--seed", type=int, default=42, help="MC seed (default 42).")
    args = parser.parse_args()

    try:
        result = compute_trade_pick(
            send=args.send,
            to=args.to,
            pick_round=args.pick_round,
            receive=args.receive,
            send_pick_round=args.send_pick_round,
            receive_player_type=args.receive_player_type,
            player_type=args.player_type,
            pick_slot=args.pick_slot,
            n_iter=args.iterations,
            seed=args.seed,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
