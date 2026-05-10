"""Trace the AVG path for one swap (Oneil Cruz -> Ian Happ replacement)
through apply_swap_delta + score_roto so we can see whether AVG is
actually 0 or just rounded to 0 in display.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantasy_baseball.draft.recs_integration import compute_rec_inputs
from fantasy_baseball.scoring import score_roto_dict
from fantasy_baseball.trades.evaluate import (
    apply_swap_delta,
    player_rest_of_season_stats,
)


def main() -> None:
    state = json.loads((ROOT / "data" / "draft_state.json").read_text())
    league_yaml = yaml.safe_load((ROOT / "config" / "league.yaml").read_text())
    board_path = ROOT / "data" / "draft_state_board.json"

    inputs = compute_rec_inputs(state, board_path, league_yaml)
    team = state["on_the_clock"]

    # Get Cruz and the OF replacement
    cruz = next(p for p in inputs.candidates if p.name == "Oneil Cruz")
    of_rep = inputs.replacements["OF"]
    print(f"Candidate:    {cruz.name}")
    print(f"Replacement:  {of_rep.name}")
    print()

    cruz_ros = player_rest_of_season_stats(cruz)
    rep_ros = player_rest_of_season_stats(of_rep)
    print(
        f"Cruz ROS: ab={cruz_ros['ab']:.0f}  hits-implied={cruz_ros['ab'] * cruz_ros['AVG']:.1f}  avg={cruz_ros['AVG']:.3f}  R={cruz_ros['R']:.1f}  HR={cruz_ros['HR']:.1f}  RBI={cruz_ros['RBI']:.1f}  SB={cruz_ros['SB']:.1f}"
    )
    print(
        f"Happ ROS: ab={rep_ros['ab']:.0f}  hits-implied={rep_ros['ab'] * rep_ros['AVG']:.1f}  avg={rep_ros['AVG']:.3f}  R={rep_ros['R']:.1f}  HR={rep_ros['HR']:.1f}  RBI={rep_ros['RBI']:.1f}  SB={rep_ros['SB']:.1f}"
    )
    print()

    # Team baseline
    all_before = {e.team_name: e.stats.to_dict() for e in inputs.projected_standings.entries}
    user_before = all_before[team]
    print(f"Team baseline ({team}):")
    for k, v in user_before.items():
        print(f"  {k:6s} = {v:.3f}")
    print()

    # Apply the swap
    new_stats = apply_swap_delta(user_before, rep_ros, cruz_ros)
    print("Team after swap:")
    for k, v in new_stats.items():
        delta = v - user_before[k]
        print(f"  {k:6s} = {v:.3f}   (delta = {delta:+.5f})")
    print()

    # Run score_roto before/after
    all_after = dict(all_before)
    all_after[team] = new_stats

    roto_before = score_roto_dict(all_before, team_sds=inputs.team_sds)
    roto_after = score_roto_dict(all_after, team_sds=inputs.team_sds)

    print("Roto points before / after for the user's team:")
    cats = ["R", "HR", "RBI", "SB", "AVG", "W", "SV", "K", "ERA", "WHIP"]
    for cat in cats:
        before = roto_before[team][f"{cat}_pts"]
        after = roto_after[team][f"{cat}_pts"]
        print(f"  {cat:6s}  before={before:.4f}  after={after:.4f}  delta={after - before:+.5f}")
    print(
        f"  TOTAL   before={roto_before[team]['total']:.4f}  after={roto_after[team]['total']:.4f}  delta={roto_after[team]['total'] - roto_before[team]['total']:+.5f}"
    )
    print()

    # Inspect team SDs for AVG
    print("Per-team AVG / SD spread:")
    for tname in sorted(all_before):
        avg = all_before[tname]["AVG"]
        sd = inputs.team_sds.get(tname, {}).get(
            __import__("fantasy_baseball.utils.constants", fromlist=["Category"]).Category.AVG, 0.0
        )
        print(f"  {tname:35s}  AVG={avg:.4f}  sd_avg={sd:.5f}")


if __name__ == "__main__":
    main()
