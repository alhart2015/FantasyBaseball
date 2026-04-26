"""One-off diagnostic: print per-category ERoto deltas for the top recs
plus a target player (Cruz) so we can see *why* he ranks where he does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantasy_baseball.draft.eroto_recs import rank_candidates  # noqa: E402
from fantasy_baseball.draft.recs_integration import compute_rec_inputs  # noqa: E402


def main() -> None:
    state = json.loads((ROOT / "data" / "draft_state.json").read_text())
    league_yaml = yaml.safe_load((ROOT / "config" / "league.yaml").read_text())
    board_path = ROOT / "data" / "draft_state_board.json"

    inputs = compute_rec_inputs(state, board_path, league_yaml)
    team = state["on_the_clock"]
    print(f"on_the_clock = {team}")
    print(f"candidates   = {len(inputs.candidates)}")
    print(f"replacements = {sorted(inputs.replacements.keys())}")
    print()
    print("Replacement player at each position:")
    for pos in sorted(inputs.replacements):
        rep = inputs.replacements[pos]
        ros = rep.rest_of_season
        sgp = getattr(ros, "sgp", None) if ros is not None else None
        print(f"  {pos:6s}  {rep.name:25s}  positions={[str(p) for p in rep.positions]}  sgp={sgp}")
    print()

    rows = rank_candidates(
        candidates=inputs.candidates,
        replacements=inputs.replacements,
        team_name=team,
        projected_standings=inputs.projected_standings,
        team_sds=inputs.team_sds,
        picks_until_next_turn=0,
        adp_table=inputs.adp_table,
    )

    # Index by player_id for ADP lookup
    adp = inputs.adp_table

    cats = ["R", "HR", "RBI", "SB", "AVG", "W", "SV", "K", "ERA", "WHIP"]

    def fmt_row(r) -> str:
        a = adp.get(r.player_id)
        per = "  ".join(f"{c}={r.per_category.get(c, 0):+.2f}" for c in cats)
        pos = "/".join(r.positions[:3])
        return f"  {r.name:25s} adp={a:6.1f} pos={pos:12s} delta={r.immediate_delta:+.3f} | {per}"

    print("=== Top 15 recs by immediate_delta ===")
    for r in rows[:15]:
        print(fmt_row(r))
    print()

    # Cruz + his ADP-30-90 peers
    cruz = next((r for r in rows if r.name == "Oneil Cruz"), None)
    if cruz is None:
        print("Oneil Cruz not in candidates")
        return
    print(f"=== Oneil Cruz (rank {rows.index(cruz) + 1}) ===")
    print(fmt_row(cruz))
    print()

    print("=== ADP 30-90 cohort, sorted by immediate_delta desc (top 20) ===")
    band = [r for r in rows if 30 <= adp.get(r.player_id) <= 90]
    band.sort(key=lambda r: r.immediate_delta, reverse=True)
    for r in band[:20]:
        print(fmt_row(r))
    print()

    # Where does Cruz rank vs his ADP cohort and the players ahead of him by ADP
    ahead_by_adp = [r for r in rows if adp.get(r.player_id) < adp.get(cruz.player_id)]
    ahead_by_adp.sort(key=lambda r: r.immediate_delta, reverse=True)
    cruz_rank_in_ahead = sum(1 for r in ahead_by_adp if r.immediate_delta > cruz.immediate_delta)
    print(
        f"Players with ADP < {adp.get(cruz.player_id):.1f}: {len(ahead_by_adp)}"
        f" — of those, {cruz_rank_in_ahead} have higher immediate_delta than Cruz"
    )


if __name__ == "__main__":
    main()
