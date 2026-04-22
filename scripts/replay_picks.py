"""Replay a mock draft pick-by-pick, showing what was available at each user pick.

Usage:
    python scripts/replay_picks.py                          # latest draft
    python scripts/replay_picks.py data/drafts/mock_*.json  # specific file
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from simulate_draft import build_board_and_context

from fantasy_baseball.draft.balance import CategoryBalance
from fantasy_baseball.draft.recommender import (
    calculate_vona_scores,
    get_filled_positions,
    get_recommendations,
)
from fantasy_baseball.draft.strategy import CLOSER_SV_THRESHOLD
from fantasy_baseball.utils.constants import Category

HART = None  # set from draft metadata


def main():
    if len(sys.argv) < 2:
        drafts_dir = PROJECT_ROOT / "data" / "drafts"
        mocks = sorted(drafts_dir.glob("*.json"))
        if not mocks:
            print("No drafts found.")
            sys.exit(1)
        draft_path = mocks[-1]
    else:
        draft_path = Path(sys.argv[1])

    with open(draft_path) as f:
        draft_data = json.load(f)

    print(f"Replaying: {draft_path.name}")
    ctx = build_board_and_context()
    config = ctx["config"]
    full_board = ctx["full_board"]
    board = ctx["board"]

    # Determine user's team from metadata
    global HART
    meta = draft_data.get("metadata", {})
    HART = meta.get("user_team")
    if not HART:
        pos = meta.get("draft_position", 8)
        HART = config.teams.get(pos, f"Team {pos}")
    print(f"User team: {HART} (position {meta.get('draft_position', '?')})")

    # Replay state
    balance = CategoryBalance()
    drafted_ids = []
    user_roster_ids = []
    user_roster = []

    hart_pick_num = 0

    for entry in draft_data["draft_log"]:
        is_hart = entry["team"] == HART

        if is_hart:
            hart_pick_num += 1
            available = board[~board["player_id"].isin(drafted_ids)]

            # Top 5 by ADP
            by_adp = available.sort_values("adp", ascending=True).head(5)
            # Top 5 by total_sgp
            by_sgp = available.sort_values("total_sgp", ascending=False).head(5)
            # Top 5 by VAR
            by_var = available.sort_values("var", ascending=False).head(5)
            # Top 5 by VONA
            vona_scores = calculate_vona_scores(available, 10)
            available_v = available.copy()
            available_v["vona"] = available_v["player_id"].map(vona_scores).fillna(0)
            by_vona = available_v.sort_values("vona", ascending=False).head(5)

            # Get the actual recommendation
            filled = get_filled_positions(user_roster_ids, full_board,
                                         roster_slots=config.roster_slots)
            recs = get_recommendations(
                board, drafted=drafted_ids,
                user_roster=user_roster,
                n=5, filled_positions=filled,
                roster_slots=config.roster_slots,
                num_teams=config.num_teams,
                scoring_mode="vona",
            )

            picked = entry["player"]
            picked_rows = board[board["player_id"] == entry["player_id"]]
            picked_player = picked_rows.iloc[0] if not picked_rows.empty else None

            # Get scoring details for the picked player
            picked_vona = vona_scores.get(entry["player_id"], 0)
            picked_sv = picked_player.get("sv", 0) if picked_player is not None else 0
            picked_var = picked_player.get("var", 0) if picked_player is not None else 0
            picked_sgp = picked_player.get("total_sgp", 0) if picked_player is not None else 0

            # Count closers
            closer_count = 0
            for pid in user_roster_ids:
                rows = board[board["player_id"] == pid]
                if rows.empty:
                    rows = full_board[full_board["player_id"] == pid]
                if not rows.empty and rows.iloc[0].get("sv", 0) >= CLOSER_SV_THRESHOLD:
                    closer_count += 1

            # Print
            print()
            print("=" * 100)
            rnd = entry["round"]
            print(f"YOUR PICK #{hart_pick_num} (Overall #{entry['pick']}, Round {rnd})"
                  f"  |  Closers: {closer_count}/3  |  Roster: {len(user_roster)} players")
            print("=" * 100)

            def _fmt_row(name, ptype, val, positions=None, sv=0):
                pos = "/".join(positions[:2]) if positions else ""
                cl = " [CL]" if sv and sv >= 20 else ""
                return f"{name:<25} {ptype[0].upper():<2} {pos:<10} {val:>7.2f}{cl}"

            print(f"\n  {'By ADP':<50} {'By SGP':<50}")
            for i in range(5):
                a = by_adp.iloc[i] if i < len(by_adp) else None
                s = by_sgp.iloc[i] if i < len(by_sgp) else None
                left = _fmt_row(a["name"], a["player_type"], a["adp"], a["positions"], a.get("sv",0)) if a is not None else ""
                right = _fmt_row(s["name"], s["player_type"], s["total_sgp"], s["positions"], s.get("sv",0)) if s is not None else ""
                print(f"  {i+1}. {left:<48} {i+1}. {right}")

            print(f"\n  {'By VAR':<50} {'By VONA (raw urgency)':<50}")
            for i in range(5):
                v = by_var.iloc[i] if i < len(by_var) else None
                vo = by_vona.iloc[i] if i < len(by_vona) else None
                left = _fmt_row(v["name"], v["player_type"], v["var"], v["positions"], v.get("sv",0)) if v is not None else ""
                right = _fmt_row(vo["name"], vo["player_type"], vo["vona"], vo["positions"], vo.get("sv",0)) if vo is not None else ""
                print(f"  {i+1}. {left:<48} {i+1}. {right}")

            # Show what strategy recommended
            print(f"\n  Strategy recommendation: {recs[0]['name'] if recs else 'N/A'}")
            print(f"  >>> PICKED: {picked}")
            if picked_player is not None:
                pos = "/".join(picked_player["positions"][:3])
                print(f"      {picked_player['player_type']} | {pos} | "
                      f"VAR={picked_var:.2f} | SGP={picked_sgp:.2f} | "
                      f"VONA={picked_vona:.2f}")

            # Explanation
            is_closer = picked_sv and picked_sv >= CLOSER_SV_THRESHOLD
            explanation = _explain_pick(
                picked, picked_player, picked_var, picked_vona,
                is_closer, closer_count, rnd, by_var, balance,
            )
            print(f"      Why: {explanation}")

        # Record pick
        pid = entry["player_id"]
        drafted_ids.append(pid)
        if is_hart:
            user_roster.append(entry["player"])
            user_roster_ids.append(pid)
            rows = board[board["player_id"] == pid]
            if not rows.empty:
                balance.add_player(rows.iloc[0])


def _explain_pick(name, player, var, vona, is_closer, closer_count,
                  rnd, by_var, balance):
    """Generate a brief explanation of why this player was recommended."""
    if player is None:
        return "Player not found on board"

    ptype = player["player_type"]

    # Check if this was a forced closer deadline
    from fantasy_baseball.draft.strategy import NO_PUNT_STAGGER_DEADLINES
    if is_closer and closer_count < 3:
        deadline = NO_PUNT_STAGGER_DEADLINES[closer_count] if closer_count < len(NO_PUNT_STAGGER_DEADLINES) else 99
        if rnd >= deadline:
            return (f"Closer deadline #{closer_count+1} triggered (round {deadline}). "
                    f"Best available closer by VAR ({var:.2f}).")

    if is_closer:
        return (f"VONA urgency ({vona:.2f}) — closer scarcity makes this pick "
                f"more urgent than higher-VAR alternatives.")

    if ptype == "pitcher":
        top_var = by_var.iloc[0]
        if name == top_var["name"]:
            return f"Top pitcher by VAR ({var:.2f})."
        return (f"VONA score ({vona:.2f}) combined with VAR ({var:.2f}) "
                f"makes this the best available.")

    # Hitter
    avg = player.get("avg", 0)
    sb = player.get("sb", 0)
    hr = player.get("hr", 0)

    strengths = []
    if sb >= 20:
        strengths.append(f"SB={sb:.0f}")
    if hr >= 25:
        strengths.append(f"HR={hr:.0f}")
    if avg >= 0.270:
        strengths.append(f"AVG={avg:.3f}")

    totals = balance.get_totals()
    reason = ""
    if (totals.get(Category.SB) or 0) < 100 and sb >= 15:
        reason = "SB is a top category need"
    elif (totals.get(Category.AVG) or 0) < 0.255 and avg >= 0.260:
        reason = "protects AVG floor"
    elif (totals.get(Category.HR) or 0) < 100 and hr >= 20:
        reason = "HR contribution needed"

    strength_str = f" ({', '.join(strengths)})" if strengths else ""
    tail = f" — {reason}" if reason else ""
    return f"Best hitter by VONA. VAR={var:.2f}{strength_str}{tail}."


if __name__ == "__main__":
    main()
