"""Simulate a full draft to evaluate strategy.

Usage:
    python scripts/simulate_draft.py

- Your team: always takes the top leverage-weighted recommendation.
- Other teams: take the highest-ADP available player they can legally roster.
- Roster limits are enforced for all teams.

Outputs projected roto standings at the end.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.draft.board import build_draft_board, apply_keepers
from fantasy_baseball.draft.tracker import DraftTracker
from fantasy_baseball.draft.balance import CategoryBalance, calculate_draft_leverage
from fantasy_baseball.draft.recommender import (
    get_recommendations,
    get_filled_positions,
)
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import can_fill_slot

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"


def _can_roster(player_positions, filled, roster_slots):
    """Check if a player can fit in any open slot."""
    for pos, total in roster_slots.items():
        if filled.get(pos, 0) < total and can_fill_slot(player_positions, pos):
            return True
    return False


def _assign_slot(player_positions, filled, roster_slots):
    """Assign a player to the best available slot, updating filled in place."""
    # Try specific slots first, then flex, then bench, then IL
    active = [p for p in roster_slots if p not in ("BN", "IL", "IF", "UTIL")]
    flex = ["IF", "UTIL"]
    overflow = ["BN", "IL"]
    for slot in active + flex + overflow:
        if slot not in roster_slots:
            continue
        if filled.get(slot, 0) < roster_slots[slot] and can_fill_slot(player_positions, slot):
            filled[slot] = filled.get(slot, 0) + 1
            return slot
    return None


def main():
    config = load_config(CONFIG_PATH)
    print(f"Simulating draft | {config.team_name} at position {config.draft_position}")
    print(f"League: {config.num_teams} teams, {sum(config.roster_slots.values())} roster slots")
    print()

    # Build board
    print("Building draft board...")
    full_board = build_draft_board(
        projections_dir=PROJECTIONS_DIR,
        positions_path=POSITIONS_PATH,
        systems=config.projection_systems,
        weights=config.projection_weights or None,
        sgp_overrides=config.sgp_overrides or None,
        roster_slots=config.roster_slots or None,
        num_teams=config.num_teams,
    )
    board = apply_keepers(full_board, config.keepers)

    # Build ADP ranking for other teams (lower ADP = picked earlier)
    # Use the board which has ADP from blending
    adp_board = board.copy()
    if "adp" not in adp_board.columns:
        print("WARNING: No ADP data found. Other teams will use VAR ranking.")
        adp_board["adp"] = range(len(adp_board))
    adp_board = adp_board.sort_values("adp", ascending=True)

    # Initialize tracker
    user_keepers = [k for k in config.keepers if k.get("team") == config.team_name]
    rounds = sum(config.roster_slots.values()) - len(user_keepers)
    tracker = DraftTracker(
        num_teams=config.num_teams,
        user_position=config.draft_position,
        rounds=rounds,
    )
    balance = CategoryBalance()

    # Per-team filled positions tracking
    team_filled = {i: {} for i in range(1, config.num_teams + 1)}

    # Register keepers
    for keeper in config.keepers:
        for num, name in config.teams.items():
            if name == keeper["team"]:
                norm = normalize_name(keeper["name"])
                matches = full_board[full_board["name_normalized"] == norm]
                if not matches.empty:
                    best = matches.loc[matches["var"].idxmax()]
                    is_user = keeper.get("team") == config.team_name
                    if is_user:
                        balance.add_player(best)
                    tracker.draft_player(best["name"], is_user=is_user,
                                         player_id=best["player_id"])
                    _assign_slot(best["positions"], team_filled[num],
                                 config.roster_slots)
                break

    print(f"Draft pool: {len(board)} players")
    print(f"Keepers registered: {len(config.keepers)}")
    print()

    # Run draft
    pick_log = []
    while tracker.current_pick <= tracker.total_picks:
        team_num = tracker.picking_team
        team_label = config.teams.get(team_num, f"Team {team_num}")
        is_user = tracker.is_user_pick

        if is_user:
            # Use the full recommendation engine
            filled = get_filled_positions(
                tracker.user_roster_ids, full_board,
                roster_slots=config.roster_slots,
            )
            leverage = calculate_draft_leverage(
                balance.get_totals(),
                picks_made=len(tracker.user_roster),
                total_picks=rounds,
            )
            recs = get_recommendations(
                board, drafted=tracker.drafted_ids,
                user_roster=tracker.user_roster,
                n=5, filled_positions=filled,
                roster_slots=config.roster_slots,
                num_teams=config.num_teams,
                draft_leverage=leverage,
            )
            if recs:
                pick_name = recs[0]["name"]
                pick_pos = recs[0]["best_position"]
                # Find the player_id
                rows = board[board["name"] == pick_name]
                if not rows.empty:
                    pid = rows.iloc[0]["player_id"]
                else:
                    pid = pick_name + "::unknown"
            else:
                # Fallback: pick best available by ADP that can be rostered anywhere
                available_ids = set(tracker.drafted_ids)
                pick_name, pid, pick_pos = None, "", ""
                for _, row in adp_board.iterrows():
                    if row["player_id"] not in available_ids:
                        pick_name = row["name"]
                        pid = row["player_id"]
                        pick_pos = row.get("best_position", "")
                        break

            if pick_name:
                tracker.draft_player(pick_name, is_user=True, player_id=pid)
                row = board[board["player_id"] == pid]
                if not row.empty:
                    balance.add_player(row.iloc[0])
                    _assign_slot(row.iloc[0]["positions"],
                                 team_filled[team_num], config.roster_slots)
            else:
                pick_name = "(no pick)"
                pick_pos = ""
        else:
            # Other teams: pick best available by ADP that they can roster
            available_ids = set(tracker.drafted_ids)
            pick_name = None
            pick_pos = ""
            pid = ""
            for _, row in adp_board.iterrows():
                if row["player_id"] in available_ids:
                    continue
                positions = row["positions"]
                if _can_roster(positions, team_filled[team_num],
                               config.roster_slots):
                    pick_name = row["name"]
                    pid = row["player_id"]
                    pick_pos = row.get("best_position", "")
                    tracker.draft_player(pick_name, is_user=False,
                                         player_id=pid)
                    _assign_slot(positions, team_filled[team_num],
                                 config.roster_slots)
                    break

            if pick_name is None:
                pick_name = "(no pick)"
                tracker.advance()
                continue

        rnd = tracker.current_round
        pick = tracker.current_pick
        marker = " <<<" if is_user else ""
        pick_log.append({
            "round": rnd, "pick": pick, "team": team_label,
            "player": pick_name, "pos": pick_pos, "is_user": is_user,
        })

        if is_user:
            print(f"  R{rnd:>2} #{pick:>3} {team_label:<30} {pick_name:<25} {pick_pos}{marker}")
        elif rnd <= 5:
            # Show early rounds for context
            print(f"  R{rnd:>2} #{pick:>3} {team_label:<30} {pick_name:<25} {pick_pos}")

        tracker.advance()

    # === Results ===
    print()
    print("=" * 80)
    print("DRAFT COMPLETE")
    print("=" * 80)

    # Show user roster
    print(f"\n{config.team_name} ROSTER:")
    for name in tracker.user_roster:
        rows = full_board[full_board["name"] == name]
        if not rows.empty:
            r = rows.iloc[0]
            print(f"  {r['name']:<25} {'/'.join(r['positions'][:3]):<12} "
                  f"{'hitter' if r['player_type']=='hitter' else 'pitcher'}")

    # Reconstruct all team rosters and project standings
    team_players = {i: [] for i in range(1, config.num_teams + 1)}

    # Keepers
    for keeper in config.keepers:
        for num, name in config.teams.items():
            if name == keeper["team"]:
                norm = normalize_name(keeper["name"])
                matches = full_board[full_board["name_normalized"] == norm]
                if not matches.empty:
                    team_players[num].append(matches.loc[matches["var"].idxmax()])
                break

    # Draft picks
    num_keepers = len(config.keepers)
    draft_entries = list(zip(
        tracker.drafted_players[num_keepers:],
        tracker.drafted_ids[num_keepers:],
    ))
    for pick_num, (name, pid) in enumerate(draft_entries, 1):
        rnd = (pick_num - 1) // config.num_teams + 1
        pos = (pick_num - 1) % config.num_teams + 1
        team = pos if rnd % 2 == 1 else config.num_teams - pos + 1
        rows = board[board["player_id"] == pid]
        if not rows.empty:
            team_players[team].append(rows.iloc[0])

    # Project stats
    results = []
    for tn in range(1, config.num_teams + 1):
        tname = config.teams.get(tn, f"Team {tn}")
        hitters = [p for p in team_players[tn] if p["player_type"] == "hitter"]
        pitchers = [p for p in team_players[tn] if p["player_type"] == "pitcher"]
        r = sum(h.get("r", 0) for h in hitters)
        hr = sum(h.get("hr", 0) for h in hitters)
        rbi = sum(h.get("rbi", 0) for h in hitters)
        sb = sum(h.get("sb", 0) for h in hitters)
        th = sum(h.get("h", 0) for h in hitters)
        tab = sum(h.get("ab", 0) for h in hitters)
        avg = th / tab if tab > 0 else 0
        w = sum(p.get("w", 0) for p in pitchers)
        k = sum(p.get("k", 0) for p in pitchers)
        sv = sum(p.get("sv", 0) for p in pitchers)
        tip = sum(p.get("ip", 0) for p in pitchers)
        ter = sum(p.get("er", 0) for p in pitchers)
        tbb = sum(p.get("bb", 0) for p in pitchers)
        tha = sum(p.get("h_allowed", 0) for p in pitchers)
        era = ter * 9 / tip if tip > 0 else 0
        whip = (tbb + tha) / tip if tip > 0 else 0
        results.append({
            "team": tname, "R": r, "HR": hr, "RBI": rbi, "SB": sb, "AVG": avg,
            "W": w, "K": k, "SV": sv, "ERA": era, "WHIP": whip,
            "nh": len(hitters), "np": len(pitchers),
        })

    # Roto points
    all_cats = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
    inverse = {"ERA", "WHIP"}
    for cat in all_cats:
        rev = cat not in inverse
        st = sorted(results, key=lambda x: x[cat], reverse=rev)
        for i, t in enumerate(st):
            t[f"{cat}_p"] = config.num_teams - i

    for t in results:
        t["tot"] = sum(t[f"{c}_p"] for c in all_cats)

    results.sort(key=lambda x: x["tot"], reverse=True)

    # Print standings
    print(f"\nPROJECTED ROTO STANDINGS")
    print("=" * 132)
    print(f"{'Rk':<3} {'Team':<32} {'Pts':>4}  "
          f"{'R':>5} {'HR':>4} {'RBI':>5} {'SB':>4} {'AVG':>6}  "
          f"{'W':>4} {'K':>5} {'SV':>4} {'ERA':>5} {'WHIP':>6}  "
          f"{'H':>2}/{'P':>2}")
    print("-" * 132)
    for i, t in enumerate(results, 1):
        m = " <<<" if t["team"] == config.team_name else ""
        print(f"{i:<3} {t['team']:<32} {t['tot']:>4}  "
              f"{t['R']:>5.0f} {t['HR']:>4.0f} {t['RBI']:>5.0f} "
              f"{t['SB']:>4.0f} {t['AVG']:>6.3f}  "
              f"{t['W']:>4.0f} {t['K']:>5.0f} {t['SV']:>4.0f} "
              f"{t['ERA']:>5.2f} {t['WHIP']:>6.3f}  "
              f"{t['nh']:>2}/{t['np']:>2}{m}")

    # Category breakdown
    print(f"\nROTO POINTS BY CATEGORY (10=best, 1=worst)")
    print("=" * 97)
    print(f"{'Team':<32} ", end="")
    for c in all_cats:
        print(f"{c:>5}", end="")
    print(f"{'TOT':>6}")
    print("-" * 97)
    for t in results:
        m = " <<<" if t["team"] == config.team_name else ""
        print(f"{t['team']:<32} ", end="")
        for c in all_cats:
            print(f"{t[f'{c}_p']:>5}", end="")
        print(f"{t['tot']:>6}{m}")

    # User team summary
    hart = next(t for t in results if t["team"] == config.team_name)
    rank = next(i + 1 for i, t in enumerate(results) if t["team"] == config.team_name)
    suf = {1: "st", 2: "nd", 3: "rd"}.get(rank, "th")
    print(f"\n{'=' * 60}")
    print(f"{config.team_name} - Projected {rank}{suf} place ({hart['tot']} pts)")
    print(f"{'=' * 60}")
    print(f"Roster: {hart['nh']}H / {hart['np']}P")

    def fmt(c, v):
        return f"{v:.3f}" if c in ("AVG", "ERA", "WHIP") else f"{v:.0f}"

    top = [(c, hart[f"{c}_p"], hart[c]) for c in all_cats if hart[f"{c}_p"] >= 8]
    mid = [(c, hart[f"{c}_p"], hart[c]) for c in all_cats if 4 <= hart[f"{c}_p"] <= 7]
    bot = [(c, hart[f"{c}_p"], hart[c]) for c in all_cats if hart[f"{c}_p"] <= 3]

    if top:
        print("\nStrengths (8-10 pts):")
        for c, p, v in sorted(top, key=lambda x: -x[1]):
            print(f"  {c:>4}: {fmt(c, v):>7} ({p} pts)")
    if mid:
        print("\nMiddle of pack (4-7 pts):")
        for c, p, v in sorted(mid, key=lambda x: -x[1]):
            print(f"  {c:>4}: {fmt(c, v):>7} ({p} pts)")
    if bot:
        print("\nWeak categories (1-3 pts):")
        for c, p, v in sorted(bot, key=lambda x: x[1]):
            print(f"  {c:>4}: {fmt(c, v):>7} ({p} pts)")


if __name__ == "__main__":
    main()
