"""Replay the REAL 2026 league draft and show what each scoring mode would have
recommended at every one of Hart's actual picks.

Source of truth: data/drafts/draft_2026-03-24_195809.json (verified 22/23 vs the
opening-day Upstash roster snapshot; the lone diff -- Contreras vs Webb at the
R15 slot -- is the unmodeled R18->TBD / R5<-TBD trade). Keepers are pre-loaded
from config; the 200-pick draft_log is replayed in order. Before each Hart pick
we reconstruct the exact board state (every team's actual picks so far) and run
each mode through the same recommend() seam the dashboard/sim use.

  python scripts/replay_real_draft.py            # immediate vs vopn vs finalslate

Valuation basis: CURRENT league calibration (sgp_denominators overrides +
DEFAULT_TEAM_IP from utils/constants), threaded via build_board_and_context.
Deliberate 2026-07-05 decision: these retro tools answer "what was the right
call given the BEST calibration we now have", unlike the draft-value tab,
which stays frozen at draft-day scale (5500 AB / 1450 IP, default denoms) to
measure picks against draft-day expectations. The two answer different
questions and are expected to disagree on pitcher/high-AVG grades.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fantasy_baseball.draft.recommend import RecommendContext, rank_for_mode, recommend
from fantasy_baseball.draft.roster_state import RosterState, get_filled_positions
from fantasy_baseball.utils.name_utils import normalize_name

REAL_DRAFT = PROJECT_ROOT / "data" / "drafts" / "draft_2026-03-24_195809.json"
HART = "Hart of the Order"
MODES = [
    ("deltaroto_immediate", "IMMEDIATE"),
    ("deltaroto_vopn", "VOPN"),
    ("deltaroto_finalslate", "FINALSLATE"),
]


class _Proxy:
    """Minimal tracker stand-in: _build_deltaroto_rec_inputs reads drafted_ids,
    user_roster_ids, and (finalslate only) current_pick/total_picks."""

    def __init__(self, drafted_ids, user_roster_ids, current_pick, total_picks):
        self.drafted_ids = list(drafted_ids)
        self.user_roster_ids = list(user_roster_ids)
        self.current_pick = current_pick
        self.total_picks = total_picks


def _resolve_pid(full_board, name):
    norm = normalize_name(name)
    matches = full_board[full_board["name_normalized"] == norm]
    if matches.empty:
        return None
    return str(matches.loc[matches["var"].idxmax()]["player_id"])


def _rec_at(
    sd,
    mode,
    board,
    full_board,
    config,
    scarcity_order,
    player_lookup,
    team_rosters,
    drafted,
    hart_pids,
    pick_no,
    total_picks,
    current_round,
    picks_until_next,
):
    """Return the mode's slot-gated recommendation name at this board state."""
    proxy = _Proxy(drafted, hart_pids, pick_no, total_picks)
    rec_inputs = sd._build_deltaroto_rec_inputs(
        board,
        full_board,
        proxy,
        config,
        team_rosters,
        player_lookup,
        team_name=HART,
        roster_ids=hart_pids,
        scoring_mode=mode,
        scarcity_order=scarcity_order,
    )
    ctx = RecommendContext(
        scoring_mode=mode,
        team_name=HART,
        picks_until_next=picks_until_next,
        inputs=rec_inputs,
        current_round=current_round,
    )
    filled = get_filled_positions(
        hart_pids, full_board, roster_slots=config.roster_slots, player_lookup=player_lookup
    )
    open_starters = RosterState.from_dicts(filled, config.roster_slots).unfilled_starter_slots()
    ranked = rank_for_mode(ctx)
    pick = recommend(
        ctx,
        strategy="default",
        open_starters=open_starters,
        pick_rank=0,
        current_round=current_round,
        closer_count=0,
        n_hitters=0,
        n_pitchers=0,
        ranked=ranked,
    )
    name = pick.name if pick is not None else "(none)"
    return name


def main():
    import simulate_draft as sd

    with open(REAL_DRAFT, encoding="utf-8") as fh:
        log = sorted(json.load(fh)["draft_log"], key=lambda e: e["pick"])
    ctx_b = sd.build_board_and_context(config_path=PROJECT_ROOT / "config" / "league.yaml")
    config = ctx_b["config"]
    board = ctx_b["board"]
    full_board = ctx_b["full_board"]
    scarcity_order = ctx_b["scarcity_order"]
    player_lookup = sd.build_player_lookup(board, full_board)

    name_by_num = dict(config.teams)
    num_by_name = {v: k for k, v in name_by_num.items()}
    total_picks = len(log)

    hart_picks = sorted(e["pick"] for e in log if e["team"] == HART)
    # picks_until_next for each Hart pick (opponent picks before his next turn).
    next_gap = {}
    for i, p in enumerate(hart_picks):
        next_gap[p] = (hart_picks[i + 1] - p - 1) if i + 1 < len(hart_picks) else 0

    rows = []  # one dict per Hart pick, filled across modes
    for mode, _label in MODES:
        # Fresh replay per mode (state is rebuilt identically; only the rec differs).
        team_rosters = {num: [] for num in name_by_num}
        drafted = set()
        for k in config.keepers:
            num = num_by_name.get(k["team"])
            pid = _resolve_pid(full_board, k["name"])
            if num is not None and pid is not None:
                team_rosters[num].append(pid)
                drafted.add(pid)

        idx = 0
        for entry in log:
            team = entry["team"]
            num = entry.get("team_num") or num_by_name.get(team)
            actual_pid = _resolve_pid(full_board, entry["player"])
            if team == HART:
                hart_pids = list(team_rosters[num])
                rec_name = _rec_at(
                    sd,
                    mode,
                    board,
                    full_board,
                    config,
                    scarcity_order,
                    player_lookup,
                    team_rosters,
                    drafted,
                    hart_pids,
                    entry["pick"],
                    total_picks,
                    entry["round"],
                    next_gap[entry["pick"]],
                )
                if mode == MODES[0][0]:
                    rows.append(
                        {
                            "round": entry["round"],
                            "pick": entry["pick"],
                            "actual": entry["player"],
                            "recs": {},
                        }
                    )
                row = rows[idx]
                row["recs"][mode] = rec_name
                idx += 1
            if actual_pid is not None:
                team_rosters[num].append(actual_pid)
                drafted.add(actual_pid)

    # Report: actual + each mode's rec, with a check per mode.
    hdr = f"{'Rd':>3} {'Pk':>4}  {'YOUR PICK':<22}"
    for _m, label in MODES:
        hdr += f"  {label:<20}"
    out = [hdr, "-" * len(hdr)]
    tally = {m: 0 for m, _ in MODES}
    for r in rows:
        line = f"{r['round']:>3} {r['pick']:>4}  {r['actual']:<22}"
        for mode, _label in MODES:
            rec = r["recs"].get(mode, "?")
            same = normalize_name(rec) == normalize_name(r["actual"])
            if same:
                tally[mode] += 1
            mark = "*" if same else " "
            line += f"  {mark}{rec:<19}"
        out.append(line)
    out.append("-" * len(hdr))
    summ = "Agreement with your actual pick:  " + "   ".join(
        f"{label} {tally[m]}/{len(rows)}" for m, label in MODES
    )
    out.append(summ)
    out.append("(* = mode's recommendation == your actual pick)")
    text = "\n".join(out)
    print(text)
    (PROJECT_ROOT / "replay_real_draft.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
