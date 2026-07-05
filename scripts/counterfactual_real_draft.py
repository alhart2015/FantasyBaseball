"""Counterfactual: would deltaroto_immediate have built Hart a better team than
his ACTUAL three_closers/var draft, in his REAL draft?

Method (single shared pool, no double-counting):
  * Walk the real draft (data/drafts/draft_2026-03-24_195809.json) in pick order.
  * At Hart's slots, immediate picks the best available (recommend() seam, default
    overlay, position-gated -- exactly what the dashboard would surface).
  * At opponents' slots, they take their REAL player if it is still available; if
    immediate sniped it, they fall back to the best available by ADP that fits an
    open slot. Every player lands on exactly one team.
Then score BOTH final states the same way the sim scores standings (active-lineup
selection via _score_roto) and compare Hart's roto points, rank, and MC win%.

  python scripts/counterfactual_real_draft.py

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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fantasy_baseball.draft.recommend import RecommendContext, rank_for_mode, recommend
from fantasy_baseball.draft.roster_state import RosterState, get_filled_positions
from fantasy_baseball.utils.name_utils import normalize_name

REAL_DRAFT = PROJECT_ROOT / "data" / "drafts" / "draft_2026-03-24_195809.json"
HART = "Hart of the Order"
N_MC = 4000
CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "ERA", "WHIP", "SV"]


class _Proxy:
    def __init__(self, drafted_ids, user_roster_ids, current_pick, total_picks):
        self.drafted_ids = list(drafted_ids)
        self.user_roster_ids = list(user_roster_ids)
        self.current_pick = current_pick
        self.total_picks = total_picks


def main():
    import simulate_draft as sd
    from mc_eval import _mc_win_rank

    with open(REAL_DRAFT, encoding="utf-8") as fh:
        log = sorted(json.load(fh)["draft_log"], key=lambda e: e["pick"])
    ctx_b = sd.build_board_and_context(config_path=PROJECT_ROOT / "config" / "league.yaml")
    config = ctx_b["config"]
    board = ctx_b["board"]
    full_board = ctx_b["full_board"]
    scarcity_order = ctx_b["scarcity_order"]
    player_lookup = sd.build_player_lookup(board, full_board)
    board_by_id, _replacements, adp_table, _ordered, _tn = sd._sim_static_inputs(board, config)

    name_by_num = dict(config.teams)
    num_by_name = {v: k for k, v in name_by_num.items()}
    total_picks = len(log)
    adp_sorted = sorted(board_by_id.keys(), key=lambda pid: adp_table.get(pid))

    _resolve_cache: dict[str, str | None] = {}

    def resolve(name):
        if name in _resolve_cache:
            return _resolve_cache[name]
        m = full_board[full_board["name_normalized"] == normalize_name(name)]
        pid = None if m.empty else str(m.loc[m["var"].idxmax()]["player_id"])
        _resolve_cache[name] = pid
        return pid

    def name_of(pid):
        row = player_lookup.get(pid)
        if row is not None:
            return row["name"]
        fb = full_board[full_board["player_id"] == pid]
        return fb.iloc[0]["name"] if not fb.empty else pid

    def preload_keepers():
        tr = {num: [] for num in name_by_num}
        drafted = set()
        for k in config.keepers:
            num = num_by_name.get(k["team"])
            pid = resolve(k["name"])
            if num is not None and pid is not None:
                tr[num].append(pid)
                drafted.add(pid)
        return tr, drafted

    def fallback_adp(num, drafted, tr):
        """Best available by ADP that fits an open slot for team `num`."""
        filled = get_filled_positions(
            tr[num], full_board, roster_slots=config.roster_slots, player_lookup=player_lookup
        )
        rstate = RosterState.from_dicts(filled, config.roster_slots)
        for pid in adp_sorted:
            if pid in drafted:
                continue
            row = player_lookup.get(pid)
            positions = row["positions"] if row is not None else []
            if rstate.any_slot_open_for(positions):
                return pid
        # nothing slot-legal: take the best available regardless
        for pid in adp_sorted:
            if pid not in drafted:
                return pid
        return None

    def immediate_pick(tr, drafted, entry):
        num = num_by_name[HART]
        hart_pids = list(tr[num])
        proxy = _Proxy(drafted, hart_pids, entry["pick"], total_picks)
        ri = sd._build_deltaroto_rec_inputs(
            board,
            full_board,
            proxy,
            config,
            tr,
            player_lookup,
            team_name=HART,
            roster_ids=hart_pids,
            scoring_mode="deltaroto_immediate",
            scarcity_order=scarcity_order,
        )
        ctx = RecommendContext(
            scoring_mode="deltaroto_immediate",
            team_name=HART,
            picks_until_next=0,
            inputs=ri,
            current_round=entry["round"],
        )
        filled = get_filled_positions(
            hart_pids, full_board, roster_slots=config.roster_slots, player_lookup=player_lookup
        )
        open_st = RosterState.from_dicts(filled, config.roster_slots).unfilled_starter_slots()
        ranked = rank_for_mode(ctx)
        pick = recommend(
            ctx,
            strategy="default",
            open_starters=open_st,
            pick_rank=0,
            current_round=entry["round"],
            closer_count=0,
            n_hitters=0,
            n_pitchers=0,
            ranked=ranked,
        )
        if pick is not None:
            return pick.player_id
        return fallback_adp(num, drafted, tr)

    def build_real():
        tr, drafted = preload_keepers()
        for e in log:
            num = e.get("team_num") or num_by_name.get(e["team"])
            pid = resolve(e["player"])
            if pid is not None:
                tr[num].append(pid)
                drafted.add(pid)
        return tr, 0

    def build_immediate():
        tr, drafted = preload_keepers()
        snipes = 0
        for e in log:
            num = e.get("team_num") or num_by_name.get(e["team"])
            if e["team"] == HART:
                pid = immediate_pick(tr, drafted, e)
            else:
                real_pid = resolve(e["player"])
                if real_pid is not None and real_pid not in drafted:
                    pid = real_pid
                else:
                    snipes += 1
                    pid = fallback_adp(num, drafted, tr)
            if pid is not None:
                tr[num].append(pid)
                drafted.add(pid)
        return tr, snipes

    def score(tr):
        team_players = {num: [] for num in name_by_num}
        for num, pids in tr.items():
            for pid in pids:
                row = player_lookup.get(pid)
                if row is None:
                    fb = full_board[full_board["player_id"] == pid]
                    row = fb.iloc[0] if not fb.empty else None
                if row is not None:
                    team_players[num].append(row)
        cap = {}
        orig = sd.score_roto_dict

        def patched(ts, team_sds=None, **kw):
            cap["stats"] = ts
            cap["sds"] = team_sds
            return orig(ts, team_sds=team_sds, **kw)

        sd.score_roto_dict = patched
        results, _ = sd._score_roto(team_players, config, full_board, board, scarcity_order)
        sd.score_roto_dict = orig
        hart = next(r for r in results if r["team"] == HART)
        rank = next(i + 1 for i, r in enumerate(results) if r["team"] == HART)
        rng = np.random.default_rng(12345)
        wp, mr = _mc_win_rank(cap["stats"], cap["sds"], HART, N_MC, rng)
        return hart, rank, wp, mr

    real_tr, _ = build_real()
    imm_tr, snipes = build_immediate()
    real_hart, real_rank, real_wp, real_mr = score(real_tr)
    imm_hart, imm_rank, imm_wp, imm_mr = score(imm_tr)

    hart_num = num_by_name[HART]
    out = []
    out.append(
        "COUNTERFACTUAL: immediate drafting Hart's real seat vs the actual three_closers/var draft"
    )
    out.append(
        f"(real opponents; {snipes} opponent picks were sniped by immediate -> ADP fallback)\n"
    )

    def roster_names(tr):
        return [name_of(p) for p in tr[hart_num]]

    real_names = roster_names(real_tr)
    imm_names = roster_names(imm_tr)
    out.append(f"{'ACTUAL (three_closers/var)':<34}{'IMMEDIATE (counterfactual)':<34}")
    out.append("-" * 68)
    for i in range(max(len(real_names), len(imm_names))):
        a = real_names[i] if i < len(real_names) else ""
        b = imm_names[i] if i < len(imm_names) else ""
        out.append(f"{a:<34}{b:<34}")
    out.append("")

    out.append(f"{'metric':<10}{'ACTUAL':>12}{'IMMEDIATE':>12}")
    out.append("-" * 34)
    for c in CATS:
        out.append(f"{c:<10}{real_hart.get(c, 0):>12.2f}{imm_hart.get(c, 0):>12.2f}")
    out.append(f"{'roto tot':<10}{real_hart['tot']:>12.1f}{imm_hart['tot']:>12.1f}")
    out.append(f"{'rank':<10}{real_rank:>12}{imm_rank:>12}")
    out.append(f"{'MC win%':<10}{real_wp * 100:>11.1f}%{imm_wp * 100:>11.1f}%")
    out.append(f"{'MC rank':<10}{real_mr:>12.2f}{imm_mr:>12.2f}")

    text = "\n".join(out)
    print(text)
    (PROJECT_ROOT / "counterfactual_real_draft.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
