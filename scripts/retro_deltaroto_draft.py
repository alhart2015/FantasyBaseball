"""Retrospective: deltaRoto recommendations at each of my real-draft picks,
OLD replacement (demand-based) vs NEW (empirical position-aware waiver lines).

Replays the real 2026 draft (data/drafts/draft_2026-03-24_195809.json). At each
"Hart of the Order" turn, reconstructs the state (keepers + picks-so-far), then
runs the ERoto/deltaRoto recommender twice -- once with the existing
find_replacement_players baseline, once with REPLACEMENT_BY_POSITION empirical
lines (sub-project #2) -- and compares both top-3s to the player actually taken.

Read-only. Throwaway analysis script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json

import pandas as pd
import yaml

import fantasy_baseball.draft.eroto_recs as eroto
from fantasy_baseball.draft.board import rebuild_board
from fantasy_baseball.draft.eroto_recs import rank_candidates
from fantasy_baseball.draft.recs_integration import (
    _build_replacements,
    _league_teams,
    build_adp_table,
    build_projected_standings,
    build_team_rosters,
    drafted_ids,
    load_board_rows,
    partition_available,
    rows_to_players,
)
from fantasy_baseball.draft.state import StateKey
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.scoring import build_team_sds
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.constants import REPLACEMENT_BY_POSITION, STARTER_IP_THRESHOLD
from fantasy_baseball.utils.name_utils import normalize_name

DRAFT_FILE = Path("data/drafts/draft_2026-03-24_195809.json")
BOARD_PATH = Path("data/_retro_board.json")
USER_TEAM = "Hart of the Order"
POOL_CAP = 200
HIT_POS = ("C", "1B", "2B", "3B", "SS", "OF")


def _empirical_replacements() -> dict[str, Player]:
    """Build replacement Players from REPLACEMENT_BY_POSITION waiver lines."""
    reps: dict[str, Player] = {}
    for pos in HIT_POS:
        ln = REPLACEMENT_BY_POSITION[pos]
        avg = ln["h"] / ln["ab"] if ln["ab"] else 0.0
        row = {
            "name": f"repl {pos}",
            "player_id": f"repl::{pos}",
            "player_type": "hitter",
            "positions": [pos],
            "r": ln["r"], "hr": ln["hr"], "rbi": ln["rbi"], "sb": ln["sb"],
            "h": ln["h"], "ab": ln["ab"], "avg": avg,
        }
        p = Player.from_dict(row)
        p.rest_of_season.compute_sgp()  # so _best_by_type ranks padding
        reps[pos] = p
    for pos in ("SP", "RP"):
        ln = REPLACEMENT_BY_POSITION[pos]
        ip = ln["ip"]
        row = {
            "name": f"repl {pos}",
            "player_id": f"repl::{pos}",
            "player_type": "pitcher",
            "positions": [pos],
            "w": ln["w"], "k": ln["k"], "sv": ln["sv"], "ip": ip,
            "er": ln["er"], "bb": ln["bb"], "h_allowed": ln["h_allowed"],
            "era": 9 * ln["er"] / ip if ip else 0.0,
            "whip": (ln["bb"] + ln["h_allowed"]) / ip if ip else 0.0,
        }
        p = Player.from_dict(row)
        p.rest_of_season.compute_sgp()
        reps[pos] = p
    return reps


def _role_aware_pick(candidate: Player, replacements):
    """Swap target for a candidate: SP/RP by IP for pitchers, primary pos for hitters."""
    if candidate.player_type == PlayerType.PITCHER:
        ros = candidate.rest_of_season
        ip = float(getattr(ros, "ip", 0) or 0) if ros else 0.0
        role = "SP" if ip >= STARTER_IP_THRESHOLD else "RP"
        return replacements.get(role) or next(iter(replacements.values()))
    primary = str(candidate.positions[0]) if candidate.positions else ""
    return replacements.get(primary) or replacements.get("OF") or next(iter(replacements.values()))


def _top3(state, candidates_all, board_by_id, teams, roster_slots, replacements, adp_table, actual_id):
    rosters = build_team_rosters(state, board_by_id, teams, roster_slots, replacements)
    standings = build_projected_standings(rosters)
    sds = build_team_sds(rosters, sd_scale=1.0)
    drafted = drafted_ids(state)
    cand = partition_available(candidates_all, drafted)[:POOL_CAP]
    if actual_id not in {c.yahoo_id for c in cand}:
        cand += [c for c in candidates_all if c.yahoo_id == actual_id]
    rows = rank_candidates(
        candidates=cand,
        replacements=replacements,
        team_name=USER_TEAM,
        projected_standings=standings,
        team_sds=sds,
        picks_until_next_turn=0,
        adp_table=adp_table,
    )
    rank = next((i + 1 for i, r in enumerate(rows) if r.player_id == actual_id), None)
    return rows, rank


def main() -> None:
    league = yaml.safe_load(Path("config/league.yaml").read_text())
    n = rebuild_board(Path("config/league.yaml"), BOARD_PATH)
    print(f"[rebuilt board: {n} players]\n")

    draft = json.loads(DRAFT_FILE.read_text())
    log = draft["draft_log"]

    rows = load_board_rows(BOARD_PATH)
    players = rows_to_players(rows)
    board_by_id = {p.yahoo_id: p for p in players if p.yahoo_id}
    teams = _league_teams(league)
    roster_slots = league.get("roster_slots") or {}
    pool = pd.DataFrame(rows)
    adp_table = build_adp_table(pool)

    by_norm = {}
    for r in rows:
        by_norm.setdefault(normalize_name(r.get("name", "")), r)
    keeper_entries = []
    for k in league.get("keepers") or []:
        row = by_norm.get(normalize_name(k["name"]))
        if row:
            keeper_entries.append(
                {"player_id": row["player_id"], "team": k["team"], "position": row.get("best_position", "")}
            )

    pick_entries = [{"player_id": e["player_id"], "team": e["team"], "position": ""} for e in log]
    user_picks = [(i, e) for i, e in enumerate(log) if e["team"] == USER_TEAM]

    old_reps = _build_replacements(pool, roster_slots, len(teams))
    new_reps = _empirical_replacements()
    orig_pick = eroto._pick_replacement

    print(f"{USER_TEAM}: {len(user_picks)} picks, {len(keeper_entries)} keepers matched\n" + "=" * 78)

    for idx, entry in user_picks:
        state = {StateKey.KEEPERS.value: keeper_entries, StateKey.PICKS.value: pick_entries[:idx]}
        actual_id = entry["player_id"]

        eroto._pick_replacement = orig_pick  # OLD: demand-based, position-primary swap
        old_rows, old_rank = _top3(state, players, board_by_id, teams, roster_slots, old_reps, adp_table, actual_id)
        eroto._pick_replacement = _role_aware_pick  # NEW: empirical, role-aware swap
        new_rows, new_rank = _top3(state, players, board_by_id, teams, roster_slots, new_reps, adp_table, actual_id)

        print(f"\nRound {entry['round']:2d} (pick {entry['pick']:3d})  YOU: {entry['player']}")
        print(f"   your pick rank:  OLD #{old_rank}   NEW #{new_rank}")
        print(f"   {'OLD (demand-based)':<40s}{'NEW (empirical waiver)':<40s}")
        for i in range(3):
            o = old_rows[i]
            nw = new_rows[i]
            om = " *" if o.player_id == actual_id else ""
            nm = " *" if nw.player_id == actual_id else ""
            ol = f"{i+1}. {o.name} ({o.immediate_delta:+.2f}){om}"
            nl = f"{i+1}. {nw.name} ({nw.immediate_delta:+.2f}){nm}"
            print(f"   {ol:<40s}{nl:<40s}")

    eroto._pick_replacement = orig_pick


if __name__ == "__main__":
    main()
