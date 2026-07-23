"""Suggest keeper-mutual consolidation trades. Keeper values are offline (local
board); rosters + the 2026 guardrail inputs are live Upstash. See
docs/superpowers/specs/2026-07-23-keeper-trade-generator-design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import keeper_value as kv_script  # scripts/keeper_value.py: build_results, BASE_YEAR

from fantasy_baseball.analysis.keeper_trades import (
    GuardrailResult,
    RosterPlayer,
    build_consolidation_proposal,
    generate_consolidation_trades,
)
from fantasy_baseball.analysis.keeper_value import discounted_total
from fantasy_baseball.config import load_config
from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
from fantasy_baseball.models.player import Player
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.rankings import fg_key, rank_key
from fantasy_baseball.trades.eval_inputs import load_trade_eval_context
from fantasy_baseball.trades.multi_trade import evaluate_multi_trade
from fantasy_baseball.web.season_data import unwrap_cache_envelope

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "league.yaml"


def _ros_value(player: Player, denoms) -> float:
    if player.rest_of_season is None:
        return float("-inf")
    return calculate_player_sgp(player.rest_of_season, denoms)


def to_roster_players(players, keeper_by_key) -> list[RosterPlayer]:
    """Attach keeper_value fg_id-primary, name fallback (spec + CLAUDE.md 'never key
    on bare names'). keeper_by_key holds BOTH fg-based board ids and rank_key(name)."""
    out = []
    for p in players:
        val = None
        if p.fg_id:
            val = keeper_by_key.get(fg_key(str(p.fg_id), p.player_type))
        if val is None:
            val = keeper_by_key.get(rank_key(p.name, p.player_type))
        out.append(
            RosterPlayer(
                player_id=p.player_key,
                name=p.name,
                keeper_value=val if val is not None else 0.0,
            )
        )
    return out


def _cache(kv, key):
    raw = kv.get(redis_key(key))
    if raw is None:
        raise RuntimeError(f"Upstash missing {key}; run a refresh first.")
    obj = json.loads(raw) if isinstance(raw, str) else raw
    return unwrap_cache_envelope(obj)


def _waiver_keys_by_ros(waiver_pool, denoms):
    """Waiver player_keys, best ROS value first. Invariant across a run, so the
    caller sorts the (large) pool once and slices per guardrail call instead of
    re-sorting on every package evaluation."""
    ordered = sorted(waiver_pool.values(), key=lambda p: _ros_value(p, denoms), reverse=True)
    return [p.player_key for p in ordered]


def _opp_drop_keys(opp_players, denoms, n):
    """Bottom-n opp drop keys by ROS value. `opp_players` already excludes the
    received stud. Pure given Players + denoms."""
    if n == 0:  # size-1 packages (the common case) drop nothing -- skip the sort
        return []
    drops = sorted(opp_players, key=lambda p: _ros_value(p, denoms))
    return [p.player_key for p in drops[:n]]


def make_guardrail(ctx, denoms, threshold):
    """Injected guardrail: resolves the opponent from `receive`, builds a
    roster-legal proposal, and returns evaluate_multi_trade's verdict for Hart."""
    waiver_keys = _waiver_keys_by_ros(ctx.waiver_pool, denoms)  # sort the pool once, not per call

    def _owning_team(receive_key: str) -> str:
        for team, players in ctx.opp_rosters.items():
            if any(p.player_key == receive_key for p in players):
                return team
        raise KeyError(f"{receive_key} is not on any opponent roster")

    def guardrail(give, receive):
        package_keys = [p.player_id for p in give]  # RosterPlayer.player_id == player_key
        n = max(0, len(package_keys) - 1)
        opp_name = _owning_team(receive.player_id)
        opp_players = [p for p in ctx.opp_rosters[opp_name] if p.player_key != receive.player_id]
        my_adds = waiver_keys[:n]
        opp_drops = _opp_drop_keys(opp_players, denoms, n)
        proposal = build_consolidation_proposal(
            opponent=opp_name,
            hart_players=ctx.hart_roster,
            package_keys=package_keys,
            receive_key=receive.player_id,
            my_adds_keys=my_adds,
            opp_drop_keys=opp_drops,
        )
        r = evaluate_multi_trade(
            proposal=proposal,
            hart_name=ctx.hart_name,
            hart_roster=ctx.hart_roster,
            opp_rosters=ctx.opp_rosters,
            waiver_pool=ctx.waiver_pool,
            projected_standings=ctx.projected_standings,
            team_sds=ctx.team_sds,
            roster_slots=ctx.roster_slots,
            fraction_remaining=ctx.fraction_remaining,
        )
        return GuardrailResult(
            legal=r.legal,
            delta_total=r.delta_total,
            ok=r.legal and r.delta_total >= -threshold,
        )

    return guardrail


def render(suggestions) -> str:
    if not suggestions:
        return "No keeper-mutual consolidation trades found."
    groups: dict[tuple[str, str], list] = {}
    for s in suggestions:
        key = (s.target_team, s.acquire.player_id)
        groups.setdefault(key, []).append(s)
    lines = ["Keeper-mutual consolidation trades (ranked by your keeper gain)", ""]
    for key, rows in groups.items():
        a = rows[0].acquire
        lines.append(f"ACQUIRE {a.name} (kv {a.keeper_value:.1f}) from {key[0]}")
        for s in rows:
            give = " + ".join(f"{p.name} ({p.keeper_value:.1f})" for p in s.give)
            g = s.guardrail
            my_keep = ", ".join(p.name for p in s.my_keepers_after)
            their_keep = ", ".join(p.name for p in s.their_keepers_after)
            lines.append(f"  give [{s.variant}]: {give}")
            lines.append(
                f"    YOU:  top-3 {s.my_top3_before:.1f} -> {s.my_top3_after:.1f} "
                f"(+{s.my_gain:.1f})   keep: {my_keep}"
            )
            lines.append(
                f"    THEM: top-3 {s.their_top3_before:.1f} -> {s.their_top3_after:.1f} "
                f"(+{s.their_gain:.1f})   keep: {their_keep}"
            )
            lines.append(
                f"    2026: roto delta {g.delta_total:+.1f}  guardrail {'OK' if g.ok else 'FAIL'}"
            )
        lines.append("")
    return "\n".join(lines)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Suggest keeper-mutual consolidation trades.")
    ap.add_argument("--discount", type=float, default=0.80)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--guardrail-threshold", type=float, default=2.0)
    ap.add_argument("--max-give", type=int, default=3)
    ap.add_argument("--no-sweetener", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parse_args(argv)
    config = load_config(CONFIG_PATH)
    denoms = get_sgp_denominators(config.sgp_overrides)

    # 1. offline keeper values -> {board player_id and rank_key(name): keeper_value}
    results, _ = kv_script.build_results(base_year=kv_script.BASE_YEAR, horizon=args.horizon)
    keeper_by_key: dict[str, float] = {}
    for r in results:
        ptype = r.player_id.rsplit("::", 1)[-1]
        val = discounted_total(r.per_year_var, kv_script.BASE_YEAR, args.discount, args.horizon)
        for k in (r.player_id, rank_key(r.name, ptype)):
            if k not in keeper_by_key or val > keeper_by_key[k]:
                keeper_by_key[k] = val

    # 2. live Upstash: assemble the eval context + rosters
    kv = build_explicit_upstash_kv()
    ctx = load_trade_eval_context(
        hart_name=config.team_name,
        roster_raw=_cache(kv, CacheKey.ROSTER),
        opp_rosters_raw=_cache(kv, CacheKey.OPP_ROSTERS),
        proj_cache=_cache(kv, CacheKey.PROJECTIONS),
        ros_cache=_cache(kv, CacheKey.ROS_PROJECTIONS),
        roster_slots=config.roster_slots,
    )

    # 3. annotate rosters with keeper_value
    rosters = {config.team_name: to_roster_players(ctx.hart_roster, keeper_by_key)}
    for team, players in ctx.opp_rosters.items():
        rosters[team] = to_roster_players(players, keeper_by_key)

    # 4. generate
    guardrail = make_guardrail(ctx, denoms, args.guardrail_threshold)
    suggestions = generate_consolidation_trades(
        config.team_name,
        rosters,
        guardrail,
        max_give=args.max_give,
        sweetener=not args.no_sweetener,
    )
    print(render(suggestions))


if __name__ == "__main__":
    main()
