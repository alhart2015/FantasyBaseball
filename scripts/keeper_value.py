"""Rank keeper candidates by forecast SGP value above replacement.

Chains onto `keeper_forecast`: that script turns 2026 into a projected 2027/2028 5x5
line, this one prices that line in standings gain points and nets it against what a
roster slot is otherwise worth.

    SGP      counting stats over the league's per-place denominators; AVG/ERA/WHIP
             priced marginally against a replacement rate, so a rate only counts for
             as much playing time as it is delivered over.
    VAR      SGP minus the replacement level -- the last player who would actually be
             rostered (10 teams x 11 hitters / 9 pitchers).

**2027 is the validated number. 2028 is an extrapolation** -- the persistence fit is a
ONE-year transition, so its drift term is one year of playing-time attrition. Applying
it to a two-year horizon understates the decay, which makes 2028 run optimistic on
volume. It is shown because keepers are held indefinitely and the trajectory matters,
not because it carries the same weight as 2027.

Replacement here is POSITION-BLIND (one hitter level, one pitcher level). A catcher is
therefore undervalued and a corner outfielder overvalued relative to a position-aware
level. That is a known gap, not an oversight.

Usage:
    python scripts/keeper_value.py                  # full board
    python scripts/keeper_value.py --team "Hart of the Order"
    python scripts/keeper_value.py --top 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Player names carry accents (Jesus Luzardo, Cristopher Sanchez) and Windows stdout
# defaults to cp1252, which mangles them. Reconfigure per the repo ASCII rule's
# stated exception for data-sourced names.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from keeper_forecast import _names, fetch_blend, forecast_pool, to_counting

from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import (
    REPLACEMENT_AVG,
    REPLACEMENT_ERA,
    REPLACEMENT_WHIP,
    calculate_counting_sgp,
    calculate_hitting_rate_sgp,
    calculate_pitching_rate_sgp,
)
from fantasy_baseball.utils.constants import DEFAULT_TEAM_AB, DEFAULT_TEAM_IP, Category
from fantasy_baseball.utils.name_utils import normalize_name

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
# 10 teams x the starting slots in config/league.yaml (11 hitters, 9 pitchers). The
# replacement level is the last man rostered, so this is the depth that defines it.
ROSTERED = {"hitter": 110, "pitcher": 90}


def sgp_frame(counting: pd.DataFrame, kind: str, denoms: dict[Category, float]) -> pd.Series:
    """Total SGP per player from a forecast counting line."""
    if kind == "hitter":
        total = sum(
            calculate_counting_sgp(counting[cat.value], denoms[cat])
            for cat in (Category.R, Category.HR, Category.RBI, Category.SB)
        )
        # `ab` is not in the counting frame; recover it from PA at the league AB/PA
        # ratio the forecast itself produced, so the marginal-hits term is scaled by
        # the at-bats this player is actually forecast to take.
        ab = counting["PA"] * 0.895
        return total + calculate_hitting_rate_sgp(
            player_avg=counting["AVG"],
            player_ab=ab,  # type: ignore[arg-type]
            replacement_avg=REPLACEMENT_AVG,
            sgp_denominator=denoms[Category.AVG],
            team_ab=DEFAULT_TEAM_AB,
        )
    total = sum(
        calculate_counting_sgp(counting[cat.value if cat is not Category.K else "K"], denoms[cat])
        for cat in (Category.W, Category.SV, Category.K)
    )
    for rate, repl, cat in (
        ("ERA", REPLACEMENT_ERA, Category.ERA),
        ("WHIP", REPLACEMENT_WHIP, Category.WHIP),
    ):
        total = total + calculate_pitching_rate_sgp(
            player_rate=counting[rate],
            player_ip=counting["IP"],
            replacement_rate=repl,
            sgp_denominator=denoms[cat],
            team_ip=DEFAULT_TEAM_IP,
            innings_divisor=1.0,
        )
    return total


def fetch_rosters(my_team: str) -> dict[tuple[str, str], str]:
    """(normalized_name, player_type) -> owning team, from the live roster blobs.

    Keyed on the NAME, not the mlbam id, because roster blobs do not carry one -- see
    issue #284. Two different players sharing a normalized name and type therefore
    collapse onto one owner. That residual is irreducible here; the fix is populating
    mlbam_id at Yahoo ingest, not more matching logic. Unmatched counts are printed so
    a silent join failure cannot pass for "nobody owns him".
    """
    os.environ["RENDER"] = "true"
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    from fantasy_baseball.data.cache_keys import CacheKey
    from fantasy_baseball.data.kv_store import build_explicit_upstash_kv

    kv = build_explicit_upstash_kv()
    owners: dict[tuple[str, str], str] = {}
    for key, mine in ((CacheKey.OPP_ROSTERS, None), (CacheKey.ROSTER, my_team)):
        raw = kv.get(f"cache:{key.value}")
        if raw is None:
            continue
        blob = json.loads(raw) if isinstance(raw, str) else raw
        data = blob.get("_data", blob)
        # opp_rosters is {team: [players]}; roster is a BARE LIST -- my own team.
        groups = data.items() if isinstance(data, dict) else [(mine, data)]
        for team, players in groups:
            if not isinstance(players, list):
                continue
            for p in players:
                if not isinstance(p, dict) or not p.get("name"):
                    continue
                owners[(normalize_name(p["name"]), str(p.get("player_type", "")))] = str(team)
    return owners


def _by_team(board: pd.DataFrame, args: argparse.Namespace) -> int:
    """Each team's best keeper candidates, weakest team first.

    Sorted on the top-`per_team` total as asked, but the KEEP total (the top
    `keep_slots`, which is what a team may actually retain) is printed too -- they can
    disagree, and only the second one is a decision.

    Unrostered players are excluded: a free agent is not anybody's keeper candidate,
    and leaving them in would invent a team called NaN.
    """
    owned = board[board["team"].notna()]
    rows = []
    for team, group in owned.groupby("team"):
        best = group.nlargest(args.per_team, "var_total")
        rows.append(
            {
                "team": str(team),
                "keep": best["var_total"].head(args.keep_slots).sum(),
                "shown": best["var_total"].sum(),
                "players": best,
            }
        )
    rows.sort(key=lambda r: r["shown"])

    print("")
    print("=" * 100)
    print(
        f"KEEPER CANDIDATES BY TEAM -- weakest first "
        f"(KEEP = best {args.keep_slots}, SHOWN = best {args.per_team})"
    )
    print(f"{'=' * 100}")
    for row in rows:
        print("")
        print(
            f"{row['team']:<26} KEEP {row['keep']:>6.2f}   top-{args.per_team} {row['shown']:>6.2f}"
        )
        for slot, (_, r) in enumerate(row["players"].iterrows(), start=1):
            tag = "H" if r["kind"] == "hitter" else "P"
            mark = "*" if slot <= args.keep_slots else " "
            print(
                f"   {mark}{slot}. {str(r['name'])[:24]:<26} {r['var_total']:>6.2f} {tag}"
                f"   27 VAR {r['var_2027']:>5.2f}   {r['vol_2027']:>5.0f} {'PA' if tag == 'H' else 'IP'}"
            )
    print("")
    print(f"  * = inside the {args.keep_slots} keeper slots.")
    print("  Values are expectations including missed time; see the notes on the main board.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-pa", type=float, default=300)
    parser.add_argument("--min-ip", type=float, default=50)
    parser.add_argument("--min-next-pa", type=float, default=250)
    parser.add_argument("--min-next-ip", type=float, default=50)
    parser.add_argument("--no-aging", action="store_true")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--team", help="restrict the board to one fantasy team")
    parser.add_argument("--stats", action="store_true", help="show the 2027 roto line")
    parser.add_argument(
        "--per-600",
        action="store_true",
        help="rescale the roto line to a common 600 PA (talent, availability removed)",
    )
    parser.add_argument("--kind", choices=("hitter", "pitcher"), help="restrict to one pool")
    parser.add_argument(
        "--by-team", action="store_true", help="each team's best N, weakest team first"
    )
    parser.add_argument("--per-team", type=int, default=5, help="rows per team for --by-team")
    parser.add_argument("--keep-slots", type=int, default=3, help="keepers allowed per team")
    parser.add_argument("--my-team", default="Hart of the Order", help="label for cache:roster")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    denoms = get_sgp_denominators(config.get("sgp_denominators"))
    print(
        "SGP denominators:",
        {k.value: v for k, v in sorted(denoms.items(), key=lambda x: x[0].value)},
    )

    payload = fetch_blend()
    owners = fetch_rosters(args.my_team)
    print(f"  roster entries loaded: {len(owners)} (joined on name+type; no mlbam, #284)")

    rows = []
    for kind in ("hitter", "pitcher"):
        per_year, volume = {}, {}
        vol_col = "PA" if kind == "hitter" else "IP"
        for year in (2027, 2028):
            args.year = year
            counting = to_counting(forecast_pool(kind, year, payload, args), kind)
            per_year[year] = sgp_frame(counting, kind, denoms)
            volume[year] = counting[vol_col]
            if year == 2027:
                # Keep the volume column: the roto view prints PA/IP as its first
                # category, so dropping it here would strand that header.
                roto = counting
        frame = pd.DataFrame(
            {
                "sgp_2027": per_year[2027],
                "sgp_2028": per_year[2028],
                # Projected workload behind the 2027 line: PA for hitters, IP for
                # pitchers. Shown because it is the heaviest single input, and a
                # rating is not readable without knowing how much playing time it
                # assumes.
                "vol_2027": volume[2027],
                "vol_2028": volume[2028],
            }
        )
        # Replacement = the last player who would actually be rostered, per year.
        for year in (2027, 2028):
            col = f"sgp_{year}"
            level = frame[col].nlargest(ROSTERED[kind]).iloc[-1]
            frame[f"var_{year}"] = frame[col] - level
            print(f"  {kind} {year} replacement level: {level:6.2f} SGP")
        # The 2027 scored line itself, so a rating can be read against the stats that
        # produced it. Hitter and pitcher category names differ, so the concat below
        # leaves the other pool's columns NaN -- which is correct, not a gap.
        for cat in roto.columns:
            frame[cat] = roto[cat]
        frame["kind"] = kind
        frame["name"] = _names(payload, kind).reindex(frame.index)
        keys = frame["name"].fillna("").map(normalize_name)
        frame["team"] = [owners.get((k, kind)) for k in keys]
        rows.append(frame)

    board = pd.concat(rows)
    board["var_total"] = board["var_2027"] + board["var_2028"]
    board = board.sort_values("var_total", ascending=False)

    if args.by_team:
        return _by_team(board, args)

    view = board
    if args.kind:
        view = view[view["kind"] == args.kind]
    if args.team:
        view = board[board["team"].fillna("").str.lower() == args.team.lower()]
    view = view.head(args.top)

    title = f"KEEPER VALUE -- {args.team}" if args.team else "KEEPER VALUE -- full board"
    print(f"\n{'=' * 108}\n{title}  (VAR = SGP above the last rostered player)\n{'=' * 108}")
    if args.stats:
        cats = (
            ["PA", "R", "HR", "RBI", "SB", "AVG"]
            if args.kind == "hitter"
            else ["IP", "W", "SV", "K", "ERA", "WHIP"]
        )
        print(
            f"{'#':>3} {'name':<24}{'VALUE':>7}"
            + "".join(f"{c:>8}" for c in cats)
            + f"{'27 VAR':>9}  team"
        )
        print("-" * 108)
        for rank, (_, r) in enumerate(view.iterrows(), start=1):
            # Per-600 puts every hitter on the same workload, so the line reads as talent
            # with availability divided out. The RATING is never rescaled -- a keeper slot
            # bears the missed time, so its value has to.
            scale = 600.0 / r["PA"] if (args.per_600 and args.kind == "hitter" and r["PA"]) else 1.0
            cells = "".join(
                f"{r[c]:>8.3f}"
                if c in {"AVG", "ERA", "WHIP"}
                else f"{r[c] * (1.0 if c in {'PA', 'IP'} else scale):>8.1f}"
                for c in cats
            )
            team = "" if pd.isna(r["team"]) else str(r["team"])[:18]
            print(
                f"{rank:>3} {str(r['name'])[:23]:<24}{r['var_total']:>7.2f}{cells}"
                f"{r['var_2027']:>9.2f}  {team}"
            )
        return 0

    # VALUE leads: it is the number the board exists to produce, and everything right
    # of it is the working that got there.
    print(
        f"{'#':>3} {'name':<24} {'VALUE':>7} {'':<2} {'27 PA/IP':>9} {'28 PA/IP':>9} "
        f"{'27 SGP':>8} {'27 VAR':>8} {'28 VAR':>8}  team"
    )
    print("-" * 108)
    for rank, (_, r) in enumerate(view.iterrows(), start=1):
        tag = "H" if r["kind"] == "hitter" else "P"
        team = "" if pd.isna(r["team"]) else str(r["team"])[:20]
        print(
            f"{rank:>3} {str(r['name'])[:23]:<24} {r['var_total']:>7.2f} {tag:<2} "
            f"{r['vol_2027']:>9.0f} {r['vol_2028']:>9.0f} {r['sgp_2027']:>8.2f} "
            f"{r['var_2027']:>8.2f} {r['var_2028']:>8.2f}  {team}"
        )
    print("\n  PA/IP and the counting stats are EXPECTATIONS over every outcome, including")
    print("  the chance of missing time -- NOT a healthy-season line. That is why they sit")
    print("  below ZiPS, which projects a nominal workload. Verified on 2025: unconditional")
    print("  bias +1% PA, +0% R, -0% RBI, -7% HR. Use --per-600 to divide availability out.")
    print("  2027 is the validated horizon; 2028 extrapolates a one-year drift term.")
    print("  Replacement is position-blind.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
