"""Rank keeper candidates by forecast SGP value above replacement.

Chains onto `keeper_forecast`: that script turns 2026 into a projected 2027/2028 5x5
line, this one prices that line in standings gain points and nets it against what a
roster slot is otherwise worth.

    SGP      counting stats over the league's per-place denominators; AVG/ERA/WHIP
             priced marginally against a replacement rate, so a rate only counts for
             as much playing time as it is delivered over.
    VAR      SGP minus the replacement level -- the last player who would actually be
             rostered (10 teams x 11 hitters / 9 pitchers).

**2027 is the validated number. 2028 is an extrapolation.** Volume iterates the
playing-time curve twice, with age advanced each step, which is the right shape. The
RATES do not: the persistence fit is a ONE-year transition and its share is applied
once, so 2028's rates are effectively 2027's and a 2026 breakout is over-credited
against a steady veteran. It is shown because keepers are held indefinitely and the
trajectory matters, not because it carries the same weight as 2027.

Replacement is POSITION-AWARE, reusing `sgp/replacement.py`'s empirical waiver floors
rather than one level per pool -- the same floors the draft board nets against, so a
keeper and a draft pick are priced on one scale. Scarcity is real and large: the
catcher floor is 7.70 SGP against an outfielder's 9.96, so a catcher earns a 2.3 SGP
credit for being hard to replace. Pitchers route to an SP or RP floor by projected
innings (9.29 vs 7.42), which is what stops a 52-inning closer being measured against
a 160-inning starter. `sgp/var.py` credits a multi-eligible player at his scarcest
slot and falls back to UTIL -- defined as the HIGHEST hitter floor, so an uncovered
player gets no scarcity credit rather than an invented one.

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

from fantasy_baseball.keepers.positions import load_positions
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import (
    REPLACEMENT_AVG,
    REPLACEMENT_ERA,
    REPLACEMENT_WHIP,
    calculate_counting_sgp,
    calculate_hitting_rate_sgp,
    calculate_pitching_rate_sgp,
)
from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import DEFAULT_TEAM_AB, DEFAULT_TEAM_IP, Category
from fantasy_baseball.utils.name_utils import normalize_name

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"


# Empirical per-position waiver floors, shared with the draft board. Pure in the
# league's denominators, so this is a module-level constant rather than a per-run fit.
LEVELS: dict[str, float] = {}


def _eligibility(names: pd.Series, kind: str) -> pd.Series:
    """Eligible slots per player, for netting against the right replacement level.

    Pitchers are a single "P" slot in this league, so `calculate_var` routes them to
    the SP or RP floor by projected innings. A hitter the position cache does not cover
    falls back to UTIL, which `position_aware_replacement_levels` defines as the
    HIGHEST hitter floor -- the conservative choice, since it assumes no scarcity
    credit rather than inventing one.
    """
    if kind == "pitcher":
        return pd.Series([["P"]] * len(names), index=names.index)
    positions = load_positions()
    return names.fillna("").map(lambda n: positions.get(normalize_name(n)) or ["UTIL"])


def sgp_frame(counting: pd.DataFrame, kind: str, denoms: dict[Category, float]) -> pd.Series:
    """Total SGP per player from a forecast counting line."""
    if kind == "hitter":
        total = sum(
            calculate_counting_sgp(counting[cat.value], denoms[cat])
            for cat in (Category.R, Category.HR, Category.RBI, Category.SB)
        )
        # The forecast's own per-player at-bats, so the marginal-hits term is scaled by
        # the AB this hitter is actually projected to take. This used to be a hardcoded
        # league 0.895 -- ab_pa really spans 0.80-0.96, which is worth ~0.34 SGP of
        # var_total to a high-walk bat, against adjacent candidates often under 0.5 apart.
        return total + calculate_hitting_rate_sgp(
            player_avg=counting["AVG"],
            player_ab=counting["AB"],  # type: ignore[arg-type]
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
    # Name the restriction: under --kind the KEEP total counts one pool only, and a
    # reader comparing teams would otherwise take it for the whole retainable roster.
    scope = f" -- {args.kind.upper()}S ONLY" if args.kind else ""
    print(
        f"KEEPER CANDIDATES BY TEAM{scope} -- weakest first "
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
    parser.add_argument(
        "--no-exit-rows", action="store_true", help="ablate the survivorship correction"
    )
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
        "--sort",
        choices=("var", "sgp"),
        default="var",
        help="rank by value above replacement (default) or by raw SGP",
    )
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

    LEVELS.update(position_aware_replacement_levels(denoms))
    print("replacement floors:", {k: round(v, 2) for k, v in sorted(LEVELS.items())})

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
        # VAR against the POSITION-AWARE empirical floors, not one constant per pool.
        # `sgp/replacement.py` already owns these (per hitter position, plus an SP/RP
        # split), and `calculate_var` already credits a multi-eligible player at his
        # scarcest slot and falls back to UTIL for DH-only bats. Reusing both keeps one
        # definition of replacement across the draft board and this one.
        frame["name"] = _names(payload, kind).reindex(frame.index)
        eligible = _eligibility(frame["name"], kind)
        for year in (2027, 2028):
            col = f"sgp_{year}"
            var_input = pd.DataFrame(
                {
                    "total_sgp": frame[col],
                    "positions": eligible,
                    # Routes a pitcher to the SP or RP floor. Full-season projected IP,
                    # so the role cannot flip on a partial line.
                    "ip": frame[f"vol_{year}"] if kind == "pitcher" else 0.0,
                }
            )
            frame[f"var_{year}"] = var_input.apply(
                lambda r: calculate_var(r, LEVELS), axis=1
            ).astype(float)
        used = sorted({p for slots in eligible for p in slots})
        print(
            f"  {kind} floors used: "
            + ", ".join(f"{p} {LEVELS[p]:.2f}" for p in used if p in LEVELS)
        )
        # The 2027 scored line itself, so a rating can be read against the stats that
        # produced it. Hitter and pitcher category names differ, so the concat below
        # leaves the other pool's columns NaN -- which is correct, not a gap.
        for cat in roto.columns:
            frame[cat] = roto[cat]
        frame["kind"] = kind
        keys = frame["name"].fillna("").map(normalize_name)
        frame["team"] = [owners.get((k, kind)) for k in keys]
        rows.append(frame)

    board = pd.concat(rows)
    board["var_total"] = board["var_2027"] + board["var_2028"]
    # Raw SGP, for comparison. Replacement is POSITION-aware, so this is not a single
    # constant per pool and it absolutely can reorder within one: the hitter floors span
    # 2.27 SGP (C 7.70 to OF 9.96), which is what moves catchers 14-20 places between
    # the two rankings. It WAS one constant per pool before c88f6167; that is where the
    # old "cannot reorder within a pool" claim came from, and it is no longer true.
    board["sgp_total"] = board["sgp_2027"] + board["sgp_2028"]
    board = board.sort_values("var_total" if args.sort == "var" else "sgp_total", ascending=False)
    # Ranked here, before any slicing: the index is NOT unique (a two-way player carries
    # one mlbam id in both pools), so reindexing a slice back against the board raises.
    # Ranks stay float because an unscoreable player is NaN, and int() on that raises.
    board["_var_rank"] = board["var_total"].rank(ascending=False, method="min")
    board["_sgp_rank"] = board["sgp_total"].rank(ascending=False, method="min")

    view = board
    if args.kind:
        view = view[view["kind"] == args.kind]
    # --by-team must see the SAME filtered view, not the raw board: returning before
    # these lines silently dropped a co-supplied --kind or --team and counted the
    # wrong pool toward each team's keeper total.
    if args.by_team:
        if args.team:
            view = view[view["team"].fillna("").str.lower() == args.team.lower()]
        return _by_team(view, args)
    if args.team:
        # Narrows `view`, NOT `board`: restarting from the board here silently threw
        # away a --kind filter applied on the line above.
        view = view[view["team"].fillna("").str.lower() == args.team.lower()]
    view = view.head(args.top)

    title = f"KEEPER VALUE -- {args.team}" if args.team else "KEEPER VALUE -- full board"
    print(f"\n{'=' * 108}\n{title}  (VAR = SGP above the last rostered player)\n{'=' * 108}")
    if args.stats:
        # Hitters and pitchers score different categories, so one header cannot serve a
        # mixed board -- without --kind this used to fall through to the pitcher columns
        # and render every hitter as six NaNs.
        if args.kind is None:
            parser.error("--stats needs --kind: the two pools score different categories")
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
        f"{'#':>3} {'name':<24} {'VALUE':>7} {'rawSGP':>8} {'':<2} {'27 PA/IP':>9} "
        f"{'VARrk':>6} {'SGPrk':>6} {'move':>6}  team"
    )
    print("-" * 108)
    for rank, (_, r) in enumerate(view.iterrows(), start=1):
        tag = "H" if r["kind"] == "hitter" else "P"
        team = "" if pd.isna(r["team"]) else str(r["team"])[:20]
        vr, sr = r["_var_rank"], r["_sgp_rank"]
        ranks = (
            f"{'-':>6}{'-':>6}{'-':>6}"
            if pd.isna(vr) or pd.isna(sr)
            else f"{int(vr):>6}{int(sr):>6}{int(sr) - int(vr):>+6}"
        )
        print(
            f"{rank:>3} {str(r['name'])[:23]:<24} {r['var_total']:>7.2f} "
            f"{r['sgp_total']:>8.2f} {tag:<2} {r['vol_2027']:>9.0f} {ranks}  {team}"
        )
    print("\n  PA/IP and the counting stats are EXPECTATIONS over every outcome, including")
    print("  the chance of missing time -- NOT a healthy-season line. That is why they sit")
    print("  below ZiPS, which projects a nominal workload. Verified on 2025: unconditional")
    print("  bias PA +1%, R +0%, RBI -0%, HR -7%, SB +14%. Use --per-600 to divide out.")
    print("  2027 is the validated horizon. 2028 iterates the VOLUME curve twice but")
    print("  applies the one-year RATE share once, so its rates are effectively 2027's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
