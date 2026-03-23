"""Analyze 2025 draft: projections vs actual stats, injury impact, over/underperformance."""
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest_2025 import DRAFT_2025, ACTUAL, ALL_CATS, INVERSE


def normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def load_projections():
    """Load and blend Steamer + ZiPS 2025 preseason projections."""
    proj_dir = PROJECT_ROOT / "data" / "projections"

    sh = pd.read_csv(proj_dir / "fangraphs-leaderboard-projections-steamer-hitters-2025.csv")
    zh = pd.read_csv(proj_dir / "fangraphs-leaderboard-projections-zips-hitters-2025.csv")
    sp = pd.read_csv(proj_dir / "fangraphs-leaderboard-projections-steamer-pitchers-2025.csv")
    zp = pd.read_csv(proj_dir / "fangraphs-leaderboard-projections-zips-pitchers-2025.csv")

    # Deduplicate by MLBAMID (keep highest WAR)
    sh = sh.sort_values("WAR", ascending=False).drop_duplicates(subset="MLBAMID", keep="first")
    zh = zh.sort_values("WAR", ascending=False).drop_duplicates(subset="MLBAMID", keep="first")
    sp = sp.sort_values("WAR", ascending=False).drop_duplicates(subset="MLBAMID", keep="first")
    zp = zp.sort_values("WAR", ascending=False).drop_duplicates(subset="MLBAMID", keep="first")

    # Blend hitters by MLBAMID
    sh = sh.set_index("MLBAMID")
    zh = zh.set_index("MLBAMID")
    common_h = sh.index.intersection(zh.index)
    num_cols_h = ["G", "PA", "AB", "H", "HR", "R", "RBI", "SB", "WAR"]
    blend_h = pd.DataFrame(index=common_h)
    for col in num_cols_h:
        if col in sh.columns and col in zh.columns:
            blend_h[col] = (sh.loc[common_h, col].astype(float) + zh.loc[common_h, col].astype(float)) / 2
    blend_h["AVG"] = blend_h["H"] / blend_h["AB"]
    blend_h["Name"] = sh.loc[common_h, "Name"]
    blend_h["NameASCII"] = sh.loc[common_h, "NameASCII"]
    blend_h["player_type"] = "hitter"

    # Also add steamer-only hitters not in ZiPS
    steamer_only_h = sh.index.difference(zh.index)
    for idx in steamer_only_h:
        row = {"Name": sh.loc[idx, "Name"], "NameASCII": sh.loc[idx, "NameASCII"], "player_type": "hitter"}
        for col in num_cols_h:
            if col in sh.columns:
                row[col] = float(sh.loc[idx, col])
        if "AB" in row and row["AB"] > 0:
            row["AVG"] = float(sh.loc[idx, "H"]) / float(sh.loc[idx, "AB"])
        blend_h.loc[idx] = row
    zips_only_h = zh.index.difference(sh.index)
    for idx in zips_only_h:
        row = {"Name": zh.loc[idx, "Name"], "NameASCII": zh.loc[idx, "NameASCII"], "player_type": "hitter"}
        for col in num_cols_h:
            if col in zh.columns:
                row[col] = float(zh.loc[idx, col])
        if "AB" in row and row["AB"] > 0:
            row["AVG"] = float(zh.loc[idx, "H"]) / float(zh.loc[idx, "AB"])
        blend_h.loc[idx] = row

    # Blend pitchers by MLBAMID
    sp = sp.set_index("MLBAMID")
    zp = zp.set_index("MLBAMID")
    common_p = sp.index.intersection(zp.index)
    num_cols_p = ["G", "GS", "IP", "W", "SV", "SO", "ER", "BB", "H", "WAR"]
    blend_p = pd.DataFrame(index=common_p)
    for col in num_cols_p:
        if col in sp.columns and col in zp.columns:
            blend_p[col] = (sp.loc[common_p, col].astype(float) + zp.loc[common_p, col].astype(float)) / 2
    if "ERA" in sp.columns and "ERA" in zp.columns:
        blend_p["ERA"] = (sp.loc[common_p, "ERA"].astype(float) + zp.loc[common_p, "ERA"].astype(float)) / 2
    if "WHIP" in sp.columns and "WHIP" in zp.columns:
        blend_p["WHIP"] = (sp.loc[common_p, "WHIP"].astype(float) + zp.loc[common_p, "WHIP"].astype(float)) / 2
    blend_p["Name"] = sp.loc[common_p, "Name"]
    blend_p["NameASCII"] = sp.loc[common_p, "NameASCII"]
    blend_p["player_type"] = "pitcher"

    steamer_only_p = sp.index.difference(zp.index)
    for idx in steamer_only_p:
        row = {"Name": sp.loc[idx, "Name"], "NameASCII": sp.loc[idx, "NameASCII"], "player_type": "pitcher"}
        for col in num_cols_p + ["ERA", "WHIP"]:
            if col in sp.columns:
                row[col] = float(sp.loc[idx, col])
        blend_p.loc[idx] = row
    zips_only_p = zp.index.difference(sp.index)
    for idx in zips_only_p:
        row = {"Name": zp.loc[idx, "Name"], "NameASCII": zp.loc[idx, "NameASCII"], "player_type": "pitcher"}
        for col in num_cols_p + ["ERA", "WHIP"]:
            if col in zp.columns:
                row[col] = float(zp.loc[idx, col])
        blend_p.loc[idx] = row

    return blend_h, blend_p


def load_actuals():
    """Load actual 2025 stats."""
    stats_dir = PROJECT_ROOT / "data" / "stats"
    ah = pd.read_csv(stats_dir / "hitters-2025.csv")
    ap = pd.read_csv(stats_dir / "pitchers-2025.csv")
    ah = ah.sort_values("WAR", ascending=False).drop_duplicates(subset="MLBAMID", keep="first")
    ap = ap.sort_values("WAR", ascending=False).drop_duplicates(subset="MLBAMID", keep="first")
    ah = ah.set_index("MLBAMID")
    ap = ap.set_index("MLBAMID")
    # Compute K for pitchers: K/9 * IP / 9
    ap["K"] = ap["K/9"] * ap["IP"] / 9
    return ah, ap


def match_player(name, proj_h, proj_p, act_h, act_p):
    """Try to match a drafted player to projections and actuals by MLBAMID."""
    name_norm = normalize_name(name)

    proj_row = None
    proj_type = None
    proj_mlbamid = None
    for df, ptype in [(proj_h, "hitter"), (proj_p, "pitcher")]:
        for idx, row in df.iterrows():
            rname = normalize_name(str(row.get("Name", row.get("NameASCII", ""))))
            if rname == name_norm:
                proj_row = row
                proj_type = ptype
                proj_mlbamid = idx
                break
            clean = name_norm.replace(" jr.", "").replace(" jr", "").replace(" sr.", "").replace(" ii", "").strip()
            rclean = rname.replace(" jr.", "").replace(" jr", "").replace(" sr.", "").replace(" ii", "").strip()
            if clean == rclean and len(clean) > 4:
                proj_row = row
                proj_type = ptype
                proj_mlbamid = idx
                break
        if proj_row is not None:
            break

    if proj_row is None:
        return None

    act_row = None
    act_df = act_h if proj_type == "hitter" else act_p
    if proj_mlbamid in act_df.index:
        act_row = act_df.loc[proj_mlbamid]
    else:
        for idx, row in act_df.iterrows():
            rname = normalize_name(str(row.get("Name", row.get("NameASCII", ""))))
            if rname == name_norm:
                act_row = row
                break
            clean = name_norm.replace(" jr.", "").replace(" jr", "").replace(" sr.", "").replace(" ii", "").strip()
            rclean = rname.replace(" jr.", "").replace(" jr", "").replace(" sr.", "").replace(" ii", "").strip()
            if clean == rclean and len(clean) > 4:
                act_row = row
                break

    return {
        "name": name,
        "type": proj_type,
        "proj": proj_row,
        "actual": act_row,
        "mlbamid": proj_mlbamid,
    }


def main():
    print("Loading projections and actuals...")
    proj_h, proj_p = load_projections()
    act_h, act_p = load_actuals()

    # Match all drafted players
    team_players = {}
    unmatched_proj = []
    unmatched_actual = []

    for rnd, player, team in DRAFT_2025:
        if team not in team_players:
            team_players[team] = []
        result = match_player(player, proj_h, proj_p, act_h, act_p)
        if result is None:
            unmatched_proj.append((rnd, player, team))
        elif result["actual"] is None:
            unmatched_actual.append((rnd, player, team))
            team_players[team].append(result)
        else:
            team_players[team].append(result)

    print(f"\nUnmatched projections: {len(unmatched_proj)}")
    for r, p, t in unmatched_proj:
        print(f"  R{r}: {p} ({t})")
    print(f"No 2025 stats (injured/minors/DNP): {len(unmatched_actual)}")
    for r, p, t in unmatched_actual:
        print(f"  R{r}: {p} ({t})")

    # =========================================================================
    # INJURY / AVAILABILITY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 100)
    print("INJURY / AVAILABILITY ANALYSIS")
    print("=" * 100)

    team_avail = {}
    for team, players in team_players.items():
        h_proj_pa, h_act_pa = 0, 0
        p_proj_ip, p_act_ip = 0, 0
        injury_list = []

        for p in players:
            if p["actual"] is None:
                if p["type"] == "hitter":
                    proj_pa = float(p["proj"].get("PA", 0) or 0)
                    h_proj_pa += proj_pa
                    injury_list.append((p["name"], p["type"], proj_pa, 0, 0.0, "PA"))
                else:
                    proj_ip = float(p["proj"].get("IP", 0) or 0)
                    p_proj_ip += proj_ip
                    injury_list.append((p["name"], p["type"], proj_ip, 0, 0.0, "IP"))
            else:
                if p["type"] == "hitter":
                    proj_pa = float(p["proj"].get("PA", 0) or 0)
                    act_pa = float(p["actual"].get("PA", 0) or 0)
                    h_proj_pa += proj_pa
                    h_act_pa += act_pa
                    pct = act_pa / proj_pa if proj_pa > 0 else 0
                    if pct < 0.70:
                        injury_list.append((p["name"], p["type"], proj_pa, act_pa, pct, "PA"))
                else:
                    proj_ip = float(p["proj"].get("IP", 0) or 0)
                    act_ip = float(p["actual"].get("IP", 0) or 0)
                    p_proj_ip += proj_ip
                    p_act_ip += act_ip
                    pct = act_ip / proj_ip if proj_ip > 0 else 0
                    if pct < 0.70:
                        injury_list.append((p["name"], p["type"], proj_ip, act_ip, pct, "IP"))

        h_avail = h_act_pa / h_proj_pa if h_proj_pa > 0 else 0
        p_avail = p_act_ip / p_proj_ip if p_proj_ip > 0 else 0
        overall = (h_act_pa + p_act_ip) / (h_proj_pa + p_proj_ip) if (h_proj_pa + p_proj_ip) > 0 else 0

        team_avail[team] = {
            "h_avail": h_avail, "p_avail": p_avail, "overall": overall,
            "h_proj_pa": h_proj_pa, "h_act_pa": h_act_pa,
            "p_proj_ip": p_proj_ip, "p_act_ip": p_act_ip,
            "injuries": injury_list,
        }

    print(f"\n{'Rk':>2} {'Team':<32} {'Hitter%':>8} {'Pitcher%':>9} {'Overall%':>9} {'Lost PA':>8} {'Lost IP':>8}")
    print("-" * 90)
    sorted_avail = sorted(team_avail.items(), key=lambda x: x[1]["overall"], reverse=True)
    for i, (team, a) in enumerate(sorted_avail, 1):
        lost_pa = a["h_proj_pa"] - a["h_act_pa"]
        lost_ip = a["p_proj_ip"] - a["p_act_ip"]
        marker = " <<<" if team == "Hart of the Order" else ""
        print(f"{i:>2} {team:<32} {a['h_avail']:>7.1%} {a['p_avail']:>8.1%} {a['overall']:>8.1%} {lost_pa:>8.0f} {lost_ip:>8.0f}{marker}")

    # Each team's major injury losses
    for team in sorted(team_avail.keys()):
        injuries = team_avail[team]["injuries"]
        if injuries:
            marker = " <<<" if team == "Hart of the Order" else ""
            print(f"\n  {team}{marker} - Major losses (<70%):")
            for name, ptype, proj, act, pct, unit in sorted(injuries, key=lambda x: x[4]):
                print(f"    {name:<25} {ptype:<8} Proj {proj:>5.0f}{unit}  Actual {act:>5.0f}{unit}  ({pct:.0%})")

    # =========================================================================
    # PLAYER-LEVEL WAR OVER/UNDERPERFORMANCE
    # =========================================================================
    all_player_perf = []
    team_perf = {}

    for team, players in team_players.items():
        team_surplus_war = 0
        team_proj_war = 0
        player_details = []

        for p in players:
            proj_war = float(p["proj"].get("WAR", 0) or 0)
            act_war = float(p["actual"].get("WAR", 0) or 0) if p["actual"] is not None else 0
            surplus = act_war - proj_war
            team_proj_war += proj_war
            team_surplus_war += surplus

            detail = {
                "name": p["name"], "type": p["type"],
                "proj_war": proj_war, "act_war": act_war, "surplus": surplus,
            }
            if p["actual"] is None:
                detail["reason"] = "DNP"

            if p["type"] == "hitter" and p["actual"] is not None:
                detail["proj_pa"] = float(p["proj"].get("PA", 0) or 0)
                detail["act_pa"] = float(p["actual"].get("PA", 0) or 0)
                detail["proj_hr"] = float(p["proj"].get("HR", 0) or 0)
                detail["act_hr"] = float(p["actual"].get("HR", 0) or 0)
                detail["proj_avg"] = float(p["proj"].get("AVG", 0) or 0)
                detail["act_avg"] = float(p["actual"].get("AVG", 0) or 0)
                detail["proj_r"] = float(p["proj"].get("R", 0) or 0)
                detail["act_r"] = float(p["actual"].get("R", 0) or 0)
                detail["proj_rbi"] = float(p["proj"].get("RBI", 0) or 0)
                detail["act_rbi"] = float(p["actual"].get("RBI", 0) or 0)
                detail["proj_sb"] = float(p["proj"].get("SB", 0) or 0)
                detail["act_sb"] = float(p["actual"].get("SB", 0) or 0)
            elif p["type"] == "pitcher" and p["actual"] is not None:
                detail["proj_ip"] = float(p["proj"].get("IP", 0) or 0)
                detail["act_ip"] = float(p["actual"].get("IP", 0) or 0)
                detail["proj_era"] = float(p["proj"].get("ERA", 0) or 0)
                detail["act_era"] = float(p["actual"].get("ERA", 0) or 0)
                detail["proj_w"] = float(p["proj"].get("W", 0) or 0)
                detail["act_w"] = float(p["actual"].get("W", 0) or 0)
                detail["proj_sv"] = float(p["proj"].get("SV", 0) or 0)
                detail["act_sv"] = float(p["actual"].get("SV", 0) or 0)
                detail["proj_k"] = float(p["proj"].get("SO", 0) or 0)
                detail["act_k"] = float(p["actual"].get("K", 0) or 0)

            player_details.append(detail)
            all_player_perf.append({
                "name": p["name"], "team": team, "type": p["type"],
                "proj_war": proj_war, "act_war": act_war, "surplus": surplus,
            })

        team_perf[team] = {
            "proj_war": team_proj_war,
            "surplus_war": team_surplus_war,
            "act_war": team_proj_war + team_surplus_war,
            "players": player_details,
        }

    # =========================================================================
    # TEAM WAR SURPLUS RANKING
    # =========================================================================
    print("\n" + "=" * 100)
    print("TEAM PERFORMANCE vs PROJECTION (WAR surplus = actual - projected)")
    print("=" * 100)
    print(f"\n{'Rk':>2} {'Team':<32} {'Proj WAR':>9} {'Act WAR':>8} {'Surplus':>8} {'Roto Pts':>9} {'Roto Rk':>8}")
    print("-" * 90)
    sorted_perf = sorted(team_perf.items(), key=lambda x: x[1]["surplus_war"], reverse=True)
    for i, (team, perf) in enumerate(sorted_perf, 1):
        a = ACTUAL.get(team, {})
        marker = " <<<" if team == "Hart of the Order" else ""
        print(f"{i:>2} {team:<32} {perf['proj_war']:>9.1f} {perf['act_war']:>8.1f} {perf['surplus_war']:>+8.1f} {a.get('total', '?'):>9} {a.get('rank', '?'):>8}{marker}")

    # =========================================================================
    # TOP OVERPERFORMERS / UNDERPERFORMERS
    # =========================================================================
    sorted_players = sorted(all_player_perf, key=lambda x: x["surplus"], reverse=True)

    print("\n" + "=" * 100)
    print("TOP 15 OVERPERFORMERS (all drafted players, by WAR surplus)")
    print("=" * 100)
    print(f"\n{'':>3} {'Player':<25} {'Team':<28} {'Type':<8} {'Proj':>5} {'Act':>5} {'Surplus':>8}")
    print("-" * 85)
    for i, p in enumerate(sorted_players[:15], 1):
        print(f"{i:>3} {p['name']:<25} {p['team']:<28} {p['type']:<8} {p['proj_war']:>5.1f} {p['act_war']:>5.1f} {p['surplus']:>+8.1f}")

    print("\n" + "=" * 100)
    print("TOP 15 UNDERPERFORMERS (all drafted players, by WAR surplus)")
    print("=" * 100)
    print(f"\n{'':>3} {'Player':<25} {'Team':<28} {'Type':<8} {'Proj':>5} {'Act':>5} {'Surplus':>8}")
    print("-" * 85)
    for i, p in enumerate(sorted_players[-15:][::-1], 1):
        print(f"{i:>3} {p['name']:<25} {p['team']:<28} {p['type']:<8} {p['proj_war']:>5.1f} {p['act_war']:>5.1f} {p['surplus']:>+8.1f}")

    # =========================================================================
    # HART OF THE ORDER DETAILED BREAKDOWN
    # =========================================================================
    print("\n" + "=" * 100)
    print("HART OF THE ORDER - DETAILED PLAYER BREAKDOWN")
    print("=" * 100)

    hart_players = team_perf["Hart of the Order"]["players"]
    hart_hitters = [p for p in hart_players if p["type"] == "hitter"]
    hart_pitchers = [p for p in hart_players if p["type"] == "pitcher"]

    print("\nHitters:")
    print(f"  {'Player':<22} {'ProjPA':>7} {'ActPA':>6} {'Avail':>6} {'ProjHR':>7} {'ActHR':>6} {'ProjAVG':>8} {'ActAVG':>7} {'ProjWAR':>8} {'ActWAR':>7} {'Surplus':>8}")
    print("  " + "-" * 95)
    for p in sorted(hart_hitters, key=lambda x: x["surplus"]):
        if p.get("reason") == "DNP":
            proj_pa = float(p.get("proj_pa", 0))
            print(f"  {p['name']:<22} {'---':>7} {'DNP':>6} {'0%':>6} {'---':>7} {'---':>6} {'---':>8} {'---':>7} {p['proj_war']:>8.1f} {0.0:>7.1f} {p['surplus']:>+8.1f}")
            continue
        proj_pa = p.get("proj_pa", 0)
        act_pa = p.get("act_pa", 0)
        avail = f"{act_pa / proj_pa:.0%}" if proj_pa > 0 else "?"
        print(f"  {p['name']:<22} {proj_pa:>7.0f} {act_pa:>6.0f} {avail:>6} {p.get('proj_hr',0):>7.0f} {p.get('act_hr',0):>6.0f} {p.get('proj_avg',0):>8.3f} {p.get('act_avg',0):>7.3f} {p['proj_war']:>8.1f} {p['act_war']:>7.1f} {p['surplus']:>+8.1f}")

    print("\nPitchers:")
    print(f"  {'Player':<22} {'ProjIP':>7} {'ActIP':>6} {'Avail':>6} {'ProjERA':>8} {'ActERA':>7} {'ProjW':>6} {'ActW':>5} {'ProjSV':>7} {'ActSV':>6} {'ProjWAR':>8} {'ActWAR':>7} {'Surplus':>8}")
    print("  " + "-" * 110)
    for p in sorted(hart_pitchers, key=lambda x: x["surplus"]):
        if p.get("reason") == "DNP":
            print(f"  {p['name']:<22} {'---':>7} {'DNP':>6} {'0%':>6} {'---':>8} {'---':>7} {'---':>6} {'---':>5} {'---':>7} {'---':>6} {p['proj_war']:>8.1f} {0.0:>7.1f} {p['surplus']:>+8.1f}")
            continue
        proj_ip = p.get("proj_ip", 0)
        act_ip = p.get("act_ip", 0)
        avail = f"{act_ip / proj_ip:.0%}" if proj_ip > 0 else "?"
        print(f"  {p['name']:<22} {proj_ip:>7.0f} {act_ip:>6.0f} {avail:>6} {p.get('proj_era',0):>8.2f} {p.get('act_era',0):>7.2f} {p.get('proj_w',0):>6.0f} {p.get('act_w',0):>5.0f} {p.get('proj_sv',0):>7.0f} {p.get('act_sv',0):>6.0f} {p['proj_war']:>8.1f} {p['act_war']:>7.1f} {p['surplus']:>+8.1f}")

    # =========================================================================
    # PROJECTION ACCURACY BY TEAM
    # =========================================================================
    print("\n" + "=" * 100)
    print("PROJECTION ACCURACY: who got closest to their projected stats?")
    print("(compares draft-day projected team roto totals vs actual season)")
    print("=" * 100)

    team_proj_stats = {}
    for team, players in team_players.items():
        proj_r, proj_hr, proj_rbi, proj_sb = 0, 0, 0, 0
        proj_h_total, proj_ab_total = 0, 0
        proj_w, proj_sv, proj_k = 0, 0, 0
        proj_er, proj_ip = 0, 0

        for p in players:
            if p["type"] == "hitter":
                proj_r += float(p["proj"].get("R", 0) or 0)
                proj_hr += float(p["proj"].get("HR", 0) or 0)
                proj_rbi += float(p["proj"].get("RBI", 0) or 0)
                proj_sb += float(p["proj"].get("SB", 0) or 0)
                proj_h_total += float(p["proj"].get("H", 0) or 0)
                proj_ab_total += float(p["proj"].get("AB", 0) or 0)
            else:
                proj_w += float(p["proj"].get("W", 0) or 0)
                proj_sv += float(p["proj"].get("SV", 0) or 0)
                proj_k += float(p["proj"].get("SO", 0) or 0)
                proj_er += float(p["proj"].get("ER", 0) or 0)
                proj_ip += float(p["proj"].get("IP", 0) or 0)

        proj_avg = proj_h_total / proj_ab_total if proj_ab_total > 0 else 0
        proj_era = proj_er * 9 / proj_ip if proj_ip > 0 else 0

        team_proj_stats[team] = {
            "R": proj_r, "HR": proj_hr, "RBI": proj_rbi, "SB": proj_sb,
            "AVG": proj_avg, "W": proj_w, "K": proj_k, "SV": proj_sv,
            "ERA": proj_era,
        }

    team_accuracy = {}
    for team in ACTUAL:
        a = ACTUAL[team]
        p = team_proj_stats.get(team, {})
        diffs = []
        cat_diffs = {}
        for cat in ["R", "HR", "RBI", "SB", "W", "K", "SV"]:
            pv = p.get(cat, 0)
            av = a.get(cat, 0)
            if pv > 0:
                d = (av - pv) / pv
                diffs.append(abs(d))
                cat_diffs[cat] = d
        for cat in ["AVG", "ERA"]:
            pv = p.get(cat, 0)
            av = a.get(cat, 0)
            if pv > 0:
                d = (av - pv) / pv
                diffs.append(abs(d))
                cat_diffs[cat] = d
        team_accuracy[team] = {
            "mean_abs_pct_err": np.mean(diffs) if diffs else 0,
            "cat_diffs": cat_diffs,
        }

    print(f"\n{'Rk':>2} {'Team':<32} {'Mean|Err%|':>11} {'Biggest Over':>28} {'Biggest Under':>28}")
    print("-" * 105)
    sorted_acc = sorted(team_accuracy.items(), key=lambda x: x[1]["mean_abs_pct_err"])
    for i, (team, acc) in enumerate(sorted_acc, 1):
        cd = acc["cat_diffs"]
        if cd:
            best_cat = max(cd, key=lambda c: cd[c])
            worst_cat = min(cd, key=lambda c: cd[c])
            best_str = f"{best_cat} {cd[best_cat]:+.0%}" if cd[best_cat] > 0 else f"{best_cat} {cd[best_cat]:+.0%}"
            worst_str = f"{worst_cat} {cd[worst_cat]:+.0%}"
        else:
            best_str = worst_str = "?"
        marker = " <<<" if team == "Hart of the Order" else ""
        print(f"{i:>2} {team:<32} {acc['mean_abs_pct_err']:>10.1%} {best_str:>28} {worst_str:>28}{marker}")

    # Hart category-by-category
    print(f"\nHART OF THE ORDER - Projected vs Actual by category:")
    print(f"  {'Cat':>5} {'Projected':>10} {'Actual':>10} {'Diff%':>8} {'Assessment':<30}")
    print("  " + "-" * 65)
    hart_proj = team_proj_stats["Hart of the Order"]
    hart_act = ACTUAL["Hart of the Order"]
    for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA"]:
        pv = hart_proj[cat]
        av = hart_act[cat]
        diff = (av - pv) / pv * 100 if pv != 0 else 0
        if cat in ("AVG", "ERA"):
            fmt_p, fmt_a = f"{pv:.3f}", f"{av:.3f}"
        else:
            fmt_p, fmt_a = f"{pv:.0f}", f"{av:.0f}"

        # Assessment considers that ERA lower is better
        effective_diff = diff
        if cat == "ERA":
            effective_diff = -diff  # positive means improvement for ERA

        if abs(effective_diff) < 5:
            assessment = "on target"
        elif effective_diff > 20:
            assessment = "MASSIVE outperformance"
        elif effective_diff > 10:
            assessment = "strong outperformance"
        elif effective_diff > 5:
            assessment = "mild outperformance"
        elif effective_diff < -20:
            assessment = "MASSIVE shortfall"
        elif effective_diff < -10:
            assessment = "significant shortfall"
        elif effective_diff < -5:
            assessment = "mild shortfall"
        else:
            assessment = ""
        print(f"  {cat:>5} {fmt_p:>10} {fmt_a:>10} {diff:>+7.1f}% {assessment:<30}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    hart_avail = team_avail["Hart of the Order"]
    league_avg_avail = np.mean([v["overall"] for v in team_avail.values()])
    hart_rank_avail = sum(1 for v in team_avail.values() if v["overall"] > hart_avail["overall"]) + 1

    hart_p = team_perf["Hart of the Order"]
    hart_rank_perf = sum(1 for v in team_perf.values() if v["surplus_war"] > hart_p["surplus_war"]) + 1

    hart_a = team_accuracy["Hart of the Order"]
    hart_rank_acc = sum(1 for v in team_accuracy.values() if v["mean_abs_pct_err"] < hart_a["mean_abs_pct_err"]) + 1

    print(f"\n  Availability (injury impact):")
    print(f"    Hart: {hart_avail['overall']:.1%} (rank {hart_rank_avail}/10, league avg {league_avg_avail:.1%})")
    print(f"    Lost {hart_avail['h_proj_pa'] - hart_avail['h_act_pa']:.0f} hitter PA, {hart_avail['p_proj_ip'] - hart_avail['p_act_ip']:.0f} pitcher IP vs projections")
    healthiest = sorted_avail[0]
    sickest = sorted_avail[-1]
    print(f"    Healthiest: {healthiest[0]} ({healthiest[1]['overall']:.1%})")
    print(f"    Most injured: {sickest[0]} ({sickest[1]['overall']:.1%})")

    print(f"\n  WAR performance vs projection:")
    print(f"    Hart: {hart_p['surplus_war']:+.1f} WAR surplus (rank {hart_rank_perf}/10)")
    best_perf = sorted_perf[0]
    worst_perf = sorted_perf[-1]
    print(f"    Most overperformed: {best_perf[0]} ({best_perf[1]['surplus_war']:+.1f})")
    print(f"    Most underperformed: {worst_perf[0]} ({worst_perf[1]['surplus_war']:+.1f})")

    print(f"\n  Projection accuracy (draft-day totals vs actual roto):")
    print(f"    Hart: {hart_a['mean_abs_pct_err']:.1%} mean abs error (rank {hart_rank_acc}/10)")
    best_acc = sorted_acc[0]
    worst_acc = sorted_acc[-1]
    print(f"    Most predictable: {best_acc[0]} ({best_acc[1]['mean_abs_pct_err']:.1%})")
    print(f"    Least predictable: {worst_acc[0]} ({worst_acc[1]['mean_abs_pct_err']:.1%})")

    print(f"\n  Actual finish: {ACTUAL['Hart of the Order']['rank']}rd, {ACTUAL['Hart of the Order']['total']} roto pts")


if __name__ == "__main__":
    main()
