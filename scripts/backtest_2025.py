"""Backtest: compare 2025 draft simulation to actual 2025 results."""
import sys
import unicodedata
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.draft.board import build_draft_board
from fantasy_baseball.utils.name_utils import normalize_name

PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"


def ascii_name(s):
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_only = nfkd.encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"\s+", " ", ascii_only.lower().strip())


# Actual 2025 final standings
ACTUAL = {
    "Hello Peanuts!": {"total": 79, "rank": 1, "R": 996, "HR": 273, "RBI": 960, "SB": 161, "AVG": .259, "W": 94, "SV": 87, "K": 1419, "ERA": 3.16, "WHIP": 1.10},
    "Tortured Baseball Department": {"total": 70, "rank": 2, "R": 985, "HR": 270, "RBI": 936, "SB": 201, "AVG": .267, "W": 97, "SV": 38, "K": 1434, "ERA": 3.84, "WHIP": 1.14},
    "Hart of the Order": {"total": 69, "rank": 3, "R": 974, "HR": 305, "RBI": 961, "SB": 193, "AVG": .248, "W": 84, "SV": 109, "K": 1282, "ERA": 3.56, "WHIP": 1.18},
    "Springfield Isotopes": {"total": 61.5, "rank": 4, "R": 1035, "HR": 294, "RBI": 988, "SB": 182, "AVG": .255, "W": 80, "SV": 71, "K": 1348, "ERA": 4.04, "WHIP": 1.21},
    "Spacemen": {"total": 57, "rank": 5, "R": 1005, "HR": 317, "RBI": 935, "SB": 156, "AVG": .254, "W": 83, "SV": 73, "K": 1276, "ERA": 3.58, "WHIP": 1.19},
    "Boston Estrellas": {"total": 55, "rank": 6, "R": 881, "HR": 291, "RBI": 918, "SB": 153, "AVG": .241, "W": 87, "SV": 74, "K": 1306, "ERA": 3.10, "WHIP": 1.12},
    "Jon's Underdogs": {"total": 54, "rank": 7, "R": 949, "HR": 242, "RBI": 932, "SB": 167, "AVG": .255, "W": 76, "SV": 75, "K": 1334, "ERA": 3.48, "WHIP": 1.14},
    "Work in Progress": {"total": 52.5, "rank": 8, "R": 944, "HR": 298, "RBI": 947, "SB": 143, "AVG": .260, "W": 79, "SV": 65, "K": 1180, "ERA": 3.52, "WHIP": 1.13},
    "SkeleThor": {"total": 31.5, "rank": 9, "R": 953, "HR": 241, "RBI": 842, "SB": 165, "AVG": .259, "W": 79, "SV": 27, "K": 1235, "ERA": 3.92, "WHIP": 1.25},
    "Crews Control": {"total": 20.5, "rank": 10, "R": 847, "HR": 266, "RBI": 844, "SB": 133, "AVG": .247, "W": 80, "SV": 36, "K": 1120, "ERA": 3.94, "WHIP": 1.21},
}

# 2025 draft (ASCII-safe names)
DRAFT_2025 = [
    (1, "Shohei Ohtani", "Work in Progress"),
    (1, "Bobby Witt Jr.", "Jon's Underdogs"),
    (1, "Aaron Judge", "Spacemen"),
    (1, "Elly De La Cruz", "Crews Control"),
    (1, "Corbin Carroll", "Hello Peanuts!"),
    (1, "Juan Soto", "Hart of the Order"),
    (1, "Jose Ramirez", "Tortured Baseball Department"),
    (1, "Paul Skenes", "Boston Estrellas"),
    (1, "Kyle Tucker", "SkeleThor"),
    (1, "Francisco Lindor", "Springfield Isotopes"),
    (2, "Fernando Tatis Jr.", "Springfield Isotopes"),
    (2, "Gunnar Henderson", "SkeleThor"),
    (2, "Tarik Skubal", "Boston Estrellas"),
    (2, "Vladimir Guerrero Jr.", "Tortured Baseball Department"),
    (2, "Julio Rodriguez", "Hart of the Order"),
    (2, "Jackson Chourio", "Hello Peanuts!"),
    (2, "Yordan Alvarez", "Crews Control"),
    (2, "Mookie Betts", "Spacemen"),
    (2, "Zack Wheeler", "Jon's Underdogs"),
    (2, "Trea Turner", "Work in Progress"),
    (3, "Logan Gilbert", "Work in Progress"),
    (3, "Jarren Duran", "Jon's Underdogs"),
    (3, "Bryce Harper", "Spacemen"),
    (3, "Jazz Chisholm Jr.", "Crews Control"),
    (3, "Garrett Crochet", "Hello Peanuts!"),
    (3, "William Contreras", "Hart of the Order"),
    (3, "Matt Olson", "Tortured Baseball Department"),
    (3, "Jackson Merrill", "Boston Estrellas"),
    (3, "Freddie Freeman", "SkeleThor"),
    (3, "Rafael Devers", "Springfield Isotopes"),
    (4, "Ketel Marte", "Springfield Isotopes"),
    (4, "Corbin Burnes", "SkeleThor"),
    (4, "Manny Machado", "Boston Estrellas"),
    (4, "Cole Ragans", "Tortured Baseball Department"),
    (4, "Chris Sale", "Hart of the Order"),
    (4, "Oneil Cruz", "Hello Peanuts!"),
    (4, "Austin Riley", "Crews Control"),
    (4, "Dylan Cease", "Spacemen"),
    (4, "Pete Alonso", "Jon's Underdogs"),
    (4, "Wyatt Langford", "Work in Progress"),
    (5, "Jose Altuve", "Work in Progress"),
    (5, "Blake Snell", "Jon's Underdogs"),
    (5, "Jacob deGrom", "Spacemen"),
    (5, "Corey Seager", "Crews Control"),
    (5, "Tyler Glasnow", "Hello Peanuts!"),
    (5, "Emmanuel Clase", "Hart of the Order"),
    (5, "Michael Harris II", "Tortured Baseball Department"),
    (5, "Yoshinobu Yamamoto", "Boston Estrellas"),
    (5, "Ronald Acuna Jr.", "SkeleThor"),
    (5, "Brent Rooker", "Springfield Isotopes"),
    (6, "Teoscar Hernandez", "Springfield Isotopes"),
    (6, "Ozzie Albies", "SkeleThor"),
    (6, "Kyle Schwarber", "Boston Estrellas"),
    (6, "Devin Williams", "Tortured Baseball Department"),
    (6, "CJ Abrams", "Hart of the Order"),
    (6, "James Wood", "Hello Peanuts!"),
    (6, "Framber Valdez", "Crews Control"),
    (6, "Brenton Doyle", "Spacemen"),
    (6, "Josh Hader", "Jon's Underdogs"),
    (6, "Marcell Ozuna", "Work in Progress"),
    (7, "Edwin Diaz", "Work in Progress"),
    (7, "Michael King", "Jon's Underdogs"),
    (7, "Raisel Iglesias", "Spacemen"),
    (7, "Mason Miller", "Crews Control"),
    (7, "Spencer Schwellenbach", "Hello Peanuts!"),
    (7, "Pablo Lopez", "Hart of the Order"),
    (7, "Shota Imanaga", "Tortured Baseball Department"),
    (7, "Ryan Helsley", "Boston Estrellas"),
    (7, "Logan Webb", "SkeleThor"),
    (7, "Aaron Nola", "Springfield Isotopes"),
    (8, "Tanner Bibee", "Springfield Isotopes"),
    (8, "Lawrence Butler", "SkeleThor"),
    (8, "Marcus Semien", "Boston Estrellas"),
    (8, "Bo Bichette", "Tortured Baseball Department"),
    (8, "Junior Caminero", "Hart of the Order"),
    (8, "Ryan Walker", "Hello Peanuts!"),
    (8, "Alex Bregman", "Crews Control"),
    (8, "Max Fried", "Spacemen"),
    (8, "Felix Bautista", "Jon's Underdogs"),
    (8, "Josh Naylor", "Work in Progress"),
    (9, "Freddy Peralta", "Work in Progress"),
    (9, "Cody Bellinger", "Jon's Underdogs"),
    (9, "Matt McLain", "Spacemen"),
    (9, "Bailey Ober", "Crews Control"),
    (9, "Andres Munoz", "Hello Peanuts!"),
    (9, "Jhoan Duran", "Hart of the Order"),
    (9, "Hunter Brown", "Tortured Baseball Department"),
    (9, "Luis Castillo", "Boston Estrellas"),
    (9, "Anthony Santander", "SkeleThor"),
    (9, "Riley Greene", "Springfield Isotopes"),
    (10, "Seiya Suzuki", "Springfield Isotopes"),
    (10, "Adley Rutschman", "SkeleThor"),
    (10, "Bryan Reynolds", "Boston Estrellas"),
    (10, "Joe Ryan", "Tortured Baseball Department"),
    (10, "Bryce Miller", "Hart of the Order"),
    (10, "Jordan Westburg", "Hello Peanuts!"),
    (10, "Christian Walker", "Crews Control"),
    (10, "Jake Burger", "Spacemen"),
    (10, "Hunter Greene", "Jon's Underdogs"),
    (10, "Mark Vientos", "Work in Progress"),
    (11, "Roki Sasaki", "Work in Progress"),
    (11, "Willy Adames", "Jon's Underdogs"),
    (11, "Mike Trout", "Spacemen"),
    (11, "Zac Gallen", "Crews Control"),
    (11, "Yainer Diaz", "Hello Peanuts!"),
    (11, "Salvador Perez", "Hart of the Order"),
    (11, "Cal Raleigh", "Tortured Baseball Department"),
    (11, "Triston Casas", "Boston Estrellas"),
    (11, "George Kirby", "SkeleThor"),
    (11, "Jeff Hoffman", "Springfield Isotopes"),
    (12, "Alec Bohm", "Springfield Isotopes"),
    (12, "Spencer Strider", "SkeleThor"),
    (12, "Sandy Alcantara", "Boston Estrellas"),
    (12, "Luis Robert Jr.", "Tortured Baseball Department"),
    (12, "Robert Suarez", "Hart of the Order"),
    (12, "Christian Yelich", "Hello Peanuts!"),
    (12, "Justin Steele", "Crews Control"),
    (12, "Willson Contreras", "Spacemen"),
    (12, "Matt Chapman", "Jon's Underdogs"),
    (12, "Will Smith", "Work in Progress"),
    (13, "Sonny Gray", "Work in Progress"),
    (13, "Dylan Crews", "Jon's Underdogs"),
    (13, "Tanner Scott", "Spacemen"),
    (13, "Ian Happ", "Crews Control"),
    (13, "Carlos Rodon", "Hello Peanuts!"),
    (13, "Luis Garcia Jr.", "Hart of the Order"),
    (13, "Steven Kwan", "Tortured Baseball Department"),
    (13, "Randy Arozarena", "Boston Estrellas"),
    (13, "Bryan Woo", "SkeleThor"),
    (13, "Jack Flaherty", "Springfield Isotopes"),
    (14, "Kevin Gausman", "Springfield Isotopes"),
    (14, "Trevor Megill", "SkeleThor"),
    (14, "Xander Bogaerts", "Boston Estrellas"),
    (14, "Brice Turang", "Tortured Baseball Department"),
    (14, "Pete Crow-Armstrong", "Hart of the Order"),
    (14, "Cristopher Sanchez", "Hello Peanuts!"),
    (14, "Brandon Nimmo", "Crews Control"),
    (14, "Kodai Senga", "Spacemen"),
    (14, "Luis Arraez", "Jon's Underdogs"),
    (14, "Nick Castellanos", "Work in Progress"),
    (15, "Vinnie Pasquantino", "Work in Progress"),
    (15, "J.T. Realmuto", "Jon's Underdogs"),
    (15, "Ezequiel Tovar", "Spacemen"),
    (15, "Josh Lowe", "Crews Control"),
    (15, "Yandy Diaz", "Hello Peanuts!"),
    (15, "Jasson Dominguez", "Hart of the Order"),
    (15, "Adolis Garcia", "Tortured Baseball Department"),
    (15, "Jurickson Profar", "Boston Estrellas"),
    (15, "Xavier Edwards", "SkeleThor"),
    (15, "Reynaldo Lopez", "Springfield Isotopes"),
    (16, "Seth Lugo", "Springfield Isotopes"),
    (16, "Ryan Pressly", "SkeleThor"),
    (16, "Lane Thomas", "Boston Estrellas"),
    (16, "Robbie Ray", "Tortured Baseball Department"),
    (16, "Shane McClanahan", "Hart of the Order"),
    (16, "Kenley Jansen", "Hello Peanuts!"),
    (16, "Shea Langeliers", "Crews Control"),
    (16, "Eugenio Suarez", "Spacemen"),
    (16, "Spencer Steer", "Jon's Underdogs"),
    (16, "Nick Pivetta", "Work in Progress"),
    (17, "Ryan Pepiot", "Work in Progress"),
    (17, "Pete Fairbanks", "Jon's Underdogs"),
    (17, "Jordan Romano", "Spacemen"),
    (17, "Andres Gimenez", "Crews Control"),
    (17, "Michael Toglia", "Hello Peanuts!"),
    (17, "Anthony Volpe", "Hart of the Order"),
    (17, "Isaac Paredes", "Tortured Baseball Department"),
    (17, "Austin Wells", "Boston Estrellas"),
    (17, "Jeremy Pena", "SkeleThor"),
    (17, "Colton Cowser", "Springfield Isotopes"),
    (18, "Jonathan India", "Springfield Isotopes"),
    (18, "Zach Eflin", "SkeleThor"),
    (18, "Taj Bradley", "Boston Estrellas"),
    (18, "Brandon Pfaadt", "Tortured Baseball Department"),
    (18, "Yusei Kikuchi", "Hart of the Order"),
    (18, "Nolan Arenado", "Hello Peanuts!"),
    (18, "Nathan Eovaldi", "Crews Control"),
    (18, "Andrew Vaughn", "Spacemen"),
    (18, "Tyler O'Neill", "Jon's Underdogs"),
    (18, "Taylor Ward", "Work in Progress"),
    (19, "Dansby Swanson", "Work in Progress"),
    (19, "Clay Holmes", "Jon's Underdogs"),
    (19, "Jorge Soler", "Spacemen"),
    (19, "David Bednar", "Crews Control"),
    (19, "Josh Jung", "Hello Peanuts!"),
    (19, "Bryson Stott", "Hart of the Order"),
    (19, "Nico Hoerner", "Tortured Baseball Department"),
    (19, "Carlos Estevez", "Boston Estrellas"),
    (19, "Kerry Carpenter", "SkeleThor"),
    (19, "MacKenzie Gore", "Springfield Isotopes"),
    (20, "Nick Lodolo", "Springfield Isotopes"),
    (20, "Tanner Houck", "SkeleThor"),
    (20, "Jackson Holliday", "Boston Estrellas"),
    (20, "Heliot Ramos", "Tortured Baseball Department"),
    (20, "Royce Lewis", "Hart of the Order"),
    (20, "Max Scherzer", "Hello Peanuts!"),
    (20, "Kyle Finnegan", "Crews Control"),
    (20, "Mitch Keller", "Spacemen"),
    (20, "Drew Rasmussen", "Jon's Underdogs"),
    (20, "Paul Goldschmidt", "Work in Progress"),
    (21, "Justin Martinez", "Work in Progress"),
    (21, "Jose Berrios", "Jon's Underdogs"),
    (21, "Michael Wacha", "Spacemen"),
    (21, "Spencer Arrighetti", "Crews Control"),
    (21, "Ryan Mountcastle", "Hello Peanuts!"),
    (21, "Gavin Williams", "Hart of the Order"),
    (21, "Jesus Luzardo", "Tortured Baseball Department"),
    (21, "Max Meyer", "Boston Estrellas"),
    (21, "Victor Robles", "SkeleThor"),
    (21, "Willi Castro", "Springfield Isotopes"),
    (22, "A.J. Puk", "Springfield Isotopes"),
    (22, "Cam Smith", "SkeleThor"),
    (22, "Carlos Correa", "Boston Estrellas"),
    (22, "Masyn Winn", "Tortured Baseball Department"),
    (22, "Tommy Edman", "Hart of the Order"),
    (22, "Joey Ortiz", "Hello Peanuts!"),
    (22, "Zach Neto", "Crews Control"),
    (22, "Aroldis Chapman", "Spacemen"),
    (22, "Shohei Ohtani", "Jon's Underdogs"),
    (22, "Logan O'Hoppe", "Work in Progress"),
    (23, "Gleyber Torres", "Work in Progress"),
    (23, "Lourdes Gurriel Jr.", "Jon's Underdogs"),
    (23, "Byron Buxton", "Spacemen"),
    (23, "Cedric Mullins", "Crews Control"),
    (23, "Rhys Hoskins", "Hello Peanuts!"),
    (23, "Shane Baz", "Hart of the Order"),
    (23, "Ronel Blanco", "Tortured Baseball Department"),
    (23, "Jake McCarthy", "Boston Estrellas"),
    (23, "Jared Jones", "SkeleThor"),
    (23, "Gabriel Moreno", "Springfield Isotopes"),
]

ALL_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
INVERSE = {"ERA", "WHIP"}
INJURY_PROB = {"pitcher": 0.45, "hitter": 0.18}
INJURY_SEVERITY = {"pitcher": (0.20, 0.60), "hitter": (0.15, 0.40)}
STAT_VARIANCE = 0.12
HITTING_COUNTING = ["r", "hr", "rbi", "sb", "h", "ab"]
PITCHING_COUNTING = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]
REPLACEMENT_HITTER = {"r": 55, "hr": 12, "rbi": 50, "sb": 5, "h": 125, "ab": 500}
REPLACEMENT_SP = {"w": 7, "k": 120, "sv": 0, "ip": 140, "er": 70, "bb": 50, "h_allowed": 139}
REPLACEMENT_RP = {"w": 2, "k": 55, "sv": 5, "ip": 60, "er": 30, "bb": 21, "h_allowed": 60}


def sim_season(rosters, rng, h_slots=13, p_slots=9):
    stats = {}
    for team, players in rosters.items():
        hitters = [p for p in players if p["player_type"] == "hitter"]
        pitchers = [p for p in players if p["player_type"] == "pitcher"]
        ah, ap = [], []
        for h in hitters:
            frac = rng.uniform(*INJURY_SEVERITY["hitter"]) if rng.random() < INJURY_PROB["hitter"] else 0
            row = {}
            for col in HITTING_COUNTING:
                base = float(h.get(col, 0) or 0)
                varied = max(0, base * (1 + rng.normal(0, STAT_VARIANCE)))
                row[col] = varied * (1 - frac) + REPLACEMENT_HITTER.get(col, 0) * frac
            ah.append(row)
        for p in pitchers:
            frac = rng.uniform(*INJURY_SEVERITY["pitcher"]) if rng.random() < INJURY_PROB["pitcher"] else 0
            repl = REPLACEMENT_RP if float(p.get("sv", 0) or 0) >= 15 else REPLACEMENT_SP
            row = {}
            for col in PITCHING_COUNTING:
                base = float(p.get(col, 0) or 0)
                varied = max(0, base * (1 + rng.normal(0, STAT_VARIANCE)))
                row[col] = varied * (1 - frac) + repl.get(col, 0) * frac
            ap.append(row)
        ah.sort(key=lambda x: x["r"] + x["hr"] + x["rbi"] + x["sb"], reverse=True)
        ap.sort(key=lambda x: (x.get("sv", 0) >= 15, x["w"] + x["k"] + x["sv"]), reverse=True)
        ah, ap = ah[:h_slots], ap[:p_slots]

        r = sum(x["r"] for x in ah)
        hr = sum(x["hr"] for x in ah)
        rbi = sum(x["rbi"] for x in ah)
        sb = sum(x["sb"] for x in ah)
        th = sum(x["h"] for x in ah)
        tab = sum(x["ab"] for x in ah)
        avg = th / tab if tab > 0 else 0
        w = sum(x["w"] for x in ap)
        k = sum(x["k"] for x in ap)
        sv = sum(x["sv"] for x in ap)
        tip = sum(x["ip"] for x in ap)
        ter = sum(x["er"] for x in ap)
        tbb = sum(x["bb"] for x in ap)
        tha = sum(x["h_allowed"] for x in ap)
        era = ter * 9 / tip if tip > 0 else 99
        whip = (tbb + tha) / tip if tip > 0 else 99
        stats[team] = {
            "R": r, "HR": hr, "RBI": rbi, "SB": sb, "AVG": avg,
            "W": w, "K": k, "SV": sv, "ERA": era, "WHIP": whip,
        }
    return stats


def main():
    board = build_draft_board(
        projections_dir=PROJECTIONS_DIR / "2025",
        positions_path=POSITIONS_PATH,
        systems=["steamer", "zips"],
        weights={"steamer": 0.50, "zips": 0.50},
        num_teams=10,
    )

    # Build ASCII lookup for fuzzy matching
    board_lookup = {}
    for idx, row in board.iterrows():
        key = ascii_name(row["name"])
        if key not in board_lookup or row["var"] > board.loc[board_lookup[key], "var"]:
            board_lookup[key] = idx

    # Match draft picks to projections
    team_rosters = {}
    matched = 0
    missed = 0
    for rnd, player, team in DRAFT_2025:
        if team not in team_rosters:
            team_rosters[team] = []
        key = ascii_name(player)
        if key in board_lookup:
            team_rosters[team].append(board.loc[board_lookup[key]])
            matched += 1
        else:
            # Try without suffixes
            key_clean = key.replace(" jr.", "").replace(" sr.", "").replace(" ii", "").strip()
            found = False
            for bkey, bidx in board_lookup.items():
                if key_clean == bkey or (len(key_clean) > 5 and key_clean in bkey):
                    team_rosters[team].append(board.loc[bidx])
                    matched += 1
                    found = True
                    break
            if not found:
                missed += 1

    print(f"Matched: {matched}, Missed: {missed}")
    for team in sorted(team_rosters.keys()):
        n = len(team_rosters[team])
        h = sum(1 for p in team_rosters[team] if p["player_type"] == "hitter")
        p = sum(1 for p in team_rosters[team] if p["player_type"] == "pitcher")
        print(f"  {team:<35} {n} ({h}H/{p}P)")

    # Run MC
    rng = np.random.default_rng(42)
    N = 1000
    all_totals = {t: [] for t in team_rosters}
    all_finishes = {t: [] for t in team_rosters}
    all_cat_med = {t: {c: [] for c in ALL_CATS} for t in team_rosters}

    for _ in range(N):
        stats = sim_season(team_rosters, rng)
        results = {}
        for cat in ALL_CATS:
            rev = cat not in INVERSE
            ranked = sorted(stats.keys(), key=lambda t: stats[t][cat], reverse=rev)
            for i, t in enumerate(ranked):
                results.setdefault(t, {})[f"{cat}_pts"] = len(stats) - i
                results[t].setdefault("stats", {})[cat] = stats[t][cat]
        for t in results:
            results[t]["total"] = sum(results[t][f"{c}_pts"] for c in ALL_CATS)
        for t in team_rosters:
            total = results[t]["total"]
            all_totals[t].append(total)
            rank = 1 + sum(1 for o in team_rosters if results[o]["total"] > total)
            all_finishes[t].append(rank)
            for cat in ALL_CATS:
                all_cat_med[t][cat].append(results[t]["stats"][cat])

    # Compare
    print(f"\n{'=' * 100}")
    print("2025 BACKTEST: Simulated vs Actual Standings")
    print(f"{'=' * 100}")
    print(f"{'Team':<28} {'Act':>4} {'Act':>5} {'Sim':>5} {'Sim':>5} {'Diff':>5}")
    print(f"{'':28} {'Rnk':>4} {'Pts':>5} {'Med':>5} {'MdRk':>5} {'Rank':>5}")
    print("-" * 100)

    sim_order = sorted(
        team_rosters.keys(),
        key=lambda t: np.median(all_totals[t]),
        reverse=True,
    )
    for team in sim_order:
        tots = np.array(all_totals[team])
        fins = np.array(all_finishes[team])
        a = ACTUAL.get(team, {})
        med_pts = np.median(tots)
        med_rank = np.median(fins)
        act_rank = a.get("rank", "?")
        act_pts = a.get("total", "?")
        rdiff = abs(med_rank - act_rank) if isinstance(act_rank, (int, float)) else "?"
        print(
            f"{team:<28} {act_rank:>4} {act_pts:>5} {med_pts:>5.0f} "
            f"{med_rank:>5.1f} {rdiff:>5.1f}"
        )

    # Category comparison for Hart
    print(f"\n{'=' * 80}")
    print("HART OF THE ORDER: Projected Stats vs Actual")
    print(f"{'=' * 80}")
    print(f"{'Cat':>5} {'Actual':>8} {'Sim Med':>8} {'Diff%':>7}")
    print("-" * 30)
    for cat in ALL_CATS:
        act_val = ACTUAL["Hart of the Order"].get(cat, 0)
        sim_val = np.median(all_cat_med["Hart of the Order"][cat])
        diff_pct = (sim_val - act_val) / act_val * 100 if act_val != 0 else 0
        if cat in ("AVG", "ERA", "WHIP"):
            print(f"{cat:>5} {act_val:>8.3f} {sim_val:>8.3f} {diff_pct:>+6.1f}%")
        else:
            print(f"{cat:>5} {act_val:>8.0f} {sim_val:>8.0f} {diff_pct:>+6.1f}%")


if __name__ == "__main__":
    main()
