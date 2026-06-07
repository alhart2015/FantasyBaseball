"""READ-ONLY: who are the top-N free agents per position, and how does averaging
depth change the replacement SB? Surfaces whether the SS/2B ~15 SB is driven by a
couple of speedsters (top-3 artifact) or holds as you average deeper.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SNAPSHOTS = [("post-draft", "2026-03-24", "preseason"), ("now", "2026-06-09", "current")]
HITTER_POS = ["C", "1B", "2B", "3B", "SS", "OF"]
PITCHER_TOKENS = {"SP", "RP", "P"}
DEPTHS = [3, 5, 10]


def _load_dotenv():
    for line in (Path(__file__).resolve().parent.parent / ".env").read_text("utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _unwrap(raw):
    if raw is None:
        return None
    obj = json.loads(raw)
    return obj["_data"] if isinstance(obj, dict) and "_data" in obj else obj


def main():
    _load_dotenv()
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key
    from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
    from fantasy_baseball.data.redis_store import get_blended_projections, get_weekly_roster_history
    from fantasy_baseball.models.player import PlayerType
    from fantasy_baseball.sgp.player_value import calculate_player_sgp
    from fantasy_baseball.utils.name_utils import normalize_name

    kv = build_explicit_upstash_kv()
    rosters_hist = get_weekly_roster_history(kv)
    positions_map = _unwrap(kv.get(redis_key(CacheKey.POSITIONS))) or {}

    yahoo_clean = {}
    for src in (_unwrap(kv.get(redis_key(CacheKey.ROSTER))) or [], *(
        _unwrap(kv.get(redis_key(CacheKey.OPP_ROSTERS))) or {}
    ).values()):
        for p in src:
            if p.get("player_id"):
                yahoo_clean[str(p["player_id"])] = normalize_name(p.get("name", ""))

    def rostered_hit(date):
        hit = set()
        for e in rosters_hist.get(date, []):
            toks = {t.strip().upper() for t in str(e.get("positions", "")).split(",")}
            if toks & PITCHER_TOKENS:
                continue
            yid = str(e.get("yahoo_id") or "")
            hit.add(yahoo_clean.get(yid) or normalize_name(e["player_name"]))
        return hit

    def load_h(which):
        if which == "preseason":
            return get_blended_projections(kv, "hitters") or []
        return (_unwrap(kv.get(redis_key(CacheKey.FULL_SEASON_PROJECTIONS))) or {}).get("hitters", [])

    def sgp_of(r):
        ab = float(r.get("ab") or 0) or 1
        return calculate_player_sgp(
            {
                "player_type": PlayerType.HITTER,
                "r": float(r.get("r") or 0),
                "hr": float(r.get("hr") or 0),
                "rbi": float(r.get("rbi") or 0),
                "sb": float(r.get("sb") or 0),
                "avg": float(r.get("avg") or 0) or float(r.get("h") or 0) / ab,
                "ab": ab,
            }
        )

    for pos in HITTER_POS:
        print(f"\n========== {pos} ==========")
        depth_sb = {n: [] for n in DEPTHS}
        for label, date, which in SNAPSHOTS:
            rost = rostered_hit(date)
            cands = []
            for r in load_h(which):
                norm = normalize_name(r.get("name") or "")
                if norm in rost or norm not in positions_map or pos not in positions_map[norm]:
                    continue
                cands.append((sgp_of(r), r.get("name"), float(r.get("sb") or 0)))
            cands.sort(reverse=True, key=lambda t: t[0])
            top = cands[: max(DEPTHS)]
            print(f"  {label} top-{max(DEPTHS)} (name, SB, SGP):")
            for sgp, name, sb in top:
                print(f"    {name[:24]:24s} SB={sb:5.1f}  SGP={sgp:5.1f}")
            for n in DEPTHS:
                if len(cands) >= 1:
                    depth_sb[n].append(sum(c[2] for c in cands[:n]) / min(n, len(cands)))
        line = "  avg replacement SB by depth:  " + "   ".join(
            f"top-{n}={sum(depth_sb[n]) / len(depth_sb[n]):.1f}" for n in DEPTHS if depth_sb[n]
        )
        print(line)


if __name__ == "__main__":
    main()
