"""Calibrate position-aware replacement levels from the league's own free agents.

For two league snapshots (post-draft and now), at each position average the top-3
UN-ROSTERED players by SGP -> a position-aware replacement line, then average the
two snapshots. Emits REPLACEMENT_BY_POSITION for utils/constants.py.

Matching notes (correctness):
  - "Rostered" is split by TYPE (hitter vs pitcher) from each roster entry's
    positions, so a two-way player rostered as a hitter still leaves his pitcher
    self in the free-agent SP pool (Yahoo rosters the two halves separately).
  - Roster names are cleaned via yahoo_id -> current-hydrated-roster name, so an
    accented rostered star can't leak into the FA pool through stored mojibake.

Read-only against prod; writes nothing.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SNAPSHOTS = [("post-draft", "2026-03-24", "preseason"), ("now", "2026-06-09", "current")]
HITTER_POS = ["C", "1B", "2B", "3B", "SS", "OF"]
PITCHER_TOKENS = {"SP", "RP", "P"}
GS_SP_THRESHOLD = 10
IP_SP_FALLBACK = 100.0
TOP_N = 3


def _load_dotenv() -> None:
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


def main() -> None:
    _load_dotenv()
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key
    from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
    from fantasy_baseball.data.redis_store import (
        get_blended_projections,
        get_weekly_roster_history,
    )
    from fantasy_baseball.models.player import PlayerType
    from fantasy_baseball.sgp.player_value import calculate_player_sgp
    from fantasy_baseball.utils.name_utils import normalize_name

    kv = build_explicit_upstash_kv()
    rosters_hist = get_weekly_roster_history(kv)
    positions_map = _unwrap(kv.get(redis_key(CacheKey.POSITIONS))) or {}

    # yahoo_id -> clean normalized name, from the current hydrated rosters (which
    # carry both player_id and an un-mangled name). Cleans accented roster names.
    yahoo_clean: dict[str, str] = {}
    cur_roster = _unwrap(kv.get(redis_key(CacheKey.ROSTER))) or []
    cur_opp = _unwrap(kv.get(redis_key(CacheKey.OPP_ROSTERS))) or {}
    for p in cur_roster:
        if p.get("player_id"):
            yahoo_clean[str(p["player_id"])] = normalize_name(p.get("name", ""))
    for plist in cur_opp.values():
        for p in plist:
            if p.get("player_id"):
                yahoo_clean[str(p["player_id"])] = normalize_name(p.get("name", ""))

    def rostered_by_type(date: str) -> tuple[set, set]:
        hit, pit = set(), set()
        for e in rosters_hist.get(date, []):
            toks = {t.strip().upper() for t in str(e.get("positions", "")).split(",")}
            is_pit = bool(toks & PITCHER_TOKENS)
            yid = str(e.get("yahoo_id") or "")
            norm = yahoo_clean.get(yid) or normalize_name(e["player_name"])
            (pit if is_pit else hit).add(norm)
        return hit, pit

    def load_pool(which: str):
        if which == "preseason":
            return (get_blended_projections(kv, "hitters") or []), (
                get_blended_projections(kv, "pitchers") or []
            )
        data = _unwrap(kv.get(redis_key(CacheKey.FULL_SEASON_PROJECTIONS))) or {}
        return data.get("hitters", []), data.get("pitchers", [])

    def to_hitter(r):
        d = {"player_type": PlayerType.HITTER}
        for k in ("r", "hr", "rbi", "sb", "h", "ab", "avg"):
            d[k] = float(r.get(k) or 0)
        if d["avg"] == 0 and d["ab"] > 0:
            d["avg"] = d["h"] / d["ab"]
        return d

    def to_pitcher(r):
        d = {"player_type": PlayerType.PITCHER}
        for k in ("w", "k", "sv", "ip", "er", "bb", "h_allowed", "era", "whip", "gs"):
            d[k] = float(r.get(k) or 0)
        if d["ip"] > 0:
            d["era"] = d["era"] or 9 * d["er"] / d["ip"]
            d["whip"] = d["whip"] or (d["bb"] + d["h_allowed"]) / d["ip"]
        return d

    def is_sp(d):
        return d["gs"] >= GS_SP_THRESHOLD if d["gs"] else d["ip"] >= IP_SP_FALLBACK

    per_snapshot: dict[str, dict[str, list]] = {}
    for label, date, which in SNAPSHOTS:
        rost_hit, rost_pit = rostered_by_type(date)
        hpool, ppool = load_pool(which)
        groups: dict[str, list] = {pos: [] for pos in HITTER_POS}
        groups["SP"], groups["RP"] = [], []

        for r in hpool:
            norm = normalize_name(r.get("name") or "")
            if norm in rost_hit or norm not in positions_map:
                continue
            d = to_hitter(r)
            d["_name"], d["_sgp"] = r.get("name"), calculate_player_sgp(d)
            for pos in HITTER_POS:
                if pos in positions_map[norm]:
                    groups[pos].append(d)

        for r in ppool:
            norm = normalize_name(r.get("name") or "")
            if norm in rost_pit:
                continue
            d = to_pitcher(r)
            if d["ip"] <= 0:
                continue
            d["_name"], d["_sgp"] = r.get("name"), calculate_player_sgp(d)
            groups["SP" if is_sp(d) else "RP"].append(d)

        per_snapshot[label] = {
            pos: sorted(c, key=lambda d: d["_sgp"], reverse=True)[:TOP_N]
            for pos, c in groups.items()
        }

    hit_cols = ["r", "hr", "rbi", "sb", "h", "ab"]
    pit_cols = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]
    all_pos = [*HITTER_POS, "SP", "RP"]

    def avg_line(rows, cols):
        return {c: sum(d[c] for d in rows) / len(rows) for c in cols} if rows else {}

    combined_by_pos: dict[str, dict] = {}
    for pos in all_pos:
        cols = pit_cols if pos in ("SP", "RP") else hit_cols
        print(f"\n=== {pos} ===")
        snap_lines = []
        for label, _, _ in SNAPSHOTS:
            rows = per_snapshot[label][pos]
            print(
                f"  {label:10s} top-3: " + ", ".join(f"{d['_name']}({d['_sgp']:.1f})" for d in rows)
            )
            snap_lines.append(avg_line(rows, cols))
        combined = {c: sum(s.get(c, 0) for s in snap_lines) / len(snap_lines) for c in cols}
        combined_by_pos[pos] = {c: round(v) for c, v in combined.items()}
        if pos in ("SP", "RP"):
            ip = combined["ip"] or 1
            print(
                f"  REPLACEMENT: {combined['ip']:.0f} IP, {combined['w']:.1f} W, {combined['k']:.0f} K,"
                f" {combined['sv']:.1f} SV, {9 * combined['er'] / ip:.2f} ERA,"
                f" {(combined['bb'] + combined['h_allowed']) / ip:.2f} WHIP"
            )
        else:
            ab = combined["ab"] or 1
            print(
                f"  REPLACEMENT: {combined['ab']:.0f} AB, {combined['r']:.0f} R, {combined['hr']:.1f} HR,"
                f" {combined['rbi']:.0f} RBI, {combined['sb']:.1f} SB, {combined['h'] / ab:.3f} AVG"
            )

    print("\n\nREPLACEMENT_BY_POSITION: dict[str, dict[str, int]] = {")
    for pos in all_pos:
        print(f"    {pos!r}: {combined_by_pos[pos]},")
    print("}")


if __name__ == "__main__":
    main()
