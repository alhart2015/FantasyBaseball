"""READ-ONLY: show 100 simulated full-season PA (hitters) / IP (pitchers) per player.

Reconstructs the live rosters from prod cache, finds the requested players, and
draws the new empirical playing-time multiplier 100 times for each (the same
sampler simulate_remaining_season uses), reporting realized volume = projected *
scale. fraction_remaining = 1.0 here so the FULL-season shape is legible (the live
MC damps this by the remaining fraction). Reads only; writes nothing.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

TARGETS = ["Juan Soto", "Bryan Woo", "Zack Wheeler", "Mason Miller", "Josh Hader"]
N_SIMS = 100
SEED = 42


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _unwrap(raw):
    if raw is None:
        return None
    obj = json.loads(raw)
    if isinstance(obj, dict) and "_data" in obj and "_meta" in obj:
        return obj["_data"]
    return obj


def main() -> None:
    _load_dotenv()
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key
    from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
    from fantasy_baseball.models.player import Player, PlayerType
    from fantasy_baseball.utils.constants import AB_PER_PA
    from fantasy_baseball.utils.name_utils import normalize_name
    from fantasy_baseball.utils.playing_time import (
        playing_time_params,
        playing_time_shape,
        scale_from_uniform,
    )

    kv = build_explicit_upstash_kv()
    opp_raw = _unwrap(kv.get(redis_key(CacheKey.OPP_ROSTERS))) or {}
    user_raw = _unwrap(kv.get(redis_key(CacheKey.ROSTER))) or []

    # Flatten every rostered player across the league.
    all_players: list[Player] = [Player.from_dict(p) for p in user_raw]
    for plist in opp_raw.values():
        all_players.extend(Player.from_dict(p) for p in plist)
    by_norm = {normalize_name(p.name): p for p in all_players}

    def projected_volume(p: Player) -> tuple[float, str]:
        """Return (volume, unit) -- PA for hitters, IP for pitchers."""
        src = p.full_season_projection or p.rest_of_season
        d = src.to_dict() if src is not None else {}
        if p.player_type == PlayerType.HITTER:
            pa = float(d.get("pa") or 0.0)
            if pa <= 0:
                pa = float(d.get("ab") or 0.0) / AB_PER_PA
            return pa, "PA"
        return float(d.get("ip") or 0.0), "IP"

    rng = np.random.default_rng(SEED)
    for name in TARGETS:
        p = by_norm.get(normalize_name(name))
        if p is None:
            print(f"\n{name}: NOT FOUND on any league roster\n")
            continue
        vol, unit = projected_volume(p)
        mean_scale, cv_pt = playing_time_params(p.player_type, vol)
        ladder = playing_time_shape(p.player_type, vol)
        us = rng.random(N_SIMS)
        scales = np.array([scale_from_uniform(mean_scale, cv_pt, ladder, float(u), 1.0) for u in us])
        realized = scales * vol

        role = (
            "hitter"
            if p.player_type == PlayerType.HITTER
            else ("SP" if vol >= 100 else "RP")
        )
        print(
            f"\n{name} ({role}) -- projected {vol:.0f} {unit}, "
            f"mean_scale={mean_scale:.3f} cv_pt={cv_pt:.3f}"
        )
        vals = sorted(int(round(v)) for v in realized)
        for i in range(0, N_SIMS, 10):
            print("  " + " ".join(f"{v:4d}" for v in vals[i : i + 10]))
        print(
            f"  -> min={realized.min():.0f}  p50={np.percentile(realized, 50):.0f}  "
            f"max={realized.max():.0f}  mean={realized.mean():.0f} {unit}  "
            f"(scale max={scales.max():.2f}x)"
        )


if __name__ == "__main__":
    main()
