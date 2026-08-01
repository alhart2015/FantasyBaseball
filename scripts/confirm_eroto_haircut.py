"""One-off confirmation: what the mean_scale playing-time haircut does to the
analytic ERoto projection, before vs after, at TODAY's fraction_remaining.

Not a permanent script -- a reproducible answer to "which is right, ERoto or the
MC, and by how much" for the hello-peanuts question. Player-level (Juan Soto +
archetypes) is exact; team-level uses the most recent local roster snapshot.

ERoto today (before): sums full rest_of_season counting stats, NO haircut.
Moment-consistent (after): per stat,
    after = base * eff_mean + repl_ros * (1 - eff_mean)
    eff_mean = 1 - (1 - mean_scale) * fraction_remaining     (matches the MC)
    repl_ros = full_season_replacement * fraction_remaining  (ROS-window backfill)
which is the leading-order mean of simulation._apply_variance line 533.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from fantasy_baseball.data.projections import blend_projections  # noqa: E402
from fantasy_baseball.models.player import PlayerType  # noqa: E402
from fantasy_baseball.simulation import _GENERIC_HITTER_REPL  # noqa: E402
from fantasy_baseball.utils.constants import REPLACEMENT_BY_POSITION  # noqa: E402
from fantasy_baseball.utils.playing_time import playing_time_params  # noqa: E402
from fantasy_baseball.utils.time_utils import compute_fraction_remaining  # noqa: E402

cfg = yaml.safe_load((ROOT / "config" / "league.yaml").read_text())
SYSTEMS = cfg["projections"]["systems"]
WEIGHTS = cfg["projections"]["weights"]
START = date.fromisoformat(str(cfg["league"]["season_start"]))
END = date.fromisoformat(str(cfg["league"]["season_end"]))
TODAY = date(2026, 6, 10)
FR = compute_fraction_remaining(START, END, TODAY)

PROJ_DIR = ROOT / "data" / "projections" / "2026" / "rest_of_season" / "2026-06-08"

H_COUNT = ["r", "hr", "rbi", "sb"]
P_COUNT = ["w", "k", "sv"]


def norm(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.strip().lower()


print(f"TODAY={TODAY}  season {START}..{END}  fraction_remaining={FR:.3f}")
print(f"systems={SYSTEMS}\n")

hit_df, pit_df, _ = blend_projections(PROJ_DIR, SYSTEMS, WEIGHTS)
hit_df = hit_df.assign(_n=hit_df["name"].map(norm))
pit_df = pit_df.assign(_n=pit_df["name"].map(norm))


def eff_mean(mean_scale: float) -> float:
    return 1.0 - (1.0 - mean_scale) * FR


def hitter_repl(positions: list[str]) -> dict:
    pos_keys = [p for p in positions if p in REPLACEMENT_BY_POSITION and p not in ("SP", "RP")]
    if not pos_keys:
        return _GENERIC_HITTER_REPL
    # highest-counting OF/MI typically; pick by total counting as a proxy
    return max(
        (REPLACEMENT_BY_POSITION[p] for p in pos_keys),
        key=lambda d: d["r"] + d["hr"] + d["rbi"] + d["sb"],
    )


def show_hitter(name: str, positions: list[str]) -> None:
    row = hit_df[hit_df["_n"] == norm(name)]
    if row.empty:
        print(f"  [hitter not found: {name}]")
        return
    r = row.iloc[0]
    ros_pa = float(r.get("pa", 0) or 0)
    fs_pa = ros_pa / FR if FR > 0 else ros_pa  # full-season pace from ROS slice
    ms, cv = playing_time_params(PlayerType.HITTER, fs_pa)
    em = eff_mean(ms)
    repl = hitter_repl(positions)
    repl_ros = {k: v * FR for k, v in repl.items()}
    print(f"  {name}  (pos={positions}, ROS PA={ros_pa:.0f}, full-season pace~{fs_pa:.0f})")
    print(f"    band: mean_scale={ms:.4f}  cv_pt={cv:.4f}  ->  eff_mean@fr={em:.4f}  "
          f"(haircut {100*(1-em):.1f}%)")
    print(f"    {'stat':>5} {'before':>8} {'after':>8} {'delta':>8}")
    for c in H_COUNT:
        base = float(r.get(c, 0) or 0)
        after = base * em + repl_ros.get(c, 0) * (1 - em)
        print(f"    {c:>5} {base:8.1f} {after:8.1f} {after - base:8.1f}")


def show_pitcher(name: str) -> None:
    row = pit_df[pit_df["_n"] == norm(name)]
    if row.empty:
        print(f"  [pitcher not found: {name}]")
        return
    r = row.iloc[0]
    ros_ip = float(r.get("ip", 0) or 0)
    fs_ip = ros_ip / FR if FR > 0 else ros_ip
    ms, cv = playing_time_params(PlayerType.PITCHER, fs_ip)
    em = eff_mean(ms)
    role = "SP" if fs_ip >= 100 else "RP"
    repl = REPLACEMENT_BY_POSITION[role]
    repl_ros = {k: v * FR for k, v in repl.items()}
    print(f"  {name}  (role~{role}, ROS IP={ros_ip:.0f}, full-season pace~{fs_ip:.0f})")
    print(f"    band: mean_scale={ms:.4f}  cv_pt={cv:.4f}  ->  eff_mean@fr={em:.4f}  "
          f"(haircut {100*(1-em):.1f}%)")
    print(f"    {'stat':>5} {'before':>8} {'after':>8} {'delta':>8}")
    for c in P_COUNT:
        base = float(r.get(c, 0) or 0)
        after = base * em + repl_ros.get(c, 0) * (1 - em)
        print(f"    {c:>5} {base:8.1f} {after:8.1f} {after - base:8.1f}")


print("=" * 64)
print("PLAYER-LEVEL: before (current ERoto) vs after (moment-consistent)")
print("=" * 64)
print("\n-- Juan Soto (the headline) --")
show_hitter("Juan Soto", ["OF"])
print("\n-- Contrast: a mid-volume regular --")
show_hitter("Jonathan India", ["2B"])
print("\n-- Contrast: an elite closer --")
show_pitcher("Edwin Diaz")
print("\n-- Contrast: a workhorse starter --")
show_pitcher("Tarik Skubal")
