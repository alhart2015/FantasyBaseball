"""Draft-value report CLI: leaderboard + per-player table + markdown.

Runs the draft-value metric (realized VAR vs draft-slot par expectation) against
the synced KV store and writes a markdown report to
``data/analysis/draft_value_report.md``.

The in-season loaders (full-season projections, game logs, transactions, rosters)
read the local KV store. If the store is stale/empty locally the report is empty;
sync first (see ``run_season_dashboard.py``; it syncs from Upstash at startup
unless ``--no-sync``). This script is read-only and does not sync on its own.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # data has accented names

from fantasy_baseball.analysis.draft_value import run_draft_value


def _fmt(x: float | None) -> str:
    return "  N/A" if x is None else f"{x:6.1f}"


def main() -> None:
    players, teams = run_draft_value()
    footer = (
        "NOTE: This grades the DRAFT only (keepers + drafted picks). Waiver / "
        "in-season pickups are evaluated separately by the transaction analyzer "
        "(deltaRoto) -- see the /transactions dashboard."
    )
    lines = ["# Draft Value Report", ""]
    lines.append("## DRAFT GRADE (keepers + drafted picks)")
    lines.append("| Team | avg | sum | drafted+kept | waiver | trade-excl |")
    lines.append("|---|---|---|---|---|---|")
    for t in sorted(
        teams,
        key=lambda r: r.avg_value if r.avg_value == r.avg_value else -9e9,
        reverse=True,
    ):
        lines.append(
            f"| {t.team} | {_fmt(t.avg_value)} | {_fmt(t.sum_value)} "
            f"| {t.credited_count} | {t.waiver_count} | {t.case3_count} |"
        )
    lines.append("")
    lines.append("## Per-player (projected)")
    lines.append(
        "| Player | kind | slot | preVAR | estVAR | par | skill | luck | value | valueYTD |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(
        players,
        key=lambda p: p.value_proj if p.value_proj is not None else -9e9,
        reverse=True,
    ):
        slot = "  -" if p.slot is None else f"{p.slot:3d}"
        par = (
            _fmt(p.est_var_proj - p.value_proj)
            if p.est_var_proj is not None and p.value_proj is not None
            else "  N/A"
        )
        lines.append(
            f"| {p.name} | {p.baseline_kind} | {slot} | {_fmt(p.preseason_var)} "
            f"| {_fmt(p.est_var_proj)} | {par} | {_fmt(p.skill)} | {_fmt(p.luck)} "
            f"| {_fmt(p.value_proj)} | {_fmt(p.value_ytd)} |"
        )
    lines.append("")
    lines.append(footer)
    report = "\n".join(lines)
    out = Path("data/analysis/draft_value_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
