"""Pins for the manual-pipeline CLI report renderer.

The renderer is presentation only, so these tests are about honesty of
presentation rather than arithmetic: the output must stay printable on a
cp1252 console, must say "no upgrade" out loud instead of leaving a blank
cell, must not truncate a real player's name, and must not reorder or
re-derive anything the audit already decided.

This source file stays ASCII per the repo rule. The accented fixture name
and the banned "pretty" glyphs are written as escapes (``\\u00e1``,
``\\u2192``, ...) so the file itself is safe to open and print anywhere
while the values under test are the real characters.
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from fantasy_baseball.lineup.roster_audit import AuditEntry
from fantasy_baseball.manual.report import MISSING, NO_UPGRADE, render_audit_report

TEAM = "Hart Attack"
EFFECTIVE = date(2026, 8, 17)
ROS_SNAPSHOT = "2026-08-16"

#: Real accented name ("Yoan Moncada" with an acute a), spelled as an escape.
ACCENTED_NAME = "Yo\u00e1n Moncada"

#: Arrow, em dash, en dash, true minus, bullet, sigma -- every glyph the repo
#: rule bans from anything that can reach ``print()`` on this Windows box.
BANNED_GLYPHS = ["\u2192", "\u2014", "\u2013", "\u2212", "\u2022", "\u03c3"]


def make_candidate(
    name: str,
    *,
    total: float,
    categories: dict[str, float] | None = None,
    p_positive: float = 0.88,
    verdict: str = "real",
    sgp: float = 8.4,
    sgp_gap: float = 6.3,
    positions: list[str] | None = None,
    before_total: float = 71.5,
) -> dict[str, object]:
    """A scored-candidate dict shaped like the ones ``audit_roster`` stores."""
    cats = categories if categories is not None else {"HR": 0.5, "SB": 1.0, "AVG": 0.0}
    return {
        "name": name,
        "player_type": "hitter",
        "positions": positions if positions is not None else ["3B"],
        "sgp": sgp,
        "gap": sgp_gap,
        "delta_roto": {
            "total": total,
            "before_total": before_total,
            "after_total": before_total + total,
            "categories": {cat: {"roto_delta": value} for cat, value in cats.items()},
        },
        "band": {
            "mean": total,
            "sd": 0.4,
            "p_positive": p_positive,
            "verdict": verdict,
        },
        "player_id": "fa-" + name.replace(" ", "-"),
    }


def make_upgrade(
    player: str,
    best_fa: str,
    *,
    total: float,
    categories: dict[str, float] | None = None,
    slot: str = "BN",
    player_sgp: float = 2.1,
    sgp_gap: float = 6.3,
    positions: list[str] | None = None,
) -> AuditEntry:
    """An entry whose audit found a genuine upgrade."""
    candidate = make_candidate(best_fa, total=total, categories=categories, sgp_gap=sgp_gap)
    return AuditEntry(
        player=player,
        player_type="hitter",
        positions=positions if positions is not None else ["3B", "UTIL"],
        slot=slot,
        player_sgp=player_sgp,
        player_id="roster-" + player.replace(" ", "-"),
        best_fa=best_fa,
        best_fa_type="hitter",
        best_fa_positions=["3B"],
        best_fa_sgp=8.4,
        best_fa_id=str(candidate["player_id"]),
        gap=sgp_gap,
        candidates=[candidate],
    )


def make_hold(player: str, *, n_candidates: int = 3) -> AuditEntry:
    """An entry the audit scored but found no upgrade for (``best_fa`` is None)."""
    return AuditEntry(
        player=player,
        player_type="hitter",
        positions=["OF"],
        slot="OF",
        player_sgp=12.3,
        player_id="roster-" + player.replace(" ", "-"),
        candidates=[make_candidate(f"Bench Guy {i}", total=-0.5) for i in range(n_candidates)],
    )


def make_il(player: str) -> AuditEntry:
    """An IL row, appended by ``audit_roster`` with ``slot='IL'`` and no SGP."""
    return AuditEntry(
        player=player,
        player_type="pitcher",
        positions=["P"],
        slot="IL",
        player_sgp=0.0,
        player_id="roster-" + player.replace(" ", "-"),
    )


def render(entries: list[AuditEntry], **kwargs: object) -> str:
    defaults: dict[str, object] = {
        "team_name": TEAM,
        "effective_date": EFFECTIVE,
        "fraction_remaining": 0.24,
        "ros_snapshot_date": ROS_SNAPSHOT,
        "kv_path": "data/manual.db",
    }
    defaults.update(kwargs)
    return render_audit_report(entries, **defaults)  # type: ignore[arg-type]


def non_ascii(text: str) -> set[str]:
    return {ch for ch in text if ord(ch) > 127}


def block(report: str, start: str, end: str | None = None) -> str:
    """The slice of ``report`` after ``start`` and before ``end``."""
    tail = report.split(start)[1]
    return tail if end is None else tail.split(end)[0]


# --------------------------------------------------------------------------
# ASCII safety -- a single stray glyph kills the run on a cp1252 console.
# --------------------------------------------------------------------------


def test_report_is_ascii_only() -> None:
    report = render(
        [
            make_upgrade("Roster Bat", "Free Agent Bat", total=1.5),
            make_hold("Aaron Judge"),
            make_il("Sandy Alcantara"),
        ],
        roto_standings=[(TEAM, 71.5), ("Other Guys", 68.0)],
    )
    report.encode("ascii")  # raises UnicodeEncodeError if the renderer slipped


def test_renderer_introduces_no_non_ascii_of_its_own() -> None:
    """An accented name passes through; the furniture around it stays ASCII."""
    report = render([make_upgrade(ACCENTED_NAME, "Free Agent Bat", total=1.5)])

    assert ACCENTED_NAME in report
    assert non_ascii(report) <= non_ascii(ACCENTED_NAME)


@pytest.mark.parametrize("banned", BANNED_GLYPHS)
def test_no_banned_pretty_glyphs(banned: str) -> None:
    report = render(
        [make_upgrade("Roster Bat", "Free Agent Bat", total=1.5), make_hold("Aaron Judge")]
    )
    assert banned not in report


def test_arrows_are_ascii_in_the_move_labels() -> None:
    report = render([make_upgrade("Roster Bat", "Free Agent Bat", total=1.5)])

    assert "->" in report


# --------------------------------------------------------------------------
# The no-upgrade cases must read as answers, not as rendering failures.
# --------------------------------------------------------------------------


def test_no_upgrades_renders_as_no_upgrade_not_empty() -> None:
    report = render([make_hold("Aaron Judge"), make_hold("Bobby Witt Jr")])

    assert "no upgrades found" in report
    assert NO_UPGRADE in report
    assert "Aaron Judge" in report
    assert "Bobby Witt Jr" in report


def test_hold_rows_are_separated_from_upgrade_rows() -> None:
    report = render(
        [make_upgrade("Roster Bat", "Free Agent Bat", total=1.5), make_hold("Aaron Judge")]
    )

    moves_at = report.index("RECOMMENDED MOVES")
    hold_at = report.index("HOLD -- NO UPGRADE AVAILABLE")
    assert moves_at < report.index("Free Agent Bat") < hold_at
    assert hold_at < report.index("Aaron Judge")


def test_empty_entries_says_the_roster_came_back_empty() -> None:
    report = render([])

    assert "NO AUDIT ENTRIES" in report
    assert "NOT that the roster is optimal" in report
    report.encode("ascii")


# --------------------------------------------------------------------------
# Provenance banner.
# --------------------------------------------------------------------------


def test_provenance_header_names_the_manual_pipeline() -> None:
    report = render([make_upgrade("Roster Bat", "Free Agent Bat", total=1.5)])

    assert "YAHOO-FREE" in report
    assert "manual (Yahoo-free" in report
    assert "data/manual.db" in report
    assert EFFECTIVE.isoformat() in report
    assert ROS_SNAPSHOT in report
    assert "24.0%" in report
    assert TEAM in report


def test_provenance_header_carries_the_fa_pool_caveat() -> None:
    report = render([make_upgrade("Roster Bat", "Free Agent Bat", total=1.5)])

    assert "projection-derived" in report
    assert "UNKNOWN" in report


def test_missing_kv_path_is_marked_not_blank() -> None:
    report = render([make_hold("Aaron Judge")], kv_path=None)

    assert re.search(r"kv store\s+: --", report) is not None


# --------------------------------------------------------------------------
# The renderer must not compute or reorder.
# --------------------------------------------------------------------------


def test_row_order_follows_the_audit_not_the_renderer() -> None:
    """Entries are printed in the order given, even if that order looks wrong."""
    small = make_upgrade("Small Gain", "FA Small", total=0.25)
    big = make_upgrade("Big Gain", "FA Big", total=3.75)
    report = render([small, big])

    assert report.index("Small Gain") < report.index("Big Gain")


def test_printed_numbers_come_straight_off_the_entry() -> None:
    report = render([make_upgrade("Roster Bat", "Free Agent Bat", total=1.5, sgp_gap=6.3)])

    assert "+1.50" in report  # delta_roto total, as stored
    assert "+6.30" in report  # AuditEntry.gap (SGP gap), as stored
    assert "2.10" in report  # player_sgp
    assert "8.40" in report  # best_fa_sgp
    assert "88%" in report  # band p_positive
    assert "real" in report  # band verdict


def test_absent_band_renders_as_missing_not_zero() -> None:
    entry = make_upgrade("Roster Bat", "Free Agent Bat", total=1.5)
    del entry.candidates[0]["band"]
    report = render([entry])
    moves = block(report, "RECOMMENDED MOVES", "PER-CATEGORY")

    # No fabricated P(helps) or verdict -- the cells must read as missing.
    assert "%" not in moves.replace("P(HELPS)", "")
    assert "--" in moves


# --------------------------------------------------------------------------
# Per-category picture -- which category does the move actually help?
# --------------------------------------------------------------------------


def test_category_columns_show_where_the_move_helps() -> None:
    report = render(
        [
            make_upgrade(
                "Roster Bat",
                "Free Agent Bat",
                total=1.5,
                categories={"HR": 0.5, "SB": 1.0, "AVG": 0.0, "ERA": -0.25},
            )
        ]
    )
    cats = block(report, "PER-CATEGORY IMPACT OF EACH MOVE", "HOLD --")

    assert "+0.50" in cats  # HR
    assert "+1.00" in cats  # SB
    assert "-0.25" in cats  # the ERA cost is shown, not hidden
    for header in ("HR", "SB", "AVG", "ERA"):
        assert header in cats


def test_zero_category_delta_is_a_dot_not_a_blank() -> None:
    report = render(
        [make_upgrade("Roster Bat", "Free Agent Bat", total=1.5, categories={"HR": 0.0})]
    )
    cats = block(report, "PER-CATEGORY IMPACT OF EACH MOVE", "HOLD --")

    assert "+0.00" not in cats
    assert "." in cats


def test_category_section_handles_no_upgrades() -> None:
    report = render([make_hold("Aaron Judge")])
    cats = block(report, "PER-CATEGORY IMPACT OF EACH MOVE", "HOLD --")

    assert "no upgrades found" in cats


# --------------------------------------------------------------------------
# Column sizing and sections.
# --------------------------------------------------------------------------


def test_long_names_are_not_truncated() -> None:
    long_name = "Bartolomeo Featherstonehaugh-Wellington III"
    report = render([make_upgrade(long_name, "Free Agent Bat", total=1.5)])

    assert long_name in report


def test_il_entries_render_in_a_trailing_section() -> None:
    report = render(
        [
            make_upgrade("Roster Bat", "Free Agent Bat", total=1.5),
            make_hold("Aaron Judge"),
            make_il("Sandy Alcantara"),
        ]
    )

    assert report.index("INJURED LIST") > report.index("HOLD -- NO UPGRADE AVAILABLE")
    assert report.index("Sandy Alcantara") > report.index("INJURED LIST")


def test_no_il_players_says_so() -> None:
    report = render([make_upgrade("Roster Bat", "Free Agent Bat", total=1.5)])

    assert "no players on the IL." in report


def test_top_n_caps_the_move_list_and_discloses_the_cap() -> None:
    entries = [make_upgrade(f"Bat {i}", f"FA {i}", total=3.0 - i * 0.5) for i in range(4)]
    report = render(entries, top_n=2)

    assert "showing the top 2 of 4 upgrades." in report
    assert "Bat 0" in report
    assert "Bat 3" not in report


def test_top_n_does_not_cap_the_hold_section() -> None:
    entries = [
        make_upgrade("Bat 0", "FA 0", total=3.0),
        make_hold("Hold One"),
        make_hold("Hold Two"),
    ]
    report = render(entries, top_n=1)

    assert "Hold One" in report
    assert "Hold Two" in report


# --------------------------------------------------------------------------
# Standings block.
# --------------------------------------------------------------------------


def test_standings_block_marks_the_users_team_and_keeps_caller_order() -> None:
    report = render(
        [make_upgrade("Roster Bat", "Free Agent Bat", total=1.5)],
        roto_standings=[(TEAM, 71.5), ("Second Place", 68.0), ("Third Place", 60.0)],
    )
    standings = block(report, "WHERE YOU STAND", "RECOMMENDED MOVES")

    assert f"{TEAM}*" in standings
    assert standings.index(TEAM) < standings.index("Second Place")
    assert standings.index("Second Place") < standings.index("Third Place")
    assert "71.50" in standings
    assert "68.00" in standings


def test_standings_block_shows_the_current_projected_total() -> None:
    report = render([make_upgrade("Roster Bat", "Free Agent Bat", total=1.5)])
    standings = block(report, "WHERE YOU STAND", "RECOMMENDED MOVES")

    assert "71.50" in standings  # delta_roto before_total, as stored


def test_standings_block_without_a_payload_says_so() -> None:
    report = render([make_hold("Aaron Judge")])
    standings = block(report, "WHERE YOU STAND", "RECOMMENDED MOVES")

    assert "no standings payload supplied." in standings


def test_projected_standings_fallback_prints_stored_category_totals() -> None:
    """Without roto points, the block shows the team's stored category totals."""
    from fantasy_baseball.models.standings import (
        CategoryStats,
        ProjectedStandings,
        ProjectedStandingsEntry,
    )

    standings = ProjectedStandings(
        effective_date=EFFECTIVE,
        entries=[
            ProjectedStandingsEntry(
                team_name=TEAM,
                stats=CategoryStats(
                    r=812,
                    hr=241,
                    rbi=799,
                    sb=118,
                    avg=0.267,
                    w=88,
                    k=1402,
                    sv=71,
                    era=3.61,
                    whip=1.18,
                ),
            ),
            ProjectedStandingsEntry(team_name="Other Guys", stats=CategoryStats()),
        ],
    )
    report = render([make_hold("Aaron Judge")], projected_standings=standings)
    standings_block = block(report, "WHERE YOU STAND", "RECOMMENDED MOVES")

    assert "projected end-of-season totals:" in standings_block
    for value in ("812", "241", "0.267", "3.61", "1.18"):
        assert value in standings_block
    for header in ("R", "HR", "RBI", "SB", "AVG", "W", "K", "ERA", "WHIP", "SV"):
        assert header in standings_block
    report.encode("ascii")


def test_projected_standings_missing_team_is_reported_not_silent() -> None:
    from fantasy_baseball.models.standings import (
        CategoryStats,
        ProjectedStandings,
        ProjectedStandingsEntry,
    )

    standings = ProjectedStandings(
        effective_date=EFFECTIVE,
        entries=[ProjectedStandingsEntry(team_name="Other Guys", stats=CategoryStats())],
    )
    report = render([make_hold("Aaron Judge")], projected_standings=standings)

    assert "is not in the projected standings payload" in report


# --------------------------------------------------------------------------
# Monte Carlo and category risk.
# --------------------------------------------------------------------------


def make_monte_carlo(
    *,
    first_pct: float = 53.4,
    category_risk: list[dict[str, object]] | None = None,
    scenario: str | None = "rest_of_season",
) -> dict[str, object]:
    """A payload shaped like ``season_data.format_monte_carlo_for_display``.

    Percentages are in percentage POINTS, exactly as the simulation stores
    them -- ``53.4`` means 53.4%, not 5340%.
    """
    risk = category_risk
    if risk is None:
        risk = [
            {
                "cat": "HR",
                "median_pts": 9.0,
                "p10": 8.0,
                "p90": 10.0,
                "first_pct": 29.4,
                "top3_pct": 99.8,
            },
            {
                "cat": "SB",
                "median_pts": 4.0,
                "p10": 1.0,
                "p90": 8.0,
                "first_pct": 2.5,
                "top3_pct": 11.0,
            },
        ]
    payload: dict[str, object] = {
        "teams": [
            {
                "name": TEAM,
                "median_pts": 95.0,
                "p10": 92.0,
                "p90": 98.0,
                "first_pct": first_pct,
                "top3_pct": 99.9,
                "is_user": True,
            },
            {
                "name": "Second Place",
                "median_pts": 77.0,
                "p10": 73.0,
                "p90": 80.0,
                "first_pct": 1.8,
                "top3_pct": 88.0,
                "is_user": False,
            },
        ],
        "category_risk": risk,
    }
    if scenario is not None:
        payload["scenario"] = scenario
    return payload


def test_monte_carlo_section_marks_the_user_and_keeps_caller_order() -> None:
    report = render([make_hold("Aaron Judge")], monte_carlo=make_monte_carlo())
    outlook = block(report, "SEASON OUTLOOK", "CATEGORY RISK")

    assert f"{TEAM}*" in outlook
    assert outlook.index(TEAM) < outlook.index("Second Place")
    assert "95.0" in outlook
    assert "92.0" in outlook and "98.0" in outlook


def test_monte_carlo_names_the_scenario_it_read() -> None:
    """base and rest_of_season answer different questions; the page must say which."""
    report = render([make_hold("Aaron Judge")], monte_carlo=make_monte_carlo())

    assert "scenario: rest_of_season" in block(report, "SEASON OUTLOOK", "CATEGORY RISK")


def test_an_unnamed_scenario_is_marked_missing_not_guessed() -> None:
    report = render([make_hold("Aaron Judge")], monte_carlo=make_monte_carlo(scenario=None))
    outlook = block(report, "SEASON OUTLOOK", "CATEGORY RISK")

    assert f"scenario: {MISSING}" in outlook
    assert "rest_of_season" not in outlook


def test_monte_carlo_percentages_are_not_rescaled() -> None:
    """The cache stores percentage POINTS. Multiplying by 100 again is the bug."""
    report = render([make_hold("Aaron Judge")], monte_carlo=make_monte_carlo(first_pct=53.4))

    assert "53.4%" in report
    assert "5340" not in report


def test_category_risk_rows_carry_the_band_and_both_probabilities() -> None:
    report = render([make_hold("Aaron Judge")], monte_carlo=make_monte_carlo())
    risk = block(report, "CATEGORY RISK", "LINEUP MOVES")

    assert "HR" in risk and "SB" in risk
    assert "29.4%" in risk and "99.8%" in risk
    # The wide-band category's own numbers, not the tight one's.
    assert "1.0" in risk and "8.0" in risk


def test_category_risk_first_pct_is_flagged_as_tie_inclusive() -> None:
    """It is a different statistic from the team 1st % directly above it."""
    report = render([make_hold("Aaron Judge")], monte_carlo=make_monte_carlo())
    risk = block(report, "CATEGORY RISK", "LINEUP MOVES")

    assert "TIE-INCLUSIVE" in risk


def test_missing_monte_carlo_payload_is_announced_not_dropped() -> None:
    """A silently absent section reads as 'nothing to say', which is not true."""
    report = render([make_hold("Aaron Judge")])

    assert "SEASON OUTLOOK" in report
    assert "CATEGORY RISK" in report
    assert "no Monte Carlo payload supplied" in block(report, "SEASON OUTLOOK", "CATEGORY RISK")
    assert "no Monte Carlo payload supplied" in block(report, "CATEGORY RISK", "LINEUP MOVES")


def test_empty_category_risk_names_the_cause() -> None:
    """category_risk is {} when the user team was absent from the simulated league."""
    report = render([make_hold("Aaron Judge")], monte_carlo=make_monte_carlo(category_risk=[]))
    risk = block(report, "CATEGORY RISK", "LINEUP MOVES")

    assert "check team_name" in risk


def test_a_missing_monte_carlo_number_renders_as_missing_not_zero() -> None:
    payload = make_monte_carlo()
    teams = payload["teams"]
    assert isinstance(teams, list)
    teams[0].pop("first_pct")
    report = render([make_hold("Aaron Judge")], monte_carlo=payload)
    outlook = block(report, "SEASON OUTLOOK", "CATEGORY RISK")

    user_row = next(line for line in outlook.split("\n") if f"{TEAM}*" in line)
    assert MISSING in user_row
    assert "0.0%" not in user_row


def test_monte_carlo_sections_stay_ascii() -> None:
    report = render(
        [make_upgrade(ACCENTED_NAME, "Free Agent Bat", total=1.5)], monte_carlo=make_monte_carlo()
    )

    assert non_ascii(report) <= non_ascii(ACCENTED_NAME)
    for glyph in BANNED_GLYPHS:
        assert glyph not in report
