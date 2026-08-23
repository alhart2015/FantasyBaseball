"""The report must surface FREE lineup swaps, not just add/drops.

Regression guard for a real reporting miss on 2026-08-22: the pipeline had
computed a start/bench swap worth +0.46 roto ("real", p=0.82) and cached it
under CacheKey.LINEUP_OPTIMAL, but the renderer printed only the add/drop
section -- whose best row was +0.29. A reader acting on the report alone would
have made a worse move, paid a waiver claim for it, and never seen the better
one. The add/drop deltas are measured AGAINST the optimal lineup, so omitting
the lineup half does not just hide a move, it misrepresents the rest.
"""

from datetime import date

from fantasy_baseball.manual.report import render_audit_report

MOVES = {
    "swaps": [
        {
            "start": {
                "player": "Bryan Woo",
                "from": "BN",
                "to": "P",
                "roto_delta": 0.4634560860263406,
                "band": {"mean": 0.46, "sd": 0.51, "p_positive": 0.818, "verdict": "real"},
            },
            "bench": {"player": "Yoendrys Gomez", "from": "P", "to": "BN"},
        }
    ],
    "unpaired_starts": [],
    "unpaired_benches": [],
}


def _an_entry():
    """One roster row. render_audit_report short-circuits on an EMPTY entry list
    (that means the roster failed to load, not that it is optimal), so every
    section test needs at least one."""
    from fantasy_baseball.lineup.roster_audit import AuditEntry

    return AuditEntry(
        player="Someone", player_type="pitcher", positions=["P"], slot="P", player_sgp=2.0
    )


def _render(lineup_moves):
    return render_audit_report(
        [_an_entry()],
        team_name="Hart of the Order",
        effective_date=date(2026, 8, 22),
        fraction_remaining=0.2,
        ros_snapshot_date="2026-08-22",
        lineup_moves=lineup_moves,
    )


def _render_with_entries(lineup_moves, entries):
    return render_audit_report(
        entries,
        team_name="Hart of the Order",
        effective_date=date(2026, 8, 22),
        fraction_remaining=0.2,
        ros_snapshot_date="2026-08-22",
        lineup_moves=lineup_moves,
    )


def test_lineup_swap_is_rendered_with_both_players_and_the_delta():
    from fantasy_baseball.lineup.roster_audit import AuditEntry

    entry = AuditEntry(
        player="Someone",
        player_type="pitcher",
        positions=["P"],
        slot="P",
        player_sgp=2.0,
    )
    out = _render_with_entries(MOVES, [entry])
    assert "Bryan Woo" in out
    assert "Yoendrys Gomez" in out
    assert "+0.46" in out
    assert "real" in out


def test_lineup_section_precedes_the_add_drop_section():
    """Free moves first -- an add/drop can be sniped, a lineup change cannot."""
    from fantasy_baseball.lineup.roster_audit import AuditEntry

    entry = AuditEntry(
        player="Someone", player_type="pitcher", positions=["P"], slot="P", player_sgp=2.0
    )
    out = _render_with_entries(MOVES, [entry])
    assert out.index("LINEUP MOVES") < out.index("RECOMMENDED MOVES")


def test_optimal_lineup_says_so_rather_than_printing_an_empty_table():
    out = _render({"swaps": [], "unpaired_starts": [], "unpaired_benches": []})
    assert "already optimal" in out


def test_missing_lineup_data_is_reported_as_missing_not_as_optimal():
    """A cache miss must never read as 'no move helps' -- that is a claim, not a gap."""
    out = _render(None)
    assert "no lineup data available" in out
    assert "already optimal" not in out


def test_report_is_ascii_only_with_a_lineup_section():
    out = _render(MOVES)
    assert out.encode("ascii")
