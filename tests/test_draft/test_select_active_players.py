"""Position-aware active-roster selection in the draft simulator.

``_select_active_players`` must respect roster position slots, not just take
the top-N hitters by counting stats. Otherwise a position-imbalanced roster
(e.g. a dozen OF and one catcher) gets its best bats started regardless of
eligibility -- benching the only catcher and leaving the C slot effectively
unfilled -- which overstates the team's projected production.
"""

import sys
from pathlib import Path


def _sim():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import simulate_draft

    return simulate_draft


def _h(name, positions, r=0, hr=0, rbi=0, sb=0):
    return {
        "name": name,
        "player_type": "hitter",
        "positions": positions,
        "r": r,
        "hr": hr,
        "rbi": rbi,
        "sb": sb,
        "h": 0,
        "ab": 0,
    }


def test_only_catcher_starts_even_when_lowest_value():
    """The sole catcher must occupy the C slot, not be benched behind OF bats."""
    sim = _sim()
    slots = {"C": 1, "OF": 1, "UTIL": 1, "P": 1, "BN": 2}
    catcher = _h("Catcher", ["C"], r=50, hr=10, rbi=50)  # lowest counting total
    of1 = _h("OF1", ["OF"], r=100, hr=40, rbi=100, sb=10)
    of2 = _h("OF2", ["OF"], r=95, hr=35, rbi=95, sb=8)
    of3 = _h("OF3", ["OF"], r=90, hr=30, rbi=90, sb=6)

    active_h, _ = sim._select_active_players([catcher, of1, of2, of3], [], slots)
    names = {h["name"] for h in active_h}

    assert "Catcher" in names  # only catcher must start (fills C)
    assert len(active_h) == 3  # C + OF + UTIL
    assert "OF3" not in names  # surplus OF benched (no slot left)


def test_unfillable_slot_left_empty():
    """A position with no eligible player stays empty -- the team fields fewer
    active hitters rather than illegally starting an ineligible player.
    """
    sim = _sim()
    slots = {"C": 1, "SS": 1, "OF": 1, "UTIL": 1, "P": 1, "BN": 2}
    of1 = _h("OF1", ["OF"], r=100, hr=40, rbi=100)
    of2 = _h("OF2", ["OF"], r=90, hr=30, rbi=90)

    active_h, _ = sim._select_active_players([of1, of2], [], slots)

    # OF -> OF, OF -> UTIL; C and SS have no eligible player -> empty.
    assert len(active_h) == 2


def test_pitchers_fill_fungible_p_pool_by_value():
    """Pitchers share one fungible P pool; top-N by value still applies."""
    sim = _sim()
    slots = {"OF": 1, "P": 2, "BN": 2}

    def _p(name, w, k, sv=0):
        return {
            "name": name,
            "player_type": "pitcher",
            "positions": ["P"],
            "w": w,
            "k": k,
            "sv": sv,
        }

    ace = _p("Ace", w=15, k=220)
    mid = _p("Mid", w=10, k=160)
    weak = _p("Weak", w=5, k=90)

    _, active_p = sim._select_active_players([], [ace, mid, weak], slots)
    names = {p["name"] for p in active_p}

    assert names == {"Ace", "Mid"}  # top 2 by value, Weak benched
