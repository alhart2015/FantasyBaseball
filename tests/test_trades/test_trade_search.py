import pytest
from fantasy_baseball.models.player import Player, HitterStats, PitcherStats
from fantasy_baseball.sgp.rankings import rank_key
from fantasy_baseball.trades.evaluate import search_trades_away

ALL_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
_EQUAL_LEVERAGE = {cat: 0.1 for cat in ALL_CATS}

ROSTER_SLOTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1,
                "OF": 4, "UTIL": 2, "P": 9, "BN": 2, "IL": 2}

SAMPLE_STANDINGS = [
    {"name": "Hart", "team_key": "t.1", "rank": 3,
     "stats": {"R": 900, "HR": 280, "RBI": 880, "SB": 120,
               "AVG": .260, "W": 80, "K": 1300, "SV": 80, "ERA": 3.50, "WHIP": 1.15}},
    {"name": "Rival", "team_key": "t.2", "rank": 5,
     "stats": {"R": 850, "HR": 250, "RBI": 870, "SB": 180,
               "AVG": .255, "W": 85, "K": 1400, "SV": 40, "ERA": 3.80, "WHIP": 1.20}},
    {"name": "Rival A", "team_key": "t.3", "rank": 4,
     "stats": {"R": 870, "HR": 260, "RBI": 860, "SB": 140,
               "AVG": .258, "W": 82, "K": 1350, "SV": 50, "ERA": 3.60, "WHIP": 1.18}},
]


def _make_hitter(name, positions, r=70, hr=20, rbi=65, sb=8, avg=.270, ab=500):
    h = int(avg * ab)
    return Player(name=name, player_type="hitter", positions=positions,
                  ros=HitterStats(pa=int(ab * 1.15), ab=ab, h=h,
                                  r=r, hr=hr, rbi=rbi, sb=sb, avg=avg))


def _make_pitcher(name, positions, ip=150, w=9, k=140, sv=0, era=3.80, whip=1.25):
    er = int(era * ip / 9)
    bb = int((whip * ip - ip * 0.8) / 1)
    h_allowed = int(whip * ip - bb)
    return Player(name=name, player_type="pitcher", positions=positions,
                  ros=PitcherStats(ip=ip, w=w, k=k, sv=sv, era=era, whip=whip,
                                   er=er, bb=bb, h_allowed=h_allowed))


class TestSearchTradesAway:
    def test_returns_grouped_by_opponent(self):
        """Results should be a list of opponent groups with 'opponent' and 'candidates' keys."""
        hart_roster = [_make_hitter("Hart OF", ["OF"], hr=15, sb=5)]
        opp_rosters = {
            "Rival": [_make_hitter("Opp OF", ["OF"], hr=25, sb=15)],
            "Rival A": [_make_hitter("Opp A OF", ["OF"], hr=22, sb=12)],
        }
        rankings = {
            rank_key("Hart OF", "hitter"): 55,
            rank_key("Opp OF", "hitter"): 50,
            rank_key("Opp A OF", "hitter"): 52,
        }
        results = search_trades_away(
            player_name="Hart OF",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
            standings=SAMPLE_STANDINGS,
            leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE, "Rival A": _EQUAL_LEVERAGE},
            roster_slots=ROSTER_SLOTS, rankings=rankings,
        )
        assert isinstance(results, list)
        for group in results:
            assert "opponent" in group
            assert "candidates" in group
            assert isinstance(group["candidates"], list)

    def test_player_not_found_returns_empty(self):
        """Searching for a player not on the roster should return empty list."""
        hart_roster = [_make_hitter("Hart OF", ["OF"])]
        results = search_trades_away(
            player_name="Nonexistent Player",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters={},
            standings=SAMPLE_STANDINGS,
            leverage_by_team={"Hart": _EQUAL_LEVERAGE},
            roster_slots=ROSTER_SLOTS, rankings={},
        )
        assert results == []

    def test_candidates_have_required_fields(self):
        """Each candidate should include send, receive, ranks, wSGP gain, and deltas."""
        hart_roster = [_make_hitter("Hart OF", ["OF"], hr=15, sb=5)]
        opp_rosters = {"Rival": [_make_hitter("Opp OF", ["OF"], hr=25, sb=15)]}
        rankings = {
            rank_key("Hart OF", "hitter"): 55,
            rank_key("Opp OF", "hitter"): 50,
        }
        results = search_trades_away(
            player_name="Hart OF",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
            standings=SAMPLE_STANDINGS,
            leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
            roster_slots=ROSTER_SLOTS, rankings=rankings,
        )
        assert len(results) > 0
        candidate = results[0]["candidates"][0]
        for key in ("send", "receive", "send_rank", "receive_rank",
                    "send_positions", "receive_positions",
                    "hart_wsgp_gain", "hart_delta", "opp_delta",
                    "hart_cat_deltas", "opp_cat_deltas"):
            assert key in candidate, f"Missing key: {key}"

    def test_rank_filter_applied(self):
        """Trades where send_rank - receive_rank > 5 should be excluded."""
        hart_roster = [_make_hitter("Hart OF", ["OF"], hr=15, sb=5)]
        opp_rosters = {"Rival": [_make_hitter("Opp OF", ["OF"], hr=25, sb=15)]}
        rankings = {
            rank_key("Hart OF", "hitter"): 60,
            rank_key("Opp OF", "hitter"): 50,
        }
        results = search_trades_away(
            player_name="Hart OF",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
            standings=SAMPLE_STANDINGS,
            leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
            roster_slots=ROSTER_SLOTS, rankings=rankings,
        )
        # rank gap = 10 > 5, should be excluded
        all_candidates = [c for g in results for c in g["candidates"]]
        assert not any(c["receive"] == "Opp OF" for c in all_candidates)

    def test_positional_weakness_included(self):
        """Each opponent group should include a positional_weakness score."""
        hart_roster = [_make_hitter("Hart SS", ["SS"], hr=15, sb=5)]
        opp_rosters = {"Rival": [_make_hitter("Opp SS", ["SS"], hr=25, sb=15)]}
        rankings = {
            rank_key("Hart SS", "hitter"): 55,
            rank_key("Opp SS", "hitter"): 50,
        }
        results = search_trades_away(
            player_name="Hart SS",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
            standings=SAMPLE_STANDINGS,
            leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
            roster_slots=ROSTER_SLOTS, rankings=rankings,
        )
        for group in results:
            assert "positional_weakness" in group
