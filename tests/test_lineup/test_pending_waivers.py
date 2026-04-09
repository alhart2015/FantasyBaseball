from fantasy_baseball.web.season_data import adjust_for_pending_moves
from fantasy_baseball.models.player import Player, PlayerType, HitterStats


def _make_player(name, ptype="hitter"):
    return Player(
        name=name,
        player_type=PlayerType(ptype),
        positions=["OF"] if ptype == "hitter" else ["SP"],
        ros=HitterStats(r=50, hr=15, rbi=45, sb=5, h=100, ab=400)
            if ptype == "hitter" else None,
    )


def _make_pending(team, adds=None, drops=None):
    return {
        "team": team,
        "adds": [{"name": n, "positions": []} for n in (adds or [])],
        "drops": [{"name": n, "positions": []} for n in (drops or [])],
    }


class TestAdjustForPendingMoves:
    def test_removes_claimed_players_from_fa_pool(self):
        fa = [_make_player("Otto Lopez"), _make_player("Juan Soto")]
        pending = [_make_pending("Any Team", adds=["Otto Lopez"])]
        _, filtered_fa = adjust_for_pending_moves([], fa, pending, "My Team")
        names = [p.name for p in filtered_fa]
        assert "Otto Lopez" not in names
        assert "Juan Soto" in names

    def test_removes_user_pending_drops_from_roster(self):
        roster = [_make_player("Marcus Semien"), _make_player("Juan Soto")]
        pending = [_make_pending("My Team", drops=["Marcus Semien"])]
        filtered_roster, _ = adjust_for_pending_moves(
            roster, [], pending, "My Team"
        )
        names = [p.name for p in filtered_roster]
        assert "Marcus Semien" not in names
        assert "Juan Soto" in names

    def test_no_pending_moves_returns_unchanged(self):
        roster = [_make_player("Player A")]
        fa = [_make_player("Player B")]
        adj_roster, adj_fa = adjust_for_pending_moves(
            roster, fa, [], "My Team"
        )
        assert [p.name for p in adj_roster] == ["Player A"]
        assert [p.name for p in adj_fa] == ["Player B"]

    def test_other_team_drops_not_removed_from_user_roster(self):
        roster = [_make_player("Marcus Semien")]
        pending = [_make_pending("Other Team", drops=["Marcus Semien"])]
        adj_roster, _ = adjust_for_pending_moves(
            roster, [], pending, "My Team"
        )
        assert len(adj_roster) == 1

    def test_claims_from_all_teams_removed_from_fa(self):
        fa = [_make_player("Player A"), _make_player("Player B")]
        pending = [
            _make_pending("Team 1", adds=["Player A"]),
            _make_pending("Team 2", adds=["Player B"]),
        ]
        _, adj_fa = adjust_for_pending_moves([], fa, pending, "My Team")
        assert len(adj_fa) == 0
