from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.web.season_data import adjust_for_pending_moves, find_unprocessed_moves
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


def _make_txn(txn_id, team, add_name=None, drop_name=None,
              add_positions=None, drop_positions=None):
    return {
        "transaction_id": txn_id,
        "type": "add/drop",
        "team": team,
        "team_key": "t.1",
        "add_name": add_name,
        "add_positions": add_positions,
        "drop_name": drop_name,
        "drop_positions": drop_positions,
    }


class TestFindUnprocessedMoves:
    def test_detects_drop_still_on_roster(self):
        """Beeter dropped but still on roster = unprocessed."""
        txns = [_make_txn("59", "My Team",
                          add_name="Michael Wacha", add_positions="SP",
                          drop_name="Clayton Beeter", drop_positions="SP")]
        roster_names = {normalize_name("Clayton Beeter"),
                        normalize_name("Juan Soto")}
        result = find_unprocessed_moves(txns, roster_names, "My Team")
        assert len(result) == 1
        assert result[0]["drops"][0]["name"] == "Clayton Beeter"
        assert result[0]["adds"][0]["name"] == "Michael Wacha"

    def test_ignores_already_processed(self):
        """If dropped player is gone and added player is on roster, skip."""
        txns = [_make_txn("59", "My Team",
                          add_name="Michael Wacha",
                          drop_name="Clayton Beeter")]
        roster_names = {normalize_name("Michael Wacha"),
                        normalize_name("Juan Soto")}
        result = find_unprocessed_moves(txns, roster_names, "My Team")
        assert len(result) == 0

    def test_ignores_other_teams(self):
        txns = [_make_txn("60", "Other Team",
                          add_name="Player A",
                          drop_name="Player B")]
        roster_names = {normalize_name("Player B")}
        result = find_unprocessed_moves(txns, roster_names, "My Team")
        assert len(result) == 0

    def test_add_only_not_on_roster(self):
        txns = [_make_txn("61", "My Team",
                          add_name="New Player", add_positions="OF")]
        roster_names = {normalize_name("Juan Soto")}
        result = find_unprocessed_moves(txns, roster_names, "My Team")
        assert len(result) == 1
        assert result[0]["adds"][0]["name"] == "New Player"
        assert result[0]["drops"] == []

    def test_empty_transactions(self):
        result = find_unprocessed_moves([], set(), "My Team")
        assert result == []
