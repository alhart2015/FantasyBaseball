import pytest
from fantasy_baseball.draft.tracker import DraftTracker


class TestDraftTracker:
    def make_tracker(self):
        return DraftTracker(num_teams=10, user_position=8)

    def test_initial_state(self):
        t = self.make_tracker()
        assert t.current_pick == 1
        assert t.current_round == 1
        assert t.picking_team == 1

    def test_round_1_order(self):
        t = self.make_tracker()
        teams = [t.picking_team]
        for _ in range(9):
            t.advance()
            teams.append(t.picking_team)
        assert teams == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_round_2_reverses(self):
        t = self.make_tracker()
        for _ in range(10):
            t.advance()
        assert t.current_round == 2
        assert t.picking_team == 10

    def test_snake_pattern(self):
        t = self.make_tracker()
        for _ in range(10):
            t.advance()
        assert t.picking_team == 10
        for _ in range(9):
            t.advance()
        assert t.picking_team == 1

    def test_is_user_pick(self):
        t = self.make_tracker()
        for _ in range(7):
            t.advance()
        assert t.is_user_pick is True

    def test_is_not_user_pick(self):
        t = self.make_tracker()
        assert t.is_user_pick is False

    def test_picks_until_next_user_turn(self):
        t = self.make_tracker()
        assert t.picks_until_user_turn == 7

    def test_picks_until_next_after_user_picks(self):
        t = self.make_tracker()
        for _ in range(7):
            t.advance()
        assert t.is_user_pick is True
        t.advance()
        assert t.picks_until_user_turn == 4

    def test_user_roster_tracking(self):
        t = self.make_tracker()
        t.draft_player("Juan Soto", is_user=True)
        assert "Juan Soto" in t.user_roster
        assert "Juan Soto" in t.drafted_players

    def test_other_team_draft(self):
        t = self.make_tracker()
        t.draft_player("Random Guy", is_user=False)
        assert "Random Guy" not in t.user_roster
        assert "Random Guy" in t.drafted_players

    def test_total_picks(self):
        t = DraftTracker(num_teams=10, user_position=8, rounds=22)
        assert t.total_picks == 220
