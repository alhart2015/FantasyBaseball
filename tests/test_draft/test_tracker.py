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

    # --- picks_until_next_turn tests ---

    def test_picks_until_next_turn_round1_position8(self):
        """Position 8 in round 1: 4 opponent picks before next turn.

        Round 1 picks: 1,2,3,4,5,6,7,[8],9,10
        Round 2 picks: 10,9,[8],...
        Opponents between: 9,10,10,9 = 4 picks.
        """
        t = self.make_tracker()
        # Advance to pick 8 (user's turn in round 1)
        for _ in range(7):
            t.advance()
        assert t.current_pick == 8
        assert t.is_user_pick is True
        assert t.picks_until_next_turn == 4

    def test_picks_until_next_turn_round2_position8(self):
        """Position 8 in round 2: 14 opponent picks before next turn.

        Round 2 picks: 10,9,[8],7,6,5,4,3,2,1
        Round 3 picks: 1,2,3,4,5,6,7,[8],...
        Opponents between: 7+7 = 14 picks.
        """
        t = self.make_tracker()
        # Advance to pick 13 (user's turn in round 2)
        for _ in range(12):
            t.advance()
        assert t.current_pick == 13
        assert t.is_user_pick is True
        assert t.picks_until_next_turn == 14

    def test_picks_until_next_turn_position1(self):
        """Position 1 in round 1: 18 opponent picks before next turn.

        Round 1 picks: [1],2,3,...,10
        Round 2 picks: 10,9,...,2,[1]
        Opponents between: 9+9 = 18 picks.
        """
        t = DraftTracker(num_teams=10, user_position=1, rounds=22)
        assert t.current_pick == 1
        assert t.is_user_pick is True
        assert t.picks_until_next_turn == 18

    def test_picks_until_next_turn_position10(self):
        """Position 10 in round 1: 0 opponent picks (back-to-back).

        Round 1 picks: 1,2,...,9,[10]
        Round 2 picks: [10],9,...,1
        No opponents pick between the two turns.
        """
        t = DraftTracker(num_teams=10, user_position=10, rounds=22)
        # Advance to pick 10 (user's turn in round 1)
        for _ in range(9):
            t.advance()
        assert t.current_pick == 10
        assert t.is_user_pick is True
        assert t.picks_until_next_turn == 0

    def test_picks_until_next_turn_position10_round2(self):
        """Position 10 in round 2: 18 opponent picks before next turn.

        Round 2 picks: [10],9,8,...,2,1
        Round 3 picks: 1,2,...,9,[10]
        Opponents between: 9+9 = 18 picks.
        """
        t = DraftTracker(num_teams=10, user_position=10, rounds=22)
        # Advance to pick 11 (user's turn in round 2, first pick of round)
        for _ in range(10):
            t.advance()
        assert t.current_pick == 11
        assert t.is_user_pick is True
        assert t.picks_until_next_turn == 18

    def test_picks_until_next_turn_not_user_pick(self):
        """When it's not the user's pick, still counts to the NEXT user turn.

        At pick 1 (team 1's turn), user is position 8.
        Next user turn is pick 8, so 7 opponent picks happen (picks 2-8,
        but pick 8 is user's so only picks 2-7 = 6... wait).

        Actually: from pick 1, temp_pick starts at 2. We count non-user
        picks until we hit a user pick.  Picks 2,3,4,5,6,7 are non-user
        (6 picks), pick 8 is user.  So picks_until_next_turn == 6.
        """
        t = self.make_tracker()  # position 8
        assert t.current_pick == 1
        assert t.is_user_pick is False
        # Picks 2,3,4,5,6,7 are opponents, pick 8 is user
        assert t.picks_until_next_turn == 6

    def test_picks_until_next_turn_last_round(self):
        """At the user's last pick in the draft, return remaining picks."""
        t = DraftTracker(num_teams=10, user_position=8, rounds=2)
        # Total picks = 20. User picks at 8 and 13.
        # Advance to pick 13 (user's last pick)
        for _ in range(12):
            t.advance()
        assert t.current_pick == 13
        assert t.is_user_pick is True
        # No more user turns — returns count of remaining picks (7)
        assert t.picks_until_next_turn == 7
