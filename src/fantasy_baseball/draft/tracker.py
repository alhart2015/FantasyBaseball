class DraftTracker:
    """Track state of a snake draft."""

    def __init__(self, num_teams: int, user_position: int, rounds: int = 22):
        self.num_teams = num_teams
        self.user_position = user_position
        self.rounds = rounds
        self.current_pick = 1
        self.drafted_players: list[str] = []
        self.user_roster: list[str] = []

    @property
    def total_picks(self) -> int:
        return self.num_teams * self.rounds

    @property
    def current_round(self) -> int:
        return (self.current_pick - 1) // self.num_teams + 1

    @property
    def pick_in_round(self) -> int:
        return (self.current_pick - 1) % self.num_teams + 1

    @property
    def picking_team(self) -> int:
        pos = self.pick_in_round
        if self.current_round % 2 == 1:
            return pos
        else:
            return self.num_teams - pos + 1

    @property
    def is_user_pick(self) -> bool:
        return self.picking_team == self.user_position

    @property
    def picks_until_user_turn(self) -> int:
        if self.is_user_pick:
            return 0
        temp_pick = self.current_pick
        count = 0
        while temp_pick <= self.total_picks:
            temp_pick += 1
            count += 1
            temp_round = (temp_pick - 1) // self.num_teams + 1
            temp_pos = (temp_pick - 1) % self.num_teams + 1
            if temp_round % 2 == 1:
                team = temp_pos
            else:
                team = self.num_teams - temp_pos + 1
            if team == self.user_position:
                return count
        return count

    def advance(self) -> None:
        self.current_pick += 1

    def draft_player(self, name: str, is_user: bool = False) -> None:
        self.drafted_players.append(name)
        if is_user:
            self.user_roster.append(name)
