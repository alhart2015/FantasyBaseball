from dataclasses import dataclass


@dataclass(frozen=True)
class Pick:
    """A single completed pick in the draft."""

    name: str
    player_id: str
    is_user: bool


class DraftTracker:
    """Track state of a snake draft."""

    def __init__(self, num_teams: int, user_position: int, rounds: int = 22):
        self.num_teams = num_teams
        self.user_position = user_position
        self.rounds = rounds
        self.current_pick = 1
        self.picks: list[Pick] = []

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

    @property
    def picks_until_next_turn(self) -> int:
        """Count opponent picks between now and the user's next turn.

        Unlike ``picks_until_user_turn`` (which returns 0 when it IS
        the user's pick), this always looks *forward* from the current
        pick to find the next user pick after this one.  This is the
        value VONA needs: how many opponents pick before you pick again.

        In a 10-team snake with user at position 8:
        - During round 1 pick 8 (user's turn): opponents make picks
          9, 10, 11, 12 before user picks again at 13 -> returns 4.
        - During round 2 pick 13 (user's turn): opponents make 14
          picks before user picks again at 28 -> returns 14.
        """
        count = 0
        temp_pick = self.current_pick + 1
        while temp_pick <= self.total_picks:
            temp_round = (temp_pick - 1) // self.num_teams + 1
            temp_pos = (temp_pick - 1) % self.num_teams + 1
            if temp_round % 2 == 1:
                team = temp_pos
            else:
                team = self.num_teams - temp_pos + 1
            if team == self.user_position:
                return count
            count += 1
            temp_pick += 1
        return count

    # ------------------------------------------------------------------
    # Backward-compat parallel-list accessors.  Prefer iterating
    # ``self.picks`` directly in new code.
    # ------------------------------------------------------------------

    @property
    def drafted_players(self) -> list[str]:
        return [p.name for p in self.picks]

    @property
    def drafted_ids(self) -> list[str]:
        return [p.player_id for p in self.picks]

    @property
    def user_roster(self) -> list[str]:
        return [p.name for p in self.picks if p.is_user]

    @property
    def user_roster_ids(self) -> list[str]:
        return [p.player_id for p in self.picks if p.is_user]

    def advance(self) -> None:
        self.current_pick += 1

    def draft_player(self, name: str, is_user: bool = False, player_id: str | None = None) -> None:
        self.picks.append(
            Pick(
                name=name,
                player_id=player_id or name,
                is_user=is_user,
            )
        )
