# Trade-for-Future-Pick Calculator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-31-trade-pick-calculator-design.md`

**Goal:** A CLI + reusable library that shows the two-sided impact of trading a current player for a next-year draft pick -- this-year win% / top-3% / per-category deltas from re-running the ROS Monte Carlo, and the marginal VAR value of the extra pick.

**Architecture:** A pure library `analysis/trade_pick.py` builds the trade scenario (remove sent player + replacement filler on your side; intact sent player + worst-of-type drop on the partner's side), runs the existing ROS Monte Carlo twice with common random numbers, and diffs the results; a second half looks up the pick's expected VAR on the frozen 2026 draft-day par curve at the post-keeper-round ordinal. A thin CLI `scripts/trade_pick_calc.py` loads stored Upstash state and renders an ASCII report. One new public helper `draft_value.projected_par_curve` exposes the par curve.

**Tech Stack:** Python 3.11+, numpy, pandas, existing modules: `simulation.run_ros_monte_carlo`, `mc_roster.build_effective_rosters`, `analysis/injury_stress` (`load_mc_inputs_from_upstash`, `McInputs`, `_replacement_ros`), `analysis/draft_value` (`ParCurve`, `build_par_curve`, board reconstruction), `models/player`.

## Global Constraints

- **ASCII-only** in all source, format strings, and report text (Windows cp1252 stdout). No true minus, em/en dash, smart quotes, sigma, arrows. Use `-`, `--`, `"`, `'`, `->`.
- **Player IDs are `name::player_type`.** Match/disambiguate by normalized name plus type; never key on bare name where a collision is possible.
- **No `x or default` for numeric values.** Use `x if x is not None else default` (a real `0.0` is valid data).
- **Reuse before writing.** Use the existing MC (`run_ros_monte_carlo`), effective-roster builder, replacement-line machinery, and par-curve construction. Do not re-implement any of them.
- **MC determinism:** baseline and scenario runs share the same `seed` (common random numbers) so the delta isolates the trade, not MC noise.
- **Defaults:** `n_iter = 2000`, `seed = 42`.
- **End-of-effort gate (Task 7):** `pytest` (named subset), `ruff check .`, `ruff format --check .`, `vulture`, `mypy` if any touched file is under `[tool.mypy].files`.

---

### Task 1: `projected_par_curve` helper in draft_value.py

**Files:**
- Modify: `src/fantasy_baseball/analysis/draft_value.py` (add one public function near `build_par_curve`, ~line 505)
- Test: `tests/test_analysis/test_draft_value.py` (append)

**Interfaces:**
- Consumes: existing `reproduce_draft_day_board`, `_frozen_var_by_player_id`, `_anchor_board_var_to_frozen`, `reconstruct_draft`, `_board_index`, `_assign_pick_types`, `build_par_curve`, module constants `_CONFIG`, `_DRAFT_STATE`, `LeagueConfig`, `load_config`, `ParCurve` (all already in the module).
- Produces: `projected_par_curve(config: LeagueConfig | None = None) -> ParCurve` -- the preseason (f=1) drafted par curve, `run_draft_value` left unmodified.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis/test_draft_value.py`:

```python
def test_projected_par_curve_matches_inline_construction():
    import json
    import math

    from fantasy_baseball.analysis import draft_value as dv
    from fantasy_baseball.config import load_config

    par = dv.projected_par_curve()
    # non-empty and sorted descending (par_for_slot(k) = k-th best drafted VAR)
    assert len(par.drafted_pars) > 0
    assert par.drafted_pars == sorted(par.drafted_pars, reverse=True)

    # cross-check: the standalone helper reproduces the exact same curve the
    # metric builds inline, guarding against setup-sequence drift (anchor before
    # board index, fraction=1.0, keepers passed to _assign_pick_types).
    config = load_config(dv._CONFIG)
    board, _scale = dv.reproduce_draft_day_board(config)
    board = dv._anchor_board_var_to_frozen(board, dv._frozen_var_by_player_id())
    state = json.loads(dv._DRAFT_STATE.read_text(encoding="utf-8"))
    picks = dv.reconstruct_draft(config, state=state)
    bindex = dv._board_index(board)
    typed = dv._assign_pick_types(picks, bindex, config.keepers)
    expected = dv.build_par_curve(typed, bindex, fraction=1.0)

    assert par.drafted_pars == expected.drafted_pars
    assert (math.isnan(par.keeper_par) and math.isnan(expected.keeper_par)) or (
        par.keeper_par == expected.keeper_par
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_draft_value.py::test_projected_par_curve_matches_inline_construction -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'projected_par_curve'`.

- [ ] **Step 3: Write minimal implementation**

In `src/fantasy_baseball/analysis/draft_value.py`, add immediately after `build_par_curve` (after ~line 505):

```python
def projected_par_curve(config: LeagueConfig | None = None) -> ParCurve:
    """Preseason (full-season, f=1) drafted par curve, standalone.

    Reproduces exactly the setup `run_draft_value` runs before its
    `par_proj = build_par_curve(typed_picks, bindex, fraction=1.0)`:
    reproduce the draft-day board, anchor its VAR to the frozen draft-day
    board (`draft_state_board.json`), reconstruct the draft, index the board,
    and assign a player_type to every pick. `run_draft_value` itself is left
    unchanged (no behavior-preserving refactor); this repeats the setup so a
    consumer that needs ONLY the projected par curve (the trade-for-pick
    calculator) does not also pay for scoring every pick.

    Pure over local files -- projection CSVs, `player_positions.json`,
    `draft_state_board.json`, `draft_state.json`, `league.yaml` -- no KV/Upstash.
    Raises (via `_anchor_board_var_to_frozen`) if the frozen board is missing,
    rather than ship a par curve on the drifted rebuilt VAR.
    """
    if config is None:
        config = load_config(_CONFIG)
    board, _scale = reproduce_draft_day_board(config)
    board = _anchor_board_var_to_frozen(board, _frozen_var_by_player_id())
    state = json.loads(_DRAFT_STATE.read_text(encoding="utf-8"))
    picks = reconstruct_draft(config, state=state)
    bindex = _board_index(board)
    typed_picks = _assign_pick_types(picks, bindex, config.keepers)
    return build_par_curve(typed_picks, bindex, fraction=1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_draft_value.py::test_projected_par_curve_matches_inline_construction -v`
Expected: PASS. (Integration-style: reads committed data; deterministic.)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py
git commit -m "feat(trade-pick): add draft_value.projected_par_curve standalone helper"
```

---

### Task 2: Next-year pick value in trade_pick.py

**Files:**
- Create: `src/fantasy_baseball/analysis/trade_pick.py`
- Test: `tests/test_analysis/test_trade_pick.py`

**Interfaces:**
- Consumes: `draft_value.projected_par_curve` (Task 1), `draft_value.ParCurve`, `config.LeagueConfig`, `config.load_config`.
- Produces:
  - `keeper_rounds_for(config: LeagueConfig) -> int`
  - `pick_ordinal_range(nominal_round: int, keeper_rounds: int, num_teams: int, curve_len: int, pick_slot: str = "round") -> tuple[int, int]` (1-based inclusive)
  - `@dataclass(frozen=True) NextYearValue` with fields `nominal_round:int, keeper_rounds:int, drafted_round:int, pick_slot:str, expected_var:float, early_var:float, keeper_par:float, ordinal_lo:int, ordinal_hi:int`
  - `pick_value(par: ParCurve, nominal_round: int, keeper_rounds: int, num_teams: int, pick_slot: str = "round") -> NextYearValue`
  - `next_year_value(config: LeagueConfig, nominal_round: int, pick_slot: str = "round") -> NextYearValue`

- [ ] **Step 1: Write the failing test**

Create `tests/test_analysis/test_trade_pick.py`:

```python
import math

import pytest

from fantasy_baseball.analysis.draft_value import ParCurve
from fantasy_baseball.analysis.trade_pick import (
    NextYearValue,
    pick_ordinal_range,
    pick_value,
)


def _curve(n=200):
    # Strictly descending so mean-over-a-range is order-sensitive and testable.
    return ParCurve(drafted_pars=[float(n - i) for i in range(n)], keeper_par=18.0)


def test_ordinal_range_round_2_is_11_to_20():
    # nominal 5, 3 keeper rounds, 10 teams -> drafted round 2 -> ordinals 11..20
    assert pick_ordinal_range(5, 3, 10, 200) == (11, 20)


def test_ordinal_range_first_drafted_round():
    assert pick_ordinal_range(4, 3, 10, 200) == (1, 10)


def test_ordinal_range_keeper_round_rejected():
    with pytest.raises(ValueError, match="keeper round"):
        pick_ordinal_range(3, 3, 10, 200)


def test_ordinal_range_beyond_curve_rejected():
    with pytest.raises(ValueError, match="beyond the par curve"):
        pick_ordinal_range(60, 3, 10, 200)  # drafted round 57 -> lo far past 200


def test_ordinal_range_clamps_upper_bound():
    # drafted round 20 -> ordinals 191..200; a 195-long curve clamps hi to 195.
    assert pick_ordinal_range(23, 3, 10, 195) == (191, 195)


def test_ordinal_range_early_mid_late_partition_the_round():
    lo_e, hi_e = pick_ordinal_range(5, 3, 10, 200, "early")
    lo_m, hi_m = pick_ordinal_range(5, 3, 10, 200, "mid")
    lo_l, hi_l = pick_ordinal_range(5, 3, 10, 200, "late")
    assert lo_e == 11 and hi_e < 20  # early starts at the round's top
    assert hi_l == 20 and lo_l > 11  # late ends at the round's bottom
    assert lo_e <= lo_m <= lo_l and hi_e <= hi_m <= hi_l


def test_pick_value_round_average_and_early_higher():
    par = _curve()
    nv = pick_value(par, 5, 3, 10, "round")
    assert isinstance(nv, NextYearValue)
    assert nv.drafted_round == 2
    assert nv.ordinal_lo == 11 and nv.ordinal_hi == 20
    # mean of par_for_slot(11..20) = mean of drafted_pars[10..19] = mean(190..181) = 185.5
    assert nv.expected_var == pytest.approx(185.5)
    # early third (higher VAR) exceeds the full-round average on a descending curve
    assert nv.early_var > nv.expected_var
    assert nv.keeper_par == 18.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_trade_pick.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fantasy_baseball.analysis.trade_pick'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/fantasy_baseball/analysis/trade_pick.py`:

```python
"""Trade-for-future-pick calculator.

Two separate views of trading a current player for a next-year draft pick:
the this-year ROS Monte Carlo impact (win% / top-3% / per-category) and the
next-year marginal value of the extra pick (VAR at the post-keeper-round draft
ordinal). See docs/superpowers/specs/2026-07-31-trade-pick-calculator-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from fantasy_baseball.analysis.draft_value import ParCurve, projected_par_curve
from fantasy_baseball.config import LeagueConfig


def keeper_rounds_for(config: LeagueConfig) -> int:
    """Number of keeper rounds = keepers per team = len(keepers) // num_teams.

    Requires an even split (every team keeps the same number, which is what
    "the first K rounds are keeper rounds" means). A non-divisible count breaks
    the nominal-to-drafted-round mapping, so fail loud rather than truncate.
    """
    nk = len(config.keepers)
    nt = config.num_teams
    if nt <= 0 or nk == 0 or nk % nt != 0:
        raise ValueError(
            f"Cannot derive keeper rounds: {nk} keepers / {nt} teams is not an "
            "even split. The nominal-to-drafted-round mapping needs a fixed "
            "keeper-rounds count."
        )
    return nk // nt


def pick_ordinal_range(
    nominal_round: int,
    keeper_rounds: int,
    num_teams: int,
    curve_len: int,
    pick_slot: str = "round",
) -> tuple[int, int]:
    """1-based inclusive par-curve ordinal range for a nominal pick round.

    drafted_round = nominal_round - keeper_rounds; the drafted round spans
    ordinals [(drafted_round-1)*num_teams + 1, drafted_round*num_teams] in the
    VAR-sorted par curve. `pick_slot` narrows to the top ("early"), middle
    ("mid"), or bottom ("late") third of that span. The upper bound is clamped
    to `curve_len`; a range entirely beyond the curve is an error.
    """
    drafted_round = nominal_round - keeper_rounds
    if drafted_round < 1:
        raise ValueError(
            f"Round {nominal_round} is a keeper round; drafted picks start at "
            f"round {keeper_rounds + 1}."
        )
    lo = (drafted_round - 1) * num_teams + 1
    hi = drafted_round * num_teams
    if lo > curve_len:
        raise ValueError(
            f"drafted round {drafted_round} (ordinals {lo}-{hi}) is beyond the "
            f"par curve ({curve_len} picks)."
        )
    hi = min(hi, curve_len)
    if pick_slot == "round":
        return lo, hi
    if pick_slot not in ("early", "mid", "late"):
        raise ValueError("pick_slot must be one of: round, early, mid, late")
    span = hi - lo + 1
    third = max(1, span // 3)
    if pick_slot == "early":
        return lo, lo + third - 1
    if pick_slot == "late":
        return hi - third + 1, hi
    # mid: the middle third, clamped to stay non-empty inside [lo, hi]
    mlo = min(lo + third, hi)
    mhi = max(hi - third, mlo)
    return mlo, mhi


@dataclass(frozen=True)
class NextYearValue:
    nominal_round: int
    keeper_rounds: int
    drafted_round: int
    pick_slot: str
    expected_var: float
    early_var: float
    keeper_par: float
    ordinal_lo: int
    ordinal_hi: int


def _mean_over(par: ParCurve, lo: int, hi: int) -> float:
    vals = [par.par_for_slot(k) for k in range(lo, hi + 1)]
    return sum(vals) / len(vals)


def pick_value(
    par: ParCurve,
    nominal_round: int,
    keeper_rounds: int,
    num_teams: int,
    pick_slot: str = "round",
) -> NextYearValue:
    """Expected pick VAR = mean par over the (narrowed) drafted-round ordinals.

    Also reports an "early in the round" VAR (top third) for context and the
    keeper-average par (`par.keeper_par`, may be NaN).
    """
    curve_len = len(par.drafted_pars)
    lo, hi = pick_ordinal_range(nominal_round, keeper_rounds, num_teams, curve_len, pick_slot)
    elo, ehi = pick_ordinal_range(nominal_round, keeper_rounds, num_teams, curve_len, "early")
    return NextYearValue(
        nominal_round=nominal_round,
        keeper_rounds=keeper_rounds,
        drafted_round=nominal_round - keeper_rounds,
        pick_slot=pick_slot,
        expected_var=_mean_over(par, lo, hi),
        early_var=_mean_over(par, elo, ehi),
        keeper_par=par.keeper_par,
        ordinal_lo=lo,
        ordinal_hi=hi,
    )


def next_year_value(
    config: LeagueConfig, nominal_round: int, pick_slot: str = "round"
) -> NextYearValue:
    """Build the projected par curve and value a nominal-round pick on it."""
    par = projected_par_curve(config)
    return pick_value(par, nominal_round, keeper_rounds_for(config), config.num_teams, pick_slot)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_trade_pick.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/trade_pick.py tests/test_analysis/test_trade_pick.py
git commit -m "feat(trade-pick): next-year pick value (keeper-round ordinal + par lookup)"
```

---

### Task 3: Trade scenario construction (filler, worst-of-type, roster swap)

**Files:**
- Modify: `src/fantasy_baseball/analysis/trade_pick.py` (append)
- Test: `tests/test_analysis/test_trade_pick.py` (append)

**Interfaces:**
- Consumes: `injury_stress.McInputs`, `injury_stress._replacement_ros`, `simulation._replacement_line`, `models.player` (`Player`, `HitterStats`, `PitcherStats`, `PlayerType`, `RankInfo`), `models.positions.Position`, `utils.name_utils.normalize_name`, `sgp.player_value.calculate_player_sgp`.
- Produces:
  - `find_sent_player(roster: list[Player], name: str, player_type: str | None = None) -> Player`
  - `build_replacement_filler(sent: Player) -> Player`
  - `worst_of_type(roster: list[Player], ptype: PlayerType, denoms) -> Player | None`
  - `build_trade_scenario(inputs: McInputs, sent: Player, partner: str) -> dict[str, list[Player]]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis/test_trade_pick.py` (reuse the injury-stress fixtures by importing them):

```python
from fantasy_baseball.analysis.trade_pick import (  # noqa: E402  (grouped with the later imports)
    build_replacement_filler,
    build_trade_scenario,
    find_sent_player,
    worst_of_type,
)
from fantasy_baseball.models.player import HitterStats, PlayerType  # noqa: E402
from fantasy_baseball.models.positions import Position  # noqa: E402


def _hit(name, *, r=90, hr=30, rbi=95, sb=12, h=165, ab=560, pa=620, g=155):
    line = {"r": r, "hr": hr, "rbi": rbi, "sb": sb, "h": h, "ab": ab, "pa": pa, "g": g}
    from fantasy_baseball.models.player import Player

    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[Position.OF],
        rest_of_season=HitterStats.from_dict(line),
        full_season_projection=HitterStats.from_dict(line),
    )


def test_find_sent_player_normalized_and_ambiguity():
    roster = [_hit("Julio Rodriguez"), _hit("Someone Else")]
    assert find_sent_player(roster, "julio rodriguez").name == "Julio Rodriguez"
    with pytest.raises(ValueError, match="not on"):
        find_sent_player(roster, "Nobody Here")


def test_replacement_filler_is_neutralized_and_renamed():
    star = _hit("Julio Rodriguez")
    filler = build_replacement_filler(star)
    assert filler.name != star.name
    assert filler.name.startswith("Replacement")
    assert filler.positions == star.positions  # can fill the vacated slot
    # Both lines neutralized below the star's real production (r/hr/rbi drop).
    for col in ("r", "hr", "rbi"):
        assert getattr(filler.rest_of_season, col) < getattr(star.rest_of_season, col)
        assert getattr(filler.full_season_projection, col) < getattr(
            star.full_season_projection, col
        )


def test_worst_of_type_picks_lowest_projection():
    from fantasy_baseball.sgp.denominators import get_sgp_denominators

    good = _hit("Good")
    bad = _hit("Bad", r=30, hr=2, rbi=25, sb=1, h=80, ab=400, pa=440, g=110)
    worst = worst_of_type([good, bad], PlayerType.HITTER, get_sgp_denominators(None))
    assert worst.name == "Bad"


def test_build_trade_scenario_keeps_sizes_and_moves_player():
    from test_injury_stress import _synth_inputs  # reuse the 2-team fixture

    inputs = _synth_inputs()
    user = inputs.user_team_name
    partner = "Opp"
    sent = find_sent_player(inputs.team_rosters[user], "Star")
    n_user0 = len(inputs.team_rosters[user])
    n_partner0 = len(inputs.team_rosters[partner])

    scen = build_trade_scenario(inputs, sent, partner)

    # user size unchanged: lost Star, gained exactly one filler
    assert len(scen[user]) == n_user0
    assert all(p.name != "Star" for p in scen[user])
    assert sum(p.name.startswith("Replacement") for p in scen[user]) == 1
    # partner size unchanged: gained the intact Star, dropped its worst hitter
    assert len(scen[partner]) == n_partner0
    assert any(p is sent for p in scen[partner])
    # inputs.team_rosters is not mutated
    assert any(p.name == "Star" for p in inputs.team_rosters[user])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_trade_pick.py -k "filler or worst_of_type or scenario or find_sent" -v`
Expected: FAIL with `ImportError: cannot import name 'build_replacement_filler'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/fantasy_baseball/analysis/trade_pick.py`. First extend the import block at the top of the file:

```python
import dataclasses

from fantasy_baseball.analysis.injury_stress import McInputs, _replacement_ros
from fantasy_baseball.models.player import (
    HitterStats,
    PitcherStats,
    Player,
    PlayerType,
    RankInfo,
)
from fantasy_baseball.simulation import _replacement_line
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.name_utils import normalize_name
```

Then append the functions:

```python
def find_sent_player(
    roster: list[Player], name: str, player_type: str | None = None
) -> Player:
    """Locate the player being traded away, by normalized (accent-safe) name.

    When two roster players normalize to the same name and differ in type (a
    hitter and a pitcher), `player_type` disambiguates; without it, an ambiguous
    match is an error rather than a silent pick.
    """
    target = normalize_name(name)
    matches = [p for p in roster if normalize_name(p.name) == target]
    if player_type is not None:
        want = PlayerType(player_type)
        matches = [p for p in matches if p.player_type == want]
    if not matches:
        raise ValueError(f"{name!r} is not on your roster.")
    if len(matches) > 1:
        raise ValueError(
            f"{name!r} is ambiguous on your roster; pass --player-type hitter|pitcher."
        )
    return matches[0]


def build_replacement_filler(sent: Player) -> Player:
    """A replacement-level filler for the slot the sent player vacates.

    Distinct name (never aliases the real player now on the partner), the sent
    player's positions (so it is eligible for the vacated slot), and BOTH its
    ROS line and full-season line neutralized to replacement level -- the MC
    reads production off the ROS line (ROS-direct engine) but the full-season
    line still drives the playing-time-curve shape and the top-k fallback, so
    neutralize both. The active-lineup selection starts this filler only when no
    better bench player is available; otherwise it benches.
    """
    is_hitter = sent.player_type == PlayerType.HITTER
    ros_repl = _replacement_ros(sent)  # scaled to the sent player's ROS volume
    repl_line = _replacement_line(sent.to_flat_dict_full_season(), is_hitter)
    stats_cls = HitterStats if is_hitter else PitcherStats
    fs_repl = stats_cls.from_dict(repl_line)  # replacement full-season line
    pos_label = str(sent.positions[0]) if sent.positions else str(sent.player_type)
    return dataclasses.replace(
        sent,
        name=f"Replacement ({pos_label})",
        rest_of_season=ros_repl,
        full_season_projection=fs_repl,
        preseason=None,
        current=None,
        rank=RankInfo(),
        selected_position=None,
        fg_id=None,
        mlbam_id=None,
        yahoo_id=None,
    )


def worst_of_type(roster: list[Player], ptype: PlayerType, denoms) -> Player | None:
    """The lowest full-season-projected player of `ptype`, or None if none exist.

    Ranked by full-season SGP so the partner drops a benched scrub (second-order)
    to fit the acquired star, keeping the partner's roster size constant. Players
    without a full-season projection are skipped (they cannot be scored).
    """
    cands = [
        p for p in roster if p.player_type == ptype and p.full_season_projection is not None
    ]
    if not cands:
        return None
    return min(
        cands, key=lambda p: calculate_player_sgp(p.full_season_projection, denoms=denoms)
    )


def build_trade_scenario(
    inputs: McInputs, sent: Player, partner: str
) -> dict[str, list[Player]]:
    """Team rosters after the trade: user loses `sent` and gains a replacement
    filler (size constant); the partner gains the intact `sent` and drops its
    worst-of-type player (size constant). `inputs.team_rosters` is not mutated.
    """
    user = inputs.user_team_name
    filler = build_replacement_filler(sent)
    new_user = [p for p in inputs.team_rosters[user] if p is not sent] + [filler]

    partner_roster = inputs.team_rosters[partner]
    drop = worst_of_type(partner_roster, sent.player_type, inputs.denoms)
    new_partner = [p for p in partner_roster if p is not drop] + [sent]

    scenario = dict(inputs.team_rosters)
    scenario[user] = new_user
    scenario[partner] = new_partner
    return scenario
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_trade_pick.py -k "filler or worst_of_type or scenario or find_sent" -v`
Expected: PASS. (The `from test_injury_stress import _synth_inputs` works because both test files live in `tests/test_analysis/`.)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/trade_pick.py tests/test_analysis/test_trade_pick.py
git commit -m "feat(trade-pick): scenario construction (replacement filler + symmetric roster sizing)"
```

---

### Task 4: This-year Monte Carlo impact

**Files:**
- Modify: `src/fantasy_baseball/analysis/trade_pick.py` (append)
- Test: `tests/test_analysis/test_trade_pick.py` (append)

**Interfaces:**
- Consumes: `mc_roster.build_effective_rosters`, `simulation.run_ros_monte_carlo`, `utils.constants.ALL_CATEGORIES`, `build_trade_scenario` (Task 3).
- Produces:
  - `@dataclass(frozen=True) CategoryDelta` (`category:str, base_first:float, new_first:float, base_top3:float, new_top3:float`)
  - `@dataclass(frozen=True) ThisYearImpact` (`base_win:float, new_win:float, base_top3:float, new_top3:float, categories:list[CategoryDelta], n_iter:int, seed:int`)
  - `run_scenario(inputs: McInputs, team_rosters: dict[str, list[Player]], n_iter: int, seed: int) -> dict`
  - `this_year_impact(inputs: McInputs, sent: Player, partner: str, *, n_iter: int = 2000, seed: int = 42) -> ThisYearImpact`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis/test_trade_pick.py`:

```python
def test_this_year_impact_star_to_rival_does_not_help_you():
    from test_injury_stress import _synth_inputs

    from fantasy_baseball.analysis.trade_pick import find_sent_player, this_year_impact

    inputs = _synth_inputs()
    sent = find_sent_player(inputs.team_rosters[inputs.user_team_name], "Star")
    # small n_iter for speed; common random numbers keep the delta meaningful
    impact = this_year_impact(inputs, sent, "Opp", n_iter=300, seed=42)

    # trading your star to the only rival must not raise your win% by more than a
    # 1.0 pt fixed-seed tolerance band (expected direction is a decrease).
    assert impact.new_win <= impact.base_win + 1.0
    # all 10 categories reported
    assert len(impact.categories) == 10
    assert {c.category for c in impact.categories} == {
        "R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"
    }
    assert impact.n_iter == 300 and impact.seed == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_trade_pick.py::test_this_year_impact_star_to_rival_does_not_help_you -v`
Expected: FAIL with `ImportError: cannot import name 'this_year_impact'`.

- [ ] **Step 3: Write minimal implementation**

Extend the import block at the top of `src/fantasy_baseball/analysis/trade_pick.py`:

```python
from fantasy_baseball.mc_roster import build_effective_rosters
from fantasy_baseball.simulation import run_ros_monte_carlo
from fantasy_baseball.utils.constants import ALL_CATEGORIES
```

Append the functions:

```python
@dataclass(frozen=True)
class CategoryDelta:
    category: str
    base_first: float
    new_first: float
    base_top3: float
    new_top3: float


@dataclass(frozen=True)
class ThisYearImpact:
    base_win: float
    new_win: float
    base_top3: float
    new_top3: float
    categories: list[CategoryDelta]
    n_iter: int
    seed: int


def run_scenario(
    inputs: McInputs, team_rosters: dict[str, list[Player]], n_iter: int, seed: int
) -> dict:
    """Rebuild effective rosters for `team_rosters` and run the ROS Monte Carlo.

    eos_baseline / team_sds are held fixed (reused from `inputs`), mirroring the
    injury stress-test: the first-order roster change flows through the rebuilt
    effective rosters and the MC scoring, while the league-context scaffolding
    stays constant so the baseline-vs-scenario delta is controlled.
    """
    eff = build_effective_rosters(
        team_rosters,
        inputs.eos_baseline,
        inputs.team_sds,
        inputs.fraction_remaining,
        denoms=inputs.denoms,
    )
    return run_ros_monte_carlo(
        team_rosters=team_rosters,
        actual_standings=inputs.actual_standings,
        fraction_remaining=inputs.fraction_remaining,
        h_slots=inputs.h_slots,
        p_slots=inputs.p_slots,
        user_team_name=inputs.user_team_name,
        n_iterations=n_iter,
        seed=seed,
        effective_rosters=eff,
    )


def this_year_impact(
    inputs: McInputs,
    sent: Player,
    partner: str,
    *,
    n_iter: int = 2000,
    seed: int = 42,
) -> ThisYearImpact:
    """Baseline vs post-trade ROS MC for the user, with common random numbers.

    Full swing: the sent player leaves the user (replaced by a bench-or-
    replacement filler) and joins the partner. Reports the user's overall win%
    and top-3% and per-category first%/top-3%.
    """
    user = inputs.user_team_name
    base = run_scenario(inputs, inputs.team_rosters, n_iter, seed)
    scen_rosters = build_trade_scenario(inputs, sent, partner)
    scen = run_scenario(inputs, scen_rosters, n_iter, seed)

    br = base["team_results"][user]
    sr = scen["team_results"][user]
    bcat = base["category_risk"]
    scat = scen["category_risk"]
    categories = [
        CategoryDelta(
            category=c.value,
            base_first=bcat[c.value]["first_pct"],
            new_first=scat[c.value]["first_pct"],
            base_top3=bcat[c.value]["top3_pct"],
            new_top3=scat[c.value]["top3_pct"],
        )
        for c in ALL_CATEGORIES
    ]
    return ThisYearImpact(
        base_win=br["first_pct"],
        new_win=sr["first_pct"],
        base_top3=br["top3_pct"],
        new_top3=sr["top3_pct"],
        categories=categories,
        n_iter=n_iter,
        seed=seed,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_trade_pick.py::test_this_year_impact_star_to_rival_does_not_help_you -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/trade_pick.py tests/test_analysis/test_trade_pick.py
git commit -m "feat(trade-pick): this-year ROS Monte Carlo impact (win%/top3%/per-category)"
```

---

### Task 5: Orchestration + ASCII report

**Files:**
- Modify: `src/fantasy_baseball/analysis/trade_pick.py` (append)
- Test: `tests/test_analysis/test_trade_pick.py` (append)

**Interfaces:**
- Consumes: `injury_stress.load_mc_inputs_from_upstash`, `config.load_config`, `this_year_impact` (Task 4), `next_year_value` (Task 2), `find_sent_player` (Task 3).
- Produces:
  - `@dataclass(frozen=True) TradePickResult` (`sent_name:str, partner:str, this_year:ThisYearImpact, next_year:NextYearValue`)
  - `resolve_partner(team_rosters: dict[str, list[Player]], to: str, user: str) -> str`
  - `compute_trade_pick(send:str, to:str, pick_round:int, *, player_type:str|None=None, pick_slot:str="round", n_iter:int=2000, seed:int=42, config_path=None) -> TradePickResult`
  - `render_report(result: TradePickResult) -> str`

- [ ] **Step 1: Write the failing test** (render is pure -> unit-testable without Upstash)

Append to `tests/test_analysis/test_trade_pick.py`:

```python
def test_resolve_partner_normalizes_and_rejects_self_and_unknown():
    from fantasy_baseball.analysis.trade_pick import resolve_partner

    rosters = {"Hart of the Order": [], "SkeleThor": []}
    assert resolve_partner(rosters, "skelethor", "Hart of the Order") == "SkeleThor"
    with pytest.raises(ValueError, match="yourself"):
        resolve_partner(rosters, "Hart of the Order", "Hart of the Order")
    with pytest.raises(ValueError, match="not a team"):
        resolve_partner(rosters, "Nonexistent", "Hart of the Order")


def test_render_report_is_ascii_and_sign_aware():
    from fantasy_baseball.analysis.trade_pick import (
        CategoryDelta,
        NextYearValue,
        ThisYearImpact,
        TradePickResult,
        render_report,
    )

    cats = [CategoryDelta(c, 30.0, 28.0, 80.0, 78.0) for c in
            ("R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP")]
    loss = TradePickResult(
        sent_name="Julio Rodriguez",
        partner="SkeleThor",
        this_year=ThisYearImpact(62.1, 57.9, 91.0, 88.2, cats, 2000, 42),
        next_year=NextYearValue(5, 3, 2, "round", 4.2, 5.1, 18.4, 11, 20),
    )
    out = render_report(loss)
    assert out.isascii()
    assert "Julio Rodriguez" in out and "SkeleThor" in out
    assert "give up" in out  # loss framing (base_win > new_win)

    gain = TradePickResult(
        sent_name="Spare Part",
        partner="SkeleThor",
        this_year=ThisYearImpact(50.0, 50.5, 80.0, 80.0, cats, 2000, 42),
        next_year=NextYearValue(5, 3, 2, "round", 4.2, 5.1, 18.4, 11, 20),
    )
    assert "roughly neutral" in render_report(gain)  # non-negative delta framing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_trade_pick.py -k "resolve_partner or render_report" -v`
Expected: FAIL with `ImportError: cannot import name 'render_report'`.

- [ ] **Step 3: Write minimal implementation**

Extend the import block at the top of `src/fantasy_baseball/analysis/trade_pick.py`:

```python
from pathlib import Path

from fantasy_baseball.analysis.injury_stress import load_mc_inputs_from_upstash
from fantasy_baseball.config import load_config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO_ROOT / "config" / "league.yaml"
```

Append:

```python
@dataclass(frozen=True)
class TradePickResult:
    sent_name: str
    partner: str
    this_year: ThisYearImpact
    next_year: NextYearValue


def resolve_partner(team_rosters: dict[str, list[Player]], to: str, user: str) -> str:
    """Resolve the trade partner's team name (normalized match). Rejects the
    user's own team and an unknown name (listing valid teams)."""
    target = normalize_name(to)
    if normalize_name(user) == target:
        raise ValueError("You cannot trade to yourself; pick a different --to team.")
    for name in team_rosters:
        if normalize_name(name) == target:
            return name
    valid = ", ".join(sorted(t for t in team_rosters if t != user))
    raise ValueError(f"{to!r} is not a team in this league. Valid partners: {valid}")


def compute_trade_pick(
    send: str,
    to: str,
    pick_round: int,
    *,
    player_type: str | None = None,
    pick_slot: str = "round",
    n_iter: int = 2000,
    seed: int = 42,
    config_path: Path | None = None,
) -> TradePickResult:
    """Load stored state, compute both halves, and return the combined result."""
    cfg_path = config_path or _CONFIG_PATH
    inputs = load_mc_inputs_from_upstash(cfg_path)
    config = load_config(cfg_path)
    partner = resolve_partner(inputs.team_rosters, to, inputs.user_team_name)
    sent = find_sent_player(inputs.team_rosters[inputs.user_team_name], send, player_type)
    this_year = this_year_impact(inputs, sent, partner, n_iter=n_iter, seed=seed)
    nxt = next_year_value(config, pick_round, pick_slot)
    return TradePickResult(sent.name, partner, this_year, nxt)


def _kp(x: float) -> str:
    """Keeper-par render: 'n/a' for NaN, else one-decimal VAR."""
    return "n/a" if x != x else f"{x:.1f}"


def render_report(result: TradePickResult) -> str:
    ty = result.this_year
    ny = result.next_year
    dwin = ty.new_win - ty.base_win
    dtop3 = ty.new_top3 - ty.base_top3
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("TRADE-FOR-PICK CALCULATOR")
    lines.append(f"  Send: {result.sent_name}  ->  {result.partner}")
    lines.append(
        f"  For:  2027 Round {ny.nominal_round} pick  "
        f"(keeper rounds: {ny.keeper_rounds}  ->  drafted round {ny.drafted_round})"
    )
    lines.append("=" * 72)

    lines.append("")
    lines.append(f"1. THIS YEAR WITHOUT {result.sent_name}  (full swing: joins {result.partner})")
    lines.append("-" * 72)
    lines.append(f"  Win%   : {ty.base_win:5.1f}%  ->  {ty.new_win:5.1f}%   ({dwin:+.1f})")
    lines.append(f"  Top-3% : {ty.base_top3:5.1f}%  ->  {ty.new_top3:5.1f}%   ({dtop3:+.1f})")
    lines.append("")
    lines.append("  Per-category odds (your team):")
    lines.append(
        f"    {'Cat':<5}{'1st base':>9}{'1st new':>9}{'d1st':>7}"
        f"{'top3 base':>11}{'top3 new':>10}{'dTop3':>8}"
    )
    for c in ty.categories:
        d1 = c.new_first - c.base_first
        d3 = c.new_top3 - c.base_top3
        lines.append(
            f"    {c.category:<5}{c.base_first:>9.1f}{c.new_first:>9.1f}{d1:>+7.1f}"
            f"{c.base_top3:>11.1f}{c.new_top3:>10.1f}{d3:>+8.1f}"
        )

    lines.append("")
    lines.append(f"2. NEXT YEAR -- extra 2027 pick (drafted round {ny.drafted_round})")
    lines.append("-" * 72)
    lines.append(f"  Expected pick value : ~{ny.expected_var:.1f} VAR")
    lines.append(
        f"  (early in the round : ~{ny.early_var:.1f} VAR ; "
        f"keeper-average keeper : ~{_kp(ny.keeper_par)} VAR)"
    )
    lines.append("  VAR is value above a replacement roster spot, so this is roughly the")
    lines.append("  pick's marginal roto-point value next year.")
    lines.append("  Estimate = the 2026 draft-day value distribution at that slot; the")
    lines.append("  specific 2027 player is unknown.")

    lines.append("")
    lines.append("-" * 72)
    if dwin < 0:
        lines.append(
            f"You give up ~{abs(dwin):.1f} win% / ~{abs(dtop3):.1f} top-3% this year "
            f"to gain a pick worth ~{ny.expected_var:.1f} VAR."
        )
    else:
        lines.append(
            f"This year is roughly neutral to positive ({dwin:+.1f} win% / "
            f"{dtop3:+.1f} top-3%), and you also gain a pick worth ~{ny.expected_var:.1f} VAR."
        )
    lines.append(
        f"MC: n_iter={ty.n_iter}, seed={ty.seed} (common random numbers across both runs)."
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_trade_pick.py -k "resolve_partner or render_report" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/trade_pick.py tests/test_analysis/test_trade_pick.py
git commit -m "feat(trade-pick): orchestration + sign-aware ASCII report"
```

---

### Task 6: CLI script

**Files:**
- Create: `scripts/trade_pick_calc.py`
- Test: `tests/test_analysis/test_trade_pick.py` (append a `--help` smoke test)

**Interfaces:**
- Consumes: `analysis.trade_pick.compute_trade_pick`, `analysis.trade_pick.render_report`.
- Produces: an executable CLI. No new library symbols.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis/test_trade_pick.py`:

```python
def test_cli_help_runs(tmp_path):
    import subprocess
    import sys

    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    script = root / "scripts" / "trade_pick_calc.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert result.returncode == 0
    assert "--send" in result.stdout and "--pick-round" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_trade_pick.py::test_cli_help_runs -v`
Expected: FAIL (script does not exist -> non-zero return / FileNotFound).

- [ ] **Step 3: Write minimal implementation**

Create `scripts/trade_pick_calc.py`:

```python
#!/usr/bin/env python
"""CLI for the trade-for-future-pick calculator.

Usage:
    python scripts/trade_pick_calc.py --send "Julio Rodriguez" --to "SkeleThor" --pick-round 5

Reads stored (last-refresh) state from Upstash; run a dashboard refresh first if
the state is stale. See docs/superpowers/specs/2026-07-31-trade-pick-calculator-design.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Scripts inject src/ into sys.path (repo convention) rather than relying solely
# on the editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Player names may carry non-ASCII (e.g. accents); reconfigure stdout so a name
# from data does not crash the report on Windows cp1252.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fantasy_baseball.analysis.trade_pick import compute_trade_pick, render_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show the two-sided impact of trading a player for a next-year draft pick."
    )
    parser.add_argument("--send", required=True, help="Player you trade away (on your roster).")
    parser.add_argument("--to", required=True, help="Trade partner's team name.")
    parser.add_argument(
        "--pick-round",
        type=int,
        required=True,
        help="Nominal round of the pick you receive (keeper rounds are subtracted).",
    )
    parser.add_argument(
        "--player-type",
        choices=["hitter", "pitcher"],
        default=None,
        help="Disambiguate when two same-named players are on your roster.",
    )
    parser.add_argument(
        "--pick-slot",
        choices=["round", "early", "mid", "late"],
        default="round",
        help="Narrow the pick's value within the drafted round (default: round average).",
    )
    parser.add_argument("--iterations", type=int, default=2000, help="MC iterations (default 2000).")
    parser.add_argument("--seed", type=int, default=42, help="MC seed (default 42).")
    args = parser.parse_args()

    try:
        result = compute_trade_pick(
            send=args.send,
            to=args.to,
            pick_round=args.pick_round,
            player_type=args.player_type,
            pick_slot=args.pick_slot,
            n_iter=args.iterations,
            seed=args.seed,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_trade_pick.py::test_cli_help_runs -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/trade_pick_calc.py tests/test_analysis/test_trade_pick.py
git commit -m "feat(trade-pick): CLI script scripts/trade_pick_calc.py"
```

---

### Task 7: End-of-effort verification gate

**Files:** none (verification only).

- [ ] **Step 1: Run the trade-pick + touched-module tests**

Run: `pytest tests/test_analysis/test_trade_pick.py tests/test_analysis/test_draft_value.py tests/test_analysis/test_injury_stress.py -v`
Expected: all PASS. (Named subset: the new module, the modified draft_value, and the injury-stress fixtures reused by the tests.)

- [ ] **Step 2: Lint**

Run: `ruff check .`
Expected: zero violations. Fix any (e.g. unused imports, import order) in the files touched this plan.

- [ ] **Step 3: Format check**

Run: `ruff format --check .`
Expected: no drift. If it reports the new files, run `ruff format src/fantasy_baseball/analysis/trade_pick.py scripts/trade_pick_calc.py tests/test_analysis/test_trade_pick.py src/fantasy_baseball/analysis/draft_value.py` and re-commit.

- [ ] **Step 4: Dead-code check**

Run: `vulture`
Expected: no NEW findings for `analysis/trade_pick.py` or `scripts/trade_pick_calc.py`. Pre-existing unrelated findings are acceptable -- call them out.

- [ ] **Step 5: Types (conditional)**

Check whether any touched file is under `[tool.mypy].files` in `pyproject.toml` (`analysis/trade_pick.py`, `analysis/draft_value.py`, `scripts/trade_pick_calc.py`). If any is listed, run `mypy` and fix findings in touched files. If none is listed, state that mypy coverage does not include these files and skip.

- [ ] **Step 6: Real-data smoke (manual, non-gating)**

Run: `python scripts/trade_pick_calc.py --send "Julio Rodriguez" --to "SkeleThor" --pick-round 5`
Expected: a two-section ASCII report. Requires current Upstash state (a prior dashboard refresh). If Upstash is stale/unavailable, note it -- this step is a manual sanity check, not a gated test.

- [ ] **Step 7: Final commit (if any fixes were applied)**

```bash
git add -A
git commit -m "chore(trade-pick): satisfy end-of-effort verification gate"
```

---

## Self-Review

**Spec coverage:**
- This-year MC (win%/top3%/per-category, full swing, filler, fixed baseline, common seed) -> Tasks 3, 4.
- Next-year marginal VAR (par curve, keeper-round ordinal, round-average + early/mid/late, keeper-par context) -> Tasks 1, 2.
- Standalone `projected_par_curve` (run_draft_value untouched, cross-check test) -> Task 1.
- Distinct filler name + both lines neutralized -> Task 3 (`build_replacement_filler`, test asserts rename + both lines drop).
- Partner drops worst-of-type, sizes constant -> Task 3 (`worst_of_type`, `build_trade_scenario`, size-invariant test).
- Sign-aware framing -> Task 5 (`render_report`, both-branch test).
- CLI + flags (`--send/--to/--pick-round/--player-type/--pick-slot/--iterations/--seed`), sys.path inject, stdout reconfigure, ASCII -> Task 6.
- Edge cases: keeper-round rejected (Task 2 test), beyond-curve rejected + clamp (Task 2 test), ambiguous/absent sent player (Task 3 test), unknown/self partner (Task 5 test), NaN keeper-par -> `_kp` (Task 5), Upstash-missing -> propagates from `load_mc_inputs_from_upstash` (Task 5 via CLI try/except in Task 6), frozen-board-missing -> propagates from `_anchor_board_var_to_frozen` (Task 1).
- Testing + verification gate -> Task 7.

No gaps found.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every test shows real assertions. Clear.

**Type consistency:** `McInputs`, `Player`, `ParCurve`, `NextYearValue`, `ThisYearImpact`, `CategoryDelta`, `TradePickResult` names and fields are used identically across tasks. `category_risk[cat.value]` / `team_results[user]` keys match `run_ros_monte_carlo`'s documented return. `build_effective_rosters` / `run_ros_monte_carlo` call shapes match `injury_stress.win_pct`. Consistent.
