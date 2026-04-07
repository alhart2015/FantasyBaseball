# Perception-Based Trade Recommendations

## Problem

The current trade recommender proposes trades based on mutual wSGP benefit, but opponents don't evaluate trades using wSGP — they look at player rankings, name recognition, and raw stats. The result is that mathematically sound proposals look lopsided to opponents and get rejected.

## Strategy

Exploit the gap between **perceived value** (unweighted SGP rankings, a proxy for how opponents evaluate players) and **actual value** (leverage-weighted wSGP tuned to our standings position). Propose trades where the player we send is roughly equally ranked or better ranked than the player we receive — looks fair or generous to the opponent — but our wSGP gain is maximized because the incoming player fills high-leverage categories.

## Design

### Filter logic (`trades/evaluate.py`)

Replace the `MAX_SGP_GAP` fairness filter and mutual benefit check with a ranking proximity filter:

- Look up both players' unweighted SGP ROS rank — the `ros` field from `RankInfo`, computed by `compute_sgp_rankings()` from blended ROS projections
- **Accept if:** `send_rank - receive_rank <= MAX_RANK_GAP` where `MAX_RANK_GAP = 5`. The player we send can be ranked up to 5 spots worse than what we receive. Sending a better-ranked player (negative gap) is always accepted.
- **Keep:** `hart_wsgp_gain > 0` — no point proposing trades that don't help us
- **Keep:** roster legality check (incoming player must fill a non-bench active slot) and position feasibility checks
- **Remove:** `MAX_SGP_GAP`, `EQUAL_LEVERAGE` constant (only used for the old SGP gap filter), and `opp_wsgp_gain >= 0` mutual benefit requirement

Rankings are already computed during the refresh pipeline. They get passed into `find_trades()` as a new parameter so the filter can use them. The ranking attachment to trade results (currently done after `find_trades()` in `season_data.py`) moves into `find_trades()` itself since it's now needed during filtering.

### Sort order

- **Primary:** `hart_wsgp_gain` descending — biggest actual benefit to us comes first
- **Tiebreaker:** `rank_gap` ascending (where `rank_gap = send_rank - receive_rank`) — between equal-benefit trades, prefer the one where we send the better-ranked player (more generous-looking, more likely accepted)

### Trade pitch (`trades/pitch.py`)

Rewrite `generate_pitch()` to focus on ranking and positional value instead of category analysis.

**Template:** 1-2 sentences that:
1. Always mention the ranking comparison (the hook — opponent sees they're getting equal or better value)
2. Add a short positional justification when players play different positions

**Examples:**
- "Sending you the #42 overall player for your #47 — I need a SS and you're deep there."
- "Straight swap — you're getting the #30 ranked player, I'm getting #33. Fills a positional need for both of us."

No category breakdowns, no stats.

**Inputs:** send rank, receive rank, send positions, receive positions.

### Files changed

- **`trades/evaluate.py`** — filter and sort logic, remove `MAX_SGP_GAP`/`EQUAL_LEVERAGE`, add rankings parameter
- **`trades/pitch.py`** — rewrite `generate_pitch()` with ranking-focused template
- **`web/season_data.py`** — pass rankings into `find_trades()`, remove post-hoc rank attachment (moved into `find_trades()`)

### No changes needed

- **Web UI** — already displays ranks, pitch, and wSGP gain
- **Config** — `MAX_RANK_GAP = 5` is a constant in `evaluate.py`, one line to tune
- **Waiver/audit systems** — untouched

## Testing

Unit tests in `tests/test_trades/`:

- **Rank filter accepts** swap at `send_rank - receive_rank = 5`
- **Rank filter rejects** swap at `send_rank - receive_rank = 6`
- **Rank filter accepts** swap where send is better-ranked (negative gap)
- **Rejects** swap with `hart_wsgp_gain <= 0`
- **Sort by wSGP gain** descending given 3 trades with different gains
- **Sort tiebreaker** by rank gap ascending given 2 trades with equal wSGP gain
- **Pitch mentions rankings** and does not contain category analysis
- **Pitch includes positional justification** when players play different positions
- **Roster legality still enforced** even when ranking looks fair

## Extensibility

The system is designed for 1-for-1 trades. Multi-player trades (2-for-1, 2-for-2) can be added later by extending the candidate generation loop. The ranking filter and perception-gap sort generalize naturally — sum of ranks for the package sent vs received.
