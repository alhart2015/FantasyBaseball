# Fantasy Baseball Draft Day Guide

## Setup

```bash
git clone https://github.com/alhart2015/FantasyBaseball.git
cd FantasyBaseball
pip install -r requirements.txt
```

## Before Draft Day

1. **Update projections** — Download latest FanGraphs CSVs (Steamer, ZiPS, ATC) into `data/projections/`
2. **Update positions** — Run `python scripts/fetch_positions.py` to get current multi-position eligibility
3. **Verify config** — Check `config/league.yaml` for correct keepers, draft position, and strategy:
   ```yaml
   draft:
     strategy: two_closers    # our strategy
     scoring_mode: var         # value above replacement
     position: 8               # snake draft position
   ```
4. **Verify draft order** — Traded picks are in `config/draft_order.json`. Confirm no new trades happened since last update.

## Running the Draft

```bash
python scripts/run_draft.py
```

This starts:
- The CLI draft assistant (in the terminal)
- A live dashboard at http://127.0.0.1:5000

### What You'll See at Startup

```
League 5652 | Draft position: 8
Team: Hart of the Order
Strategy: two_closers + var
Keepers: 30 players across 10 teams
Draft order loaded: 200 picks, 20 yours, 6 traded picks

Building draft board...
Draft pool: 275 players (after removing 30 keepers)

TOP 25 AVAILABLE PLAYERS
...
```

### During the Draft

The script walks through every pick in order. There are three types of picks:

#### Other Team's Pick
```
======================================================================
ROUND 4 | Pick 31 | Send in the Cavalli
======================================================================

Pick (player, 'team player', or 'mine'): corbin burnes
  -> Drafted: Corbin Burnes
```
Type the player name — fuzzy matching handles partial names and typos.

#### Your Pick
```
======================================================================
ROUND 4 | Pick 38 | Hart of the Order *** YOUR PICK ***
======================================================================

Picks until next turn: 4

RECOMMENDATIONS:
  1. George Kirby (P) VAR: 4.2 score: 5.1
  2. Framber Valdez (P) VAR: 3.8 score: 4.7
  3. CJ Abrams (SS) VAR: 3.5 score: 4.3
  4. Dylan Cease (P) VAR: 3.4 score: 4.1
  5. Josh Naylor (1B) VAR: 3.2 score: 3.9

ROSTER BALANCE:
  R:88 HR:32 RBI:90 SB:25 AVG:.278
  W:0 K:0 SV:0 ERA:N/A WHIP:N/A
```

**To pick: type `1` to take the #1 recommendation**, or type a player name if you want someone else.

#### Traded Picks
```
======================================================================
  !!!!! TRADED PICK !!!!!
  Originally Tortured Baseball Department's pick -> now Hart of the Order's
  !!!!! TRADED PICK !!!!!
ROUND 5 | Pick 42 | Hart of the Order *** YOUR PICK ***
======================================================================
```

You have an extra pick in Round 5 (acquired from TBD) and no pick in Round 18 (traded to TBD).

### Input Reference

| Input | What it does |
|-------|-------------|
| `corbin burnes` | Drafts Corbin Burnes (fuzzy matched) |
| `1`, `2`, etc. | Picks that numbered recommendation (your pick only) |
| `spacemen gausman` | Records Kevin Gausman drafted by Spacemen (useful when a team traded for someone else's pick) |
| `mine` | When it's another team's pick but you own it via trade — switches to your pick mode |
| `skip` | Skip entering a pick (if you missed one) |
| `quit` | Exit the draft |

### Strategy: two_closers + var

The system will recommend picks using VAR (Value Above Replacement) scoring. It guarantees two closers by rounds 8 and 14. Between closer deadlines, it picks the best player available weighted by your category needs.

Watch for these alerts:
- **SV DANGER** — You need a closer, draft one now
- **OPPORTUNISTIC** — A closer has fallen past their ADP, consider grabbing them
- **[LOW AVG]** — This hitter would drag your team AVG below .250

### Tips

- **Trust the recommendations** — they account for positional scarcity, category balance, and VONA urgency
- **Don't panic on pitching early** — the system will front-load SPs if you kept 3 hitters, that's by design
- **Watch the roster balance** — the bottom of each pick shows your projected category totals
- **The dashboard** (http://127.0.0.1:5000) shows the same info in a visual layout — useful on a second monitor

## After the Draft

The draft log is auto-saved to `data/drafts/draft_YYYY-MM-DD_HHMMSS.json`.

### Run Monte Carlo Projections

```bash
python scripts/monte_carlo.py --iterations 1000
```

Estimates win probability and risk profile by simulating injuries and stat variance across 1000 seasons.

## Mock Drafts

Practice without keepers:

```bash
python scripts/run_draft.py --mock
python scripts/run_draft.py --mock --position 3  # try a different draft slot
```

## Running Simulations

```bash
# Single deterministic sim
python scripts/simulate_draft.py --strategy two_closers --scoring var

# Full strategy sweep (deterministic)
python scripts/batch_sweep.py

# Strategy sweep with jitter (stable results, recommended)
python scripts/batch_sweep.py --iterations 30

# ADP-only opponents with jitter
python scripts/sweep_adp_jitter.py
```
