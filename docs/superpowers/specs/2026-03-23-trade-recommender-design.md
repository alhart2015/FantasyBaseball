# Trade Recommender (1-for-1) — Design Spec

## Goal

Propose the top 5 one-for-one trades that improve Hart's team, are realistic for the opponent to accept, and include a human-readable pitch with projected roto point impact.

## Data Sources

- **Yahoo Fantasy API:** All 10 teams' rosters (player names, positions), league standings (cumulative category stats, ranks, team keys)
- **ROS projections:** Blended Steamer/ZiPS/ATC with recency blend (same as lineup optimizer)
- **Leverage:** Per-team category leverage weights computed from standings gaps

## Core Logic

For every possible 1-for-1 swap between Hart and each opponent:

1. **Trade value (is this worth doing?):** Compute wSGP gain for Hart using Hart's leverage, and wSGP gain for the opponent using their leverage. Both sides must gain (or opponent at worst breaks even).

2. **Standings impact (how many roto points change?):** Take each team's current Yahoo standings totals. Subtract the traded player's ROS projection, add the received player's ROS projection. Re-rank all 10 teams in each category. Delta = new roto points minus old roto points. This correctly models that accumulated stats stay — only future production changes.

3. **Roster legality:** After the swap, each team must still be able to fill all required roster slots. In practice this only blocks edge cases like trading away your only catcher.

4. **Pitch generation:** For the opponent, identify their 2-3 weakest categories that improve from the trade and their strongest category that takes the hit. Generate 1-2 sentences framing the trade as "you gain where you need it, you lose where you can afford it."

## Constraints

- 1-for-1 only (multi-player trades are a future TODO)
- No position type restriction — any player for any player as long as rosters remain legal
- Filter to trades where Hart gains positive roto points AND opponent gains positive roto points (or breaks even)
- Rank by Hart's projected roto point gain, break ties by opponent's gain (more realistic trades first)

## Output

Top 5 trades, each showing:

```
1. SEND: Bryson Stott (2B)  →  Springfield Isotopes
   GET:  Emmanuel Clase (RP) ←  Springfield Isotopes

   Hart gains: +4.2 roto pts projected (+6 SV, +2 ERA, -2 HR, -2 RBI)
   They gain:  +1.5 roto pts projected (+2 SB, +1 AVG, -1 SV, -1 ERA)

   Pitch: "You're 8th in SB and 7th in AVG — Stott gives you both.
   You're already 2nd in saves, so trading Clase barely costs you."
```

## Architecture

**Script:** `scripts/run_trades.py` — standalone CLI, similar pattern to `run_lineup.py`

**New module:** `src/fantasy_baseball/trades/recommender.py` — core trade evaluation logic:
- `evaluate_trade(hart_stats, opp_stats, all_standings, hart_loses_ros, hart_gains_ros, opp_loses_ros, opp_gains_ros)` → returns roto point deltas per category for both teams
- `generate_pitch(opp_name, opp_leverage, opp_gains, opp_losses, opp_standings_rank)` → returns 1-2 sentence pitch string
- `find_trades(hart_roster, opp_rosters, projections, standings, leverage_by_team, roster_slots)` → returns ranked list of trade proposals

**Reuses:** `calculate_leverage` from leverage.py, `calculate_weighted_sgp` from weighted_sgp.py, Yahoo API functions from yahoo_roster.py, projection blending from projections.py, recency blend from recency.py.

## Scope

This is a CLI tool run on-demand during the season. No automation or scheduled execution. Output is printed to stdout for the user to review and act on manually via Yahoo.
