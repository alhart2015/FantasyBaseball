# TODO — Bugs, Flaws & Improvements

## Bugs

- [x] **League key substring match** (`yahoo_auth.py:46`) — `str(league_id) in lid` falsely matches partial IDs. Should split on `.l.` and compare exactly.
- [x] **No warnings for ERA/WHIP/AVG** (`balance.py:56-63`) — `get_warnings()` skips all rate stats. A 5.50 ERA or .220 AVG produces no warning.
- [x] **Dashboard "Hide drafted" toggle is dead** (`dashboard.html:163`) — `available_players` already excludes drafted players in the state JSON, so the JS filter never matches.
- [x] **Roster grid shows "Filled" not player names** (`dashboard.html:226`) — The `text` variable is computed but unused; every slot just says "Filled".
- [x] **`scale_by_schedule()` defined but never called** (`run_lineup.py:36-58`) — Dead code. Lineup optimizer uses raw full-season projections with no weekly adjustment.
- [x] **Two-way player projections broken** (`run_lineup.py:117-134`) — Players with hitter+pitcher positions get added to both lists with the same (first-matched) projection. Pitching value is lost.
- [x] **DH-only players have inflated VAR** (`var.py:25-28`, `replacement.py:17-18`) — No replacement level for UTIL/DH, so `best_var = total_sgp` with nothing subtracted.
- [x] **`_handle_user_pick` temporarily corrupts tracker state** (`run_draft.py:167-170`) — Mutates `current_pick` to peek ahead, then restores. An exception between mutation and restore permanently breaks the tracker.
- [x] **`get_filled_positions` uses exact name match** (`recommender.py:65`) — Uses raw `name` instead of `name_normalized`. Accented names (José Ramírez) can fail to match.

## Design Flaws

- [ ] **Leverage only looks at team above** (`leverage.py:29-32`) — Ignores teams below that could catch you. Should consider both neighbors for attack/defense leverage.
- [ ] **Silent exception swallowing in Yahoo API calls** (`yahoo_players.py:19`, `yahoo_roster.py:83`) — Network errors, auth failures, rate limits all disappear silently. Should at minimum log.
- [ ] **Hardcoded constants that should come from config** — `STARTERS_PER_POSITION`, `ROSTER_SLOTS` in recommender, and dashboard JS `ROSTER_SLOTS` are all hardcoded instead of derived from league config.
- [ ] **State JSON includes ALL available players on every 2s poll** (`state.py:25-46`) — Large payload for 300+ player pools. Board data is mostly static; should send once and use deltas.
- [ ] **No SRI hash on CDN-loaded htmx** (`dashboard.html:8`) — Missing `integrity` and `crossorigin` attributes on the unpkg script tag.
- [ ] **No validation on projection directory existence** — Neither script checks if `data/projections/` exists before building the board. Errors surface as opaque pandas exceptions.
- [ ] **`CategoryBalance` defaults ERA/WHIP to 0.0 with no pitchers** (`balance.py:42-43`) — Shows "perfect" 0.00 ERA during early draft rounds. Should display N/A or be visually distinguished.
