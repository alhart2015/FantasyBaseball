"""Historical players whose REALIZED career looked like this one's (#358).

Given a player's realized SGP by age, find the historical player-seasons whose own
realized career, over the same window of ages, minimizes RMSE against it -- then show
what those players went on to do.

THIS IS THE BACKWARD MATCH, and it replaced a forward one. `comp_paths.closest_paths`
took the model's already-predicted five-year path and returned the historical seasons
whose realized path landed closest to it. That set is selected ON THE OUTCOME: it is the
prediction redrawn, so it cannot be evidence for the prediction and its spread is not
uncertainty about anything. Measured on Yordan Alvarez, the five stored forward comps
tracked his predicted path at RMSE 1.20-2.04.

What is matched here is a statement about HISTORY, and no predicted value enters it. The
question is "which players actually looked like this one, and what happened to them?",
which is answerable without our model and therefore usable as evidence about it. The
forward paths that come back fan out, and that fan is real.

The two sets are almost disjoint in practice. Measured 2026-08-24 on Alvarez at 29, against
the 2000-2026 hitter panel anchored to the 2026-08-22 rest-of-season snapshot -- which is
THIS code's output, not the issue's prototype's (see `MIN_OVERLAP_FRACTION` for why the two
differ). Forward matching returned Todd Frazier 2015, Edgar Renteria 2005, Howie Kendrick
2013, Evan Longoria 2015 and Melky Cabrera 2014. Backward matching returns:

    Starling Marte  2018  rmse 2.96  16.5 16.0 19.2 13.9  6.7
    Ian Kinsler     2011       3.96  13.9 13.1 16.0 13.2 16.5
    Joey Votto      2013       4.41   4.0 16.6 16.8 16.6  9.4
    Alex Rios       2010       4.61   7.5 17.3 18.6  9.8  5.6
    Carlos Gonzalez 2015       4.98  13.8  8.1 10.1  1.1  0.0
    Adam Dunn       2009       5.02  14.4  2.6 12.6 10.9  8.3

Recognisably the same KIND of player -- high-peak bats with interruption years -- and the
outcomes are the point: Kinsler is still at 16.5 five years on, Gonzalez is out of the
league. That fan is what the old set could not show, because it was picked for agreeing
with the forecast.

RE-MEASURE RATHER THAN TRUST THIS PARAGRAPH. It is pinned to one panel vintage and one
snapshot, and both move.

TWO CONVENTIONS FOR A MISSING SEASON, on purpose. See `Prepared.back` for the full
reasoning; in short, an age the player did not play is dropped from the BACKWARD error
(absence says nothing about the kind of player he was) and is a real 0.0 in the FORWARD
path (absence is exactly what happened to him). Filling the backward side with zeros is
what matched an injured star to a replacement-level journeyman; requiring a played season
on the forward side would quietly drop everyone who washed out and bias the displayed
outcomes optimistic.

Ids, never names: naming needs the people cache, and keeping it out of here is what lets
this be tested against a hand-built `Prepared` with no data files at all.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from fantasy_baseball.trajectory.shape import Prepared

#: Comps per player: what `push_trajectory_board.py` STORES and what the chart's `n`
#: control may ASK FOR. One number, because they are one requirement -- the blob is
#: built hours earlier by another process, so the control can only ever slice what was
#: stored, and a ceiling above the stored count asks for comps that do not exist.
#:
#: Ten, not the five the chart draws by default: every legal `n` has to be servable.
#: It lives HERE, beside `closest_careers`, because this is the module both sides go
#: through; it was defined twice, once per side, and kept honest by a test asserting the
#: two literals were equal.
#:
#: WHAT IT COSTS, re-measured 2026-08-24 because backward matching moved it. The old note
#: quoted "~370 bytes a player over five" and was written for the forward matcher; that
#: figure no longer describes anything. Built at max_horizon=5 against the 2000-2026
#: panels and the 2026-08-22 snapshot, the whole `cache:trajectory_chart_data` blob is
#: **1,739 KB** over 1,290 players -- against **1,515 KB** for the forward-matched blob
#: currently deployed, so +15%.
#:
#: The growth is NOT the two new per-comp fields. It is the deduped `careers` map going
#: 1,216 -> 2,755 arcs: forward comps were selected for landing near one predicted path,
#: so the same few hundred careers served the whole board, while backward comps are drawn
#: from a far wider and less correlated pool and share far less. Comp ENTRIES actually
#: fell, 12,590 -> 10,980, because the `MIN_OVERLAP` floor leaves the very young with
#: none. Well inside what the KV takes -- the deployed blob is already this size -- but
#: it is the number to check before raising this constant, since the careers map is what
#: grows and it grows faster than the comp count.
MAX_COMPS = 10

#: Share of the subject's own realized ages a candidate must also have played to be
#: matched against him at all. Fraction rather than a flat count because the flat count
#: silently excludes the young: 6-of-8 is the right cut for a 29-year-old and is
#: unreachable for a 23-year-old who has only three professional seasons -- and a
#: 23-year-old is precisely the ambiguous keeper call this board exists for.
#:
#: 0.75 reproduces the >= 6 of 8 the #358 prototype used. THE RULE, NOT THE NUMBERS --
#: the shipped comp list is close to the issue's prototype but not equal to it, and
#: claiming parity would send the next reader hunting for a regression that is not there.
#: Six things differ, none of them a drift from a decision anyone made:
#:
#:  1. the approved deviation on survivorship admits candidates the prototype's
#:     `all(a in hist for a in FWD)` dropped, which changes who is in the top ten;
#:  2. the subject's anchor-age value is `player.now`, the ROS-anchored full-season
#:     estimate, where the prototype used the raw partial-season row;
#:  3. the panel here is `load_scored_panel`-filtered (observed, `_in_role`, volume > 0,
#:     `_scale_short_schedules`) where the prototype read the CSV bare;
#:  4. era factors are computed pre-injection by `ros_anchor.load_anchored_panels`, not
#:     `era_normalize(score(raw))`;
#:  5. `build_history` drops each player's first panel season, so it can never anchor;
#:  6. `exclude_id` is explicit here.
MIN_OVERLAP_FRACTION = 0.75

#: ...and the floor under it, in ages. Two, so a "career comp" is never a single point.
#: One shared age is the anchor season alone, which is level matching -- the question
#: `comps.py` asked and #325 retired -- wearing a career comp's label.
#:
#: A HARD floor, not clamped to what the subject has. A player with one realized season
#: has no career to match on, so he gets no comps rather than a level match under a
#: career comp's heading; the page says so in words. Clamping it produced 100% coverage
#: across both 2026 pools, every age from 20 up -- which reads as the feature working and
#: is really the floor being defeated at exactly the ages nobody can check by eye.
MIN_OVERLAP = 2


def required_overlap(subject_ages: int) -> int:
    """How many shared ages a candidate needs, given how many the subject has.

    Scales with the subject rather than being fixed, so a 23-year-old is matched on the
    three seasons he has instead of being refused for not having eight -- but never
    below `MIN_OVERLAP`, which no subject can buy his way under by having played less.
    """
    return max(MIN_OVERLAP, math.ceil(MIN_OVERLAP_FRACTION * subject_ages))


@dataclass(frozen=True)
class CareerComp:
    """One historical player-season whose realized career resembles the subject's."""

    mlbam_id: int
    #: The season AT THE ANCHOR AGE -- the year he was as old as the subject is now.
    #: For labelling; the match spans the years before it and the display spans the
    #: years after.
    season: int
    #: Root mean squared error on the REALIZED career, over the shared ages only. No
    #: predicted value enters it.
    #:
    #: A THIN MATCH SCORES SLIGHTLY BETTER, and nothing here corrects for it: RMSE is a
    #: mean, so fewer shared ages is fewer chances to disagree rather than a smaller
    #: sum. `MIN_OVERLAP_FRACTION` bounds how thin a match can get, and `overlap` is
    #: carried so a reader can see it, which is the honest treatment -- a correction
    #: factor would be invented rather than measured.
    rmse: float
    #: Ages that actually entered `rmse`. Below the subject's own count when the
    #: candidate has a hole in his career, or when the subject's window reaches back
    #: past where the panel begins.
    overlap: int
    #: What happened NEXT: realized SGP at each horizon, ascending, 0.0 for a year he
    #: was out of the league. This is the payload -- the match is only how he was chosen.
    path: tuple[float, ...]


def closest_careers(
    prepared: Prepared,
    career: Mapping[int, float],
    age: int,
    n: int,
    *,
    exclude_id: int | None = None,
) -> list[CareerComp]:
    """The `n` historical careers closest to `career` at `age`, best first.

    `career` is the subject's REALIZED SGP by age -- his complete seasons, plus the
    anchored in-progress one at `age` (see `ros_anchor`). Ages outside the lookback
    window are ignored, so a caller may pass a whole career.

    THE ANCHOR AGE IS NOT PURELY REALIZED, and this is the one place "matched on realized
    career only" needs qualifying. While the base season is in progress the caller passes
    `SweptPlayer.now` for that age: season-to-date PLUS a rest-of-season projection,
    re-scored (#348). Deliberate -- dropping the newest and most decision-relevant year
    would match a player on a career that stops a year before the question being asked,
    and passing his two-thirds-season partial line would read his missed time as decline.
    What matters for #358 is that no value from the SHAPE FIT enters, which is what makes
    the resulting comps independent of the forecast they are drawn against; that is
    pinned by `test_a_comp_is_selected_on_the_career_never_on_the_prediction`. A
    rest-of-season blend is not a trajectory prediction and does not reintroduce the
    circularity. On a COMPLETE base season the distinction vanishes.

    `exclude_id` drops the subject from his own candidate pool. The forward-observability
    mask usually removes him anyway -- his anchor season is the most recent one -- but
    that is a coincidence of the current panel and not a rule, and a player listed as his
    own closest comp at RMSE 0.00 is the kind of wrong that is only ever noticed by
    someone who already distrusts the page.

    Raises:
        ValueError: `career` has no value at `age` itself. Every candidate is selected at
            exactly that age, so with no anchor value there is nothing to anchor to -- and
            a caller reaching this has a construction bug, not an empty result.
    """
    if age not in career:
        raise ValueError(
            f"career carries no value at the anchor age {age} (has {sorted(career)}); "
            "the in-progress season must be supplied by the caller"
        )
    lookback = len(prepared.back)
    horizons = tuple(sorted(prepared.horizons))

    # The subject's side of the comparison: his own realized ages inside the window,
    # newest last. Ages he never played are simply absent, which is the same treatment
    # `back` gives the candidates.
    ages = [a for a in range(age - lookback + 1, age + 1) if a in career]
    subject = np.array([career[a] for a in ages], dtype=float)
    need = required_overlap(len(ages))

    # EXACT age, and every forward year observABLE. `forward` stores a real 0.0 for "out
    # of the league", which is indistinguishable from "has not happened yet" -- so the
    # censoring has to come from the season, not the value. Horizons ascend, so clearing
    # the longest clears them all.
    eligible = (prepared.age == float(age)) & (prepared.season + horizons[-1] <= prepared.last)
    if exclude_id is not None:
        eligible &= prepared.mlbam_id != exclude_id
    candidates = np.flatnonzero(eligible)
    if candidates.size == 0:
        return []

    # One column per shared age, NaN where the candidate has no season there. `age - a`
    # is an age offset and a season offset at once -- a player's age advances by exactly
    # one per season, verified against the shipped panel (0 exceptions in 29,424 rows).
    window = np.column_stack([prepared.back[age - a][candidates] for a in ages])
    diff = window - subject
    matched = ~np.isnan(diff)
    overlap = matched.sum(axis=1)

    keep = overlap >= need
    if not keep.any():
        return []
    candidates, overlap = candidates[keep], overlap[keep]
    # Squared error over the SHARED ages only: a NaN column contributes nothing to the
    # sum and nothing to the divisor, so an absent season neither helps nor hurts.
    squared = np.where(matched[keep], np.nan_to_num(diff[keep], nan=0.0) ** 2, 0.0)
    rmse = np.sqrt(squared.sum(axis=1) / overlap)

    paths = np.column_stack([prepared.forward[h][candidates] for h in horizons])

    # Sorted on the full key, not just rmse: two identical careers would otherwise swap
    # between reads on nothing but row order. `lexsort` reads its keys LAST-IS-PRIMARY,
    # so this tuple is (rmse, mlbam_id, season) priority spelled backwards.
    order = np.lexsort((prepared.season[candidates], prepared.mlbam_id[candidates], rmse))
    return [
        CareerComp(
            mlbam_id=int(prepared.mlbam_id[candidates[i]]),
            season=int(prepared.season[candidates[i]]),
            rmse=float(rmse[i]),
            overlap=int(overlap[i]),
            path=tuple(float(v) for v in paths[i]),
        )
        for i in order[:n]
    ]
