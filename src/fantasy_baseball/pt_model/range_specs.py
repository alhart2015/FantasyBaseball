"""The player types ``ranges`` makes outcome ranges for (#505): what each one's playing
time is, what each stat is counted over, and what his PT range reads."""

from __future__ import annotations

from fantasy_baseball.pt_model.ranges import HISTORY_SEASONS, Spec, Stat

HITTER = Spec(
    name="hitter",
    volume="pa",
    trials={"ab": "ab_pa"},
    stats=(
        Stat("r", "pa", binomial=True, counting=True, numerator=("r",)),
        Stat("hr", "pa", binomial=True, counting=True, numerator=("hr",)),
        Stat("rbi", "pa", binomial=True, counting=True, numerator=("rbi",)),
        Stat("sb", "pa", binomial=True, counting=True, numerator=("sb",)),
        # An AVG draw needs 30 AB.
        Stat("avg", "ab", binomial=True, counting=False, numerator=("h",), min_trials=30.0),
    ),
    features=(
        "rank_pa",
        "age",
        "catcher",
        "dh",
        *(f"{c}{k}" for k in range(1, HISTORY_SEASONS + 1) for c in ("g", "pa")),
    ),
    never_blank=("rank_pa", "catcher", "dh"),
    history_columns=("g", "pa"),
    # Steamer's 150+ PA projections were about 390 a season in 2022-2026, the population
    # the method was validated on.
    top_n=390,
    bands=(0.0, 300.0, 450.0, 550.0, float("inf")),
    # The saturating PT scale's ceiling, over the most PA in the training rows. Held out
    # 2023-2026 (PA pinball loss, P10 / P50 / P90; predictions of a 700+ median and the
    # real PA's share under them): straight 27.5 / 59.3 / 22.9, 13 at 92% (Steamer PT)
    # and 28.0 / 64.0 / 25.6, 33 at 94% (last season's PA); 2.0: 27.9 / 59.3 / 21.7, 2 and
    # 28.5 / 63.1 / 23.7, 8. 1.02-1.5 squeezed the 620-660 PA medians low (37-46% under);
    # 3.0 scored as 2.0 but let 700+ medians back (6 and 15, 83-87% under).
    headroom=2.0,
    # A past season needs 50 PA for its rate miss to join the luck pool.
    min_rate_volume=50.0,
)
