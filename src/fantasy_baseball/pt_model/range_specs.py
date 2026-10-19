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
    # Held out 2022-2026 (Steamer PT), quantile loss vs one-size: PA -4.0%, R -2.1%, HR
    # -2.0%, RBI -2.7% (100 sure).
    player_specific=True,
)


def _pitcher(
    role: str,
    top_n: int,
    bands: tuple[float, ...],
    player_specific: bool,
    luck_bands: tuple[float, ...] = (),
) -> Spec:
    """A pitcher role (#505). Playing time is innings; K, W and SV are per batter faced
    (BF from IP by ``bf_ip``, the PT source's), ERA and WHIP per inning, their misses
    shrinking as 1/sqrt(IP). Ranked and fit within the role: a starter's 180 IP and a
    reliever's 65 are different jobs, and saves come from the role (their misses
    persist season to season, +0.26, where K, W, ERA and WHIP's don't)."""
    return Spec(
        name=role,
        volume="ip",
        trials={"bf": "bf_ip"},
        stats=(
            Stat("k", "bf", binomial=True, counting=True, numerator=("k",)),
            Stat("w", "bf", binomial=True, counting=True, numerator=("w",)),
            Stat("sv", "bf", binomial=True, counting=True, numerator=("sv",)),
            # An ERA or WHIP draw needs 10 IP.
            Stat(
                "era",
                "ip",
                binomial=False,
                counting=False,
                numerator=("er",),
                scale=9.0,
                min_trials=10.0,
            ),
            Stat(
                "whip", "ip", binomial=False, counting=False, numerator=("h", "bb"), min_trials=10.0
            ),
        ),
        features=(
            "rank_ip",
            "age",
            *(f"{c}{k}" for k in range(1, HISTORY_SEASONS + 1) for c in ("g", "ip", "gs")),
        ),
        never_blank=("rank_ip",),
        history_columns=("g", "ip", "gs"),
        top_n=top_n,
        bands=bands,
        headroom=2.0,
        # A past season needs 15 IP for its rate miss to join the luck pool.
        min_rate_volume=15.0,
        player_specific=player_specific,
        role=role,
        luck_by="sv" if luck_bands else None,
        luck_bands=luck_bands,
    )


# Steamer's preseason files, 2022-2026: about 150-165 projected starters (GS/G >= 0.5)
# with 60+ IP and 360-450 relievers with 30+ IP a season.
# Starters' ranges are one-size in playing time: held out 2022-2026 (Steamer PT, run
# 035-saves-net rates), quantile loss vs one-size, the player-specific range was IP
# -0.9% (83 sure), K and W -0.0%, ERA +1.1% and WHIP +0.7% (one-size 97-98 sure
# better); fewer inputs or more L1 (alpha 0.01-0.03) got IP to -1.4% but ERA stayed
# +0.5% and K/W/WHIP even. Steamer's starter innings already carry what his history
# says (season-to-season persistence +0.06). With last season's line as the PT
# source (2022-2026): IP -0.7%, the rest within 0.3%.
STARTER = _pitcher(
    "starter", top_n=150, bands=(0.0, 100.0, 140.0, 170.0, float("inf")), player_specific=False
)
# Relievers' are player-specific: held out 2022-2026, Steamer PT, quantile loss vs
# one-size IP -5.7%, K -5.6%, W -3.8%, SV -1.7%, WHIP -0.8% (100 sure), ERA -0.1% (63);
# last season's line, IP -9.5%, K -7.4%, W -2.9%, SV +0.0%, ERA +0.8%, WHIP +0.7%.
# Their luck is drawn within bands of projected saves (cut at the 50th, 75th and 90th
# percentile of the past relievers'): a setup man's 10 lucky saves are ~13 coin flips
# at his projected rate, and replayed on a closer gave seasons of 100. Steamer PT: SV
# loss -2.0% vs no bands (97 sure), IP -0.6%, the rest within 0.4%; seasons of
# projected 5-15 SV relievers under their P10 11%, not 31%. Bands at 60/85, 75/90 or
# 50/80/93 scored within noise of these. The projected 25+ SV closers' P90 still runs
# high (1 of 18 seasons above it, with ranges.fit's per-stat cap at the most saves in
# the training rows, 49; none without): save chances come from the team and aren't
# drawn.
RELIEVER = _pitcher(
    "reliever",
    top_n=400,
    bands=(0.0, 35.0, 50.0, 65.0, float("inf")),
    player_specific=True,
    luck_bands=(0.5, 0.75, 0.9),
)
# A projected starter: GS / G at least this.
STARTER_SHARE = 0.5
