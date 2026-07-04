"""deltaRoto -- roto-point impact metric for player swaps.

Uses EV-based score_roto, so deltaRoto.total is simply the change in
total expected roto points across all categories. No tuning knobs,
no tie bands, no defensive-comfort heuristic -- the Gaussian pairwise
win-probabilities price projection uncertainty and vulnerability
directly into the score.

The confidence band (:class:`DeltaRotoBand`) is closed-form, not Monte
Carlo: its ``mean`` reuses the EV deltaRoto so it is identical to the
point estimate, and its ``sd`` propagates the swapped players'
per-category stat variance through each category's Gaussian
roto-points curve via a fixed Gauss-Hermite rule. Deterministic and
cheap enough to run inline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fantasy_baseball.models.standings import CategoryStats
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES,
    COUNTING_STATS,
    INVERSE_STATS,
    Category,
)

if TYPE_CHECKING:
    from fantasy_baseball.models.player import Player
    from fantasy_baseball.models.standings import ProjectedStandings


@dataclass
class CategoryDelta:
    roto_delta: float


@dataclass
class DeltaRotoResult:
    total: float
    categories: dict[str, CategoryDelta]
    before_total: float
    after_total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 2),
            "before_total": round(self.before_total, 2),
            "after_total": round(self.after_total, 2),
            "categories": {
                cat: {"roto_delta": round(cd.roto_delta, 2)} for cat, cd in self.categories.items()
            },
        }


def score_swap(
    roto_before: dict[str, dict[str, float]],
    roto_after: dict[str, dict[str, float]],
    team_name: str,
) -> DeltaRotoResult:
    """Per-category deltaRoto from before/after ``score_roto`` outputs.

    Total is the change in team total EV roto points. Each category's
    ``roto_delta`` is the change in that category's EV points for the
    user's team. No discounts, no penalties -- the EV already reflects
    projection uncertainty, defensive vulnerability, and boundary
    proximity via the sigmoid on pairwise win probabilities.
    """
    categories: dict[str, CategoryDelta] = {}
    for cat in ALL_CATEGORIES:
        rd = roto_after[team_name][f"{cat.value}_pts"] - roto_before[team_name][f"{cat.value}_pts"]
        categories[cat.value] = CategoryDelta(roto_delta=rd)

    return DeltaRotoResult(
        total=roto_after[team_name]["total"] - roto_before[team_name]["total"],
        categories=categories,
        before_total=roto_before[team_name]["total"],
        after_total=roto_after[team_name]["total"],
    )


def compute_delta_roto(
    drop_name: str,
    add_player: Player,
    user_roster: list[Player],
    projected_standings: ProjectedStandings,
    team_name: str,
    *,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
) -> DeltaRotoResult:
    """Compute deltaRoto for dropping one player and adding another.

    When ``team_sds`` is provided, ``score_roto`` uses pairwise Gaussian
    win-probabilities so a swap's impact reflects projection
    uncertainty (ERoto). Pass ``None`` explicitly for exact-rank
    semantics -- no default: callers must make the choice so we can't
    silently fall back to integer roto by forgetting the argument.

    Args:
        drop_name: roster player to drop.
        add_player: Player to add.
        user_roster: current roster (used to resolve the dropped player's ROS).
        projected_standings: end-of-season stats for all teams.
        team_name: user's team name.
        team_sds: ``{team: {Category: sd}}`` for EV scoring, or ``None``
            for rank-based. Required keyword -- no default.

    Raises:
        ValueError: if drop_name is not found on the roster.
    """
    from fantasy_baseball.scoring import score_roto_dict
    from fantasy_baseball.trades.evaluate import build_swap_standings, find_player_by_name

    dropped = find_player_by_name(drop_name, user_roster)
    if dropped is None:
        raise ValueError(f"Player '{drop_name}' not found on roster")

    all_before, all_after = build_swap_standings(
        dropped, add_player, projected_standings, team_name
    )
    roto_before = score_roto_dict(all_before, team_sds=team_sds)
    roto_after = score_roto_dict(all_after, team_sds=team_sds)

    return score_swap(roto_before, roto_after, team_name)


@dataclass
class DeltaRotoBand:
    """Closed-form confidence band for a before->after deltaRoto.

    ``mean`` is the EV point estimate from :func:`compute_delta_roto`
    (identical, not an approximation). ``sd`` is the analytic standard
    deviation of the deltaRoto under the swapped players' per-category
    stat uncertainty, and ``p_positive`` is the Gaussian probability the
    swap helps. ``to_dict`` calls :func:`band_class` so the P(helps)
    verdict is computed once, in Python, for every surface.
    """

    mean: float
    sd: float
    p_positive: float

    def to_dict(self) -> dict[str, float | str]:
        from fantasy_baseball.lineup.band_format import band_class

        return {
            "mean": round(self.mean, 2),
            "sd": round(self.sd, 2),
            "p_positive": round(self.p_positive, 3),
            "verdict": band_class(self.p_positive),
        }


# Fixed 9-node Gauss-Hermite (probabilists') quadrature for integrating a
# function of dX ~ N(d_mu, sigma^2). Nodes/weights from
# numpy.polynomial.hermite_e.hermegauss(9), whose weight function is
# exp(-x^2/2), so E[f(X)] = sum_k (w_k / sqrt(2*pi)) * f(d_mu + sigma * z_k).
# Hardcoded so the band carries no numpy dependency and stays deterministic.
_GH_NODES: tuple[float, ...] = (
    -4.512745863399783,
    -3.20542900285647,
    -2.07684797867783,
    -1.0232556637891326,
    0.0,
    1.0232556637891326,
    2.07684797867783,
    3.20542900285647,
    4.512745863399783,
)
_GH_WEIGHTS: tuple[float, ...] = (
    5.601272441031135e-05,
    0.0069913404977412184,
    0.12512187656567714,
    0.6118617025232779,
    1.0185664100087874,
    0.6118617025232779,
    0.12512187656567714,
    0.0069913404977412184,
    5.601272441031135e-05,
)
_GH_NORM = math.sqrt(2.0 * math.pi)


def _normal_cdf(z: float) -> float:
    """Standard-normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _swap_sets(
    before_players: list[Player],
    after_players: list[Player],
) -> tuple[list[Player], list[Player]]:
    """Players entering (IN) and leaving (OUT) the user's roster.

    A player is IN if in ``after`` but not ``before``, OUT if in
    ``before`` but not ``after``. Players shared by both lists cancel
    (common random numbers) and contribute no variance. Keyed on the
    canonical ``name::player_type`` identity -- a two-way player's
    hitter and pitcher rows share a name and must swap independently.
    """
    if before_players is after_players:
        # Identity split (the legacy anchor rebuilding its own row).
        return [], []
    # Function-local like this module's other trades.* imports:
    # multi_trade imports delta_roto lazily, so a top-level import here
    # would half-close an import cycle.
    from fantasy_baseball.trades.multi_trade import player_key

    before_keyed = [(player_key(p), p) for p in before_players]
    after_keyed = [(player_key(p), p) for p in after_players]
    before_keys = {k for k, _ in before_keyed}
    after_keys = {k for k, _ in after_keyed}
    in_players = [p for k, p in after_keyed if k not in before_keys]
    out_players = [p for k, p in before_keyed if k not in after_keys]
    return in_players, out_players


def _ev_delta_and_stats(
    before_players: list[Player],
    after_players: list[Player],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    reference_players: list[Player] | None = None,
) -> tuple[float, CategoryStats, CategoryStats, list[Player], list[Player]]:
    """EV deltaRoto total, before/after stat rows, and the IN/OUT sets.

    Mean mechanism (identical to :func:`compute_delta_roto`,
    :mod:`multi_trade`, and the audit): rebuild each endpoint's user row
    from the ``projected_standings`` row by subtracting the aggregated ROS
    of players leaving and adding players entering, score both with
    ``team_sds``, take the total delta. For a one-for-one swap this is
    exactly ``compute_delta_roto(...).total`` because a single-player
    ``aggregate_player_stats`` equals ``player_rest_of_season_stats`` and
    ``apply_swap_delta`` is the same call.

    **Anchor contract:** the standings row is the roto total of one
    specific lineup. ``reference_players`` names that lineup, and both
    endpoints are rebuilt relative to it. When ``None`` (the legacy
    default), ``before_players`` is assumed to BE that lineup -- valid for
    the waiver/trade/audit callers, whose "before" is the current roster.
    Callers whose "before" is itself hypothetical (the lineup optimizer's
    per-starter counterfactuals) MUST pass ``reference_players``:
    anchoring an alt lineup on the current-lineup row double-counts every
    player who is in the current lineup but not in "before" (a current
    starter's stats end up in the anchor AND in the IN set), which is how
    reliever rows showed a fictional 101-save team.

    Also returns the user team's before/after :class:`CategoryStats` rows
    so the sd path can read the per-category baseline mean and shift, and
    the before->after IN/OUT player lists from :func:`_swap_sets` so the
    caller need not recompute the split.
    """
    from fantasy_baseball.scoring import score_roto_dict
    from fantasy_baseball.trades.evaluate import (
        aggregate_player_stats,
        apply_swap_delta,
        team_baseline_volumes,
    )

    in_players, out_players = _swap_sets(before_players, after_players)

    all_rows = {e.team_name: e.stats.to_dict() for e in projected_standings.entries}
    anchor_dict = all_rows[team_name]
    # Pull AB/IP off the projected standings entry; pre-PR-110 / legacy
    # entries decay to None via team_baseline_volumes so apply_swap_delta
    # falls back to the constant heuristic.
    user_ab, user_ip = team_baseline_volumes(projected_standings.by_team()[team_name])

    reference = reference_players if reference_players is not None else before_players

    def _row_for(ins: list[Player], outs: list[Player]) -> dict[str, Any]:
        """The user row for a hypothetical lineup, rebuilt off the anchor."""
        if not ins and not outs:
            return anchor_dict
        return apply_swap_delta(
            anchor_dict,
            aggregate_player_stats(outs),
            aggregate_player_stats(ins),
            team_ab=user_ab,
            team_ip=user_ip,
        )

    user_before_dict = _row_for(*_swap_sets(reference, before_players))
    user_after_dict = _row_for(*_swap_sets(reference, after_players))

    all_before = dict(all_rows)
    all_before[team_name] = user_before_dict
    all_after = dict(all_rows)
    all_after[team_name] = user_after_dict

    roto_before = score_roto_dict(all_before, team_sds=team_sds)
    roto_after = score_roto_dict(all_after, team_sds=team_sds)
    total = roto_after[team_name]["total"] - roto_before[team_name]["total"]

    before_cs = CategoryStats.from_dict(user_before_dict)
    after_cs = CategoryStats.from_dict(user_after_dict)
    return total, before_cs, after_cs, in_players, out_players


def _swap_category_variance(
    cat: Category,
    in_players: list[Player],
    out_players: list[Player],
    before_players: list[Player],
    after_players: list[Player],
    fraction_remaining: float,
) -> float:
    """Variance of the swap's change in the user's category-``cat`` total.

    Counting categories (R, HR, RBI, SB, W, K, SV): variances add across
    players, so ``sigma2 = fraction_remaining * sum over IN and OUT
    players of player_category_variance(p)[cat]``. Both the entering and
    leaving players' uncertainties contribute. Scaled by
    ``fraction_remaining`` because variance is proportional to the
    remaining season (matches ``build_team_sds`` using
    ``sd_scale = sqrt(fraction_remaining)``).

    Rate categories (AVG, ERA, WHIP): not additive per player (shared
    denominator), so the marginal variance is the change in the team's
    rate variance between the after- and before-rosters. We take the
    absolute difference of the two team-level rate variances from
    ``project_team_sds`` (a defensible marginal-variance estimate: the
    rate-variance change attributable to the swap), scaled by
    ``fraction_remaining``.

    Caveat: a swap that shifts the team's rate MEAN but leaves its rate
    VARIANCE roughly unchanged (e.g. swapping a high-volume arm for an
    equal-volume arm with a worse ERA -- same IP and similar er^2, so
    ``project_team_sds`` barely moves) yields a near-zero rate variance
    here. The band width does NOT track that mean shift; the mean-only
    move is captured by the EV ``mean`` (via ``apply_swap_delta``), not by
    this term. Readers should not infer the rate band reflects the mean
    change.
    """
    from fantasy_baseball.scoring import player_category_variance, project_team_sds

    if cat in COUNTING_STATS:
        total = 0.0
        for p in (*in_players, *out_players):
            total += player_category_variance(p).get(cat, 0.0)
        return fraction_remaining * total

    # Rate category: derive from before/after team-level rate SDs.
    sd_before = project_team_sds(before_players, displacement=False).get(cat, 0.0)
    sd_after = project_team_sds(after_players, displacement=False).get(cat, 0.0)
    return fraction_remaining * abs(sd_after * sd_after - sd_before * sd_before)


def _category_points(
    x: float,
    field_means: list[float],
    combined_sds: list[float],
    *,
    higher_is_better: bool,
) -> float:
    """User's category points at realized total ``x``.

    ``pts(x) = 1 + sum_j Phi((x - mu_j) / s_j)`` over the fixed field,
    the same Gaussian-pairwise expectation ``score_roto`` sums. For
    inverse categories (ERA, WHIP) lower is better, so the difference is
    flipped. A zero combined SD degrades to the rank step function.
    """
    pts = 1.0
    for mu_j, s_j in zip(field_means, combined_sds, strict=True):
        diff = (x - mu_j) if higher_is_better else (mu_j - x)
        if s_j <= 0.0:
            pts += 1.0 if diff > 0 else (0.0 if diff < 0 else 0.5)
        else:
            pts += _normal_cdf(diff / s_j)
    return pts


def _category_delta_variance(
    cat: Category,
    before_cs: CategoryStats,
    after_cs: CategoryStats,
    field_stats: Mapping[str, CategoryStats],
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    team_name: str,
    sigma2: float,
) -> float:
    """Variance of the category-``cat`` roto-points delta under the swap.

    The user's realized category total is ``mu_b + dX`` where ``mu_b`` is
    the before-swap category mean and ``dX ~ N(d_mu, sigma2)`` with
    ``d_mu`` the EV stat shift. The category points delta is
    ``dpts = pts(mu_b + dX) - pts(mu_b)``, holding the (cancelling)
    shared roster at its mean. We integrate ``Var(dpts) = E[dpts^2] -
    E[dpts]^2`` with a fixed Gauss-Hermite rule.

    With ``team_sds=None`` the curve degrades to the rank step function
    (same as ``score_roto``); the band then reflects only whether the
    EV stat shift crosses a rank boundary.
    """
    if sigma2 <= 0.0:
        return 0.0

    sds = team_sds or {}
    higher_is_better = cat not in INVERSE_STATS
    sd_me = sds.get(team_name, {}).get(cat, 0.0)
    field_means: list[float] = []
    combined_sds: list[float] = []
    for tname, cs in field_stats.items():
        field_means.append(cs[cat])
        sd_j = sds.get(tname, {}).get(cat, 0.0)
        combined_sds.append(math.sqrt(sd_me * sd_me + sd_j * sd_j))

    mu_b = before_cs[cat]
    d_mu = after_cs[cat] - before_cs[cat]
    sigma = math.sqrt(sigma2)

    base_pts = _category_points(mu_b, field_means, combined_sds, higher_is_better=higher_is_better)

    e_dpts = 0.0
    e_dpts_sq = 0.0
    for node, weight in zip(_GH_NODES, _GH_WEIGHTS, strict=True):
        dx = d_mu + sigma * node
        dpts = (
            _category_points(
                mu_b + dx, field_means, combined_sds, higher_is_better=higher_is_better
            )
            - base_pts
        )
        w = weight / _GH_NORM
        e_dpts += w * dpts
        e_dpts_sq += w * dpts * dpts

    var = e_dpts_sq - e_dpts * e_dpts
    return max(var, 0.0)


def band_reference_lineup(
    candidates: list[Player], other_half: list[Player] | None = None
) -> list[Player] | None:
    """Infer the anchor lineup for :func:`compute_delta_roto_band`.

    The projected-standings user row is built from the CURRENT Yahoo
    lineup, so the band anchor is the currently-active subset of
    ``candidates`` -- the same slot-first partition the standings row
    itself uses (``scoring._classify_roster``, so None/unrecognized slots
    count as active there and here) -- plus the fixed ``other_half``. The
    result need only equal the anchor lineup *modulo players present in
    both endpoints*: shared extras (e.g. benched pitchers in the hitter
    optimizer's fixed half) cancel in :func:`_swap_sets`.

    Returns ``None`` when no candidate carries a selected position (bare
    test fixtures, pre-fetch callers) -- the band then falls back to the
    legacy before-is-the-anchor contract.
    """
    if all(p.selected_position is None for p in candidates):
        # Bare fixtures / pre-fetch callers: no inferable anchor -- fall back
        # to the legacy before-is-the-anchor contract.
        return None
    from fantasy_baseball.scoring import _classify_roster

    active, _, _ = _classify_roster(candidates)
    # An empty active set (every slotted candidate benched, e.g. a lineup-set
    # transition window) still yields a valid anchor: the standings row's
    # lineup for this half IS empty, so the reference is other_half alone.
    return [*active, *(other_half or [])]


def compute_delta_roto_band(
    before_players: list[Player],
    after_players: list[Player],
    field_stats: Mapping[str, CategoryStats],
    team_name: str,
    fraction_remaining: float,
    *,
    projected_standings: ProjectedStandings,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    reference_players: list[Player] | None = None,
) -> DeltaRotoBand:
    """Closed-form confidence band for a before->after roster change.

    ``mean`` is the EV deltaRoto (identical to
    :func:`compute_delta_roto` for a one-for-one swap; see
    :func:`_ev_delta_and_stats`). ``sd`` propagates the swapped players'
    per-category stat variance through each category's Gaussian
    roto-points curve, summing per-category variances under a
    category-independence assumption. Deterministic -- no sampling.

    Args:
        before_players: the user's roster before the change.
        after_players: the user's roster after the change. ``before`` and
            ``after`` differ only by the swapped players; shared players
            cancel.
        field_stats: the other teams' fixed point :class:`CategoryStats`,
            keyed by team name (``projected_standings.field_stats(team)``).
        team_name: the user's team name (the key the swap is scored under).
        fraction_remaining: portion of the season left -- scales variance.
        projected_standings: full projected standings, used to anchor the
            EV mean on the user's projected row.
        team_sds: ``{team: {Category: sd}}``, already scaled by
            ``sqrt(fraction_remaining)`` -- the same combined-SD softness
            ``score_roto`` uses for the points curve. ``None`` falls back
            to the rank step function (no curve softness).
        reference_players: the lineup the ``projected_standings`` user row
            reflects. REQUIRED whenever ``before_players`` is itself a
            hypothetical lineup rather than that roster (see the anchor
            contract on :func:`_ev_delta_and_stats`); ``None`` keeps the
            legacy before-is-the-anchor behavior.

    Known limitation: the endpoint rows are rebuilt by face-value component
    arithmetic, while the anchor row itself carries displacement scaling
    (IL players at partial credit, displaced actives scaled down -- see
    ``scoring._apply_displacement``). Swapping a *displaced* player
    therefore moves his full ROS line against a partially-scaled anchor, a
    second-order error measured at <=0.05 roto points on live data (vs the
    ~1.0-point anchor artifact this contract exists to prevent). The
    pool-model ``roto_delta`` next to the band remains the exact
    selection-grade number.
    """
    mean, before_cs, after_cs, in_players, out_players = _ev_delta_and_stats(
        before_players,
        after_players,
        projected_standings,
        team_name,
        team_sds,
        reference_players=reference_players,
    )

    var_total = 0.0
    for cat in ALL_CATEGORIES:
        sigma2 = _swap_category_variance(
            cat, in_players, out_players, before_players, after_players, fraction_remaining
        )
        if sigma2 <= 0.0:
            continue
        var_total += _category_delta_variance(
            cat, before_cs, after_cs, field_stats, team_sds, team_name, sigma2
        )

    sd = math.sqrt(var_total)
    if sd > 0.0:
        p_positive = _normal_cdf(mean / sd)
    elif mean > 0:
        p_positive = 1.0
    elif mean < 0:
        p_positive = 0.0
    else:
        # mean == 0, sd == 0 (identity swap): neutral, maps to coin-flip.
        p_positive = 0.5
    return DeltaRotoBand(mean=mean, sd=sd, p_positive=p_positive)


def compute_one_for_one_band(
    drop_name: str,
    add_player: Player,
    active_players: list[Player],
    field_stats: Mapping[str, CategoryStats],
    team_name: str,
    fraction_remaining: float,
    *,
    projected_standings: ProjectedStandings,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
) -> DeltaRotoBand:
    """Band for a one-for-one swap: drop ``drop_name``, add ``add_player``.

    Thin wrapper that builds the before/after rosters and delegates to
    :func:`compute_delta_roto_band`. The band's ``mean`` equals
    ``compute_delta_roto(drop_name, add_player, ...).total``.
    """
    before = list(active_players)
    after = [p for p in active_players if p.name != drop_name] + [add_player]
    return compute_delta_roto_band(
        before,
        after,
        field_stats,
        team_name,
        fraction_remaining,
        projected_standings=projected_standings,
        team_sds=team_sds,
    )
