"""The shipped output of the #266 calibration study.

These are the numbers increment 2 applies. They live in code rather than only in
the finding document so the production path imports them instead of retyping a
markdown table, and so `tests/test_keepers/test_coefficients.py` can hold them
against the study's own CSV output.

Full derivation and every caveat: `docs/superpowers/keeper-calibration-finding-2026-07-27.md`.
Regenerate the CSVs with `python scripts/keeper_calibration.py`.

Three things a caller must not get wrong:

  * **The coefficients are conditional on `n0`.** `k` and the shrink enter the
    fold multiplicatively, and the study found `k` scales almost exactly
    inversely with the shrink WEIGHT while held-out error does not move -- so `k`
    RISES with `n0` (hr_pa: 0.407 at n0=100, 0.494 at 200, 0.655 at 400). Only the
    product `k * w` is identified. Using these `k` values with a different `n0` is
    meaningless (finding B.5).
  * **Apply `fold.gate_ramp`, not `fold.gate_mask`.** The hard gate is a cliff at
    the threshold: 78.7% across two plate appearances for hitters, 44.6% across
    two innings for pitchers, because the playing-time term is unshrunk.
    `gate_mask` selects fit-sample rows; the ramp is what production applies
    (finding B.7).
  * **`pa` and `k_ip` are endpoint fallbacks, not fitted values.** Both lost to
    `k=1` on held-out error under the pre-registered rule. For `pa` the mechanism
    is understood and unresolved -- see finding B.3, the largest open item.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from fantasy_baseball.keepers.fold import gate_ramp, shrink


@dataclass(frozen=True)
class FoldPolicy:
    """Everything the production fold needs for one player type."""

    coefficients: Mapping[str, float]
    """Per-column transfer coefficient `k`, conditional on `n0`."""

    n0: float
    """Shrink constant: the playing time at which observation and projection get
    equal weight, in `w = n / (n + n0)`."""

    gate: float
    """Realized playing time below which a player is not folded at all."""

    ramp_width: float
    """Width of the linear on-ramp above `gate`, for `fold.gate_ramp`."""

    pt_col: str
    """Which coefficient is playing time. It is the one fit UNSHRUNK, so it is
    the one `serve_weights` must not damp."""

    def serve_weights(self, realized_pt: pd.Series) -> dict[str, pd.Series]:
        """The per-column fold weights for the production path.

        This exists so the three rules that are otherwise only prose become
        mechanical, and so no caller has to assemble them from primitives:

          * **Ramp, not hard gate.** The hard gate is a cliff at the threshold --
            78.7% across two plate appearances for hitters, 44.6% across two
            innings for pitchers -- because the PT term is unshrunk. `gate_ramp`
            removes the step; `gate_mask` is for selecting fit-sample rows, never
            for serving.
          * **Rates are shrunk, playing time is not.** Damping the PT residual by
            a function of the playing time an injury suppressed is circular and
            would make the coefficient unable to learn from lost time (spec 5.3).
          * **`k` is conditional on `n0`.** The shrink here uses this policy's own
            `n0`, the one its coefficients were fit against.
        """
        ramp = gate_ramp(realized_pt, self.gate, self.ramp_width)
        rate_weight = shrink(realized_pt, self.n0) * ramp
        return {col: (ramp if col == self.pt_col else rate_weight) for col in self.coefficients}


# Fitted 2026-07-27 over pairs 2022->2023, 2023->2024, 2024->2025.
# `pa` is the k=1 fallback (fitted 0.646, CI [0.584, 0.708]); see finding B.3.
_HITTER = FoldPolicy(
    coefficients=MappingProxyType(
        {
            "hr_pa": 0.494,
            "r_pa": 0.531,
            "rbi_pa": 0.532,
            "sb_pa": 0.637,  # provisional -- widest CI, 2023 rules break (B.4)
            "h_ab": 0.428,
            "ab_pa": 0.687,
            # fallback:k=1 per the pre-registered rule. NOT a settled result: the
            # fitted value is 0.646 (CI [0.584, 0.708], excluding 1.0), and k=1 wins
            # out of sample only because it stands in for an unshipped -83 PA level
            # term. Finding B.3 calls this the largest open item for increment 2,
            # and note main's shipped keeper path applies an up-only PT heal
            # (analysis/keeper_value.py DEFAULT_PT_HEAL_CAP) that pulls the other way.
            "pa": 1.000,
        }
    ),
    n0=200.0,
    gate=100.0,
    ramp_width=100.0,
    pt_col="pa",
)

# `k_ip` is the k=1 fallback (fitted 0.970, CI [0.866, 1.073]). The two coefficients
# are 3.0% apart but their HELD-OUT ERROR is only 0.36% apart, which is why the
# pre-registered rule resolves it to the endpoint: strikeout rate carries forward
# in full. Note the clamp DID bind in the ex-2023 fold (raw 1.024) and at n0=100
# (raw 1.276); the report flags that per fold as `folds_clamped`.
_PITCHER = FoldPolicy(
    coefficients=MappingProxyType(
        {
            "k_ip": 1.000,  # fallback:k=1
            "w_ip": 0.491,
            "er_ip": 0.343,
            "bb_ip": 0.697,
            "h_ip": 0.385,
            "ip": 0.631,
        }
    ),
    n0=50.0,
    gate=50.0,
    ramp_width=50.0,
    pt_col="ip",
)

POLICIES = MappingProxyType({"hitter": _HITTER, "pitcher": _PITCHER})


# The verdict vocabulary. `scripts/keeper_calibration.py` emits these and
# `policy_from_study` parses them, so one module owns the strings rather than
# both retyping them.
PASS_VERDICT = "pass"
FALLBACK_PREFIX = "fallback:k="
CHOSEN_ESTIMATOR = "fitted-k"


def policy_from_study(report: pd.DataFrame, pt_col: str, ramp_width: float) -> FoldPolicy:
    """Derive a `FoldPolicy` from a `keeper_calibration_<type>.csv` report.

    This is the rule that turns a study row into a shipped constant, and it lives
    here rather than in prose or in a test so the re-fit promised for `sb_pa`
    after the 2026 season is a mechanical diff instead of a hand transcription of
    thirteen floats (six hitter rates + PA, five pitcher rates + IP):

      * `verdict == "pass"`   -> ship the fitted `k_full`
      * `verdict == "fallback:k=X"` -> ship the ENDPOINT X that beat it, not the
        fitted value. The fallback means the fitted coefficient lost on held-out
        error, so shipping it anyway would ignore the study's own bar (spec 6.6).

    `n0` and `gate` come from the report too -- the coefficients are only
    interpretable against the ones they were fit under.
    """
    fitted = report.loc[report["estimator"] == CHOSEN_ESTIMATOR]
    if fitted.empty:
        raise ValueError(f"report contains no {CHOSEN_ESTIMATOR!r} rows")
    coefficients: dict[str, float] = {}
    for column, row in fitted.set_index("column").iterrows():
        verdict = str(row["verdict"])
        if verdict == PASS_VERDICT:
            coefficients[str(column)] = round(float(row["k_full"]), 3)
        elif verdict.startswith(FALLBACK_PREFIX):
            coefficients[str(column)] = float(verdict.removeprefix(FALLBACK_PREFIX))
        else:
            raise ValueError(f"unrecognized verdict for {column!r}: {verdict!r}")
    n0 = fitted["n0"].unique()
    gate = fitted["gate"].unique()
    if len(n0) != 1 or len(gate) != 1:
        raise ValueError(f"report mixes fit settings: n0={n0}, gate={gate}")
    return FoldPolicy(
        coefficients=MappingProxyType(coefficients),
        n0=float(n0[0]),
        gate=float(gate[0]),
        ramp_width=ramp_width,
        pt_col=pt_col,
    )
