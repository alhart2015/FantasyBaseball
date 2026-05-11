from __future__ import annotations

from datetime import datetime

from fantasy_baseball.streaks.data.load_model_fits import upsert_model_fits
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.models import ModelFit


def _fit(
    model_id: str = "hr_hot_2023-2024",
    val_auc: float = 0.58,
) -> ModelFit:
    return ModelFit(
        model_id=model_id,
        category="hr",
        direction="above",
        season_set="2023-2024",
        window_days=14,
        cold_method="poisson_p20",
        chosen_C=1.0,
        cv_auc_mean=0.57,
        cv_auc_std=0.02,
        val_auc=val_auc,
        n_train_rows=20_000,
        n_val_rows=10_000,
        fit_timestamp=datetime(2026, 5, 10, 12, 0, 0),
    )


def test_upsert_model_fits_inserts_rows() -> None:
    conn = get_connection(":memory:")
    upsert_model_fits(conn, [_fit("a"), _fit("b")])
    n = conn.execute("SELECT COUNT(*) FROM model_fits").fetchone()[0]
    assert n == 2


def test_upsert_model_fits_replaces_on_pk_collision() -> None:
    conn = get_connection(":memory:")
    upsert_model_fits(conn, [_fit("a", val_auc=0.50)])
    upsert_model_fits(conn, [_fit("a", val_auc=0.62)])
    val = conn.execute("SELECT val_auc FROM model_fits WHERE model_id='a'").fetchone()[0]
    assert val == 0.62


def test_upsert_model_fits_empty_input_is_noop() -> None:
    conn = get_connection(":memory:")
    upsert_model_fits(conn, [])
    n = conn.execute("SELECT COUNT(*) FROM model_fits").fetchone()[0]
    assert n == 0
