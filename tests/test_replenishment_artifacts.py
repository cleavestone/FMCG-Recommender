"""Round-trip test for artifacts.save_replenishment_artifacts / load_replenishment_artifacts."""
import pandas as pd

from fmcg_reco.artifacts import load_replenishment_artifacts, save_replenishment_artifacts
from fmcg_reco.models.replenishment import ReplenishmentForecaster


def _toy_products() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": [1, 2],
        "department": ["GROCERY", "KIOSK-GAS"],
        "commodity_desc": ["MILK", "FUEL"],
    })


def _toy_train() -> pd.DataFrame:
    rows = []
    for i, day in enumerate([0, 10, 20, 30]):
        rows.append({"household_key": 1, "basket_id": i, "product_id": 1, "day": day, "quantity": 2})
    return pd.DataFrame(rows)


def test_loaded_forecaster_matches_original(tmp_path):
    product_df = _toy_products()
    original = ReplenishmentForecaster(product_df, min_purchases=4).fit(_toy_train())

    save_replenishment_artifacts(original, tmp_path)
    loaded = load_replenishment_artifacts(tmp_path, product_df)

    pd.testing.assert_frame_equal(
        loaded.forecast(1, as_of_day=35).reset_index(drop=True),
        original.forecast(1, as_of_day=35).reset_index(drop=True),
    )
    assert loaded.min_purchases == original.min_purchases
