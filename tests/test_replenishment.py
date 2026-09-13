"""Unit tests for models.replenishment.ReplenishmentForecaster."""
import pandas as pd

from fmcg_reco.models.replenishment import ReplenishmentForecaster


def _toy_products() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": [1, 2, 3, 4],
        "department": ["GROCERY", "KIOSK-GAS", "GROCERY", "GROCERY"],
        "commodity_desc": ["MILK", "FUEL", "COUPON/MISC ITEMS", "BREAD"],
    })


def _toy_train() -> pd.DataFrame:
    rows = []
    # household 1, product 1 (milk): 4 purchases, gap of exactly 10 days each time, qty=2 always
    for i, day in enumerate([0, 10, 20, 30]):
        rows.append({"household_key": 1, "basket_id": i, "product_id": 1, "day": day, "quantity": 2})
    # household 1, product 2 (gas): many purchases, huge quantity - must be excluded entirely
    for i, day in enumerate([0, 5, 10, 15, 20]):
        rows.append({"household_key": 1, "basket_id": 100 + i, "product_id": 2, "day": day, "quantity": 30000})
    # household 1, product 3 (coupon/misc): must be excluded entirely
    for i, day in enumerate([0, 5, 10, 15]):
        rows.append({"household_key": 1, "basket_id": 200 + i, "product_id": 3, "day": day, "quantity": 500})
    # household 1, product 4 (bread): only 2 purchases - below min_purchases threshold
    for i, day in enumerate([0, 10]):
        rows.append({"household_key": 1, "basket_id": 300 + i, "product_id": 4, "day": day, "quantity": 1})
    return pd.DataFrame(rows)


def test_excludes_non_replenishable_departments_and_commodities():
    forecaster = ReplenishmentForecaster(_toy_products(), min_purchases=4).fit(_toy_train())
    forecasted_products = set(forecaster.pair_stats_["product_id"])
    assert 2 not in forecasted_products  # KIOSK-GAS
    assert 3 not in forecasted_products  # COUPON/MISC ITEMS


def test_excludes_pairs_below_min_purchases():
    forecaster = ReplenishmentForecaster(_toy_products(), min_purchases=4).fit(_toy_train())
    assert 4 not in set(forecaster.pair_stats_["product_id"])  # only 2 purchases


def test_computes_gap_and_quantity_correctly():
    forecaster = ReplenishmentForecaster(_toy_products(), min_purchases=4).fit(_toy_train())
    row = forecaster.pair_stats_[forecaster.pair_stats_["product_id"] == 1].iloc[0]
    assert row["median_gap"] == 10
    assert row["gap_std"] == 0
    assert row["median_qty"] == 2
    assert row["last_day"] == 30
    assert row["n_purchases"] == 4


def test_forecast_includes_active_habit():
    forecaster = ReplenishmentForecaster(_toy_products(), min_purchases=4).fit(_toy_train())
    # last purchase day 30, median_gap 10 -> expected_day 40; as_of_day=35 -> days_until_due=5
    result = forecaster.forecast(household_id=1, as_of_day=35)
    assert list(result["product_id"]) == [1]
    assert result.iloc[0]["days_until_due"] == 5
    assert result.iloc[0]["expected_quantity"] == 2


def test_forecast_excludes_abandoned_habit():
    forecaster = ReplenishmentForecaster(_toy_products(), min_purchases=4).fit(_toy_train())
    # days_since_last = 60 - 30 = 30, threshold = 2 * median_gap(10) = 20 -> 30 > 20 -> excluded
    result = forecaster.forecast(household_id=1, as_of_day=60)
    assert result.empty


def test_forecast_unknown_household_returns_empty():
    forecaster = ReplenishmentForecaster(_toy_products(), min_purchases=4).fit(_toy_train())
    result = forecaster.forecast(household_id=9999, as_of_day=35)
    assert result.empty


def test_reference_day_defaults_to_last_training_day():
    forecaster = ReplenishmentForecaster(_toy_products(), min_purchases=4).fit(_toy_train())
    assert forecaster.reference_day_ == 30  # max day across all of _toy_train()

    with_default = forecaster.forecast(household_id=1)
    with_explicit = forecaster.forecast(household_id=1, as_of_day=30)
    pd.testing.assert_frame_equal(with_default, with_explicit)
