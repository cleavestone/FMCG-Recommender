"""Replenishment forecasting: when a household will likely need a product
again, and how much they'll likely buy.

Extracted from notebooks/08_replenishment_forecasting.ipynb.

Deliberately not a BaseRecommender subclass - ranking ("recommend k items")
and forecasting ("when/how much for items you already know are relevant")
are different task shapes.

Two honest, verified results from that notebook worth remembering:

- Quantity forecasting works: per-pair median quantity beats a naive
  "always predict 1" baseline by ~28% MAE (0.28 vs 0.39 units).
- Timing forecasting does not currently beat a simpler household-average-
  gap baseline (56.3 vs 52.1 days MAE) - per-pair gap medians are
  estimated from too few historical gaps to be more reliable than pooling
  across a household's whole purchase history. Kept anyway because it
  still supports a coarse "roughly due" signal (57.9% of actual gaps fall
  within the predicted median +/- 1 std), just not a precise date - the
  concrete next step (not built here) is shrinking per-pair estimates
  toward the household average rather than trusting either in isolation.

`forecast()` excludes pairs where the household appears to have simply
stopped buying a product (more than 2x its own typical gap since the last
purchase) rather than being "very overdue" on an active habit - see the
notebook for why sorting by raw overdue-ness alone surfaced abandoned
habits, not upcoming needs.
"""
import pandas as pd

NON_REPLENISHABLE_DEPARTMENTS = {"KIOSK-GAS"}
NON_REPLENISHABLE_COMMODITIES = {"COUPON/MISC ITEMS"}


class ReplenishmentForecaster:
    def __init__(
        self,
        product_df: pd.DataFrame,
        min_purchases: int = 4,
        quantity_cap_quantile: float = 0.99,
    ):
        self.product_df = product_df
        self.min_purchases = min_purchases
        self.quantity_cap_quantile = quantity_cap_quantile
        self.pair_stats_: pd.DataFrame = pd.DataFrame()
        self.reference_day_: int = 0

    def fit(self, train_interactions: pd.DataFrame) -> "ReplenishmentForecaster":
        non_replenishable = (
            self.product_df["department"].isin(NON_REPLENISHABLE_DEPARTMENTS)
            | self.product_df["commodity_desc"].isin(NON_REPLENISHABLE_COMMODITIES)
        )
        excluded = set(self.product_df.loc[non_replenishable, "product_id"])
        df = train_interactions[~train_interactions["product_id"].isin(excluded)]
        self.reference_day_ = int(train_interactions["day"].max())

        events = (
            df.groupby(["household_key", "product_id", "basket_id"])
            .agg(day=("day", "first"), quantity=("quantity", "sum"))
            .reset_index()
            .sort_values(["household_key", "product_id", "day"])
        )
        counts = events.groupby(["household_key", "product_id"]).size()
        eligible = counts[counts >= self.min_purchases].index
        events = events.set_index(["household_key", "product_id"]).loc[eligible].reset_index()
        events["gap"] = events.groupby(["household_key", "product_id"])["day"].diff()

        cap = df["quantity"].quantile(self.quantity_cap_quantile)
        events["quantity_capped"] = events["quantity"].clip(upper=cap)

        self.pair_stats_ = events.groupby(["household_key", "product_id"]).agg(
            last_day=("day", "max"),
            median_gap=("gap", "median"),
            gap_std=("gap", "std"),
            median_qty=("quantity_capped", "median"),
            n_purchases=("day", "size"),
        ).reset_index()
        return self

    def forecast(self, household_id: int, as_of_day: int | None = None, max_overdue_multiple: float = 2.0) -> pd.DataFrame:
        if as_of_day is None:
            as_of_day = self.reference_day_
        rows = self.pair_stats_[self.pair_stats_["household_key"] == household_id].copy()
        rows["expected_day"] = rows["last_day"] + rows["median_gap"]
        rows["days_until_due"] = rows["expected_day"] - as_of_day
        rows["window_days"] = rows["gap_std"].fillna(0)
        rows["expected_quantity"] = rows["median_qty"]

        days_since_last = as_of_day - rows["last_day"]
        still_active = days_since_last <= max_overdue_multiple * rows["median_gap"]
        rows = rows[still_active]

        return rows.sort_values("days_until_due")[
            ["product_id", "days_until_due", "window_days", "expected_quantity", "n_purchases"]
        ].reset_index(drop=True)
