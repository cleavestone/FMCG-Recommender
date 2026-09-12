"""Non-personalized and repeat-purchase baselines.

Extracted from notebooks/02_eval_harness_and_baselines.ipynb (sections 7-8).
On the validation split, PersonalRepeatPurchaseRecommender beat
PopularityRecommender by ~2.6x on MAP@10 (0.326 vs 0.124) — the bar every
later model in this project has to clear.
"""
import pandas as pd

from fmcg_reco.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Ranks candidate items by distinct-basket count in train. No personalization."""

    def __init__(self, candidate_items: set | None = None):
        self.candidate_items = candidate_items
        self.ranking_: list = []

    def fit(self, train_interactions: pd.DataFrame) -> "PopularityRecommender":
        df = train_interactions
        if self.candidate_items is not None:
            df = df[df["product_id"].isin(self.candidate_items)]
        self.ranking_ = (
            df.groupby("product_id")["basket_id"].nunique()
            .sort_values(ascending=False)
            .index.tolist()
        )
        return self

    def recommend(self, household_key: int, k: int) -> list:
        return self.ranking_[:k]


class PersonalRepeatPurchaseRecommender(BaseRecommender):
    """Ranks each household's own previously purchased items by recurrence,
    falling back to global popularity to fill remaining slots.
    """

    def __init__(self, candidate_items: set | None = None):
        self.candidate_items = candidate_items
        self.personal_map_: dict[int, list] = {}
        self.fallback_: PopularityRecommender | None = None

    def fit(self, train_interactions: pd.DataFrame) -> "PersonalRepeatPurchaseRecommender":
        df = train_interactions
        if self.candidate_items is not None:
            df = df[df["product_id"].isin(self.candidate_items)]

        counts = (
            df.groupby(["household_key", "product_id"])["basket_id"].nunique()
            .reset_index(name="n_baskets")
            .sort_values(["household_key", "n_baskets"], ascending=[True, False])
        )
        self.personal_map_ = counts.groupby("household_key")["product_id"].apply(list).to_dict()

        self.fallback_ = PopularityRecommender(self.candidate_items).fit(train_interactions)
        return self

    def recommend(self, household_key: int, k: int) -> list:
        items = list(self.personal_map_.get(household_key, []))
        if len(items) < k:
            for item in self.fallback_.recommend(household_key, k + len(items)):
                if item not in items:
                    items.append(item)
                if len(items) >= k:
                    break
        return items[:k]


class SegmentPopularityRecommender(BaseRecommender):
    """Ranks items by popularity within a household's demographic segment
    (`hh_comp_desc` by default), falling back to global popularity when the
    segment is unknown or too sparse to fill k slots.

    Extracted from notebooks/05_hybrid_and_cold_start.ipynb (section 1). This
    is the cold-start path for households with no candidate-item purchase
    history at all (18 of 2,499 in train) — the double fallback matters
    because only 32% of households have demographics (notebook 01).
    """

    def __init__(self, candidate_items: set, demographic_df: pd.DataFrame, segment_col: str = "hh_comp_desc"):
        self.candidate_items = candidate_items
        self.demographic_df = demographic_df
        self.segment_col = segment_col
        self.segment_rankings_: dict = {}
        self.household_to_segment_: dict = {}
        self.global_fallback_: PopularityRecommender | None = None

    def fit(self, train_interactions: pd.DataFrame) -> "SegmentPopularityRecommender":
        self.global_fallback_ = PopularityRecommender(self.candidate_items).fit(train_interactions)
        self.household_to_segment_ = (
            self.demographic_df.set_index("household_key")[self.segment_col].to_dict()
        )

        df = train_interactions[train_interactions["product_id"].isin(self.candidate_items)].copy()
        df["segment"] = df["household_key"].map(self.household_to_segment_)
        df = df.dropna(subset=["segment"])

        for segment, g in df.groupby("segment"):
            self.segment_rankings_[segment] = (
                g.groupby("product_id")["basket_id"].nunique()
                .sort_values(ascending=False).index.tolist()
            )
        return self

    def recommend(self, household_key: int, k: int) -> list:
        segment = self.household_to_segment_.get(household_key)
        items = list(self.segment_rankings_.get(segment, []))
        if len(items) < k:
            for item in self.global_fallback_.recommend(household_key, k + len(items)):
                if item not in items:
                    items.append(item)
                if len(items) >= k:
                    break
        return items[:k]
