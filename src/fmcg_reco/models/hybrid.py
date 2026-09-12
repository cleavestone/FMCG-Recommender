"""Hybrid recommender: rank-fusion of content, collaborative, and
repeat-purchase signals, with demographic-segment cold-start fallback.

Extracted from notebooks/05_hybrid_and_cold_start.ipynb (sections 3-5).

Honest result from that notebook's weight grid search (tuned against
validation Recall@20): the winning weights were content=0, cf=0, repeat=1.0
— i.e. the best-scoring "hybrid" is mathematically identical to
PersonalRepeatPurchaseRecommender for warm households. This is a real
property of linear rank-fusion when one signal dominates the region a
ranking metric rewards, not a bug — see the notebook 05 findings for the
full reasoning and why it motivates a learning-to-rank reranker (roadmap
P4) rather than a bigger grid search here.

Content-based is kept and still reachable via `similar_items()` for the
cold-start/discovery use case demonstrated in that notebook (scoring an
item with zero purchase history), even though it doesn't win the blend.
"""
import pandas as pd

from fmcg_reco.models.base import BaseRecommender
from fmcg_reco.models.collaborative import ALSRecommender
from fmcg_reco.models.content import ContentBasedRecommender
from fmcg_reco.models.popularity import (
    PersonalRepeatPurchaseRecommender,
    SegmentPopularityRecommender,
)

DEFAULT_WEIGHTS = {"content": 0.0, "cf": 0.0, "repeat": 1.0}
DEFAULT_N_PER_SIGNAL = 300


class HybridRecommender(BaseRecommender):
    def __init__(
        self,
        product_df: pd.DataFrame,
        demographic_df: pd.DataFrame,
        candidate_items: set,
        weights: dict | None = None,
        n_per_signal: int = DEFAULT_N_PER_SIGNAL,
    ):
        self.product_df = product_df
        self.demographic_df = demographic_df
        self.candidate_items = candidate_items
        self.weights = weights if weights is not None else dict(DEFAULT_WEIGHTS)
        self.n_per_signal = n_per_signal
        self.content_model_: ContentBasedRecommender | None = None
        self.als_model_: ALSRecommender | None = None
        self.repeat_model_: PersonalRepeatPurchaseRecommender | None = None
        self.segment_model_: SegmentPopularityRecommender | None = None
        self.warm_households_: set = set()

    def fit(self, train_interactions: pd.DataFrame) -> "HybridRecommender":
        self.content_model_ = ContentBasedRecommender(self.product_df, self.candidate_items).fit(train_interactions)
        self.als_model_ = ALSRecommender(
            candidate_items=self.candidate_items, factors=32, regularization=0.01, iterations=15
        ).fit(train_interactions)
        self.repeat_model_ = PersonalRepeatPurchaseRecommender(candidate_items=self.candidate_items).fit(train_interactions)
        self.segment_model_ = SegmentPopularityRecommender(self.candidate_items, self.demographic_df).fit(train_interactions)
        self.warm_households_ = set(self.content_model_.household_id_to_idx_)
        return self

    def _blend(self, household_key: int, k: int) -> list:
        combined: dict = {}
        for name, model in [
            ("content", self.content_model_),
            ("cf", self.als_model_),
            ("repeat", self.repeat_model_),
        ]:
            ranked = model.recommend(household_key, self.n_per_signal)
            n = len(ranked)
            weight = self.weights[name]
            for rank, item in enumerate(ranked):
                score = 1.0 - (rank / n) if n else 0.0
                combined[item] = combined.get(item, 0.0) + weight * score
        ranked = sorted(combined.items(), key=lambda pair: -pair[1])
        return [item for item, _ in ranked[:k]]

    def recommend(self, household_key: int, k: int) -> list:
        if household_key not in self.warm_households_:
            return self.segment_model_.recommend(household_key, k)
        return self._blend(household_key, k)

    def similar_items(self, product_id: int, k: int) -> list:
        return self.content_model_.similar_items(product_id, k)
