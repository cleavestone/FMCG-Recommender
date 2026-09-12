"""Collaborative filtering via ALS (Alternating Least Squares).

Extracted from notebooks/04_collaborative_filtering.ipynb (section 2).
Best config found by grid search against validation Recall@20:
factors=32, regularization=0.01, iterations=15 — fewer factors won because
the interaction matrix is small and sparse (2,481 households x 5,000 items,
5.3% dense); more factors overfit that sparse signal rather than helping.

On the validation split this underperformed both `popularity` and
`buy_it_again` (MAP@10 0.048 vs 0.124 and 0.326) — kept as one input signal
for models.hybrid, not used standalone.
"""
import pandas as pd
from implicit.als import AlternatingLeastSquares

from fmcg_reco.data.interactions import DEFAULT_ALPHA, build_interaction_matrix
from fmcg_reco.models.base import BaseRecommender


class ALSRecommender(BaseRecommender):
    def __init__(
        self,
        candidate_items: set,
        factors: int = 32,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = DEFAULT_ALPHA,
        random_state: int = 42,
    ):
        self.candidate_items = candidate_items
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        self.user_items_ = None
        self.household_ids_: list = []
        self.item_ids_: list = []
        self.household_id_to_idx_: dict = {}
        self.item_id_to_idx_: dict = {}
        self.model_: AlternatingLeastSquares | None = None

    def fit(self, train_interactions: pd.DataFrame) -> "ALSRecommender":
        (self.user_items_, self.household_ids_, self.item_ids_,
         self.household_id_to_idx_, self.item_id_to_idx_) = build_interaction_matrix(
            train_interactions, self.candidate_items, alpha=self.alpha
        )
        self.model_ = AlternatingLeastSquares(
            factors=self.factors, regularization=self.regularization,
            iterations=self.iterations, random_state=self.random_state,
        )
        self.model_.fit(self.user_items_, show_progress=False)
        return self

    def recommend(self, household_key: int, k: int) -> list:
        idx = self.household_id_to_idx_.get(household_key)
        if idx is None:
            return []  # ALS has no factors for a household absent from train
        ids, _ = self.model_.recommend(
            idx, self.user_items_[idx], N=k, filter_already_liked_items=False,
        )
        return [self.item_ids_[i] for i in ids]

    def similar_items(self, product_id: int, k: int) -> list:
        idx = self.item_id_to_idx_.get(product_id)
        if idx is None:
            raise KeyError(f"product_id {product_id} has no ALS factors (unseen in train)")
        ids, _ = self.model_.similar_items(idx, N=k + 1)
        return [self.item_ids_[i] for i in ids if i != idx][:k]
