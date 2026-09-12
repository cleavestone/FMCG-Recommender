"""Content-based recommender: item hierarchy similarity + household profile.

Extracted from notebooks/03_content_based_model.ipynb (sections 3-4).
Solves item cold-start (a new product has a content vector immediately) but
not household cold-start (recommend() returns [] for a household with no
candidate-item purchase history) — see models.hybrid for the fallback that
covers that gap.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from fmcg_reco.features.item_features import build_item_feature_matrix
from fmcg_reco.models.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    def __init__(self, product_df: pd.DataFrame, candidate_items: set):
        self.product_df = product_df
        self.candidate_items = candidate_items
        self.item_matrix_ = None
        self.item_ids_: list = []
        self.item_id_to_idx_: dict = {}
        self.profile_matrix_ = None
        self.household_id_to_idx_: dict = {}

    def fit(self, train_interactions: pd.DataFrame) -> "ContentBasedRecommender":
        self.item_matrix_, self.item_ids_, _ = build_item_feature_matrix(self.product_df, self.candidate_items)
        self.item_id_to_idx_ = {pid: i for i, pid in enumerate(self.item_ids_)}

        df = train_interactions[train_interactions["product_id"].isin(self.candidate_items)].copy()
        df["item_idx"] = df["product_id"].map(self.item_id_to_idx_)
        weights = (
            df.groupby(["household_key", "item_idx"])["basket_id"].nunique()
            .reset_index(name="w")
        )

        households = sorted(weights["household_key"].unique())
        self.household_id_to_idx_ = {h: i for i, h in enumerate(households)}

        row = weights["household_key"].map(self.household_id_to_idx_).to_numpy()
        col = weights["item_idx"].to_numpy()
        data = weights["w"].to_numpy(dtype=float)
        weight_matrix = csr_matrix((data, (row, col)), shape=(len(households), len(self.item_ids_)))

        profile_matrix = weight_matrix @ self.item_matrix_
        norms = np.sqrt(profile_matrix.multiply(profile_matrix).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        self.profile_matrix_ = diags(1.0 / norms) @ profile_matrix
        return self

    def recommend(self, household_key: int, k: int) -> list:
        idx = self.household_id_to_idx_.get(household_key)
        if idx is None:
            return []
        profile = self.profile_matrix_[idx]
        scores = np.asarray((profile @ self.item_matrix_.T).todense()).ravel()
        top_idx = np.argsort(-scores)[:k]
        return [self.item_ids_[i] for i in top_idx]

    def similar_items(self, product_id: int, k: int) -> list:
        idx = self.item_id_to_idx_[product_id]
        scores = np.asarray((self.item_matrix_[idx] @ self.item_matrix_.T).todense()).ravel()
        order = np.argsort(-scores)
        return [self.item_ids_[i] for i in order if i != idx][:k]
