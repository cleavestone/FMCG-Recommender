"""Implicit-feedback interaction matrix construction for collaborative filtering.

Extracted from notebooks/04_collaborative_filtering.ipynb (section 1).

Uses a confidence-weighted count (Hu, Koren & Volinsky, 2008), not raw
purchase counts: `confidence = 1 + alpha * log1p(r)`, where `r` is the
number of distinct baskets a household bought an item in. This compresses
the long tail of very frequent repeat purchases (notebook 01: baskets per
household range from 1 to 1,300) instead of letting a handful of heavy
shoppers dominate the factorization.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DEFAULT_ALPHA = 15.0


def build_interaction_matrix(
    train_interactions: pd.DataFrame, candidate_items: set, alpha: float = DEFAULT_ALPHA
):
    """Returns (confidence matrix, household_ids, item_ids, household_id_to_idx, item_id_to_idx).

    The matrix is (n_households x n_items), dtype float32 as `implicit` expects.
    """
    df = train_interactions[train_interactions["product_id"].isin(candidate_items)]
    counts = (
        df.groupby(["household_key", "product_id"])["basket_id"].nunique()
        .reset_index(name="r")
    )

    households = sorted(counts["household_key"].unique())
    items = sorted(candidate_items)
    household_to_idx = {h: i for i, h in enumerate(households)}
    item_to_idx = {p: i for i, p in enumerate(items)}

    confidence = (1.0 + alpha * np.log1p(counts["r"].to_numpy())).astype(np.float32)
    row = counts["household_key"].map(household_to_idx).to_numpy()
    col = counts["product_id"].map(item_to_idx).to_numpy()
    matrix = csr_matrix((confidence, (row, col)), shape=(len(households), len(items)))
    return matrix, households, items, household_to_idx, item_to_idx
