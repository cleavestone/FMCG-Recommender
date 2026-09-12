"""Candidate universe scoping — restricts what a model is allowed to recommend.

Extracted from notebooks/02_eval_harness_and_baselines.ipynb (section 3).
Every model evaluated against the same harness must share the same
candidate universe, or comparisons between them are meaningless.
"""
import pandas as pd


def top_n_candidate_items(train_interactions: pd.DataFrame, n: int) -> set[int]:
    """Return the n most-purchased products in train, ranked by distinct
    basket count (robust to one heavy-shopping household inflating a raw
    quantity-based ranking — see notebook 01 finding: baskets/household
    range from 1 to 1,300).
    """
    popularity = train_interactions.groupby("product_id")["basket_id"].nunique()
    return set(popularity.sort_values(ascending=False).head(n).index)
