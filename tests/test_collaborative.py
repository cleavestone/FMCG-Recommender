"""Unit tests for models.collaborative.ALSRecommender.

ALS's learned rankings on tiny synthetic data aren't meaningful, so these
check structural contracts (shapes, cold-start handling, candidate-universe
respect) rather than ranking quality — ranking quality is what the notebook
04 harness run against real data is for.
"""
import numpy as np
import pandas as pd

from fmcg_reco.models.collaborative import ALSRecommender


def _toy_train(n_households: int = 15, n_items: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_rows = 200
    return pd.DataFrame({
        "household_key": rng.integers(0, n_households, n_rows),
        "basket_id": np.arange(n_rows),
        "product_id": rng.integers(0, n_items, n_rows),
    })


def test_recommend_returns_k_items_from_candidate_universe():
    candidate_items = set(range(10))
    model = ALSRecommender(candidate_items, factors=4, iterations=3).fit(_toy_train())
    recs = model.recommend(household_key=0, k=5)
    assert len(recs) == 5
    assert set(recs) <= candidate_items


def test_recommend_returns_empty_for_household_unseen_in_train():
    candidate_items = set(range(10))
    model = ALSRecommender(candidate_items, factors=4, iterations=3).fit(_toy_train())
    assert model.recommend(household_key=9999, k=5) == []


def test_similar_items_excludes_query_item_and_respects_k():
    candidate_items = set(range(10))
    model = ALSRecommender(candidate_items, factors=4, iterations=3).fit(_toy_train())
    sims = model.similar_items(product_id=0, k=3)
    assert len(sims) == 3
    assert 0 not in sims


def test_similar_items_raises_for_item_outside_candidate_universe():
    candidate_items = set(range(10))
    model = ALSRecommender(candidate_items, factors=4, iterations=3).fit(_toy_train())
    try:
        model.similar_items(product_id=999, k=3)
        assert False, "expected KeyError for an item with no ALS factors"
    except KeyError:
        pass
