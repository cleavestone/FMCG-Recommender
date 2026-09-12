"""Unit tests for models.hybrid.HybridRecommender.

HybridRecommender trains a real ALSRecommender internally (factors=32), so
the synthetic dataset needs to be large enough for that to run cleanly —
these tests check structural contracts (cold-start routing, candidate
universe, similar_items delegation), not exact ranking quality.
"""
import numpy as np
import pandas as pd

from fmcg_reco.models.hybrid import HybridRecommender


def _toy_products(n_items: int = 20) -> pd.DataFrame:
    departments = ["PRODUCE", "GROCERY", "DAIRY", "MEAT"]
    return pd.DataFrame({
        "product_id": list(range(n_items)),
        "department": [departments[i % len(departments)] for i in range(n_items)],
        "commodity_desc": [f"COMMODITY_{i % 5}" for i in range(n_items)],
        "sub_commodity_desc": [f"SUBCOMMODITY_{i}" for i in range(n_items)],
        "brand": ["National" if i % 2 == 0 else "Private" for i in range(n_items)],
        "manufacturer": [i % 3 for i in range(n_items)],
    })


def _toy_demographic(n_households: int = 30) -> pd.DataFrame:
    segments = ["Single Male", "Single Female", "2 Adults No Kids"]
    return pd.DataFrame({
        "household_key": list(range(n_households)),
        "hh_comp_desc": [segments[i % len(segments)] for i in range(n_households)],
    })


def _toy_train(n_households: int = 30, n_items: int = 20, warm_households: int = 28, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_rows = 400
    # households [warm_households, n_households) get zero rows -> genuinely cold.
    household_keys = rng.integers(0, warm_households, n_rows)
    return pd.DataFrame({
        "household_key": household_keys,
        "basket_id": np.arange(n_rows),
        "product_id": rng.integers(0, n_items, n_rows),
    })


def test_cold_household_routes_to_segment_popularity():
    candidate_items = set(range(20))
    model = HybridRecommender(_toy_products(), _toy_demographic(), candidate_items).fit(_toy_train())

    cold_household = 29  # outside the warm_households=28 range used to build train
    assert cold_household not in model.warm_households_
    recs = model.recommend(cold_household, k=5)
    assert recs == model.segment_model_.recommend(cold_household, k=5)


def test_warm_household_gets_blended_recommendations():
    candidate_items = set(range(20))
    model = HybridRecommender(_toy_products(), _toy_demographic(), candidate_items).fit(_toy_train())

    warm_household = 0
    assert warm_household in model.warm_households_
    recs = model.recommend(warm_household, k=5)
    assert len(recs) == 5
    assert set(recs) <= candidate_items


def test_default_weights_reduce_to_pure_repeat_purchase():
    # notebook 05's finding: the tuned weights collapse to content=0, cf=0,
    # repeat=1.0 -> recommend() for a warm household must exactly match
    # PersonalRepeatPurchaseRecommender's own ranking.
    candidate_items = set(range(20))
    model = HybridRecommender(_toy_products(), _toy_demographic(), candidate_items).fit(_toy_train())

    warm_household = 0
    assert model.recommend(warm_household, k=5) == model.repeat_model_.recommend(warm_household, k=5)


def test_similar_items_delegates_to_content_model():
    candidate_items = set(range(20))
    model = HybridRecommender(_toy_products(), _toy_demographic(), candidate_items).fit(_toy_train())

    assert model.similar_items(0, k=3) == model.content_model_.similar_items(0, k=3)
