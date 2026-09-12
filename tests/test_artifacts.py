"""Round-trip test for artifacts.save_artifacts / load_artifacts.

The real test here is behavioural: a loaded HybridRecommender must produce
identical recommendations to the one that was fit and saved, for both warm
and cold households, without ever calling .fit() again.
"""
import numpy as np
import pandas as pd

from fmcg_reco.artifacts import load_artifacts, save_artifacts
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
    household_keys = rng.integers(0, warm_households, n_rows)
    return pd.DataFrame({
        "household_key": household_keys,
        "basket_id": np.arange(n_rows),
        "product_id": rng.integers(0, n_items, n_rows),
    })


def test_loaded_hybrid_matches_original_for_warm_and_cold_households(tmp_path):
    candidate_items = set(range(20))
    product_df, demographic_df = _toy_products(), _toy_demographic()
    original = HybridRecommender(product_df, demographic_df, candidate_items).fit(_toy_train())

    save_artifacts(original, tmp_path)
    loaded = load_artifacts(tmp_path, product_df, demographic_df)

    warm_household, cold_household = 0, 29
    assert loaded.recommend(warm_household, k=5) == original.recommend(warm_household, k=5)
    assert loaded.recommend(cold_household, k=5) == original.recommend(cold_household, k=5)
    assert loaded.similar_items(0, k=3) == original.similar_items(0, k=3)


def test_loaded_hybrid_does_not_require_refitting(tmp_path):
    candidate_items = set(range(20))
    product_df, demographic_df = _toy_products(), _toy_demographic()
    original = HybridRecommender(product_df, demographic_df, candidate_items).fit(_toy_train())
    save_artifacts(original, tmp_path)

    loaded = load_artifacts(tmp_path, product_df, demographic_df)
    # a freshly-loaded object must be immediately usable - no train_interactions in sight
    assert loaded.recommend(0, k=3) is not None
