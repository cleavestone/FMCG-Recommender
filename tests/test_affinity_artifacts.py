"""Round-trip test for artifacts.save_affinity_artifacts / load_affinity_artifacts."""
import pandas as pd

from fmcg_reco.artifacts import load_affinity_artifacts, save_affinity_artifacts
from fmcg_reco.models.affinity import BasketAffinityRecommender

CANDIDATE_ITEMS = {1, 2, 3, 4, 5}


def _toy_train() -> pd.DataFrame:
    rows = []
    basket_id = 0
    for household in (10, 11, 12, 13, 14, 15, 16, 17):
        for product_id in (1, 2, 3):
            rows.append({"basket_id": basket_id, "product_id": product_id, "household_key": household})
        basket_id += 1
    for product_id in (1, 2):
        rows.append({"basket_id": basket_id, "product_id": product_id, "household_key": 1})
    basket_id += 1
    for _ in range(10):
        for product_id in (4, 5):
            rows.append({"basket_id": basket_id, "product_id": product_id, "household_key": 2})
        basket_id += 1
    return pd.DataFrame(rows)


def test_loaded_affinity_model_matches_original(tmp_path):
    original = BasketAffinityRecommender(CANDIDATE_ITEMS, min_support=0.05, min_confidence=0.1).fit(_toy_train())
    save_affinity_artifacts(original, tmp_path)

    loaded = load_affinity_artifacts(tmp_path)

    assert loaded.recommend(1, k=5) == original.recommend(1, k=5)
    assert loaded.recommend(9999, k=5) == original.recommend(9999, k=5)
    pd.testing.assert_frame_equal(
        loaded.explain(1, 3).reset_index(drop=True),
        original.explain(1, 3).reset_index(drop=True),
    )
