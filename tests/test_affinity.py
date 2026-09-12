"""Unit tests for models.affinity.BasketAffinityRecommender."""
import pandas as pd

from fmcg_reco.models.affinity import BasketAffinityRecommender

CANDIDATE_ITEMS = {1, 2, 3, 4, 5}


def _toy_train() -> pd.DataFrame:
    # household 1 already owns bread(1) + peanut_butter(2); a strong {1,2}->3
    # rule exists from other households' baskets, so household 1 should be
    # recommended jelly(3) - a product it has never bought.
    # Noise baskets (unrelated items 4,5) are needed so jelly's overall support
    # is diluted below 1 - without them every basket contains 1 and/or 2,
    # making lift({1,2}->3) compute to exactly 1.0 (no real association).
    rows = []
    basket_id = 0
    for household in (10, 11, 12, 13, 14, 15, 16, 17):
        for product_id in (1, 2, 3):
            rows.append({"basket_id": basket_id, "product_id": product_id, "household_key": household})
        basket_id += 1
    # household 1: owns bread + peanut_butter, never bought jelly
    for product_id in (1, 2):
        rows.append({"basket_id": basket_id, "product_id": product_id, "household_key": 1})
    basket_id += 1
    for _ in range(10):
        for product_id in (4, 5):
            rows.append({"basket_id": basket_id, "product_id": product_id, "household_key": 2})
        basket_id += 1
    return pd.DataFrame(rows)


def test_recommends_a_never_purchased_item_via_affinity():
    model = BasketAffinityRecommender(candidate_items=CANDIDATE_ITEMS, min_support=0.05, min_confidence=0.1).fit(_toy_train())
    recs = model.recommend(household_key=1, k=5)
    assert 3 in recs
    assert 1 not in recs and 2 not in recs  # already owned - never recommended


def test_never_recommends_an_owned_item():
    model = BasketAffinityRecommender(candidate_items=CANDIDATE_ITEMS, min_support=0.05, min_confidence=0.1).fit(_toy_train())
    owned = model.household_items_[1]
    recs = model.recommend(household_key=1, k=10)
    assert not (set(recs) & owned)


def test_unknown_household_gets_no_recommendations():
    model = BasketAffinityRecommender(candidate_items=CANDIDATE_ITEMS, min_support=0.05, min_confidence=0.1).fit(_toy_train())
    assert model.recommend(household_key=9999, k=5) == []


def test_explain_returns_the_matching_rule():
    model = BasketAffinityRecommender(candidate_items=CANDIDATE_ITEMS, min_support=0.05, min_confidence=0.1).fit(_toy_train())
    explanation = model.explain(household_key=1, item_id=3)
    assert len(explanation) >= 1
    assert frozenset({1, 2}) in explanation["antecedents"].values
