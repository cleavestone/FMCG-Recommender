"""Unit tests for models.content.ContentBasedRecommender."""
import pandas as pd

from fmcg_reco.models.content import ContentBasedRecommender


def _toy_products() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": [1, 2, 3],
        "department": ["PRODUCE", "PRODUCE", "GROCERY"],
        "commodity_desc": ["TROPICAL FRUIT", "TROPICAL FRUIT", "SOFT DRINKS"],
        "sub_commodity_desc": ["BANANAS", "MANGO", "COLA"],
        "brand": ["National", "National", "Private"],
        "manufacturer": [1, 1, 2],
    })


def _toy_train() -> pd.DataFrame:
    # household 1 exclusively buys bananas -> profile should favour mango (same
    # commodity) over cola (different department) when recommending unseen items.
    return pd.DataFrame({
        "household_key": [1, 1, 1],
        "basket_id": [1, 2, 3],
        "product_id": [1, 1, 1],
    })


def test_recommends_similar_items_to_purchase_history():
    model = ContentBasedRecommender(_toy_products(), candidate_items={1, 2, 3}).fit(_toy_train())
    recs = model.recommend(household_key=1, k=3)
    # mango (same commodity as bananas) should outrank cola (different department)
    assert recs.index(2) < recs.index(3)


def test_household_with_no_history_gets_empty_recommendations():
    model = ContentBasedRecommender(_toy_products(), candidate_items={1, 2, 3}).fit(_toy_train())
    assert model.recommend(household_key=999, k=3) == []


def test_similar_items_excludes_the_query_item_itself():
    model = ContentBasedRecommender(_toy_products(), candidate_items={1, 2, 3}).fit(_toy_train())
    sims = model.similar_items(product_id=1, k=2)
    assert 1 not in sims
    assert sims[0] == 2  # mango is the closest neighbour to bananas
