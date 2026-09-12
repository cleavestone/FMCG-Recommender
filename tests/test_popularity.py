"""Unit tests for models.popularity — PopularityRecommender,
PersonalRepeatPurchaseRecommender, and SegmentPopularityRecommender against
small synthetic interaction sets.
"""
import pandas as pd

from fmcg_reco.models.popularity import (
    PersonalRepeatPurchaseRecommender,
    PopularityRecommender,
    SegmentPopularityRecommender,
)


def _toy_train() -> pd.DataFrame:
    # household 1 bought product 10 in three separate baskets (a clear repeat),
    # product 100 is globally the most popular (bought by both households).
    return pd.DataFrame({
        "household_key": [1, 1, 1, 1, 2, 2],
        "basket_id":     [1, 2, 3, 4, 5, 6],
        "product_id":    [10, 10, 10, 100, 100, 20],
    })


def test_popularity_ranks_by_distinct_basket_count():
    model = PopularityRecommender().fit(_toy_train())
    # product 100 appears in 2 distinct baskets, product 10 in 3 -> product 10 ranks first
    assert model.recommend(household_key=999, k=1) == [10]


def test_popularity_ignores_household_identity():
    model = PopularityRecommender().fit(_toy_train())
    assert model.recommend(1, k=3) == model.recommend(2, k=3)


def test_popularity_respects_candidate_universe():
    model = PopularityRecommender(candidate_items={20, 100}).fit(_toy_train())
    recs = model.recommend(1, k=3)
    assert 10 not in recs
    assert set(recs) <= {20, 100}


def test_personal_repeat_purchase_prioritises_own_history():
    model = PersonalRepeatPurchaseRecommender().fit(_toy_train())
    recs = model.recommend(household_key=1, k=1)
    assert recs == [10]  # household 1's own most-repeated product


def test_personal_repeat_purchase_falls_back_to_popularity_when_short_on_history():
    model = PersonalRepeatPurchaseRecommender().fit(_toy_train())
    # household 2 only has 2 distinct products in history; asking for 3 must
    # fall back to global popularity to fill the remaining slot without duplicates.
    recs = model.recommend(household_key=2, k=3)
    assert len(recs) == 3
    assert len(set(recs)) == 3
    assert set(recs[:2]) == {100, 20}  # household 2's own history, ranked first


def test_personal_repeat_purchase_handles_unseen_household():
    model = PersonalRepeatPurchaseRecommender().fit(_toy_train())
    recs = model.recommend(household_key=999, k=2)
    assert recs == [10, 100]  # pure fallback to global popularity


def _toy_demographic() -> pd.DataFrame:
    return pd.DataFrame({
        "household_key": [1, 2, 3],
        "hh_comp_desc": ["Single Male", "Single Male", "2 Adults Kids"],
    })


def _toy_segment_train() -> pd.DataFrame:
    # households 1 and 2 share a segment and both favour product 50;
    # household 3 is in a different segment and favours product 60.
    # household 4 has no demographic row at all.
    return pd.DataFrame({
        "household_key": [1, 1, 2, 2, 3, 3, 4],
        "basket_id":     [1, 2, 3, 4, 5, 6, 7],
        "product_id":    [50, 50, 50, 40, 60, 60, 40],
    })


def test_segment_popularity_ranks_within_household_segment():
    model = SegmentPopularityRecommender(
        candidate_items={40, 50, 60}, demographic_df=_toy_demographic()
    ).fit(_toy_segment_train())
    # household 3's segment ("2 Adults Kids") favours 60 over 40
    assert model.recommend(household_key=3, k=1) == [60]


def test_segment_popularity_falls_back_to_global_for_unknown_segment():
    model = SegmentPopularityRecommender(
        candidate_items={40, 50, 60}, demographic_df=_toy_demographic()
    ).fit(_toy_segment_train())
    # household 4 has no demographic row -> pure global-popularity fallback
    recs = model.recommend(household_key=4, k=1)
    assert recs == model.global_fallback_.recommend(4, 1)
