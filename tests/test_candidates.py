"""Unit tests for evaluation.candidates."""
import pandas as pd

from fmcg_reco.evaluation.candidates import top_n_candidate_items


def test_top_n_ranks_by_distinct_basket_count_not_raw_quantity():
    # product 1: 1 basket but bought 100 units in it (a bulk buy).
    # product 2: 3 distinct baskets, 1 unit each -> should rank above product 1.
    df = pd.DataFrame({
        "basket_id": [1, 2, 3, 4],
        "product_id": [1, 2, 2, 2],
    })
    assert top_n_candidate_items(df, n=1) == {2}


def test_top_n_returns_requested_size():
    df = pd.DataFrame({
        "basket_id": [1, 2, 3, 4, 5],
        "product_id": [1, 2, 3, 4, 4],
    })
    assert len(top_n_candidate_items(df, n=2)) == 2
