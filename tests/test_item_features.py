"""Unit tests for features.item_features."""
import pandas as pd

from fmcg_reco.features.item_features import build_item_feature_matrix


def _toy_products() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": [1, 2, 3, 4],
        "department": ["PRODUCE", "PRODUCE", "GROCERY", "GROCERY"],
        "commodity_desc": ["TROPICAL FRUIT", "TROPICAL FRUIT", "SOFT DRINKS", "SOFT DRINKS"],
        "sub_commodity_desc": ["BANANAS", "MANGO", "COLA", "COLA"],
        "brand": ["National", "National", "Private", "Private"],
        "manufacturer": [1, 1, 2, 2],
    })


def test_returns_one_row_per_candidate_item():
    matrix, item_ids, _ = build_item_feature_matrix(_toy_products(), candidate_items={1, 2, 3})
    assert matrix.shape[0] == 3
    assert set(item_ids) == {1, 2, 3}


def test_excludes_items_outside_candidate_set():
    _, item_ids, _ = build_item_feature_matrix(_toy_products(), candidate_items={1, 2})
    assert 3 not in item_ids and 4 not in item_ids


def test_same_department_items_are_more_similar_than_different_departments():
    matrix, item_ids, _ = build_item_feature_matrix(_toy_products(), candidate_items={1, 2, 3, 4})
    idx = {pid: i for i, pid in enumerate(item_ids)}

    banana, mango, cola1 = idx[1], idx[2], idx[3]
    sim_same_dept = (matrix[banana] @ matrix[mango].T).toarray()[0, 0]
    sim_diff_dept = (matrix[banana] @ matrix[cola1].T).toarray()[0, 0]
    assert sim_same_dept > sim_diff_dept


def test_identical_hierarchy_items_are_tied():
    # products 3 and 4 share every hierarchy field (the pack-size-blind-spot case).
    matrix, item_ids, _ = build_item_feature_matrix(_toy_products(), candidate_items={3, 4})
    idx = {pid: i for i, pid in enumerate(item_ids)}
    sim = (matrix[idx[3]] @ matrix[idx[4]].T).toarray()[0, 0]
    assert abs(sim - 1.0) < 1e-9
