"""Unit tests for data.interactions.build_interaction_matrix."""
import numpy as np
import pandas as pd

from fmcg_reco.data.interactions import build_interaction_matrix


def _toy_train() -> pd.DataFrame:
    return pd.DataFrame({
        "household_key": [1, 1, 2],
        "basket_id":     [1, 2, 3],
        "product_id":    [10, 10, 20],
    })


def test_matrix_shape_matches_households_and_candidate_items():
    matrix, households, items, _h2i, _i2i = build_interaction_matrix(
        _toy_train(), candidate_items={10, 20, 30}
    )
    assert matrix.shape == (2, 3)  # 2 households, 3 candidate items (30 unpurchased but still a column)
    assert set(households) == {1, 2}
    assert set(items) == {10, 20, 30}


def test_confidence_formula_matches_hu_koren_volinsky():
    alpha = 15.0
    matrix, _households, _items, h2i, i2i = build_interaction_matrix(
        _toy_train(), candidate_items={10, 20}, alpha=alpha
    )
    # household 1 bought product 10 in 2 distinct baskets -> r=2
    expected = 1.0 + alpha * np.log1p(2)
    actual = matrix[h2i[1], i2i[10]]
    assert abs(actual - expected) < 1e-4


def test_excludes_items_outside_candidate_set():
    df = pd.DataFrame({
        "household_key": [1, 1],
        "basket_id": [1, 2],
        "product_id": [10, 99],
    })
    _, _, items, _, _ = build_interaction_matrix(df, candidate_items={10})
    assert items == [10]
