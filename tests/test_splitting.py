"""Unit tests for evaluation.splitting — boundary correctness and no leakage."""
import pandas as pd

from fmcg_reco.evaluation.splitting import relevant_items_by_household, time_based_split


def _toy_interactions() -> pd.DataFrame:
    return pd.DataFrame({
        "household_key": [1, 1, 1, 2, 2, 2],
        "product_id": [10, 20, 30, 10, 40, 50],
        "week_no": [1, 88, 89, 95, 96, 102],
    })


def test_split_boundaries_are_inclusive_and_disjoint():
    df = _toy_interactions()
    train, val, test = time_based_split(df, train_end=88, val_end=95, test_end=102)

    assert set(train["week_no"]) == {1, 88}
    assert set(val["week_no"]) == {89, 95}
    assert set(test["week_no"]) == {96, 102}


def test_no_future_week_leaks_into_train():
    df = _toy_interactions()
    train, val, test = time_based_split(df, train_end=88, val_end=95, test_end=102)

    assert train["week_no"].max() <= 88
    assert val["week_no"].min() > 88
    assert test["week_no"].min() > 95


def test_splits_partition_the_input_exactly():
    df = _toy_interactions()
    train, val, test = time_based_split(df, train_end=88, val_end=95, test_end=102)

    assert len(train) + len(val) + len(test) == len(df)
    assert set(train.index) | set(val.index) | set(test.index) == set(df.index)


def test_relevant_items_by_household():
    df = _toy_interactions()
    relevant = relevant_items_by_household(df)

    assert relevant[1] == {10, 20, 30}
    assert relevant[2] == {10, 40, 50}
