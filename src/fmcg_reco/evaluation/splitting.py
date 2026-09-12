"""Time-based train/val/test splitting for implicit-feedback interaction data.

Extracted from notebooks/02_eval_harness_and_baselines.ipynb (section 2).
"""
import pandas as pd


def time_based_split(
    df: pd.DataFrame, train_end: int, val_end: int, test_end: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions on `week_no` into train/val/test, each a disjoint,
    contiguous window. A single global cutoff avoids the future-leaks-into-
    past leakage a per-row random split would introduce.
    """
    train = df[df["week_no"] <= train_end]
    val = df[(df["week_no"] > train_end) & (df["week_no"] <= val_end)]
    test = df[(df["week_no"] > val_end) & (df["week_no"] <= test_end)]
    return train, val, test


def relevant_items_by_household(df: pd.DataFrame) -> dict[int, set[int]]:
    """Map each household to the set of distinct products it purchased in `df`."""
    return df.groupby("household_key")["product_id"].apply(set).to_dict()
