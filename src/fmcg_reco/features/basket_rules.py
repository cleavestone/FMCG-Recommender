"""Association rule mining over basket-level co-purchase data (FP-Growth).

Extracted from notebooks/07_market_basket_analysis.ipynb (sections 1-3).
Parameters (min_support=0.002, min_lift=1.2, min_confidence=0.1) were
chosen there after a small sweep and validated against held-out
validation-period baskets (train/val confidence correlation 0.70).
"""
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

RULE_COLUMNS = ["antecedents", "consequents", "consequent_item", "support", "confidence", "lift"]


def mine_association_rules(
    train_interactions: pd.DataFrame,
    candidate_items: set,
    min_support: float = 0.002,
    min_lift: float = 1.2,
    min_confidence: float = 0.1,
    max_len: int = 3,
) -> pd.DataFrame:
    """Mine single-consequent association rules from basket co-purchases.

    Returns a DataFrame with `antecedents`/`consequents` as frozenset[int],
    `consequent_item` as int, plus support/confidence/lift, sorted by lift
    descending. Empty (but correctly-shaped) if no baskets or itemsets
    survive filtering.
    """
    basket_items = (
        train_interactions[train_interactions["product_id"].isin(candidate_items)]
        .groupby("basket_id")["product_id"].apply(set)
    )
    basket_items = basket_items[basket_items.apply(len) >= 2]
    if basket_items.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)

    encoder = TransactionEncoder()
    onehot = encoder.fit_transform(basket_items.tolist(), sparse=True)
    # mlxtend requires string column names for sparse input with integer product ids
    encoded = pd.DataFrame.sparse.from_spmatrix(onehot, columns=[str(c) for c in encoder.columns_])

    frequent_itemsets = fpgrowth(encoded, min_support=min_support, use_colnames=True, max_len=max_len)
    if frequent_itemsets.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)

    rules = association_rules(
        frequent_itemsets, num_itemsets=len(basket_items), metric="lift", min_threshold=min_lift
    )
    rules = rules[rules["consequents"].apply(len) == 1].copy()
    rules = rules[rules["confidence"] >= min_confidence]
    if rules.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)

    rules["antecedents"] = rules["antecedents"].apply(lambda s: frozenset(int(x) for x in s))
    rules["consequents"] = rules["consequents"].apply(lambda s: frozenset(int(x) for x in s))
    rules["consequent_item"] = rules["consequents"].apply(lambda s: next(iter(s)))
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)[RULE_COLUMNS]
