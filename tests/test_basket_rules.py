"""Unit tests for features.basket_rules.mine_association_rules."""
import pandas as pd

from fmcg_reco.features.basket_rules import mine_association_rules


def _toy_train() -> pd.DataFrame:
    # bread+peanut_butter -> jelly is a strong, clear association (8/10 co-occurring
    # baskets also have jelly). The noise baskets (items 40/41, unrelated, 2 items
    # each so they survive the >=2-items filter) matter for correctness: without
    # them every basket contains {1,2}, which makes jelly's support equal to
    # confidence({1,2}->3) and lift compute to exactly 1.0 - i.e. no measurable
    # association at all, even though the co-occurrence "looks" strong. The noise
    # dilutes jelly's overall support so a genuine lift > 1 shows up.
    rows = []
    basket_id = 0
    # 8 baskets with bread(1) + peanut_butter(2) + jelly(3)
    for _ in range(8):
        for product_id in (1, 2, 3):
            rows.append({"basket_id": basket_id, "product_id": product_id, "household_key": 1})
        basket_id += 1
    # 2 baskets with bread + peanut_butter but no jelly
    for _ in range(2):
        for product_id in (1, 2):
            rows.append({"basket_id": basket_id, "product_id": product_id, "household_key": 1})
        basket_id += 1
    # 10 unrelated 2-item noise baskets (dilutes jelly's overall support)
    for _ in range(10):
        for product_id in (40, 41):
            rows.append({"basket_id": basket_id, "product_id": product_id, "household_key": 2})
        basket_id += 1
    return pd.DataFrame(rows)


def test_finds_the_strong_association():
    rules = mine_association_rules(
        _toy_train(), candidate_items={1, 2, 3, 40, 41}, min_support=0.05, min_lift=1.2, min_confidence=0.1
    )
    assert not rules.empty
    match = rules[
        (rules["antecedents"] == frozenset({1, 2})) & (rules["consequent_item"] == 3)
    ]
    assert len(match) == 1
    assert match.iloc[0]["confidence"] >= 0.75  # 8/10 baskets with {1,2} also have 3
    assert match.iloc[0]["lift"] > 1.2  # a genuine association, not just coincidental co-occurrence


def test_excludes_items_outside_candidate_set():
    rules = mine_association_rules(
        _toy_train(), candidate_items={1, 2}, min_support=0.05, min_lift=1.0, min_confidence=0.1
    )
    all_items = set()
    for s in rules["antecedents"]:
        all_items |= s
    for s in rules["consequents"]:
        all_items |= s
    assert 3 not in all_items  # jelly excluded once it's outside the candidate universe


def test_empty_when_no_multi_item_baskets():
    single_item_only = pd.DataFrame({
        "basket_id": [1, 2, 3],
        "product_id": [10, 20, 30],
        "household_key": [1, 1, 1],
    })
    rules = mine_association_rules(single_item_only, candidate_items={10, 20, 30})
    assert rules.empty
    assert list(rules.columns) == ["antecedents", "consequents", "consequent_item", "support", "confidence", "lift"]
