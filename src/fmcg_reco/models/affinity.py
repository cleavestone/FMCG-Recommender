"""Basket-affinity recommender: association-rule-driven cross-sell / basket
growth, complementary to (not a replacement for) HybridRecommender.

Extracted from notebooks/07_market_basket_analysis.ipynb (sections 5-7).

This answers a different question than every other model in this project:
not "what will this household buy again" but "what should they add that
they've never bought, because it goes with what they already buy."
recommend() explicitly excludes anything the household already owns, so it
is structurally incapable of a repeat hit (verified: repeat_hits == 0.0 on
the validation harness) — its `novel_hits` were ~53x HybridRecommender's on
that same run. Ship this as a second recommendation surface
(`/complete-basket`-style endpoint), not blended into the hybrid's score.
"""
import pandas as pd

from fmcg_reco.features.basket_rules import mine_association_rules
from fmcg_reco.models.base import BaseRecommender


class BasketAffinityRecommender(BaseRecommender):
    def __init__(
        self,
        candidate_items: set,
        min_support: float = 0.002,
        min_lift: float = 1.2,
        min_confidence: float = 0.1,
        max_len: int = 3,
    ):
        self.candidate_items = candidate_items
        self.min_support = min_support
        self.min_lift = min_lift
        self.min_confidence = min_confidence
        self.max_len = max_len
        self.rules_: pd.DataFrame = pd.DataFrame()
        self.household_items_: dict = {}

    def fit(self, train_interactions: pd.DataFrame) -> "BasketAffinityRecommender":
        self.rules_ = mine_association_rules(
            train_interactions, self.candidate_items,
            min_support=self.min_support, min_lift=self.min_lift,
            min_confidence=self.min_confidence, max_len=self.max_len,
        )
        df = train_interactions[train_interactions["product_id"].isin(self.candidate_items)]
        self.household_items_ = df.groupby("household_key")["product_id"].apply(set).to_dict()
        return self

    def recommend(self, household_key: int, k: int) -> list:
        owned = self.household_items_.get(household_key, set())
        scores: dict = {}
        for antecedent, consequent, lift in zip(
            self.rules_["antecedents"], self.rules_["consequent_item"], self.rules_["lift"]
        ):
            if consequent in owned:
                continue
            if antecedent <= owned:
                scores[consequent] = max(scores.get(consequent, 0.0), lift)
        ranked = sorted(scores.items(), key=lambda pair: -pair[1])
        return [item for item, _ in ranked[:k]]

    def explain(self, household_key: int, item_id: int) -> pd.DataFrame:
        """Which rule(s) drove a recommendation - for a customer-facing
        "why am I seeing this" explanation, not just an opaque score.
        """
        owned = self.household_items_.get(household_key, set())
        matches = self.rules_[
            (self.rules_["consequent_item"] == item_id)
            & self.rules_["antecedents"].apply(lambda a: a <= owned)
        ]
        return matches[["antecedents", "confidence", "lift"]].sort_values("lift", ascending=False)
