"""Evaluation harness: score any BaseRecommender against held-out interactions.

Extracted from notebooks/02_eval_harness_and_baselines.ipynb (section 9).
"""
import pandas as pd

from fmcg_reco.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    repeat_vs_novel_hits,
)
from fmcg_reco.models.base import BaseRecommender


def evaluate_recommender(
    model: BaseRecommender,
    relevant_map: dict[int, set],
    k_values: list[int],
    previously_purchased_map: dict[int, set] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run `model` against every household in `relevant_map` and score it at
    each k in `k_values`. Households with no relevant items are skipped —
    recall/precision are undefined for them, not zero, so including them
    would silently bias every metric.

    Returns (per-household results, mean summary indexed by k).
    """
    rows = []
    max_k = max(k_values)
    for household_key, relevant in relevant_map.items():
        if not relevant:
            continue
        recs = model.recommend(household_key, max_k)
        prev = previously_purchased_map.get(household_key, set()) if previously_purchased_map else set()
        for k in k_values:
            row = {
                "household_key": household_key,
                "k": k,
                "precision": precision_at_k(recs, relevant, k),
                "recall": recall_at_k(recs, relevant, k),
                "ndcg": ndcg_at_k(recs, relevant, k),
                "map": average_precision_at_k(recs, relevant, k),
            }
            if previously_purchased_map is not None:
                repeat_hits, novel_hits = repeat_vs_novel_hits(recs, relevant, prev, k)
                row["repeat_hits"], row["novel_hits"] = repeat_hits, novel_hits
            rows.append(row)

    results = pd.DataFrame(rows)
    metric_cols = [col for col in ["precision", "recall", "ndcg", "map", "repeat_hits", "novel_hits"] if col in results]
    summary = results.groupby("k")[metric_cols].mean().round(4)
    return results, summary
