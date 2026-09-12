"""Ranking metrics for implicit-feedback recommendation evaluation.

Relevance is binary (purchased vs. not) — there are no ratings in this
dataset. Extracted from notebooks/02_eval_harness_and_baselines.ipynb
(sections 4-5), where each function was hand-verified against a worked
example before being trusted.
"""
import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Of the top-k recommendations, what fraction were actually relevant?"""
    if k <= 0:
        return np.nan
    topk = recommended[:k]
    hits = sum(1 for item in topk if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Of everything relevant, what fraction did the top-k surface?

    Undefined (NaN) when `relevant` is empty — a household with no purchases
    in the eval window contributes nothing to this metric, not a zero.
    """
    if not relevant:
        return np.nan
    topk = recommended[:k]
    hits = sum(1 for item in topk if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Recall-like, but a hit ranked higher counts more (1/log2(rank+1) discount)."""
    if not relevant:
        return np.nan
    topk = recommended[:k]
    dcg = sum((1.0 if item in relevant else 0.0) / np.log2(i + 2) for i, item in enumerate(topk))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else np.nan


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Mean average precision contribution for one household: rewards hits
    that come both early and often, normalised by min(|relevant|, k).
    """
    if not relevant:
        return np.nan
    topk = recommended[:k]
    hits, sum_precision = 0, 0.0
    for i, item in enumerate(topk, start=1):
        if item in relevant:
            hits += 1
            sum_precision += hits / i
    denom = min(len(relevant), k)
    return sum_precision / denom if denom > 0 else np.nan


def repeat_vs_novel_hits(recommended: list, relevant: set, previously_purchased: set, k: int) -> tuple[int, int]:
    """Split top-k hits into ones the household already bought before
    (`repeat`) vs. ones genuinely new to it (`novel`). Grocery recommendation
    is repeat-purchase dominated, so recall alone can be won by replaying
    history — this keeps that distinction visible.
    """
    hits = [item for item in recommended[:k] if item in relevant]
    repeat_hits = sum(1 for item in hits if item in previously_purchased)
    novel_hits = len(hits) - repeat_hits
    return repeat_hits, novel_hits
