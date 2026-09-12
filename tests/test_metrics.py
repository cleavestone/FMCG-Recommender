"""Unit tests for evaluation.metrics — hand-computed examples, no data files needed."""
import math

from fmcg_reco.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    repeat_vs_novel_hits,
)

RECOMMENDED = ["A", "B", "C", "D", "E"]
RELEVANT = {"B", "D", "F"}


def test_precision_at_k():
    assert precision_at_k(RECOMMENDED, RELEVANT, 5) == 2 / 5


def test_recall_at_k():
    assert recall_at_k(RECOMMENDED, RELEVANT, 5) == 2 / 3


def test_recall_at_k_undefined_when_no_relevant_items():
    assert math.isnan(recall_at_k(RECOMMENDED, set(), 5))


def test_ndcg_at_k():
    assert abs(ndcg_at_k(RECOMMENDED, RELEVANT, 5) - 0.498) < 0.01


def test_average_precision_at_k():
    assert abs(average_precision_at_k(RECOMMENDED, RELEVANT, 5) - 0.333) < 0.01


def test_precision_at_k_perfect_ranking():
    recommended = ["B", "D", "F", "X", "Y"]
    assert precision_at_k(recommended, RELEVANT, 3) == 1.0
    assert recall_at_k(recommended, RELEVANT, 3) == 1.0
    assert ndcg_at_k(recommended, RELEVANT, 3) == 1.0


def test_precision_at_k_no_hits():
    recommended = ["X", "Y", "Z"]
    assert precision_at_k(recommended, RELEVANT, 3) == 0.0
    assert recall_at_k(recommended, RELEVANT, 3) == 0.0
    assert ndcg_at_k(recommended, RELEVANT, 3) == 0.0
    assert average_precision_at_k(recommended, RELEVANT, 3) == 0.0


def test_repeat_vs_novel_hits():
    previously_purchased = {"B"}
    repeat_hits, novel_hits = repeat_vs_novel_hits(RECOMMENDED, RELEVANT, previously_purchased, 5)
    assert repeat_hits == 1  # B: relevant hit, previously purchased
    assert novel_hits == 1  # D: relevant hit, not previously purchased
