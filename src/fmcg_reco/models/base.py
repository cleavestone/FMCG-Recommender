"""Common interface every recommender in this project implements.

Fixing this shape now (fit/recommend/similar_items) means the evaluation
harness — and later, the serving API — never has to change as models get
more sophisticated (popularity -> content -> collaborative -> hybrid ->
learning-to-rank).
"""
from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, train_interactions: pd.DataFrame) -> "BaseRecommender":
        ...

    @abstractmethod
    def recommend(self, household_key: int, k: int) -> list:
        ...

    def similar_items(self, product_id: int, k: int) -> list:
        raise NotImplementedError(f"{type(self).__name__} does not support item-item similarity")
