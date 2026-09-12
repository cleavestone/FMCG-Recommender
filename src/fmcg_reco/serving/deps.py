import pandas as pd
from fastapi import Request

from fmcg_reco.models.affinity import BasketAffinityRecommender
from fmcg_reco.models.hybrid import HybridRecommender


def get_hybrid_model(request: Request) -> HybridRecommender:
    return request.app.state.hybrid_model


def get_affinity_model(request: Request) -> BasketAffinityRecommender:
    return request.app.state.affinity_model


def get_product_catalog(request: Request) -> pd.DataFrame:
    return request.app.state.product_catalog
