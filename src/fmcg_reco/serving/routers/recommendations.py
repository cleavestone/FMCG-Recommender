import pandas as pd
from fastapi import APIRouter, Depends, Query

from fmcg_reco.models.affinity import BasketAffinityRecommender
from fmcg_reco.models.hybrid import HybridRecommender
from fmcg_reco.serving.auth.dependencies import get_current_user
from fmcg_reco.serving.deps import get_affinity_model, get_hybrid_model, get_product_catalog
from fmcg_reco.serving.presentation import describe_item
from fmcg_reco.serving.schemas import (
    BasketCompletionItem,
    BasketCompletionResponse,
    RecommendationResponse,
    RuleExplanation,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"], dependencies=[Depends(get_current_user)])


@router.get("/{household_id}", response_model=RecommendationResponse, summary="Reorder recommendations")
def recommend(
    household_id: int,
    k: int = Query(10, ge=1, le=50),
    model: HybridRecommender = Depends(get_hybrid_model),
    catalog: pd.DataFrame = Depends(get_product_catalog),
) -> RecommendationResponse:
    items = [describe_item(catalog, pid) for pid in model.recommend(household_id, k)]
    return RecommendationResponse(household_id=household_id, items=items)


@router.get(
    "/{household_id}/complete-basket",
    response_model=BasketCompletionResponse,
    summary="Basket-growth (cross-sell) recommendations",
)
def complete_basket(
    household_id: int,
    k: int = Query(10, ge=1, le=50),
    model: BasketAffinityRecommender = Depends(get_affinity_model),
    catalog: pd.DataFrame = Depends(get_product_catalog),
) -> BasketCompletionResponse:
    items = []
    for product_id in model.recommend(household_id, k):
        base = describe_item(catalog, product_id)
        explanation = [
            RuleExplanation(
                antecedent_product_ids=sorted(row.antecedents),
                confidence=row.confidence,
                lift=row.lift,
            )
            for row in model.explain(household_id, product_id).itertuples()
        ]
        items.append(BasketCompletionItem(**base.model_dump(), explanation=explanation))
    return BasketCompletionResponse(household_id=household_id, items=items)
