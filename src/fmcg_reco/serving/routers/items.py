import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from fmcg_reco.models.hybrid import HybridRecommender
from fmcg_reco.serving.auth.dependencies import get_current_user
from fmcg_reco.serving.deps import get_hybrid_model, get_product_catalog
from fmcg_reco.serving.presentation import describe_item
from fmcg_reco.serving.schemas import SimilarItemsResponse

router = APIRouter(prefix="/items", tags=["items"], dependencies=[Depends(get_current_user)])


@router.get("/{product_id}/similar", response_model=SimilarItemsResponse, summary="Content-based similar items")
def similar(
    product_id: int,
    k: int = Query(10, ge=1, le=50),
    model: HybridRecommender = Depends(get_hybrid_model),
    catalog: pd.DataFrame = Depends(get_product_catalog),
) -> SimilarItemsResponse:
    try:
        similar_ids = model.similar_items(product_id, k)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"product {product_id} not found") from exc
    items = [describe_item(catalog, pid) for pid in similar_ids]
    return SimilarItemsResponse(product_id=product_id, items=items)
