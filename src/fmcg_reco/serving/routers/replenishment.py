import pandas as pd
from fastapi import APIRouter, Depends, Query

from fmcg_reco.models.replenishment import ReplenishmentForecaster
from fmcg_reco.serving.auth.dependencies import get_current_user
from fmcg_reco.serving.deps import get_product_catalog, get_replenishment_model
from fmcg_reco.serving.presentation import describe_item
from fmcg_reco.serving.schemas import ReplenishmentItem, ReplenishmentResponse

router = APIRouter(prefix="/households", tags=["replenishment"], dependencies=[Depends(get_current_user)])


@router.get(
    "/{household_id}/replenishment", response_model=ReplenishmentResponse, summary="What's due soon"
)
def replenishment(
    household_id: int,
    k: int = Query(10, ge=1, le=50),
    model: ReplenishmentForecaster = Depends(get_replenishment_model),
    catalog: pd.DataFrame = Depends(get_product_catalog),
) -> ReplenishmentResponse:
    forecast = model.forecast(household_id).head(k)
    items = []
    for row in forecast.itertuples():
        base = describe_item(catalog, row.product_id)
        items.append(ReplenishmentItem(
            **base.model_dump(),
            days_until_due=row.days_until_due,
            window_days=row.window_days,
            expected_quantity=row.expected_quantity,
        ))
    return ReplenishmentResponse(household_id=household_id, as_of_day=model.reference_day_, items=items)
