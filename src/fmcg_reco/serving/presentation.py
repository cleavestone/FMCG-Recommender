import pandas as pd
from fastapi import HTTPException, status

from fmcg_reco.serving.schemas import RecommendedItem


def describe_item(catalog: pd.DataFrame, product_id: int) -> RecommendedItem:
    try:
        row = catalog.loc[product_id]
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"product {product_id} not found") from exc
    return RecommendedItem(
        product_id=product_id,
        department=row["department"],
        commodity_desc=row["commodity_desc"],
        sub_commodity_desc=row["sub_commodity_desc"],
    )
