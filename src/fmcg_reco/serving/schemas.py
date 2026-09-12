from pydantic import BaseModel


class RecommendedItem(BaseModel):
    product_id: int
    department: str
    commodity_desc: str
    sub_commodity_desc: str


class RecommendationResponse(BaseModel):
    household_id: int
    items: list[RecommendedItem]


class SimilarItemsResponse(BaseModel):
    product_id: int
    items: list[RecommendedItem]


class RuleExplanation(BaseModel):
    antecedent_product_ids: list[int]
    confidence: float
    lift: float


class BasketCompletionItem(RecommendedItem):
    explanation: list[RuleExplanation]


class BasketCompletionResponse(BaseModel):
    household_id: int
    items: list[BasketCompletionItem]
