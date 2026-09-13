import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from fmcg_reco.serving.auth.models import User
from fmcg_reco.serving.auth.router import router as auth_router
from fmcg_reco.serving.auth.security import hash_password
from fmcg_reco.serving.auth.store import get_session
from fmcg_reco.serving.deps import (
    get_affinity_model,
    get_hybrid_model,
    get_product_catalog,
    get_replenishment_model,
)
from fmcg_reco.serving.routers.health import router as health_router
from fmcg_reco.serving.routers.items import router as items_router
from fmcg_reco.serving.routers.recommendations import router as recommendations_router
from fmcg_reco.serving.routers.replenishment import router as replenishment_router


class FakeHybridModel:
    def recommend(self, household_id: int, k: int) -> list:
        return [10, 20, 30][:k]

    def similar_items(self, product_id: int, k: int) -> list:
        if product_id not in {10, 20, 30}:
            raise KeyError(product_id)
        return [20, 30][:k]


class FakeAffinityModel:
    def recommend(self, household_id: int, k: int) -> list:
        return [40][:k]

    def explain(self, household_id: int, item_id: int) -> pd.DataFrame:
        return pd.DataFrame([{"antecedents": frozenset({10}), "confidence": 0.5, "lift": 2.0}])


class FakeReplenishmentModel:
    reference_day_ = 100

    def forecast(self, household_id: int, as_of_day: int | None = None) -> pd.DataFrame:
        return pd.DataFrame([
            {"product_id": 10, "days_until_due": 3.0, "window_days": 1.5, "expected_quantity": 2.0},
            {"product_id": 20, "days_until_due": 7.0, "window_days": 2.0, "expected_quantity": 1.0},
        ])


def _fake_catalog() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": [10, 20, 30, 40],
        "department": ["GROCERY"] * 4,
        "commodity_desc": ["A", "B", "C", "D"],
        "sub_commodity_desc": ["A1", "B1", "C1", "D1"],
    }).set_index("product_id")


@pytest.fixture
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(User(username="demo", hashed_password=hash_password("demo-password")))
        session.commit()

    def override_get_session():
        with Session(engine) as session:
            yield session

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(recommendations_router)
    app.include_router(replenishment_router)
    app.include_router(items_router)
    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_hybrid_model] = lambda: FakeHybridModel()
    app.dependency_overrides[get_affinity_model] = lambda: FakeAffinityModel()
    app.dependency_overrides[get_replenishment_model] = lambda: FakeReplenishmentModel()
    app.dependency_overrides[get_product_catalog] = lambda: _fake_catalog()

    app.state.hybrid_model = FakeHybridModel()
    app.state.affinity_model = FakeAffinityModel()
    app.state.replenishment_model = FakeReplenishmentModel()

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client):
    response = client.post("/auth/token", data={"username": "demo", "password": "demo-password"})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
