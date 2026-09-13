from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from fmcg_reco.artifacts import (
    load_affinity_artifacts,
    load_artifacts,
    load_replenishment_artifacts,
)
from fmcg_reco.config import PROCESSED_DIR, RAW_DIR, settings
from fmcg_reco.serving.auth.router import router as auth_router
from fmcg_reco.serving.auth.store import init_db
from fmcg_reco.serving.routers.health import router as health_router
from fmcg_reco.serving.routers.items import router as items_router
from fmcg_reco.serving.routers.recommendations import router as recommendations_router
from fmcg_reco.serving.routers.replenishment import router as replenishment_router

DESCRIPTION = """
Hybrid FMCG recommender trained on the Dunnhumby *Complete Journey* dataset,
exposing three independent recommendation surfaces:

- **Reorder** (`/recommendations/{household_id}`) — rank-fusion hybrid of
  content, collaborative, and repeat-purchase signals.
- **Basket growth** (`/recommendations/{household_id}/complete-basket`) —
  association-rule cross-sell, with the triggering rule returned alongside
  each suggestion.
- **Replenishment** (`/households/{household_id}/replenishment`) — what's
  probably due soon, with an honest uncertainty window rather than a
  precise date.

Authenticate via `/auth/token` (OAuth2 password grant) before calling any
`recommendations`, `households`, or `items` endpoint.
"""

TAGS_METADATA = [
    {"name": "auth", "description": "OAuth2 password grant, JWT bearer tokens."},
    {"name": "health", "description": "Liveness and readiness checks."},
    {"name": "recommendations", "description": "Reorder and basket-growth recommendations, per household."},
    {"name": "replenishment", "description": "Forecasted timing and quantity of a household's next purchases."},
    {"name": "items", "description": "Item-to-item content similarity."},
]

SCALAR_HTML = """<!doctype html>
<html>
<head>
  <title>FMCG Recommender API</title>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
  <script id="api-reference" data-url="/openapi.json"></script>
  <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    product = pd.read_parquet(PROCESSED_DIR / "product_dim.parquet")
    demographic = pd.read_csv(RAW_DIR / "hh_demographic.csv")
    demographic.columns = [c.strip().lower() for c in demographic.columns]

    app.state.product_catalog = product.set_index("product_id")
    app.state.hybrid_model = load_artifacts(settings.hybrid_artifact_dir, product, demographic)
    app.state.affinity_model = load_affinity_artifacts(settings.affinity_artifact_dir)
    app.state.replenishment_model = load_replenishment_artifacts(settings.replenishment_artifact_dir, product)

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="FMCG Recommender API",
        description=DESCRIPTION,
        version="1.0.0",
        openapi_tags=TAGS_METADATA,
        docs_url=None,
        lifespan=lifespan,
    )
    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(recommendations_router)
    app.include_router(replenishment_router)
    app.include_router(items_router)

    @app.get("/docs", include_in_schema=False)
    def docs() -> HTMLResponse:
        return HTMLResponse(SCALAR_HTML)

    return app
