from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness check")
def health() -> dict:
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness check")
def ready(request: Request) -> dict:
    is_ready = all(
        hasattr(request.app.state, attr)
        for attr in ("hybrid_model", "affinity_model", "replenishment_model")
    )
    return {"ready": is_ready}
