from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness check")
def health() -> dict:
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness check")
def ready(request: Request) -> dict:
    is_ready = hasattr(request.app.state, "hybrid_model") and hasattr(request.app.state, "affinity_model")
    return {"ready": is_ready}
