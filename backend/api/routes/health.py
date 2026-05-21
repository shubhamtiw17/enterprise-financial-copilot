from fastapi import APIRouter
from backend.models.response_models import HealthResponse
from backend.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.app_env,
        llm_provider=settings.llm_provider,
    )
