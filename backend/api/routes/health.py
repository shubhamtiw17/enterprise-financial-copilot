from fastapi import APIRouter
from backend.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.app_env,
        "llm_provider": settings.llm_provider,
    }