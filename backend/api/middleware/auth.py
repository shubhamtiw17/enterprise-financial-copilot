from fastapi import Request, HTTPException
from backend.config import get_settings

settings = get_settings()


async def verify_api_key(request: Request, call_next):
    # Auth is disabled in development; enable by setting API_KEY in .env
    api_key = getattr(settings, "api_key", None)
    if api_key:
        request_key = request.headers.get("X-API-Key")
        if request_key != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return await call_next(request)
