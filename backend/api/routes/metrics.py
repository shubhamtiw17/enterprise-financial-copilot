import time
from fastapi import APIRouter

router = APIRouter()

_start_time = time.time()


@router.get("/metrics")
def get_metrics():
    return {
        "uptime_seconds": round(time.time() - _start_time, 2),
        "status": "ok",
    }
