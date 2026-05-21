import os
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from backend.config import get_settings
from backend.models.response_models import UploadResponse
from backend.utils.constants import ALLOWED_UPLOAD_TYPES, UPLOAD_DIR
from backend.utils.helpers import ensure_upload_dir
from ingestion.pipelines.ingestion_pipeline import run_ingestion

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

ensure_upload_dir()

ingestion_status: dict = {}


@router.get("/upload/status/{document_id}")
async def get_ingestion_status(document_id: str):
    return {"document_id": document_id, "status": ingestion_status.get(document_id, "processing")}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename)[-1].lower()
    if extension not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{extension}' not supported. Allowed: {ALLOWED_UPLOAD_TYPES}",
        )

    document_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{document_id}{extension}")

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    logger.info(f"Saved: {file.filename} -> {save_path}")
    background_tasks.add_task(_ingest, document_id, save_path)

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        status="uploaded",
        message="File received. Ingestion started in background.",
    )


async def _ingest(document_id: str, save_path: str) -> None:
    try:
        ingestion_status[document_id] = "processing"
        await run_ingestion(save_path, document_id)
        ingestion_status[document_id] = "ready"
        if os.path.exists(save_path):
            os.remove(save_path)
    except Exception as e:
        ingestion_status[document_id] = f"error: {e}"
        logger.error(f"Ingestion failed for {document_id}: {e}")
