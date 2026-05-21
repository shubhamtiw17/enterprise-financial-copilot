import os
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from backend.config import get_settings
from ingestion.pipelines.ingestion_pipeline import run_ingestion

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

ALLOWED_TYPES = {".pdf", ".csv", ".txt", ".md"}
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Track ingestion status in memory
ingestion_status = {}


def set_status(document_id: str, status: str):
    ingestion_status[document_id] = status


@router.get("/upload/status/{document_id}")
async def get_ingestion_status(document_id: str):
    status = ingestion_status.get(document_id, "processing")
    return {"document_id": document_id, "status": status}


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    filename = file.filename
    extension = os.path.splitext(filename)[-1].lower()

    if extension not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{extension}' not supported. Allowed: {ALLOWED_TYPES}"
        )

    document_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{document_id}{extension}")

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    logger.info(f"Saved file: {filename} -> {save_path}")

    background_tasks.add_task(run_ingestion_task, document_id, save_path)

    return {
        "document_id": document_id,
        "filename": filename,
        "status": "uploaded",
        "message": "File received. Ingestion started in background."
    }


async def run_ingestion_task(document_id: str, save_path: str):
    try:
        set_status(document_id, "processing")
        await run_ingestion(save_path, document_id)
        set_status(document_id, "ready")
    except Exception as e:
        set_status(document_id, f"error: {str(e)}")
        logger.error(f"Ingestion failed for {document_id}: {e}")