import os
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from backend.config import get_settings

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

ALLOWED_TYPES = {".pdf", ".csv", ".txt", ".md"}
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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

    background_tasks.add_task(run_ingestion_placeholder, document_id, filename, save_path)

    return {
        "document_id": document_id,
        "filename": filename,
        "status": "uploaded",
        "message": "File received. Ingestion started in background."
    }


async def run_ingestion_placeholder(document_id: str, filename: str, path: str):
    logger.info(f"[Ingestion queued] {filename} ({document_id}) at {path}")