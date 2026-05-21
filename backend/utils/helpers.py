import os
from backend.utils.constants import UPLOAD_DIR


def ensure_upload_dir() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_filename(file_path: str) -> str:
    return file_path.split("/")[-1].split("\\")[-1]


def truncate(text: str, max_chars: int = 200) -> str:
    return text[:max_chars]
