import logging
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> List[Dict]:
    """
    Extracts text from a PDF file page by page.
    Returns a list of dicts, one per page, with text and metadata.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(str(path))
    pages = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        # Skip empty pages
        if not text or not text.strip():
            logger.warning(f"Empty page {page_num} in {path.name}, skipping")
            continue

        pages.append({
            "page_number": page_num + 1,
            "text": text,
            "source": path.name,
            "file_path": str(path),
            "total_pages": len(reader.pages)
        })

    logger.info(f"Extracted {len(pages)} pages from {path.name}")
    return pages
