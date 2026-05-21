import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


def load_markdown(file_path: str) -> List[Dict]:
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    sections = content.split("\n## ")

    pages = []
    for i, section in enumerate(sections):
        text = section.strip()
        if text:
            pages.append({
                "page_number": i + 1,
                "text": text,
                "source": path.name,
                "file_path": str(path),
                "total_pages": len(sections),
            })

    logger.info(f"Loaded {len(pages)} sections from Markdown: {path.name}")
    return pages
