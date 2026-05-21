import csv
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def load_csv(file_path: str) -> List[Dict]:
    rows = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = " | ".join(f"{k}: {v}" for k, v in row.items())
            rows.append({
                "page_number": i + 1,
                "text": text,
                "source": file_path.split("\\")[-1].split("/")[-1],
                "file_path": file_path,
                "total_pages": None,
            })
    logger.info(f"Loaded {len(rows)} rows from CSV: {file_path}")
    return rows
