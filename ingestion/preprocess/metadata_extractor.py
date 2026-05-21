import re
from typing import Dict


def extract_metadata(text: str, source: str) -> Dict:
    year = None
    match = re.search(r'\b(20\d{2})\b', text[:500])
    if match:
        year = int(match.group(1))

    ticker = None
    match = re.search(r'\b([A-Z]{1,5})\b', source)
    if match:
        ticker = match.group(1)

    return {
        "fiscal_year": year,
        "ticker": ticker,
        "source": source,
        "word_count": len(text.split()),
    }
