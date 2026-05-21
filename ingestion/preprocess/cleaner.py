import re
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # Remove standalone page numbers (e.g. "- 42 -" or "42")
    text = re.sub(r'^\s*-?\s*\d+\s*-?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84]', '', text)
    return text.strip()


def clean_pages(pages: list) -> list:
    cleaned = []
    for page in pages:
        cleaned_text = clean_text(page["text"])
        if cleaned_text:
            page["text"] = cleaned_text
            cleaned.append(page)
    logger.info(f"Cleaned {len(cleaned)} pages")
    return cleaned
