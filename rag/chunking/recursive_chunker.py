import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_index = 0

    for page in pages:
        for text in splitter.split_text(page["text"]):
            chunks.append({
                "chunk_index": chunk_index,
                "text": text,
                "page_number": page["page_number"],
                "source": page["source"],
                "file_path": page["file_path"],
            })
            chunk_index += 1

    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks
