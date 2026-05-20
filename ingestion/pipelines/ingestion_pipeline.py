import logging
import asyncio
from ingestion.loaders.pdf_loader import load_pdf
from ingestion.preprocess.cleaner import clean_pages
from rag.chunking.recursive_chunker import chunk_pages
from rag.embeddings.huggingface_embeddings import embed_chunks
from vectorstore.pgvector_client import insert_document, insert_chunks

logger = logging.getLogger(__name__)


def run_ingestion(file_path: str, document_id: str) -> list:
    """
    Full ingestion pipeline:
    PDF -> Extract -> Clean -> Chunk -> Embed -> Store
    """
    logger.info(f"Starting ingestion for {file_path}")

    # Step 1: Extract text from PDF
    pages = load_pdf(file_path)
    logger.info(f"Step 1 complete: extracted {len(pages)} pages")

    # Step 2: Clean the text
    pages = clean_pages(pages)
    logger.info(f"Step 2 complete: cleaned {len(pages)} pages")

    # Step 3: Chunk into smaller pieces
    chunks = chunk_pages(pages)
    logger.info(f"Step 3 complete: created {len(chunks)} chunks")

    # Step 4: Embed each chunk
    chunks = embed_chunks(chunks)
    logger.info(f"Step 4 complete: embedded {len(chunks)} chunks")

    # Step 5: Store in pgvector
    filename = file_path.split("/")[-1].split("\\")[-1]
    asyncio.run(_store(document_id, filename, file_path, chunks))
    logger.info(f"Step 5 complete: stored in database")

    return chunks


async def _store(document_id: str, filename: str, file_path: str, chunks: list):
    await insert_document(document_id, filename, file_path)
    await insert_chunks(chunks, document_id)