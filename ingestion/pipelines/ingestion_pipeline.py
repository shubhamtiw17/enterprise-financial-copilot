import logging
from ingestion.loaders.pdf_loader import load_pdf
from ingestion.preprocess.cleaner import clean_pages
from rag.chunking.recursive_chunker import chunk_pages
from rag.embeddings.huggingface_embeddings import embed_chunks
from vectorstore.pgvector_client import insert_document, insert_chunks

logger = logging.getLogger(__name__)


async def run_ingestion(file_path: str, document_id: str) -> list:
    """
    Full ingestion pipeline:
    PDF -> Extract -> Clean -> Chunk -> Embed -> Store
    """
    logger.info(f"Starting ingestion for {file_path}")

    filename = file_path.split("/")[-1].split("\\")[-1]

    # Step 1: Extract
    pages = load_pdf(file_path)
    logger.info(f"Step 1 complete: extracted {len(pages)} pages")

    # Step 2: Clean
    pages = clean_pages(pages)
    logger.info(f"Step 2 complete: cleaned {len(pages)} pages")

    # Step 3: Chunk
    chunks = chunk_pages(pages)
    logger.info(f"Step 3 complete: created {len(chunks)} chunks")

    # Step 4: Embed
    chunks = embed_chunks(chunks)
    logger.info(f"Step 4 complete: embedded {len(chunks)} chunks")

    # Step 5: Store
    await insert_document(document_id, filename, file_path)
    await insert_chunks(chunks, document_id)
    logger.info(f"Step 5 complete: stored {len(chunks)} chunks in database")

    return chunks
