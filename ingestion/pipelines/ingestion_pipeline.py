import logging
from ingestion.loaders.pdf_loader import load_pdf
from ingestion.preprocess.cleaner import clean_pages
from rag.chunking.recursive_chunker import chunk_pages
from rag.embeddings.huggingface_embeddings import embed_chunks
from vectorstore.pgvector_client import insert_document, insert_chunks

logger = logging.getLogger(__name__)


async def run_ingestion(file_path: str, document_id: str) -> list:
    logger.info(f"Starting ingestion for {file_path}")

    filename = file_path.split("/")[-1].split("\\")[-1]

    pages = load_pdf(file_path)
    logger.info(f"Extracted {len(pages)} pages")

    pages = clean_pages(pages)
    logger.info(f"Cleaned {len(pages)} pages")

    chunks = chunk_pages(pages)
    logger.info(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    logger.info(f"Embedded {len(chunks)} chunks")

    await insert_document(document_id, filename, file_path)
    await insert_chunks(chunks, document_id)
    logger.info(f"Stored {len(chunks)} chunks in database")

    return chunks
