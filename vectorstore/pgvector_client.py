import logging
from typing import List, Optional
import asyncpg
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def get_connection():
    """Creates a single database connection."""
    return await asyncpg.connect(settings.database_url)


async def setup_database():
    """Runs the schema SQL to create tables if they don't exist."""
    conn = await get_connection()
    try:
        with open("vectorstore/schema.sql", "r") as f:
            schema = f.read()
        await conn.execute(schema)
        logger.info("Database schema applied successfully")
    finally:
        await conn.close()


async def insert_document(document_id: str, filename: str, file_path: str) -> None:
    """Registers a document in the documents table."""
    conn = await get_connection()
    try:
        await conn.execute(
            """
            INSERT INTO documents (id, filename, file_path)
            VALUES ($1, $2, $3)
            ON CONFLICT (id) DO NOTHING
            """,
            document_id, filename, file_path
        )
        logger.info(f"Inserted document: {filename}")
    finally:
        await conn.close()


async def insert_chunks(chunks: List[dict], document_id: str) -> None:
    """Stores all chunks with their embeddings into the chunks table."""
    conn = await get_connection()
    try:
        for chunk in chunks:
            await conn.execute(
                """
                INSERT INTO chunks 
                    (document_id, chunk_index, text, page_number, source, embedding)
                VALUES 
                    ($1, $2, $3, $4, $5, $6)
                """,
                document_id,
                chunk["chunk_index"],
                chunk["text"],
                chunk["page_number"],
                chunk["source"],
                str(chunk["embedding"])
            )

        await conn.execute(
            "UPDATE documents SET total_chunks = $1 WHERE id = $2",
            len(chunks), document_id
        )
        logger.info(f"Inserted {len(chunks)} chunks for document {document_id}")
    finally:
        await conn.close()


async def similarity_search(
    query_embedding: List[float],
    top_k: int = 6,
    document_ids: Optional[List[str]] = None
) -> List[dict]:
    """
    Finds the top_k chunks most similar to the query embedding.
    Optionally scoped to specific document IDs.
    """
    conn = await get_connection()
    try:
        if document_ids:
            rows = await conn.fetch(
                """
                SELECT 
                    c.id,
                    c.document_id,
                    c.chunk_index,
                    c.text,
                    c.page_number,
                    c.source,
                    1 - (c.embedding <=> $1::vector) AS score
                FROM chunks c
                WHERE c.document_id = ANY($2)
                ORDER BY c.embedding <=> $1::vector
                LIMIT $3
                """,
                str(query_embedding), document_ids, top_k
            )
        else:
            rows = await conn.fetch(
                """
                SELECT 
                    c.id,
                    c.document_id,
                    c.chunk_index,
                    c.text,
                    c.page_number,
                    c.source,
                    1 - (c.embedding <=> $1::vector) AS score
                FROM chunks c
                ORDER BY c.embedding <=> $1::vector
                LIMIT $2
                """,
                str(query_embedding), top_k
            )

        return [dict(row) for row in rows]
    finally:
        await conn.close()