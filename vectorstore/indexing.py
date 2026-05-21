import logging
from vectorstore.pgvector_client import get_connection

logger = logging.getLogger(__name__)


async def create_ivfflat_index(lists: int = 100) -> None:
    conn = await get_connection()
    try:
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
            f"ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
        )
        logger.info(f"IVFFlat index created with lists={lists}")
    finally:
        await conn.close()


async def drop_index() -> None:
    conn = await get_connection()
    try:
        await conn.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
        logger.info("Index dropped")
    finally:
        await conn.close()
