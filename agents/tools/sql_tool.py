import logging
from langchain_core.tools import tool
from database.connection import get_pool

logger = logging.getLogger(__name__)


@tool
async def query_documents_table(sql: str) -> list:
    """
    Executes a read-only SQL query against the documents table.
    Use to look up ingested document metadata (filenames, chunk counts).
    Only SELECT statements are permitted.
    """
    if not sql.strip().upper().startswith("SELECT"):
        return [{"error": "Only SELECT queries are permitted"}]

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql)
        return [dict(row) for row in rows]
