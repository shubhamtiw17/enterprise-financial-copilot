import asyncio
from typing import List, Optional
from langchain_core.tools import tool
from backend.services.rag_service import retrieve_relevant_chunks


@tool
async def retrieve_financial_context(
    question: str,
    document_ids: Optional[List[str]] = None,
    top_k: int = 6,
) -> List[dict]:
    """
    Retrieves relevant chunks from ingested financial documents for a given question.
    Uses pgvector similarity search followed by cross-encoder reranking.
    Returns a list of chunk dicts with keys: text, source, page_number, score.
    """
    chunks = await retrieve_relevant_chunks(
        question=question,
        top_k=top_k,
        document_ids=document_ids,
    )
    return chunks
