import logging
from typing import List, Optional
from backend.services.embedding_service import embed_query
from backend.services.retrieval_service import retrieve_chunks
from backend.services.reranker_service import rerank
from backend.services.citation_service import build_prompt, build_citations
from backend.services.llm_service import call_llm
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def retrieve_relevant_chunks(
    question: str,
    top_k: int = 6,
    document_ids: Optional[List[str]] = None,
) -> List[dict]:
    embedding = embed_query(question)
    candidates = await retrieve_chunks(
        query_embedding=embedding,
        top_k=top_k * 2,
        document_ids=document_ids,
    )
    reranked = rerank(question, candidates, top_k=top_k)
    logger.info(f"Retrieved {len(candidates)} candidates, reranked to {len(reranked)}")
    return reranked


async def run_rag_query(
    question: str,
    top_k: int = 6,
    document_ids: Optional[List[str]] = None,
) -> dict:
    from langsmith import traceable

    @traceable(name="rag_pipeline", run_type="chain")
    async def _pipeline(q: str):
        chunks = await retrieve_relevant_chunks(q, top_k, document_ids)

        if not chunks:
            return {
                "answer": "No relevant documents found. Please upload financial documents first.",
                "citations": [],
                "model_used": settings.llm_provider,
            }

        messages = build_prompt(q, chunks)
        answer = await call_llm(messages)

        return {
            "answer": answer,
            "citations": build_citations(chunks),
            "model_used": settings.llm_provider,
        }

    return await _pipeline(question)
