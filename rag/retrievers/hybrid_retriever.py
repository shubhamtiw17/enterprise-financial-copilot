from typing import List, Optional
from rag.retrievers.vector_retriever import vector_retrieve
from rag.retrievers.bm25_retriever import bm25_retrieve


async def hybrid_retrieve(
    query: str,
    query_embedding: List[float],
    corpus: List[dict],
    top_k: int = 10,
    document_ids: Optional[List[str]] = None,
    vector_weight: float = 0.6,
) -> List[dict]:
    vector_results = await vector_retrieve(
        query_embedding, top_k=top_k * 2, document_ids=document_ids
    )
    bm25_results = bm25_retrieve(query, corpus, top_k=top_k * 2)

    scores: dict = {}
    for doc in vector_results:
        idx = doc["chunk_index"]
        scores[idx] = scores.get(idx, 0) + vector_weight * doc.get("score", 0)
    for doc in bm25_results:
        idx = doc["chunk_index"]
        scores[idx] = scores.get(idx, 0) + (1 - vector_weight) * doc.get("bm25_score", 0)

    all_docs = {d["chunk_index"]: d for d in vector_results + bm25_results}
    ranked = sorted(all_docs.values(), key=lambda d: scores.get(d["chunk_index"], 0), reverse=True)
    return ranked[:top_k]
