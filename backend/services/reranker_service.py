from typing import List
from rag.reranking.cross_encoder_reranker import rerank_chunks


def rerank(question: str, chunks: List[dict], top_k: int = 6) -> List[dict]:
    return rerank_chunks(question, chunks, top_k=top_k)
