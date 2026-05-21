import logging
from typing import List
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

model = CrossEncoder(MODEL_NAME)
logger.info(f"Loaded reranker model: {MODEL_NAME}")


def rerank_chunks(question: str, chunks: List[dict], top_k: int = 6) -> List[dict]:
    if not chunks:
        return chunks

    pairs = [[question, chunk["text"]] for chunk in chunks]
    scores = model.predict(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    result = reranked[:top_k]

    logger.info(f"Reranked {len(chunks)} chunks -> top {len(result)}")
    return result
