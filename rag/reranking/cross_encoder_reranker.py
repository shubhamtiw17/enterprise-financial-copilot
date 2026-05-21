import logging
from typing import List
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# This model runs locally — no API key needed
# cross-encoder/ms-marco-MiniLM-L-6-v2 is fast and accurate for passage ranking
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Load once at module level
model = CrossEncoder(MODEL_NAME)
logger.info(f"Loaded reranker model: {MODEL_NAME}")


def rerank_chunks(question: str, chunks: List[dict], top_k: int = 6) -> List[dict]:
    """
    Takes retrieved chunks and reranks them using a cross-encoder.
    Returns top_k chunks sorted by reranker score.
    """
    if not chunks:
        return chunks

    # Build pairs of [question, chunk_text] for the cross-encoder
    pairs = [[question, chunk["text"]] for chunk in chunks]

    # Score each pair — higher score = more relevant
    scores = model.predict(pairs)

    # Attach reranker score to each chunk
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    # Sort by reranker score descending
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    # Return only top_k
    result = reranked[:top_k]

    logger.info(f"Reranked {len(chunks)} chunks -> top {len(result)}")
    return result