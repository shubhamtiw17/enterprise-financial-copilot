from typing import List


def evaluate_relevance(question: str, chunks: List[dict]) -> float:
    """
    Scores the average reranker score of retrieved chunks as a proxy for relevance.
    """
    if not chunks:
        return 0.0
    scores = [c.get("rerank_score", c.get("score", 0.0)) for c in chunks]
    return round(sum(scores) / len(scores), 4)
