from typing import List
from rank_bm25 import BM25Okapi


def bm25_retrieve(query: str, corpus: List[dict], top_k: int = 10) -> List[dict]:
    tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())

    for i, doc in enumerate(corpus):
        doc["bm25_score"] = float(scores[i])

    return sorted(corpus, key=lambda x: x["bm25_score"], reverse=True)[:top_k]
