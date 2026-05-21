from typing import List
from rag.embeddings.huggingface_embeddings import embed_query as _embed_query
from rag.embeddings.huggingface_embeddings import embed_chunks as _embed_chunks


def embed_query(question: str) -> List[float]:
    return _embed_query(question)


def embed_chunks(chunks: List[dict]) -> List[dict]:
    return _embed_chunks(chunks)
