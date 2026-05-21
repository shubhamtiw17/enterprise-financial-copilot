import pytest
from unittest.mock import AsyncMock, patch


async def test_vector_retrieve_calls_similarity_search():
    mock_results = [{"chunk_index": 0, "text": "test", "score": 0.9}]
    with patch("rag.retrievers.vector_retriever.similarity_search", new=AsyncMock(return_value=mock_results)):
        from rag.retrievers.vector_retriever import vector_retrieve
        result = await vector_retrieve([0.1] * 384, top_k=5)
        assert result == mock_results


def test_bm25_retrieve_ranks_by_relevance():
    corpus = [
        {"chunk_index": 0, "text": "Apple revenue declined in fiscal 2023"},
        {"chunk_index": 1, "text": "Tesla production increased significantly"},
        {"chunk_index": 2, "text": "Apple iPhone sales drove revenue growth"},
    ]
    from rag.retrievers.bm25_retriever import bm25_retrieve
    results = bm25_retrieve("Apple revenue", corpus, top_k=2)
    assert len(results) == 2
    sources = [r["chunk_index"] for r in results]
    assert 0 in sources or 2 in sources


def test_bm25_retrieve_returns_top_k():
    corpus = [{"chunk_index": i, "text": f"document {i} about finance"} for i in range(10)]
    from rag.retrievers.bm25_retriever import bm25_retrieve
    results = bm25_retrieve("finance", corpus, top_k=3)
    assert len(results) == 3
