import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_build_prompt_contains_question(sample_chunks):
    from backend.services.citation_service import build_prompt
    messages = build_prompt("What was Apple revenue?", sample_chunks)
    assert "What was Apple revenue?" in messages[1]["content"]


def test_build_prompt_includes_all_sources(sample_chunks):
    from backend.services.citation_service import build_prompt
    messages = build_prompt("test", sample_chunks)
    user_msg = messages[1]["content"]
    assert "[Source 1:" in user_msg
    assert "[Source 2:" in user_msg
    assert "AAPL_10K_2023.pdf" in user_msg


def test_build_prompt_message_structure(sample_chunks):
    from backend.services.citation_service import build_prompt
    messages = build_prompt("test", sample_chunks)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_prompt_empty_chunks():
    from backend.services.citation_service import build_prompt
    messages = build_prompt("What was Apple revenue?", [])
    assert len(messages) == 2
    assert "Question:" in messages[1]["content"]


def test_build_citations_shape(sample_chunks):
    from backend.services.citation_service import build_citations
    citations = build_citations(sample_chunks)
    assert len(citations) == len(sample_chunks)
    for c in citations:
        assert "document_id" in c
        assert "filename" in c
        assert "score" in c


async def test_retrieve_returns_reranked_chunks(sample_chunks):
    with (
        patch("backend.services.rag_service.embed_query", return_value=[0.1] * 384),
        patch("backend.services.rag_service.retrieve_chunks", new=AsyncMock(return_value=sample_chunks)),
        patch("backend.services.rag_service.rerank", return_value=sample_chunks),
    ):
        from backend.services.rag_service import retrieve_relevant_chunks
        result = await retrieve_relevant_chunks("Apple revenue", top_k=2)
        assert len(result) == 2


async def test_retrieve_returns_empty_when_no_docs():
    with (
        patch("backend.services.rag_service.embed_query", return_value=[0.0] * 384),
        patch("backend.services.rag_service.retrieve_chunks", new=AsyncMock(return_value=[])),
        patch("backend.services.rag_service.rerank", return_value=[]),
    ):
        from backend.services.rag_service import retrieve_relevant_chunks
        result = await retrieve_relevant_chunks("irrelevant question")
        assert result == []


async def test_run_rag_query_returns_expected_shape(sample_chunks, mock_llm_response):
    with (
        patch("backend.services.rag_service.retrieve_relevant_chunks", new=AsyncMock(return_value=sample_chunks)),
        patch("langchain_groq.ChatGroq.ainvoke", new=AsyncMock(return_value=mock_llm_response)),
    ):
        from backend.services.rag_service import run_rag_query
        result = await run_rag_query("What was Apple revenue?")
        assert "answer" in result
        assert "citations" in result
        assert "model_used" in result


async def test_run_rag_query_no_chunks_returns_fallback():
    with patch("backend.services.rag_service.retrieve_relevant_chunks", new=AsyncMock(return_value=[])):
        from backend.services.rag_service import run_rag_query
        result = await run_rag_query("anything")
        assert "No relevant documents" in result["answer"]
        assert result["citations"] == []
