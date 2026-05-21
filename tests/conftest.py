import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    """FastAPI test client, shared across the whole test session."""
    # Patch LangSmith setup and DB pool so the app boots without live services
    with patch("llmops.monitoring.langsmith_tracing.setup_langsmith", return_value=False):
        from backend.main import app
        return TestClient(app)


@pytest.fixture
def sample_chunks():
    """Realistic chunk dicts returned by similarity_search / rerank."""
    return [
        {
            "document_id": "doc-001",
            "source": "AAPL_10K_2023.pdf",
            "page_number": 42,
            "chunk_index": 7,
            "text": (
                "Apple's revenue for fiscal 2023 was $383.3 billion, "
                "a decrease of 3% compared to fiscal 2022."
            ),
            "score": 0.91,
            "rerank_score": 0.87,
        },
        {
            "document_id": "doc-001",
            "source": "AAPL_10K_2023.pdf",
            "page_number": 15,
            "chunk_index": 3,
            "text": (
                "The Company's products and services may be affected by "
                "macroeconomic conditions, currency fluctuations, and supply chain disruptions."
            ),
            "score": 0.85,
            "rerank_score": 0.82,
        },
    ]


@pytest.fixture
def mock_llm_response():
    """Fake AIMessage returned by ChatGroq.ainvoke()."""
    msg = MagicMock()
    msg.content = "Apple's revenue declined 3% in fiscal 2023 to $383.3 billion [Source 1]."
    return msg
