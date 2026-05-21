import pytest
from unittest.mock import AsyncMock, patch


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_shape(client):
    data = client.get("/health").json()
    assert "status" in data
    assert "llm_provider" in data
    assert data["status"] == "ok"


# ── Query ─────────────────────────────────────────────────────────────────────

def test_query_returns_answer(client, sample_chunks):
    mock_result = {
        "answer": "Apple's revenue was $383.3 billion in fiscal 2023.",
        "citations": [
            {
                "document_id": "doc-001",
                "filename": "AAPL_10K_2023.pdf",
                "chunk_index": 7,
                "excerpt": "Apple's revenue for fiscal 2023 was $383.3 billion...",
                "score": 0.87,
            }
        ],
        "model_used": "groq",
    }

    with patch("backend.api.routes.query.run_rag_query", new=AsyncMock(return_value=mock_result)):
        response = client.post("/query", json={"question": "What was Apple revenue?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == mock_result["answer"]
    assert len(data["citations"]) == 1
    assert data["question"] == "What was Apple revenue?"


def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 400


def test_query_requires_question_field(client):
    response = client.post("/query", json={})
    assert response.status_code == 422


# ── Agent Query ───────────────────────────────────────────────────────────────

def test_agent_query_returns_full_response(client):
    mock_result = {
        "answer": "Apple faces currency, supply chain, and macro risks.",
        "citations": [
            {
                "document_id": "doc-001",
                "filename": "AAPL_10K_2023.pdf",
                "chunk_index": 3,
                "excerpt": "macroeconomic conditions, currency fluctuations...",
                "score": 0.82,
            }
        ],
        "model_used": "groq/llama-3.3-70b-versatile",
        "sub_questions": ["What are Apple's main revenue risks?"],
        "analysis": "**Risk Factors**\n- Currency fluctuations\n- Supply chain disruptions",
    }

    with patch("backend.api.routes.agent_query.run_agent_query", new=AsyncMock(return_value=mock_result)):
        response = client.post("/agent-query", json={"question": "What are Apple's revenue risks?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sub_questions" in data
    assert "analysis" in data
    assert isinstance(data["sub_questions"], list)


def test_agent_query_rejects_empty_question(client):
    response = client.post("/agent-query", json={"question": ""})
    assert response.status_code == 400
