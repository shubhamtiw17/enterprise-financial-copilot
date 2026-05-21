import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_state(query="What was Apple revenue?", **kwargs):
    """Helper: build a minimal AgentState dict for testing."""
    return {
        "original_query": query,
        "document_ids": None,
        "sub_questions": [],
        "research_results": [],
        "analysis": "",
        "final_answer": "",
        "citations": [],
        "model_used": "",
        **kwargs,
    }


# ── PlannerAgent ──────────────────────────────────────────────────────────────

async def test_planner_simple_query_returns_one_sub_question():
    mock_msg = MagicMock()
    mock_msg.content = '["What was Apple revenue?"]'
    with patch("langchain_groq.ChatGroq.ainvoke", new=AsyncMock(return_value=mock_msg)):
        from agents.planner_agent import planner_node
        result = await planner_node(_make_state("What was Apple revenue?"))
        assert len(result["sub_questions"]) == 1
        assert result["sub_questions"][0] == "What was Apple revenue?"


async def test_planner_multi_company_query_returns_two_sub_questions():
    mock_msg = MagicMock()
    mock_msg.content = '["What are Apple revenue risks?", "What are Microsoft revenue risks?"]'
    with patch("langchain_groq.ChatGroq.ainvoke", new=AsyncMock(return_value=mock_msg)):
        from agents.planner_agent import planner_node
        result = await planner_node(_make_state("Compare Apple and Microsoft revenue risks"))
        assert len(result["sub_questions"]) == 2


async def test_planner_falls_back_on_bad_json():
    """If the LLM returns non-JSON, planner uses the original query as-is."""
    mock_msg = MagicMock()
    mock_msg.content = "Sorry, I cannot decompose this."
    with patch("langchain_groq.ChatGroq.ainvoke", new=AsyncMock(return_value=mock_msg)):
        from agents.planner_agent import planner_node
        result = await planner_node(_make_state("What was Apple revenue?"))
        assert result["sub_questions"] == ["What was Apple revenue?"]


async def test_planner_strips_markdown_fences():
    """Planner handles LLM wrapping JSON in ```json ... ``` fences."""
    mock_msg = MagicMock()
    mock_msg.content = '```json\n["What are Apple risks?"]\n```'
    with patch("langchain_groq.ChatGroq.ainvoke", new=AsyncMock(return_value=mock_msg)):
        from agents.planner_agent import planner_node
        result = await planner_node(_make_state("What are Apple risks?"))
        assert result["sub_questions"] == ["What are Apple risks?"]


# ── ResearchAgent ─────────────────────────────────────────────────────────────

async def test_research_returns_one_result_per_sub_question(sample_chunks):
    mock_msg = MagicMock()
    mock_msg.content = "Apple revenue declined 3% to $383.3B."
    state = _make_state(sub_questions=["What was Apple revenue?", "What are Apple risks?"])
    with (
        patch("agents.research_agent.retrieve_relevant_chunks", new=AsyncMock(return_value=sample_chunks)),
        patch("langchain_groq.ChatGroq.ainvoke", new=AsyncMock(return_value=mock_msg)),
    ):
        from agents.research_agent import research_node
        result = await research_node(state)
        assert len(result["research_results"]) == 2


async def test_research_result_contains_expected_keys(sample_chunks):
    mock_msg = MagicMock()
    mock_msg.content = "Apple summary."
    state = _make_state(sub_questions=["What was Apple revenue?"])
    with (
        patch("agents.research_agent.retrieve_relevant_chunks", new=AsyncMock(return_value=sample_chunks)),
        patch("langchain_groq.ChatGroq.ainvoke", new=AsyncMock(return_value=mock_msg)),
    ):
        from agents.research_agent import research_node
        result = await research_node(state)
        r = result["research_results"][0]
        assert "sub_question" in r
        assert "chunks" in r
        assert "summary" in r


async def test_research_handles_no_chunks():
    """When retrieval returns nothing, research reports no info found."""
    mock_msg = MagicMock()
    mock_msg.content = "No info."
    state = _make_state(sub_questions=["Obscure question"])
    with patch("agents.research_agent.retrieve_relevant_chunks", new=AsyncMock(return_value=[])):
        from agents.research_agent import research_node
        result = await research_node(state)
        assert "No relevant information" in result["research_results"][0]["summary"]


# ── SummarizerAgent — _collect_citations (pure function) ─────────────────────

def test_collect_citations_deduplicates(sample_chunks):
    from agents.summarizer_agent import _collect_citations
    # Same chunks in two research results — should deduplicate
    research_results = [
        {"sub_question": "q1", "chunks": sample_chunks},
        {"sub_question": "q2", "chunks": sample_chunks},
    ]
    citations = _collect_citations(research_results)
    assert len(citations) == len(sample_chunks)


def test_collect_citations_shape(sample_chunks):
    from agents.summarizer_agent import _collect_citations
    citations = _collect_citations([{"sub_question": "q", "chunks": sample_chunks}])
    for c in citations:
        assert "document_id" in c
        assert "filename" in c
        assert "chunk_index" in c
        assert "excerpt" in c
        assert "score" in c


def test_collect_citations_empty_results():
    from agents.summarizer_agent import _collect_citations
    assert _collect_citations([]) == []


# ── SummarizerAgent ───────────────────────────────────────────────────────────

async def test_summarizer_returns_final_answer(sample_chunks):
    mock_msg = MagicMock()
    mock_msg.content = "Apple's revenue declined 3% [Source: AAPL_10K_2023.pdf, Page 42]."
    state = _make_state(
        analysis="Key metrics: revenue $383.3B, down 3%.",
        research_results=[{"sub_question": "q", "chunks": sample_chunks, "summary": "summary"}],
    )
    with patch("langchain_groq.ChatGroq.ainvoke", new=AsyncMock(return_value=mock_msg)):
        from agents.summarizer_agent import summarizer_node
        result = await summarizer_node(state)
        assert result["final_answer"] != ""
        assert isinstance(result["citations"], list)
        assert result["model_used"].startswith("groq/")


# ── FinancialAnalysisAgent ────────────────────────────────────────────────────

async def test_analysis_returns_structured_output(sample_chunks):
    mock_msg = MagicMock()
    mock_msg.content = "**Key Metrics**\n- Revenue: $383.3B\n**Risk Factors**\n- Currency risk"
    state = _make_state(
        research_results=[{"sub_question": "q", "chunks": sample_chunks, "summary": "Apple revenue $383.3B"}],
    )
    with patch("langchain_groq.ChatGroq.ainvoke", new=AsyncMock(return_value=mock_msg)):
        from agents.financial_analysis_agent import financial_analysis_node
        result = await financial_analysis_node(state)
        assert result["analysis"] != ""


async def test_analysis_handles_empty_research():
    state = _make_state(research_results=[])
    from agents.financial_analysis_agent import financial_analysis_node
    result = await financial_analysis_node(state)
    assert "No research data" in result["analysis"]
