import asyncio
import logging
from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import AgentState, ResearchResult
from backend.services.rag_service import retrieve_relevant_chunks
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

RESEARCH_SYSTEM_PROMPT = """You are a financial document analyst.
Given retrieved excerpts from financial filings, write a concise factual summary
that directly answers the question. Include specific numbers, percentages, and
dates when present. Keep it under 200 words. Do not speculate beyond the sources."""


async def _research_one(
    llm: ChatGroq,
    sub_question: str,
    document_ids: List[str],
) -> ResearchResult:
    """Retrieves chunks for one sub-question and summarizes them."""
    chunks = await retrieve_relevant_chunks(
        question=sub_question,
        top_k=6,
        document_ids=document_ids or None,
    )

    if not chunks:
        return ResearchResult(
            sub_question=sub_question,
            chunks=[],
            summary="No relevant information found in the uploaded documents.",
        )

    context = "\n\n".join(
        f"[Source: {c['source']}, Page {c['page_number']}]\n{c['text']}"
        for c in chunks
    )

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Question: {sub_question}\n\nDocument excerpts:\n{context}"
        ),
    ]

    response = await llm.ainvoke(messages)
    return ResearchResult(
        sub_question=sub_question,
        chunks=chunks,
        summary=response.content.strip(),
    )


async def research_node(state: AgentState) -> AgentState:
    """
    LangGraph node: runs all sub-questions through RAG concurrently.
    Each sub-question gets its own retrieval + summarization pass.
    Updates state with research_results list.
    """
    sub_questions = state["sub_questions"]
    document_ids = state.get("document_ids") or []

    logger.info(f"Research: running {len(sub_questions)} sub-question(s) concurrently")

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.1,
        max_tokens=512,
    )

    results: List[ResearchResult] = await asyncio.gather(
        *[_research_one(llm, q, document_ids) for q in sub_questions]
    )

    logger.info(f"Research complete: {len(results)} result(s)")
    return {**state, "research_results": list(results)}
