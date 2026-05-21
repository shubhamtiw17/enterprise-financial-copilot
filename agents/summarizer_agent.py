import logging
from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import AgentState
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SUMMARIZER_SYSTEM_PROMPT = """You are a financial reporting assistant.
Given a structured analysis, write a clear, professional final answer to the user's question.

Requirements:
- Open with a direct answer to the question
- Support your answer with specific facts and numbers from the analysis
- Cite sources as [Source: <filename>, Page <N>] inline
- Use clear paragraph structure, not bullet points
- Keep the response under 350 words
- End with a one-sentence key takeaway"""


def _collect_citations(research_results: list) -> List[dict]:
    seen = set()
    citations = []
    for result in research_results:
        for chunk in result.get("chunks", []):
            key = (chunk.get("document_id"), chunk.get("chunk_index"))
            if key not in seen:
                seen.add(key)
                citations.append({
                    "document_id": chunk.get("document_id", ""),
                    "filename": chunk.get("source", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "excerpt": chunk.get("text", "")[:200],
                    "score": round(float(
                        chunk.get("rerank_score", chunk.get("score", 0.0))
                    ), 4),
                })
    return citations


async def summarizer_node(state: AgentState) -> AgentState:
    analysis = state["analysis"]
    original_query = state["original_query"]
    research_results = state["research_results"]

    logger.info("Summarizer: generating final answer")

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.2,
        max_tokens=1024,
    )

    messages = [
        SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"User question: {original_query}\n\nStructured analysis:\n{analysis}"
        ),
    ]

    response = await llm.ainvoke(messages)
    citations = _collect_citations(research_results)

    logger.info(f"Summarizer complete. {len(citations)} citation(s) collected.")
    return {
        **state,
        "final_answer": response.content.strip(),
        "citations": citations,
        "model_used": f"groq/{settings.groq_model}",
    }
