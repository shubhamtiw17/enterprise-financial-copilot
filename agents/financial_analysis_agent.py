import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import AgentState
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

ANALYSIS_SYSTEM_PROMPT = """You are a senior financial analyst.
Given research summaries from financial documents, produce a structured analysis.

Your analysis must include:
1. Key Metrics - specific numbers (revenue, margins, growth rates, EPS, etc.)
2. Risk Factors - material risks mentioned in the filings
3. Comparative Insights - similarities and differences if multiple companies are covered
4. Trends - year-over-year or sequential changes if data allows

Be precise and factual. Use bullet points. Reference company names explicitly.
Keep the analysis under 400 words."""


async def financial_analysis_node(state: AgentState) -> AgentState:
    """
    LangGraph node: synthesizes research results into a structured financial analysis.
    Updates state with analysis string.
    """
    research_results = state["research_results"]
    original_query = state["original_query"]

    if not research_results:
        return {**state, "analysis": "No research data available to analyze."}

    research_block = "\n\n".join(
        f"### Sub-question: {r['sub_question']}\n{r['summary']}"
        for r in research_results
    )

    logger.info(f"Analysis: synthesizing {len(research_results)} research result(s)")

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.1,
        max_tokens=1024,
    )

    messages = [
        SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original question: {original_query}\n\n"
                f"Research summaries:\n\n{research_block}"
            )
        ),
    ]

    response = await llm.ainvoke(messages)
    logger.info("Analysis complete")
    return {**state, "analysis": response.content.strip()}
