import json
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import AgentState
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

PLANNER_SYSTEM_PROMPT = """You are a financial research planning assistant.
Decompose a financial question into 2-4 focused sub-questions that can each be answered
by searching financial documents independently.

Rules:
- If the question is self-contained, return it as a single sub-question.
- If it compares multiple companies, create one sub-question per company.
- If it spans multiple topics, split by topic.
- Return ONLY a valid JSON array of strings, no explanation.

Example input: "Compare Apple and Tesla revenue risks"
Example output: ["What are Apple's main revenue risks?", "What are Tesla's main revenue risks?"]"""


async def planner_node(state: AgentState) -> AgentState:
    query = state["original_query"]
    logger.info(f"Planner: decomposing query: {query[:80]}")

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.0,
        max_tokens=512,
    )

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Decompose this financial question: {query}"),
    ]

    response = await llm.ainvoke(messages)
    raw = response.content.strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        sub_questions = json.loads(raw)
        if not isinstance(sub_questions, list):
            raise ValueError("Expected a JSON array")
    except Exception as e:
        logger.warning(f"Planner JSON parse failed ({e}), using original query")
        sub_questions = [query]

    logger.info(f"Planner produced {len(sub_questions)} sub-question(s)")
    return {**state, "sub_questions": sub_questions}
