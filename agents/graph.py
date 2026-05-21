from typing import List, Optional
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.planner_agent import planner_node
from agents.research_agent import research_node
from agents.financial_analysis_agent import financial_analysis_node
from agents.summarizer_agent import summarizer_node


def build_agent_graph():
    """
    Builds and compiles the LangGraph pipeline.

    Graph topology (linear):
      planner → research → analysis → summarizer → END

    Returns a compiled graph that accepts an AgentState dict as input
    and returns an updated AgentState dict as output.
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)
    graph.add_node("analysis", financial_analysis_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "analysis")
    graph.add_edge("analysis", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()


# Module-level compiled graph — import this in the route handler.
# Compiled once on startup, reused for every request.
agent_graph = build_agent_graph()


async def run_agent_query(
    question: str,
    document_ids: Optional[List[str]] = None,
) -> dict:
    """
    Entry point called by the FastAPI route.
    Initialises AgentState and runs the full graph.
    Returns the final state dict with answer and citations.
    """
    initial_state: AgentState = {
        "original_query": question,
        "document_ids": document_ids,
        "sub_questions": [],
        "research_results": [],
        "analysis": "",
        "final_answer": "",
        "citations": [],
        "model_used": "",
    }

    final_state = await agent_graph.ainvoke(initial_state)

    return {
        "answer": final_state["final_answer"],
        "citations": final_state["citations"],
        "model_used": final_state["model_used"],
        "sub_questions": final_state["sub_questions"],
        "analysis": final_state["analysis"],
    }
