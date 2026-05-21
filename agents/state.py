from typing import List, Optional, TypedDict


class ResearchResult(TypedDict):
    sub_question: str
    chunks: List[dict]
    summary: str


class AgentState(TypedDict):
    """
    Shared state passed between every node in the LangGraph pipeline.

    Flow:
      original_query  → PlannerAgent    → sub_questions
      sub_questions   → ResearchAgent   → research_results
      research_results→ AnalysisAgent   → analysis
      analysis        → SummarizerAgent → final_answer, citations
    """
    original_query: str
    document_ids: Optional[List[str]]

    # Planner output
    sub_questions: List[str]

    # Research output
    research_results: List[ResearchResult]

    # Analysis output
    analysis: str

    # Final output
    final_answer: str
    citations: List[dict]
    model_used: str
