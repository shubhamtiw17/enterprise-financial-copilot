import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.graph import run_agent_query

router = APIRouter()
logger = logging.getLogger(__name__)


class AgentQueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None


class CitationModel(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    excerpt: str
    score: float


class AgentQueryResponse(BaseModel):
    question: str
    sub_questions: List[str]
    analysis: str
    answer: str
    citations: List[CitationModel]
    model_used: str


@router.post("/agent-query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest):
    """
    Multi-agent RAG endpoint.
    Runs: Planner → Research (parallel) → FinancialAnalysis → Summarizer.
    Use for complex queries that span multiple companies or topics.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"Agent query: {request.question[:80]}")

    result = await run_agent_query(
        question=request.question,
        document_ids=request.document_ids,
    )

    return AgentQueryResponse(
        question=request.question,
        sub_questions=result["sub_questions"],
        analysis=result["analysis"],
        answer=result["answer"],
        citations=result["citations"],
        model_used=result["model_used"],
    )
