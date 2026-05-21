import logging
from fastapi import APIRouter, HTTPException
from backend.models.request_models import AgentQueryRequest
from backend.models.response_models import AgentQueryResponse
from agents.graph import run_agent_query

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/agent-query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest):
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
