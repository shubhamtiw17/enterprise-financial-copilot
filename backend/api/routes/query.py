import logging
from fastapi import APIRouter, HTTPException
from backend.models.request_models import QueryRequest
from backend.models.response_models import QueryResponse
from backend.services.rag_service import run_rag_query

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"Query: {request.question[:80]}")

    result = await run_rag_query(
        question=request.question,
        top_k=request.top_k,
        document_ids=request.document_ids,
    )

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        citations=result["citations"],
        model_used=result["model_used"],
    )
