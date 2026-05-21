import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.rag_service import run_rag_query

router = APIRouter()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    top_k: int = 6


class CitationModel(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    excerpt: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationModel]
    model_used: str


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"Query received: {request.question[:80]}")

    result = await run_rag_query(
        question=request.question,
        top_k=request.top_k,
        document_ids=request.document_ids
    )

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        citations=result["citations"],
        model_used=result["model_used"]
    )
