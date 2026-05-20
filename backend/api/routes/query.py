import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.config import get_settings

router = APIRouter()
settings = get_settings()
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

    return QueryResponse(
        question=request.question,
        answer="RAG pipeline not connected yet - coming next.",
        citations=[],
        model_used=settings.llm_provider,
    )