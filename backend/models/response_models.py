from typing import List
from pydantic import BaseModel


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


class AgentQueryResponse(BaseModel):
    question: str
    sub_questions: List[str]
    analysis: str
    answer: str
    citations: List[CitationModel]
    model_used: str


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    llm_provider: str
