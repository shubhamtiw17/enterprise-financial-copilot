from typing import List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    top_k: int = 6


class AgentQueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
