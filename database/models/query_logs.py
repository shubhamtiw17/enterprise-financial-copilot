from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class QueryLog:
    question: str
    answer: str
    model_used: str
    latency_ms: float
    citation_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
