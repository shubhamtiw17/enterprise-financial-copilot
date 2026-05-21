from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    id: str
    document_id: str
    chunk_index: int
    text: str
    page_number: int
    source: str
    embedding: List[float]
