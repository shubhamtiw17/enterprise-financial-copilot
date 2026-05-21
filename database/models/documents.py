from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    id: str
    filename: str
    file_path: str
    total_chunks: Optional[int] = None
