import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_chunk_pages(pages: List[Dict], similarity_threshold: float = 0.7) -> List[Dict]:
    chunks = []
    chunk_index = 0

    for page in pages:
        sentences = [s.strip() for s in page["text"].split(".") if s.strip()]
        if not sentences:
            continue

        embeddings = model.encode(sentences)
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]) /
                        (np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-8))
            if sim >= similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append({
                    "chunk_index": chunk_index,
                    "text": ". ".join(current_chunk) + ".",
                    "page_number": page["page_number"],
                    "source": page["source"],
                    "file_path": page["file_path"],
                })
                chunk_index += 1
                current_chunk = [sentences[i]]

        if current_chunk:
            chunks.append({
                "chunk_index": chunk_index,
                "text": ". ".join(current_chunk) + ".",
                "page_number": page["page_number"],
                "source": page["source"],
                "file_path": page["file_path"],
            })
            chunk_index += 1

    logger.info(f"Semantic chunking produced {len(chunks)} chunks")
    return chunks
