import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
logger.info(f"Loaded embedding model: {MODEL_NAME}")


def embed_chunks(chunks: List[dict]) -> List[dict]:
    texts = [chunk["text"] for chunk in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    logger.info(f"Done. Embedding dim: {len(chunks[0]['embedding'])}")
    return chunks


def embed_query(question: str) -> List[float]:
    return model.encode(question).tolist()
