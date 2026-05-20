import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# This model runs locally - no API key needed, free forever
# all-MiniLM-L6-v2 is small, fast, and good enough for financial text
MODEL_NAME = "all-MiniLM-L6-v2"

# Load model once at module level - expensive to reload every time
model = SentenceTransformer(MODEL_NAME)
logger.info(f"Loaded embedding model: {MODEL_NAME}")


def embed_chunks(chunks: List[dict]) -> List[dict]:
    """
    Takes a list of chunk dicts, adds an 'embedding' field to each.
    Returns the same chunks with embeddings attached.
    """
    texts = [chunk["text"] for chunk in chunks]

    logger.info(f"Embedding {len(texts)} chunks...")

    # encode() returns a numpy array of shape (num_chunks, embedding_dim)
    embeddings = model.encode(texts, show_progress_bar=True)

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    logger.info(f"Done. Each embedding has {len(chunks[0]['embedding'])} dimensions")
    return chunks


def embed_query(question: str) -> List[float]:
    """
    Embeds a single question for similarity search.
    """
    embedding = model.encode(question)
    return embedding.tolist()