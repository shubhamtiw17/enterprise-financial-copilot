from typing import List, Optional
from vectorstore.pgvector_client import similarity_search


async def retrieve_chunks(
    query_embedding: List[float],
    top_k: int = 6,
    document_ids: Optional[List[str]] = None,
) -> List[dict]:
    return await similarity_search(
        query_embedding=query_embedding,
        top_k=top_k,
        document_ids=document_ids,
    )
