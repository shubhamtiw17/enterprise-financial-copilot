from typing import List


def build_prompt(question: str, chunks: List[dict]) -> List[dict]:
    context_parts = [
        f"[Source {i+1}: {c['source']}, Page {c['page_number']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ]
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a financial analyst assistant.\n"
        "Answer questions using ONLY the provided context from financial documents.\n"
        "Always cite your sources by referencing [Source N] in your answer.\n"
        "If the answer is not in the context, say so clearly."
    )

    user_message = (
        f"Context from financial documents:\n\n{context}\n\n---\n\n"
        f"Question: {question}\n\nAnswer based on the context above, citing sources:"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def build_citations(chunks: List[dict]) -> List[dict]:
    return [
        {
            "document_id": chunk["document_id"],
            "filename": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "excerpt": chunk["text"][:200],
            "score": round(float(
                chunk.get("rerank_score", chunk.get("score", 0.0))
            ), 4),
        }
        for chunk in chunks
    ]
