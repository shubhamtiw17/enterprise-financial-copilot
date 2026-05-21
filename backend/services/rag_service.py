import logging
from typing import List, Optional
from rag.embeddings.huggingface_embeddings import embed_query
from vectorstore.pgvector_client import similarity_search
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def retrieve_relevant_chunks(
    question: str,
    top_k: int = 6,
    document_ids: Optional[List[str]] = None
) -> List[dict]:
    from rag.reranking.cross_encoder_reranker import rerank_chunks

    # Retrieve more candidates than needed for reranking
    candidates = await similarity_search(
        query_embedding=embed_query(question),
        top_k=top_k * 2,      # fetch double, reranker picks best half
        document_ids=document_ids
    )

    # Rerank candidates
    reranked = rerank_chunks(question, candidates, top_k=top_k)

    logger.info(f"Retrieved {len(candidates)} candidates, reranked to {len(reranked)}")
    return reranked


def build_prompt(question: str, chunks: List[dict]) -> List[dict]:
    """
    Builds the message list to send to the LLM.
    Injects retrieved chunks as context.
    """
    # Format each chunk with its source
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['source']}, Page {chunk['page_number']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are a financial analyst assistant. 
Answer questions using ONLY the provided context from financial documents.
Always cite your sources by referencing [Source N] in your answer.
If the answer is not in the context, say so clearly."""

    user_message = f"""Context from financial documents:

{context}

---

Question: {question}

Answer based on the context above, citing sources:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]


async def run_rag_query(
    question: str,
    top_k: int = 6,
    document_ids: Optional[List[str]] = None
) -> dict:
    """
    Full RAG pipeline:
    Question -> Retrieve chunks -> Build prompt -> LLM -> Answer
    """
    # Step 1: Retrieve relevant chunks
    chunks = await retrieve_relevant_chunks(question, top_k, document_ids)

    if not chunks:
        return {
            "answer": "No relevant documents found. Please upload financial documents first.",
            "citations": [],
            "model_used": settings.llm_provider
        }

    # Step 2: Build prompt with context
    messages = build_prompt(question, chunks)

    # Step 3: Call LLM
    answer = await call_llm(messages)

    # Step 4: Format citations
    citations = [
        {
            "document_id": chunk["document_id"],
            "filename": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "excerpt": chunk["text"][:200],
            "score": round(float(chunk["score"]), 4),
            "page_number": chunk["page_number"]
        }
        for chunk in chunks
    ]

    return {
        "answer": answer,
        "citations": citations,
        "model_used": settings.llm_provider
    }


async def call_llm(messages: List[dict]) -> str:
    if settings.llm_provider == "openai":
        return await call_openai(messages)
    elif settings.llm_provider == "bedrock":
        return await call_bedrock(messages)
    elif settings.llm_provider == "gemini":
        return await call_gemini(messages)
    elif settings.llm_provider == "groq":
        return await call_groq(messages)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


async def call_openai(messages: List[dict]) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.2,
        max_tokens=1500
    )
    return response.choices[0].message.content

async def call_groq(messages: List[dict]) -> str:
    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.2,
        max_tokens=1500
    )
    return response.choices[0].message.content


async def call_bedrock(messages: List[dict]) -> str:
    import boto3, json, asyncio

    client = boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1500,
        "messages": messages
    }

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.invoke_model(
            modelId=settings.bedrock_model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

async def call_gemini(messages: List[dict]) -> str:
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_model)

    system_text = ""
    user_text = ""

    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        elif msg["role"] == "user":
            user_text = msg["content"]

    full_prompt = f"{system_text}\n\n{user_text}"

    response = model.generate_content(full_prompt)
    return response.text