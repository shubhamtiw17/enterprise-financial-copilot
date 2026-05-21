import logging
from typing import List
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def call_llm(messages: List[dict]) -> str:
    provider = settings.llm_provider
    if provider == "groq":
        return await _call_groq(messages)
    elif provider == "openai":
        return await _call_openai(messages)
    elif provider == "gemini":
        return await _call_gemini(messages)
    elif provider == "bedrock":
        return await _call_bedrock(messages)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def _call_groq(messages: List[dict]) -> str:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.2,
        max_tokens=MAX_TOKENS,
    )
    lc_messages = [
        SystemMessage(content=m["content"]) if m["role"] == "system"
        else HumanMessage(content=m["content"])
        for m in messages
    ]
    response = await llm.ainvoke(lc_messages)
    return response.content


async def _call_openai(messages: List[dict]) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.2,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


async def _call_gemini(messages: List[dict]) -> str:
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_model)
    system_text = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
    response = model.generate_content(f"{system_text}\n\n{user_text}")
    return response.text


async def _call_bedrock(messages: List[dict]) -> str:
    import boto3
    import json
    import asyncio
    client = boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "messages": messages,
    }
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.invoke_model(
            modelId=settings.bedrock_model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        ),
    )
    return json.loads(response["body"].read())["content"][0]["text"]


MAX_TOKENS = 1500
