import asyncio
from backend.services.rag_service import run_rag_query

async def test():
    result = await run_rag_query(
        question="What are Apple's main revenue risks?",
        top_k=6
    )
    print("ANSWER:")
    print(result["answer"])
    print("\nCITATIONS:")
    for c in result["citations"]:
        print(f"  - {c['filename']} page {c['page_number']} (score: {c['score']})")

asyncio.run(test())