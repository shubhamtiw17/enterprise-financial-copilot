import asyncio
import asyncpg
from backend.config import get_settings

settings = get_settings()

async def verify():
    conn = await asyncpg.connect(settings.database_url)
    
    docs = await conn.fetch("SELECT id, filename, total_chunks FROM documents")
    print("Documents:")
    for doc in docs:
        print(f"  {doc['filename']} — {doc['total_chunks']} chunks")
    
    count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
    print(f"\nTotal chunks in database: {count}")
    
    sample = await conn.fetchrow("SELECT chunk_index, text, source FROM chunks LIMIT 1")
    print(f"\nSample chunk:")
    print(f"  Source: {sample['source']}")
    print(f"  Index: {sample['chunk_index']}")
    print(f"  Text: {sample['text'][:200]}")
    
    await conn.close()

asyncio.run(verify())