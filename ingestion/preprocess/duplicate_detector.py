import hashlib
from typing import List


def deduplicate_chunks(chunks: List[dict]) -> List[dict]:
    seen = set()
    unique = []
    for chunk in chunks:
        fingerprint = hashlib.md5(chunk["text"].encode()).hexdigest()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(chunk)
    return unique
