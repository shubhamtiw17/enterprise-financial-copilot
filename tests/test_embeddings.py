import pytest
from unittest.mock import patch, MagicMock


def test_embed_query_returns_list():
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1] * 384)
    with patch("rag.embeddings.huggingface_embeddings.model", mock_model):
        from rag.embeddings.huggingface_embeddings import embed_query
        result = embed_query("What was Apple revenue?")
        assert isinstance(result, list)
        assert len(result) == 384


def test_embed_chunks_adds_embedding_field():
    mock_model = MagicMock()
    import numpy as np
    mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
    chunks = [
        {"text": "Apple revenue was $383B", "chunk_index": 0},
        {"text": "Revenue declined 3%", "chunk_index": 1},
    ]
    with patch("rag.embeddings.huggingface_embeddings.model", mock_model):
        from rag.embeddings.huggingface_embeddings import embed_chunks
        result = embed_chunks(chunks)
        assert all("embedding" in c for c in result)
        assert len(result[0]["embedding"]) == 384


def test_embed_query_returns_floats():
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.5] * 384)
    with patch("rag.embeddings.huggingface_embeddings.model", mock_model):
        from rag.embeddings.huggingface_embeddings import embed_query
        result = embed_query("test")
        assert all(isinstance(v, float) for v in result)
