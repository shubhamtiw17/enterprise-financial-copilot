-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Stores metadata about each uploaded document
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    total_pages INT DEFAULT 0,
    total_chunks INT DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Stores each chunk with its embedding
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    text        TEXT NOT NULL,
    page_number INT,
    source      TEXT,
    embedding   vector(384),
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Index for fast similarity search
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);