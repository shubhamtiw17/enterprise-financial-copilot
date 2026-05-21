FROM python:3.10-slim

# System deps: libpq for asyncpg, build tools for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (~200 MB vs ~800 MB GPU wheel).
# sentence-transformers will reuse this instead of pulling the GPU build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source packages
COPY backend/     ./backend/
COPY agents/      ./agents/
COPY rag/         ./rag/
COPY vectorstore/ ./vectorstore/
COPY ingestion/   ./ingestion/
COPY llmops/      ./llmops/
COPY scripts/     ./scripts/

# Directory for uploaded PDFs (also mounted as a volume in compose)
RUN mkdir -p /app/uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
