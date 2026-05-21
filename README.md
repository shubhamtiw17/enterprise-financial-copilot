# Enterprise Financial Copilot

A question-answering system for financial documents. You upload SEC filings or annual reports, ask questions in plain English, and get answers with exact source citations pointing back to the original document pages.

---

## What It Does

1. You upload a PDF (e.g. Apple 10-K)
2. The system extracts text, splits it into chunks, converts each chunk into a vector, and stores everything in PostgreSQL
3. When you ask a question, it finds the most relevant chunks using vector similarity, reranks them with a cross-encoder model, and sends them along with your question to an LLM
4. The LLM answers using only those chunks and cites which source it used

There is also a multi-agent mode for complex questions. Instead of one retrieval pass, a planner breaks the question into sub-questions, each sub-question runs retrieval in parallel, a financial analysis step structures the findings, and a summarizer writes the final answer.

---

## How Each Component Works

### Ingestion — turning a PDF into searchable vectors

```
PDF file
  → pdf_loader.py       extracts text page by page
  → cleaner.py          removes whitespace artifacts, page numbers, control characters
  → recursive_chunker.py splits text into overlapping chunks (800 chars, 100 overlap)
  → huggingface_embeddings.py converts each chunk to a 384-dimensional vector
  → pgvector_client.py  stores the document metadata and chunks in PostgreSQL
```

The chunking overlap means a sentence that falls at a chunk boundary appears in both chunks, so retrieval does not miss context at the edges.

---

### Retrieval — finding relevant chunks for a question

```
Question
  → embed_query()           converts the question to a 384-dim vector
  → similarity_search()     finds the top 12 closest chunks in pgvector using cosine distance
  → rerank_chunks()         cross-encoder model scores each (question, chunk) pair
  → top 6 returned          sorted by reranker score, not original similarity score
```

Two-stage retrieval exists because vector similarity finds broadly related chunks, but the cross-encoder (which reads question and chunk together) scores actual relevance more accurately. Running the cross-encoder on all chunks would be too slow, so vector similarity narrows the candidates first.

---

### Standard RAG Query

```
Question → retrieve_relevant_chunks() → build_prompt() → call_llm() → answer + citations
```

`build_prompt()` in `citation_service.py` formats the chunks as numbered sources and instructs the LLM to cite them. The LLM never sees the raw database — only the text of the retrieved chunks.

---

### Multi-Agent Query

```
Question
  → planner_node()             LLM breaks question into 2–4 sub-questions (JSON array)
  → research_node()            runs retrieve + summarize for each sub-question concurrently
  → financial_analysis_node()  LLM structures findings into metrics, risks, comparisons
  → summarizer_node()          LLM writes final answer; citations collected from all chunks
```

All sub-questions run concurrently via `asyncio.gather`. For a single-company question the planner returns one sub-question and the pipeline behaves identically to standard mode.

---

## Project Structure

```
enterprise-financial-copilot/
│
├── backend/                        FastAPI application
│   ├── main.py                     app setup, middleware, route registration
│   ├── config.py                   reads .env into typed Settings object
│   ├── dependencies.py             FastAPI dependency injection helpers
│   │
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py           GET  /health
│   │   │   ├── upload.py           POST /upload, GET /upload/status/{id}
│   │   │   ├── query.py            POST /query
│   │   │   ├── agent_query.py      POST /agent-query
│   │   │   └── metrics.py          GET  /metrics
│   │   └── middleware/
│   │       ├── auth.py             optional API key check (disabled in dev)
│   │       └── logging.py          logs method, path, status, latency per request
│   │
│   ├── services/
│   │   ├── rag_service.py          orchestrates the full query pipeline
│   │   ├── embedding_service.py    wraps embed_query and embed_chunks
│   │   ├── retrieval_service.py    wraps pgvector similarity_search
│   │   ├── reranker_service.py     wraps cross-encoder rerank_chunks
│   │   ├── citation_service.py     builds LLM prompt and citation list from chunks
│   │   └── llm_service.py          calls Groq / OpenAI / Gemini / Bedrock
│   │
│   ├── models/
│   │   ├── request_models.py       Pydantic models for incoming requests
│   │   └── response_models.py      Pydantic models for API responses
│   │
│   └── utils/
│       ├── logger.py               centralised logging formatter
│       ├── helpers.py              small utilities (filename extraction, truncation)
│       └── constants.py            shared constants (upload types, token limits)
│
├── agents/                         multi-step query pipeline using LangGraph
│   ├── state.py                    AgentState TypedDict shared across all nodes
│   ├── graph.py                    builds and compiles the LangGraph StateGraph
│   ├── planner_agent.py            decomposes question into sub-questions
│   ├── research_agent.py           runs retrieval + summarisation per sub-question
│   ├── financial_analysis_agent.py structures findings into metrics and risk factors
│   ├── summarizer_agent.py         writes final answer, deduplicates citations
│   ├── memory_manager.py           stores conversation turns (used for multi-turn chat)
│   └── tools/
│       ├── retrieval_tool.py       LangChain @tool wrapping retrieve_relevant_chunks
│       ├── sql_tool.py             LangChain @tool for read-only document metadata queries
│       └── calculator_tool.py      LangChain @tool for arithmetic (margins, growth rates)
│
├── rag/                            retrieval and embedding modules
│   ├── chunking/
│   │   ├── recursive_chunker.py    splits pages using RecursiveCharacterTextSplitter
│   │   └── semantic_chunker.py     splits by sentence similarity instead of character count
│   │
│   ├── embeddings/
│   │   ├── huggingface_embeddings.py  all-MiniLM-L6-v2 running locally (384 dimensions)
│   │   └── bedrock_embeddings.py      Amazon Titan embeddings via AWS Bedrock
│   │
│   ├── retrievers/
│   │   ├── vector_retriever.py     pgvector cosine similarity search
│   │   ├── bm25_retriever.py       keyword-based BM25 ranking over a corpus
│   │   └── hybrid_retriever.py     weighted combination of vector and BM25 scores
│   │
│   ├── reranking/
│   │   └── cross_encoder_reranker.py  ms-marco-MiniLM-L-6-v2 cross-encoder
│   │
│   └── prompts/
│       ├── system_prompt.txt       base system instruction for the LLM
│       ├── rag_prompt.txt          user message template with {context} and {question}
│       └── citation_prompt.txt     instruction for inline citation format
│
├── ingestion/                      document loading and preprocessing
│   ├── loaders/
│   │   ├── pdf_loader.py           extracts text per page using pypdf
│   │   ├── csv_loader.py           converts each CSV row into a text representation
│   │   └── markdown_loader.py      splits markdown by ## headings into sections
│   │
│   ├── preprocess/
│   │   ├── cleaner.py              removes whitespace noise and control characters
│   │   ├── metadata_extractor.py   extracts fiscal year and ticker from text
│   │   └── duplicate_detector.py   MD5 fingerprint deduplication of chunks
│   │
│   └── pipelines/
│       └── ingestion_pipeline.py   runs all steps in sequence: load→clean→chunk→embed→store
│
├── vectorstore/                    PostgreSQL + pgvector interface
│   ├── schema.sql                  CREATE TABLE statements for documents and chunks
│   ├── pgvector_client.py          insert_document, insert_chunks, similarity_search
│   └── indexing.py                 creates and drops the IVFFlat index on the embedding column
│
├── database/                       data models and connection pool
│   ├── connection.py               asyncpg connection pool (get_pool, close_pool)
│   └── models/
│       ├── documents.py            Document dataclass matching the documents table
│       ├── embeddings.py           Chunk dataclass matching the chunks table
│       └── query_logs.py           QueryLog dataclass for recording query history
│
├── llmops/                         observability and evaluation
│   ├── monitoring/
│   │   ├── langsmith_tracing.py    sets LANGSMITH_* env vars so all LLM calls are traced
│   │   ├── metrics.py              in-memory counters and latency recording
│   │   └── token_tracking.py       tracks input/output token counts per model
│   │
│   ├── evaluations/
│   │   ├── faithfulness_eval.py    scores whether the answer is grounded in the context
│   │   ├── relevance_eval.py       scores average reranker score of retrieved chunks
│   │   └── latency_eval.py         @track_latency decorator that logs function duration
│   │
│   └── prompt_versions/
│       ├── v1.yaml                 baseline prompt (system + user template)
│       └── v2.yaml                 structured prompt requesting confidence level
│
├── frontend/                       Streamlit web interface
│   ├── app.py                      entry point, page routing
│   ├── pages/
│   │   ├── chat.py                 chat interface with citation expanders
│   │   ├── upload.py               file upload with ingestion progress polling
│   │   └── analytics.py            API health and uptime metrics
│   └── components/
│       ├── sidebar.py              upload widget, mode toggle, API status
│       └── citation_card.py        renders a single citation with filename and excerpt
│
├── tests/
│   ├── conftest.py                 shared fixtures: TestClient, sample chunks, mock LLM
│   ├── test_api.py                 HTTP contract tests for all endpoints
│   ├── test_rag_pipeline.py        unit tests for prompt building and retrieval logic
│   ├── test_agents.py              unit tests for each agent node
│   ├── test_embeddings.py          unit tests for embed_query and embed_chunks
│   └── test_retrieval.py           unit tests for vector and BM25 retrievers
│
├── scripts/
│   ├── ingest.py                   CLI: ingest a local PDF directly (bypasses upload route)
│   └── verify_db.py                CLI: prints documents and chunk count in the database
│
├── docker/
│   ├── backend.Dockerfile          python:3.10-slim, installs CPU torch + requirements
│   ├── frontend.Dockerfile         python:3.10-slim, installs streamlit + httpx
│   └── postgres.Dockerfile         pgvector:pg16, runs schema.sql on first start
│
├── kubernetes/
│   ├── backend-deployment.yaml     2 replicas, readiness probe on /health
│   ├── frontend-deployment.yaml    1 replica, API_BASE_URL points to backend service
│   ├── postgres-deployment.yaml    1 replica with PersistentVolumeClaim
│   └── ingress.yaml                routes /api to backend, / to frontend
│
├── monitoring/
│   └── prometheus/
│       └── prometheus.yml          scrapes /metrics from backend every 15 seconds
│
├── .github/workflows/
│   ├── ci.yml                      runs flake8 + pytest on every push and pull request
│   └── docker-build.yml            builds both Docker images on every push to main
│
├── docker-compose.yml              runs postgres, redis, backend, frontend together
├── requirements.txt                all Python dependencies
├── pytest.ini                      asyncio_mode=auto, pythonpath=. for CI compatibility
└── Makefile                        convenience commands (see below)
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Web framework | FastAPI | async-native, automatic OpenAPI docs |
| Frontend | Streamlit | rapid UI without a separate JS project |
| Database | PostgreSQL + pgvector | stores both document metadata and embedding vectors in one place |
| Embeddings | all-MiniLM-L6-v2 | runs locally, no API key, 384 dimensions, fast |
| Reranking | ms-marco-MiniLM-L-6-v2 cross-encoder | more accurate relevance scoring than dot-product alone |
| LLM | Groq (LLaMA 3.3-70b) | fast inference, no GPU needed |
| Agent framework | LangGraph | stateful graph with typed state passed between nodes |
| Tracing | LangSmith | records every LLM call with inputs, outputs, and latency |
| Containerisation | Docker + Docker Compose | one command to start the full stack |

---

## Setup

### Prerequisites

- Python 3.10
- Docker Desktop
- A Groq API key (free at console.groq.com)
- A LangSmith API key (free at smith.langchain.com)

### 1. Clone and install

```bash
git clone https://github.com/shubhamtiw17/enterprise-financial-copilot.git
cd enterprise-financial-copilot
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in:

```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
DATABASE_URL=postgresql://copilot:copilot@localhost:5432/copilot
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=financial-copilot
```

### 3. Start the database

```bash
docker-compose up postgres redis -d
```

---

## Running

### Option A — Local development

```bash
# Terminal 1 — backend
python -m uvicorn backend.main:app --reload

# Terminal 2 — frontend
streamlit run frontend/app.py
```

- Frontend: http://localhost:8501
- API docs: http://localhost:8000/docs

### Option B — Full Docker stack

```bash
docker-compose up --build
```

All four services (postgres, redis, backend, frontend) start together. The backend image installs dependencies at build time; the HuggingFace models download on first boot into a persistent volume.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns app status and LLM provider |
| GET | `/metrics` | Returns API uptime |
| POST | `/upload` | Accepts a PDF/CSV/MD file, starts ingestion in background |
| GET | `/upload/status/{id}` | Returns ingestion status: processing / ready / error |
| POST | `/query` | Standard single-pass retrieval and answer |
| POST | `/agent-query` | Multi-agent pipeline with planner and parallel research |

### POST /query

```json
{
  "question": "What was Apple gross margin in 2023?",
  "top_k": 6
}
```

### POST /agent-query

```json
{
  "question": "Compare Apple and Microsoft revenue risks"
}
```

Response includes `sub_questions` (what the planner generated) and `analysis` (intermediate structured findings) in addition to `answer` and `citations`.

---

## Running Tests

```bash
pytest tests/ -v
```

35 tests across 5 files. All tests mock external dependencies (LLM, database) so no live services are needed.

| File | What it tests |
|---|---|
| `test_api.py` | HTTP status codes and response shapes for all endpoints |
| `test_rag_pipeline.py` | prompt building, retrieval, citation extraction |
| `test_agents.py` | each agent node in isolation |
| `test_embeddings.py` | embed_query and embed_chunks output shape and type |
| `test_retrieval.py` | vector retriever and BM25 ranking behaviour |

---

## Makefile Commands

```bash
make dev          # start backend + frontend locally
make docker       # docker-compose up --build
make test         # pytest tests/ -v
make lint         # flake8 on all source directories
make ingest FILE=path/to/file.pdf   # ingest a PDF directly
make verify       # print documents and chunk count in database
```

---

## LangSmith Tracing

Every LLM call is traced automatically. After running a query, open smith.langchain.com and look for the `financial-copilot` project. Each trace shows:

- `rag_pipeline` parent run
  - `ChatGroq` child run with input messages, output text, token count, and latency

For agent queries the trace tree is deeper: planner → N research runs → analysis → summarizer.
