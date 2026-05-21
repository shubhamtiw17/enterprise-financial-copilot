# 🔧 Technical Decisions — Pros, Cons & Alternatives

This document explains every major technology choice in the Enterprise Financial Copilot why it was selected, what it trades off, and what you would use instead in different scenarios.

---

## 1. Web Framework — FastAPI

### Why FastAPI
FastAPI is an async Python web framework built on Pydantic and Starlette. It was chosen because ingestion and LLM calls are I/O-bound operations async handling means the server can process multiple requests without blocking.

### Pros
- Async-first: handles slow LLM calls without blocking other requests
- Auto-generates Swagger UI at `/docs` with zero extra code
- Pydantic validation: invalid request shapes are rejected automatically before your code runs
- Extremely fast: benchmarks close to Node.js and Go for I/O-bound work

### Cons
- More complex than Flask for simple use cases
- Async debugging is harder than synchronous code
- Smaller ecosystem than Django for non-API features (auth, admin panels)

### Alternatives

| Alternative | When to use instead |
|-------------|-------------------|
| **Flask** | Simpler projects, synchronous workloads, quick prototypes |
| **Django + DRF** | Full-stack apps needing ORM, admin panel, auth out of the box |
| **LiteStar** | When you need the fastest possible async framework with strict typing |
| **Express (Node.js)** | Team is JavaScript-first, real-time features needed |

---

## 2. Frontend — Streamlit

### Why Streamlit
Streamlit lets you build interactive data apps in pure Python. No HTML, CSS, or JavaScript. For a proof-of-concept AI application targeting data engineers and ML practitioners, it's the fastest path to a working UI.

### Pros
- Zero frontend knowledge required: entire UI in Python
- Built-in components for file upload, chat, progress bars, expanders
- Hot-reload: save the file and the browser updates instantly
- Free hosting on Streamlit Community Cloud

### Cons
- Not suitable for production consumer apps: re-runs entire script on every interaction
- Limited customisation: can't build complex layouts or custom components easily
- Session state is fragile: refreshing the browser loses chat history
- Not scalable: single-threaded by default

### Alternatives

| Alternative | When to use instead |
|-------------|-------------------|
| **Gradio** | ML demos, model showcases: even simpler than Streamlit |
| **React + TypeScript** | Production consumer app, complex UI, custom design system |
| **Next.js** | SEO needed, server-side rendering, full-stack JavaScript |
| **Chainlit** | Purpose-built chat UI for LLM apps: more polished than Streamlit for chatbots |

---

## 3. Vector Database — PostgreSQL + pgvector

### Why pgvector
pgvector adds vector similarity search to PostgreSQL. The project already needs PostgreSQL for document metadata and query logs pgvector means one database handles both relational and vector data.

### Pros
- No additional infrastructure: extends existing PostgreSQL
- Full SQL available alongside vector search: join vectors with metadata easily
- ACID transactions: data consistency guaranteed
- IVFFlat and HNSW index types for fast approximate nearest neighbour search
- Free and open source

### Cons
- Not purpose-built for vectors: dedicated vector databases are faster at scale
- IVFFlat index requires knowing the number of lists upfront (`lists = 100`)
- Slower than Pinecone or Weaviate for very large datasets (10M+ vectors)
- Needs tuning at scale: not zero-config like managed services

### Alternatives

| Alternative | When to use instead |
|-------------|-------------------|
| **Pinecone** | Managed, zero-ops, scales to billions of vectors, production SaaS |
| **Weaviate** | Need built-in hybrid search, GraphQL API, multi-modal support |
| **Qdrant** | High performance, Rust-based, great filtering, self-hosted |
| **Chroma** | Local development, prototyping, zero setup, in-memory option |
| **Milvus** | Large-scale production, billions of vectors, enterprise |
| **Redis (with RedisSearch)** | Already using Redis, need low-latency vector search + caching together |

---

## 4. Embedding Model — all-MiniLM-L6-v2

### Why all-MiniLM-L6-v2
A local sentence-transformer model that runs on CPU. No API key, no cost, no latency from network calls. For a document corpus of hundreds to thousands of chunks, it performs well.

### Pros
- Completely free no API costs regardless of volume
- Runs locally on CPU, no GPU required
- Fast for small-medium corpora
- 384 dimensions, smaller storage footprint than 1536-dim OpenAI embeddings
- Good performance on semantic similarity tasks

### Cons
- Lower quality than OpenAI `text-embedding-3-large` on complex financial language
- 384 dimensions may miss subtle semantic distinctions
- Needs ~90MB download on first run
- Not fine-tuned on financial domain text

### Alternatives

| Alternative | When to use instead |
|-------------|-------------------|
| **OpenAI text-embedding-3-small** | Better quality, cheap ($0.02/1M tokens), 1536 dims |
| **OpenAI text-embedding-3-large** | Highest quality, more expensive, 3072 dims |
| **Cohere embed-v3** | Strong multilingual support, good for non-English filings |
| **all-mpnet-base-v2** | Larger local model, better quality than MiniLM, slower |
| **FinBERT embeddings** | Fine-tuned on financial text, better for SEC filings specifically |
| **Voyage Finance** | Purpose-built for financial documents, commercial |

---

## 5. Reranking — Cross-Encoder (ms-marco-MiniLM-L-6-v2)

### Why Cross-Encoder Reranking
Vector similarity is approximate. It finds chunks that are semantically close but doesn't deeply understand the question-passage relationship. A cross-encoder sees both the question and the passage together, giving a much more accurate relevance score.

### Pros
- Significantly improves answer quality over pure vector search
- Runs locally, no API cost
- Can catch relevant passages that vector search ranked poorly
- Reduces hallucination by ensuring truly relevant context reaches the LLM

### Cons
- Slower than vector search — O(n) inference for n candidates
- Must be run after retrieval, adding latency
- Not fine-tuned on financial text

### Alternatives

| Alternative | When to use instead |
|-------------|-------------------|
| **Cohere Rerank API** | Managed, high quality, no local compute needed, costs per call |
| **cross-encoder/ms-marco-electra-base** | Higher accuracy than MiniLM, slower |
| **BGE Reranker** | Strong open-source alternative, multilingual |
| **No reranking** | Ultra-low latency requirements, large top-k already sufficient |
| **LLM-based reranking** | Use the LLM itself to score passages, highest quality, expensive |

---

## 6. LLM — Groq (LLaMA 3.3 70B)

### Why Groq
Groq's free tier provides access to LLaMA 3.3 70B a model comparable to GPT-4 in many benchmarks at no cost, with extremely fast inference due to Groq's custom LPU hardware.

### Pros
- Free tier: generous daily limits
- Very fast inference: LPU hardware significantly faster than GPU
- LLaMA 3.3 70B is highly capable for financial reasoning
- OpenAI-compatible API: easy to swap providers

### Cons
- Free tier has rate limits: can hit quota with heavy usage
- Less reliable than paid OpenAI for production
- LLaMA 3.3 slightly behind GPT-4o on complex reasoning tasks
- No fine-tuning option on free tier

### Alternatives

| Alternative | When to use instead |
|-------------|-------------------|
| **OpenAI GPT-4o** | Highest capability, reliable, paid best for production |
| **Anthropic Claude (Bedrock)** | Strong reasoning, large context window, enterprise compliance |
| **Google Gemini 2.0 Flash** | Free tier, multimodal, good for mixed text+image filings |
| **Mistral Large** | Strong European option, GDPR-friendly, good multilingual |
| **Local Ollama** | Completely private, no data leaves machine, needs good GPU |
| **Azure OpenAI** | Enterprise compliance, data residency guarantees, SOC2 |

---

## 7. Document Chunking — RecursiveCharacterTextSplitter

### Why Recursive Chunking
LangChain's `RecursiveCharacterTextSplitter` splits on paragraph breaks first, then sentences, then words trying to preserve semantic units. The 100-character overlap ensures sentences spanning chunk boundaries aren't lost.

### Pros
- Preserves paragraph and sentence structure
- Overlap prevents context loss at boundaries
- Configurable chunk size and overlap
- Works well for most document types out of the box

### Cons
- Character-based, not token-based actual token counts vary
- Doesn't understand document structure (headers, tables, footnotes)
- 800 characters may be too small for complex financial tables
- No awareness of section boundaries in SEC filings

### Alternatives

| Alternative | When to use instead |
|-------------|-------------------|
| **Semantic chunking** | Split on meaning changes rather than character count higher quality, slower |
| **Token-based splitter** | When you need exact token control for context window management |
| **MarkdownHeaderTextSplitter** | Documents with clear header structure |
| **Unstructured.io** | Complex PDFs with tables, images, mixed layouts |
| **Fixed-size with no overlap** | Simple use cases, storage-constrained |
| **Agentic chunking** | Use LLM to determine natural chunk boundaries highest quality, expensive |

---

## 8. Observability — LangSmith

### Why LangSmith
LangSmith traces every LLM call inputs, outputs, latency, token usage. Built by LangChain, it integrates automatically when using LangChain components. Free tier available.

### Pros
- Automatic tracing when using LangChain zero extra code
- Visual trace explorer see exactly what prompt was sent
- Latency and token tracking per run
- Can compare prompt versions
- Dataset and evaluation tools built in

### Cons
- Tightly coupled to LangChain ecosystem
- Free tier has data retention limits
- Sends data to LangChain's servers compliance concern for sensitive data
- Limited alerting compared to dedicated APM tools

### Alternatives

| Alternative | When to use instead |
|-------------|-------------------|
| **MLflow** | Open source, self-hosted, experiment tracking, model registry |
| **Weights & Biases (W&B)** | ML experiment tracking, model versioning, strong visualisations |
| **Prometheus + Grafana** | Infrastructure metrics, custom dashboards, self-hosted |
| **Helicone** | LLM-specific observability, cost tracking, caching |
| **Arize AI** | Production ML monitoring, drift detection, enterprise |
| **Custom logging** | Full control, no external dependencies, more engineering effort |

---

## 9. Caching — Redis

### Why Redis
Redis is an in-memory key-value store. In this project it's included for semantic caching storing LLM responses for repeated or similar questions so you don't call the LLM every time.

### Pros
- Extremely fast microsecond read/write latency
- Persistent storage option available
- Rich data structures, lists, sets, sorted sets beyond simple key-value
- LangChain has built-in Redis semantic cache integration

### Cons
- Not yet fully implemented in this project
- Memory-based expensive per GB compared to disk storage
- Cache invalidation when documents are updated is complex
- Semantic similarity threshold for cache hits needs tuning

### Alternatives

| Alternative | When to use instead |
|-------------|-------------------|
| **In-memory Python dict** | Single server, prototyping, no persistence needed |
| **Memcached** | Simpler key-value only, slightly faster for pure caching |
| **GPTCache** | Purpose-built semantic cache for LLM calls |
| **No caching** | Low traffic, always-fresh answers required |

---

## 10. Containerisation — Docker Compose

### Why Docker Compose
Docker Compose runs PostgreSQL+pgvector and Redis as containers. This means anyone can run the full stack with one command regardless of their OS, without installing databases manually.

### Pros
- One command setup — `docker-compose up -d`
- Consistent environment across development machines
- Easy to version-control infrastructure config
- Volumes persist data between container restarts

### Cons
- Not production-grade single host, no orchestration
- Docker Desktop required on Windows/Mac heavy application
- No auto-scaling or self-healing

### Alternatives

| Alternative | When to use instead |
|-------------|-------------------|
| **Kubernetes (k8s)** | Production, auto-scaling, self-healing, multi-node |
| **Minikube + Helm** | Local Kubernetes development |
| **Managed cloud DBs** | AWS RDS, GCP Cloud SQL no operational overhead |
| **Railway / Render** | Simple PaaS deployment, no Kubernetes complexity |
| **Podman** | Docker alternative, rootless containers, RHEL environments |

---

## Summary Table

| Component | Chosen | Best Free Alternative | Best Production Alternative |
|-----------|--------|----------------------|----------------------------|
| Web Framework | FastAPI | Flask | FastAPI (already good) |
| Frontend | Streamlit | Gradio | React + Next.js |
| Vector DB | pgvector | Chroma | Pinecone / Qdrant |
| Embeddings | MiniLM-L6 | all-mpnet-base-v2 | OpenAI text-embedding-3-small |
| Reranking | ms-marco-MiniLM | BGE Reranker | Cohere Rerank API |
| LLM | Groq (LLaMA 3.3) | Gemini Flash | OpenAI GPT-4o |
| Chunking | Recursive | Semantic Chunker | Unstructured.io |
| Observability | LangSmith | MLflow | Datadog LLM Observability |
| Caching | Redis | In-memory dict | GPTCache + Redis |
| Containers | Docker Compose | — | Kubernetes + Helm |
