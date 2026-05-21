import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from llmops.monitoring.langsmith_tracing import setup_langsmith

setup_langsmith()

from backend.api.routes import health, upload, query, agent_query, metrics  # noqa: E402

settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} [{settings.app_env}]")
    logger.info(f"LLM provider: {settings.llm_provider}")
    logger.info(
        "LangSmith: %s",
        "active" if settings.langsmith_api_key else "disabled",
    )
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Enterprise Financial Copilot",
    description="RAG system for querying financial documents with cited answers",
    version="1.0.0",
    lifespan=lifespan,
    separate_input_output_schemas=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(metrics.router, tags=["Metrics"])
app.include_router(upload.router, tags=["Upload"])
app.include_router(query.router, tags=["Query"])
app.include_router(agent_query.router, tags=["Agent"])
