import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from llmops.monitoring.langsmith_tracing import setup_langsmith

# Must run before any LangChain/langsmith modules are imported by the routes,
# so that LANGSMITH_TRACING env vars are present when those modules initialize.
setup_langsmith()

from backend.api.routes import health, upload, query, agent_query  # noqa: E402

settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(
        "LangSmith tracing: %s (project: %s)",
        "active" if settings.langsmith_api_key else "disabled (no API key)",
        settings.langsmith_project,
    )
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Enterprise Financial Copilot",
    description="RAG-powered AI system for querying financial documents",
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
app.include_router(upload.router, tags=["Upload"], prefix="")
app.include_router(query.router, tags=["Query"], prefix="")
app.include_router(agent_query.router, tags=["Agent"], prefix="")
