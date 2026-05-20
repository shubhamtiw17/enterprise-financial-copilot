import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.api.routes import health, upload, query

settings = get_settings()

# This controls what you see in your terminal logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything before yield runs on STARTUP
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    yield
    # Everything after yield runs on SHUTDOWN
    logger.info("Shutting down...")


app = FastAPI(
    title="Enterprise Financial Copilot",
    description="RAG-powered AI system for querying financial documents",
    version="1.0.0",
    lifespan=lifespan,
    separate_input_output_schemas=False,
)

# CORS tells your API which frontends are allowed to talk to it
# Without this, your Streamlit app gets blocked by the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router, tags=["Health"])
app.include_router(upload.router, tags=["Upload"], prefix="")
app.include_router(query.router, tags=["Query"], prefix="")