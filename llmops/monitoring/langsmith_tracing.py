import os
import logging
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def setup_langsmith():
    if not settings.langsmith_api_key:
        logger.warning("LANGSMITH_API_KEY not set - tracing disabled")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    # Also set LangChain variants
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    logger.info(f"LangSmith tracing enabled - project: {settings.langsmith_project}")
    return True
