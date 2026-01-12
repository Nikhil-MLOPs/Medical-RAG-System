from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from src.core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Medical RAG System starting up")
    yield
    logger.info("Medical RAG System shutting down")


app = FastAPI(
    title="Medical RAG System",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}