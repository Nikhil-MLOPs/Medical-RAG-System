import logging
from pathlib import Path
import yaml
from math import ceil

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from src.experiments.chunk_documents import run as chunk_run
from src.core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/embeddings.yaml")


def batch_documents(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i : i + batch_size]


def run():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    batch_size = config.get("batch_size", 32)

    logger.info("Starting chunking step")
    chunks = chunk_run()
    total_batches = ceil(len(chunks) / batch_size)

    logger.info("Initializing Ollama embeddings")
    embeddings = OllamaEmbeddings(model=config["model"])

    logger.info("Initializing / loading Chroma vector store")
    vectorstore = Chroma(
        persist_directory=config["persist_directory"],
        embedding_function=embeddings,
    )

    for idx, batch in enumerate(batch_documents(chunks, batch_size), start=1):
        logger.info(f"Embedding batch {idx}/{total_batches} (size={len(batch)})")
        vectorstore.add_documents(batch)

    logger.info("All embeddings stored successfully (incremental)")


if __name__ == "__main__":
    run()