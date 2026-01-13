import json
import logging
from pathlib import Path
import yaml

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
CONFIG_PATH = Path("configs/chunking.yaml")


def load_documents() -> list[Document]:
    documents = []
    for file in PROCESSED_DIR.glob("*.jsonl"):
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                documents.append(
                    Document(
                        page_content=record["page_content"],
                        metadata=record["metadata"],
                    )
                )
    return documents


def run():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=[config["separator"]],
    )

    documents = load_documents()
    logger.info(f"Loaded {len(documents)} documents")

    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    return chunks


if __name__ == "__main__":
    run()