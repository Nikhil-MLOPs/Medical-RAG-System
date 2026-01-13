import json
from pathlib import Path
import yaml
import logging

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/retrieval.yaml")
PROCESSED_DIR = Path("data/processed")


def load_documents():
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


def get_keyword_retriever():
    retrieval_config = yaml.safe_load(CONFIG_PATH.read_text())
    documents = load_documents()

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = retrieval_config["keyword_k"]

    return retriever