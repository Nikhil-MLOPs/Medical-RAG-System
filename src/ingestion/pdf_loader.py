import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf_as_documents(pdf_path: Path) -> List[Document]:
    """
    Load a PDF using LangChain and return page-level Documents.
    Each page becomes a Document with strong provenance metadata.
    """
    logger.info(f"Loading PDF via LangChain: {pdf_path.name}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_pdf"] = pdf_path.name

    logger.info(f"Loaded {len(documents)} pages from {pdf_path.name}")
    return documents