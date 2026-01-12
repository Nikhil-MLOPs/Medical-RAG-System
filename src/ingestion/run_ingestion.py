import json
import logging
from pathlib import Path

from langchain_core.documents import Document

from src.core.logging_config import setup_logging
from src.ingestion.pdf_loader import load_pdf_as_documents
from src.ingestion.document_cleaner import clean_document

setup_logging()
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)


def serialize_document(doc: Document) -> dict:
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    }


def run():
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError("No PDF files found in data/raw")

    for pdf_path in pdf_files:
        logger.info(f"Starting ingestion for {pdf_path.name}")
        documents = load_pdf_as_documents(pdf_path)

        output_file = PROCESSED_DIR / f"{pdf_path.stem}.jsonl"

        with output_file.open("w", encoding="utf-8") as f:
            for doc in documents:
                cleaned_doc = clean_document(doc)
                if not cleaned_doc.page_content:
                    continue
                f.write(json.dumps(serialize_document(cleaned_doc)) + "\n")

        logger.info(f"Saved processed LangChain docs to {output_file}")


if __name__ == "__main__":
    run()