import re
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def clean_document(doc: Document) -> Document:
    """
    Perform conservative cleaning on LangChain Document content
    while preserving medical semantics and metadata.
    """
    original_length = len(doc.page_content)

    text = doc.page_content
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page\s+\d+", "", text, flags=re.IGNORECASE)

    doc.page_content = text.strip()

    logger.debug(
        f"Cleaned page {doc.metadata.get('page')} "
        f"from {original_length} to {len(doc.page_content)} chars"
    )

    return doc