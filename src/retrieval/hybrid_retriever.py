from typing import List, Optional
from langchain_core.documents import Document

from src.retrieval.vector_retriever import get_vector_retriever
from src.retrieval.keyword_retriever import get_keyword_retriever


def get_hybrid_retriever(
    query: str,
    *,
    vector_retriever=None,
    keyword_retriever=None,
) -> List[Document]:
    """
    Hybrid retrieval = vector retrieval + keyword retrieval.
    Retrievers can be injected (used for testing).
    """

    if vector_retriever is None:
        vector_retriever = get_vector_retriever()

    if keyword_retriever is None:
        keyword_retriever = get_keyword_retriever()

    vector_docs = vector_retriever.invoke(query)

    keyword_docs = []
    if keyword_retriever is not None:
        keyword_docs = keyword_retriever.invoke(query)

    combined = {doc.page_content: doc for doc in vector_docs + keyword_docs}
    return list(combined.values())