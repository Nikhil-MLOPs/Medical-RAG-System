from typing import List
from langchain_core.documents import Document

from src.retrieval.vector_retriever import get_vector_retriever
from src.retrieval.keyword_retriever import get_keyword_retriever


def get_hybrid_retriever(query: str) -> List[Document]:
    """
    Hybrid retrieval = vector retrieval + keyword retrieval,
    merged and deduplicated.
    """
    vector_retriever = get_vector_retriever()
    keyword_retriever = get_keyword_retriever()

    # LangChain 1.x standard invocation
    vector_docs = vector_retriever.invoke(query)
    keyword_docs = keyword_retriever.invoke(query)

    # Deduplicate using page_content as key
    combined = {doc.page_content: doc for doc in vector_docs + keyword_docs}

    return list(combined.values())