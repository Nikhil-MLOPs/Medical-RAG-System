from langchain_core.documents import Document
from src.retrieval.hybrid_retriever import get_hybrid_retriever


class FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query):
        return self._docs


def test_hybrid_retriever_returns_list():
    fake_vector_docs = [
        Document(page_content="Diabetes treatment includes insulin.", metadata={"page": 1})
    ]

    fake_keyword_docs = [
        Document(page_content="Insulin dosage depends on patient.", metadata={"page": 2})
    ]

    fake_vector_retriever = FakeRetriever(fake_vector_docs)
    fake_keyword_retriever = FakeRetriever(fake_keyword_docs)

    docs = get_hybrid_retriever(
        "diabetes treatment",
        vector_retriever=fake_vector_retriever,
        keyword_retriever=fake_keyword_retriever,
    )

    assert isinstance(docs, list)
    assert len(docs) == 2