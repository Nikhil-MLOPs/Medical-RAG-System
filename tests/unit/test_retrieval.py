from src.retrieval.hybrid_retriever import get_hybrid_retriever


def test_hybrid_retriever_returns_list():
    docs = get_hybrid_retriever("diabetes treatment")
    assert isinstance(docs, list)