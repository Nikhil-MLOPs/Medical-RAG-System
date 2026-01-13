from typing import Dict, List

from langchain_core.documents import Document

from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.rag.prompt import load_rag_prompt
from src.rag.llm import load_llm


def format_context(documents: List[Document]) -> str:
    """
    Convert retrieved documents into a single context string
    with citations.
    """
    context_blocks = []

    for doc in documents:
        source = doc.metadata.get("source_pdf", "unknown")
        page = doc.metadata.get("page", "unknown")

        block = f"[Source: {source}, Page: {page}]\n{doc.page_content}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def run_rag(question: str) -> Dict[str, str]:
    # 1. Retrieve documents
    documents = get_hybrid_retriever(question)

    # 2. Format context
    context = format_context(documents)

    # 3. Load prompt + LLM
    prompt = load_rag_prompt()
    llm = load_llm()

    # 4. Create chain
    chain = prompt | llm

    # 5. Run chain
    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "answer": response.content,
        "context": context,
    }