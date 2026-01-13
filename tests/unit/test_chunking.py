from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def test_chunking_produces_documents():
    docs = [
        Document(page_content="This is a medical sentence.", metadata={"page": 1})
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
    chunks = splitter.split_documents(docs)

    assert len(chunks) > 0
    assert isinstance(chunks[0], Document)