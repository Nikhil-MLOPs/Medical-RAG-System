import logging
from pathlib import Path
import yaml

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/retrieval.yaml")
EMBED_CONFIG_PATH = Path("configs/embeddings.yaml")


def get_vector_retriever():
    retrieval_config = yaml.safe_load(CONFIG_PATH.read_text())
    embed_config = yaml.safe_load(EMBED_CONFIG_PATH.read_text())

    embeddings = OllamaEmbeddings(model=embed_config["model"])

    vectorstore = Chroma(
        persist_directory=embed_config["persist_directory"],
        embedding_function=embeddings,
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": retrieval_config["vector_k"]}
    )