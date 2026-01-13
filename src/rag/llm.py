from pathlib import Path
import yaml

from langchain_ollama import ChatOllama


CONFIG_PATH = Path("configs/rag.yaml")


def load_llm():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    llm_config = config["llm"]

    return ChatOllama(
        model=llm_config["model"],
        temperature=llm_config["temperature"],
    )