from pathlib import Path
import yaml

from langchain_core.prompts import ChatPromptTemplate


CONFIG_PATH = Path("configs/rag.yaml")


def load_rag_prompt() -> ChatPromptTemplate:
    config = yaml.safe_load(CONFIG_PATH.read_text())

    system_prompt = config["prompt"]["system"]
    human_prompt = config["prompt"]["human"]

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )