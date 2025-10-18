from dataclasses import dataclass
from typing import Final
import os

from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())


@dataclass(frozen=True)
class ConfigLLM:
    name: Final[str] = os.environ.get("NAME_LLM", default="vis-mistralai/mistral-small-3.1-24b-instruct")
    token: Final[str] = os.environ.get("LLM_TOKEN")
    base_url: Final[str] = os.environ.get("LLM_BASE_URL", default="https://api.vsegpt.ru/v1")


@dataclass(frozen=True)
class ConfigVectorModel:
    name: Final[str] = os.environ.get(
        "EMBEDDING_NAME_VECTOR_MODEL",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    embedding_type: Final[str] = os.environ.get("EMBEDDING_TYPE", default="huggingface")
    device: Final[str] = os.environ.get("EMBEDDING_DEVICE", default="cuda")
