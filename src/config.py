from __future__ import annotations

from pathlib import Path

import dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class BaseConfig(BaseSettings):  # type: ignore
    """
    A base config class for your DS project that:
      - Reads environment variables from .env
      - Allows overriding any field via environment variables or code
    """

    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    project_dir: Path = Field(
        default=Path(__file__).resolve().parents[1],
        env="PROJECT_DIR",
    )
    data_dir: Path = Field(
        default=Path(__file__).resolve().parents[1] / Path("data"),
        env="DATA_DIR",
    )
    model_dir: Path = Field(
        default=Path(__file__).resolve().parents[1] / Path("models"),
        env="MODEL_DIR",
    )

    env_file: str | None = Field(default=".env", env="ENV_FILE")

    random_seed: int = Field(default=42, env="RANDOM_SEED")

    # Models config

    # deepseek-ai/deepseek-vl2-small will also be tested
    baseline_model: str = Field(
        default="deepseek-ai/deepseek-vl2-small",
        env="BASELINE_MODEL",
    )

    layout_detection_model: str = Field(
        default="cmarkea/detr-layout-detection",
        env="LAYOUT_DETECTION_MODEL",
    )
    # deepseek-ai/deepseek-vl2-tiny will also be tested
    picture_information_extraction_model: str = Field(
        default="mPLUG/DocOwl1.5",
        env="PICTURE_INFORMATION_EXTRACTION_MODEL",
    )
    # Qwen/Qwen2.5-1.5B / deepseek-ai/deepseek-llm-7b-chat /
    # deepseek-ai/DeepSeek-V2-Lite-Chat will also be tested
    llm_model: str = Field(default="meta-llama/Llama-3.2-3B", env="LLM_MODEL")

    # deepset/sentence_bert is also suitable for this task
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL",
    )

    # Data config

    chart_qa_dataset: str = Field(
        default="ahmed-masry/ChartQA",
        env="CHART_QA_DATASET",
    )
    table_extraction_dataset: str = Field(
        default="bsmock/pubtables-1m",
        env="TABLE_EXTRACTION_DATASET",
    )
    document_qa_generator: str = Field(
        default="ds4sd/DocLayNet",
        env="DOCUMENT_QA_GENERATOR",
    )
    papers_dataset: str = Field(
        default="Cornell-University/arxiv",
        env="PAPERS_DATASET",
    )

    # Hugging Face credentials

    huggingface_token: str | None = Field(
        default=None,
        env="HUGGINGFACE_TOKEN",
    )


config = BaseConfig()
