from __future__ import annotations

from pathlib import Path

from huggingface_hub import login
from loguru import logger
from transformers import AutoImageProcessor
from transformers import AutoModel
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers.models.detr import DetrForSegmentation

from src.config import config


def load_huggingface_llm(model_name: str, output_dir: Path) -> None:
    """
    Load a Hugging Face language model to the output_dir.

    :param model_name: The name of the model to load.
    :param output_dir: The directory where the model is saved.
    :return: None
    """

    logger.info(
        f"Loading the Hugging Face language model `{model_name}`... (model will be saved to {output_dir})",
    )

    # Check if the model is already downloaded
    if output_dir.exists():
        logger.warning(
            f"Found possible existing model at {output_dir}. Skipping download.",
        )
        return

    # Create the output_dir if it doesn't exist
    (output_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    (output_dir / "model").mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    # Load model
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.save_pretrained(output_dir / "model")


def load_huggingface_vision_model(model_name: str, output_dir: Path) -> None:
    """
    Load a Hugging Face vision model to the output_dir.

    :param model_name: The name of the model to load.
    :param output_dir: The directory where the model is saved.
    :return: None
    """

    logger.info(
        f"Loading the Hugging Face vision model `{model_name}`... (model will be saved to {output_dir})",
    )

    # Check if the model is already downloaded
    if output_dir.exists():
        logger.warning(
            f"Found possible existing model at {output_dir}. Skipping download.",
        )
        return

    # Create the output_dir if it doesn't exist
    (output_dir / "processor").mkdir(parents=True, exist_ok=True)
    (output_dir / "model").mkdir(parents=True, exist_ok=True)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)
    processor.save_pretrained(output_dir / "processor")

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(model_name)
    model.save_pretrained(output_dir / "model")


def load_huggingface_automodel(model_name: str, output_dir: Path) -> None:
    """
    Load a Hugging Face AutoModel to the output_dir.

    :param model_name: The name of the model to load.
    :param output_dir: The directory where the model is saved.
    :return: None
    """

    logger.info(
        f"Loading the Hugging Face AutoModel `{model_name}`... (model will be saved to {output_dir})",
    )

    # Check if the model is already downloaded
    if output_dir.exists():
        logger.warning(
            f"Found possible existing model at {output_dir}. Skipping download.",
        )
        return

    # Create the output_dir if it doesn't exist
    (output_dir / "model").mkdir(parents=True, exist_ok=True)

    # Load model
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.save_pretrained(output_dir / "model")


def load_huggingface_detr(model_name: str, output_dir: Path) -> None:
    """
    Load a Hugging Face DETR model to the output_dir.

    :param model_name: The name of the model to load.
    :param output_dir: The directory where the model is saved.
    :return: None
    """

    logger.info(
        f"Loading the Hugging Face DETR model `{model_name}`... (model will be saved to {output_dir})",
    )

    # Check if the model is already downloaded
    if output_dir.exists():
        logger.warning(
            f"Found possible existing model at {output_dir}. Skipping download.",
        )
        return

    # Create the output_dir if it doesn't exist
    (output_dir / "processor").mkdir(parents=True, exist_ok=True)
    (output_dir / "model").mkdir(parents=True, exist_ok=True)

    # Load processor
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    processor.save_pretrained(output_dir / "processor")

    # Load model
    model = DetrForSegmentation.from_pretrained(model_name)
    model.save_pretrained(output_dir / "model")


def download_models() -> None:
    """
    Download the models specified in the config.

    :return: None
    """

    # Log in to Hugging before downloading the models
    login(token=config.huggingface_token)

    # Load the baseline model
    load_huggingface_vision_model(
        model_name=config.baseline_model,
        output_dir=config.model_dir / "baseline_model",
    )

    # Load the layout detection model
    load_huggingface_detr(
        model_name=config.layout_detection_model,
        output_dir=config.model_dir / "layout_detection_model",
    )

    # Load the picture information extraction model
    load_huggingface_llm(
        model_name=config.picture_information_extraction_model,
        output_dir=config.model_dir / "picture_information_extraction_model",
    )

    # Load the LLM model
    load_huggingface_llm(
        model_name=config.llm_model,
        output_dir=config.model_dir / "llm_model",
    )

    # Load the embedding model
    load_huggingface_llm(
        model_name=config.embedding_model,
        output_dir=config.model_dir / "embedding_model",
    )


if __name__ == "__main__":
    download_models()
