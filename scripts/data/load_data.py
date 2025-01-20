from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset
from loguru import logger

from src.config import config


def download_chartqa_datasets(
    output_dir: Path = config.data_dir / "chartQA",
) -> None:
    """
    Download the `chartQA` datasets to the output_dir.

    The `chartQA` dataset is a collection of 28k+ question-answer pairs for 14k+ charts.

    :param output_dir: The directory where the datasets are saved.
    :return: None
    """

    logger.info(
        f"Downloading the `chartQA` datasets... (data will be saved to {output_dir})",
    )

    # Check if the dataset is already downloaded
    if output_dir.exists():
        logger.warning(
            f"Found possible existing dataset at {output_dir}. Skipping download.",
        )
        return

    # Create the output_dir if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download the `chartQA` datasets
    dataset = load_dataset("ahmed-masry/ChartQA")

    # Save the datasets to the output_dir
    dataset.save_to_disk(output_dir)


def dowload_pubtables_datasets(
    output_dir: Path = config.data_dir / "pubtables",
) -> None:
    """
    Download the `pubtables` datasets to the output_dir.

    The `pubtables` dataset is a collection of 1.5k+ tables from scientific papers.

    :param output_dir: The directory where the datasets are saved.
    :return: None
    """

    logger.info(
        "Downloading the `pubtables` datasets... (data will be saved to {output_dir})",
    )
    logger.info(
        "!!!ATTENTION!!! You should have lfs module installed to download the `pubtables` datasets.",
    )

    # Check if the dataset is already downloaded
    if output_dir.exists():
        logger.warning(
            f"Found possible existing dataset at {output_dir}. Skipping download.",
        )
        return

    # Create the output_dir if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download the `pubtables` datasets
    os.system(
        f"cd {output_dir} && curl -LO"
        "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-PDF_Annotations.tar.gz",
    )


def dowload_doclaynet_datasets(
    output_dir: Path = config.data_dir / "layoutset",
) -> None:
    """
    Download the `doclaynet` datasets to the output_dir.

    The `doclaynet` dataset is a collection of 1.5k+ tables from scientific papers.

    :param output_dir: The directory where the datasets are saved.
    :return: None
    """

    logger.info(
        "Downloading the `doclaynet` datasets... (data will be saved to {output_dir})",
    )

    # Check if the dataset is already downloaded
    if output_dir.exists():
        logger.warning(
            f"Found possible existing dataset at {output_dir}. Skipping download.",
        )
        return

    # Create the output_dir if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download the `dcolaynet` datasets
    dataset = load_dataset("ds4sd/DocLayNet", trust_remote_code=True)

    # Save the datasets to the output_dir
    dataset.save_to_disk(output_dir)


def download_datasets(
    output_dir: Path = config.data_dir,
) -> None:
    """
    Download datasets from the internet and save them to the output_dir.

    Checkout the `EDA` block for more details on the datasets.

    :param output_dir: The directory where the datasets are saved.
    :return: None
    """
    # Create the output_dir if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download the `chartQA` datasets
    download_chartqa_datasets(output_dir / "chartQA")

    # Download the `pubtables` datasets
    dowload_pubtables_datasets(output_dir / "pubtables")

    # Download the `doclaynet` datasets
    dowload_doclaynet_datasets(output_dir / "layoutset")


if __name__ == "__main__":
    download_datasets()
