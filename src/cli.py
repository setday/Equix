from __future__ import annotations

from pathlib import Path
from typing import Callable

import click

from src.baseline import Baseline
from src.mainline import ChatChainModel


def run_chat(prompt_callback: Callable[[str], str]) -> None:
    """
    Run the chatbot in chat mode.

    :param prompt_callback: The callback function for the chatbot.
    :return: None
    """

    while True:
        prompt = input("You: ")
        if prompt == "exit":
            break

        response = prompt_callback(prompt)
        print(f"Bot: {response}")


@click.command()  # type: ignore
@click.option(
    "--mode",
    type=click.Choice(["chat", "image"]),
    default="chat",
    help="Mode of operation.",
)  # type: ignore
@click.option(
    "--document",
    type=click.Path(exists=True),
    required=True,
    help="Path to the PDF file or image.",
)  # type: ignore
@click.option(
    "--baseline",
    is_flag=True,
    default=False,
    help="Run the baseline chatbot.",
)  # type: ignore
@click.option(
    "--prompt",
    type=str,
    default=None,
    help="A single prompt mode.",
)  # type: ignore
@click.option(
    "--crop",
    type=(int, int, int, int),
    default=None,
    help="Crop the image.",
)  # type: ignore
def main(
    mode: str,
    document: str,
    baseline: bool,
    prompt: str | None,
    crop: tuple[int, int, int, int] | None,
) -> None:
    """
    Main function for the CLI.

    :param mode: The mode of operation.
    :param document: The document to process.
    :param baseline: The baseline flag.
    :param prompt: The prompt for the chatbot.
    :param crop: The crop for the image.
    :return: None
    """

    is_image_mode = mode == "image"

    path = Path(document)

    model_pipeline: Baseline | ChatChainModel | None = None
    if baseline:
        model_pipeline = Baseline(path, image_mode=is_image_mode, crop=crop)
    else:
        model_pipeline = ChatChainModel(path, image_mode=is_image_mode, crop=crop)

    if prompt is not None:
        response = model_pipeline.handle_prompt(prompt)
        print(f"Bot: {response}")
    else:
        run_chat(model_pipeline.handle_prompt)


if __name__ == "__main__":
    main()
