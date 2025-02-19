from __future__ import annotations

from pathlib import Path
from typing import Callable

import click


def run_chat(prompt_callback: Callable[[str], str]) -> None:
    """
    Run the chatbot in chat mode.

    :param prompt_callback: The callback function for the chatbot.
    :return: None
    """

    while True:
        prompt = input()
        if prompt == "exit":
            break

        response = prompt_callback(prompt)
        print(response)


@click.command()  # type: ignore
@click.option(
    "--document_mode",
    type=click.Choice(["document", "image"]),
    default="document",
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
    "--extract_layout",
    is_flag=True,
    default=False,
    help="Run the layout chatbot.",
)  # type: ignore
@click.option(
    "--prompt",
    type=str,
    default=None,
    help="A single prompt mode.",
)  # type: ignore
@click.option(
    "--crop",
    type=(int, int, int, int, int),
    default=None,
    help="Crop the image.",
)  # type: ignore
def main(
    document_mode: str,
    document: str,
    baseline: bool,
    extract_layout: bool,
    prompt: str | None,
    crop: tuple[int, int, int, int, int] | None,
) -> None:
    """
    Main function for the CLI.

    :param mode: The mode of operation.
    :param document: The document to process.
    :param baseline: The baseline flag.
    :param prompt: The prompt for the chatbot.
    :param crop: The crop for the image ([page_number, x0, y0, x1, y1]).
    :return: None
    """

    assert not (
        baseline and extract_layout
    ), "There is no layout block in the baseline."

    is_image_mode = document_mode == "image"

    path = Path(document)

    model_pipeline: Baseline | ChatChainModel | None = None
    if baseline:
        from src.baseline import Baseline

        model_pipeline = Baseline(path, image_mode=is_image_mode, crop=crop)
    else:
        from src.mainline import ChatChainModel

        model_pipeline = ChatChainModel(path, image_mode=is_image_mode, crop=crop)

    if extract_layout:
        layout = model_pipeline.get_layout()
        dict_layout = layout.to_dict(graphics_only=True)
        print(dict_layout)
        return

    if prompt is not None:
        response = model_pipeline.handle_prompt(prompt)
        print(response)
    else:
        run_chat(model_pipeline.handle_prompt)


if __name__ == "__main__":
    main()
