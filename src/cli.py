from __future__ import annotations

from pathlib import Path
from typing import Callable

import click

from src.baseline import Baseline


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
    "--pdf",
    type=click.Path(exists=True),
    required=True,
    help="Path to the PDF file.",
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
def main(mode: str, pdf: str, baseline: bool, prompt: str | None) -> None:
    """
    Main function for the CLI.
    """

    if not baseline:
        raise NotImplementedError("Baseline chatbot is not implemented yet.")

    if mode == "image":
        raise NotImplementedError("Image mode is not implemented yet.")

    if prompt is not None:
        raise NotImplementedError("Single prompt mode is not implemented yet.")

    path = Path(pdf)

    baseline_model = Baseline(path)
    run_chat(baseline_model.handle_prompt)


if __name__ == "__main__":
    main()
