from __future__ import annotations

import click


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
def main(mode: str, pdf: str, baseline: bool) -> None:
    """
    Main function for the CLI.
    """

    if baseline:
        raise NotImplementedError("Baseline chatbot is not implemented yet.")

    if mode == "image":
        raise NotImplementedError("Image mode is not implemented yet.")


if __name__ == "__main__":
    main()
