from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import config

# import matplotlib.pyplot as plt
# import seaborn as sns


def generate_randowm_plot(
    functions: list[str],
    description: str,
    anotation: str,
    output_path: Path,
) -> None:
    """
    Generate a random plot.

    :param function: The function to plot.
    :param description: The description of the plot.
    :param anotation: The anotation of the plot.
    :param output_path: The path to save the plot.
    """

    raise NotImplementedError("This function is not implemented yet.")


if __name__ == "__main__":
    generation_size = 100
    generation_dir = config.data_dir / "generation"

    functions_file = Path(__file__).resolve() / ".." / "plot_functions.txt"
    descriptions_file = Path(__file__).resolve() / ".." / "plot_descriptions.txt"
    anotation_file = Path(__file__).resolve() / ".." / "plot_anotations.txt"

    with open(functions_file) as f:
        functions = f.readlines()

    with open(descriptions_file) as f:
        descriptions = f.readlines()

    with open(anotation_file) as f:
        anotations = f.readlines()

    for i in range(generation_size):
        functions_count = np.random.randint(1, 4)

        selected_functions = np.random.choice(functions, functions_count)
        selected_descriptions = np.random.choice(descriptions)
        selected_anotations = np.random.choice(anotations)

        output_path = generation_dir / f"plot_{i}.png"

        generate_randowm_plot(
            functions=selected_functions,
            description=selected_descriptions,
            anotation=selected_anotations,
            output_path=output_path,
        )
