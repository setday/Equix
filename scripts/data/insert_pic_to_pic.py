from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.config import config


def insert_pic_to_pic(
    background_pic: Image.Image,
    foreground_pic: Image.Image,
) -> Image.Image:
    """
    Insert the foreground_pic into the background_pic.

    :param background_pic: The background picture.
    :param foreground_pic: The foreground picture.
    :return: The combined picture.
    """

    x = np.random.randint(0, background_pic.width - foreground_pic.width)
    y = np.random.randint(0, background_pic.height - foreground_pic.height)

    background_pic.paste(foreground_pic, (x, y))

    return background_pic


def mix_images(
    background_pic_path: Path,
    foreground_pic_path: Path,
    output_path: Path,
) -> None:
    """
    Mix the foreground_pic into the background_pic.

    :param background_pic_path: The path to the background picture.
    :param foreground_pic_path: The path to the foreground picture.
    :return: The combined picture.
    """

    background_pic = Image.open(background_pic_path)
    foreground_pic = Image.open(foreground_pic_path)

    combined_pic = insert_pic_to_pic(
        background_pic=background_pic,
        foreground_pic=foreground_pic,
    )

    combined_pic.save(output_path)


if __name__ == "__main__":
    generation_size = 100
    generation_dir = config.data_dir / "generation"

    papers_dir = config.data_dir / "papers"
    plots_dir = config.data_dir / "plots"

    paper_files = list(papers_dir.iterdir())
    plot_files = list(plots_dir.iterdir())

    for i in range(generation_size):
        paper_file = np.random.choice(paper_files)
        plot_file = np.random.choice(plot_files)

        mix_images(
            background_pic_path=paper_file,
            foreground_pic_path=plot_file,
            output_path=generation_dir / f"generated_{i}.png",
        )
