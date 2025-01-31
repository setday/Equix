from __future__ import annotations

import json

import numpy as np

from src.config import config
from src.tools.models.layout_extractor import global_layout_extractor
from src.tools.pdf_reader import PDFReader


if __name__ == "__main__":
    markup_size = 100

    papers_dir = config.data_dir / "papers"
    papers_files = list(papers_dir.iterdir())

    markup_dir = config.data_dir / "markup"
    markup_dir.mkdir(parents=True, exist_ok=True)

    for i in range(markup_size):
        paper_file = np.random.choice(papers_files)
        reader = PDFReader(pdf_path=paper_file)

        layouts = []

        for j, image in enumerate(reader.images):
            image.save(markup_dir / f"image_{i}_{j}.png")

            layout = global_layout_extractor.make_layout(image)
            layouts.append(
                {
                    "image_path": f"image_{i}_{j}.png",
                    "layout": layout,
                },
            )

        with open(markup_dir / f"markup_{i}.json", "w") as f:
            json.dump(layouts, f)
