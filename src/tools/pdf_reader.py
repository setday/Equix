from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pdf2image
from PIL import Image


class PDFReader:
    def __init__(self, pdf_path: Path) -> None:
        """
        Initialize the PDFReader class.

        :param pdf_path: The path to the PDF file.
        :return: None
        """

        assert pdf_path.exists(), f"PDF file not found at {pdf_path}"

        pdf_data = None

        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        self.pdf_data = pdf_data
        self.pdf_path = pdf_path
        self.images = self._extract_images()

    def _extract_images(self) -> list[Image.Image]:
        """
        Extract images from the PDF file.

        :return: A list of images.
        """

        return pdf2image.convert_from_path(self.pdf_path)  # type: ignore

    def as_single_image(self) -> Image.Image:
        """
        Convert the images to a single image.

        :return: A single image.
        """

        images = self.images
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_image = Image.new("RGB", (total_width, max_height))

        x_offset = 0
        for image in images:
            new_image.paste(image, (x_offset, 0))
            x_offset += image.size[0]

        return new_image

    def save_images(self, output_dir: Path) -> None:
        """
        Save the images to the output_dir.

        :param output_dir: The directory where the images are saved.
        :return: None
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(self.images):
            image.save(output_dir / f"image_{i}.png")

    def __repr__(self) -> str:
        return f"PDFReader(pdf_path={self.pdf_path})"

    def __str__(self) -> str:
        return f"PDFReader(pdf_path={self.pdf_path})"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Image.Image:
        return self.images[index]

    def __iter__(self) -> Iterator[Image.Image]:
        return iter(self.images)
