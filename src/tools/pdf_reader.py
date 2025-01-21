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

        images = []
        with pdf2image.create(self.pdf_path) as pdf:
            for _i, page in enumerate(pdf.pages):
                image = page.to_pil()
                images.append(image)
        return images

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
