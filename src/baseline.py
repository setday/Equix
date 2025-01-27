from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.tools.models.vllm_model import global_vllm_model
from src.tools.pdf_reader import PDFReader


class Baseline:
    def __init__(
        self,
        document_path: Path,
        crop: tuple[int, int, int, int] | None = None,
        image_mode: bool = False,
    ):
        """
        Initialize the Baseline class.

        :param document_path: The path to the document.
        :param crop: The crop for the document (for specific regions).
        :param image_mode:
        """

        self.document_path = document_path
        self.image_mode = image_mode
        self.image: Image.Image | None = None

        if not self.image_mode:
            self._read_document()
        else:
            self._read_image()

        if crop is not None and self.image is not None:
            self.image = self.image.crop(crop)

    def _read_document(self) -> None:
        """
        Read the document.

        :return: None
        """

        self.pdf_reader = PDFReader(self.document_path)
        self.image = self.pdf_reader.as_single_image()

    def _read_image(self) -> None:
        """
        Read the image.

        :return: None
        """

        self.image = Image.open(self.document_path)

    def handle_prompt(self, prompt: str) -> str:
        """
        Handle the prompt and return the response.

        :param prompt: The prompt to handle.
        :return: The response.
        """

        response = global_vllm_model.ask_for(prompt, self.image)
        return response
