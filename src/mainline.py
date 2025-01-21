from __future__ import annotations

from pathlib import Path

from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from PIL import Image

from src.base.layout import Layout
from src.tools.pdf_reader import PDFReader


class ChatChainModel:
    def __init__(
        self,
        document_path: Path,
        crop: tuple[int, int, int, int] | None = None,
        image_mode: bool = False,
    ):
        """
        A chat chain model that uses a layout to generate a conversation.

        :param document_path: The path to the document.
        :param crop: The crop for the document (for specific regions).
        :param image_mode: Select the mode of the document.
        """

        self.chain = self._build_chain()
        self.layout = self._build_document(document_path, crop, image_mode)

    def _build_document(
        self,
        document_path: Path,
        crop: tuple[int, int, int, int] | None = None,
        image_mode: bool = False,
    ) -> Layout:
        """
        Make layout of the image.

        :param document_path: The input image.
        :param crop: The crop for the document (for specific regions).
        :param image_mode: Select the mode of the document.
        :return: None
        """

        self.image: Image.Image | None = None
        if not image_mode:
            self.image = self._read_document(document_path)
        else:
            self.image = self._read_image(document_path)

        if crop is not None and self.image is not None:
            self.image = self.image.crop(crop)

        return Layout([])

    def _read_document(
        self,
        document_path: Path,
    ) -> Image.Image:
        """
        Read the document.

        :param document_path: The path to the document.
        :return: None
        """

        self.pdf_reader = PDFReader(document_path)
        return self.pdf_reader.as_single_image()

    def _read_image(
        self,
        document_path: Path,
    ) -> Image.Image:
        """
        Read the image.

        :param document_path: The path to the document.
        :return: None
        """

        return Image.open(document_path)

    def _build_chain(self) -> StateGraph:
        """
        Build the chain for the chat model.

        :return: The chain for the chat model.
        """

        chain = StateGraph()

        chain.add_state("start")
        chain.add_state("end")

        chain.add_edge(
            "start",
            "end",
            PromptTemplate("Ask for information about document", layout=self.layout),
        )

        return chain

    def handle_prompt(self, prompt: str) -> str:
        """
        Handle the prompt and return the response.

        :param prompt: The prompt to handle.
        :return: The response.
        """

        result = self.chain.generate_conversation()

        assert isinstance(result, str)

        return result


class ExtractionChainModel:
    def __init__(
        self,
        layout: Layout,
    ):
        """
        An extraction chain model that uses a layout to extract information from the document.

        :param layout: The layout of the document.
        """

        self.chain = self._build_chain()
        self.layout = layout

    def _build_chain(self) -> StateGraph:
        """
        Build the chain for the extraction model.

        :return: The chain for the extraction model.
        """

        chain = StateGraph()

        chain.add_state("start")
        chain.add_state("end")

        chain.add_edge(
            "start",
            "end",
            PromptTemplate("Ask for information about document", layout=self.layout),
        )

        return chain

    def extract_information(self) -> str:
        """
        Extract information based on the layout.

        :return: The extracted information.
        """

        result = self.chain.generate_conversation()

        assert isinstance(result, str)

        return result
