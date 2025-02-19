from __future__ import annotations

from enum import Enum
from typing import Any

from PIL import Image


class LayoutBlockType(Enum):
    """
    An enumeration class that represents the type of a layout block.
    """

    UNKNOWN = 0

    CAPTION = 1
    FOOTNOTE = 2
    FORMULA = 3

    LIST_ITEM = 4

    PAGE_FOOTER = 5
    PAGE_HEADER = 6

    PICTURE = 7

    SECTION_HEADER = 8

    TABLE = 9
    TEXT = 10

    CHART = 100


class LayoutBlockSpecification(Enum):
    """
    An enumeration class that represents the specification of a layout block.
    """

    UNKNOWN = 0
    HEADER = 1
    FOOTER = 2
    ANNOTATION = 3
    CAPTION = 4
    PAGE_NUMBER = 5


def block_string_to_enum(
    block_string: str,
) -> tuple[LayoutBlockType, LayoutBlockSpecification]:
    """
    Convert the block string to a block enum.

    :param block_string: The block string.
    :return: The block enum.
    """

    match block_string:
        case "Table":
            return LayoutBlockType.TABLE, LayoutBlockSpecification.UNKNOWN
        case "Picture":
            return LayoutBlockType.PICTURE, LayoutBlockSpecification.UNKNOWN
        case "Text":
            return LayoutBlockType.TEXT, LayoutBlockSpecification.UNKNOWN
        case "Chart":
            return LayoutBlockType.CHART, LayoutBlockSpecification.UNKNOWN
        case "Header":
            return LayoutBlockType.UNKNOWN, LayoutBlockSpecification.HEADER
        case "Footer":
            return LayoutBlockType.UNKNOWN, LayoutBlockSpecification.FOOTER
        case "Annotation":
            return LayoutBlockType.UNKNOWN, LayoutBlockSpecification.ANNOTATION
        case "Caption":
            return LayoutBlockType.UNKNOWN, LayoutBlockSpecification.CAPTION
        case "Page Number":
            return LayoutBlockType.UNKNOWN, LayoutBlockSpecification.PAGE_NUMBER
        case _:
            return LayoutBlockType.UNKNOWN, LayoutBlockSpecification.UNKNOWN


class LayoutBlock:
    block_type: LayoutBlockType
    block_specification: LayoutBlockSpecification

    text_content: str | None
    byte_content: bytes | None

    anotation: str | None

    bbox: tuple[float, float, float, float]
    page_number: int

    def __init__(
        self,
        block_type: LayoutBlockType,
        block_specification: LayoutBlockSpecification,
        bbox: tuple[float, float, float, float],
        page_number: int,
        text_content: str | None = None,
        byte_content: bytes | None = None,
        anotation: str | None = None,
    ):
        """
        A layout block class that represents a block in a document layout.

        :param block_type: The type of the block.
        :param block_specification: The specification of the block.
        :param bbox: The bounding box of the block.
        :param page_number: The page number of the block.
        :param text_content: The text content (for text blocks or extracted text from images, tables, etc.).
        :param byte_content: The byte content (for images, tables, etc.).
        :param anotation: The anotation of the block (for images, tables, etc.).
        """

        self.block_type = block_type
        self.block_specification = block_specification

        self.bbox = bbox
        self.page_number = page_number

        self.text_content = text_content
        self.byte_content = byte_content

        self.anotation = anotation

    @staticmethod
    def from_dict(
        block_data: dict[str, Any],
    ) -> LayoutBlock:
        """
        A layout block class that represents a block in a document layout.

        :param block_data: The data of the block.
        :return: The layout block.
        """

        btype = block_data.get("block_type", None)

        if isinstance(btype, str):
            block_type, block_specification = block_string_to_enum(btype)
        else:
            block_type = LayoutBlockType(block_data["block_type"])
            block_specification = LayoutBlockSpecification(
                block_data.get("block_specification", 0),
            )

        return LayoutBlock(
            block_type=block_type,
            block_specification=block_specification,
            bbox=block_data.get("bbox", None),
            page_number=block_data.get("page_number", None),
            text_content=block_data.get("text_content", None),
            byte_content=block_data.get("byte_content", None),
            anotation=block_data.get("anotation", None),
        )

    def extract_image_block(self, pages: list[Image.Image]) -> Image.Image:
        """
        Extract the image block from the pages.

        :param pages: The pages of the document.
        :return: The image block.
        """

        x0, y0, x1, y1 = self.bbox
        page = pages[self.page_number]
        image_block = page.crop((x0, y0, x1, y1))

        return image_block

    def to_text(self) -> str:
        """
        Convert the layout block to text.

        :return: The text representation of the layout block.
        """

        if self.text_content:
            return self.text_content

        if self.anotation:
            return self.anotation

        return f"LayoutBlock(block_type={self.block_type}, block_specification={self.block_specification})"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the layout block to a dictionary.

        :return: The dictionary representation of the layout block.
        """

        return {
            "block_type": self.block_type.name,
            "block_specification": self.block_specification.name,
            "bbox": self.bbox,
            "page_number": self.page_number,
            "text_content": self.text_content,
            "byte_content": self.byte_content,
            "anotation": self.anotation,
        }

    def __repr__(self) -> str:
        return self.to_text()

    def __str__(self) -> str:
        return self.to_text()


class Layout:
    blocks: list[LayoutBlock]

    def __init__(
        self,
        blocks: list[LayoutBlock],
    ):
        """
        A layout class that represents a layout of a document.

        :param blocks: The blocks in the layout.
        """

        self.blocks = blocks

    @staticmethod
    def from_dict(
        layout_data: dict[str, list[dict[str, Any]]],
    ) -> Layout:
        """
        A layout class that represents a layout of a document.

        :param layout_data: The data of the layout.
        :return: The layout.
        """

        return Layout(
            blocks=[
                LayoutBlock.from_dict(block_data)
                for block_data in layout_data["blocks"]
            ],
        )

    def to_text(self) -> str:
        """
        Convert the layout to text.

        :return: The text representation of the layout.
        """

        return "\n".join([str(block) for block in self.blocks])

    def to_dict(self, graphics_only: bool = False) -> dict[str, Any]:
        """
        Convert the layout to a dictionary.

        :param graphics_only: A flag to include only graphics in the layout.
        :return: The dictionary representation of the layout.
        """

        return {
            "blocks": [
                block.to_dict()  # .update({"index": index})
                for index, block in enumerate(self.blocks)
                if not graphics_only
                or (
                    block.block_type == LayoutBlockType.PICTURE
                    or block.block_type == LayoutBlockType.TABLE
                    or block.block_type == LayoutBlockType.CHART
                )
            ],
        }

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> LayoutBlock:
        return self.blocks[index]

    def __repr__(self) -> str:
        return self.to_text()

    def __str__(self) -> str:
        return self.to_text()
