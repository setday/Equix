from __future__ import annotations

from enum import Enum


class LayoutBlockType(Enum):
    """
    An enumeration class that represents the type of a layout block.
    """

    UNKNOWN = 0
    TEXT = 1
    IMAGE = 2
    TABLE = 3
    CHART = 4


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

    def to_text(self) -> str:
        """
        Convert the layout to text.

        :return: The text representation of the layout.
        """

        return "\n".join([str(block) for block in self.blocks])

    def __repr__(self) -> str:
        return self.to_text()

    def __str__(self) -> str:
        return self.to_text()
