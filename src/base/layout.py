from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from PIL import Image


class LayoutBlockType(Enum):
    """
    An enumeration class that represents the type of a layout block.
    """

    UNKNOWN = -1

    CAPTION = 0
    FOOTNOTE = 1
    FORMULA = 2

    LIST_ITEM = 3

    PAGE_FOOTER = 4
    PAGE_HEADER = 5

    PICTURE = 6

    SECTION_HEADER = 7

    TABLE = 8
    TEXT = 9

    TITLE = 10

    CHART = 11


class LayoutBlockSpecification(Enum):
    """
    An enumeration class that represents the specification of a layout block.
    """

    UNKNOWN = -1
    HEADER = 0
    FOOTER = 1
    ANNOTATION = 2
    CAPTION = 3
    PAGE_NUMBER = 4


def block_string_to_enum(
    block_string: str,
) -> tuple[LayoutBlockType, LayoutBlockSpecification]:
    """
    Convert the block string to a block enum.

    :param block_string: The block string.
    :return: The block enum.
    """
    # Define mappings to reduce complexity
    block_type_mappings = {
        "Table": LayoutBlockType.TABLE,
        "TABLE": LayoutBlockType.TABLE,
        "Picture": LayoutBlockType.PICTURE,
        "Image": LayoutBlockType.PICTURE,
        "PICTURE": LayoutBlockType.PICTURE,
        "Text": LayoutBlockType.TEXT,
        "TEXT": LayoutBlockType.TEXT,
        "Chart": LayoutBlockType.CHART,
        "CHART": LayoutBlockType.CHART,
        "List Item": LayoutBlockType.LIST_ITEM,
        "LIST_ITEM": LayoutBlockType.LIST_ITEM,
        "Formula": LayoutBlockType.FORMULA,
        "FORMULA": LayoutBlockType.FORMULA,
        "Footnote": LayoutBlockType.FOOTNOTE,
        "FOOTNOTE": LayoutBlockType.FOOTNOTE,
        "Section Header": LayoutBlockType.SECTION_HEADER,
        "SECTION_HEADER": LayoutBlockType.SECTION_HEADER,
        "Page Footer": LayoutBlockType.PAGE_FOOTER,
        "PAGE_FOOTER": LayoutBlockType.PAGE_FOOTER,
        "Page Header": LayoutBlockType.PAGE_HEADER,
        "PAGE_HEADER": LayoutBlockType.PAGE_HEADER,
        "Caption": LayoutBlockType.CAPTION,
        "CAPTION": LayoutBlockType.CAPTION,
    }

    specification_mappings = {
        "Header": LayoutBlockSpecification.HEADER,
        "HEADER": LayoutBlockSpecification.HEADER,
        "Footer": LayoutBlockSpecification.FOOTER,
        "FOOTER": LayoutBlockSpecification.FOOTER,
        "Annotation": LayoutBlockSpecification.ANNOTATION,
        "ANNOTATION": LayoutBlockSpecification.ANNOTATION,
        "Caption": LayoutBlockSpecification.CAPTION,
        "CAPTION": LayoutBlockSpecification.CAPTION,
        "Page Number": LayoutBlockSpecification.PAGE_NUMBER,
        "PAGE_NUMBER": LayoutBlockSpecification.PAGE_NUMBER,
    }

    # Check block type mappings first
    if block_string in block_type_mappings:
        return block_type_mappings[block_string], LayoutBlockSpecification.UNKNOWN

    # Check specification mappings
    if block_string in specification_mappings:
        return LayoutBlockType.UNKNOWN, specification_mappings[block_string]

    # Default case
    return LayoutBlockType.UNKNOWN, LayoutBlockSpecification.UNKNOWN


@dataclass
class LayoutBlockBoundingBox:
    """
    A layout block bounding box class that represents the bounding box of a block.

    :param x: The x-coordinate of the top-left corner of the bounding box.
    :param y: The y-coordinate of the top-left corner of the bounding box.
    :param width: The width of the bounding box.
    :param height: The height of the bounding box.
    """

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        """
        Post-initialization method to ensure the bounding box is valid.
        """

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive values.")

    @staticmethod
    def from_points(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> LayoutBlockBoundingBox:
        """
        Create a LayoutBlockBoundingBox from points.

        :param x0: The x-coordinate of the top-left corner of the bounding box.
        :param y0: The y-coordinate of the top-left corner of the bounding box.
        :param x1: The x-coordinate of the bottom-right corner of the bounding box.
        :param y1: The y-coordinate of the bottom-right corner of the bounding box.
        :return: The layout block bounding box.
        """

        return LayoutBlockBoundingBox(
            x=x0,
            y=y0,
            width=x1 - x0,
            height=y1 - y0,
        )

    def to_points(self) -> tuple[float, float, float, float]:
        """
        Convert the layout block bounding box to points.

        :return: The points of the bounding box.
        """

        return self.x, self.y, self.x + self.width, self.y + self.height

    @staticmethod
    def from_dict(
        bounding_box_data: dict[str, float],
    ) -> LayoutBlockBoundingBox:
        """
        Create a LayoutBlockBoundingBox from a dictionary.

        :param bounding_box_data: The data of the bounding box.
        :return: The layout block bounding box.
        """

        return LayoutBlockBoundingBox(
            x=bounding_box_data["x"],
            y=bounding_box_data["y"],
            width=bounding_box_data["width"],
            height=bounding_box_data["height"],
        )

    def to_dict(self) -> dict[str, float]:
        """
        Convert the layout block bounding box to a dictionary.

        :return: The dictionary representation of the layout block bounding box.
        """

        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    def __iter__(self) -> tuple[float, float, float, float]:
        """
        Iterate over the bounding box coordinates.

        :return: The coordinates of the bounding box.
        """

        return self.x, self.y, self.x + self.width, self.y + self.height

    def __repr__(self) -> str:
        return f"(x={self.x}, y={self.y}, width={self.width}, height={self.height})"

    def __str__(self) -> str:
        return f"(x={self.x}, y={self.y}, width={self.width}, height={self.height})"


@dataclass
class LayoutBlock:
    """
    A layout block class that represents a block in a document layout.

    :param id: The ID of the block.
    :param type: The type of the block.
    :param specification: The specification of the block.
    :param bounding_box: The bounding box of the block.
    :param page_number: The page number of the block.
    :param text_content: The text content (for text blocks or extracted text from images, tables, etc.).
    :param byte_content: The byte content (for images, tables, etc.).
    :param anotation: The anotation of the block (for images, tables, etc.).
    :param confidence: The confidence of the block.
    :param metadata: The metadata of the block.
    :param children: The children of the block (for nested blocks).
    """

    id: int

    type: LayoutBlockType
    specification: LayoutBlockSpecification

    bounding_box: LayoutBlockBoundingBox
    page_number: int

    text_content: str | None = None
    byte_content: bytes | None = None

    anotation: str | None = None

    confidence: float = 1.0
    metadata: dict[str, Any] | None = None
    children: list[LayoutBlock] | None = None

    @staticmethod
    def from_dict(
        block_data: dict[str, Any],
    ) -> LayoutBlock:
        """
        A layout block class that represents a block in a document layout.

        :param block_data: The data of the block.
        :return: The layout block.
        """

        btype = block_data.get("type", "UNKNOWN")

        if isinstance(btype, str):
            block_type, specification = block_string_to_enum(btype)
        else:
            block_type = LayoutBlockType(block_data["type"])
            specification = LayoutBlockSpecification(
                block_data.get("specification", 0),
            )

        return LayoutBlock(
            id=block_data.get("id", -1),
            type=block_type,
            specification=specification,
            bounding_box=LayoutBlockBoundingBox.from_dict(block_data["bounding_box"]),
            page_number=block_data.get("page_number", None),
            text_content=block_data.get("text_content", None),
            byte_content=block_data.get("byte_content", None),
            anotation=block_data.get("anotation", None),
            confidence=block_data.get("confidence", 1.0),
            metadata=block_data.get("metadata", None),
            children=(
                [
                    LayoutBlock.from_dict(child_data)
                    for child_data in block_data["children"]
                ]
                if "children" in block_data and block_data["children"] is not None
                else None
            ),
        )

    def extract_image_block(self, pages: list[Image.Image]) -> Image.Image:
        """
        Extract the image block from the pages.

        :param pages: The pages of the document.
        :return: The image block.
        """

        x0, y0, x1, y1 = self.bounding_box.to_points()
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

        return f"LayoutBlock(type={self.type}, specification={self.specification})"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the layout block to a dictionary.

        :return: The dictionary representation of the layout block.
        """

        return {
            "id": self.id,
            "block_type": self.type.name,
            "block_specification": self.specification.name,
            "bounding_box": self.bounding_box.to_dict(),
            "page_number": self.page_number,
            "text_content": self.text_content,
            "byte_content": self.byte_content,
            "anotation": self.anotation,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "children": (
                [child.to_dict() for child in self.children] if self.children else None
            ),
        }

    def with_fields(
        self,
        block_id: int | None = None,
        block_type: LayoutBlockType | None = None,
        block_specification: LayoutBlockSpecification | None = None,
        bounding_box: (
            tuple[float, float, float, float] | LayoutBlockBoundingBox | None
        ) = None,
        page_number: int | None = None,
        text_content: str | None = None,
        byte_content: bytes | None = None,
        anotation: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
        children: list[LayoutBlock] | None = None,
    ) -> LayoutBlock:
        """
        Create a new LayoutBlock with updated fields.
        """

        if isinstance(bounding_box, tuple):
            bounding_box = LayoutBlockBoundingBox.from_points(*bounding_box)

        return LayoutBlock(
            block_id or self.id,
            block_type or self.type,
            block_specification or self.specification,
            bounding_box or self.bounding_box,
            page_number or self.page_number,
            text_content or self.text_content,
            byte_content or self.byte_content,
            anotation or self.anotation,
            confidence or self.confidence,
            metadata or self.metadata,
            children or self.children,
        )

    def __repr__(self) -> str:
        return self.to_text()

    def __str__(self) -> str:
        return self.to_text()


@dataclass
class Layout:
    """
    A layout class that represents a layout of a document.

    :param blocks: The blocks in the layout.
    :param page_count: The number of pages in the layout.
    :param metadata: The metadata of the layout.
    """

    blocks: list[LayoutBlock]
    page_count: int = 0
    metadata: dict[str, Any] | None = None

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
            [LayoutBlock.from_dict(block_data) for block_data in layout_data["blocks"]],
            layout_data.get("page_count", 0),  # type: ignore
            layout_data.get("metadata", None),  # type: ignore
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
                block.with_fields(block_id=index).to_dict()
                for index, block in enumerate(self.blocks)
                if not graphics_only
                or (
                    block.type == LayoutBlockType.PICTURE
                    or block.type == LayoutBlockType.TABLE
                    or block.type == LayoutBlockType.CHART
                )
            ],
            "page_count": self.page_count,
            "metadata": self.metadata,
        }

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> LayoutBlock:
        return self.blocks[index]

    def __repr__(self) -> str:
        return self.to_text()

    def __str__(self) -> str:
        return self.to_text()
