"""Improved PDF reader with proper error handling and interface implementation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pdf2image
from PIL import Image

from src.config import config
from src.core.exceptions import PDFReadError, ValidationError
from src.core.interfaces import PDFReaderInterface
from src.core.logging import get_logger

logger = get_logger()


class PDFReader(PDFReaderInterface):
    """Enhanced PDF reader with caching and error handling."""

    def __init__(self, pdf_path: Path | None = None) -> None:
        """Initialize the PDFReader.

        Args:
            pdf_path: Optional path to PDF file to load immediately
        """
        self._pdf_path: Path | None = None
        self._images: list[Image.Image] | None = None
        self._pdf_hash: str | None = None
        
        if pdf_path:
            self.load_pdf(pdf_path)

    def load_pdf(self, pdf_path: Path) -> None:
        """Load a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Raises:
            ValidationError: If file validation fails
            PDFReadError: If PDF reading fails
        """
        self._validate_pdf_file(pdf_path)
        
        try:
            self._pdf_path = pdf_path
            self._pdf_hash = self._calculate_file_hash(pdf_path)
            self._images = self._extract_images()
            
            logger.info(
                f"Successfully loaded PDF: {pdf_path}",
                pdf_path=str(pdf_path),
                page_count=len(self._images),
                file_size=pdf_path.stat().st_size,
            )
            
        except Exception as e:
            raise PDFReadError(
                f"Failed to load PDF {pdf_path}: {e}",
                "PDF_LOAD_ERROR",
                {"pdf_path": str(pdf_path), "original_error": str(e)},
            ) from e

    def read_pdf(self, pdf_path: Path) -> list[Image.Image]:
        """Read a PDF file and return images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PIL Images, one per page
        """
        if self._pdf_path != pdf_path or self._images is None:
            self.load_pdf(pdf_path)
        
        assert self._images is not None, "PDF images not loaded | Unexpected state"

        return self._images.copy()  # Return copy to prevent external modification

    def get_page_count(self, pdf_path: Path | None = None) -> int:
        """Get the number of pages in the PDF.
        
        Args:
            pdf_path: Optional path to PDF file. If None, uses loaded PDF.
            
        Returns:
            Number of pages
        """
        if pdf_path and pdf_path != self._pdf_path:
            self.load_pdf(pdf_path)
        elif self._images is None:
            raise PDFReadError(
                "No PDF loaded. Please provide pdf_path or call load_pdf first.",
                "NO_PDF_LOADED",
            )
        
        assert self._images is not None, "PDF images not loaded | Unexpected state"
        
        return len(self._images)

    def _validate_pdf_file(self, pdf_path: Path) -> None:
        """Validate PDF file.
        
        Args:
            pdf_path: Path to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if not pdf_path.exists():
            raise ValidationError(
                f"PDF file not found: {pdf_path}",
                "FILE_NOT_FOUND",
                {"pdf_path": str(pdf_path)},
            )

        if not pdf_path.is_file():
            raise ValidationError(
                f"Path is not a file: {pdf_path}",
                "NOT_A_FILE",
                {"pdf_path": str(pdf_path)},
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValidationError(
                f"File is not a PDF: {pdf_path}",
                "INVALID_FILE_TYPE",
                {"pdf_path": str(pdf_path), "suffix": pdf_path.suffix},
            )

        # Check file size
        file_size = pdf_path.stat().st_size
        max_size = config.data.max_file_size_mb * 1024 * 1024
        if file_size > max_size:
            raise ValidationError(
                f"PDF file too large: {file_size} bytes (max: {max_size} bytes)",
                "FILE_TOO_LARGE",
                {"file_size": file_size, "max_size": max_size},
            )

    def _calculate_file_hash(self, pdf_path: Path) -> str:
        """Calculate hash of PDF file for caching.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            SHA256 hash of file
        """
        hash_sha256 = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _extract_images(self) -> list[Image.Image]:
        """Extract images from the loaded PDF.
        
        Returns:
            List of PIL Images
            
        Raises:
            PDFReadError: If extraction fails
        """
        if not self._pdf_path:
            raise PDFReadError(
                "No PDF path set",
                "NO_PDF_PATH",
            )

        try:
            images = pdf2image.convert_from_path(
                self._pdf_path,
                dpi=config.data.pdf_dpi,
                fmt="RGB",
            )
            
            if not images:
                raise PDFReadError(
                    f"No images extracted from PDF: {self._pdf_path}",
                    "NO_IMAGES_EXTRACTED",
                    {"pdf_path": str(self._pdf_path)},
                )
            
            return images
            
        except Exception as e:
            raise PDFReadError(
                f"Failed to extract images from PDF: {e}",
                "IMAGE_EXTRACTION_ERROR",
                {"pdf_path": str(self._pdf_path), "original_error": str(e)},
            ) from e

    def as_single_image(self) -> Image.Image:
        """Convert all pages to a single horizontally concatenated image.
        
        Returns:
            Single PIL Image containing all pages
            
        Raises:
            PDFReadError: If no images are loaded
        """
        if not self._images:
            raise PDFReadError(
                "No images loaded. Call load_pdf first.",
                "NO_IMAGES_LOADED",
            )

        try:
            images = self._images
            widths, heights = zip(*(img.size for img in images))

            total_width = sum(widths)
            max_height = max(heights)

            new_image = Image.new("RGB", (total_width, max_height), "white")

            x_offset = 0
            for image in images:
                new_image.paste(image, (x_offset, 0))
                x_offset += image.size[0]

            return new_image
            
        except Exception as e:
            raise PDFReadError(
                f"Failed to create single image: {e}",
                "SINGLE_IMAGE_ERROR",
                {"original_error": str(e)},
            ) from e

    def save_images(self, output_dir: Path, prefix: str = "page") -> list[Path]:
        """Save all pages as separate image files.
        
        Args:
            output_dir: Directory to save images
            prefix: Filename prefix
            
        Returns:
            List of saved image paths
            
        Raises:
            PDFReadError: If no images are loaded or saving fails
        """
        if not self._images:
            raise PDFReadError(
                "No images loaded. Call load_pdf first.",
                "NO_IMAGES_LOADED",
            )

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            saved_paths = []
            
            for i, image in enumerate(self._images):
                filename = f"{prefix}_{i:03d}.png"
                image_path = output_dir / filename
                image.save(image_path, "PNG")
                saved_paths.append(image_path)
            
            logger.info(
                f"Saved {len(saved_paths)} images to {output_dir}",
                output_dir=str(output_dir),
                image_count=len(saved_paths),
            )
            
            return saved_paths
            
        except Exception as e:
            raise PDFReadError(
                f"Failed to save images: {e}",
                "IMAGE_SAVE_ERROR",
                {"output_dir": str(output_dir), "original_error": str(e)},
            ) from e

    @property
    def images(self) -> list[Image.Image]:
        """Get loaded images.
        
        Returns:
            List of loaded images
            
        Raises:
            PDFReadError: If no images are loaded
        """
        if not self._images:
            raise PDFReadError(
                "No images loaded. Call load_pdf first.",
                "NO_IMAGES_LOADED",
            )
        return self._images.copy()

    @property
    def pdf_path(self) -> Path | None:
        """Get the current PDF path."""
        return self._pdf_path

    @property
    def pdf_hash(self) -> str | None:
        """Get the hash of the current PDF."""
        return self._pdf_hash

    def __len__(self) -> int:
        """Get number of pages."""
        return len(self._images) if self._images else 0

    def __getitem__(self, index: int) -> Image.Image:
        """Get page by index."""
        if not self._images:
            raise PDFReadError(
                "No images loaded. Call load_pdf first.",
                "NO_IMAGES_LOADED",
            )
        return self._images[index]

    def __iter__(self) -> Iterator[Image.Image]:
        """Iterate over pages."""
        if not self._images:
            raise PDFReadError(
                "No images loaded. Call load_pdf first.",
                "NO_IMAGES_LOADED",
            )
        return iter(self._images)

    def __repr__(self) -> str:
        """String representation."""
        return f"PDFReader(pdf_path={self._pdf_path}, pages={len(self) if self._images else 0})"

    def __str__(self) -> str:
        """String representation."""
        return self.__repr__()
