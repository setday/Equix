from __future__ import annotations

import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.base.layout import Layout
from src.tools.models.layout_extractor import global_layout_extractor
from src.tools.pdf_reader import PDFReader


app = FastAPI(
    title="Equix Layout Extraction API",
    description="API for extracting layout from PDF documents.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LayoutResponse(BaseModel):  # type: ignore
    layout: Layout
    document_id: str
    processing_time: float

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Layout: lambda v: v.to_dict(),
        }
        json_schema_extra = {
            "example": {
                "layout": {
                    "blocks": [
                        {
                            "type": "text",
                            "specification": "unknown",
                            "bounding_box": [0, 0, 0.1, 0.1],
                            "page_number": 0,
                            "text_content": "Sample text",
                            "byte_content": None,
                            "anotation": None,
                            "confidence": 0.9,
                            "metadata": None,
                            "children": None,
                        },
                    ],
                },
                "document_id": "sample_document_id",
                "processing_time": 100.0,
            },
        }


def extract_layout(path: Path) -> Layout:
    def enhance_box(
        box: dict[str, Any],
        page_number: int,
        page_dimensions: tuple[int, int],
    ) -> dict[str, Any]:
        box.update(
            {
                "page_number": page_number,
                "bounding_box": {
                    "x": box["bounding_box"]["x"] / page_dimensions[0],
                    "y": box["bounding_box"]["y"] / page_dimensions[1],
                    "width": box["bounding_box"]["width"] / page_dimensions[0],
                    "height": box["bounding_box"]["height"] / page_dimensions[1],
                },
            },
        )
        return box

    pdf_reader = PDFReader(path)

    return Layout.from_dict(
        {
            "blocks": [
                enhance_box(box, page_number, image.size)
                for page_number, image in enumerate(pdf_reader.images)
                for box in global_layout_extractor.make_layout(image)
            ],
            "page_count": len(pdf_reader.images),  # type: ignore
        },
    )


@app.post("/layout-extraction", response_model=LayoutResponse)  # type: ignore
async def layout_extraction(document: UploadFile) -> LayoutResponse:
    """
    Endpoint for extracting layout from a PDF document.
    """

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await document.read())
        tmp_path = tmp.name

    processing_time_start = time.time()

    try:
        layout = extract_layout(Path(tmp_path))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during layout extraction: {str(e)}",
        ) from None

    processing_time_end = time.time()

    with open(Path(tmp_path).with_suffix(".json"), "wb") as f:
        pickle.dump(layout, f)

    return LayoutResponse(
        layout=layout,
        document_id=tmp_path,
        processing_time=processing_time_end - processing_time_start,
    )


@app.get("/health")  # type: ignore
async def health() -> dict[str, str]:
    """
    Health check endpoint.
    """

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5123)
