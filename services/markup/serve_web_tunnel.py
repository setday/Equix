from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import UploadFile
from pydantic import BaseModel

from src.base.layout import Layout
from src.tools.models.layout_extractor import global_layout_extractor
from src.tools.pdf_reader import PDFReader


app = FastAPI()


class LayoutResponse(BaseModel):
    layout: Layout
    document_id: str

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
                            "page_number": 0,
                            "text": "Sample text",
                            "bounding_box": {
                                "x1": 0,
                                "y1": 0,
                                "x2": 100,
                                "y2": 100,
                            },
                        }
                    ],
                },
                "document_id": "sample_document_id",
            }
        }
        

@app.post("/layout-extraction", response_model=LayoutResponse)
async def layout_extraction(document: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await document.read())
        tmp_path = tmp.name

    try:
        pdf_reader = PDFReader(Path(tmp_path))

        layout = Layout.from_dict(
            {
                "blocks": [
                    box.update({"page_number": page_number}) or box
                    for page_number, image in enumerate(pdf_reader.images)
                    for box in global_layout_extractor.make_layout(image)
                ],
            },
        )
    except HTTPException as e:
        # Handle the error and return a response
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    
    with open(Path(tmp_path).with_suffix(".json"), "w") as f:
        json.dump(layout, f)

    return LayoutResponse(layout=layout, document_id=tmp_path)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5123)
