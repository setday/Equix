from __future__ import annotations

import pickle

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import UploadFile
from pydantic import BaseModel

app = FastAPI()


class LayoutResponse(BaseModel):  # type: ignore
    layout: Any
    document_id: str


class ExtractionResponse(BaseModel):  # type: ignore
    status: str
    text: str


class InformationResponse(BaseModel):  # type: ignore
    status: str
    answer: str


# TODO: Make a process to avoid rerunning the model each time


def run_cli_command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    return result.stdout
    # return "{'blocks': []}"


@app.post("/layout-extraction", response_model=LayoutResponse)  # type: ignore
async def layout_extraction(document: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await document.read())
        tmp_path = tmp.name

    command = [".\\venv\\Scripts\\python.exe", "-m", "src.cli", "--document", tmp_path, "--extract_layout"]

    try:
        output = run_cli_command(command).splitlines()[-1]
    except HTTPException as e:
        # Handle the error and return a response
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    with open(Path(tmp_path).with_suffix(".json"), "w") as f:
        f.write(output)

    return LayoutResponse(layout=output, document_id=tmp_path)


# TODO: Make specific prompts for each type of extraction


@app.post("/graphics-extraction", response_model=ExtractionResponse)  # type: ignore
async def graphics_extraction(layout_block_id: int, document_id: str, output_type: str):
    document_path = Path(document_id)
    with open(document_path.with_suffix(".json")) as f:
        layout = json.load(f)

    bbox = layout[layout_block_id]["bbox"]
    page_number = layout[layout_block_id]["page_number"]

    command = [
        "python",
        "src/cli.py",
        "--document",
        document_id,
        "--prompt",
        f"Extract graphics from block {layout_block_id}",
        "--crop",
        f"{page_number},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
    ]
    output = run_cli_command(command)
    return ExtractionResponse(status="success", text=output)


@app.post("/information-extraction", response_model=InformationResponse)  # type: ignore
async def information_extraction(prompt: str, document_id: str):
    command = ["python", "src/cli.py", "--document", document_id, "--prompt", prompt]
    output = run_cli_command(command)
    return InformationResponse(status="success", answer=output)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
