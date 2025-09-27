# FastAPI backend for Doc UI (Angular frontend)
from fastapi import FastAPI, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import hashlib
import json
import os
import shutil
from typing import Dict, Any
from pydantic import ValidationError

# Import parser singleton and Pydantic models for schema exposure
from doc_ui.backend.shared.parser_wrapper import get_parser
from extraction_io.ExtractionItems import ExtractionItem, ExtractionItems
from extraction_io.ExtractionOutputs import (
    KeyValueOutput,
    BulletPointsOutput,
    SummaryOutput,
    CheckboxOutput,
    TableOutput,
)

app = FastAPI(title="Doc UI Backend", version="1.0.0")

# CORS for Angular dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "*",  # can be tightened in prod
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "output"
DE_CONFIG_DIR = ROOT / "de_config"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/perform_de")
async def perform_de(pdf: UploadFile, config_name: str = Form(None), config_json: str = Form(None)):
    dataset_path = DATASET_DIR / pdf.filename
    output_path = OUTPUT_DIR / pdf.filename.replace(".pdf", ".json")

    if not dataset_path.exists():
        with dataset_path.open("wb") as f:
            shutil.copyfileobj(pdf.file, f)

    try:
        if config_json:
            # Prefer inline config JSON if provided
            try:
                extraction_config = json.loads(config_json)
                # Validate config
                _ = ExtractionItems.model_validate(extraction_config)
            except (json.JSONDecodeError, ValidationError) as ve:
                return JSONResponse(status_code=400, content={"error": "Invalid config_json", "details": getattr(ve, 'errors', lambda: str(ve))() if hasattr(ve, 'errors') else str(ve)})
        else:
            if not config_name:
                return JSONResponse(status_code=400, content={"error": "Either config_name or config_json must be provided."})
            config_path = DE_CONFIG_DIR / config_name
            with config_path.open("r") as file:
                extraction_config = json.load(file)

        get_parser().perform_de(str(dataset_path), extraction_config, str(output_path))
        with output_path.open("r") as f:
            result = json.load(f)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/configs/validate")
def validate_config(payload: Any = Body(...)):
    try:
        _ = ExtractionItems.model_validate(payload)
        return {"ok": True}
    except ValidationError as ve:
        return JSONResponse(status_code=400, content={"ok": False, "errors": ve.errors()})


@app.post("/configs/save")
def save_config(name: str = Body(..., embed=True), content: Any = Body(..., embed=True)):
    # Basic safety: only allow .json and no path traversal
    if not name.endswith(".json") or "/" in name or ".." in name:
        return JSONResponse(status_code=400, content={"error": "Invalid config name"})
    try:
        # Validate before saving
        _ = ExtractionItems.model_validate(content)
        path = DE_CONFIG_DIR / name
        with path.open("w") as f:
            json.dump(content, f, indent=2)
        return {"ok": True, "name": name}
    except ValidationError as ve:
        return JSONResponse(status_code=400, content={"ok": False, "errors": ve.errors()})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/schemas/extraction_item")
def schema_extraction_item():
    return ExtractionItem.model_json_schema()


@app.get("/schemas/outputs")
def schema_outputs():
    return {
        "key_value": KeyValueOutput.model_json_schema(),
        "bullet_points": BulletPointsOutput.model_json_schema(),
        "summary": SummaryOutput.model_json_schema(),
        "checkbox": CheckboxOutput.model_json_schema(),
        "table": TableOutput.model_json_schema(),
    }


@app.get("/schemas/version")
def schema_version():
    blob = json.dumps(
        {
            "extraction_item": ExtractionItem.model_json_schema(),
            "outputs": {
                "key_value": KeyValueOutput.model_json_schema(),
                "bullet_points": BulletPointsOutput.model_json_schema(),
                "summary": SummaryOutput.model_json_schema(),
                "checkbox": CheckboxOutput.model_json_schema(),
                "table": TableOutput.model_json_schema(),
            },
        },
        sort_keys=True,
    ).encode("utf-8")
    h = hashlib.sha256(blob).hexdigest()
    return {"schema_hash": h}


@app.get("/configs/list")
def list_configs():
    files = [p.name for p in DE_CONFIG_DIR.glob("*.json")]
    return {"configs": sorted(files)}


@app.get("/configs/get")
def get_config(name: str):
    path = DE_CONFIG_DIR / name
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": f"Config not found: {name}"})
    with path.open("r") as f:
        data = json.load(f)
    return data


@app.get("/")
def root_index():
    return {
        "service": "doc-ui-backend",
        "ok": True,
        "hint": "This is the backend. Open the Angular UI at http://localhost:4200",
        "use": [
            "/health",
            "/perform_de",
            "/configs/list",
            "/configs/get?name=...",
            "/configs/validate",
            "/configs/save",
            "/schemas/extraction_item",
            "/schemas/outputs",
            "/schemas/version",
        ],
        "docs": "http://localhost:8001/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "doc_ui.backend.server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
