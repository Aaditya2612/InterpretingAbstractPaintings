"""Abstract Painting Interpreter — FastAPI server.

Serves the static frontend and exposes a single analysis endpoint. Heavy models
are pre-warmed in a background thread on startup.
"""

from __future__ import annotations

import base64
import threading

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from core import config, pipeline

app = FastAPI(title="Abstract Painting Interpreter", version="2.0.0")


def _heatmap_b64(arr: np.ndarray) -> str | None:
    import cv2

    if not isinstance(arr, np.ndarray):
        return None
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.get("/health")
def health():
    return {"status": "ok", "models": pipeline.status()}


@app.get("/api/samples")
def samples():
    samples_dir = config.SAMPLES_DIR
    if not samples_dir.exists():
        return []
    items = []
    for p in sorted(samples_dir.glob("*")):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            items.append({"name": p.name, "url": f"/static/samples/{p.name}"})
    return items


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    # Reject non-images early.
    header = data[:12]
    if not (
        header.startswith(b"\xff\xd8")          # JPEG
        or header.startswith(b"\x89PNG")         # PNG
        or header.startswith(b"RIFF")            # WEBP
    ):
        raise HTTPException(status_code=415, detail="Unsupported image type")

    result = await _run_blocking(data, file.filename or "upload.png")

    # Build a fresh response dict so we never mutate the cached result.
    response = dict(result)
    response["heatmap"] = _heatmap_b64(result.get("heatmap"))
    return JSONResponse(response)


async def _run_blocking(data: bytes, name: str):
    # Offload CPU-bound work off the event loop.
    return await run_in_threadpool(pipeline.run, data, name)


# --- Static frontend --------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


# --- Warmup ----------------------------------------------------------------

if config.WARMUP:
    threading.Thread(target=pipeline.warmup, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=False)
