"""Community collection site. It stores clips and never stores who sent them."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from voxdetect.audio import AudioReadError
from voxdetect.labels import MAX_BYTES, LabelError, public_schema, validate_submission
from voxdetect.collect.store import (
    collection_stats,
    delete_submission,
    ensure_layout,
    public_record,
    save_submission,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
RATE_LIMIT = 60
RATE_WINDOW_SECONDS = 60.0


def create_app(collection_root: str | Path = "data/collected") -> FastAPI:
    root = Path(collection_root).expanduser().resolve()
    ensure_layout(root)
    app = FastAPI(title="Voxdetect Collect", docs_url=None, redoc_url=None, openapi_url=None)
    recent: deque[float] = deque()

    @app.middleware("http")
    async def privacy_headers(request, call_next):
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Permissions-Policy"] = "microphone=(self)"
        if "set-cookie" in response.headers:
            del response.headers["set-cookie"]
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/schema")
    def schema():
        return public_schema()

    @app.get("/api/stats")
    def stats():
        return collection_stats(root)

    @app.post("/api/submissions")
    async def submit(
        audio: UploadFile = File(...),
        label: str = Form(...),
        consent: str = Form(""),
        language: str = Form(""),
        noise_kind: str = Form(""),
        prompt_id: str = Form(""),
    ):
        _throttle(recent)
        payload = await audio.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="That file is larger than 8 MB.")
        try:
            normalized = validate_submission(
                label=label,
                language=language,
                noise_kind=noise_kind,
                prompt_id=prompt_id,
                consent=_consent(consent),
            )
            record = save_submission(root, payload, normalized)
        except LabelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AudioReadError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return public_record(record)

    @app.delete("/api/submissions/{submission_id}")
    def remove(submission_id: str):
        try:
            removed = delete_submission(root, submission_id)
        except LabelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not removed:
            raise HTTPException(status_code=404, detail="That clip is not in the catalog.")
        return {"deleted": submission_id}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _consent(value: str) -> bool:
    return value.strip().lower() in {"yes", "true", "1", "on"}


def _throttle(recent: deque[float]) -> None:
    now = time.monotonic()
    while recent and now - recent[0] > RATE_WINDOW_SECONDS:
        recent.popleft()
    if len(recent) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="The collector is busy. Wait a minute and try again.")
    recent.append(now)
