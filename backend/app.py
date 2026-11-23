from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model_runtime import ModelRuntime
from .settings import Settings


settings = Settings.from_env()
runtime: ModelRuntime | None = None
runtime_error: str | None = None
forecast_lock = threading.Lock()


def initialize_runtime() -> None:
    global runtime, runtime_error
    try:
        runtime = ModelRuntime(settings.model_dir, settings.output_dir)
        runtime_error = None
    except Exception as exc:  # Keep the UI alive so configuration errors are visible.
        runtime = None
        runtime_error = f"{type(exc).__name__}: {exc}"


def generate_forecast() -> bool:
    global runtime_error
    if runtime is None:
        return False
    if not forecast_lock.acquire(blocking=False):
        return False
    try:
        runtime.run_24h(
            settings.input_parquet,
            recursive=settings.recursive_mode,
            horizons=settings.forecast_horizons,
        )
        runtime_error = None
        return True
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        return False
    finally:
        forecast_lock.release()


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_runtime()
    if settings.auto_run_forecast and runtime is not None and settings.input_parquet.exists():
        worker = threading.Thread(target=generate_forecast, daemon=True)
        worker.start()
    yield


app = FastAPI(
    title="India Lightning 24-Hour Demo",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/api/health")
def health():
    latest = settings.output_dir / "latest_forecast.json"
    return {
        "status": "ready" if runtime is not None and runtime_error is None else "attention_required",
        "model_loaded": runtime is not None,
        "input_exists": settings.input_parquet.exists(),
        "forecast_exists": latest.exists(),
        "forecast_running": forecast_lock.locked(),
        "recursive_mode": settings.recursive_mode,
        "error": runtime_error,
    }


@app.get("/api/config")
def config():
    return {
        "forecast_horizons": settings.forecast_horizons,
        "recursive_mode": settings.recursive_mode,
        "input_file": settings.input_parquet.name,
        "model": runtime.describe() if runtime else None,
        "warning": "Research demonstration only; not an operational public warning service.",
    }


@app.post("/api/forecast/run")
def run_forecast():
    if runtime is None:
        raise HTTPException(status_code=503, detail=runtime_error or "Model is not loaded")
    if not forecast_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A forecast is already running")
    try:
        return runtime.run_24h(
            settings.input_parquet,
            recursive=settings.recursive_mode,
            horizons=settings.forecast_horizons,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        forecast_lock.release()


@app.get("/api/forecast/latest")
def latest_forecast():
    path = settings.output_dir / "latest_forecast.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No forecast has been generated yet")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/boundary")
def boundary():
    path = settings.static_dir / "india_boundary.geojson"
    if not path.exists():
        raise HTTPException(status_code=404, detail="India boundary asset is missing")
    return FileResponse(path, media_type="application/geo+json")


app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="frontend")
