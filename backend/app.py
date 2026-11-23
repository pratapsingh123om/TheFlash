from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .bootstrap_assets import (
    DEFAULT_BUNDLE_SHA256,
    DEFAULT_BUNDLE_URI,
    bootstrap_assets,
)
from .settings import Settings, env_bool

if TYPE_CHECKING:
    from .model_runtime import ModelRuntime


settings = Settings.from_env()
runtime: "ModelRuntime | None" = None
runtime_error: str | None = None
forecast_lock = threading.Lock()
runtime_initialization_lock = threading.Lock()


def prepare_runtime_assets(oidc_token: str | None = None) -> None:
    global settings
    if settings.model_dir.is_dir() and settings.input_parquet.is_file():
        return
    auto_bootstrap = env_bool("AUTO_BOOTSTRAP_ASSETS", env_bool("VERCEL", False))
    if not auto_bootstrap:
        return
    result = bootstrap_assets(
        os.environ.get("MODEL_BUNDLE_URI", DEFAULT_BUNDLE_URI),
        os.environ.get("MODEL_BUNDLE_SHA256", DEFAULT_BUNDLE_SHA256),
        Path(os.environ.get("ASSET_ROOT", "/tmp/india-lightning-assets")),
        oidc_token=oidc_token,
    )
    settings = replace(
        settings,
        model_dir=Path(str(result["model_dir"])).resolve(),
        input_parquet=Path(str(result["input_parquet"])).resolve(),
    )


def initialize_runtime() -> None:
    global runtime, runtime_error
    try:
        from .model_runtime import ModelRuntime

        runtime = ModelRuntime(settings.model_dir, settings.output_dir)
        runtime_error = None
    except Exception as exc:  # Keep the UI alive so configuration errors are visible.
        runtime = None
        runtime_error = f"{type(exc).__name__}: {exc}"


def ensure_runtime(request: Request | None = None) -> None:
    global runtime, runtime_error
    if runtime is not None:
        return

    with runtime_initialization_lock:
        if runtime is not None:
            return

        oidc_token = None
        if request is not None:
            oidc_token = request.headers.get("x-vercel-oidc-token")
        if not oidc_token:
            oidc_token = os.environ.get("VERCEL_OIDC_TOKEN")

        is_vercel = env_bool("VERCEL", False)
        if is_vercel and not oidc_token:
            runtime_error = "AssetBootstrapError: RuntimeError: Vercel OIDC header is missing"
            return

        try:
            prepare_runtime_assets(oidc_token=oidc_token)
            initialize_runtime()
        except Exception as exc:
            runtime = None
            runtime_error = f"AssetBootstrapError: {type(exc).__name__}: {exc}"


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
    global runtime_error
    # In Vercel serverless functions, authentication tokens are request-scoped.
    # Initialization is deferred to ensure_runtime(request) on incoming requests.
    if not env_bool("VERCEL", False):
        try:
            prepare_runtime_assets()
            initialize_runtime()
        except Exception as exc:
            runtime_error = f"AssetBootstrapError: {type(exc).__name__}: {exc}"
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
def health(request: Request = None):
    ensure_runtime(request)
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
def config(request: Request = None):
    ensure_runtime(request)
    return {
        "forecast_horizons": settings.forecast_horizons,
        "recursive_mode": settings.recursive_mode,
        "input_file": settings.input_parquet.name,
        "model": runtime.describe() if runtime else None,
        "warning": "Research demonstration only; not an operational public warning service.",
    }


@app.post("/api/forecast/run")
def run_forecast(request: Request = None):
    ensure_runtime(request)
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


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


if settings.static_dir.is_dir():
    app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="frontend")
else:
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def missing_frontend():
        return HTMLResponse(
            "<h1>India Lightning API</h1>"
            "<p>The frontend directory was not included in this deployment.</p>"
            "<p>Inspect <a href='/api/health'>/api/health</a> for backend status.</p>",
            status_code=503,
        )
