from __future__ import annotations

import base64
import json
import os
import threading
import zipfile
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from .bootstrap_assets import (
    DEFAULT_BUNDLE_SHA256,
    DEFAULT_BUNDLE_URI,
    bootstrap_assets,
    download_bundle,
)
from .settings import Settings, env_bool

if TYPE_CHECKING:
    from .model_runtime import ModelRuntime


settings = Settings.from_env()
runtime: "ModelRuntime | None" = None
runtime_error: str | None = None
forecast_lock = threading.Lock()
runtime_initialization_lock = threading.Lock()
lightning_history_download_lock = threading.Lock()


def extract_jwt_claims(token: str | None) -> dict:
    if not token:
        return {}
    try:
        parts = token.split(".")
        if len(parts) >= 2:
            padded = parts[1] + "=" * (-len(parts[1]) % 4)
            return json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except Exception:
        pass
    return {}


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
            claims = extract_jwt_claims(oidc_token)
            sub_info = f" [OIDC sub: {claims.get('sub')}]" if claims.get("sub") else ""
            runtime_error = f"AssetBootstrapError: {type(exc).__name__}: {exc}{sub_info}"


def resolve_lightning_history(request: Request | None = None) -> Path | None:
    if settings.lightning_history_parquet is not None:
        if not settings.lightning_history_parquet.is_file():
            raise FileNotFoundError(
                "LIGHTNING_HISTORY_PARQUET does not exist: "
                f"{settings.lightning_history_parquet}"
            )
        return settings.lightning_history_parquet

    if not settings.lightning_history_uri:
        return None

    oidc_token = request.headers.get("x-vercel-oidc-token") if request else None
    if not oidc_token:
        oidc_token = os.environ.get("VERCEL_OIDC_TOKEN")
    target = settings.output_dir / "latest_lightning_history.parquet"
    temporary = target.with_suffix(".parquet.tmp")
    with lightning_history_download_lock:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            download_bundle(settings.lightning_history_uri, temporary, oidc_token=oidc_token)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def generate_forecast() -> bool:
    global runtime_error
    if runtime is None:
        return False
    if not forecast_lock.acquire(blocking=False):
        return False
    try:
        history_path = (
            resolve_lightning_history()
            if settings.default_forecast_mode == "lightning_only_today"
            else None
        )
        runtime.run_24h(
            settings.input_parquet,
            recursive=settings.recursive_mode,
            horizons=settings.forecast_horizons,
            mode=settings.default_forecast_mode,
            history_path=history_path,
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
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)


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
        "default_forecast_mode": settings.default_forecast_mode,
        "external_lightning_history_configured": bool(
            settings.lightning_history_uri
            or (
                settings.lightning_history_parquet
                and settings.lightning_history_parquet.is_file()
            )
        ),
        "error": runtime_error,
    }


@app.get("/api/config")
def config(request: Request = None):
    ensure_runtime(request)
    return {
        "forecast_horizons": settings.forecast_horizons,
        "recursive_mode": settings.recursive_mode,
        "default_forecast_mode": settings.default_forecast_mode,
        "supported_forecast_modes": [
            "lightning_only_today",
            "full_features_replay",
        ],
        "external_lightning_history_file": (
            settings.lightning_history_parquet.name
            if settings.lightning_history_parquet
            else None
        ),
        "external_lightning_history_uri_configured": bool(
            settings.lightning_history_uri
        ),
        "input_file": settings.input_parquet.name,
        "model": runtime.describe() if runtime else None,
        "warning": "Research demonstration only; not an operational public warning service.",
    }


@app.post("/api/forecast/run")
def run_forecast(request: Request = None, mode: str | None = None):
    ensure_runtime(request)
    if runtime is None:
        raise HTTPException(status_code=503, detail=runtime_error or "Model is not loaded")
    selected_mode = (mode or settings.default_forecast_mode).strip().lower()
    if selected_mode not in {"lightning_only_today", "full_features_replay"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "mode must be lightning_only_today or full_features_replay"
            ),
        )
    if not forecast_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A forecast is already running")
    try:
        history_path = (
            resolve_lightning_history(request)
            if selected_mode == "lightning_only_today"
            else None
        )
        return runtime.run_24h(
            settings.input_parquet,
            recursive=settings.recursive_mode,
            horizons=settings.forecast_horizons,
            mode=selected_mode,
            history_path=history_path,
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


ARCHIVE_ZIP_URI = os.environ.get(
    "ARCHIVE_ZIP_URI",
    "gs://nuclei-lenses/flash-model/india_lightning_2023_daily_archive.zip",
)
archive_initialization_lock = threading.Lock()
COMPARISON_ARCHIVE_URI = os.environ.get(
    "COMPARISON_ARCHIVE_URI",
    "gs://nuclei-lenses/flash-model/india_lightning_2023_comparison_archive.zip",
)
comparison_initialization_lock = threading.Lock()


def get_archive_path(oidc_token: str | None = None) -> Path:
    local_candidates = [
        settings.root_dir / "data" / "india_lightning_2023_daily_archive.zip",
        Path(os.environ.get("ASSET_ROOT", "/tmp/india-lightning-assets")).resolve()
        / "india_lightning_2023_daily_archive.zip",
        settings.output_dir / "india_lightning_2023_daily_archive.zip",
        Path("c:/DEV/minor-project/artifacts/india_lightning_2023_daily_archive.zip"),
    ]
    for candidate in local_candidates:
        if candidate.is_file() and candidate.stat().st_size > 1000:
            return candidate

    target = (
        Path(os.environ.get("ASSET_ROOT", "/tmp/india-lightning-assets")).resolve()
        / "india_lightning_2023_daily_archive.zip"
    )
    if target.is_file() and target.stat().st_size > 1000:
        return target

    with archive_initialization_lock:
        if target.is_file() and target.stat().st_size > 1000:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_suffix(".tmp")
        download_bundle(ARCHIVE_ZIP_URI, temp_path, oidc_token=oidc_token)
        temp_path.replace(target)
        return target


def get_comparison_archive_path(oidc_token: str | None = None) -> Path:
    local_candidates = [
        settings.root_dir / "data" / "india_lightning_2023_comparison_archive.zip",
        Path(os.environ.get("ASSET_ROOT", "/tmp/india-lightning-assets")).resolve()
        / "india_lightning_2023_comparison_archive.zip",
        settings.output_dir / "india_lightning_2023_comparison_archive.zip",
    ]
    for candidate in local_candidates:
        if candidate.is_file() and candidate.stat().st_size > 1000:
            return candidate

    target = (
        Path(os.environ.get("ASSET_ROOT", "/tmp/india-lightning-assets")).resolve()
        / "india_lightning_2023_comparison_archive.zip"
    )
    with comparison_initialization_lock:
        if target.is_file() and target.stat().st_size > 1000:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".zip.tmp")
        try:
            download_bundle(COMPARISON_ARCHIVE_URI, temporary, oidc_token=oidc_token)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target

@app.get("/api/forecast/archive/dates")
def archive_dates(request: Request = None):
    token = request.headers.get("x-vercel-oidc-token") if request else None
    try:
        zip_path = get_archive_path(token)
        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            return manifest
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to read archive manifest: {exc}"
        ) from exc


@app.get("/api/forecast/archive/day")
def archive_day(date: str, request: Request = None):
    token = request.headers.get("x-vercel-oidc-token") if request else None
    clean_date = date.strip()
    if (
        len(clean_date) != 10
        or clean_date[4] != "-"
        or clean_date[7] != "-"
        or not clean_date.startswith("2023-")
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {date}. Expected YYYY-MM-DD in 2023.",
        )

    try:
        zip_path = get_archive_path(token)
        with zipfile.ZipFile(zip_path, "r") as zf:
            filename = f"daily/{clean_date}.json"
            if filename not in zf.namelist():
                raise HTTPException(
                    status_code=404,
                    detail=f"Date {clean_date} has no recorded lightning in 2023 archive",
                )
            return json.loads(zf.read(filename).decode("utf-8"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read day {clean_date} from archive: {exc}",
        ) from exc


@app.get("/api/comparison/dates")
def comparison_dates(request: Request = None):
    token = request.headers.get("x-vercel-oidc-token") if request else None
    try:
        zip_path = get_comparison_archive_path(token)
        with zipfile.ZipFile(zip_path, "r") as archive:
            return json.loads(archive.read("manifest.json").decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The paired 2023 model-vs-observed archive is not available yet. "
                f"Expected {COMPARISON_ARCHIVE_URI}. {type(exc).__name__}: {exc}"
            ),
        ) from exc


@app.get("/api/comparison/day")
def comparison_day(date: str, request: Request = None):
    token = request.headers.get("x-vercel-oidc-token") if request else None
    clean_date = date.strip()
    if (
        len(clean_date) != 10
        or clean_date[4] != "-"
        or clean_date[7] != "-"
        or not clean_date.startswith("2023-")
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid comparison date: {date}. Expected YYYY-MM-DD in 2023.",
        )
    try:
        zip_path = get_comparison_archive_path(token)
        with zipfile.ZipFile(zip_path, "r") as archive:
            filename = f"daily/{clean_date}.json"
            if filename not in archive.namelist():
                raise HTTPException(
                    status_code=404,
                    detail=f"No paired model-vs-observed result exists for {clean_date}",
                )
            return json.loads(archive.read(filename).decode("utf-8"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The paired 2023 model-vs-observed archive is not available yet. "
                f"Expected {COMPARISON_ARCHIVE_URI}. {type(exc).__name__}: {exc}"
            ),
        ) from exc


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
