from __future__ import annotations

from fastapi import FastAPI


try:
    from backend.app import app
except Exception as exc:
    startup_error = f"{type(exc).__name__}: {exc}"
    app = FastAPI(title="India Lightning Startup Diagnostics")

    @app.get("/")
    def startup_failure():
        return {
            "status": "startup_failed",
            "message": "The application could not be imported.",
            "error": startup_error,
        }

    @app.get("/api/health")
    def startup_health():
        return {
            "status": "startup_failed",
            "model_loaded": False,
            "input_exists": False,
            "forecast_exists": False,
            "error": startup_error,
        }
