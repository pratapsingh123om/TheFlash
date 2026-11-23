from __future__ import annotations

import uvicorn

from .settings import Settings


if __name__ == "__main__":
    settings = Settings.from_env()
    uvicorn.run(
        "backend.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
