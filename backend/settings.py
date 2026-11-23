from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    model_dir: Path
    input_parquet: Path
    output_dir: Path
    static_dir: Path
    recursive_mode: bool
    auto_run_forecast: bool
    forecast_horizons: int
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parents[1])).resolve()
        vercel = env_bool("VERCEL", False)
        default_output = Path("/tmp/india-lightning-output") if vercel else root / "runtime"
        return cls(
            root_dir=root,
            model_dir=Path(os.environ.get("MODEL_DIR", root / "models")).resolve(),
            input_parquet=Path(
                os.environ.get("INPUT_PARQUET", root / "data" / "next_24h_features.parquet")
            ).resolve(),
            output_dir=Path(os.environ.get("OUTPUT_DIR", default_output)).resolve(),
            static_dir=(root / "frontend").resolve(),
            recursive_mode=env_bool("RECURSIVE_MODE", True),
            auto_run_forecast=env_bool("AUTO_RUN_FORECAST", not vercel),
            forecast_horizons=max(1, min(4, int(os.environ.get("FORECAST_HORIZONS", "4")))),
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8080")),
        )
