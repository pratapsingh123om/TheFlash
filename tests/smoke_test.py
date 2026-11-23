from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.model_runtime import HISTORY_FEATURES, ModelRuntime


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def train_model(path: Path, features: list[str], severity: bool) -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(80, len(features))).astype(np.float32)
    if severity:
        target = np.log1p(np.maximum(0, 8 + 2 * values[:, 0])).astype(np.float32)
        params = {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.2}
    else:
        target = (values[:, 0] + values[:, 1] > 0).astype(np.float32)
        params = {"objective": "binary:logistic", "max_depth": 2, "eta": 0.2}
    model = xgb.train(
        params,
        xgb.DMatrix(values, label=target, feature_names=features),
        num_boost_round=5,
    )
    model.save_model(path)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="lightning_demo_smoke_") as temporary:
        root = Path(temporary)
        model_dir = root / "models"
        output_dir = root / "runtime"
        model_dir.mkdir()
        features = ["grid_lat_center", "grid_lon_center", *HISTORY_FEATURES]
        configuration = {
            "features": features,
            "forecast_horizon_hours": 6,
            "same_window_weather_included": False,
            "posthoc_multiplier": False,
            "split": {"train": "2019-2021", "validation": "2022", "test": "2023"},
        }
        write_json(model_dir / "run_configuration.json", configuration)
        write_json(
            model_dir / "locked_storm_category_thresholds.json",
            {"q95": 68, "q99": 203},
        )
        write_json(
            model_dir / "alarm_thresholds_selected_on_2022.json",
            {
                "thresholds": {"none": 0.5, "mild": 0.25, "strong": 0.2, "heavy": 0.15},
                "cumulative_head_thresholds": {"any": 0.2, "q95": 0.2, "q99": 0.15},
            },
        )
        write_json(model_dir / "selection_frozen_before_2023.json", {"status": "frozen"})
        write_json(model_dir / "training_complete.json", {"status": "complete"})
        for name in ("any", "q95", "q99"):
            train_model(model_dir / f"ordinal_{name}.json", features, severity=False)
            write_json(
                model_dir / f"isotonic_{name}_selected_on_2022.json",
                {"x_thresholds": [-8, 8], "y_thresholds": [0.01, 0.99]},
            )
        for name in ("mild", "strong", "heavy"):
            train_model(model_dir / f"severity_{name}.json", features, severity=True)

        start = datetime(2023, 9, 29, 12)
        rows = []
        for horizon in range(4):
            issue = start + timedelta(hours=6 * horizon)
            valid = issue + timedelta(hours=6)
            for grid_id in range(5):
                lag1 = float(grid_id + 1)
                lag2 = float(grid_id)
                lag3 = float(max(grid_id - 1, 0))
                lag4 = float(max(grid_id - 2, 0))
                row = {
                    "forecast_issue_time_ist": issue,
                    "forecast_valid_time_ist": valid,
                    "grid_id": grid_id,
                    "grid_lat_center": 20.0 + grid_id * 0.25,
                    "grid_lon_center": 78.0 + grid_id * 0.25,
                    "lightning_lag6h_count": lag1,
                    "lightning_lag12h_count": lag2,
                    "lightning_lag24h_count": lag4,
                    "lightning_recent12h_sum": lag1 + lag2,
                    "lightning_recent24h_sum": lag1 + lag2 + lag3 + lag4,
                    "lightning_local_trend_6h": lag1 - lag2,
                    "lightning_local_acceleration_6h": lag1 - 2 * lag2 + lag3,
                    "lightning_active_lag6h": float(lag1 > 0),
                }
                for name in HISTORY_FEATURES:
                    row.setdefault(name, 0.0)
                rows.append(row)
        input_path = root / "input.parquet"
        pl.DataFrame(rows).write_parquet(input_path)

        runtime = ModelRuntime(model_dir, output_dir)
        result = runtime.run_24h(input_path, recursive=True, horizons=4)
        assert result["status"] == "complete"
        assert len(result["periods"]) == 4
        assert all(len(period["cells"]) == 5 for period in result["periods"])
        assert (output_dir / "latest_forecast.json").exists()

        frontend = root / "frontend"
        frontend.mkdir()
        (frontend / "index.html").write_text("<h1>smoke</h1>", encoding="utf-8")
        os.environ.update(
            {
                "APP_ROOT": str(root),
                "MODEL_DIR": str(model_dir),
                "INPUT_PARQUET": str(input_path),
                "OUTPUT_DIR": str(output_dir),
                "AUTO_RUN_FORECAST": "0",
                "FORECAST_HORIZONS": "4",
            }
        )
        from backend import app as api

        api.initialize_runtime()
        health = api.health()
        assert health["model_loaded"] is True
        response = api.run_forecast()
        assert len(response["periods"]) == 4
        print("SMOKE TEST PASSED", len(result["periods"]), "horizons")


if __name__ == "__main__":
    main()
