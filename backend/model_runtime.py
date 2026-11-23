from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


CLASS_NAMES = ("none", "mild", "strong", "heavy")
HISTORY_FEATURES = (
    "lightning_lag6h_count",
    "lightning_lag12h_count",
    "lightning_lag24h_count",
    "lightning_recent12h_sum",
    "lightning_recent24h_sum",
    "lightning_local_trend_6h",
    "lightning_local_acceleration_6h",
    "lightning_active_lag6h",
    "neighbor50_lag6h_sum",
    "neighbor50_lag12h_sum",
    "neighbor50_lag24h_sum",
    "neighbor50_lag6h_active_cells",
    "neighbor100_lag6h_sum",
    "neighbor100_lag12h_sum",
    "neighbor100_lag24h_sum",
    "neighbor100_recent12h_sum",
    "neighbor100_recent24h_sum",
    "neighbor100_lag6h_active_cells",
    "neighbor100_propagation_6h",
    "neighbor50_degree",
    "neighbor100_degree",
)
REQUIRED_ARTIFACTS = (
    "run_configuration.json",
    "locked_storm_category_thresholds.json",
    "alarm_thresholds_selected_on_2022.json",
    "selection_frozen_before_2023.json",
    "training_complete.json",
    "ordinal_any.json",
    "ordinal_q95.json",
    "ordinal_q99.json",
    "severity_mild.json",
    "severity_strong.json",
    "severity_heavy.json",
    "isotonic_any_selected_on_2022.json",
    "isotonic_q95_selected_on_2022.json",
    "isotonic_q99_selected_on_2022.json",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_value(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="s")
    return str(value).replace(" ", "T")


@dataclass(frozen=True)
class IsotonicCalibrator:
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "IsotonicCalibrator":
        payload = read_json(path)
        x = np.asarray(payload["x_thresholds"], dtype=np.float64)
        y = np.asarray(payload["y_thresholds"], dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.size < 2 or x.size != y.size:
            raise RuntimeError(f"Invalid isotonic calibration file: {path}")
        return cls(x=x, y=y)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.clip(np.interp(values, self.x, self.y), 1e-8, 1.0 - 1e-8)


def adjacency(latitudes: np.ndarray, longitudes: np.ndarray, radius_km: float) -> csr_matrix:
    lat = np.radians(np.asarray(latitudes, dtype=np.float64))
    lon = np.radians(np.asarray(longitudes, dtype=np.float64))
    xyz = np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )
    chord = 2.0 * math.sin(radius_km / (2.0 * 6371.0088))
    groups = cKDTree(xyz).query_ball_point(xyz, r=chord + 1e-12)
    rows: list[int] = []
    columns: list[int] = []
    for row, group in enumerate(groups):
        rows.extend([row] * len(group))
        columns.extend(group)
    return csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(lat), len(lat)),
        dtype=np.float32,
    )


def sparse_sum(matrix: csr_matrix, values: np.ndarray) -> np.ndarray:
    return np.asarray(matrix.dot(np.nan_to_num(values, nan=0.0))).ravel().astype(np.float32)


class ModelRuntime:
    def __init__(self, model_dir: Path, output_dir: Path):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._validate_artifacts()

        self.configuration = read_json(self.model_dir / "run_configuration.json")
        self.features = list(self.configuration["features"])
        if not self.features or len(self.features) != len(set(self.features)):
            raise RuntimeError("Model feature list is empty or contains duplicates")
        self.feature_positions = {name: index for index, name in enumerate(self.features)}

        self.classifiers = {
            name: self._load_booster(self.model_dir / f"ordinal_{name}.json")
            for name in ("any", "q95", "q99")
        }
        self.severity_models = {
            name: self._load_booster(self.model_dir / f"severity_{name}.json")
            for name in ("mild", "strong", "heavy")
        }
        self.calibrators = {
            name: IsotonicCalibrator.load(
                self.model_dir / f"isotonic_{name}_selected_on_2022.json"
            )
            for name in ("any", "q95", "q99")
        }
        alarm_payload = read_json(self.model_dir / "alarm_thresholds_selected_on_2022.json")
        self.class_alarm_thresholds = alarm_payload["thresholds"]
        self.head_alarm_thresholds = alarm_payload["cumulative_head_thresholds"]
        self.count_thresholds = read_json(self.model_dir / "locked_storm_category_thresholds.json")
        self.frozen_selection = read_json(self.model_dir / "selection_frozen_before_2023.json")

    def _validate_artifacts(self) -> None:
        missing = [name for name in REQUIRED_ARTIFACTS if not (self.model_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Deployment model directory is incomplete; missing {missing} in {self.model_dir}"
            )

    @staticmethod
    def _load_booster(path: Path) -> xgb.Booster:
        model = xgb.Booster()
        model.load_model(path)
        return model

    def describe(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "feature_count": len(self.features),
            "forecast_horizon_hours": self.configuration.get("forecast_horizon_hours", 6),
            "training_split": self.configuration.get("split"),
            "q95_count": self.count_thresholds.get("q95"),
            "q99_count": self.count_thresholds.get("q99"),
            "selection_frozen": self.frozen_selection.get("status") == "frozen",
            "posthoc_multiplier": self.configuration.get("posthoc_multiplier", False),
        }

    @staticmethod
    def _iteration_range(model: xgb.Booster) -> dict[str, tuple[int, int]]:
        best = model.attr("best_iteration")
        return {"iteration_range": (0, int(best) + 1)} if best is not None else {}

    def _predict_margin(self, model: xgb.Booster, matrix: xgb.DMatrix) -> np.ndarray:
        return np.asarray(
            model.predict(matrix, output_margin=True, **self._iteration_range(model)),
            dtype=np.float64,
        )

    def _predict_value(self, model: xgb.Booster, matrix: xgb.DMatrix) -> np.ndarray:
        return np.asarray(
            model.predict(matrix, **self._iteration_range(model)), dtype=np.float64
        )

    @staticmethod
    def _coherent_probabilities(
        p_any: np.ndarray, p_q95: np.ndarray, p_q99: np.ndarray
    ) -> np.ndarray:
        p_any = np.clip(p_any, 1e-8, 1.0 - 1e-8)
        p_q95 = np.minimum(np.clip(p_q95, 1e-8, 1.0 - 1e-8), p_any)
        p_q99 = np.minimum(np.clip(p_q99, 1e-8, 1.0 - 1e-8), p_q95)
        probabilities = np.column_stack(
            [1.0 - p_any, p_any - p_q95, p_q95 - p_q99, p_q99]
        )
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)

    def _predict(self, x_values: np.ndarray) -> dict[str, np.ndarray]:
        matrix = xgb.DMatrix(
            x_values, feature_names=self.features, missing=np.nan
        )
        ordinal = {
            name: self.calibrators[name].predict(
                self._predict_margin(model, matrix)
            )
            for name, model in self.classifiers.items()
        }
        probabilities = self._coherent_probabilities(
            ordinal["any"], ordinal["q95"], ordinal["q99"]
        )
        severity = {
            name: np.clip(
                np.expm1(self._predict_value(model, matrix)), 0.0, None
            )
            for name, model in self.severity_models.items()
        }
        expected = (
            probabilities[:, 1] * severity["mild"]
            + probabilities[:, 2] * severity["strong"]
            + probabilities[:, 3] * severity["heavy"]
        )
        argmax_code = np.argmax(probabilities, axis=1).astype(np.uint8)
        warning_code = np.zeros(len(expected), dtype=np.uint8)
        for index, name in enumerate(CLASS_NAMES[1:], start=1):
            warning_code[probabilities[:, index] >= float(self.class_alarm_thresholds[name])] = index
        return {
            "probabilities": probabilities,
            "ordinal_any": ordinal["any"],
            "ordinal_q95": ordinal["q95"],
            "ordinal_q99": ordinal["q99"],
            "severity_mild": severity["mild"],
            "severity_strong": severity["strong"],
            "severity_heavy": severity["heavy"],
            "expected_region50_count": expected,
            "argmax_code": argmax_code,
            "warning_code": warning_code,
        }

    def _history_state(self, first: pl.DataFrame) -> list[np.ndarray]:
        lag1 = first["lightning_lag6h_count"].to_numpy().astype(np.float32)
        lag2 = first["lightning_lag12h_count"].to_numpy().astype(np.float32)
        lag4 = first["lightning_lag24h_count"].to_numpy().astype(np.float32)
        recent24 = first["lightning_recent24h_sum"].to_numpy().astype(np.float32)
        lag3 = recent24 - lag1 - lag2 - lag4
        return [lag1, lag2, lag3, lag4]

    @staticmethod
    def _history_values(
        lags: list[np.ndarray], neighbor50: csr_matrix, neighbor100: csr_matrix
    ) -> dict[str, np.ndarray]:
        lag1, lag2, lag3, lag4 = lags
        n50_lag1 = sparse_sum(neighbor50, lag1)
        n100_lag1 = sparse_sum(neighbor100, lag1)
        n100_lag2 = sparse_sum(neighbor100, lag2)
        return {
            "lightning_lag6h_count": lag1,
            "lightning_lag12h_count": lag2,
            "lightning_lag24h_count": lag4,
            "lightning_recent12h_sum": lag1 + lag2,
            "lightning_recent24h_sum": lag1 + lag2 + lag3 + lag4,
            "lightning_local_trend_6h": lag1 - lag2,
            "lightning_local_acceleration_6h": lag1 - 2.0 * lag2 + lag3,
            "lightning_active_lag6h": (np.nan_to_num(lag1) > 0).astype(np.float32),
            "neighbor50_lag6h_sum": n50_lag1,
            "neighbor50_lag12h_sum": sparse_sum(neighbor50, lag2),
            "neighbor50_lag24h_sum": sparse_sum(neighbor50, lag4),
            "neighbor50_lag6h_active_cells": sparse_sum(
                neighbor50, (np.nan_to_num(lag1) > 0).astype(np.float32)
            ),
            "neighbor100_lag6h_sum": n100_lag1,
            "neighbor100_lag12h_sum": n100_lag2,
            "neighbor100_lag24h_sum": sparse_sum(neighbor100, lag4),
            "neighbor100_recent12h_sum": n100_lag1 + n100_lag2,
            "neighbor100_recent24h_sum": (
                n100_lag1
                + n100_lag2
                + sparse_sum(neighbor100, lag3)
                + sparse_sum(neighbor100, lag4)
            ),
            "neighbor100_lag6h_active_cells": sparse_sum(
                neighbor100, (np.nan_to_num(lag1) > 0).astype(np.float32)
            ),
            "neighbor100_propagation_6h": n100_lag1 - n100_lag2,
            "neighbor50_degree": np.asarray(neighbor50.sum(axis=1)).ravel().astype(np.float32),
            "neighbor100_degree": np.asarray(neighbor100.sum(axis=1)).ravel().astype(np.float32),
        }

    def _validate_input(self, frame: pl.DataFrame, horizons: int) -> list[Any]:
        required = {
            *self.features,
            "forecast_issue_time_ist",
            "forecast_valid_time_ist",
            "grid_id",
            "grid_lat_center",
            "grid_lon_center",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"Input Parquet is missing required columns: {missing}")
        times = (
            frame.select("forecast_valid_time_ist")
            .unique()
            .sort("forecast_valid_time_ist")
            .to_series()
            .to_list()
        )
        if len(times) < horizons:
            raise RuntimeError(
                f"Need {horizons} consecutive six-hour valid times, found {len(times)}"
            )
        return times[:horizons]

    def run_24h(
        self, input_path: Path, recursive: bool = True, horizons: int = 4
    ) -> dict[str, Any]:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Forecast feature input not found: {input_path}")
        with self.lock:
            frame = pl.read_parquet(input_path)
            times = self._validate_input(frame, horizons)
            frame = frame.filter(pl.col("forecast_valid_time_ist").is_in(times)).sort(
                ["forecast_valid_time_ist", "grid_id"]
            )
            slices = [
                frame.filter(pl.col("forecast_valid_time_ist") == valid_time).sort("grid_id")
                for valid_time in times
            ]
            grid_ids = slices[0]["grid_id"].to_numpy()
            latitudes = slices[0]["grid_lat_center"].to_numpy().astype(np.float64)
            longitudes = slices[0]["grid_lon_center"].to_numpy().astype(np.float64)
            for item in slices[1:]:
                if not np.array_equal(item["grid_id"].to_numpy(), grid_ids):
                    raise RuntimeError("Grid IDs are not identical across all four horizons")

            neighbor50 = adjacency(latitudes, longitudes, 50.0)
            neighbor100 = adjacency(latitudes, longitudes, 100.0)
            lags = self._history_state(slices[0])
            periods: list[dict[str, Any]] = []
            for horizon_index, item in enumerate(slices):
                x_values = item.select(self.features).to_numpy().astype(np.float32, copy=False)
                if recursive:
                    history = self._history_values(lags, neighbor50, neighbor100)
                    for name, values in history.items():
                        position = self.feature_positions.get(name)
                        if position is not None:
                            x_values[:, position] = values
                prediction = self._predict(x_values)
                probabilities = prediction["probabilities"]
                cells = []
                for row in range(len(grid_ids)):
                    cells.append(
                        {
                            "grid_id": int(grid_ids[row]),
                            "lat": round(float(latitudes[row]), 4),
                            "lon": round(float(longitudes[row]), 4),
                            "p_none": round(float(probabilities[row, 0]), 6),
                            "p_mild": round(float(probabilities[row, 1]), 6),
                            "p_strong": round(float(probabilities[row, 2]), 6),
                            "p_heavy": round(float(probabilities[row, 3]), 6),
                            "p_strong_or_heavy_within50": round(
                                float(prediction["ordinal_q95"][row]), 6
                            ),
                            "p_heavy_within50": round(float(prediction["ordinal_q99"][row]), 6),
                            "expected_region50_count": round(
                                float(prediction["expected_region50_count"][row]), 3
                            ),
                            "class_code": int(prediction["argmax_code"][row]),
                            "warning_code": int(prediction["warning_code"][row]),
                        }
                    )
                warning_counts = {
                    CLASS_NAMES[index]: int((prediction["warning_code"] == index).sum())
                    for index in range(4)
                }
                periods.append(
                    {
                        "horizon_hours": (horizon_index + 1) * 6,
                        "issue_time_ist": iso_value(item["forecast_issue_time_ist"][0]),
                        "valid_time_ist": iso_value(item["forecast_valid_time_ist"][0]),
                        "warning_counts": warning_counts,
                        "max_strong_or_heavy_probability": round(
                            float(np.nanmax(prediction["ordinal_q95"])), 6
                        ),
                        "max_heavy_probability": round(
                            float(np.nanmax(prediction["ordinal_q99"])), 6
                        ),
                        "cells": cells,
                    }
                )
                if recursive:
                    region_size = np.asarray(neighbor50.sum(axis=1)).ravel() + 1.0
                    local_proxy = (
                        prediction["expected_region50_count"] / np.maximum(region_size, 1.0)
                    ).astype(np.float32)
                    local_proxy = np.nan_to_num(local_proxy, nan=0.0, posinf=0.0, neginf=0.0)
                    lags = [local_proxy, lags[0], lags[1], lags[2]]

            payload = {
                "status": "complete",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "input_file": input_path.name,
                "input_sha256": file_sha256(input_path),
                "recursive_mode": recursive,
                "recursive_history_policy": (
                    "Expected 50 km regional count divided by region cell count is fed as a local-count "
                    "proxy after the first horizon; weather is supplied independently for every horizon."
                    if recursive
                    else "History features are read directly from every input horizon."
                ),
                "warning": (
                    "This 24-hour recursion is a demonstration. The frozen model was trained for one "
                    "six-hour step with observed lightning history, so uncertainty grows after horizon 1."
                ),
                "model": self.describe(),
                "periods": periods,
            }
            temporary = self.output_dir / "latest_forecast.json.tmp"
            final = self.output_dir / "latest_forecast.json"
            temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            temporary.replace(final)
            return payload
