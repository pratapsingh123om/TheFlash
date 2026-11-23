"""Batch frozen +6-hour holdout prediction generator for 2023.

This script executes frozen, independent +6-hour forecasts across all 1,460
six-hour windows of test year 2023 (January 1 through December 31, 2023).

Every 2023 six-hour window is predicted independently from information available
at its own issue time. This reproduces the validated holdout protocol. Recursive
+12/+18/+24 forecasting is a separate live-demo product and must not be mixed
into the retrospective 2023 comparison player.

Run directly in Google Colab (with Google Drive mounted) or on any machine with
access to the native 0.25-degree ERA5 physics panel partitions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

CLASS_NAMES = ("none", "mild", "strong", "heavy")
REQUIRED_MODEL_FILES = (
    "run_configuration.json",
    "locked_storm_category_thresholds.json",
    "alarm_thresholds_selected_on_2022.json",
    "selection_frozen_before_2023.json",
    "training_complete.json",
    "evaluation_summary_2023.json",
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

SEASON_MAP = {
    1: "Winter", 2: "Winter",
    3: "Pre-Monsoon", 4: "Pre-Monsoon", 5: "Pre-Monsoon",
    6: "SW Monsoon", 7: "SW Monsoon", 8: "SW Monsoon", 9: "SW Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon", 12: "Post-Monsoon"
}


def mount_colab_drive() -> None:
    try:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
            print("Google Drive mounted successfully.")
    except ImportError:
        pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def locate_model_directory(root: Path) -> Path:
    root = Path(root)
    candidates = [root]
    if root.exists():
        candidates.extend(path.parent for path in root.rglob("run_configuration.json"))
    valid = [
        candidate
        for candidate in candidates
        if all((candidate / name).is_file() for name in REQUIRED_MODEL_FILES)
    ]
    if not valid:
        raise FileNotFoundError(
            f"No complete deployment model was found under {root}. "
            f"Required files include {REQUIRED_MODEL_FILES[:4]}."
        )
    return max(valid, key=lambda path: (path / "training_complete.json").stat().st_mtime)


class IsotonicCalibrator:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @classmethod
    def load(cls, path: Path) -> "IsotonicCalibrator":
        payload = read_json(path)
        x = np.asarray(payload["x_thresholds"], dtype=np.float64)
        y = np.asarray(payload["y_thresholds"], dtype=np.float64)
        return cls(x=x, y=y)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.clip(np.interp(values, self.x, self.y), 1e-8, 1.0 - 1e-8)


def prediction_iteration_range(model: xgb.Booster) -> dict[str, tuple[int, int]]:
    """Use the same early-stopping tree range as frozen model evaluation."""
    best = model.attr("best_iteration")
    return {"iteration_range": (0, int(best) + 1)} if best is not None else {}


def predict_margin(model: xgb.Booster, matrix: xgb.DMatrix) -> np.ndarray:
    return np.asarray(
        model.predict(
            matrix,
            output_margin=True,
            **prediction_iteration_range(model),
        ),
        dtype=np.float64,
    )


def predict_value(model: xgb.Booster, matrix: xgb.DMatrix) -> np.ndarray:
    return np.asarray(
        model.predict(matrix, **prediction_iteration_range(model)),
        dtype=np.float64,
    )


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


def coherent_probabilities(p_any: np.ndarray, p_q95: np.ndarray, p_q99: np.ndarray) -> np.ndarray:
    p_any = np.clip(p_any, 1e-8, 1.0 - 1e-8)
    p_q95 = np.minimum(np.clip(p_q95, 1e-8, 1.0 - 1e-8), p_any)
    p_q99 = np.minimum(np.clip(p_q99, 1e-8, 1.0 - 1e-8), p_q95)
    probabilities = np.column_stack([1.0 - p_any, p_any - p_q95, p_q95 - p_q99, p_q99])
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)


def history_values(lags: list[np.ndarray], neighbor50: csr_matrix, neighbor100: csr_matrix) -> dict[str, np.ndarray]:
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
        "neighbor50_lag6h_active_cells": sparse_sum(neighbor50, (np.nan_to_num(lag1) > 0).astype(np.float32)),
        "neighbor100_lag6h_sum": n100_lag1,
        "neighbor100_lag12h_sum": n100_lag2,
        "neighbor100_lag24h_sum": sparse_sum(neighbor100, lag4),
        "neighbor100_recent12h_sum": n100_lag1 + n100_lag2,
        "neighbor100_recent24h_sum": (
            n100_lag1 + n100_lag2 + sparse_sum(neighbor100, lag3) + sparse_sum(neighbor100, lag4)
        ),
        "neighbor100_lag6h_active_cells": sparse_sum(neighbor100, (np.nan_to_num(lag1) > 0).astype(np.float32)),
        "neighbor100_propagation_6h": n100_lag1 - n100_lag2,
        "neighbor50_degree": np.asarray(neighbor50.sum(axis=1)).ravel().astype(np.float32),
        "neighbor100_degree": np.asarray(neighbor100.sum(axis=1)).ravel().astype(np.float32),
    }


def run_monthly_recursive_predictions(
    panel_path: Path,
    models: dict[str, Any],
    calibrators: dict[str, IsotonicCalibrator],
    features: list[str],
    alarm_thresholds: dict[str, float],
    recursive: bool = True,
    horizons_per_sequence: int = 4
) -> pl.DataFrame:
    """Run model predictions on a single monthly panel file.

    ``recursive=False`` is the scientifically valid 2023 holdout mode: every
    row retains the observed pre-issue lightning-history features stored in the
    panel. Recursive replacement is retained only for legacy experiments and
    must not be used to create the comparison archive.
    """
    frame = pl.read_parquet(panel_path)
    required_columns = {
        "lightning_count",
        "target_region50_sum",
        "forecast_row_eligible",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise RuntimeError(
            f"{panel_path.name} cannot reproduce frozen evaluation; missing {missing_columns}"
        )
    valid_times = frame.select("forecast_valid_time_ist").unique().sort("forecast_valid_time_ist").to_series().to_list()
    
    feature_pos = {name: idx for idx, name in enumerate(features)}
    
    # Pre-compute spatial graph on first time slice
    sample_slice = frame.filter(pl.col("forecast_valid_time_ist") == valid_times[0]).sort("grid_id")
    grid_ids = sample_slice["grid_id"].to_numpy()
    lats = sample_slice["grid_lat_center"].to_numpy().astype(np.float64)
    lons = sample_slice["grid_lon_center"].to_numpy().astype(np.float64)
    neighbor50 = adjacency(lats, lons, 50.0)
    neighbor100 = adjacency(lats, lons, 100.0)
    region_size = np.asarray(neighbor50.sum(axis=1)).ravel() + 1.0

    output_rows = []
    
    # Chunk valid times into 24-hour sequences (4 six-hour horizons each)
    num_windows = len(valid_times)
    for start_idx in range(0, num_windows, horizons_per_sequence):
        seq_times = valid_times[start_idx : min(start_idx + horizons_per_sequence, num_windows)]
        seq_frame = frame.filter(pl.col("forecast_valid_time_ist").is_in(seq_times)).sort(["forecast_valid_time_ist", "grid_id"])
        
        # Initialize lags from observed history of the first interval
        first_slice = seq_frame.filter(pl.col("forecast_valid_time_ist") == seq_times[0]).sort("grid_id")
        lag1 = first_slice["lightning_lag6h_count"].to_numpy().astype(np.float32)
        lag2 = first_slice["lightning_lag12h_count"].to_numpy().astype(np.float32)
        lag4 = first_slice["lightning_lag24h_count"].to_numpy().astype(np.float32)
        recent24 = first_slice["lightning_recent24h_sum"].to_numpy().astype(np.float32)
        lag3 = recent24 - lag1 - lag2 - lag4
        lags = [lag1, lag2, lag3, lag4]
        
        for h_idx, v_time in enumerate(seq_times):
            h_slice = seq_frame.filter(pl.col("forecast_valid_time_ist") == v_time).sort("grid_id")
            x_vals = h_slice.select(features).to_numpy().astype(np.float32, copy=False)
            
            if recursive and h_idx > 0:
                h_features = history_values(lags, neighbor50, neighbor100)
                for f_name, f_vals in h_features.items():
                    if f_name in feature_pos:
                        x_vals[:, feature_pos[f_name]] = f_vals

            eligible = h_slice["forecast_row_eligible"].to_numpy() == 1
            selected = np.flatnonzero(eligible)
            if not selected.size:
                continue
            selected_x = x_vals[selected]
            dmat = xgb.DMatrix(selected_x, feature_names=features, missing=np.nan)
            
            # Predict ordinal heads
            ord_any = calibrators["any"].predict(predict_margin(models["ordinal_any"], dmat))
            ord_q95 = calibrators["q95"].predict(predict_margin(models["ordinal_q95"], dmat))
            ord_q99 = calibrators["q99"].predict(predict_margin(models["ordinal_q99"], dmat))
            
            probs = coherent_probabilities(ord_any, ord_q95, ord_q99)
            
            # Severity predictions
            sev_mild = np.clip(np.expm1(predict_value(models["severity_mild"], dmat)), 0.0, None)
            sev_strong = np.clip(np.expm1(predict_value(models["severity_strong"], dmat)), 0.0, None)
            sev_heavy = np.clip(np.expm1(predict_value(models["severity_heavy"], dmat)), 0.0, None)
            
            expected_count = probs[:, 1] * sev_mild + probs[:, 2] * sev_strong + probs[:, 3] * sev_heavy
            observed_region50 = (
                h_slice["target_region50_sum"].to_numpy().astype(np.float32)[selected]
            )
            
            # Alarms
            warning_code = np.zeros(len(expected_count), dtype=np.uint8)
            for idx, cname in enumerate(CLASS_NAMES[1:], start=1):
                warning_code[probs[:, idx] >= float(alarm_thresholds[cname])] = idx
                
            # Accumulate predictions
            for prediction_row, grid_position in enumerate(selected):
                output_rows.append({
                    "valid_time_ist": v_time,
                    "horizon_hours": (h_idx + 1) * 6 if recursive else 6,
                    "grid_id": int(grid_ids[grid_position]),
                    "grid_lat": round(float(lats[grid_position]), 4),
                    "grid_lon": round(float(lons[grid_position]), 4),
                    "p_none": round(float(probs[prediction_row, 0]), 5),
                    "p_mild": round(float(probs[prediction_row, 1]), 5),
                    "p_strong": round(float(probs[prediction_row, 2]), 5),
                    "p_heavy": round(float(probs[prediction_row, 3]), 5),
                    "p_strong_or_heavy_50km": round(float(ord_q95[prediction_row]), 5),
                    "p_heavy_50km": round(float(ord_q99[prediction_row]), 5),
                    "expected_region50_count": round(float(expected_count[prediction_row]), 3),
                    "observed_region50_count": int(round(float(observed_region50[prediction_row]))),
                    "warning_code": int(warning_code[prediction_row]),
                })
                
            if recursive:
                expected_full = np.zeros(len(grid_ids), dtype=np.float32)
                expected_full[selected] = expected_count.astype(np.float32)
                local_proxy = (expected_full / np.maximum(region_size, 1.0)).astype(np.float32)
                local_proxy = np.nan_to_num(local_proxy, nan=0.0, posinf=0.0, neginf=0.0)
                lags = [local_proxy, lags[0], lags[1], lags[2]]

    return pl.DataFrame(output_rows)


def append_month_to_comparison_archive(
    archive: zipfile.ZipFile,
    predictions: pl.DataFrame,
    dates: list[str],
    stats: dict[str, dict[str, float]],
) -> None:
    daily = (
        predictions.with_columns(
            pl.col("valid_time_ist").cast(pl.Datetime).dt.date().cast(pl.String).alias("date")
        )
        .group_by(["date", "grid_id", "grid_lat", "grid_lon"])
        .agg(
            pl.col("expected_region50_count").sum().alias("predicted"),
            pl.col("observed_region50_count").sum().alias("observed"),
            pl.col("p_heavy").max().alias("p_heavy"),
            pl.col("warning_code").max().alias("warning_code"),
        )
        .sort(["date", "grid_id"])
    )

    for date_value in daily["date"].unique(maintain_order=True).to_list():
        day = daily.filter(pl.col("date") == date_value).sort("grid_id")
        predicted = day["predicted"].to_numpy().astype(np.float64)
        observed = day["observed"].to_numpy().astype(np.float64)
        cell_count = len(predicted)
        top_count = max(1, int(math.ceil(cell_count * 0.10)))
        predicted_top = set(np.argpartition(predicted, -top_count)[-top_count:].tolist())
        observed_top = set(np.argpartition(observed, -top_count)[-top_count:].tolist())
        top_overlap = len(predicted_top & observed_top) / top_count

        predicted_rank = pl.Series(predicted).rank("average").to_numpy()
        observed_rank = pl.Series(observed).rank("average").to_numpy()
        if np.std(predicted_rank) > 0 and np.std(observed_rank) > 0:
            spatial_spearman = float(np.corrcoef(predicted_rank, observed_rank)[0, 1])
        else:
            spatial_spearman = 0.0

        cells = [
            [
                int(grid_id),
                round(float(lat), 3),
                round(float(lon), 3),
                round(float(pred), 2),
                int(round(float(obs))),
                round(float(p_heavy), 4),
                int(warning),
            ]
            for grid_id, lat, lon, pred, obs, p_heavy, warning in zip(
                day["grid_id"],
                day["grid_lat"],
                day["grid_lon"],
                predicted,
                observed,
                day["p_heavy"],
                day["warning_code"],
            )
        ]
        payload = {
            "date": date_value,
            "target": "daily sum of 50 km regional lightning counts",
            "prediction_mode": "independent_6h_holdout_with_observed_history",
            "aggregation_note": (
                "Totals sum overlapping 50 km regional targets across grid cells; "
                "they are not unique India-wide lightning-event totals."
            ),
            "predicted_total": round(float(predicted.sum()), 2),
            "observed_total": int(round(float(observed.sum()))),
            "spatial_spearman": round(spatial_spearman, 4),
            "top10_exact_overlap": round(top_overlap, 4),
            "predicted_max": round(float(predicted.max()), 2),
            "observed_max": int(round(float(observed.max()))),
            "cells": cells,
        }
        archive.writestr(
            f"daily/{date_value}.json",
            json.dumps(payload, separators=(",", ":")),
        )
        dates.append(date_value)
        stats[date_value] = {
            "predicted_total": payload["predicted_total"],
            "observed_total": payload["observed_total"],
            "spatial_spearman": payload["spatial_spearman"],
            "top10_exact_overlap": payload["top10_exact_overlap"],
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate the frozen independent +6-hour 2023 holdout archive."
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/MINOR_PROJECT/0.25_spatial_models/india_lightning_era5_physics_0p25_forecast_panel"),
        help="Path to ERA5 0.25-degree physics panel directory."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/MINOR_PROJECT/0.25_spatial_models/india_lightning_0p25_spatial_models"),
        help="Path to trained model artifacts directory."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("2023_independent_6h_holdout_results"),
        help="Directory to save generated predictions and figures."
    )
    parser.add_argument(
        "--months",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12",
        help="Comma-separated month integers to process (e.g. '6,7,8,9' for SW monsoon)."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help=(
            "Deprecated for this holdout builder. The paired comparison archive "
            "always uses independent +6-hour predictions with observed history."
        ),
    )
    parser.add_argument(
        "--comparison-zip",
        type=Path,
        default=None,
        help=(
            "Paired daily model-vs-observed archive for the web player. "
            "Defaults to OUTPUT_DIR/india_lightning_2023_comparison_archive_v2.zip."
        ),
    )
    args = parser.parse_args()

    mount_colab_drive()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_comparison_zip = args.comparison_zip or (
        args.output_dir / "india_lightning_2023_comparison_archive_v2.zip"
    )
    final_comparison_zip.parent.mkdir(parents=True, exist_ok=True)
    comparison_zip = final_comparison_zip.with_name(final_comparison_zip.name + ".tmp")
    comparison_zip.unlink(missing_ok=True)

    # Locate model files
    model_dir = locate_model_directory(args.model_dir)
    print(f"Using model directory: {model_dir}")

    print("Loading models and calibrators...")
    config = read_json(model_dir / "run_configuration.json")
    frozen_evaluation = read_json(model_dir / "evaluation_summary_2023.json")
    features = list(config["features"])
    alarm_thresholds = read_json(model_dir / "alarm_thresholds_selected_on_2022.json")["thresholds"]

    models = {}
    for name in ("ordinal_any", "ordinal_q95", "ordinal_q99", "severity_mild", "severity_strong", "severity_heavy"):
        booster = xgb.Booster()
        booster.load_model(model_dir / f"{name}.json")
        models[name] = booster

    calibrators = {
        name: IsotonicCalibrator.load(model_dir / f"isotonic_{name}_selected_on_2022.json")
        for name in ("any", "q95", "q99")
    }

    months = [int(m.strip()) for m in args.months.split(",")]
    if not months or any(month < 1 or month > 12 for month in months):
        raise ValueError(f"Invalid --months value: {args.months}")
    if len(months) != len(set(months)):
        raise ValueError(f"Duplicate months are not allowed: {args.months}")
    if args.recursive:
        print(
            "NOTE: --recursive is ignored. The 2023 comparison must reproduce the "
            "validated independent +6-hour holdout protocol."
        )
    comparison_recursive = False
    print(f"Processing 2023 months: {months} (Independent +6-hour holdout mode)")

    monthly_frames = []
    comparison_dates: list[str] = []
    comparison_stats: dict[str, dict[str, float]] = {}
    with zipfile.ZipFile(
        comparison_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as comparison_archive:
        for month in months:
            panel_file = args.panel_dir / "parts" / f"india_lightning_physics_0p25_forecast_2023_{month:02d}.parquet"
            if not panel_file.exists():
                # Try alternate naming
                alt = list((args.panel_dir / "parts").glob(f"*2023_{month:02d}*.parquet"))
                if alt:
                    panel_file = alt[0]
                else:
                    print(f"Warning: Panel partition for 2023-{month:02d} not found at {panel_file}, skipping.")
                    continue

            print(
                f"\n--- Generating independent +6h predictions for "
                f"2023-{month:02d} ({SEASON_MAP[month]}) ---"
            )
            start_time = datetime.now()
            pred_df = run_monthly_recursive_predictions(
                panel_file,
                models,
                calibrators,
                features,
                alarm_thresholds,
                recursive=comparison_recursive,
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Completed 2023-{month:02d} in {elapsed:.1f}s ({pred_df.height:,} prediction rows)")

            # Save monthly predictions and append paired daily maps.
            out_file = args.output_dir / f"holdout_6h_predictions_2023_{month:02d}.parquet"
            pred_df.write_parquet(out_file, compression="zstd")
            append_month_to_comparison_archive(
                comparison_archive, pred_df, comparison_dates, comparison_stats
            )
            print(f"Saved: {out_file}")

            monthly_frames.append(
                pred_df.group_by(["grid_id", "grid_lat", "grid_lon"]).agg(
                    pl.col("p_heavy").sum().alias("p_heavy_sum"),
                    pl.col("p_strong_or_heavy_50km").sum().alias("p_severe_sum"),
                    pl.len().alias("window_count"),
                    (pl.col("warning_code") >= 2).sum().alias("warning_windows"),
                    pl.col("expected_region50_count").sum().alias("expected_sum"),
                )
            )
            del pred_df

        annual_predicted = float(
            sum(item["predicted_total"] for item in comparison_stats.values())
        )
        annual_observed = float(
            sum(item["observed_total"] for item in comparison_stats.values())
        )
        annual_ratio = annual_predicted / max(annual_observed, 1.0)
        reproduction_check: dict[str, Any] = {
            "generated_predicted_sum": annual_predicted,
            "generated_observed_sum": annual_observed,
            "generated_predicted_to_observed_ratio": annual_ratio,
            "full_year_verified_against_frozen_evaluation": False,
        }

        if set(months) == set(range(1, 13)):
            if len(set(comparison_dates)) != 365:
                raise RuntimeError(
                    f"Full-year archive must contain 365 dates; found "
                    f"{len(set(comparison_dates))}"
                )
            frozen_counts = frozen_evaluation["regional_count_metrics"]
            frozen_predicted = float(frozen_counts["predicted_region50_count_sum"])
            frozen_observed = float(frozen_counts["actual_region50_count_sum"])
            predicted_relative_error = abs(annual_predicted - frozen_predicted) / max(
                frozen_predicted, 1.0
            )
            observed_relative_error = abs(annual_observed - frozen_observed) / max(
                frozen_observed, 1.0
            )
            reproduction_check.update(
                {
                    "frozen_predicted_sum": frozen_predicted,
                    "frozen_observed_sum": frozen_observed,
                    "predicted_relative_error": predicted_relative_error,
                    "observed_relative_error": observed_relative_error,
                    "tolerance": 0.005,
                }
            )
            if predicted_relative_error > 0.005 or observed_relative_error > 0.005:
                raise RuntimeError(
                    "Generated 2023 archive does not reproduce the frozen model "
                    "evaluation within 0.5%. Refusing to publish it. "
                    f"prediction error={predicted_relative_error:.3%}, "
                    f"observation error={observed_relative_error:.3%}."
                )
            reproduction_check["full_year_verified_against_frozen_evaluation"] = True

        manifest = {
            "schema_version": 2,
            "year": 2023,
            "resolution_degrees": 0.25,
            "target": "daily sum of 50 km regional lightning counts",
            "selection": "model and thresholds selected on 2022; 2023 untouched holdout",
            "prediction_mode": "independent_6h_holdout_with_observed_history",
            "forecast_horizon_hours": 6,
            "aggregation_note": (
                "predicted_total and observed_total sum overlapping 50 km regional "
                "targets across cells and are not unique national event totals"
            ),
            "reproduction_check": reproduction_check,
            "dates": sorted(comparison_dates),
            "stats": comparison_stats,
        }
        comparison_archive.writestr(
            "manifest.json", json.dumps(manifest, separators=(",", ":"))
        )

    comparison_zip.replace(final_comparison_zip)
    comparison_zip = final_comparison_zip
    print(f"Paired comparison archive: {comparison_zip}")

    if monthly_frames:
        print("\nAggregating seasonal and annual summaries...")
        all_preds = pl.concat(monthly_frames)
        
        # Grid level seasonal aggregation
        summary = (
            all_preds.group_by(["grid_id", "grid_lat", "grid_lon"])
            .agg(
                pl.col("p_heavy_sum").sum().alias("p_heavy_sum"),
                pl.col("p_severe_sum").sum().alias("p_severe_sum"),
                pl.col("window_count").sum().alias("window_count"),
                pl.col("warning_windows").sum().alias("strong_or_heavy_warning_windows"),
                pl.col("expected_sum").sum().alias("predicted_annual_expected_strikes")
            )
            .with_columns(
                (pl.col("p_heavy_sum") / pl.col("window_count")).alias("mean_p_heavy"),
                (pl.col("p_severe_sum") / pl.col("window_count")).alias("mean_p_severe_50km"),
            )
            .drop(["p_heavy_sum", "p_severe_sum", "window_count"])
            .sort("predicted_annual_expected_strikes", descending=True)
        )
        summary_csv = args.output_dir / "predictions_2023_grid_annual_summary.csv"
        summary.write_csv(summary_csv)
        print(f"Summary written to: {summary_csv}")
        print("\n=======================================================")
        print(" 2023 Full-Year Recursive Forecast Completed!")
        print("=======================================================")


if __name__ == "__main__":
    main()
