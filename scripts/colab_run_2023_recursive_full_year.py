# =====================================================================================
# ONE-CELL GOOGLE COLAB SCRIPT: 2023 FULL-YEAR RECURSIVE LIGHTNING FORECASTER & SHOWCASE
# =====================================================================================
# This script executes the complete multi-horizon recursive forecasting loop across all
# 1,460 six-hour windows of 2023 (Jan 1 to Dec 31, 2023) using the native 0.25-degree model.
#
# Paths are pre-configured to match your Google Drive project structure.
# Simply paste and run this entire block in a Google Colab notebook cell.
# =====================================================================================

from __future__ import annotations

import os
import sys
import gc
import json
import math
import time
import subprocess
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# -------------------------------------------------------------------------------------
# 1. MOUNT GOOGLE DRIVE & AUTO-INSTALL PACKAGES
# -------------------------------------------------------------------------------------
try:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
        print("✓ Google Drive mounted successfully.")
except ImportError:
    pass

def ensure_dependencies():
    packages = {
        "polars": "polars",
        "xgboost": "xgboost",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "numpy": "numpy"
    }
    missing = [pip for mod, pip in packages.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"Installing missing packages: {missing}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        print("✓ Dependencies installed.")

ensure_dependencies()

import numpy as np
import polars as pl
import xgboost as xgb
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------
# 2. EXACT GOOGLE DRIVE PROJECT PATHS (Pre-configured from previous scripts)
# -------------------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("MINOR_PROJECT_ROOT", "/content/drive/MyDrive/MINOR_PROJECT"))
WORK_DIR     = PROJECT_ROOT / "0.25_spatial_models"

# 60 monthly ERA5 physics panel partitions (106 features at 0.25 deg)
PANEL_DIR    = Path(os.environ.get("PANEL_DIR", str(WORK_DIR / "india_lightning_era5_physics_0p25_forecast_panel")))

# Trained model artifacts directory
MODEL_ROOT   = Path(os.environ.get("MODEL_ROOT", str(WORK_DIR / "india_lightning_0p25_spatial_models")))

# Verified observed WWLLN stroke events
OBSERVED_DIR = Path(os.environ.get("OBSERVED_DIR", str(WORK_DIR / "india_lightning_events_exact_0p25_v2")))

# Output directory for 2023 predictions and showcase maps
OUTPUT_DIR   = Path(os.environ.get("OUTPUT_DIR", str(PROJECT_ROOT / "2023_full_year_recursive_results")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Select months to run: default is all 12 months (pass e.g. [6, 7, 8, 9] for SW Monsoon only)
MONTHS_TO_RUN = list(range(1, 13))

REQUIRED_MODEL_FILES = (
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

CLASS_NAMES = ("none", "mild", "strong", "heavy")
SEASON_MAP  = {
    1: "Winter", 2: "Winter",
    3: "Pre-Monsoon", 4: "Pre-Monsoon", 5: "Pre-Monsoon",
    6: "SW Monsoon", 7: "SW Monsoon", 8: "SW Monsoon", 9: "SW Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon", 12: "Post-Monsoon"
}

# -------------------------------------------------------------------------------------
# 3. HELPER FUNCTIONS & CALIBRATOR CLASS
# -------------------------------------------------------------------------------------
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def locate_model_directory() -> Path:
    """Find the latest complete native 0.25-degree model directory."""
    if (MODEL_ROOT / "run_configuration.json").exists():
        return MODEL_ROOT
    candidates = []
    for d in MODEL_ROOT.glob("native_0p25_spatial_categories_*"):
        if all((d / f).exists() for f in REQUIRED_MODEL_FILES):
            candidates.append(d)
    if not candidates:
        # Check fallback search paths
        for d in PROJECT_ROOT.rglob("native_0p25_spatial_categories_*"):
            if all((d / f).exists() for f in REQUIRED_MODEL_FILES):
                candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"Could not locate frozen model artifacts below {MODEL_ROOT}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

class IsotonicCalibrator:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @classmethod
    def load(cls, path: Path) -> "IsotonicCalibrator":
        d = read_json(path)
        return cls(np.asarray(d["x_thresholds"], dtype=np.float64), np.asarray(d["y_thresholds"], dtype=np.float64))

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.clip(np.interp(values, self.x, self.y), 1e-8, 1.0 - 1e-8)

def adjacency(latitudes: np.ndarray, longitudes: np.ndarray, radius_km: float) -> csr_matrix:
    lat = np.radians(np.asarray(latitudes, dtype=np.float64))
    lon = np.radians(np.asarray(longitudes, dtype=np.float64))
    xyz = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    chord = 2.0 * math.sin(radius_km / (2.0 * 6371.0088))
    groups = cKDTree(xyz).query_ball_point(xyz, r=chord + 1e-12)
    rows, cols = [], []
    for r_idx, grp in enumerate(groups):
        rows.extend([r_idx] * len(grp))
        cols.extend(grp)
    return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(lat), len(lat)), dtype=np.float32)

def sparse_sum(mat: csr_matrix, vals: np.ndarray) -> np.ndarray:
    return np.asarray(mat.dot(np.nan_to_num(vals, nan=0.0))).ravel().astype(np.float32)

def coherent_probabilities(p_any: np.ndarray, p_q95: np.ndarray, p_q99: np.ndarray) -> np.ndarray:
    p_any = np.clip(p_any, 1e-8, 1.0 - 1e-8)
    p_q95 = np.minimum(np.clip(p_q95, 1e-8, 1.0 - 1e-8), p_any)
    p_q99 = np.minimum(np.clip(p_q99, 1e-8, 1.0 - 1e-8), p_q95)
    probs = np.column_stack([1.0 - p_any, p_any - p_q95, p_q95 - p_q99, p_q99])
    probs = np.clip(probs, 0.0, 1.0)
    return probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)

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
        "neighbor100_recent24h_sum": (n100_lag1 + n100_lag2 + sparse_sum(neighbor100, lag3) + sparse_sum(neighbor100, lag4)),
        "neighbor100_lag6h_active_cells": sparse_sum(neighbor100, (np.nan_to_num(lag1) > 0).astype(np.float32)),
        "neighbor100_propagation_6h": n100_lag1 - n100_lag2,
        "neighbor50_degree": np.asarray(neighbor50.sum(axis=1)).ravel().astype(np.float32),
        "neighbor100_degree": np.asarray(neighbor100.sum(axis=1)).ravel().astype(np.float32),
    }

# -------------------------------------------------------------------------------------
# 4. RECURSIVE ENGINE (4-Horizon 24-hour cycles across all monthly windows)
# -------------------------------------------------------------------------------------
def run_recursive_month(
    panel_path: Path,
    models: dict[str, Any],
    calibrators: dict[str, IsotonicCalibrator],
    features: list[str],
    alarm_thresholds: dict[str, float]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    frame = pl.read_parquet(panel_path)
    valid_times = frame.select("forecast_valid_time_ist").unique().sort("forecast_valid_time_ist").to_series().to_list()
    feature_pos = {name: idx for idx, name in enumerate(features)}

    sample_slice = frame.filter(pl.col("forecast_valid_time_ist") == valid_times[0]).sort("grid_id")
    grid_ids = sample_slice["grid_id"].to_numpy()
    lats = sample_slice["grid_lat_center"].to_numpy().astype(np.float64)
    lons = sample_slice["grid_lon_center"].to_numpy().astype(np.float64)
    neighbor50 = adjacency(lats, lons, 50.0)
    neighbor100 = adjacency(lats, lons, 100.0)
    region_size = np.asarray(neighbor50.sum(axis=1)).ravel() + 1.0

    # Accumulators for month-level grid summary to save RAM
    grid_expected_sum = np.zeros(len(grid_ids), dtype=np.float64)
    grid_p_heavy_max  = np.zeros(len(grid_ids), dtype=np.float32)
    grid_p_q95_max    = np.zeros(len(grid_ids), dtype=np.float32)
    grid_warning_hits = np.zeros(len(grid_ids), dtype=np.int32)
    daily_rows = []

    # Process in 4-step rolling cycles (each sequence represents +6h, +12h, +18h, +24h recursion)
    step_size = 4
    for s_idx in range(0, len(valid_times), step_size):
        seq_times = valid_times[s_idx : min(s_idx + step_size, len(valid_times))]
        seq_frame = frame.filter(pl.col("forecast_valid_time_ist").is_in(seq_times)).sort(["forecast_valid_time_ist", "grid_id"])

        # Horizon 1 initializes from observed history
        first_slice = seq_frame.filter(pl.col("forecast_valid_time_ist") == seq_times[0]).sort("grid_id")
        lag1 = first_slice["lightning_lag6h_count"].to_numpy().astype(np.float32)
        lag2 = first_slice["lightning_lag12h_count"].to_numpy().astype(np.float32)
        lag4 = first_slice["lightning_lag24h_count"].to_numpy().astype(np.float32)
        recent24 = first_slice["lightning_recent24h_sum"].to_numpy().astype(np.float32)
        lag3 = recent24 - lag1 - lag2 - lag4
        lags = [lag1, lag2, lag3, lag4]

        for h_idx, v_time in enumerate(seq_times):
            h_slice = seq_frame.filter(pl.col("forecast_valid_time_ist") == v_time).sort("grid_id")
            x_vals  = h_slice.select(features).to_numpy().astype(np.float32, copy=False)

            # Horizons 2, 3, 4 dynamically update lightning lag graph features
            if h_idx > 0:
                h_feats = history_values(lags, neighbor50, neighbor100)
                for f_name, f_vals in h_feats.items():
                    if f_name in feature_pos:
                        x_vals[:, feature_pos[f_name]] = f_vals

            dmat = xgb.DMatrix(x_vals, feature_names=features, missing=np.nan)

            # Predict ordinal heads
            ord_any = calibrators["any"].predict(np.asarray(models["ordinal_any"].predict(dmat, output_margin=True), dtype=np.float64))
            ord_q95 = calibrators["q95"].predict(np.asarray(models["ordinal_q95"].predict(dmat, output_margin=True), dtype=np.float64))
            ord_q99 = calibrators["q99"].predict(np.asarray(models["ordinal_q99"].predict(dmat, output_margin=True), dtype=np.float64))
            probs   = coherent_probabilities(ord_any, ord_q95, ord_q99)

            # Severity heads
            sev_mild   = np.clip(np.expm1(models["severity_mild"].predict(dmat)), 0.0, None)
            sev_strong = np.clip(np.expm1(models["severity_strong"].predict(dmat)), 0.0, None)
            sev_heavy  = np.clip(np.expm1(models["severity_heavy"].predict(dmat)), 0.0, None)
            expected   = probs[:, 1] * sev_mild + probs[:, 2] * sev_strong + probs[:, 3] * sev_heavy

            # Alarms
            is_warning = (probs[:, 2] >= alarm_thresholds["strong"]) | (probs[:, 3] >= alarm_thresholds["heavy"])
            
            # Accumulate statistics
            grid_expected_sum += expected
            grid_p_heavy_max   = np.maximum(grid_p_heavy_max, probs[:, 3])
            grid_p_q95_max     = np.maximum(grid_p_q95_max, ord_q95)
            grid_warning_hits += is_warning.astype(np.int32)

            daily_rows.append({
                "valid_time_ist": str(v_time),
                "horizon_hours": (h_idx + 1) * 6,
                "predicted_india_strikes": float(expected.sum()),
                "heavy_warning_cells": int(is_warning.sum()),
                "max_heavy_probability": float(probs[:, 3].max())
            })

            # Recursive step: update local proxy and push into lags
            local_proxy = (expected / np.maximum(region_size, 1.0)).astype(np.float32)
            local_proxy = np.nan_to_num(local_proxy, nan=0.0, posinf=0.0, neginf=0.0)
            lags = [local_proxy, lags[0], lags[1], lags[2]]

    grid_summary = pl.DataFrame({
        "grid_id": grid_ids,
        "grid_lat": lats,
        "grid_lon": lons,
        "predicted_strokes": grid_expected_sum,
        "max_p_heavy": grid_p_heavy_max,
        "max_p_severe_50km": grid_p_q95_max,
        "warning_windows": grid_warning_hits
    })
    return grid_summary, pl.DataFrame(daily_rows)

# -------------------------------------------------------------------------------------
# 5. EXECUTION & PRESENTATION VISUALIZATIONS
# -------------------------------------------------------------------------------------
def main():
    print("==========================================================================")
    print("      2023 FULL-YEAR RECURSIVE LIGHTNING FORECASTING PIPELINE (0.25°)")
    print("==========================================================================")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Panel dir    : {PANEL_DIR}")
    print(f"Output dir   : {OUTPUT_DIR}")

    model_dir = locate_model_directory()
    print(f"Model dir    : {model_dir}\n")

    config = read_json(model_dir / "run_configuration.json")
    features = list(config["features"])
    alarm_thresholds = read_json(model_dir / "alarm_thresholds_selected_on_2022.json")["thresholds"]

    print("Loading frozen boosters and calibrators...")
    models = {}
    for name in ("ordinal_any", "ordinal_q95", "ordinal_q99", "severity_mild", "severity_strong", "severity_heavy"):
        booster = xgb.Booster()
        booster.load_model(model_dir / f"{name}.json")
        models[name] = booster

    calibrators = {
        name: IsotonicCalibrator.load(model_dir / f"isotonic_{name}_selected_on_2022.json")
        for name in ("any", "q95", "q99")
    }
    print("✓ All 6 XGBoost models and isotonic calibrators loaded.\n")

    seasonal_preds = {"Winter": [], "Pre-Monsoon": [], "SW Monsoon": [], "Post-Monsoon": []}
    seasonal_obs   = {"Winter": [], "Pre-Monsoon": [], "SW Monsoon": [], "Post-Monsoon": []}
    monthly_rows   = []

    for month in MONTHS_TO_RUN:
        panel_file = Path(PANEL_DIR) / "parts" / f"india_lightning_physics_0p25_forecast_2023_{month:02d}.parquet"
        if not panel_file.exists():
            panel_file = Path(PANEL_DIR) / f"india_lightning_physics_0p25_forecast_2023_{month:02d}.parquet"
        if not panel_file.exists():
            matches = list(Path(PANEL_DIR).rglob(f"*2023_{month:02d}*.parquet"))
            if matches:
                panel_file = matches[0]
            else:
                print(f"⚠ Warning: Panel for 2023-{month:02d} not found, skipping.")
                continue

        season = SEASON_MAP[month]
        print(f"Running recursive forecast for 2023-{month:02d} ({season})...")
        t0 = time.time()
        grid_pred, window_df = run_recursive_month(panel_file, models, calibrators, features, alarm_thresholds)
        elapsed = time.time() - t0
        print(f"  ✓ Finished in {elapsed:.1f}s | Predicted strikes: {grid_pred['predicted_strokes'].sum():,.0f}")

        # Save monthly summary
        grid_pred.write_parquet(OUTPUT_DIR / f"grid_summary_2023_{month:02d}.parquet")
        seasonal_preds[season].append(grid_pred)

        # Load observed strokes if available for comparison
        obs_file = OBSERVED_DIR / f"india_lightning_exact_0p25_2023_{month:02d}.parquet"
        if obs_file.exists():
            df_obs = pl.scan_parquet(obs_file).select(["grid_id", "grid_lat_center", "grid_lon_center"]).collect()
            obs_agg = df_obs.group_by(["grid_id", "grid_lat_center", "grid_lon_center"]).agg(pl.len().alias("observed_strokes"))
            seasonal_obs[season].append(obs_agg)
            monthly_rows.append({
                "month": month,
                "season": season,
                "predicted_strokes": float(grid_pred["predicted_strokes"].sum()),
                "observed_strokes": df_obs.height,
                "active_predicted_cells": int((grid_pred["predicted_strokes"] > 1.0).sum()),
                "active_observed_cells": df_obs["grid_id"].n_unique()
            })

        gc.collect()

    # Save Monthly & Seasonal Performance Report
    if monthly_rows:
        summary_table = pl.DataFrame(monthly_rows)
        summary_table.write_csv(OUTPUT_DIR / "2023_monthly_predicted_vs_observed_summary.csv")
        print("\n==========================================================================")
        print("          2023 MONTHLY PREDICTED VS OBSERVED SUMMARY TABLE")
        print("==========================================================================")
        print(summary_table)

    # ---------------------------------------------------------------------------------
    # Generate Presentation 4-Panel Seasonal Map (SW Monsoon Highlights All-India Convection)
    # ---------------------------------------------------------------------------------
    print("\nGenerating presentation-ready 4-panel seasonal comparison maps...")
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.5), constrained_layout=True, sharex=True, sharey=True)
    seasons = ["Winter", "Pre-Monsoon", "SW Monsoon", "Post-Monsoon"]

    for ax, s_name in zip(axes, seasons):
        if not seasonal_preds[s_name]:
            continue
        comb_pred = pl.concat(seasonal_preds[s_name]).group_by(["grid_id", "grid_lat", "grid_lon"]).agg(pl.col("predicted_strokes").sum())
        lons = comb_pred["grid_lon"].to_numpy()
        lats = comb_pred["grid_lat"].to_numpy()
        strikes = comb_pred["predicted_strokes"].to_numpy()
        log_s = np.log10(np.maximum(strikes, 1))

        sc = ax.scatter(lons, lats, c=log_s, cmap="inferno", s=8, vmin=0, vmax=4.5)
        ax.set_title(f"{s_name} (2023)\nPred strikes: {strikes.sum():,.0f} | Active: {(strikes > 1).sum():,} cells", fontsize=11, fontweight="bold")
        ax.set_xlim(67.5, 98.5)
        ax.set_ylim(6.0, 37.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Latitude (°N)", fontsize=9)

    cb = fig.colorbar(sc, ax=axes, orientation="horizontal", shrink=0.5, pad=0.08)
    cb.set_label("Log10(Predicted Seasonal Strikes per 0.25° Grid Cell) [1 = 10, 2 = 100, 3 = 1,000, 4 = 10,000+]", fontsize=10)
    fig.suptitle("Model-Predicted 2023 India Lightning Distribution Across All Four Seasons (Recursive 0.25° Model)", fontsize=14, fontweight="bold")

    fig_path = OUTPUT_DIR / "india_2023_predicted_seasonal_lightning.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved seasonal map to: {fig_path}")

    # Display in Colab
    try:
        from IPython.display import Image as IPythonImage, display
        display(IPythonImage(filename=str(fig_path)))
    except ImportError:
        pass

    print("\n==========================================================================")
    print(" ✓ 2023 FULL-YEAR RECURSIVE RUN COMPLETE!")
    print(f" All outputs and figures saved to: {OUTPUT_DIR}")
    print("==========================================================================")

if __name__ == "__main__":
    main()
