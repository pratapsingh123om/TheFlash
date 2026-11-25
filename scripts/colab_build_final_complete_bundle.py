"""Build the final India-lightning bucket ZIP from Google Drive sources.

This Colab script deliberately does not download, reuse, or inspect an older
deployment ZIP.  It validates the exact frozen 0.25-degree model and all twelve
2023 panel partitions, rebuilds the paired 2023 comparison archive, extracts a
four-window replay seed, copies the current application source, and creates one
CRC-checked, SHA-256-addressed final bundle.

No Google Cloud authentication or bucket access is performed.  The finished
ZIP and its receipt remain on Drive and are also offered as a browser download.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from google.colab import drive, files


# ---------------------------------------------------------------------------
# Exact project paths.  These are not searched or guessed.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/content/drive/MyDrive/MINOR_PROJECT")
SPATIAL_ROOT = PROJECT_ROOT / "0.25_spatial_models"
PANEL_DIR = SPATIAL_ROOT / "india_lightning_era5_physics_0p25_forecast_panel"
MODEL_DIR = (
    SPATIAL_ROOT
    / "india_lightning_0p25_spatial_models"
    / "native_0p25_spatial_categories_e0e51e94ad8d"
)
RESULTS_DIR = SPATIAL_ROOT / "2023_independent_6h_holdout_results"
COMPARISON_ZIP = RESULTS_DIR / "india_lightning_2023_comparison_archive_v2.zip"
FINAL_OUTPUT_DIR = SPATIAL_ROOT / "final_deployment_bundle"
DIAGNOSTIC_DIR = FINAL_OUTPUT_DIR / "diagnostics"
REPO_DIR = Path("/content/TheFlash")
REPO_URL = "https://github.com/pratapsingh123om/TheFlash.git"

# The bundled replay is a historical demonstration seed, not a current feed.
DEMO_ISSUE_TIME_IST = "2023-09-29 12:00:00"
FORCE_REBUILD_COMPARISON = True

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

OPTIONAL_MODEL_FILES = (
    "test_2023_spatial_metrics.csv",
    "test_2023_class_metrics.csv",
    "feature_importance.csv",
    "spatial_tolerance_metrics.csv",
)

PANEL_BASE_COLUMNS = {
    "forecast_issue_time_ist",
    "forecast_valid_time_ist",
    "grid_id",
    "grid_lat_center",
    "grid_lon_center",
    "lightning_count",
    "target_region50_sum",
    "forecast_row_eligible",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_visible(command: list[str], log_path: Path, cwd: Path | None = None) -> None:
    """Stream a subprocess while persisting its complete combined output."""
    print("\nCOMMAND:", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
            tail.append(line.rstrip())
            if len(tail) > 250:
                tail.pop(0)
        return_code = process.wait()

    if return_code:
        print("\n" + "=" * 78)
        print(f"SUBPROCESS FAILED WITH EXIT CODE {return_code}")
        print(f"Complete log: {log_path}")
        print("Last output lines:")
        print("\n".join(tail))
        print("=" * 78)
        raise RuntimeError(
            f"Command failed with exit code {return_code}. Full log: {log_path}"
        )


def prepare_repository() -> None:
    if REPO_DIR.exists() and not (REPO_DIR / ".git").is_dir():
        shutil.rmtree(REPO_DIR)
    if (REPO_DIR / ".git").is_dir():
        run_visible(
            ["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
            DIAGNOSTIC_DIR / "git_pull.log",
        )
    else:
        run_visible(
            ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
            DIAGNOSTIC_DIR / "git_clone.log",
        )
    generator = REPO_DIR / "scripts" / "generate_2023_full_year_recursive_predictions.py"
    require(generator.is_file(), f"Generator is missing after repository setup: {generator}")


def install_dependencies() -> None:
    run_visible(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "polars>=1.20,<2",
            "pyarrow>=17,<23",
            "xgboost>=2,<4",
            "scipy>=1.11,<2",
        ],
        DIAGNOSTIC_DIR / "pip_install.log",
    )


def exact_panel_parts() -> dict[int, Path]:
    parts_dir = PANEL_DIR / "parts"
    require(parts_dir.is_dir(), f"Panel parts directory is absent: {parts_dir}")
    result: dict[int, Path] = {}
    for month in range(1, 13):
        matches = sorted(parts_dir.glob(f"*2023_{month:02d}*.parquet"))
        require(
            len(matches) == 1,
            f"Expected exactly one 2023-{month:02d} partition, found "
            f"{len(matches)}: {[str(path) for path in matches]}",
        )
        result[month] = matches[0]
    return result


def validate_model_and_panel() -> tuple[dict[str, Any], dict[int, Path], dict[str, Any]]:
    import polars as pl
    import pyarrow.parquet as pq
    import xgboost as xgb

    require(MODEL_DIR.is_dir(), f"Exact frozen model directory is absent: {MODEL_DIR}")
    missing_model = [name for name in REQUIRED_MODEL_FILES if not (MODEL_DIR / name).is_file()]
    require(not missing_model, f"Frozen model is incomplete; missing: {missing_model}")

    config = read_json(MODEL_DIR / "run_configuration.json")
    features = list(config.get("features", []))
    require(features, "run_configuration.json contains no features")
    require(len(features) == len(set(features)), "Model feature names contain duplicates")
    require(
        config.get("forecast_horizon_hours") == 6,
        f"Expected 6-hour model, found {config.get('forecast_horizon_hours')}",
    )
    require(
        config.get("same_window_weather_included") is False,
        "Model configuration permits same-window weather leakage",
    )
    require(
        config.get("posthoc_multiplier") is False,
        "Model configuration uses a forbidden post-hoc multiplier",
    )

    thresholds = read_json(MODEL_DIR / "alarm_thresholds_selected_on_2022.json").get(
        "thresholds", {}
    )
    require(
        {"mild", "strong", "heavy"}.issubset(thresholds),
        f"Alarm thresholds are incomplete: {sorted(thresholds)}",
    )

    evaluation = read_json(MODEL_DIR / "evaluation_summary_2023.json")
    regional = evaluation.get("regional_count_metrics", {})
    require(
        {
            "predicted_region50_count_sum",
            "actual_region50_count_sum",
        }.issubset(regional),
        "evaluation_summary_2023.json lacks frozen regional-count totals",
    )

    # Actually load every trained XGBoost head and inspect every calibrator.
    for stem in (
        "ordinal_any",
        "ordinal_q95",
        "ordinal_q99",
        "severity_mild",
        "severity_strong",
        "severity_heavy",
    ):
        booster = xgb.Booster()
        booster.load_model(MODEL_DIR / f"{stem}.json")
        require(booster.num_boosted_rounds() > 0, f"Empty XGBoost model: {stem}.json")
    for name in ("any", "q95", "q99"):
        payload = read_json(MODEL_DIR / f"isotonic_{name}_selected_on_2022.json")
        x_values = payload.get("x_thresholds", [])
        y_values = payload.get("y_thresholds", [])
        require(
            len(x_values) >= 2 and len(x_values) == len(y_values),
            f"Invalid isotonic_{name} calibrator",
        )

    parts = exact_panel_parts()
    required_columns = PANEL_BASE_COLUMNS | set(features)
    panel_report: list[dict[str, Any]] = []
    total_rows = 0
    reference_schema: set[str] | None = None
    for month, path in parts.items():
        require(path.stat().st_size > 10_000, f"Panel partition is unexpectedly small: {path}")
        metadata = pq.ParquetFile(path).metadata
        schema_names = set(pl.scan_parquet(path).collect_schema().names())
        missing_columns = sorted(required_columns - schema_names)
        require(
            not missing_columns,
            f"{path.name} is missing {len(missing_columns)} required columns: "
            f"{missing_columns[:30]}",
        )
        if reference_schema is None:
            reference_schema = schema_names
        else:
            require(
                schema_names == reference_schema,
                f"{path.name} schema differs from January; "
                f"missing={sorted(reference_schema - schema_names)[:20]}, "
                f"extra={sorted(schema_names - reference_schema)[:20]}",
            )
        # Force an actual data-page read, not just a metadata check.
        sample = pl.read_parquet(
            path,
            columns=[
                "forecast_issue_time_ist",
                "forecast_valid_time_ist",
                "grid_id",
                "forecast_row_eligible",
                "target_region50_sum",
            ],
            n_rows=16,
        )
        require(sample.height > 0, f"Panel partition has no readable rows: {path}")
        total_rows += metadata.num_rows
        panel_report.append(
            {
                "month": month,
                "path": str(path),
                "bytes": path.stat().st_size,
                "rows": metadata.num_rows,
                "row_groups": metadata.num_row_groups,
                "columns": len(schema_names),
            }
        )

    report = {
        "status": "preflight_passed",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(MODEL_DIR),
        "model_feature_count": len(features),
        "panel_dir": str(PANEL_DIR),
        "panel_partitions": panel_report,
        "total_2023_panel_rows": total_rows,
    }
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    (DIAGNOSTIC_DIR / "preflight_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return config, parts, report


def validate_comparison_archive(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Comparison ZIP was not created: {path}")
    require(path.stat().st_size > 1000, f"Comparison ZIP is unexpectedly small: {path}")
    with zipfile.ZipFile(path, "r") as archive:
        corrupt = archive.testzip()
        require(corrupt is None, f"Comparison ZIP CRC failed at: {corrupt}")
        names = set(archive.namelist())
        require("manifest.json" in names, "Comparison ZIP has no manifest.json")
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        dates = manifest.get("dates", [])
        require(manifest.get("schema_version") == 2, "Comparison schema_version is not 2")
        require(manifest.get("year") == 2023, "Comparison year is not 2023")
        require(len(dates) == 365, f"Expected 365 comparison dates, found {len(dates)}")
        require(
            manifest.get("prediction_mode")
            == "independent_6h_holdout_with_observed_history",
            f"Incorrect comparison prediction mode: {manifest.get('prediction_mode')}",
        )
        require(
            manifest.get("reproduction_check", {}).get(
                "full_year_verified_against_frozen_evaluation"
            )
            is True,
            "Frozen full-year reproduction check did not pass",
        )
        missing_days = [date for date in dates if f"daily/{date}.json" not in names]
        require(not missing_days, f"Comparison ZIP is missing days: {missing_days[:10]}")
        sample = json.loads(archive.read(f"daily/{dates[0]}.json").decode("utf-8"))
        required_fields = {
            "date",
            "predicted_total",
            "observed_total",
            "spatial_spearman",
            "top10_exact_overlap",
            "cells",
        }
        require(
            required_fields.issubset(sample),
            f"Daily comparison payload lacks: {sorted(required_fields - set(sample))}",
        )
        require(len(sample["cells"]) > 4000, "Daily payload contains too few grid cells")
    return manifest


def rebuild_comparison_archive() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if FORCE_REBUILD_COMPARISON:
        COMPARISON_ZIP.unlink(missing_ok=True)
        COMPARISON_ZIP.with_name(COMPARISON_ZIP.name + ".tmp").unlink(missing_ok=True)
    generator = REPO_DIR / "scripts" / "generate_2023_full_year_recursive_predictions.py"
    command = [
        sys.executable,
        str(generator),
        "--panel-dir",
        str(PANEL_DIR),
        "--model-dir",
        str(MODEL_DIR),
        "--output-dir",
        str(RESULTS_DIR),
        "--comparison-zip",
        str(COMPARISON_ZIP),
        "--months",
        "1,2,3,4,5,6,7,8,9,10,11,12",
    ]
    run_visible(command, DIAGNOSTIC_DIR / "comparison_generator.log")
    return validate_comparison_archive(COMPARISON_ZIP)


def build_replay_seed(
    config: dict[str, Any], parts: dict[int, Path], destination: Path
) -> dict[str, Any]:
    import polars as pl

    issue_start = datetime.fromisoformat(DEMO_ISSUE_TIME_IST)
    issue_times = [issue_start + timedelta(hours=6 * index) for index in range(4)]
    features = list(config["features"])
    columns = list(
        dict.fromkeys(
            [
                "forecast_issue_time_ist",
                "forecast_valid_time_ist",
                "grid_id",
                "grid_lat_center",
                "grid_lon_center",
                *features,
            ]
        )
    )
    frame = (
        pl.scan_parquet([str(path) for path in parts.values()])
        .filter(pl.col("forecast_issue_time_ist").is_in(issue_times))
        .select(columns)
        .collect()
        .sort(["forecast_valid_time_ist", "grid_id"])
    )
    require(frame.height > 0, "Replay extraction returned zero rows")
    require(frame["forecast_issue_time_ist"].n_unique() == 4, "Replay has fewer than 4 issue times")
    require(frame["forecast_valid_time_ist"].n_unique() == 4, "Replay has fewer than 4 valid times")
    duplicates = frame.select(
        pl.struct(["forecast_valid_time_ist", "grid_id"]).is_duplicated().sum()
    ).item()
    require(duplicates == 0, f"Replay input has {duplicates} duplicate time/grid rows")
    rows_by_time = frame.group_by("forecast_valid_time_ist").len().sort("forecast_valid_time_ist")
    require(rows_by_time["len"].n_unique() == 1, "Replay windows have inconsistent grid sizes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(destination, compression="zstd", compression_level=5, statistics=True)
    return {
        "issue_start_ist": DEMO_ISSUE_TIME_IST,
        "issue_times": [value.isoformat(sep=" ") for value in issue_times],
        "rows": frame.height,
        "grid_cells_per_window": int(rows_by_time["len"][0]),
        "target_columns_packaged": False,
        "sha256": sha256(destination),
    }


def copy_application(destination: Path) -> None:
    ignored = shutil.ignore_patterns(
        ".git",
        ".pytest_cache",
        "__pycache__",
        "*.pyc",
        "*.zip",
        "*.parquet",
        ".env",
        ".env.production",
        ".venv",
        "runtime",
        "runtime-assets",
        "models",
        "data",
    )
    shutil.copytree(REPO_DIR, destination, ignore=ignored)
    (destination / "models").mkdir(parents=True, exist_ok=True)
    (destination / "data").mkdir(parents=True, exist_ok=True)
    (destination / "runtime").mkdir(parents=True, exist_ok=True)


def copy_model_artifacts(destination: Path) -> None:
    for name in REQUIRED_MODEL_FILES:
        shutil.copy2(MODEL_DIR / name, destination / name)
    for name in OPTIONAL_MODEL_FILES:
        source = MODEL_DIR / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def make_final_zip(package_root: Path, output_path: Path) -> None:
    temporary = output_path.with_suffix(".zip.tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        for path in sorted(package_root.rglob("*")):
            if not path.is_file():
                continue
            archive_name = (Path(package_root.name) / path.relative_to(package_root)).as_posix()
            compression = (
                zipfile.ZIP_STORED
                if path.suffix.lower() in {".zip", ".parquet"}
                else zipfile.ZIP_DEFLATED
            )
            archive.write(path, archive_name, compress_type=compression)
    temporary.replace(output_path)


def validate_final_zip(path: Path, comparison_sha: str) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 1000, f"Final ZIP is absent/empty: {path}")
    prefix = "india_lightning_24h_demo/"
    with zipfile.ZipFile(path, "r") as archive:
        corrupt = archive.testzip()
        require(corrupt is None, f"Final ZIP CRC failed at: {corrupt}")
        names = set(archive.namelist())
        required = {
            prefix + "backend/app.py",
            prefix + "frontend/index.html",
            prefix + "frontend/reference-theme.css",
            prefix + "models/run_configuration.json",
            prefix + "data/next_24h_features.parquet",
            prefix + "data/india_lightning_2023_comparison_archive_v2.zip",
            prefix + "bundle_manifest.json",
        }
        require(required.issubset(names), f"Final ZIP is missing: {sorted(required - names)}")
        embedded = archive.read(prefix + "data/india_lightning_2023_comparison_archive_v2.zip")
        require(
            hashlib.sha256(embedded).hexdigest() == comparison_sha,
            "Embedded comparison archive checksum changed during packaging",
        )
    return {
        "status": "complete",
        "zip": str(path),
        "bytes": path.stat().st_size,
        "size_mib": round(path.stat().st_size / 1024**2, 2),
        "sha256": sha256(path),
    }


def build() -> tuple[Path, Path]:
    drive.mount("/content/drive", force_remount=False)
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    prepare_repository()
    install_dependencies()

    print("\n" + "=" * 78)
    print("FINAL BUNDLE PREFLIGHT")
    print("Model :", MODEL_DIR)
    print("Panel :", PANEL_DIR)
    print("Output:", FINAL_OUTPUT_DIR)
    print("=" * 78)
    config, parts, preflight = validate_model_and_panel()
    print(
        f"PREFLIGHT PASSED: {len(config['features'])} features, "
        f"{len(parts)} partitions, {preflight['total_2023_panel_rows']:,} panel rows"
    )

    manifest = rebuild_comparison_archive()
    comparison_sha = sha256(COMPARISON_ZIP)
    print(
        f"COMPARISON PASSED: {len(manifest['dates'])} days, "
        f"{COMPARISON_ZIP.stat().st_size / 1024**2:.2f} MiB, SHA256={comparison_sha}"
    )

    with tempfile.TemporaryDirectory(prefix="india_lightning_final_bundle_") as temporary:
        package_root = Path(temporary) / "india_lightning_24h_demo"
        copy_application(package_root)
        copy_model_artifacts(package_root / "models")
        replay = build_replay_seed(
            config, parts, package_root / "data" / "next_24h_features.parquet"
        )
        shutil.copy2(
            COMPARISON_ZIP,
            package_root / "data" / "india_lightning_2023_comparison_archive_v2.zip",
        )

        bundle_manifest = {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "package_name": package_root.name,
            "source_repository": REPO_URL,
            "source_commit": subprocess.check_output(
                ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_model": str(MODEL_DIR),
            "model_feature_count": len(config["features"]),
            "forecast_horizon_hours": 6,
            "recursive_demo_horizons": [6, 12, 18, 24],
            "same_window_weather_included": False,
            "posthoc_multiplier": False,
            "comparison": {
                "path": "data/india_lightning_2023_comparison_archive_v2.zip",
                "mode": manifest["prediction_mode"],
                "days": len(manifest["dates"]),
                "sha256": comparison_sha,
            },
            "replay_seed": replay,
        }
        (package_root / "bundle_manifest.json").write_text(
            json.dumps(bundle_manifest, indent=2), encoding="utf-8"
        )

        descriptor = {
            "model": MODEL_DIR.name,
            "config_sha256": sha256(MODEL_DIR / "run_configuration.json"),
            "comparison_sha256": comparison_sha,
            "replay_sha256": replay["sha256"],
            "commit": bundle_manifest["source_commit"],
        }
        signature = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        output_zip = FINAL_OUTPUT_DIR / f"india_lightning_final_complete_{signature}.zip"
        make_final_zip(package_root, output_zip)

    receipt = validate_final_zip(output_zip, comparison_sha)
    receipt.update(
        {
            "model_dir": str(MODEL_DIR),
            "panel_dir": str(PANEL_DIR),
            "comparison_zip": str(COMPARISON_ZIP),
            "comparison_sha256": comparison_sha,
            "diagnostics": str(DIAGNOSTIC_DIR),
        }
    )
    receipt_path = output_zip.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("FINAL INDIA LIGHTNING ZIP READY")
    print("ZIP        :", output_zip)
    print("Size MiB   :", receipt["size_mib"])
    print("SHA-256    :", receipt["sha256"])
    print("Receipt    :", receipt_path)
    print("Diagnostics:", DIAGNOSTIC_DIR)
    print("Cloud access/upload performed: NO")
    print("=" * 78)
    return output_zip, receipt_path


try:
    FINAL_ZIP, FINAL_RECEIPT = build()
except Exception as exc:
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failure_path = FINAL_OUTPUT_DIR / "final_bundle_failure.json"
    failure = {
        "status": "failed",
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "model_dir": str(MODEL_DIR),
        "panel_dir": str(PANEL_DIR),
        "comparison_zip": str(COMPARISON_ZIP),
        "diagnostics": str(DIAGNOSTIC_DIR),
    }
    failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print("\nFINAL BUNDLE BUILD FAILED")
    print("Exact failure report:", failure_path)
    print(failure["traceback"])
    raise
else:
    print("\nStarting browser download of the verified final ZIP...")
    files.download(str(FINAL_ZIP))

