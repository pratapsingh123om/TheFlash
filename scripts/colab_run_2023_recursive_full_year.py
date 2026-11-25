"""One-cell Colab launcher for the corrected 2023 +6-hour holdout archive.

Despite this legacy filename, this launcher intentionally does NOT run a
recursive 2023 experiment. It reproduces the frozen independent +6-hour test;
recursion remains limited to the separate today +24-hour demonstration.
"""

from __future__ import annotations

import subprocess
import sys
import hashlib
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

from google.colab import auth, drive


drive.mount("/content/drive", force_remount=False)

REPOSITORY = Path("/content/TheFlash")
if not (REPOSITORY / ".git").is_dir():
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/pratapsingh123om/TheFlash.git",
            str(REPOSITORY),
        ],
        check=True,
    )
else:
    subprocess.run(["git", "-C", str(REPOSITORY), "pull", "--ff-only"], check=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "polars", "xgboost", "scipy"],
    check=True,
)

project = Path("/content/drive/MyDrive/MINOR_PROJECT")
work = project / "0.25_spatial_models"
output = work / "2023_independent_6h_holdout_results"
archive = output / "india_lightning_2023_comparison_archive_v2.zip"


def validate_comparison_archive(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 1000:
        raise FileNotFoundError(f"Comparison archive is absent or empty: {path}")
    with zipfile.ZipFile(path, "r") as source:
        corrupt_member = source.testzip()
        if corrupt_member:
            raise RuntimeError(f"ZIP CRC validation failed at {corrupt_member}")
        names = set(source.namelist())
        if "manifest.json" not in names:
            raise RuntimeError("Comparison ZIP has no manifest.json")
        manifest = json.loads(source.read("manifest.json").decode("utf-8"))
        dates = manifest.get("dates", [])
        if manifest.get("schema_version") != 2:
            raise RuntimeError("Comparison manifest is not schema version 2")
        if manifest.get("year") != 2023 or len(dates) != 365:
            raise RuntimeError(
                f"Comparison manifest must contain all 365 dates in 2023; found {len(dates)}"
            )
        if manifest.get("prediction_mode") != "independent_6h_holdout_with_observed_history":
            raise RuntimeError("Archive does not contain independent +6-hour holdout predictions")
        if not manifest.get("reproduction_check", {}).get(
            "full_year_verified_against_frozen_evaluation", False
        ):
            raise RuntimeError("Frozen-evaluation reproduction check is not marked as passed")
        missing_days = [date for date in dates if f"daily/{date}.json" not in names]
        if missing_days:
            raise RuntimeError(f"Archive is missing daily payloads: {missing_days[:5]}")
    return manifest


force_rebuild = os.environ.get("FORCE_REBUILD_COMPARISON", "0").strip().lower() in {
    "1", "true", "yes"
}
archive_is_valid = False
if archive.is_file() and not force_rebuild:
    try:
        existing_manifest = validate_comparison_archive(archive)
        archive_is_valid = True
        print(
            "Reusing validated Drive archive: "
            f"{archive} ({archive.stat().st_size / 1024**2:.1f} MiB, "
            f"{len(existing_manifest['dates'])} dates)"
        )
    except Exception as exc:
        print(f"Existing archive failed validation and will be rebuilt: {exc}")

# Authentication is needed for the private frozen model bundle and final GCS upload.
auth.authenticate_user()
subprocess.run(["gcloud", "config", "set", "project", "data-mines"], check=True)

# Use the exact immutable deployment bundle, not whichever model directory was
# modified most recently on Drive.
model_bundle_url = (
    "https://storage.googleapis.com/nuclei-lenses/flash-model/"
    "india_lightning_24h_demo_ed5dec2770f7.zip"
)
model_bundle_sha256 = (
    "0b565c63f89be15818920ab6798a46530b0e0cbdfffd6c4a5b7c8dbacf7a39d1"
)
model_bundle = Path("/content/india_lightning_24h_demo_ed5dec2770f7.zip")
model_cache = Path("/content/india_lightning_exact_model_ed5dec2770f7")
model_dir = model_cache / "india_lightning_24h_demo" / "models"

if not archive_is_valid:
    drive_bundle = work / "india_lightning_24h_demo_ed5dec2770f7.zip"
    if not model_bundle.is_file() and drive_bundle.is_file():
        print(f"Found deployment model bundle on Google Drive: {drive_bundle}")
        model_bundle = drive_bundle

    if not model_bundle.is_file() or hashlib.sha256(model_bundle.read_bytes()).hexdigest() != model_bundle_sha256:
        print("Obtaining the exact frozen deployment model bundle...")
        try:
            urllib.request.urlretrieve(model_bundle_url, model_bundle)
        except Exception as exc:
            print(f"Direct URL download failed ({exc}); using authenticated GCS copy...")
            subprocess.run(
                [
                    "gcloud", "storage", "cp",
                    "gs://nuclei-lenses/flash-model/india_lightning_24h_demo_ed5dec2770f7.zip",
                    str(model_bundle),
                ],
                check=True,
            )

    actual_sha256 = hashlib.sha256(model_bundle.read_bytes()).hexdigest()
    if actual_sha256 != model_bundle_sha256:
        raise RuntimeError(
            f"Model bundle checksum mismatch: expected {model_bundle_sha256}, found {actual_sha256}"
        )

    if not (model_dir / "run_configuration.json").is_file():
        print("Extracting checksum-verified deployment model...")
        model_cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(model_bundle, "r") as source:
            source.extractall(model_cache)

    command = [
        sys.executable,
        str(REPOSITORY / "scripts" / "generate_2023_full_year_recursive_predictions.py"),
        "--panel-dir",
        str(work / "india_lightning_era5_physics_0p25_forecast_panel"),
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(output),
        "--comparison-zip",
        str(archive),
    ]

    print("Running corrected independent +6-hour 2023 holdout inference...")
    subprocess.run(command, check=True)

manifest = validate_comparison_archive(archive)
archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
print(
    f"\nArchive validation passed: {len(manifest['dates'])} dates, "
    f"{archive.stat().st_size / 1024**2:.1f} MiB, SHA-256={archive_sha256}"
)

print("\nUploading corrected v2 archive to Google Cloud Storage...")
subprocess.run(
    [
        "gcloud", "storage", "cp",
        str(archive),
        "gs://nuclei-lenses/flash-model/india_lightning_2023_comparison_archive_v2.zip",
    ],
    check=True,
)
subprocess.run(
    [
        "gcloud", "storage", "objects", "describe",
        "gs://nuclei-lenses/flash-model/india_lightning_2023_comparison_archive_v2.zip",
        "--format=value(name,size,updateTime)",
    ],
    check=True,
)

production_url = "https://the-flash-hxuh.vercel.app/api/comparison/dates"
for attempt in range(1, 7):
    try:
        with urllib.request.urlopen(production_url, timeout=60) as response:
            deployed_manifest = json.loads(response.read().decode("utf-8"))
        if len(deployed_manifest.get("dates", [])) != 365:
            raise RuntimeError("Deployed manifest does not expose all 365 dates")
        print("\nSUCCESS: Vercel loaded the corrected archive and exposed all 365 dates.")
        break
    except Exception as exc:
        if attempt == 6:
            raise RuntimeError(
                "Upload succeeded, but production did not load the archive after six checks"
            ) from exc
        print(f"Production check {attempt}/6 not ready yet: {exc}")
        time.sleep(10)
