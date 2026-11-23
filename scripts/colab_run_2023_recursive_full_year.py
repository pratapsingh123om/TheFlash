"""One-cell Colab launcher for the corrected 2023 +6-hour holdout archive.

Despite this legacy filename, this launcher intentionally does NOT run a
recursive 2023 experiment. It reproduces the frozen independent +6-hour test;
recursion remains limited to the separate today +24-hour demonstration.
"""

from __future__ import annotations

import subprocess
import sys
import hashlib
import urllib.request
import zipfile
from pathlib import Path

from google.colab import drive


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

if not model_bundle.is_file() or hashlib.sha256(model_bundle.read_bytes()).hexdigest() != model_bundle_sha256:
    print("Downloading the exact frozen deployment model bundle...")
    urllib.request.urlretrieve(model_bundle_url, model_bundle)

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
print(f"\nREADY TO UPLOAD:\n{archive}")
