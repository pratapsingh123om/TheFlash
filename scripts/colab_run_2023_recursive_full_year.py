"""One-cell Colab launcher for the corrected 2023 +6-hour holdout archive.

Despite this legacy filename, this launcher intentionally does NOT run a
recursive 2023 experiment. It reproduces the frozen independent +6-hour test;
recursion remains limited to the separate today +24-hour demonstration.
"""

from __future__ import annotations

import subprocess
import sys
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

command = [
    sys.executable,
    str(REPOSITORY / "scripts" / "generate_2023_full_year_recursive_predictions.py"),
    "--panel-dir",
    str(work / "india_lightning_era5_physics_0p25_forecast_panel"),
    "--model-dir",
    str(work / "india_lightning_0p25_spatial_models"),
    "--output-dir",
    str(output),
    "--comparison-zip",
    str(archive),
]

print("Running corrected independent +6-hour 2023 holdout inference...")
subprocess.run(command, check=True)
print(f"\nREADY TO UPLOAD:\n{archive}")
