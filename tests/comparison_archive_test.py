from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_2023_full_year_recursive_predictions import (
    append_month_to_comparison_archive,
)


def main() -> None:
    rows = []
    for hour in (0, 6, 12, 18):
        for grid_id in range(4):
            rows.append(
                {
                    "valid_time_ist": datetime(2023, 7, 21, hour),
                    "grid_id": grid_id,
                    "grid_lat": 20.0 + grid_id * 0.25,
                    "grid_lon": 78.0 + grid_id * 0.25,
                    "expected_region50_count": float((grid_id + 1) * (hour + 1)),
                    "observed_region50_count": int((4 - grid_id) * (hour + 1)),
                    "p_heavy": 0.01 * (grid_id + 1),
                    "warning_code": min(grid_id, 3),
                }
            )

    with tempfile.TemporaryDirectory(prefix="comparison_archive_test_") as temporary:
        archive_path = Path(temporary) / "comparison.zip"
        dates: list[str] = []
        stats: dict[str, dict[str, float]] = {}
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            append_month_to_comparison_archive(
                archive, pl.DataFrame(rows), dates, stats
            )
            archive.writestr(
                "manifest.json",
                json.dumps({"dates": dates, "stats": stats}),
            )

        with zipfile.ZipFile(archive_path, "r") as archive:
            manifest = json.loads(archive.read("manifest.json"))
            payload = json.loads(archive.read("daily/2023-07-21.json"))

        assert manifest["dates"] == ["2023-07-21"]
        assert payload["date"] == "2023-07-21"
        assert len(payload["cells"]) == 4
        assert payload["predicted_total"] > 0
        assert payload["observed_total"] > 0
        assert payload["prediction_mode"] == "independent_6h_holdout_with_observed_history"
        assert "not unique India-wide" in payload["aggregation_note"]
        assert 0 <= payload["top10_exact_overlap"] <= 1
        assert -1 <= payload["spatial_spearman"] <= 1

    print("COMPARISON ARCHIVE TEST PASSED")


if __name__ == "__main__":
    main()
