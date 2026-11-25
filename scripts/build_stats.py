from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


SCRIPT_VERSION = "statistical-showcase-v1"
MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}
PERIOD_ORDER = {"Night": 0, "Morning": 1, "Afternoon": 2, "Evening": 3}
SEASON_ORDER = {
    "Winter": 0,
    "Pre-monsoon": 1,
    "Southwest monsoon": 2,
    "Post-monsoon": 3,
}


def rows(frame: pl.DataFrame) -> list[dict]:
    return frame.to_dicts()


def write_csv(path: Path, values: list[dict]) -> None:
    if not values:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def collect(query: pl.LazyFrame) -> pl.DataFrame:
    return query.collect(engine="streaming")


def main() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    repo_root = script_path.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=(
            project_root
            / "Lightning_Data"
            / "processed"
            / "india_lightning_events_exact_0p25_v2"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "frontend" / "data" / "stats",
    )
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    dataset_manifest_path = source_dir / "dataset_manifest.json"
    partition_manifest_path = source_dir / "partition_manifest.csv"
    if not dataset_manifest_path.is_file() or not partition_manifest_path.is_file():
        raise FileNotFoundError(f"Audited source manifests are absent under {source_dir}")

    source_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    partitions = sorted(source_dir.glob("india_lightning_exact_0p25_*.parquet"))
    if len(partitions) != 60:
        raise RuntimeError(f"Expected 60 monthly Parquets, found {len(partitions)}")

    with partition_manifest_path.open(newline="", encoding="utf-8") as handle:
        partition_rows = list(csv.DictReader(handle))
    manifest_partition_rows = sum(int(item["row_count"]) for item in partition_rows)
    scan = pl.scan_parquet([str(path) for path in partitions])

    annual = collect(
        scan.group_by("year_utc")
        .agg(pl.len().alias("strokes"))
        .sort("year_utc")
    )
    total_strokes = int(annual["strokes"].sum())
    annual_values = [
        {
            "year": int(item["year_utc"]),
            "strokes": int(item["strokes"]),
            "share_percent": round(100 * int(item["strokes"]) / total_strokes, 2),
        }
        for item in rows(annual)
    ]

    monthly = collect(
        scan.group_by("month_ist")
        .agg(pl.len().alias("strokes"))
        .sort("month_ist")
    )
    monthly_values = [
        {
            "month": int(item["month_ist"]),
            "label": MONTH_NAMES[int(item["month_ist"])],
            "strokes": int(item["strokes"]),
            "five_year_mean": round(int(item["strokes"]) / 5, 1),
            "share_percent": round(100 * int(item["strokes"]) / total_strokes, 2),
        }
        for item in rows(monthly)
    ]

    season_time = collect(
        scan.group_by(["season_ist", "time_period_ist"])
        .agg(pl.len().alias("strokes"))
    )
    season_time_values = [
        {
            "season": str(item["season_ist"]),
            "period": str(item["time_period_ist"]),
            "strokes": int(item["strokes"]),
            "share_percent": round(100 * int(item["strokes"]) / total_strokes, 2),
        }
        for item in rows(season_time)
    ]
    season_time_values.sort(
        key=lambda item: (
            SEASON_ORDER.get(item["season"], 99),
            PERIOD_ORDER.get(item["period"], 99),
        )
    )

    season_totals: dict[str, int] = {}
    period_totals: dict[str, int] = {}
    for item in season_time_values:
        season_totals[item["season"]] = season_totals.get(item["season"], 0) + item[
            "strokes"
        ]
        period_totals[item["period"]] = period_totals.get(item["period"], 0) + item[
            "strokes"
        ]

    grid = collect(
        scan.group_by(["grid_id", "grid_lat_center", "grid_lon_center"])
        .agg(pl.len().alias("strokes"))
        .sort("strokes", descending=True)
    )
    hotspot_values = [
        {
            "rank": rank,
            "grid_id": int(item["grid_id"]),
            "latitude": round(float(item["grid_lat_center"]), 2),
            "longitude": round(float(item["grid_lon_center"]), 2),
            "strokes": int(item["strokes"]),
            "share_percent": round(100 * int(item["strokes"]) / total_strokes, 3),
        }
        for rank, item in enumerate(rows(grid.head(12)), start=1)
    ]

    daily = collect(
        scan.group_by("date_ist")
        .agg(pl.len().alias("strokes"))
        .sort("strokes", descending=True)
    )
    top_day_values = [
        {
            "rank": rank,
            "date": str(item["date_ist"]),
            "strokes": int(item["strokes"]),
            "share_percent": round(100 * int(item["strokes"]) / total_strokes, 3),
        }
        for rank, item in enumerate(rows(daily.head(8)), start=1)
    ]

    expected_windows = int(source_manifest["expected_six_hour_windows"])
    scored_windows = int(source_manifest["complete_six_hour_windows"])
    masked_windows = int(source_manifest["masked_six_hour_windows"])
    years = [item["year"] for item in annual_values]
    peak_year = max(annual_values, key=lambda item: item["strokes"])
    peak_month = max(monthly_values, key=lambda item: item["strokes"])
    peak_season_name, peak_season_strokes = max(
        season_totals.items(), key=lambda item: item[1]
    )
    peak_period_name, peak_period_strokes = max(
        period_totals.items(), key=lambda item: item[1]
    )

    checks = {
        "partition_count_is_60": len(partitions) == 60,
        "partition_manifest_count_is_60": len(partition_rows) == 60,
        "parquet_total_matches_dataset_manifest": total_strokes
        == int(source_manifest["retained_rows"]),
        "partition_manifest_total_matches_parquet": manifest_partition_rows
        == total_strokes,
        "annual_total_matches_parquet": sum(item["strokes"] for item in annual_values)
        == total_strokes,
        "monthly_total_matches_parquet": sum(item["strokes"] for item in monthly_values)
        == total_strokes,
        "season_time_total_matches_parquet": sum(
            item["strokes"] for item in season_time_values
        )
        == total_strokes,
        "windows_reconcile": scored_windows + masked_windows == expected_windows,
        "all_source_fingerprints_verified": bool(
            source_manifest["validation_gates"][
                "all_partition_content_fingerprints_verified"
            ]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Statistical reconciliation failed: {failed}")

    takeaways = {
        "coverage": (
            f"{scored_windows:,} of {expected_windows:,} six-hour windows "
            f"({100 * scored_windows / expected_windows:.1f}%) were scored; "
            f"{masked_windows:,} remained masked rather than being treated as zero."
        ),
        "annual": (
            f"{peak_year['year']} had the largest audited annual volume at "
            f"{peak_year['strokes']:,} strokes, supporting a chronological split "
            "instead of random sampling."
        ),
        "monthly": (
            f"{peak_month['label']} contributed {peak_month['share_percent']:.1f}% "
            "of all audited strokes, supporting season-aware calendar features."
        ),
        "daily_cycle": (
            f"{peak_period_name} contained {peak_period_strokes:,} strokes "
            f"({100 * peak_period_strokes / total_strokes:.1f}%), supporting "
            "six-hour IST issue periods."
        ),
        "spatial": (
            f"The leading 0.25° cell was centred at "
            f"{hotspot_values[0]['latitude']:.2f}°N, "
            f"{hotspot_values[0]['longitude']:.2f}°E with "
            f"{hotspot_values[0]['strokes']:,} strokes, supporting native-grid "
            "spatial modelling rather than a single national total."
        ),
    }

    payload = {
        "manifest": {
            "script_version": SCRIPT_VERSION,
            "built_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": "india_lightning_events_exact_0p25_v2",
            "source_partition_count": len(partitions),
            "source_partitions": [
                {
                    "name": item["parquet_file"],
                    "rows": int(item["row_count"]),
                }
                for item in partition_rows
            ],
            "timezone_for_display": "Asia/Kolkata (IST)",
            "window_definition": "Four fixed six-hour IST periods per calendar day",
            "years": [min(years), max(years)],
            "retained_strokes": total_strokes,
            "expected_windows": expected_windows,
            "scored_windows": scored_windows,
            "masked_windows": masked_windows,
            "source_days_missing": int(source_manifest["source_days_missing"]),
            "source_days_unreadable": int(source_manifest["source_days_unreadable"]),
            "missing_policy": source_manifest["missing_data_policy"],
            "reconciliation_checks": checks,
            "reconciliation_status": "passed",
        },
        "summary": {
            "retained_strokes": total_strokes,
            "scored_windows": scored_windows,
            "masked_windows": masked_windows,
            "years_covered": f"{min(years)}–{max(years)}",
            "peak_season": peak_season_name,
            "peak_season_share_percent": round(
                100 * peak_season_strokes / total_strokes, 2
            ),
        },
        "annual": annual_values,
        "monthly": monthly_values,
        "season_time": season_time_values,
        "hotspots": hotspot_values,
        "top_days": top_day_values,
        "takeaways": takeaways,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "statistical-summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "annual.csv", annual_values)
    write_csv(output_dir / "monthly.csv", monthly_values)
    write_csv(output_dir / "season-time.csv", season_time_values)
    write_csv(output_dir / "hotspots.csv", hotspot_values)
    write_csv(output_dir / "top-days.csv", top_day_values)

    print("STATISTICAL SHOWCASE DATA READY")
    print(f"Strokes          : {total_strokes:,}")
    print(f"Partitions       : {len(partitions)}")
    print(f"Windows scored   : {scored_windows:,}")
    print(f"Windows masked   : {masked_windows:,}")
    print(f"Peak month       : {peak_month['label']} ({peak_month['strokes']:,})")
    print(f"Peak period      : {peak_period_name} ({peak_period_strokes:,})")
    print(f"Output           : {output_dir}")
    print(f"Reconciliation   : {checks}")


if __name__ == "__main__":
    main()
