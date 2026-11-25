from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
DATA = FRONTEND / "data" / "stats"


def test_statistical_summary_reconciles() -> None:
    payload = json.loads((DATA / "statistical-summary.json").read_text(encoding="utf-8"))
    manifest = payload["manifest"]
    assert manifest["reconciliation_status"] == "passed"
    assert all(manifest["reconciliation_checks"].values())
    assert manifest["source_partition_count"] == 60
    assert payload["summary"]["retained_strokes"] == 11_102_575
    assert payload["summary"]["scored_windows"] + payload["summary"]["masked_windows"] == 7_304
    assert sum(item["strokes"] for item in payload["annual"]) == 11_102_575
    assert sum(item["strokes"] for item in payload["monthly"]) == 11_102_575
    assert sum(item["strokes"] for item in payload["season_time"]) == 11_102_575


def test_downloadable_tables_are_nonempty() -> None:
    for name in ("annual.csv", "monthly.csv", "season-time.csv", "hotspots.csv", "top-days.csv"):
        with (DATA / name).open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        assert records, f"{name} must contain data"


def test_page_contract_and_navigation() -> None:
    page = (FRONTEND / "statistical-analysis.html").read_text(encoding="utf-8")
    index = (FRONTEND / "index.html").read_text(encoding="utf-8")
    script = (FRONTEND / "statistics.js").read_text(encoding="utf-8")
    for required in (
        'id="annual-chart"',
        'id="monthly-chart"',
        'id="season-time-matrix"',
        'id="hotspot-table"',
        'id="top-days"',
        "/data/stats/statistical-summary.json",
    ):
        assert required in page
    assert "/statistical-analysis.html" in index
    assert "fetch(SUMMARY_URL" in script
    assert "11,102,575" not in page
    assert "11,102,575" not in script
