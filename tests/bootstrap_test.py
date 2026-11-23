from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.bootstrap_assets import PACKAGE_NAME, REQUIRED_MODEL_FILES, bootstrap_assets


def build_test_bundle(root: Path) -> Path:
    package = root / PACKAGE_NAME
    model_dir = package / "models"
    data_dir = package / "data"
    model_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    configuration = {
        "features": ["grid_lat_center", "grid_lon_center"],
        "forecast_horizon_hours": 6,
        "same_window_weather_included": False,
        "posthoc_multiplier": False,
    }
    for name in REQUIRED_MODEL_FILES:
        payload = configuration if name == "run_configuration.json" else {"status": "test"}
        (model_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (data_dir / "next_24h_features.parquet").write_bytes(b"test-parquet-placeholder")

    archive = root / "bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(root).as_posix())
    return archive


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="lightning_bootstrap_test_") as temporary:
        root = Path(temporary)
        archive = build_test_bundle(root)
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        asset_root = root / "assets"

        first = bootstrap_assets(str(archive), checksum, asset_root)
        assert first["cached"] is False
        assert Path(str(first["model_dir"])).is_dir()
        assert Path(str(first["input_parquet"])).is_file()

        second = bootstrap_assets(str(archive), checksum, asset_root)
        assert second["cached"] is True
        assert first["package_root"] == second["package_root"]

        try:
            bootstrap_assets(str(archive), "0" * 64, root / "bad-assets")
        except RuntimeError as exc:
            assert "checksum mismatch" in str(exc)
        else:
            raise AssertionError("A bad checksum was not rejected")

        print("ASSET BOOTSTRAP TEST PASSED")


if __name__ == "__main__":
    main()
