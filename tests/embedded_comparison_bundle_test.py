from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.app as backend_app


def main() -> None:
    original_settings = backend_app.settings
    with tempfile.TemporaryDirectory(prefix="embedded-comparison-test-") as temporary:
        root = Path(temporary)
        data_dir = root / "india_lightning_24h_demo" / "data"
        data_dir.mkdir(parents=True)
        replay = data_dir / "next_24h_features.parquet"
        replay.write_bytes(b"test replay")
        comparison = data_dir / backend_app.COMPARISON_ARCHIVE_FILENAME
        with zipfile.ZipFile(comparison, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps({"dates": ["2023-01-01"]}))
            archive.writestr("daily/2023-01-01.json", os.urandom(4096))

        backend_app.settings = replace(original_settings, input_parquet=replay)
        try:
            resolved = backend_app.get_comparison_archive_path()
            assert resolved == comparison
        finally:
            backend_app.settings = original_settings

    print("EMBEDDED COMPARISON BUNDLE TEST PASSED")


if __name__ == "__main__":
    main()
