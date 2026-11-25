from __future__ import annotations

import zipfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import classify_comparison_archive_error


def main() -> None:
    cases = (
        (RuntimeError("404 GET: No such object"), "object_not_found"),
        (RuntimeError("403 Forbidden: permission denied"), "permission_denied"),
        (zipfile.BadZipFile("File is not a zip file"), "invalid_zip"),
        (KeyError("There is no item named 'manifest.json'"), "manifest_missing"),
        (TimeoutError("download timeout"), "download_timeout"),
        (OSError("No space left on device"), "temporary_storage_exhausted"),
        (RuntimeError("connection reset"), "retrieval_or_validation_failed"),
    )
    for error, expected in cases:
        actual = classify_comparison_archive_error(error)
        if actual != expected:
            raise AssertionError(f"{error!r}: expected {expected}, received {actual}")
    print("COMPARISON ERROR CLASSIFICATION TEST PASSED")


if __name__ == "__main__":
    main()
