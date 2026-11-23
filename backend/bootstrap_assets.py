from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.cloud import storage

try:
    from google.auth import identity_pool

    class StaticOidcTokenSupplier(identity_pool.SubjectTokenSupplier):
        def __init__(self, token: str) -> None:
            self._token = token

        def get_subject_token(self, context, request) -> str:
            return self._token

except (ImportError, AttributeError):
    class StaticOidcTokenSupplier:  # type: ignore[no-redef]
        def __init__(self, token: str) -> None:
            self._token = token

        def get_subject_token(self, context, request) -> str:
            return self._token


DEFAULT_BUNDLE_URI = (
    "gs://nuclei-lenses/flash-model/"
    "india_lightning_24h_demo_ed5dec2770f7.zip"
)
DEFAULT_BUNDLE_SHA256 = (
    "0b565c63f89be15818920ab6798a46530b0e0cbdfffd6c4a5b7c8dbacf7a39d1"
)
DEFAULT_GCP_PROJECT_ID = "data-mines"
DEFAULT_GCP_PROJECT_NUMBER = "939450581122"
DEFAULT_GCP_SERVICE_ACCOUNT_EMAIL = "theflash-vercel@data-mines.iam.gserviceaccount.com"
DEFAULT_WIF_POOL_ID = "vercel"
DEFAULT_WIF_PROVIDER_ID = "vercel"
PACKAGE_NAME = "india_lightning_24h_demo"
REQUIRED_MODEL_FILES = {
    "run_configuration.json",
    "locked_storm_category_thresholds.json",
    "alarm_thresholds_selected_on_2022.json",
    "selection_frozen_before_2023.json",
    "training_complete.json",
    "ordinal_any.json",
    "ordinal_q95.json",
    "ordinal_q99.json",
    "severity_mild.json",
    "severity_strong.json",
    "severity_heavy.json",
    "isotonic_any_selected_on_2022.json",
    "isotonic_q95_selected_on_2022.json",
    "isotonic_q99_selected_on_2022.json",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("MODEL_BUNDLE_SHA256 must be a 64-character hexadecimal digest")
    return normalized


def vercel_gcp_storage_client(oidc_token: str) -> "storage.Client":
    from google.auth import identity_pool
    from google.cloud import storage

    project_id = os.environ.get("GCP_PROJECT_ID", DEFAULT_GCP_PROJECT_ID)
    project_number = os.environ.get("GCP_PROJECT_NUMBER", DEFAULT_GCP_PROJECT_NUMBER)
    service_account = os.environ.get(
        "GCP_SERVICE_ACCOUNT_EMAIL", DEFAULT_GCP_SERVICE_ACCOUNT_EMAIL
    )
    pool_id = os.environ.get("GCP_WORKLOAD_IDENTITY_POOL_ID", DEFAULT_WIF_POOL_ID)
    provider_id = os.environ.get(
        "GCP_WORKLOAD_IDENTITY_POOL_PROVIDER_ID", DEFAULT_WIF_PROVIDER_ID
    )

    missing = [
        name
        for name, val in [
            ("GCP_PROJECT_ID", project_id),
            ("GCP_PROJECT_NUMBER", project_number),
            ("GCP_SERVICE_ACCOUNT_EMAIL", service_account),
            ("GCP_WORKLOAD_IDENTITY_POOL_ID", pool_id),
            ("GCP_WORKLOAD_IDENTITY_POOL_PROVIDER_ID", provider_id),
        ]
        if not val
    ]
    if missing:
        raise KeyError(
            f"Missing required Workload Identity environment variable(s): {', '.join(missing)}"
        )

    audience = (
        f"//iam.googleapis.com/projects/{project_number}/locations/global/"
        f"workloadIdentityPools/{pool_id}/providers/{provider_id}"
    )
    impersonation_url = (
        "https://iamcredentials.googleapis.com/v1/projects/-/"
        f"serviceAccounts/{service_account}:generateAccessToken"
    )

    credentials = identity_pool.Credentials(
        audience=audience,
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        token_url="https://sts.googleapis.com/v1/token",
        service_account_impersonation_url=impersonation_url,
        subject_token_supplier=StaticOidcTokenSupplier(oidc_token),
        scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
    )
    return storage.Client(project=project_id, credentials=credentials)


def download_bundle(
    uri: str,
    destination: Path,
    storage_client: "storage.Client | None" = None,
    oidc_token: str | None = None,
) -> None:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme == "gs":
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-storage is required to download a gs:// bundle"
            ) from exc
        if not parsed.netloc or not parsed.path.lstrip("/"):
            raise ValueError(f"Invalid Google Cloud Storage URI: {uri}")
        if storage_client is not None:
            client = storage_client
        elif oidc_token:
            client = vercel_gcp_storage_client(oidc_token)
        else:
            service_account_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON", "").strip()
            if service_account_json:
                try:
                    from google.oauth2 import service_account

                    service_account_info = json.loads(service_account_json)
                    credentials = service_account.Credentials.from_service_account_info(
                        service_account_info,
                        scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
                    )
                    client = storage.Client(
                        project=service_account_info.get("project_id"),
                        credentials=credentials,
                    )
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    raise RuntimeError("GCP_SERVICE_ACCOUNT_JSON is not valid service-account JSON") from exc
            else:
                client = storage.Client()
        blob = client.bucket(parsed.netloc).blob(parsed.path.lstrip("/"))
        blob.download_to_filename(str(destination))
        return

    if parsed.scheme in {"http", "https"}:
        request = urllib.request.Request(uri, headers={"User-Agent": "india-lightning-demo/1.0"})
        with urllib.request.urlopen(request, timeout=300) as response, destination.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        return

    source = Path(urllib.request.url2pathname(parsed.path) if parsed.scheme == "file" else uri)
    if not source.is_file():
        raise FileNotFoundError(f"Local model bundle does not exist: {source}")
    shutil.copy2(source, destination)


def safe_extract(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe path in model bundle: {member.filename}")
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise RuntimeError(f"Symbolic links are not allowed in model bundle: {member.filename}")
            target = (destination_root / member_path).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise RuntimeError(f"Archive member escapes extraction root: {member.filename}")
        bundle.extractall(destination_root)


def validate_package(package_root: Path) -> dict[str, str | int]:
    model_dir = package_root / "models"
    input_parquet = package_root / "data" / "next_24h_features.parquet"
    missing = sorted(name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file())
    if missing:
        raise RuntimeError(f"Model bundle is missing required artifacts: {missing}")
    if not input_parquet.is_file():
        raise RuntimeError(f"Model bundle is missing replay input: {input_parquet}")

    configuration = json.loads((model_dir / "run_configuration.json").read_text(encoding="utf-8"))
    if configuration.get("forecast_horizon_hours") != 6:
        raise RuntimeError("Frozen model is not configured for a six-hour forecast horizon")
    if configuration.get("same_window_weather_included") is not False:
        raise RuntimeError("Frozen model configuration permits same-window weather leakage")
    if configuration.get("posthoc_multiplier") is not False:
        raise RuntimeError("Frozen model configuration contains a post-hoc multiplier")

    return {
        "package_root": str(package_root),
        "model_dir": str(model_dir),
        "input_parquet": str(input_parquet),
        "feature_count": len(configuration.get("features", [])),
    }


def acquire_lock(lock_path: Path, ready_marker: Path, timeout_seconds: int = 300) -> int | None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if ready_marker.is_file():
            return None
        try:
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                age_seconds = time.time() - lock_path.stat().st_mtime
                if age_seconds > timeout_seconds:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for asset bootstrap lock: {lock_path}")
            time.sleep(0.25)


def bootstrap_assets(
    uri: str,
    expected_sha256: str,
    asset_root: Path,
    force_download: bool = False,
    storage_client: "storage.Client | None" = None,
    oidc_token: str | None = None,
) -> dict[str, str | int | bool]:
    expected_sha256 = validate_sha256(expected_sha256)
    asset_root = asset_root.resolve()
    downloads_dir = asset_root / "downloads"
    releases_dir = asset_root / "releases"
    locks_dir = asset_root / "locks"
    archive = downloads_dir / f"{expected_sha256}.zip"
    release_dir = releases_dir / expected_sha256
    package_root = release_dir / PACKAGE_NAME
    ready_marker = release_dir / "asset_bundle_ready.json"
    lock_path = locks_dir / f"{expected_sha256}.lock"
    for directory in (downloads_dir, releases_dir, locks_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if ready_marker.is_file() and not force_download:
        details = validate_package(package_root)
        return {**details, "cached": True, "bundle_sha256": expected_sha256}

    lock_fd = acquire_lock(lock_path, ready_marker)
    if lock_fd is None:
        details = validate_package(package_root)
        return {**details, "cached": True, "bundle_sha256": expected_sha256}

    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        if force_download or not archive.is_file() or file_sha256(archive) != expected_sha256:
            partial = archive.with_suffix(".zip.partial")
            partial.unlink(missing_ok=True)
            download_bundle(uri, partial, storage_client=storage_client, oidc_token=oidc_token)
            actual_sha256 = file_sha256(partial)
            if actual_sha256 != expected_sha256:
                partial.unlink(missing_ok=True)
                raise RuntimeError(
                    "Model bundle checksum mismatch: "
                    f"expected {expected_sha256}, received {actual_sha256}"
                )
            partial.replace(archive)

        if ready_marker.is_file() and not force_download:
            details = validate_package(package_root)
            return {**details, "cached": True, "bundle_sha256": expected_sha256}

        with tempfile.TemporaryDirectory(prefix=".extract-", dir=releases_dir) as temporary:
            staging = Path(temporary)
            safe_extract(archive, staging)
            staged_package = staging / PACKAGE_NAME
            if not staged_package.is_dir():
                candidates = [
                    path.parent
                    for path in staging.rglob("run_configuration.json")
                    if path.parent.name == "models"
                ]
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"Expected one {PACKAGE_NAME} package root; found {len(candidates)}"
                    )
                candidate_root = candidates[0].parent
                staged_package = staging / PACKAGE_NAME
                candidate_root.replace(staged_package)

            details = validate_package(staged_package)
            marker = {
                "status": "ready",
                "bundle_uri": uri,
                "bundle_sha256": expected_sha256,
                "package_name": PACKAGE_NAME,
                "feature_count": details["feature_count"],
            }
            (staging / "asset_bundle_ready.json").write_text(
                json.dumps(marker, indent=2), encoding="utf-8"
            )
            if release_dir.exists():
                raise RuntimeError(f"Incomplete release directory already exists: {release_dir}")
            staging.replace(release_dir)

        details = validate_package(package_root)
        return {**details, "cached": False, "bundle_sha256": expected_sha256}
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify frozen model assets")
    parser.add_argument("--uri", default=os.environ.get("MODEL_BUNDLE_URI", DEFAULT_BUNDLE_URI))
    parser.add_argument(
        "--sha256",
        default=os.environ.get("MODEL_BUNDLE_SHA256", DEFAULT_BUNDLE_SHA256),
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.environ.get("ASSET_ROOT", "/tmp/india-lightning-assets")),
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--oidc-token",
        default=os.environ.get("VERCEL_OIDC_TOKEN"),
        help="Optional OIDC token for Workload Identity Federation",
    )
    arguments = parser.parse_args()
    result = bootstrap_assets(
        arguments.uri,
        arguments.sha256,
        arguments.asset_root,
        force_download=arguments.force_download,
        oidc_token=arguments.oidc_token,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
