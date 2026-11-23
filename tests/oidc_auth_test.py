from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from starlette.requests import Request
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.bootstrap_assets import (
    StaticOidcTokenSupplier,
    download_bundle,
    vercel_gcp_storage_client,
)
from backend import app as api


def test_static_oidc_token_supplier():
    test_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.mock_token"
    supplier = StaticOidcTokenSupplier(test_token)
    assert supplier.get_subject_token(None, None) == test_token
    print("test_static_oidc_token_supplier PASSED")


def test_vercel_gcp_storage_client_missing_env():
    # Save original env
    env_keys = [
        "GCP_PROJECT_ID",
        "GCP_PROJECT_NUMBER",
        "GCP_SERVICE_ACCOUNT_EMAIL",
        "GCP_WORKLOAD_IDENTITY_POOL_ID",
        "GCP_WORKLOAD_IDENTITY_POOL_PROVIDER_ID",
    ]
    old_env = {k: os.environ.get(k) for k in env_keys}
    for k in env_keys:
        os.environ.pop(k, None)

    try:
        vercel_gcp_storage_client("mock_token")
    except KeyError as exc:
        assert "GCP_PROJECT_ID" in str(exc)
        assert "GCP_PROJECT_NUMBER" in str(exc)
        print("test_vercel_gcp_storage_client_missing_env PASSED")
    else:
        raise AssertionError("Expected KeyError when required env vars are missing")
    finally:
        for k, v in old_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)


def test_vercel_gcp_storage_client_success():
    env_vars = {
        "GCP_PROJECT_ID": "data-mines",
        "GCP_PROJECT_NUMBER": "123456789012",
        "GCP_SERVICE_ACCOUNT_EMAIL": "theflash-vercel@data-mines.iam.gserviceaccount.com",
        "GCP_WORKLOAD_IDENTITY_POOL_ID": "vercel",
        "GCP_WORKLOAD_IDENTITY_POOL_PROVIDER_ID": "vercel",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        client = vercel_gcp_storage_client("sample_oidc_token_123")
        assert client.project == "data-mines"
        credentials = client._credentials
        expected_audience = (
            "//iam.googleapis.com/projects/123456789012/locations/global/"
            "workloadIdentityPools/vercel/providers/vercel"
        )
        expected_impersonation_url = (
            "https://iamcredentials.googleapis.com/v1/projects/-/"
            "serviceAccounts/theflash-vercel@data-mines.iam.gserviceaccount.com:generateAccessToken"
        )
        assert credentials._audience == expected_audience
        assert credentials._service_account_impersonation_url == expected_impersonation_url
        assert "https://www.googleapis.com/auth/devstorage.read_only" in credentials._scopes
        assert credentials._subject_token_supplier.get_subject_token(None, None) == "sample_oidc_token_123"
        print("test_vercel_gcp_storage_client_success PASSED")


def test_download_bundle_with_oidc_token():
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    with patch("backend.bootstrap_assets.vercel_gcp_storage_client", return_value=mock_client) as mock_factory:
        dest = Path("dummy_dest.zip")
        download_bundle("gs://nuclei-lenses/flash-model/test.zip", dest, oidc_token="token_abc")
        mock_factory.assert_called_once_with("token_abc")
        mock_client.bucket.assert_called_once_with("nuclei-lenses")
        mock_bucket.blob.assert_called_once_with("flash-model/test.zip")
        mock_blob.download_to_filename.assert_called_once_with(str(dest))
        print("test_download_bundle_with_oidc_token PASSED")


def test_ensure_runtime_vercel_missing_header():
    # Save original state
    orig_runtime = api.runtime
    orig_error = api.runtime_error

    try:
        api.runtime = None
        api.runtime_error = None

        with patch.dict(os.environ, {"VERCEL": "1"}, clear=False):
            # Request without x-vercel-oidc-token header
            req = Request({"type": "http", "headers": []})
            api.ensure_runtime(req)
            assert api.runtime is None
            assert api.runtime_error is not None
            assert "Vercel OIDC header is missing" in api.runtime_error

            # /api/health returns attention_required
            health_response = api.health(req)
            assert health_response["status"] == "attention_required"
            assert health_response["model_loaded"] is False
            assert "Vercel OIDC header is missing" in health_response["error"]

            # /api/forecast/run raises 503
            try:
                api.run_forecast(req)
            except HTTPException as exc:
                assert exc.status_code == 503
                assert "Vercel OIDC header is missing" in exc.detail
            else:
                raise AssertionError("Expected HTTPException 503 on run_forecast without runtime")

            print("test_ensure_runtime_vercel_missing_header PASSED")
    finally:
        api.runtime = orig_runtime
        api.runtime_error = orig_error


def test_ensure_runtime_vercel_with_header():
    orig_runtime = api.runtime
    orig_error = api.runtime_error

    try:
        api.runtime = None
        api.runtime_error = None

        mock_token = "secret-jwt-token-999"
        req = Request({"type": "http", "headers": [(b"x-vercel-oidc-token", mock_token.encode("utf-8"))]})

        called_with_token = []

        def mock_prepare(oidc_token=None):
            called_with_token.append(oidc_token)

        def mock_init():
            api.runtime = MagicMock()
            api.runtime_error = None

        with patch.dict(os.environ, {"VERCEL": "1"}, clear=False), \
             patch.object(api, "prepare_runtime_assets", side_effect=mock_prepare), \
             patch.object(api, "initialize_runtime", side_effect=mock_init):
            api.ensure_runtime(req)
            assert called_with_token == [mock_token]
            assert api.runtime is not None
            assert api.runtime_error is None

            # Subsequent call when runtime is already loaded should return immediately
            api.ensure_runtime(req)
            assert len(called_with_token) == 1  # Not called again

            print("test_ensure_runtime_vercel_with_header PASSED")
    finally:
        api.runtime = orig_runtime
        api.runtime_error = orig_error


def test_token_not_logged_in_error():
    orig_runtime = api.runtime
    orig_error = api.runtime_error

    try:
        api.runtime = None
        api.runtime_error = None

        mock_token = "very_sensitive_oidc_jwt_token_12345"
        req = Request({"type": "http", "headers": [(b"x-vercel-oidc-token", mock_token.encode("utf-8"))]})

        def mock_failing_prepare(oidc_token=None):
            raise RuntimeError("Storage download failed")

        with patch.dict(os.environ, {"VERCEL": "1"}, clear=False), \
             patch.object(api, "prepare_runtime_assets", side_effect=mock_failing_prepare):
            api.ensure_runtime(req)
            assert api.runtime is None
            assert api.runtime_error is not None
            # Must never log the secret token
            assert mock_token not in api.runtime_error
            print("test_token_not_logged_in_error PASSED")
    finally:
        api.runtime = orig_runtime
        api.runtime_error = orig_error


def main():
    test_static_oidc_token_supplier()
    test_vercel_gcp_storage_client_missing_env()
    test_vercel_gcp_storage_client_success()
    test_download_bundle_with_oidc_token()
    test_ensure_runtime_vercel_missing_header()
    test_ensure_runtime_vercel_with_header()
    test_token_not_logged_in_error()
    print("ALL OIDC AUTH TESTS PASSED")


if __name__ == "__main__":
    main()
