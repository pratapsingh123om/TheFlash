# Issue: Authenticate Vercel to the private GCS model bundle without service-account keys

## Status

Blocked on keyless Google Cloud authentication from the Vercel runtime.

The application itself is working:

- Google Cloud Shell downloads and verifies the model bundle.
- The 106-feature XGBoost model loads successfully.
- The frontend and backend run together on port 8080.
- GitHub CI passes on Linux.
- Vercel builds successfully and serves the frontend with HTTP 200.
- Vercel `/api/health` responds, but the model remains unloaded because Google credentials are unavailable.

Current Vercel health response:

```json
{
  "status": "attention_required",
  "model_loaded": false,
  "input_exists": false,
  "forecast_exists": false,
  "error": "AssetBootstrapError: DefaultCredentialsError: Your default credentials were not found."
}
```

## Root cause

Cloud Shell and Vercel are separate execution environments.

Cloud Shell runs inside Google Cloud and automatically receives short-lived Application Default Credentials for the signed-in user. Vercel cannot inherit, copy, or connect to that Cloud Shell identity.

Creating a permanent service-account JSON key also fails:

```text
FAILED_PRECONDITION: Key creation is not allowed on this service account.
constraints/iam.disableServiceAccountKeyCreation
```

This organization policy should remain enabled. Do not weaken it merely to deploy this demonstration.

The correct solution is:

```text
Vercel request OIDC token
        ↓
Google Workload Identity Federation
        ↓
Short-lived service-account impersonation token
        ↓
Read-only access to the single private GCS model bucket
```

No persistent private key is required.

## Relevant resources

- Vercel OIDC overview: <https://vercel.com/docs/oidc>
- Vercel → Google Cloud configuration: <https://vercel.com/docs/oidc/gcp>
- Google Workload Identity Federation: <https://cloud.google.com/iam/docs/workload-identity-federation>
- Python external-account credentials: <https://google-auth.readthedocs.io/en/latest/reference/google.auth.identity_pool.html>

## Existing resources

```text
Google Cloud project: data-mines
GCS bucket: gs://nuclei-lenses
Model object: gs://nuclei-lenses/flash-model/india_lightning_24h_demo_ed5dec2770f7.zip
Service account: theflash-vercel@data-mines.iam.gserviceaccount.com
Vercel production project: the-flash-hxuh
```

The service account already needs only this bucket role:

```text
roles/storage.objectViewer
```

## Phase 1 — Enable Vercel OIDC

In the Vercel project `the-flash-hxuh`:

1. Open **Settings → Security**.
2. Find **Secure backend access with OIDC Federation**.
3. Select **Team issuer**.
4. Enable OIDC.
5. Record the exact issuer URL shown by Vercel.

It should resemble:

```text
https://oidc.vercel.com/VERCEL_TEAM_SLUG
```

Do not guess the team slug. Copy the value shown by Vercel.

## Phase 2 — Configure Google Workload Identity Federation

Run the following in Google Cloud Shell after replacing `VERCEL_TEAM_SLUG` with the exact value from Vercel.

```bash
set -euo pipefail

export GCP_PROJECT_ID="data-mines"
export GCP_PROJECT_NUMBER="$(
  gcloud projects describe "$GCP_PROJECT_ID" \
    --format='value(projectNumber)'
)"
export GCP_SERVICE_ACCOUNT_EMAIL="theflash-vercel@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

export WIF_POOL_ID="vercel"
export WIF_PROVIDER_ID="vercel"

export VERCEL_TEAM_SLUG="REPLACE_WITH_EXACT_TEAM_SLUG"
export VERCEL_PROJECT_NAME="the-flash-hxuh"
export VERCEL_ENVIRONMENT="production"
export VERCEL_ISSUER="https://oidc.vercel.com/${VERCEL_TEAM_SLUG}"
export VERCEL_TOKEN_AUDIENCE="https://vercel.com/${VERCEL_TEAM_SLUG}"

gcloud services enable \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  storage.googleapis.com \
  --project "$GCP_PROJECT_ID"

gcloud iam workload-identity-pools describe "$WIF_POOL_ID" \
  --project "$GCP_PROJECT_ID" \
  --location global >/dev/null 2>&1 || \
gcloud iam workload-identity-pools create "$WIF_POOL_ID" \
  --project "$GCP_PROJECT_ID" \
  --location global \
  --display-name "Vercel production"

gcloud iam workload-identity-pools providers describe "$WIF_PROVIDER_ID" \
  --project "$GCP_PROJECT_ID" \
  --location global \
  --workload-identity-pool "$WIF_POOL_ID" >/dev/null 2>&1 || \
gcloud iam workload-identity-pools providers create-oidc "$WIF_PROVIDER_ID" \
  --project "$GCP_PROJECT_ID" \
  --location global \
  --workload-identity-pool "$WIF_POOL_ID" \
  --display-name "Vercel OIDC" \
  --issuer-uri "$VERCEL_ISSUER" \
  --allowed-audiences "$VERCEL_TOKEN_AUDIENCE" \
  --attribute-mapping "google.subject=assertion.sub"

export VERCEL_PRODUCTION_SUBJECT="owner:${VERCEL_TEAM_SLUG}:project:${VERCEL_PROJECT_NAME}:environment:${VERCEL_ENVIRONMENT}"
export VERCEL_PRODUCTION_PRINCIPAL="principal://iam.googleapis.com/projects/${GCP_PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL_ID}/subject/${VERCEL_PRODUCTION_SUBJECT}"

gcloud iam service-accounts add-iam-policy-binding \
  "$GCP_SERVICE_ACCOUNT_EMAIL" \
  --project "$GCP_PROJECT_ID" \
  --role roles/iam.workloadIdentityUser \
  --member "$VERCEL_PRODUCTION_PRINCIPAL"

gcloud storage buckets add-iam-policy-binding gs://nuclei-lenses \
  --member "serviceAccount:${GCP_SERVICE_ACCOUNT_EMAIL}" \
  --role roles/storage.objectViewer

printf 'Project number: %s\n' "$GCP_PROJECT_NUMBER"
printf 'Issuer: %s\n' "$VERCEL_ISSUER"
printf 'Allowed audience: %s\n' "$VERCEL_TOKEN_AUDIENCE"
printf 'Principal: %s\n' "$VERCEL_PRODUCTION_PRINCIPAL"
```

If the provider already exists with an incorrect issuer, audience, or attribute mapping, update or recreate it rather than assuming the existing provider is correct.

## Phase 3 — Configure non-secret Vercel environment variables

Add these variables to the `the-flash-hxuh` Vercel project for **Production**:

```text
GCP_PROJECT_ID=data-mines
GCP_PROJECT_NUMBER=VALUE_PRINTED_BY_THE_SCRIPT
GCP_SERVICE_ACCOUNT_EMAIL=theflash-vercel@data-mines.iam.gserviceaccount.com
GCP_WORKLOAD_IDENTITY_POOL_ID=vercel
GCP_WORKLOAD_IDENTITY_POOL_PROVIDER_ID=vercel
AUTO_BOOTSTRAP_ASSETS=1
AUTO_RUN_FORECAST=0
```

These values are configuration, not private keys. `GCP_SERVICE_ACCOUNT_JSON` is no longer required and should be removed from Vercel.

Redeploy after changing environment variables.

## Phase 4 — Update the Python backend for request-scoped OIDC

Vercel provides the OIDC token differently at build time and runtime:

- Build/local environment: `VERCEL_OIDC_TOKEN`
- Vercel Function request: `x-vercel-oidc-token` HTTP header

The production application must read the request header. It must not attempt to retrieve the token at module-import time.

### 4.1 Add a subject-token supplier

Use `google.auth.identity_pool.SubjectTokenSupplier`:

```python
from google.auth import identity_pool


class StaticOidcTokenSupplier(identity_pool.SubjectTokenSupplier):
    def __init__(self, token: str) -> None:
        self._token = token

    def get_subject_token(self, context, request) -> str:
        return self._token
```

### 4.2 Construct short-lived Google credentials

```python
import os
from google.auth import identity_pool
from google.cloud import storage


def vercel_gcp_storage_client(oidc_token: str) -> storage.Client:
    project_id = os.environ["GCP_PROJECT_ID"]
    project_number = os.environ["GCP_PROJECT_NUMBER"]
    service_account = os.environ["GCP_SERVICE_ACCOUNT_EMAIL"]
    pool_id = os.environ["GCP_WORKLOAD_IDENTITY_POOL_ID"]
    provider_id = os.environ["GCP_WORKLOAD_IDENTITY_POOL_PROVIDER_ID"]

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
```

### 4.3 Initialize lazily from an API request

The current backend initializes model assets in the FastAPI lifespan. That cannot access Vercel's request-scoped OIDC header.

Required change:

1. Accept `Request` in `/api/health`, `/api/config`, and `/api/forecast/run`.
2. Read `request.headers.get("x-vercel-oidc-token")`.
3. Under a process-level initialization lock:
   - exchange the token for short-lived Google credentials;
   - download and verify the GCS ZIP;
   - load the model;
   - reuse the initialized runtime for later requests on the same warm instance.
4. Keep Cloud Shell behavior unchanged by continuing to use Application Default Credentials outside Vercel.

Example shape:

```python
from fastapi import Request


def ensure_runtime(request: Request) -> None:
    if runtime is not None:
        return

    oidc_token = request.headers.get("x-vercel-oidc-token")
    if os.environ.get("VERCEL") and not oidc_token:
        raise RuntimeError("Vercel OIDC header is missing")

    with runtime_initialization_lock:
        if runtime is not None:
            return
        # Pass oidc_token into the asset bootstrap/storage client.
        prepare_runtime_assets(oidc_token=oidc_token)
        initialize_runtime()
```

Do not log the OIDC token.

## Phase 5 — Validate

After redeployment, request health. The first request can take longer because it exchanges credentials, downloads approximately 23 MB, verifies the archive, and loads XGBoost.

```bash
curl -fsS https://the-flash-hxuh.vercel.app/api/health
```

Expected response:

```json
{
  "status": "ready",
  "model_loaded": true,
  "input_exists": true,
  "forecast_exists": false,
  "forecast_running": false,
  "recursive_mode": true,
  "error": null
}
```

Then run the forecast:

```bash
curl -fsS -X POST \
  https://the-flash-hxuh.vercel.app/api/forecast/run
```

Finally reload:

```text
https://the-flash-hxuh.vercel.app/
```

## Troubleshooting

### `x-vercel-oidc-token` is missing

- OIDC is not enabled in the correct Vercel project.
- The deployment was not redeployed after enabling OIDC.
- The request is reaching the other linked Vercel project.
- The code tried to access the token during module import or lifespan instead of from `Request`.

### Google STS returns `invalid_target` or an audience error

- Confirm the provider issuer exactly matches Vercel's displayed Team issuer.
- Confirm the provider allowed audience is `https://vercel.com/VERCEL_TEAM_SLUG`.
- Confirm the credential `audience` uses the Google provider resource path with the numeric project number.

### `iam.serviceAccounts.getAccessToken` is denied

- The `roles/iam.workloadIdentityUser` binding is absent or attached to the wrong service account.
- The subject does not exactly match:

```text
owner:TEAM_SLUG:project:the-flash-hxuh:environment:production
```

- Preview deployments use `environment:preview` and require a separate principal if they should receive access.

### `storage.objects.get` is denied

Verify the service account has:

```text
roles/storage.objectViewer
```

on `gs://nuclei-lenses`.

### First request times out

- Confirm Fluid Compute is enabled.
- Keep `maxDuration` at 300 seconds in `vercel.json`.
- Confirm the CPU-only XGBoost dependency is installed.
- Inspect Vercel runtime logs for memory termination or download latency.

## Security requirements

- Do not disable `iam.disableServiceAccountKeyCreation`.
- Do not make the whole bucket public.
- Do not commit a service-account JSON key.
- Do not store an OIDC token; it is short-lived and request-scoped.
- Do not log `x-vercel-oidc-token`.
- Restrict the Google principal to the exact Vercel team, project, and production environment.
- Grant only `roles/storage.objectViewer` to the runtime service account.

## Acceptance criteria

- [ ] Vercel Team OIDC issuer enabled for `the-flash-hxuh`.
- [ ] Google workload identity pool and OIDC provider configured.
- [ ] Provider maps `google.subject=assertion.sub`.
- [ ] Provider audience matches the Vercel team audience.
- [ ] Exact production subject can impersonate `theflash-vercel`.
- [ ] Service account has read-only access to `gs://nuclei-lenses`.
- [ ] No `GCP_SERVICE_ACCOUNT_JSON` secret exists in Vercel or Git.
- [ ] Backend reads `x-vercel-oidc-token` only from the incoming request.
- [ ] `/api/health` reports `status=ready` and `model_loaded=true`.
- [ ] `/api/forecast/run` returns all four six-hour horizons.
- [ ] Frontend renders the returned forecast map.
- [ ] Cloud Shell deployment remains functional with Application Default Credentials.
