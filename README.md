# India Lightning 24-Hour Forecast Demo

GitHub-ready backend and frontend for four rolling six-hour lightning-storm category forecasts over India. It keeps retrospective 2023 observations separate from a current-time, lightning-history-only recursive candidate. Source code comes from GitHub, while the frozen model and seed Parquet stay in the private Cloud Storage ZIP.

> **Research prototype:** this is not an operational public-warning service. A forecast is current only when the API reports `lightning_history_seed.is_current=true`. Horizons 2–4 recursively update lightning-history proxies, so uncertainty increases through the 24-hour demonstration.

## Proven deployment flow

```text
GitHub checkout
  ├─ FastAPI backend
  ├─ HTML/CSS/JavaScript frontend
  └─ verified Cloud Shell launcher
             │
             ├─ downloads private GCS ZIP once
             ├─ checks immutable SHA-256
             ├─ validates frozen model configuration
             └─ points the backend at models + replay Parquet

Browser ──same URL──> frontend ──/api──> FastAPI ──> XGBoost models
```

The browser does **not** open the ZIP or execute XGBoost itself. FastAPI loads the artifacts from the verified ZIP and serves the frontend on the same origin. This avoids exposing the private bucket and requires no CORS configuration.

## Frozen bucket artifact

```text
GCS URI: gs://nuclei-lenses/flash-model/india_lightning_24h_demo_ed5dec2770f7.zip
SHA-256: 0b565c63f89be15818920ab6798a46530b0e0cbdfffd6c4a5b7c8dbacf7a39d1
```

The public HTTPS form may return `403 Forbidden`; that is expected for a private object. Cloud Shell uses the signed-in Google account through Application Default Credentials.

## Repository contents

- `backend/app.py` — API and same-origin static frontend
- `backend/model_runtime.py` — frozen four-horizon inference
- `backend/bootstrap_assets.py` — GCS/local download, SHA-256 verification, safe extraction, schema checks, persistent cache
- `frontend/` — responsive forecast dashboard and India boundary
- `scripts/run_cloud_shell.sh` — one-command Cloud Shell setup/restart
- `tests/bootstrap_test.py` — checksum/cache bootstrap test
- `tests/smoke_test.py` — synthetic model, four-horizon inference, and API test
- `.github/workflows/ci.yml` — automatic Python, JavaScript, bootstrap, and inference validation

Models, Parquet files, runtime outputs, ZIPs, credentials, and virtual environments are excluded by `.gitignore`.

## Push this folder to GitHub

```bash
git init
git branch -M main
git add .
git commit -m "Finalize India lightning forecast demo"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

GitHub Actions tests the complete synthetic inference path on every push and pull request. It does not need access to the private production ZIP.

## Deploy on Vercel

The repository declares the custom FastAPI entrypoint in `pyproject.toml`:

```toml
[project]
requires-python = ">=3.12,<3.13"
dependencies = ["fastapi==0.115.12", "...other locked runtime packages..."]

[tool.vercel]
entrypoint = "app:app"
```

The runtime packages are listed in both `pyproject.toml` and `requirements.txt`: Vercel installs from the project metadata, while the Cloud Shell launcher installs from `requirements.txt`.

The root `app.py` imports the application from `backend.app` and provides a safe diagnostic response if Vercel cannot import a native dependency. This keeps deployment failures visible at `/api/health` instead of returning only `FUNCTION_INVOCATION_FAILED`.

Because the model ZIP is private, Vercel cannot use the Google credentials from Cloud Shell. The deployment uses Vercel OIDC, Google Workload Identity Federation, and short-lived service-account impersonation. It does not use a service-account JSON key. The complete setup is documented in [`ISSUE.md`](ISSUE.md).

```text
GCP_PROJECT_ID=data-mines
GCP_PROJECT_NUMBER=YOUR_NUMERIC_PROJECT_NUMBER
GCP_SERVICE_ACCOUNT_EMAIL=theflash-vercel@data-mines.iam.gserviceaccount.com
GCP_WORKLOAD_IDENTITY_POOL_ID=vercel
GCP_WORKLOAD_IDENTITY_POOL_PROVIDER_ID=vercel
AUTO_BOOTSTRAP_ASSETS=1
AUTO_RUN_FORECAST=0
```

`MODEL_BUNDLE_URI` and `MODEL_BUNDLE_SHA256` already have frozen defaults. Never create or commit a service-account JSON key, and never expose the request-scoped OIDC token to the frontend.

On a cold start, the FastAPI lifespan downloads the ZIP to Vercel's writable `/tmp` directory, verifies its SHA-256 and model contract, then loads the XGBoost artifacts. Warm function instances reuse that cache. The browser receives neither the credentials nor the bucket URL.

`vercel.json` explicitly includes `frontend/**` in the FastAPI function and allows up to 300 seconds for cold-start download/model initialization. Native ML imports are deferred until startup, so `/api/health` can report an actionable error instead of the entire function returning an opaque invocation failure.

Vercel installs the official `xgboost-cpu` wheel rather than the much larger GPU-capable wheel. The application performs inference on CPU, and both packages expose the same Python `xgboost` module and JSON model loader.

If Vercel reports that the Python function is too large because of NumPy, SciPy, Polars, and XGBoost, add this project environment variable and redeploy:

```text
VERCEL_SUPPORT_LARGE_FUNCTIONS=1
```

The dashboard intentionally waits for the user to press **Forecast today · next 24 h** on Vercel rather than starting inference during cold start.

## Run from Google Cloud Shell

```bash
git clone https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
chmod +x scripts/*.sh
./scripts/run_cloud_shell.sh
```

The launcher:

1. Creates a persistent virtual environment under `~/india-lightning-cloudshell`.
2. Installs dependencies only when `requirements.txt` changes.
3. Downloads and checks the bucket ZIP only once per SHA-256.
4. Validates the model is six-hour-ahead, excludes same-window weather, and uses no post-hoc multiplier.
5. Starts the GitHub backend/frontend code on port `8080`.
6. Waits for `/api/health` and prints the health JSON or the failure log.

In Cloud Shell, select **Web preview → Preview on port 8080**. The app remains available while the Cloud Shell VM/session remains active; Cloud Shell is suitable for the project demonstration, not permanent production hosting.

### Restart after a GitHub update

```bash
git pull
./scripts/run_cloud_shell.sh
```

The launcher safely stops only the previous process that used its own persistent virtual environment, then starts the updated source. Cached model assets are reused when their checksum is unchanged.

### Useful diagnostics

```bash
curl -fsS http://127.0.0.1:8080/api/health
curl -fsS http://127.0.0.1:8080/api/config
cat ~/india-lightning-cloudshell/server.log
```

Force a clean bucket re-download without deleting other state:

```bash
cd YOUR_REPOSITORY
source ~/india-lightning-cloudshell/.venv/bin/activate
python -m backend.bootstrap_assets \
  --asset-root ~/india-lightning-cloudshell/assets \
  --force-download
./scripts/run_cloud_shell.sh
```

## API

- `GET /api/health` — model, input, running forecast, and output readiness
- `GET /api/config` — frozen model/configuration summary
- `POST /api/forecast/run?mode=lightning_only_today` — current-time autoregressive lightning-history candidate
- `POST /api/forecast/run?mode=full_features_replay` — frozen retrospective feature replay
- `GET /api/forecast/latest` — return the latest map payload
- `GET /api/boundary` — return the India boundary used by the dashboard

Only one forecast can execute at a time. A simultaneous run request receives HTTP `409`, preventing two processes from racing the output file.

## Local tests

Python 3.11 is the reference runtime.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test
```

The tests use temporary synthetic XGBoost artifacts and a temporary ZIP. They never download the private production bundle.

## Frontend/backend contract

`frontend/app.js` deliberately uses relative endpoints such as `/api/health` and `/api/forecast/latest`. Therefore the dashboard always calls the FastAPI process that served it, whether opened through Cloud Shell Web Preview or at `http://127.0.0.1:8080`. No bucket URL, token, model path, or secret is sent to the browser.

The output contains four periods with per-cell latitude, longitude, category probabilities, warning code, and expected count. The dashboard draws those cells over the included India boundary and provides horizon selection and summary cards.

## Supplying current lightning history

Set either `LIGHTNING_HISTORY_PARQUET` to a local file or `LIGHTNING_HISTORY_URI` to a private GCS object containing one row per model `grid_id` and these columns:

```text
grid_id
observation_time_ist
lightning_lag6h_count
lightning_lag12h_count
lightning_lag18h_count
lightning_lag24h_count
```

`lightning_recent24h_sum` may replace `lightning_lag18h_count`. Every model grid cell must appear exactly once. The backend rejects duplicate cells, incomplete spatial coverage, and missing lag values. A seed from no more than 12 hours before the issue boundary is marked current. Without this file, the app still demonstrates the recursive loop but clearly labels the bundled 2023 seed as stale.

For Vercel, configure a non-secret URI such as:

```text
LIGHTNING_HISTORY_URI=gs://nuclei-lenses/flash-model/latest_lightning_history.parquet
```

The object is downloaded with the same request-scoped OIDC credentials as the model bundle. Update that object from the lightning-ingestion process before requesting the forecast; no bucket credential is exposed to the browser.

## Limitations to state in the demonstration

- The bundled history seed is from 2023 and is not a live input.
- `lightning_only_today` masks all learned weather predictors. It becomes a genuine current candidate only after supplying recent lightning history.
- The lightning-only candidate is an ablation of the trained full-feature model, not a separately prospectively validated operational model.
- The first interval uses supplied history; later intervals use recursively updated proxies.
- The result is a regional storm-category probability map, not individual strike coordinates.
- The model remains a retrospective research prototype and has not passed prospective operational validation.
- Cloud Shell stops background services when its VM is recycled; rerun the launcher to restore the demo.

## Updating the model bundle

Never overwrite the existing GCS object. Upload a new versioned ZIP, calculate its SHA-256, and update the URI/digest defaults in:

1. `backend/bootstrap_assets.py`
2. `.env.example`
3. `scripts/run_cloud_shell.sh`

The launcher refuses a mismatched checksum or incomplete model package instead of silently serving changed artifacts.
