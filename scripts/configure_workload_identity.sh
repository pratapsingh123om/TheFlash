#!/usr/bin/env bash
set -euo pipefail

# Configure Google Workload Identity Federation for Vercel
# Run this script in Google Cloud Shell.

export GCP_PROJECT_ID="${GCP_PROJECT_ID:-data-mines}"
export GCP_PROJECT_NUMBER="$(
  gcloud projects describe "$GCP_PROJECT_ID" \
    --format='value(projectNumber)'
)"
export GCP_SERVICE_ACCOUNT_EMAIL="${GCP_SERVICE_ACCOUNT_EMAIL:-theflash-vercel@${GCP_PROJECT_ID}.iam.gserviceaccount.com}"

export WIF_POOL_ID="${WIF_POOL_ID:-vercel}"
export WIF_PROVIDER_ID="${WIF_PROVIDER_ID:-vercel}"

export VERCEL_TEAM_SLUG="${VERCEL_TEAM_SLUG:-gps9}"
export VERCEL_PROJECT_NAME="${VERCEL_PROJECT_NAME:-the-flash-hxuh}"
export VERCEL_ENVIRONMENT="${VERCEL_ENVIRONMENT:-production}"
export VERCEL_ISSUER="https://oidc.vercel.com/${VERCEL_TEAM_SLUG}"
export VERCEL_TOKEN_AUDIENCE="https://vercel.com/${VERCEL_TEAM_SLUG}"

echo "Enabling required Google Cloud services..."
gcloud services enable \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  storage.googleapis.com \
  --project "$GCP_PROJECT_ID"

echo "Checking workload identity pool '$WIF_POOL_ID'..."
gcloud iam workload-identity-pools describe "$WIF_POOL_ID" \
  --project "$GCP_PROJECT_ID" \
  --location global >/dev/null 2>&1 || \
gcloud iam workload-identity-pools create "$WIF_POOL_ID" \
  --project "$GCP_PROJECT_ID" \
  --location global \
  --display-name "Vercel production"

echo "Checking workload identity provider '$WIF_PROVIDER_ID'..."
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

echo "Binding roles/iam.workloadIdentityUser to service account for production and preview..."
for PROJECT in "the-flash-hxuh" "the-flash"; do
  for ENV in "production" "preview"; do
    SUBJECT="owner:${VERCEL_TEAM_SLUG}:project:${PROJECT}:environment:${ENV}"
    PRINCIPAL="principal://iam.googleapis.com/projects/${GCP_PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL_ID}/subject/${SUBJECT}"
    gcloud iam service-accounts add-iam-policy-binding \
      "$GCP_SERVICE_ACCOUNT_EMAIL" \
      --project "$GCP_PROJECT_ID" \
      --role roles/iam.workloadIdentityUser \
      --member "$PRINCIPAL" >/dev/null 2>&1 || true
  done
done
  --role roles/iam.workloadIdentityUser \
  --member "$VERCEL_PRODUCTION_PRINCIPAL"

echo "Binding roles/storage.objectViewer on bucket gs://nuclei-lenses..."
gcloud storage buckets add-iam-policy-binding gs://nuclei-lenses \
  --member "serviceAccount:${GCP_SERVICE_ACCOUNT_EMAIL}" \
  --role roles/storage.objectViewer

echo ""
echo "=== Configuration Complete ==="
printf 'Project ID: %s\n' "$GCP_PROJECT_ID"
printf 'Project Number: %s\n' "$GCP_PROJECT_NUMBER"
printf 'Service Account: %s\n' "$GCP_SERVICE_ACCOUNT_EMAIL"
printf 'Issuer: %s\n' "$VERCEL_ISSUER"
printf 'Allowed audience: %s\n' "$VERCEL_TOKEN_AUDIENCE"
printf 'Principal: %s\n' "$VERCEL_PRODUCTION_PRINCIPAL"
echo ""
echo "Set these Vercel Production Environment Variables:"
printf 'GCP_PROJECT_ID=%s\n' "$GCP_PROJECT_ID"
printf 'GCP_PROJECT_NUMBER=%s\n' "$GCP_PROJECT_NUMBER"
printf 'GCP_SERVICE_ACCOUNT_EMAIL=%s\n' "$GCP_SERVICE_ACCOUNT_EMAIL"
printf 'GCP_WORKLOAD_IDENTITY_POOL_ID=%s\n' "$WIF_POOL_ID"
printf 'GCP_WORKLOAD_IDENTITY_POOL_PROVIDER_ID=%s\n' "$WIF_PROVIDER_ID"
printf 'AUTO_BOOTSTRAP_ASSETS=1\n'
printf 'AUTO_RUN_FORECAST=0\n'
