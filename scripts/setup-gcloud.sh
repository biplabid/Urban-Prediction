#!/bin/bash
# ============================================================
# Google Cloud + GitHub setup for Guwahati Urban Prediction
# Run this ONCE to configure everything.
# Prerequisites: gcloud CLI, gh CLI, both authenticated
# ============================================================

set -euo pipefail

PROJECT_ID="vivid-brand-182819"
REGION="asia-south1"
SERVICE_NAME="guwahati-urban-prediction"
GAR_REPO="guwahati-urban-prediction"
SA_NAME="github-deploy"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
POOL_NAME="github-pool"
PROVIDER_NAME="github-provider"

# Get GitHub repo from current directory
GITHUB_REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo "")
if [ -z "$GITHUB_REPO" ]; then
  echo "ERROR: Not in a GitHub repo. Run 'gh repo create' first."
  exit 1
fi
echo "GitHub repo: $GITHUB_REPO"

echo ""
echo "=== Step 1: Enable Google Cloud APIs ==="
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com \
  earthengine.googleapis.com \
  --project="$PROJECT_ID"
echo "APIs enabled."

echo ""
echo "=== Step 2: Create Artifact Registry repository ==="
gcloud artifacts repositories create "$GAR_REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --project="$PROJECT_ID" \
  --description="Guwahati Urban Prediction Docker images" \
  2>/dev/null || echo "Repository already exists."

echo ""
echo "=== Step 3: Create service account for GitHub Actions ==="
gcloud iam service-accounts create "$SA_NAME" \
  --display-name="GitHub Actions Deploy" \
  --project="$PROJECT_ID" \
  2>/dev/null || echo "Service account already exists."

# Grant roles
for ROLE in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$ROLE" \
    --quiet
done

# Register the SA with Earth Engine (needed for GEE API access)
echo ""
echo "=== Step 4: Register service account with Earth Engine ==="
echo "IMPORTANT: Go to https://code.earthengine.google.com/"
echo "  -> Assets tab -> Register a new Cloud project"
echo "  -> Make sure project '$PROJECT_ID' is registered"
echo "  -> Then allowlist the service account: $SA_EMAIL"
echo ""
echo "Alternatively, run in Python:"
echo "  import ee"
echo "  ee.data.createServiceAccountKey('${SA_EMAIL}')"
echo ""

echo "=== Step 5: Set up Workload Identity Federation ==="
# Create pool
gcloud iam workload-identity-pools create "$POOL_NAME" \
  --location="global" \
  --display-name="GitHub Actions Pool" \
  --project="$PROJECT_ID" \
  2>/dev/null || echo "Pool already exists."

# Create provider
gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_NAME" \
  --location="global" \
  --workload-identity-pool="$POOL_NAME" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --project="$PROJECT_ID" \
  2>/dev/null || echo "Provider already exists."

# Allow GitHub repo to impersonate service account
WIF_PROVIDER="projects/$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')/locations/global/workloadIdentityPools/${POOL_NAME}/providers/${PROVIDER_NAME}"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --project="$PROJECT_ID" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')/locations/global/workloadIdentityPools/${POOL_NAME}/attribute.repository/${GITHUB_REPO}"

echo ""
echo "=== Step 6: Set GitHub secrets ==="
gh secret set WIF_PROVIDER --body "$WIF_PROVIDER"
gh secret set WIF_SERVICE_ACCOUNT --body "$SA_EMAIL"
echo "GitHub secrets set."

echo ""
echo "=== Step 7: Create Earth Engine service account key ==="
echo "For Cloud Run to authenticate with GEE, you need ONE of:"
echo ""
echo "  Option A (Recommended): Use default credentials"
echo "    Grant the Cloud Run service account Earth Engine access:"
echo "    gcloud run services update $SERVICE_NAME --region=$REGION \\"
echo "      --service-account=${SA_EMAIL}"
echo ""
echo "  Option B: Use a key file"
echo "    gcloud iam service-accounts keys create service-account-key.json \\"
echo "      --iam-account=${SA_EMAIL}"
echo "    Then add it as a secret and mount in Cloud Run."
echo ""

echo ""
echo "============================================"
echo "SETUP COMPLETE!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Register $SA_EMAIL with Earth Engine (Step 4 above)"
echo "  2. git add -A && git commit -m 'Initial deployment setup'"
echo "  3. git push origin main"
echo "  4. GitHub Actions will build and deploy to Cloud Run"
echo "  5. Your app URL: https://${SERVICE_NAME}-*.run.app"
echo ""
echo "WIF_PROVIDER: $WIF_PROVIDER"
echo "WIF_SERVICE_ACCOUNT: $SA_EMAIL"
