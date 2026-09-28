# ============================================================
# Google Cloud + GitHub setup for Guwahati Urban Prediction
# Run this ONCE in PowerShell after installing gcloud CLI.
#
# Prerequisites:
#   winget install Google.CloudSDK
#   gcloud auth login
#   gcloud config set project vivid-brand-182819
#   gh auth login
# ============================================================

$ErrorActionPreference = "Stop"

$PROJECT_ID     = "vivid-brand-182819"
$REGION         = "asia-south1"
$SERVICE_NAME   = "urban-prediction"
$GAR_REPO       = "urban-prediction"
$SA_NAME        = "github-deploy"
$SA_EMAIL       = "$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
$POOL_NAME      = "github-pool"
$PROVIDER_NAME  = "github-provider"

# Get GitHub repo
$GITHUB_REPO = gh repo view --json nameWithOwner -q .nameWithOwner 2>$null
if (-not $GITHUB_REPO) {
    Write-Error "Not in a GitHub repo or not authenticated. Run 'gh auth login' first."
    exit 1
}
Write-Host "GitHub repo: $GITHUB_REPO" -ForegroundColor Cyan

# ── Step 1: Enable APIs ──
Write-Host "`n=== Step 1: Enable Google Cloud APIs ===" -ForegroundColor Yellow
gcloud services enable `
    run.googleapis.com `
    artifactregistry.googleapis.com `
    iamcredentials.googleapis.com `
    earthengine.googleapis.com `
    --project="$PROJECT_ID"
Write-Host "APIs enabled." -ForegroundColor Green

# ── Step 2: Create Artifact Registry ──
Write-Host "`n=== Step 2: Create Artifact Registry ===" -ForegroundColor Yellow
try {
    gcloud artifacts repositories create $GAR_REPO `
        --repository-format=docker `
        --location="$REGION" `
        --project="$PROJECT_ID" `
        --description="Urban Prediction Docker images"
} catch { Write-Host "Repository already exists." }

# ── Step 3: Create service account ──
Write-Host "`n=== Step 3: Create service account ===" -ForegroundColor Yellow
try {
    gcloud iam service-accounts create $SA_NAME `
        --display-name="GitHub Actions Deploy" `
        --project="$PROJECT_ID"
} catch { Write-Host "Service account already exists." }

foreach ($ROLE in @("roles/run.admin", "roles/artifactregistry.writer", "roles/iam.serviceAccountUser")) {
    gcloud projects add-iam-policy-binding $PROJECT_ID `
        --member="serviceAccount:$SA_EMAIL" `
        --role="$ROLE" `
        --quiet 2>$null
}
Write-Host "Roles granted." -ForegroundColor Green

# ── Step 4: Workload Identity Federation ──
Write-Host "`n=== Step 4: Workload Identity Federation ===" -ForegroundColor Yellow
try {
    gcloud iam workload-identity-pools create $POOL_NAME `
        --location="global" `
        --display-name="GitHub Actions Pool" `
        --project="$PROJECT_ID"
} catch { Write-Host "Pool already exists." }

try {
    gcloud iam workload-identity-pools providers create-oidc $PROVIDER_NAME `
        --location="global" `
        --workload-identity-pool="$POOL_NAME" `
        --display-name="GitHub Provider" `
        --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" `
        --issuer-uri="https://token.actions.githubusercontent.com" `
        --project="$PROJECT_ID"
} catch { Write-Host "Provider already exists." }

$PROJECT_NUMBER = gcloud projects describe $PROJECT_ID --format="value(projectNumber)"
$WIF_PROVIDER = "projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL_NAME/providers/$PROVIDER_NAME"

gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL `
    --project="$PROJECT_ID" `
    --role="roles/iam.workloadIdentityUser" `
    --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL_NAME/attribute.repository/$GITHUB_REPO"

# ── Step 5: Set GitHub secrets ──
Write-Host "`n=== Step 5: Set GitHub secrets ===" -ForegroundColor Yellow
gh secret set WIF_PROVIDER --body "$WIF_PROVIDER"
gh secret set WIF_SERVICE_ACCOUNT --body "$SA_EMAIL"
Write-Host "GitHub secrets set." -ForegroundColor Green

# ── Done ──
Write-Host "`n============================================" -ForegroundColor Green
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT — Register this service account with Earth Engine:" -ForegroundColor Red
Write-Host "  1. Go to https://code.earthengine.google.com/"
Write-Host "  2. Register project '$PROJECT_ID' if not already done"
Write-Host "  3. Allowlist: $SA_EMAIL"
Write-Host ""
Write-Host "Then deploy:" -ForegroundColor Cyan
Write-Host "  git add -A"
Write-Host "  git commit -m 'Add Cloud Run deployment'"
Write-Host "  git push origin main"
Write-Host ""
Write-Host "App URL will be: https://$SERVICE_NAME-*.run.app"
Write-Host ""
Write-Host "WIF_PROVIDER:        $WIF_PROVIDER"
Write-Host "WIF_SERVICE_ACCOUNT: $SA_EMAIL"
