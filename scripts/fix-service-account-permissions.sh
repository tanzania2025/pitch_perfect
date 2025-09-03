#!/bin/bash
# Fix service account permissions for GitHub Actions

PROJECT_ID="pitchperfect-lewagon"
SERVICE_ACCOUNT="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"

echo "🔧 Updating permissions for service account: $SERVICE_ACCOUNT"
echo "=================================================="

# Set the project
gcloud config set project $PROJECT_ID

# Grant required roles
echo "📋 Granting required roles..."

# Service Usage Admin - to enable APIs
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/serviceusage.serviceUsageAdmin"

# Artifact Registry Writer
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/artifactregistry.writer"

# Cloud Run Admin
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/run.admin"

# Service Account User
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/iam.serviceAccountUser"

# Additional permission to act as the default compute service account
COMPUTE_SA="792590041292-compute@developer.gserviceaccount.com"
gcloud iam service-accounts add-iam-policy-binding $COMPUTE_SA \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/iam.serviceAccountUser"

# Secret Manager Secret Accessor
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

# Cloud Build Editor (if using Cloud Build)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/cloudbuild.builds.editor"

echo "✅ Permissions updated successfully!"
echo ""
echo "The service account now has:"
echo "- Service Usage Admin (to enable APIs)"
echo "- Artifact Registry Writer"
echo "- Cloud Run Admin"
echo "- Service Account User"
echo "- Secret Manager Secret Accessor"
echo "- Cloud Build Editor"