#!/bin/bash
# Fix Cloud Run deployment permission for GitHub Actions

PROJECT_ID="pitchperfect-lewagon"
GITHUB_SA="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"
COMPUTE_SA="792590041292-compute@developer.gserviceaccount.com"

echo "🔧 Fixing Cloud Run deployment permissions"
echo "=========================================="
echo ""

# Set the project
gcloud config set project $PROJECT_ID

echo "📋 Granting serviceAccountUser permission..."
echo "   GitHub Actions SA: $GITHUB_SA"
echo "   Compute SA: $COMPUTE_SA"
echo ""

# Grant permission for GitHub Actions SA to act as the compute SA
gcloud iam service-accounts add-iam-policy-binding $COMPUTE_SA \
    --member="serviceAccount:$GITHUB_SA" \
    --role="roles/iam.serviceAccountUser"

echo "✅ Permission granted!"
echo ""
echo "The GitHub Actions service account can now deploy to Cloud Run."
echo "Re-run the GitHub Actions workflow to complete deployment."