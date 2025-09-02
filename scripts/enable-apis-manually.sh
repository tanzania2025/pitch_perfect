#!/bin/bash
# Manually enable required APIs for Pitch Perfect

PROJECT_ID="pitchperfect-lewagon"

echo "🔌 Enabling required Google Cloud APIs for $PROJECT_ID"
echo "===================================================="

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📋 Enabling APIs..."

gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable secretmanager.googleapis.com

echo "✅ APIs enabled successfully!"
echo ""
echo "Enabled APIs:"
echo "- Cloud Build API"
echo "- Cloud Run API"
echo "- Artifact Registry API"
echo "- Secret Manager API"