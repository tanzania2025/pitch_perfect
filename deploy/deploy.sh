#!/bin/bash

# Pitch Perfect Backend Deployment Script for Google Cloud Platform
set -e

# Configuration - Load from .env if available
if [ -f .env ]; then
    source .env
fi

PROJECT_ID=${PROJECT_ID:-"pitchperfect-lewagon"}
REGION=${REGION:-"europe-west1"}
REPOSITORY=${REPOSITORY:-"pitch-perfect-repo"}
SERVICE_NAME=${SERVICE_NAME:-"pitch-perfect-backend"}
IMAGE_NAME="pitch-perfect-backend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Pitch Perfect Backend Deployment${NC}"

# Check if required tools are installed
check_requirements() {
    echo -e "${YELLOW}📋 Checking requirements...${NC}"

    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}❌ Google Cloud CLI not found. Please install: https://cloud.google.com/sdk/docs/install${NC}"
        exit 1
    fi

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found. Please install Docker.${NC}"
        exit 1
    fi

    # Check if gcloud is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        echo -e "${RED}❌ Not authenticated with Google Cloud. Please run: gcloud auth login${NC}"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ Docker daemon is not running. Please start Docker Desktop.${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ All requirements satisfied${NC}"
}

# Set up Google Cloud project
setup_gcloud() {
    echo -e "${YELLOW}🔧 Setting up Google Cloud project...${NC}"

    if [ "$PROJECT_ID" = "your-project-id" ]; then
        echo -e "${RED}❌ Please set PROJECT_ID environment variable or update the script${NC}"
        echo "   export PROJECT_ID=your-actual-project-id"
        exit 1
    fi

    echo "Setting project: $PROJECT_ID"
    gcloud config set project $PROJECT_ID

    echo "Configuring Docker authentication for $REGION-docker.pkg.dev..."
    # Use --quiet flag to avoid interactive prompts
    if ! gcloud auth configure-docker $REGION-docker.pkg.dev --quiet; then
        echo -e "${RED}❌ Docker authentication failed. Trying alternative method...${NC}"
        # Alternative: configure docker directly
        gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin $REGION-docker.pkg.dev
    fi

    echo -e "${GREEN}✅ Google Cloud configured${NC}"
}

# Create Artifact Registry repository
create_repository() {
    echo -e "${YELLOW}📦 Creating Artifact Registry repository...${NC}"

    # Check if repository exists
    if gcloud artifacts repositories describe $REPOSITORY --location=$REGION &> /dev/null; then
        echo -e "${GREEN}✅ Repository $REPOSITORY already exists${NC}"
    else
        gcloud artifacts repositories create $REPOSITORY \
            --repository-format=docker \
            --location=$REGION \
            --description="Pitch Perfect Backend Docker Images"
        echo -e "${GREEN}✅ Repository created: $REPOSITORY${NC}"
    fi
}

# Build and push Docker image
build_and_push() {
    echo -e "${YELLOW}🔨 Building Docker image...${NC}"

    # Full image path
    IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest"

    # Build the image
    docker build --platform linux/amd64 -f docker/Dockerfile -t $IMAGE_URI .

    echo -e "${YELLOW}📤 Pushing image to Artifact Registry...${NC}"
    docker push $IMAGE_URI

    echo -e "${GREEN}✅ Image pushed: $IMAGE_URI${NC}"
}

# Enable required APIs
enable_apis() {
    echo -e "${YELLOW}🔌 Enabling required Google Cloud APIs...${NC}"

    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable run.googleapis.com
    gcloud services enable artifactregistry.googleapis.com
    gcloud services enable secretmanager.googleapis.com

    echo -e "${GREEN}✅ APIs enabled${NC}"
}

# Create secrets for API keys
create_secrets() {
    echo -e "${YELLOW}🔐 Setting up secrets...${NC}"

    # Check if secrets exist, create if not
    secrets=("openai-api-key" "elevenlabs-api-key")

    for secret in "${secrets[@]}"; do
        if gcloud secrets describe $secret &> /dev/null; then
            echo -e "${GREEN}✅ Secret $secret already exists${NC}"
        else
            echo -e "${YELLOW}Creating secret: $secret${NC}"
            echo "Please enter your ${secret//-/ } (input will be hidden):"
            read -s secret_value
            if [ -n "$secret_value" ]; then
                echo -n "$secret_value" | gcloud secrets create $secret --data-file=-
                echo -e "${GREEN}✅ Secret $secret created${NC}"
            else
                echo -e "${YELLOW}⚠️  Skipping empty secret: $secret${NC}"
            fi
        fi
    done
}

# Deploy to Cloud Run
deploy_cloudrun() {
    echo -e "${YELLOW}☁️  Deploying to Cloud Run...${NC}"

    IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest"

    gcloud run deploy $SERVICE_NAME \
        --image=$IMAGE_URI \
        --region=$REGION \
        --platform=managed \
        --allow-unauthenticated \
        --memory=4Gi \
        --cpu=2 \
        --timeout=900 \
        --concurrency=10 \
        --max-instances=100 \
        --set-env-vars="PYTHONPATH=/home/app" \
        --set-secrets="OPENAI_API_KEY=openai-api-key:latest" \
        --set-secrets="ELEVENLABS_API_KEY=elevenlabs-api-key:latest"

    # Get the service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

    echo -e "${GREEN}🎉 Deployment complete!${NC}"
    echo -e "${GREEN}📍 Service URL: $SERVICE_URL${NC}"
    echo -e "${GREEN}📚 API Docs: $SERVICE_URL/docs${NC}"
    echo -e "${GREEN}🏥 Health Check: $SERVICE_URL/health${NC}"
}

# Test deployment
test_deployment() {
    echo -e "${YELLOW}🧪 Testing deployment...${NC}"

    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

    # Test health endpoint
    if curl -f "$SERVICE_URL/health" &> /dev/null; then
        echo -e "${GREEN}✅ Health check passed${NC}"
    else
        echo -e "${RED}❌ Health check failed${NC}"
        exit 1
    fi
}

# Main deployment flow
main() {
    check_requirements
    setup_gcloud
    
    enable_apis
    
    create_repository
    
    create_secrets
    
    build_and_push
    deploy_cloudrun
    test_deployment

    echo -e "${GREEN}🎊 Deployment completed successfully!${NC}"
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Test your API at: $(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")/docs"
    echo "2. Monitor logs: gcloud logs tail --follow --resource=cloud_run_revision --service=$SERVICE_NAME"
    echo "3. Update API keys in Secret Manager if needed"
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "build-only")
        check_requirements
        setup_gcloud
        create_repository
        build_and_push
        ;;
    "secrets-only")
        check_requirements
        setup_gcloud
        enable_apis
        create_secrets
        ;;
    *)
        echo "Usage: $0 [deploy|build-only|secrets-only]"
        echo "  deploy      - Full deployment (default)"
        echo "  build-only  - Only build and push Docker image"
        echo "  secrets-only - Only create/update secrets"
        exit 1
        ;;
esac
