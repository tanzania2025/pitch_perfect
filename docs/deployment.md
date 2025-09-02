# =� Pitch Perfect Backend Deployment Guide

Complete guide to build, test, and deploy the Pitch Perfect FastAPI backend to Google Cloud Run.

## =� Prerequisites

### Required Tools
- **Docker Desktop** - [Install here](https://www.docker.com/products/docker-desktop/)
- **Google Cloud CLI** - [Install here](https://cloud.google.com/sdk/docs/install)
- **Python 3.11+** (for local development)

### Required Accounts & Keys
- **Google Cloud Project** with billing enabled
- **OpenAI API Key** - [Get here](https://platform.openai.com/api-keys)
- **ElevenLabs API Key** - [Get here](https://elevenlabs.io/docs/api-reference/authentication)

## <� Architecture Overview

```
FastAPI Backend � Docker Container � Google Artifact Registry � Google Cloud Run
```

**Key Features:**
- FastAPI backend with async file handling
- Docker containerization for consistent deployment
- Google Cloud Run for serverless scaling
- Secret Manager for secure API key storage
- Multi-stage build for optimized images

## >� Local Testing

### 1. Start Docker Desktop
Make sure Docker Desktop is running:
```bash
docker --version
# Should show: Docker version 20.x.x or higher
```

### 2. Build Docker Image
```bash
cd pitch_perfect
docker build -f docker/Dockerfile -t pitch-perfect-backend:local .
```

### 3. Test Locally with Docker Compose
Create `.env` file:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Run locally:
```bash
docker-compose -f docker/docker-compose.yml up
```

### 4. Test API Endpoints
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Complete API Reference**: See [API Documentation](api_documentation.md)

##  Google Cloud Deployment

### Method 1: Automated Deployment Script (Recommended)

#### 1. Setup Environment
```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"  # or your preferred region
```

#### 2. Run Deployment Script
```bash
chmod +x deploy/deploy.sh
./deploy/deploy.sh
```

The script will:
-  Check requirements
-  Configure Google Cloud
-  Enable required APIs
-  Create Artifact Registry repository
-  Set up Secret Manager for API keys
-  Build and push Docker image
-  Deploy to Cloud Run
-  Test deployment

#### 3. Monitor Deployment
```bash
# View logs
gcloud logs tail --follow --resource=cloud_run_revision --service=pitch-perfect-backend

# Check service status
gcloud run services describe pitch-perfect-backend --region=us-central1
```

### Method 2: Manual Step-by-Step Deployment

#### 1. Enable APIs
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

#### 2. Create Artifact Registry Repository
```bash
gcloud artifacts repositories create pitch-perfect-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Pitch Perfect Backend Images"
```

#### 3. Configure Docker Authentication
```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

#### 4. Create Secrets
```bash
# Create secrets for API keys
echo -n "your-openai-api-key" | gcloud secrets create openai-api-key --data-file=-
echo -n "your-elevenlabs-api-key" | gcloud secrets create elevenlabs-api-key --data-file=-
```

#### 5. Build and Push Image
```bash
# Build image
docker build -f docker/Dockerfile -t us-central1-docker.pkg.dev/$PROJECT_ID/pitch-perfect-repo/pitch-perfect-backend:latest .

# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/$PROJECT_ID/pitch-perfect-repo/pitch-perfect-backend:latest
```

#### 6. Deploy to Cloud Run
```bash
gcloud run deploy pitch-perfect-backend \
    --image=us-central1-docker.pkg.dev/$PROJECT_ID/pitch-perfect-repo/pitch-perfect-backend:latest \
    --region=us-central1 \
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
```

### Method 3: Using Cloud Build (CI/CD)

#### 1. Setup Cloud Build Trigger
```bash
# Connect your repository to Cloud Build
gcloud builds triggers create github \
    --name="pitch-perfect-deploy" \
    --repo-name="pitch_perfect" \
    --repo-owner="your-username" \
    --branch-pattern="^main$" \
    --build-config="deploy/cloudbuild.yaml"
```

#### 2. Trigger Build
Push to main branch or manually trigger:
```bash
gcloud builds submit --config=deploy/cloudbuild.yaml .
```

## >� Testing Your Deployment

### 1. Get Service URL
```bash
SERVICE_URL=$(gcloud run services describe pitch-perfect-backend --region=us-central1 --format="value(status.url)")
echo "Service URL: $SERVICE_URL"
```

### 2. Test Endpoints
```bash
# Health check
curl $SERVICE_URL/health

# API Documentation
open "$SERVICE_URL/docs"

# For detailed API usage examples, see docs/api_documentation.md
```

## =' Configuration & Monitoring

### Environment Variables
Key environment variables set in Cloud Run:
- `PYTHONPATH=/home/app`
- `OPENAI_API_KEY` (from Secret Manager)
- `ELEVENLABS_API_KEY` (from Secret Manager)

### Resource Configuration
- **Memory**: 4Gi (for ML models)
- **CPU**: 2 vCPUs
- **Timeout**: 900 seconds (15 minutes)
- **Concurrency**: 10 requests per instance
- **Max Instances**: 100

### Monitoring
```bash
# View logs
gcloud logs tail --follow --resource=cloud_run_revision --service=pitch-perfect-backend

# Monitor metrics
gcloud monitoring dashboards list
```

## =� Troubleshooting

### Common Issues

#### Docker Build Fails
```bash
# Check Docker daemon
docker info

# Clean build cache
docker system prune -a
```

#### Out of Memory During Build
```bash
# Use smaller base image or multi-stage build
# Already implemented in docker/Dockerfile
```

#### Cloud Run Cold Starts
```bash
# Set minimum instances
gcloud run services update pitch-perfect-backend \
    --min-instances=1 \
    --region=us-central1
```

#### API Key Issues
```bash
# Check secrets
gcloud secrets versions list openai-api-key

# Update secret
echo -n "new-api-key" | gcloud secrets versions add openai-api-key --data-file=-
```

### Performance Optimization

#### 1. Enable Cloud Run CPU Allocation
```bash
gcloud run services update pitch-perfect-backend \
    --cpu-boost \
    --region=us-central1
```

#### 2. Use Cloud Storage for Large Files
- Store processed audio in Cloud Storage
- Return signed URLs instead of direct file responses

#### 3. Implement Caching
- Cache ML model outputs
- Use Cloud Memorystore for frequently accessed data

## =� Cost Optimization

### Cloud Run Pricing
- **CPU**: $0.00002400 per vCPU-second
- **Memory**: $0.00000250 per GiB-second
- **Requests**: $0.40 per million requests

### Optimization Tips
1. **Set max instances** to control costs
2. **Use minimum instances** sparingly
3. **Optimize Docker image** size
4. **Implement request batching**

## = Security Best Practices

### 1. API Authentication (Production)
Remove `--allow-unauthenticated` and implement:
```bash
gcloud run services update pitch-perfect-backend \
    --no-allow-unauthenticated \
    --region=us-central1
```

### 2. VPC Connector (Optional)
For private network access:
```bash
gcloud run services update pitch-perfect-backend \
    --vpc-connector=your-connector \
    --vpc-egress=private-ranges-only \
    --region=us-central1
```

### 3. IAM Roles
Grant minimal required permissions:
```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:your-service@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## =� API Endpoints

The deployed FastAPI backend provides these endpoints:

### Core Endpoints
- `GET /` - Health check
- `GET /health` - Detailed health check
- `GET /docs` - Interactive API documentation
- `GET /redoc` - Alternative API documentation

### Processing Endpoints
- `POST /process-audio` - Main audio processing endpoint
  - **Parameters**:
    - `audio_file` (required): Audio file to process
    - `voice_sample` (optional): Voice sample for cloning
    - `target_style`: professional, casual, academic, motivational
    - `improvement_focus`: all, clarity, confidence, engagement
    - `save_audio`: Whether to save generated audio

- `GET /download-audio/{filename}` - Download processed audio files
- `POST /cleanup` - Clean up temporary files
- `GET /config` - Get current configuration (secrets hidden)

### Example API Usage
```bash
# Process audio with curl
curl -X POST "https://your-service-url/process-audio" \
     -H "Content-Type: multipart/form-data" \
     -F "audio_file=@speech.wav" \
     -F "target_style=professional" \
     -F "improvement_focus=clarity"
```

## 📦 Deployed Service

The deployed FastAPI backend provides:

- **Interactive API Documentation**: `https://your-service-url/docs`
- **Health Check**: `https://your-service-url/health`
- **Complete API Reference**: See [API Documentation](api_documentation.md) for detailed endpoint information, request/response formats, and integration examples

## 🎯 Next Steps

1. **Set up monitoring** and alerting
2. **Implement CI/CD** pipeline
3. **Add load testing**
4. **Configure custom domain**
5. **Set up staging environment**
6. **Connect frontend** application

## 📞 Support

- **Logs**: `gcloud logs tail --follow --resource=cloud_run_revision --service=pitch-perfect-backend`
- **Status**: Check Google Cloud Console → Cloud Run
- **API Documentation**: Visit your service URL + `/docs` or see [api_documentation.md](api_documentation.md)
- **Configuration**: Check `config/config.yaml` for settings

---

**Deployment Complete!** Your Pitch Perfect backend is now running on Google Cloud Run with automatic scaling and managed infrastructure.
