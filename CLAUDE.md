# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pitch Perfect is an AI-powered speech analysis and improvement system that processes audio through a 5-stage pipeline: speech-to-text (Whisper), sentiment analysis, tonal analysis, LLM-powered text improvement (GPT-4), and text-to-speech synthesis (ElevenLabs). The system features voice cloning capabilities and prosody analysis for presentation enhancement.

## Common Commands

### Development
```bash
# Install dependencies and setup
make install

# Code quality and formatting
make lint          # Run flake8, black, isort
make format        # Format code with black and isort
make check         # Run all quality checks

# Testing
make test          # Run pytest tests
python -m pytest tests/  # Run specific test directory

# Run the FastAPI application
make run-api       # Start the API server
uvicorn pitchperfect.api.main:app --reload  # Development server with auto-reload
```

### Docker Development
```bash
# Build and run with Docker
make docker-build
make docker-run

# Using docker-compose for full stack
docker-compose up --build
```

### Cloud Deployment
```bash
# Deploy to Google Cloud Run
make deploy

# Build for cloud deployment
make cloud-build

# Setup cloud permissions
bash scripts/fix-service-account-permissions.sh
```

### Data Processing
```bash
# Download and process MELD dataset
make download-meld-data
make process-meld-data
```

## Architecture

### Pipeline Components
The system follows a modular architecture with 5 core components plus orchestration:

1. **Speech-to-Text** (`pitchperfect/speech_to_text/`) - Whisper-based audio transcription
2. **Text Sentiment Analysis** (`pitchperfect/text_sentiment_analysis/`) - VADER/transformers-based emotion classification
3. **Tonal Analysis** (`pitchperfect/tonal_analysis/`) - Prosody and acoustic feature extraction using librosa/parselmouth
4. **LLM Processing** (`pitchperfect/llm_processing/`) - GPT-4 powered content improvement and SSML generation
5. **Text-to-Speech** (`pitchperfect/text_to_speech/`) - ElevenLabs voice cloning and synthesis
6. **Pipeline Orchestration** (`pitchperfect/pipeline/`) - End-to-end workflow coordination

### Key Patterns
- **Dependency Injection**: All modules accept optional config dictionaries for flexibility
- **Configuration-Driven**: YAML-based settings (`config/config.yaml`) with environment variable substitution
- **Lazy Loading**: Memory-efficient data processing with on-demand model loading
- **Async Support**: FastAPI endpoints with async file handling

### Technology Stack
- **Backend**: FastAPI with Pydantic models
- **AI/ML**: OpenAI (Whisper, GPT-4), ElevenLabs, PyTorch, Transformers, NLTK, spaCy
- **Audio Processing**: librosa, parselmouth for prosody analysis
- **Infrastructure**: Docker, Google Cloud Run, Cloud Build, Artifact Registry
- **Development**: Black, flake8, isort, pytest, pre-commit hooks

## Configuration

### Environment Variables
Key environment variables needed:
- `OPENAI_API_KEY` - For Whisper and GPT-4 APIs
- `ELEVENLABS_API_KEY` - For voice cloning and TTS
- `GOOGLE_APPLICATION_CREDENTIALS` - For GCP services (when not using Cloud Run)

### Config Structure
Configuration is managed through `config/config.yaml` with sections for each pipeline component. Environment variables can override any config value using the format `PITCH_PERFECT_<SECTION>_<KEY>`.

## Development Notes

### Code Quality
- Use Black for formatting (line length 88)
- flake8 for linting with specific ignore rules
- isort for import sorting
- Type hints required for all public functions

### Testing
- Test files exist in `tests/` but are currently empty - implementation needed
- Use pytest framework with fixtures for component testing
- Mock external APIs (OpenAI, ElevenLabs) in tests

### Docker Development
- Multi-stage Dockerfile optimizes for production deployment
- Non-root user (uid=1000) for security
- Health check endpoint at `/health`
- Supports both local development and cloud deployment

### Cloud Deployment
- Deploys to Google Cloud Run in europe-west1
- Uses Cloud Build for automated CI/CD
- Service account permissions managed via scripts
- Secrets stored in Google Secret Manager

## Error Handling Guidelines

When errors are shown or fixes are needed:
1. Make fixes directly in the code
2. Write as few new lines as possible
3. Prefer modifying existing code over creating new files

## Feature Design Guidelines

When designing new features or modules:
1. Start with the simplest architecture that works
2. Present the basic implementation first
3. Ask if the user wants options for more complex/comprehensive versions
4. Only add complexity when explicitly requested