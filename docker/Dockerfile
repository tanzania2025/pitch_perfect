# Multi-stage build for Pitch Perfect Backend
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio processing dependencies
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libasound2-dev \
    # Build dependencies
    build-essential \
    git \
    wget \
    curl \
    # Additional dependencies for ML libraries
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy requirements first for better Docker layer caching
COPY --chown=app:app requirements.txt ./

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Download spaCy model (required for NLP processing)
RUN python -m spacy download en_core_web_sm

# Copy only essential application code
COPY --chown=app:app pitchperfect/ ./pitchperfect/
COPY --chown=app:app app/ ./app/
COPY --chown=app:app config/ ./config/
COPY --chown=app:app setup.py ./


# Create necessary directories
RUN mkdir -p outputs/logs outputs/generated_audio data/processed models/pretrained

# Install the package in development mode
RUN pip install --user -e .

# Add the user's local bin to PATH
ENV PATH="/home/app/.local/bin:${PATH}"

# Expose the FastAPI port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command to run the FastAPI server
CMD ["python", "app/main.py"]
