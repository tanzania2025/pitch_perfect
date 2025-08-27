.PHONY: help setup install clean test lint format docker-build docker-run

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Initial project setup
	@echo "Setting up the project..."
	cp .env.example .env
	mkdir -p data/{processed,interim,external}
	mkdir -p models/{checkpoints,pretrained,custom}
	mkdir -p outputs/{transcripts,analysis_results,generated_audio,logs}
	mkdir -p tests/fixtures
	@echo "Project structure created. Please edit .env file with your API keys."

install: ## Install dependencies
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	@echo "Dependencies installed successfully."

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install pre-commit
	pre-commit install
	python -m spacy download en_core_web_sm
	@echo "Development environment ready."

clean: ## Clean temporary files and cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	@echo "Cleaned temporary files."

test: ## Run tests
	pytest tests/ -v --cov=pitchperfect --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

lint: ## Run linting
	flake8 pitchperfect/ tests/
	black --check pitchperfect/ tests/
	isort --check-only pitchperfect/ tests/


format: ## Format code
	black pitchperfect/ tests/
	isort pitchperfect/ tests/
	@echo "Code formatted successfully."

preprocess-data: ## Preprocess MELD dataset
	python scripts/preprocess_meld.py

download-models: ## Download required models
	python scripts/download_models.py

demo: ## Run demo script
	python scripts/demo.py

docker-build: ## Build Docker image
	docker-compose -f docker/docker-compose.yml build

docker-run: ## Run Docker container
	docker-compose -f docker/docker-compose.yml up

docker-dev: ## Run Docker in development mode
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up

notebook: ## Start Jupyter notebook server
	jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser

# Team-specific targets
stt-dev: ## Development environment for Speech-to-Text team
	@echo "Setting up STT development environment..."
	pip install whisper-openai librosa soundfile

sentiment-dev: ## Development environment for Sentiment Analysis team
	@echo "Setting up Sentiment Analysis development environment..."
	python -m spacy download en_core_web_sm
	pip install transformers torch

tonal-dev: ## Development environment for Tonal Analysis team
	@echo "Setting up Tonal Analysis development environment..."
	pip install parselmouth python_speech_features librosa

llm-dev: ## Development environment for LLM team
	@echo "Setting up LLM development environment..."
	pip install openai anthropic langchain tiktoken

tts-dev: ## Development environment for TTS team
	@echo "Setting up TTS development environment..."
	pip install elevenlabs pydub

# Utility targets
check-env: ## Check if environment variables are set
	@echo "Checking environment variables..."
	@python - <<-'PY'
	import os
	required_vars = ['OPENAI_API_KEY', 'ELEVENLABS_API_KEY']
	optional_vars = ['ANTHROPIC_API_KEY']
	missing_required = [var for var in required_vars if not os.getenv(var)]
	missing_optional = [var for var in optional_vars if not os.getenv(var)]
	if missing_required:
	    print('❌ Missing required environment variables:', ', '.join(missing_required))
	    raise SystemExit(1)
	if missing_optional:
	    print('⚠️  Missing optional environment variables:', ', '.join(missing_optional))
	print('✅ Environment variables check passed')
	PY

pipeline-test: ## Test the complete pipeline
	python - <<-'PY'
	from src.pipeline.orchestrator import MainPipeline
	pipeline = MainPipeline()
	print('✅ Pipeline initialization successful')
	PY

# Database/Data management
backup-models: ## Backup trained models
	tar -czf models_backup_$$(date +%Y%m%d_%H%M%S).tar.gz models/

restore-models: ## Restore models from backup (specify BACKUP_FILE)
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Please specify BACKUP_FILE=backup.tar.gz"; exit 1; fi
	tar -xzf $(BACKUP_FILE)
