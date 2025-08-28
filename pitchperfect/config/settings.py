import os
from pathlib import Path

# Get the project root directory (2 levels up from this config file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# MELD specific paths
MELD_RAW_DIR = EXTERNAL_DATA_DIR / "MELD.Raw"
MELD_INTERIM_DIR = INTERIM_DATA_DIR / "MELD"
MELD_WAV_DIR = PROCESSED_DATA_DIR / "meld_wav"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
ANALYSIS_RESULTS_DIR = OUTPUT_DIR / "analysis_results"
GENERATED_AUDIO_DIR = OUTPUT_DIR / "generated_audio"
LOGS_DIR = OUTPUT_DIR / "logs"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
CUSTOM_MODELS_DIR = MODELS_DIR / "custom"

# Ensure directories exist
for directory in [DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
