pitch_perfect/
│
├── README.md
├── requirements.txt
├── environment.yml
├── .env.example
├── .gitignore
├── setup.py
├── Makefile
│
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── model_configs.py
│
├── raw_data/
│   ├── MELD/
│   │   ├── train/
│   │   ├── dev/
│   │   └── test/
│   └── README.md
│
├── data/
│   ├── __init__.py
│   ├── processed/
│   │   └── meld_wav/          # Converted WAV files from MELD
│   ├── interim/
│   │   └── MELD/              # Organized MELD splits (train/dev/test)
│   └── external/
│       └── MELD.Raw/          # Original MELD dataset
│
├── pitchperfect/
│   ├── __init__.py
│   │
│   ├── speech_to_text/
│   │   ├── __init__.py
│   │   └── transcriber.py
│   │
│   ├── text_sentiment_analysis/
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   ├── models.py
│   │   └── preprocessing.py
│   │
│   ├── tonal_analysis/
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   ├── feature_extraction.py
│   │   └── models.py
│   │
│   ├── llm_processing/
│   │   ├── __init__.py
│   │   ├── prompt_generator.py
│   │   ├── text_improver.py          # AI text improvement & prosody analysis
│   │   ├── identify_issues.py        # Issue detection for speech & text
│   │   ├── prosody_guide.py          # Prosody calculation & guidance
│   │   ├── helper_functions.py       # Utility functions for text processing
│   │   ├── ssml_generate.py          # SSML markup generation
│   │   └── templates.py
│   │
│   ├── text_to_speech/
│   │   ├── __init__.py
│   │   ├── elevenlabs_client.py
│   │   ├── voice_cloning.py
│   │   └── synthesis.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── data_flow.py
│   │   └── validators.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── audio_processing.py
│   │   ├── file_handlers.py
│   │   ├── logging_config.py
│   │   └── visualization.py
│   │
│   └── data/
│       ├── __init__.py
│       └── meld_loader.py      # MELD dataset loader and utilities
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_speech_to_text.py
│   │   ├── test_sentiment_analysis.py
│   │   ├── test_tonal_analysis.py
│   │   ├── test_llm_processing.py
│   │   └── test_text_to_speech.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_end_to_end.py
│   └── fixtures/
│       ├── sample_audio.wav
│       └── test_data.json
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_speech_to_text_analysis.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_tonal_analysis.ipynb
│   ├── 05_llm_experiments.ipynb
│   └── 06_end_to_end_demo.ipynb
│
├── scripts/
│   ├── setup_environment.sh
│   ├── download_models.py
│   ├── preprocess_meld.py
│   ├── organize_meld.py        # Organize MELD data from external to interim
│   ├── train_models.py
│   └── demo.py
│
├── models/
│   ├── checkpoints/
│   ├── pretrained/
│   └── custom/
│
├── outputs/
│   ├── transcripts/
│   ├── analysis_results/
│   ├── generated_audio/
│   └── logs/
│
├── docs/
│   ├── api_documentation.md
│   ├── setup_guide.md
│   ├── component_details.md
│   └── deployment.md
│
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements-docker.txt
