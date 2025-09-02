⏺ Pitch Perfect Codebase Overview

  Architecture: Modular AI pipeline system with 5 core components plus orchestration layer

  Core Components

  1. Speech-to-Text (pitchperfect/speech_to_text/)
    - Uses OpenAI Whisper for transcription
    - Returns text with confidence scores and segments
    - Supports multiple languages and model sizes
  2. Text Sentiment Analysis (pitchperfect/text_sentiment_analysis/)
    - Transformer-based emotion classification
    - Calculates valence/arousal metrics
    - Returns emotion scores and sentiment classification
  3. Tonal Analysis (pitchperfect/tonal_analysis/)
    - Acoustic feature extraction (pitch, rate, energy)
    - Identifies delivery issues (monotone, too fast/slow)
    - Uses librosa and parselmouth for audio analysis
  4. LLM Processing (pitchperfect/llm_processing/)
    - Issue Identifier: Detects content/delivery problems
    - Text Improver: Uses GPT to enhance text clarity
    - Emphasis Identifier: Finds key words/phrases
    - Prosody Calculator: Adjusts speaking parameters
    - SSML Generator: Creates markup for TTS
  5. Text-to-Speech (pitchperfect/text_to_speech/)
    - ElevenLabs integration for voice synthesis
    - Support for voice cloning from samples
    - SSML processing for prosody control

  Pipeline Orchestration

  - PipelineOrchestrator: Manages 5-step workflow
  - Parallel processing where possible
  - Error handling with graceful fallbacks
  - Comprehensive result tracking

  User Interface

  - Gradio web app (app/main.py) with:
    - Voice recording capability
    - File upload support
    - Real-time processing feedback
    - Audio playback and download

  Data & Models

  - MELD emotional dataset integration
  - Model checkpoints and caching
  - Preprocessing utilities for audio conversion

  Technical Stack

  - ML: PyTorch, Transformers, Whisper
  - Audio: librosa, soundfile, parselmouth
  - NLP: spaCy, NLTK, VADER
  - APIs: OpenAI GPT, ElevenLabs TTS
  - Web: Gradio, FastAPI support
  - Testing: pytest with fixtures

  Key Workflow: Audio → Transcribe → Analyze (sentiment+tonal) → Improve (LLM) → Synthesize → Enhanced Audio
