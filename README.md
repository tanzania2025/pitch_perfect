# Pitch Perfect 🎤

**AI voice clone + text sentiment and voice tonal analysis**

A comprehensive AI-powered system that combines speech-to-text, sentiment analysis, tonal analysis, and text-to-speech with voice cloning capabilities to create perfect pitch presentations.

## 🚀 Features

- **Speech-to-Text**: Advanced audio transcription using OpenAI Whisper
- **Sentiment Analysis**: Multi-layered text sentiment analysis with NLTK, spaCy, and VADER
- **Tonal Analysis**: Voice pitch, rhythm, and emotional tone analysis
- **LLM Processing**: AI-powered text improvement and prompt generation
- **Voice Cloning**: ElevenLabs integration for realistic voice synthesis
- **End-to-End Pipeline**: Seamless workflow from audio input to improved audio output

## 🏗️ Project Structure

```
pitch_perfect/
├── config/                 # Configuration files and model settings
├── raw_data/              # Raw audio datasets (MELD)
├── data/                  # Processed and intermediate data
├── pitchperfect/          # Core source code package
│   ├── speech_to_text/    # Audio transcription modules
│   ├── text_sentiment_analysis/  # Sentiment analysis
│   ├── tonal_analysis/    # Voice tone and pitch analysis
│   ├── llm_processing/    # AI text improvement
│   ├── text_to_speech/    # Voice synthesis and cloning
│   ├── pipeline/          # End-to-end orchestration
│   └── utils/             # Helper utilities
├── tests/                 # Unit and integration tests
├── notebooks/             # Jupyter notebooks for exploration
├── scripts/               # Setup and utility scripts
├── models/                # Model checkpoints and weights
├── outputs/               # Generated results and logs
├── docs/                  # Documentation
└── docker/                # Containerization files
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- FFmpeg for audio processing

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pitch_perfect
   ```

2. **Set up environment**
   ```bash
   # Using conda
   conda env create -f environment.yml
   conda activate pitch_perfect

   # Or using pip
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Environment setup**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Organize and prepare MELD dataset**
   ```bash
   # Organize MELD data from external to interim folder
   make organize-meld

   # Convert MP4 files to WAV format (requires ffmpeg)
   make meld-to-wav
   ```

5. **Download models and data**
   ```bash
   python scripts/download_models.py
   python scripts/preprocess_meld.py
   ```

## 📊 MELD Dataset Workflow

The project includes a streamlined workflow for processing the MELD (Multimodal EmotionLines Dataset):

1. **Organize**: `make organize-meld` - Copies and organizes MELD.Raw data from `data/external/` to `data/interim/MELD/` with proper train/dev/test splits
2. **Convert**: `make meld-to-wav` - Converts MP4 files to WAV format using ffmpeg, preserving directory structure
3. **Process**: Use the processed WAV files in `data/processed/meld_wav/` for training and analysis

**Requirements**: ffmpeg must be installed and available in PATH for audio conversion.

## 🔧 Configuration

### API Keys Required

- **OpenAI API Key**: For Whisper transcription and GPT models
- **ElevenLabs API Key**: For voice cloning and synthesis
- **Anthropic API Key**: For Claude models (optional)

### Environment Variables

Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
MODEL_CACHE_DIR=./models
DATA_DIR=./data
OUTPUT_DIR=./outputs
```

## 📖 Usage

### Basic Pipeline

```python
from pitchperfect.pipeline.orchestrator import PipelineOrchestrator

# Initialize the pipeline
orchestrator = PipelineOrchestrator()

# Process audio file
result = orchestrator.process_audio(
    audio_path="path/to/audio.wav",
    target_voice="professional",
    improve_content=True
)

# Access results
print(f"Transcript: {result.transcript}")
print(f"Sentiment: {result.sentiment}")
print(f"Improved text: {result.improved_text}")
print(f"Generated audio: {result.output_audio_path}")
```

### Individual Components

#### Speech-to-Text
```python
from pitchperfect.speech_to_text.transcriber import AudioTranscriber

transcriber = AudioTranscriber()
transcript = transcriber.transcribe("audio.wav")
```

#### Sentiment Analysis
```python
from pitchperfect.text_sentiment_analysis.analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment = analyzer.analyze("Your text here")
```

#### Voice Cloning
```python
from pitchperfect.text_to_speech.voice_cloning import VoiceCloner

cloner = VoiceCloner()
audio = cloner.clone_voice("Text to speak", "target_voice_id")
```

## 🧪 Testing

Run the test suite:
```bash
# All tests
pytest

# With coverage
pytest --cov=pitchperfect

# Specific test categories
pytest tests/unit/
pytest tests/integration/
```

## 📊 Jupyter Notebooks

Explore the system capabilities through interactive notebooks:

- `01_data_exploration.ipynb` - Dataset analysis and visualization
- `02_speech_to_text_analysis.ipynb` - Audio transcription experiments
- `03_sentiment_analysis.ipynb` - Sentiment analysis examples
- `04_tonal_analysis.ipynb` - Voice tone analysis
- `05_llm_experiments.ipynb` - AI text improvement
- `06_end_to_end_demo.ipynb` - Complete workflow demonstration

## 🐳 Docker

Run with Docker:
```bash
docker-compose up -d
```

Or build manually:
```bash
docker build -t pitch_perfect .
docker run -it pitch_perfect
```

## 📚 Documentation

- [API Documentation](docs/api_documentation.md)
- [Setup Guide](docs/setup_guide.md)
- [Component Details](docs/component_details.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Set up pre-commit hooks
pre-commit install

# Code formatting
black pitchperfect/ tests/
isort pitchperfect/ tests/
flake8 pitchperfect/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [ElevenLabs](https://elevenlabs.io/) for voice cloning technology
- [MELD Dataset](https://github.com/declare-lab/MELD) for emotional audio data
- [Hugging Face Transformers](https://huggingface.co/transformers) for NLP models

## 📞 Support

For questions and support:
- Open an [issue](../../issues) on GitHub
- Check the [documentation](docs/) for detailed guides
- Review the [notebooks](notebooks/) for examples

---

**Made with ❤️ for perfect pitches everywhere**
