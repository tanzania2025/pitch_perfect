# Pitch Perfect 🎤

**AI voice clone + text sentiment and voice tonal analysis**

A comprehensive AI-powered system that combines speech-to-text, sentiment analysis, tonal analysis, and text-to-speech with voice cloning capabilities to create perfect pitch presentations.

## 🚀 Features

- **Speech-to-Text**: Advanced audio transcription using OpenAI Whisper
- **Sentiment Analysis**: Multi-layered text sentiment analysis with NLTK, spaCy, and VADER
- **Tonal Analysis**: Voice pitch, rhythm, and emotional tone analysis
- **LLM Processing**: AI-powered text improvement, prosody analysis, and speech enhancement
  - Text improvement using OpenAI GPT models
  - Prosody analysis and speaking rate optimization
  - Emphasis word identification and pause point detection
  - Volume and pitch adjustment recommendations
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

## 🎯 Enhanced LLM Processing

The system now includes comprehensive AI-powered text improvement and prosody analysis:

### Text Improvement
- **Issue Detection**: Automatically identifies filler words, poor structure, and emotional disengagement
- **AI Enhancement**: Uses OpenAI GPT models to improve text clarity, structure, and engagement
- **Smart Filtering**: Removes unnecessary phrases while maintaining original meaning

### Prosody Analysis
- **Speaking Rate Optimization**: Analyzes and recommends optimal speaking speeds
- **Pitch Variation**: Detects monotone speech and suggests pitch adjustments
- **Emphasis Identification**: Highlights key words, numbers, and power phrases
- **Pause Point Detection**: Identifies natural break points for better speech flow
- **Volume Control**: Recommends volume adjustments based on energy levels

### Integration
- **Seamless Workflow**: Integrates with the complete pipeline from transcription to synthesis
- **Customizable**: Configurable parameters for different speaking styles and contexts
- **Error Handling**: Graceful fallback to original text if improvement fails

## 📊 MELD Dataset Workflow

The project includes a streamlined workflow for processing the MELD (Multimodal EmotionLines Dataset):

1. **Organize**: `make organize-meld` - Copies and organizes MELD.Raw data from `data/external/` to `data/interim/MELD/` with proper train/dev/test splits
2. **Convert**: `make meld-to-wav` - Converts MP4 files to WAV format using ffmpeg, preserving directory structure
3. **Process**: Use the processed WAV files in `data/processed/meld_wav/` for training and analysis

**Requirements**: ffmpeg must be installed and available in PATH for audio conversion.

### Data Loading Functions

```python
from pitchperfect.data.meld_loader import process_meld_split

# Lazy loading (memory efficient)
for wav_path in process_meld_split(split='train', mode='lazy'):
    process_audio(wav_path)

# Batch loading (get all results at once)
results = process_meld_split(split='dev', mode='batch')
for mp4_path, wav_path in results:
    print(f"{mp4_path.name} -> {wav_path.name}")
```

**Modes:**
- `mode='lazy'` (default): Returns an iterator, processes files one at a time
- `mode='batch'`: Returns a list of (mp4_path, wav_path) tuples

## 🔧 Configuration

### API Keys Required

- **OpenAI API Key**: For Whisper transcription, GPT models, and LLM text processing
- **ElevenLabs API Key**: For voice cloning and synthesis

### Environment Variables

Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
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

#### LLM Processing
```python
from pitchperfect.llm_processing.text_improver import ImprovementGenerator

# Initialize with your API key
improver = ImprovementGenerator(api_key="your-openai-api-key")

# Generate improvements
improved_text, prosody_guide = improver.generate_improvements(
    text="Your text here",
    text_sentiment={'emotion': 'neutral', 'score': 0.85},
    acoustic_features={
        'pitch_mean': 150,
        'speaking_rate': 3.2,
        'energy': 0.05,
        'pause_ratio': 0.15,
        'monotone_score': 0.7
    }
)

print(f"Improved text: {improved_text}")
print(f"Prosody guide: {prosody_guide}")
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
- `05_llm_experiments.ipynb` - AI text improvement and prosody analysis
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
