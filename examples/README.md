# Pitch Perfect Examples

This directory contains example scripts demonstrating the functionality of each core module in the Pitch Perfect system.

## 📁 Example Scripts

### Core Modules

1. **[speech_to_text_example.py](speech_to_text_example.py)**
   - Demonstrates audio transcription using OpenAI Whisper
   - Shows transcription results with timestamps and confidence scores
   - Examples of supported audio formats and features

2. **[sentiment_analysis_example.py](sentiment_analysis_example.py)**
   - Analyzes emotional content and sentiment in text
   - Demonstrates emotion detection and confidence scoring
   - Shows practical use cases for sentiment analysis

3. **[tonal_analysis_example.py](tonal_analysis_example.py)**
   - Extracts acoustic features from audio (pitch, pace, volume)
   - Demonstrates prosody analysis and speech pattern detection
   - Identifies common speaking issues and improvements

4. **[llm_processing_example.py](llm_processing_example.py)**
   - Shows text improvement using LLM processing
   - Demonstrates SSML generation for speech synthesis
   - Provides speaking tips and feedback generation

5. **[text_to_speech_example.py](text_to_speech_example.py)**
   - Synthesizes improved speech using ElevenLabs
   - Demonstrates voice cloning capabilities
   - Shows SSML markup for enhanced speech control

6. **[full_pipeline_example.py](full_pipeline_example.py)**
   - Complete end-to-end pipeline demonstration
   - Integrates all modules for comprehensive speech improvement
   - Shows performance metrics and processing flow

## 🚀 Quick Start

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys** in your environment or `.env` file:
   ```bash
   OPENAI_API_KEY=your_openai_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_key_here
   ```

3. **Create required directories:**
   ```bash
   mkdir -p data outputs/generated_audio
   ```

### Running Examples

Each example can be run independently:

```bash
# Run individual module examples
python examples/speech_to_text_example.py
python examples/sentiment_analysis_example.py
python examples/tonal_analysis_example.py
python examples/llm_processing_example.py
python examples/text_to_speech_example.py

# Run complete pipeline demo
python examples/full_pipeline_example.py
```

## 📄 Example Data

For best results, add sample audio files to the `data/` directory:

- **sample_audio.wav** - Any audio file for transcription and tonal analysis
- **voice_sample.wav** - 30-60 second voice sample for cloning
- **sample_speech.wav** - Complete speech sample for full pipeline demo

### Supported Audio Formats
- WAV (recommended)
- MP3
- M4A
- FLAC

## 🎯 What Each Example Demonstrates

### Speech-to-Text
- Audio transcription with timestamps
- Language detection
- Confidence scoring
- Segment-level analysis

### Sentiment Analysis
- Emotion recognition (joy, sadness, anger, fear, etc.)
- Sentiment classification (positive, negative, neutral)
- Confidence scoring for predictions
- Multi-emotion detection

### Tonal Analysis
- Speaking rate analysis (WPM)
- Pitch characteristics (mean, variation, range)
- Volume/energy patterns
- Pause detection and timing
- Acoustic problem identification

### LLM Processing
- Text improvement (removing fillers, improving structure)
- Speaking tip generation
- SSML markup creation
- Prosody adjustment recommendations
- Personalized feedback

### Text-to-Speech
- High-quality speech synthesis
- Voice cloning from samples
- SSML-enhanced speech control
- Multiple output formats

### Full Pipeline
- End-to-end processing workflow
- Integration of all modules
- Performance metrics
- Comprehensive results

## 🔧 Customization

### User Preferences
You can customize processing by modifying user preferences:

```python
user_preferences = {
    "target_style": "professional",  # professional, casual, academic, motivational
    "improvement_focus": "clarity",  # clarity, confidence, engagement, all
    "pace_preference": "slower"      # slower, maintain, faster
}
```

### Mock Data Mode
If API keys are not available, examples will run with mock data to demonstrate expected output formats and functionality.

## 💡 Tips for Best Results

1. **Audio Quality**: Use clear, high-quality audio recordings
2. **Length**: Longer samples (30+ seconds) provide more reliable analysis
3. **Environment**: Minimize background noise
4. **Voice Samples**: For cloning, use 30-60 seconds of varied speech
5. **API Keys**: Ensure valid API keys for full functionality

## 🐛 Troubleshooting

### Common Issues

1. **Missing API Keys**: Examples will show mock data if keys aren't configured
2. **Audio File Not Found**: Examples will create synthetic results for demonstration
3. **Import Errors**: Ensure you're running from the project root directory
4. **Permission Errors**: Check file permissions in `outputs/` directory

### Getting Help

- Check the main project documentation
- Verify API key configuration
- Ensure all dependencies are installed
- Review error messages for specific guidance

## 📚 Next Steps

After running these examples:

1. Try with your own audio files
2. Experiment with different user preferences
3. Explore the FastAPI backend (`app/main.py`)
4. Build custom applications using the individual modules
5. Check out the full pipeline orchestrator for production use

Happy speech improving! 🎤✨