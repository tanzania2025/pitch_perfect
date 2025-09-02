# =Ú Pitch Perfect API Documentation

Complete reference for the Pitch Perfect FastAPI backend endpoints, request/response formats, and integration examples.

## < Base URL

- **Local Development**: `http://localhost:8000`
- **Production**: `https://your-service-url` (from Cloud Run deployment)

## = Interactive Documentation

Once deployed, visit these URLs for interactive API exploration:

- **Swagger UI**: `{BASE_URL}/docs`
- **ReDoc**: `{BASE_URL}/redoc`
- **OpenAPI Schema**: `{BASE_URL}/openapi.json`

## =Í API Endpoints

### Health & Status Endpoints

#### `GET /`
Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.123456",
  "version": "0.1.0"
}
```

#### `GET /health`
Detailed health check with service information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.123456",
  "version": "0.1.0"
}
```

### Core Processing Endpoints

#### `POST /process-audio`
Main endpoint for speech analysis and improvement.

**Content-Type:** `multipart/form-data`

**Parameters:**
- `audio_file` *(required)*: Audio file to process
  - **Format**: WAV, MP3, M4A, FLAC
  - **Max Size**: 50MB
  - **Duration**: 5-300 seconds recommended
- `voice_sample` *(optional)*: Voice sample for cloning
  - **Format**: Same as audio_file
  - **Duration**: 10-60 seconds recommended
- `target_style` *(optional)*: Target speaking style
  - **Options**: `professional`, `casual`, `academic`, `motivational`
  - **Default**: `professional`
- `improvement_focus` *(optional)*: Areas to focus on
  - **Options**: `all`, `clarity`, `confidence`, `engagement`
  - **Default**: `all`
- `save_audio` *(optional)*: Whether to save generated audio
  - **Type**: boolean
  - **Default**: `true`

**Example Request:**
```bash
curl -X POST "http://localhost:8000/process-audio" \
     -H "Content-Type: multipart/form-data" \
     -F "audio_file=@speech.wav" \
     -F "target_style=professional" \
     -F "improvement_focus=clarity" \
     -F "save_audio=true"
```

**Response:**
```json
{
  "session_id": "session_20240115_103045",
  "processing_status": "completed",
  "timestamp": "2024-01-15T10:30:45.123456",
  "input_audio": "/tmp/audio_20240115_103045.wav",
  
  "transcription": {
    "text": "Hello everyone, welcome to today's presentation about artificial intelligence.",
    "confidence": 0.95,
    "segments": [
      {
        "start": 0.0,
        "end": 2.5,
        "text": "Hello everyone",
        "confidence": 0.98
      }
    ],
    "language": "en"
  },
  
  "sentiment": {
    "emotion": "neutral",
    "confidence": 0.87,
    "sentiment": "positive",
    "valence": 0.15,
    "arousal": 0.45,
    "emotion_scores": {
      "neutral": 0.65,
      "joy": 0.20,
      "confidence": 0.10,
      "sadness": 0.05
    },
    "text_length": 12
  },
  
  "tonal": {
    "prosodic_features": {
      "pitch_mean": 150.5,
      "pitch_std": 25.3,
      "speaking_rate": 145.2,
      "pause_ratio": 0.18
    },
    "voice_quality": {
      "energy": 0.65,
      "spectral_centroid": 2500.0,
      "zero_crossing_rate": 0.05
    },
    "acoustic_problems": ["monotone", "low_energy"],
    "spectral_features": {
      "mfcc": [12.5, -8.3, 4.1, 2.8],
      "spectral_rolloff": 3500.0
    }
  },
  
  "improvements": {
    "improved_text": "Hello everyone! Welcome to today's exciting presentation about artificial intelligence.",
    "original_text": "Hello everyone, welcome to today's presentation about artificial intelligence.",
    "issues": {
      "text_issues": ["low_engagement"],
      "delivery_issues": ["monotone", "low_energy"],
      "severity": "medium"
    },
    "feedback": {
      "summary": "Your speech is good but could benefit from some refinements.",
      "key_improvements": [
        "Added variety to combat monotone delivery",
        "Improved engagement and energy"
      ],
      "speaking_tips": [
        "Vary your pitch more to sound more engaging",
        "Increase your speaking energy",
        "Add natural pauses for emphasis"
      ],
      "severity": "medium",
      "issues_found": 3
    },
    "prosody_guide": {
      "rate_multiplier": 1.1,
      "pitch_multiplier": 1.2,
      "pitch_variation_multiplier": 1.5,
      "volume_multiplier": 1.1,
      "pause_locations": [2, 8],
      "emphasis_count": 2,
      "target_wpm": 160
    },
    "ssml_markup": "<speak><prosody rate='110%' pitch='120%'>Hello everyone!</prosody> <break time='0.5s'/> Welcome to today's <emphasis level='moderate'>exciting</emphasis> presentation about artificial intelligence.</speak>"
  },
  
  "synthesis": {
    "status": "completed",
    "output_path": "/home/app/outputs/generated_audio/improved_session_20240115_103045.mp3",
    "filename": "improved_session_20240115_103045.mp3",
    "audio_duration": 4.2
  },
  
  "metrics": {
    "processing_time_seconds": 12.5,
    "original_word_count": 12,
    "improved_word_count": 13,
    "issues_found": 3,
    "severity": "medium"
  }
}
```

**Error Responses:**
```json
// 400 - Bad Request
{
  "detail": "Unsupported audio format: audio/ogg"
}

// 413 - File Too Large
{
  "detail": "File too large. Maximum size is 50MB"
}

// 500 - Processing Error
{
  "detail": "Processing failed: Unable to transcribe audio"
}
```

#### `GET /download-audio/{filename}`
Download processed audio files.

**Parameters:**
- `filename`: Name of the audio file to download

**Example Request:**
```bash
curl -O "http://localhost:8000/download-audio/improved_session_20240115_103045.mp3"
```

**Response:**
- **Success**: Audio file download
- **404**: File not found

### Utility Endpoints

#### `POST /cleanup`
Clean up temporary files older than specified time.

**Parameters:**
- `max_age_hours` *(optional)*: Maximum age of files to keep
  - **Type**: integer
  - **Default**: 1
  - **Range**: 1-24

**Example Request:**
```bash
curl -X POST "http://localhost:8000/cleanup" \
     -H "Content-Type: application/json" \
     -d '{"max_age_hours": 2}'
```

**Response:**
```json
{
  "cleaned_files": 5,
  "status": "success"
}
```

#### `GET /config`
Get current configuration (sensitive keys hidden).

**Response:**
```json
{
  "speech_to_text": {
    "model": "whisper",
    "model_size": "base",
    "language": "en"
  },
  "text_sentiment_analysis": {
    "model": "j-hartmann/emotion-english-distilroberta-base",
    "device": "auto"
  },
  "llm_processing": {
    "openai": {
      "api_key": "***HIDDEN***",
      "model": "gpt-4",
      "temperature": 0.7
    }
  },
  "text_to_speech": {
    "provider": "elevenlabs",
    "api_key": "***HIDDEN***",
    "default_voice": "adam"
  }
}
```

## =Ê Data Models

### Audio Processing Pipeline

1. **Speech-to-Text**: OpenAI Whisper transcription
2. **Sentiment Analysis**: Emotion classification using transformers
3. **Tonal Analysis**: Acoustic feature extraction with librosa
4. **LLM Processing**: GPT-powered text improvement and prosody analysis
5. **Text-to-Speech**: ElevenLabs voice synthesis with cloning

### Supported Audio Formats

| Format | Extension | MIME Type | Max Size |
|--------|-----------|-----------|----------|
| WAV | `.wav` | `audio/wav`, `audio/x-wav` | 50MB |
| MP3 | `.mp3` | `audio/mpeg`, `audio/mp3` | 50MB |
| M4A | `.m4a` | `audio/m4a`, `audio/x-m4a` | 50MB |
| FLAC | `.flac` | `audio/flac` | 50MB |
| MP4 | `.mp4` | `audio/mp4` | 50MB |

### Target Styles

| Style | Description | Use Cases |
|-------|-------------|-----------|
| `professional` | Clear, confident business communication | Presentations, meetings, interviews |
| `casual` | Relaxed, conversational tone | Podcasts, informal talks |
| `academic` | Structured, educational delivery | Lectures, research presentations |
| `motivational` | Energetic, inspiring speech | Speeches, coaching, training |

### Improvement Focus Areas

| Focus | Description | Optimizes |
|-------|-------------|-----------|
| `all` | Comprehensive improvement | Text + delivery + engagement |
| `clarity` | Clear communication | Filler words, structure, pronunciation |
| `confidence` | Confident delivery | Pace, volume, hesitations |
| `engagement` | Audience engagement | Energy, emphasis, emotional connection |

## =' Integration Examples

### Python Client
```python
import requests
import json

# Process audio file
url = "http://localhost:8000/process-audio"
files = {
    'audio_file': open('speech.wav', 'rb')
}
data = {
    'target_style': 'professional',
    'improvement_focus': 'clarity'
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Improved text: {result['improvements']['improved_text']}")
print(f"Processing time: {result['metrics']['processing_time_seconds']}s")

# Download improved audio
if result['synthesis']['filename']:
    audio_url = f"http://localhost:8000/download-audio/{result['synthesis']['filename']}"
    audio_response = requests.get(audio_url)
    
    with open('improved_speech.mp3', 'wb') as f:
        f.write(audio_response.content)
```

### JavaScript/Node.js Client
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function processAudio(audioPath) {
    const form = new FormData();
    form.append('audio_file', fs.createReadStream(audioPath));
    form.append('target_style', 'professional');
    form.append('improvement_focus', 'all');
    
    try {
        const response = await axios.post(
            'http://localhost:8000/process-audio',
            form,
            {
                headers: {
                    ...form.getHeaders(),
                },
                timeout: 300000, // 5 minutes
            }
        );
        
        console.log('Processing completed:', response.data);
        return response.data;
    } catch (error) {
        console.error('Processing failed:', error.response?.data || error.message);
        throw error;
    }
}

// Usage
processAudio('./speech.wav')
    .then(result => {
        console.log('Improved text:', result.improvements.improved_text);
        console.log('Issues found:', result.improvements.issues);
    })
    .catch(console.error);
```

### cURL Examples
```bash
# Basic processing
curl -X POST "http://localhost:8000/process-audio" \
     -F "audio_file=@speech.wav" \
     -F "target_style=professional"

# With voice cloning
curl -X POST "http://localhost:8000/process-audio" \
     -F "audio_file=@speech.wav" \
     -F "voice_sample=@target_voice.wav" \
     -F "target_style=motivational" \
     -F "improvement_focus=engagement"

# Health check
curl "http://localhost:8000/health"

# Download result
curl -O "http://localhost:8000/download-audio/improved_session_20240115_103045.mp3"
```

## ¡ Performance & Limits

### Processing Times
- **Small files** (< 30s): 5-15 seconds
- **Medium files** (30s-2min): 15-45 seconds  
- **Large files** (2-5min): 45-120 seconds

### Rate Limits
- **Concurrent requests**: 10 per instance
- **File size**: 50MB maximum
- **Audio duration**: 5-300 seconds recommended
- **Timeout**: 15 minutes maximum

### Optimization Tips
1. **Use WAV format** for best transcription quality
2. **Record in quiet environment** for better analysis
3. **Keep files under 2 minutes** for faster processing
4. **Use voice samples 10-60 seconds** for optimal cloning

## = Security & Authentication

### Current Setup (Development)
- **Public access**: No authentication required
- **CORS**: Enabled for all origins

### Production Recommendations
1. **Enable authentication**: Remove `--allow-unauthenticated`
2. **API keys**: Implement API key validation
3. **Rate limiting**: Add request rate limits
4. **CORS**: Restrict to specific domains
5. **Input validation**: Enhanced file type and size checks

## = Error Handling

### Common Error Codes
- `400`: Bad Request (invalid parameters, unsupported format)
- `413`: File Too Large
- `422`: Validation Error
- `500`: Internal Server Error (processing failure)
- `503`: Service Unavailable (overloaded)

### Retry Strategy
- **Transient errors**: Retry with exponential backoff
- **File errors**: Check format and size
- **Processing errors**: Try with different parameters

---

For deployment instructions, see [deployment.md](deployment.md).