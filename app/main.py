# app/main.py - FastAPI Backend
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional

import base64
import aiofiles
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import load_config
from pitchperfect.pipeline.orchestrator import PipelineOrchestrator
from pitchperfect.text_to_speech.elevenlabs_client import ElevenLabsClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class ProcessRequest(BaseModel):
    target_style: str = "professional"
    improvement_focus: str = "all"
    save_audio: bool = True
    voice_id: Optional[str] = None


class VoiceOption(BaseModel):
    voice_id: str
    name: str
    category: str
    description: str


class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str = "0.1.0"


# Initialize FastAPI app
app = FastAPI(
    title="Pitch Perfect API",
    description="AI-powered speech analysis and improvement backend",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SpeechImprovementService:
    """Backend service for speech improvement processing"""

    def __init__(self):
        self.config = load_config()
        self.pipeline = PipelineOrchestrator(self.config)

        # Create output directories
        self.output_dir = Path("outputs/generated_audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.temp_dir = Path(tempfile.gettempdir()) / "pitch_perfect"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Speech Improvement Service initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Temp directory: {self.temp_dir}")

    async def save_uploaded_file(
        self, file: UploadFile, prefix: str = "uploaded"
    ) -> str:
        """Save uploaded file to temp directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix if file.filename else ".wav"
        temp_filename = f"{prefix}_{timestamp}{file_extension}"
        temp_path = self.temp_dir / temp_filename

        async with aiofiles.open(temp_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        logger.info(f"Saved uploaded file: {temp_path}")
        return str(temp_path)

    def process_speech(
        self,
        audio_path: str,
        voice_sample_path: Optional[str] = None,
        preferences: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Process speech through the pipeline"""

        if not os.path.exists(audio_path):
            raise HTTPException(status_code=400, detail="Audio file not found")

        # Set default preferences
        default_preferences = {
            "target_style": "professional",
            "improvement_focus": "all",
        }
        user_preferences = {**default_preferences, **(preferences or {})}

        try:
            # Generate unique session ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"session_{timestamp}"

            # Process through pipeline
            logger.info(f"Processing audio: {audio_path}")
            results = self.pipeline.process(
                audio_path=audio_path,
                voice_sample_path=voice_sample_path,
                output_path=None,  # We'll handle this separately
                user_preferences=user_preferences,
            )

            # Save audio if synthesis was successful
            if "synthesis" in results and results["synthesis"].get("audio"):
                output_filename = f"improved_{session_id}.mp3"
                output_path = self.output_dir / output_filename

                audio_bytes = results["synthesis"]["audio"]
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)

                results["synthesis"]["output_path"] = str(output_path)
                results["synthesis"]["filename"] = output_filename
                logger.info(f"Audio saved: {output_path}")

                # Convert binary audio to base64 for JSON serialization
                audio_bytes = results["synthesis"]["audio"]
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                results["synthesis"]["audio_data"] = audio_base64
                results["synthesis"]["audio_format"] = "mp3"  # Specify format for frontend
                del results["synthesis"]["audio"]  # Remove original bytes

            # Add session info
            results["session_id"] = session_id
            results["processing_status"] = "completed"

            return results

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    def cleanup_temp_files(self, max_age_hours: int = 1):
        """Clean up temporary files older than specified hours"""
        try:
            import time

            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            cleaned_count = 0
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                            logger.info(f"Cleaned up temp file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to clean up {file_path}: {e}")

            logger.info(f"Cleanup completed: {cleaned_count} files removed")
            return {"cleaned_files": cleaned_count, "status": "success"}

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {"error": str(e), "status": "failed"}


# Initialize service with lazy loading for faster startup
speech_service = None

def get_speech_service():
    global speech_service
    if speech_service is None:
        speech_service = SpeechImprovementService()
    return speech_service


@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint"""
    return HealthCheck(status="healthy", timestamp=datetime.now())


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check"""
    return HealthCheck(status="healthy", timestamp=datetime.now())


@app.post("/process-audio")
async def process_audio(
    audio_file: UploadFile = File(..., description="Audio file to process"),
    voice_sample: Optional[UploadFile] = File(
        None, description="Optional voice sample for cloning"
    ),
    target_style: str = Form("professional"),
    improvement_focus: str = Form("all"),
    save_audio: bool = Form(True),
    voice_id: Optional[str] = Form(None),
):
    """
    Process uploaded audio file through the speech improvement pipeline

    - **audio_file**: Audio file (WAV, MP3, M4A, FLAC)
    - **voice_sample**: Optional voice sample for cloning
    - **target_style**: professional, casual, academic, motivational
    - **improvement_focus**: all, clarity, confidence, engagement
    - **save_audio**: Whether to save the improved audio file
    - **voice_id**: Optional ElevenLabs voice ID for TTS (use /voices endpoint to get options)
    """

    logger.info(f"[ENDPOINT] Received parameters:")
    logger.info(f"  - voice_id: {voice_id}")
    logger.info(f"  - target_style: {target_style}")
    logger.info(f"  - improvement_focus: {improvement_focus}")

    # Validate file types
    allowed_types = [
        "audio/wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/m4a",
        "audio/flac",
        "audio/x-wav",
        "audio/x-m4a",
    ]

    if audio_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {audio_file.content_type}",
        )

    try:
        # Save uploaded files
        audio_path = await get_speech_service().save_uploaded_file(audio_file, "audio")

        voice_sample_path = None
        if (
            voice_sample and voice_sample.filename
        ):  # Check if voice_sample exists and has content
            if voice_sample.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported voice sample format: {voice_sample.content_type}",
                )
            voice_sample_path = await get_speech_service().save_uploaded_file(
                voice_sample, "voice_sample"
            )

        # Process audio
        preferences = {
            "target_style": target_style,
            "improvement_focus": improvement_focus,
            "voice_id": voice_id,
        }

        logger.info(f"[MAIN.PY] Received voice_id: {voice_id}")
        logger.info(f"[MAIN.PY] Preferences: {preferences}")

        results = get_speech_service().process_speech(
            audio_path=audio_path,
            voice_sample_path=voice_sample_path,
            preferences=preferences,
        )

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/download-audio/{filename}")
async def download_audio(filename: str):
    """Download generated audio file"""
    file_path = get_speech_service().output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(path=file_path, media_type="audio/mpeg", filename=filename)


@app.post("/cleanup")
async def cleanup_temp_files(max_age_hours: int = 1):
    """Clean up temporary files older than specified hours"""
    result = get_speech_service().cleanup_temp_files(max_age_hours)
    return result


@app.get("/config")
async def get_config():
    """Get current configuration (excluding sensitive keys)"""
    config = get_speech_service().config.copy()

    # Remove sensitive information
    sensitive_keys = ["api_key", "openai", "elevenlabs"]
    for key in sensitive_keys:
        if key in config.get("llm_processing", {}):
            config["llm_processing"][key] = "***HIDDEN***"
        if key in config.get("text_to_speech", {}):
            config["text_to_speech"][key] = "***HIDDEN***"

    return config


@app.get("/voices")
async def get_available_voices():
    """Get available ElevenLabs voices for selection"""
    try:
        from elevenlabs import set_api_key, voices

        # Get API key from config
        api_key = get_speech_service().config.get("text_to_speech", {}).get("api_key")
        if not api_key:
            raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")

        set_api_key(api_key)

        # Get available voices
        available_voices = voices()

        voice_options = []
        for voice in available_voices:
            # Ensure description is never None
            description = getattr(voice, 'description', None)
            if description is None or description == '':
                description = 'No description available.'

            voice_options.append(VoiceOption(
                voice_id=voice.voice_id,
                name=voice.name,
                category=getattr(voice, 'category', 'Unknown'),
                description=description
            ))

        return {"voices": voice_options}

    except Exception as e:
        logger.error(f"Failed to fetch voices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch voices: {str(e)}")


if __name__ == "__main__":
    # Create necessary directories
    Path("outputs/generated_audio").mkdir(parents=True, exist_ok=True)
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", 8080))

    # Determine if we're running in production (Cloud Run)
    is_production = os.environ.get("K_SERVICE") is not None

    # Run the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=not is_production,  # Disable reload in production
        log_level="info",
        timeout_keep_alive=300  # Keep connections alive longer
    )
