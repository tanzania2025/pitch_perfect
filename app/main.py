# app/main.py - FastAPI Backend
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import logging
import tempfile
import os
from datetime import datetime
from typing import Optional, Dict, Any
import aiofiles
from pydantic import BaseModel

from config import load_config
from pitchperfect.pipeline.orchestrator import PipelineOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ProcessRequest(BaseModel):
    target_style: str = "professional"
    improvement_focus: str = "all"
    save_audio: bool = True

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
    redoc_url="/redoc"
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

    async def save_uploaded_file(self, file: UploadFile, prefix: str = "uploaded") -> str:
        """Save uploaded file to temp directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix if file.filename else ".wav"
        temp_filename = f"{prefix}_{timestamp}{file_extension}"
        temp_path = self.temp_dir / temp_filename

        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        logger.info(f"Saved uploaded file: {temp_path}")
        return str(temp_path)

    def process_speech(self,
                      audio_path: str,
                      voice_sample_path: Optional[str] = None,
                      preferences: Optional[Dict] = None) -> Dict[str, Any]:
        """Process speech through the pipeline"""

        if not os.path.exists(audio_path):
            raise HTTPException(status_code=400, detail="Audio file not found")

        # Set default preferences
        default_preferences = {
            'target_style': 'professional',
            'improvement_focus': 'all'
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
                user_preferences=user_preferences
            )

            # Save audio if synthesis was successful
            if 'synthesis' in results and results['synthesis'].get('audio'):
                output_filename = f"improved_{session_id}.mp3"
                output_path = self.output_dir / output_filename

                audio_bytes = results['synthesis']['audio']
                with open(output_path, 'wb') as f:
                    f.write(audio_bytes)

                results['synthesis']['output_path'] = str(output_path)
                results['synthesis']['filename'] = output_filename
                logger.info(f"Audio saved: {output_path}")
                
                # Remove binary audio data from response to avoid JSON serialization issues
                del results['synthesis']['audio']

            # Add session info
            results['session_id'] = session_id
            results['processing_status'] = 'completed'

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

# Initialize service
speech_service = SpeechImprovementService()

@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now()
    )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now()
    )

@app.post("/process-audio")
async def process_audio(
    audio_file: UploadFile = File(..., description="Audio file to process"),
    voice_sample: Optional[UploadFile] = File(None, description="Optional voice sample for cloning"),
    target_style: str = "professional",
    improvement_focus: str = "all",
    save_audio: bool = True
):
    """
    Process uploaded audio file through the speech improvement pipeline

    - **audio_file**: Audio file (WAV, MP3, M4A, FLAC)
    - **voice_sample**: Optional voice sample for cloning
    - **target_style**: professional, casual, academic, motivational
    - **improvement_focus**: all, clarity, confidence, engagement
    - **save_audio**: Whether to save the improved audio file
    """

    # Validate file types
    allowed_types = [
        "audio/wav", "audio/mpeg", "audio/mp3",
        "audio/mp4", "audio/m4a", "audio/flac",
        "audio/x-wav", "audio/x-m4a"
    ]

    if audio_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {audio_file.content_type}"
        )

    try:
        # Save uploaded files
        audio_path = await speech_service.save_uploaded_file(audio_file, "audio")

        voice_sample_path = None
        if voice_sample and voice_sample.filename:  # Check if voice_sample exists and has content
            if voice_sample.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported voice sample format: {voice_sample.content_type}"
                )
            voice_sample_path = await speech_service.save_uploaded_file(voice_sample, "voice_sample")

        # Process audio
        preferences = {
            'target_style': target_style,
            'improvement_focus': improvement_focus
        }

        results = speech_service.process_speech(
            audio_path=audio_path,
            voice_sample_path=voice_sample_path,
            preferences=preferences
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
    file_path = speech_service.output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=filename
    )

@app.post("/cleanup")
async def cleanup_temp_files(max_age_hours: int = 1):
    """Clean up temporary files older than specified hours"""
    result = speech_service.cleanup_temp_files(max_age_hours)
    return result

@app.get("/config")
async def get_config():
    """Get current configuration (excluding sensitive keys)"""
    config = speech_service.config.copy()

    # Remove sensitive information
    sensitive_keys = ['api_key', 'openai', 'elevenlabs']
    for key in sensitive_keys:
        if key in config.get('llm_processing', {}):
            config['llm_processing'][key] = "***HIDDEN***"
        if key in config.get('text_to_speech', {}):
            config['text_to_speech'][key] = "***HIDDEN***"

    return config

if __name__ == "__main__":
    # Create necessary directories
    Path("outputs/generated_audio").mkdir(parents=True, exist_ok=True)
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", 8000))
    # Run the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
