# pitchperfect/text_to_speech/elevenlabs_client.py
from elevenlabs import generate, set_api_key, Voice
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class ElevenLabsClient:
    """Client for ElevenLabs TTS API"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        tts_config = self.config.get('text_to_speech', {})

        api_key = tts_config.get('api_key')
        if not api_key:
            raise ValueError("ElevenLabs API key required")

        set_api_key(api_key)
        self.default_voice = tts_config.get('default_voice', 'onwK4e9ZLuTAKqWW03F9')
        logger.info("ElevenLabs client initialized")

    def generate(self, text: str, voice_id: str = None, model: str = None) -> bytes:
        """Generate speech from text"""
        voice = voice_id or self.default_voice
        model = model or "eleven_monolingual_v1"

        try:
            audio = generate(
                text=text,
                voice=voice,
                model=model
            )
            return audio
        except Exception as e:
            logger.error(f"ElevenLabs generation failed with voice '{voice}': {e}")
            # Try with a known fallback voice if the original fails
            if voice != "onwK4e9ZLuTAKqWW03F9":  # Daniel voice ID
                logger.warning(f"Retrying with fallback voice (Daniel)")
                try:
                    audio = generate(
                        text=text,
                        voice="onwK4e9ZLuTAKqWW03F9",
                        model=model
                    )
                    return audio
                except Exception as fallback_e:
                    logger.error(f"Fallback voice also failed: {fallback_e}")
            raise

    def generate_with_settings(self, text: str, voice_id: str,
                               stability: float = 0.5,
                               similarity_boost: float = 0.75) -> bytes:
        """Generate with custom voice settings"""
        try:
            audio = generate(
                text=text,
                voice=Voice(
                    voice_id=voice_id,
                    settings={
                        'stability': stability,
                        'similarity_boost': similarity_boost
                    }
                )
            )
            return audio
        except Exception as e:
            logger.error(f"ElevenLabs generation failed: {e}")
            raise
