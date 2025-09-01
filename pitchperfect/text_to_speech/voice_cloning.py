# pitchperfect/text_to_speech/voice_cloning.py
from elevenlabs import clone
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class VoiceCloner:
    """Voice cloning functionality"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.cloned_voices = {}

    def clone(self, audio_path: str, name: str = None) -> str:
        """Clone voice from audio sample"""
        name = name or f"cloned_voice_{len(self.cloned_voices)}"

        try:
            logger.info(f"Cloning voice from {audio_path}")

            voice = clone(
                name=name,
                files=[audio_path],
                description="Cloned voice for speech improvement"
            )

            voice_id = voice.voice_id
            self.cloned_voices[name] = voice_id

            logger.info(f"Voice cloned successfully: {voice_id}")
            return voice_id

        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            raise

    def get_cloned_voice(self, name: str) -> Optional[str]:
        """Get cloned voice ID by name"""
        return self.cloned_voices.get(name)
