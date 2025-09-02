# pitchperfect/text_to_speech/synthesis.py
import re
from typing import Optional, Dict
from .elevenlabs_client import ElevenLabsClient
from .voice_cloning import VoiceCloner
from pitchperfect.utils.audio_processing import AudioProcessor
import logging


logger = logging.getLogger(__name__)

class Synthesizer:
    """Main text-to-speech synthesizer with SSML support"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.client = ElevenLabsClient(config)
        self.cloner = VoiceCloner(config)
        self.audio_processor = AudioProcessor(config)
        self.default_voice = self.config.get('text_to_speech', {}).get('default_voice', 'onwK4e9ZLuTAKqWW03F9')

    def synthesize(self, text: str = None, ssml: str = None,
                  voice_id: str = None, output_path: str = None) -> Dict:
        """
        Synthesize speech from text or SSML

        Returns:
            Dict with audio data and metadata
        """
        if not text and not ssml:
            raise ValueError("Either text or SSML required")

        # Use provided voice or default
        voice = voice_id or self.default_voice

        # Process SSML if provided
        if ssml:
            text = self._parse_ssml(ssml)
            # Extract prosody settings from SSML
            prosody = self._extract_prosody(ssml)
        else:
            prosody = {}

        # Generate audio
        try:
            logger.info(f"Synthesizing with voice: {voice}")
            audio_bytes = self.client.generate(text, voice)

            # Save if path provided
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(audio_bytes)
                logger.info(f"Audio saved to {output_path}")

            return {
                'audio': audio_bytes,
                'voice_id': voice,
                'text': text,
                'prosody_applied': prosody,
                'output_path': output_path
            }

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise

    def synthesize_with_clone(self, text: str, clone_audio_path: str,
                            ssml: str = None) -> Dict:
        """Synthesize using cloned voice"""
        # Clone voice
        voice_id = self.cloner.clone(clone_audio_path)

        # Synthesize
        return self.synthesize(text=text, ssml=ssml, voice_id=voice_id)

    def _parse_ssml(self, ssml: str) -> str:
        """Extract text from SSML"""
        # Remove all tags for basic extraction
        text = re.sub(r'<[^>]+>', '', ssml)
        return text.strip()

    def _extract_prosody(self, ssml: str) -> Dict:
        """Extract prosody settings from SSML"""
        prosody = {}

        # Extract rate
        rate_match = re.search(r'rate="(\d+)%"', ssml)
        if rate_match:
            prosody['rate'] = int(rate_match.group(1)) / 100

        # Extract pitch
        pitch_match = re.search(r'pitch="([+-]?\d+)%"', ssml)
        if pitch_match:
            prosody['pitch'] = int(pitch_match.group(1)) / 100

        # Extract volume
        volume_match = re.search(r'volume="([+-]?\d+\.?\d*)dB"', ssml)
        if volume_match:
            prosody['volume'] = float(volume_match.group(1))

        return prosody
