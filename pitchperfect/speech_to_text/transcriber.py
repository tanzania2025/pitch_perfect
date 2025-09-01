# pitchperfect/speech_to_text/transcriber.py
import whisper
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class Transcriber:
    """Speech to text transcription using Whisper"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        model_size = self.config.get('speech_to_text', {}).get('model_size', 'base')
        self.model = whisper.load_model(model_size)
        self.language = self.config.get('speech_to_text', {}).get('language', 'en')
        logger.info(f"Transcriber initialized with Whisper {model_size}")

    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribe audio to text

        Returns:
            Dict with text, segments, and confidence
        """
        try:
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                word_timestamps=True
            )

            return {
                'text': result['text'].strip(),
                'segments': result.get('segments', []),
                'language': result.get('language', self.language),
                'confidence': self._calculate_confidence(result)
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate average confidence from segments"""
        segments = result.get('segments', [])
        if not segments:
            return 0.0

        confidences = [seg.get('confidence', 0.5) for seg in segments
                      if 'confidence' in seg]

        return sum(confidences) / len(confidences) if confidences else 0.5
