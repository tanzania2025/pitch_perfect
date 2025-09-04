# pitchperfect/speech_to_text/transcriber.py
import logging
from typing import Dict, Optional

import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


logger = logging.getLogger(__name__)


class Transcriber:
    """Speech to text transcription using Whisper"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        model_size = self.config.get("speech_to_text", {}).get("model_size", "base")
        model_name = f"openai/whisper-{model_size}"
        
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.language = self.config.get("speech_to_text", {}).get("language", "en")
        
        logger.info(f"Transcriber initialized with {model_name}")

    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribe audio to text

        Returns:
            Dict with text, segments, and confidence
        """
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process audio through Whisper
            input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features, return_timestamps=True)
            
            # Decode the transcription
            result = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            text = result[0] if result else ""

            return {
                "text": text.strip(),
                "segments": [],  # Transformers Whisper doesn't provide segments by default
                "language": self.language,
                "confidence": 0.8,  # Default confidence since segments aren't available
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate average confidence from segments"""
        segments = result.get("segments", [])
        if not segments:
            return 0.0

        confidences = [
            seg.get("confidence", 0.5) for seg in segments if "confidence" in seg
        ]

        return sum(confidences) / len(confidences) if confidences else 0.5
