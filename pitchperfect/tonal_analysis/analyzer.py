# pitchperfect/tonal_analysis/analyzer.py
import numpy as np
import librosa
from typing import Dict, List, Union, Optional
from .feature_extraction import ProsodyExtractor
from pitchperfect.utils.audio_processing import AudioProcessor
import logging

logger = logging.getLogger(__name__)

class TonalAnalyzer:
    """Analyze tonal qualities of speech"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        tonal_config = self.config.get('tonal_analysis', {})
        self.sample_rate = tonal_config.get('sample_rate', 16000)

        self.prosody_extractor = ProsodyExtractor(self.sample_rate)
        self.audio_processor = AudioProcessor({'sample_rate': self.sample_rate})

        # Load thresholds
        self.thresholds = self.config.get('llm_processing', {}).get('thresholds', {})

        logger.info("Tonal analyzer initialized")

    def analyze(self, audio_input: Union[str, np.ndarray]) -> Dict:
        """
        Analyze tonal features from audio

        Returns schema for llm_processing:
        {
            'prosodic_features': dict,
            'voice_quality': dict,
            'acoustic_problems': list,
            'spectral_features': dict
        }
        """
        # Load audio
        if isinstance(audio_input, str):
            audio, sr = self.audio_processor.load_audio(audio_input)
        else:
            audio = audio_input
            sr = self.sample_rate

        # Extract all features
        prosodic = self.prosody_extractor.extract(audio, sr)
        voice_quality = self._assess_voice_quality(prosodic)
        problems = self._identify_problems(prosodic, voice_quality)
        spectral = self._extract_spectral_features(audio, sr)

        return {
            'prosodic_features': prosodic,
            'voice_quality': voice_quality,
            'acoustic_problems': problems,
            'spectral_features': spectral,
            'audio_quality': self._assess_audio_quality(audio, sr)
        }

    def _assess_voice_quality(self, prosodic: Dict) -> Dict:
        """Assess voice quality from prosodic features"""
        pitch_std = prosodic.get('pitch', {}).get('std_hz', 25)

        # Monotone score (0=varied, 1=monotone)
        monotone_score = 1.0 - min(pitch_std / 50, 1.0)

        # Clarity based on voiced ratio
        voiced_ratio = prosodic.get('pitch', {}).get('voiced_ratio', 0.5)
        clarity_score = min(voiced_ratio * 1.2, 1.0)

        # Confidence based on energy consistency
        energy_std = abs(prosodic.get('energy', {}).get('std_db', 5))
        confidence_score = 1.0 - min(energy_std / 10, 1.0)

        # Stress based on pitch range
        pitch_range = prosodic.get('pitch', {}).get('range_hz', 100)
        stress_level = min(pitch_range / 200, 1.0) * 0.5

        return {
            'monotone_score': float(monotone_score),
            'clarity_score': float(clarity_score),
            'confidence_score': float(confidence_score),
            'stress_level': float(stress_level),
            'overall_quality': float((clarity_score + confidence_score + (1 - monotone_score)) / 3)
        }

    def _identify_problems(self, prosodic: Dict, quality: Dict) -> List[str]:
        """Identify acoustic problems"""
        problems = []

        # Check speaking rate
        wpm = prosodic.get('tempo', {}).get('speaking_rate_wpm', 160)
        if wpm > self.thresholds.get('fast_speech_wpm', 200):
            problems.append('too_fast')
        elif wpm < self.thresholds.get('slow_speech_wpm', 120):
            problems.append('too_slow')

        # Check monotone
        if quality['monotone_score'] > self.thresholds.get('monotone_threshold', 0.6):
            problems.append('monotone')

        # Check volume
        energy_db = prosodic.get('energy', {}).get('mean_db', -15)
        if energy_db < self.thresholds.get('low_energy_db', -30):
            problems.append('too_quiet')
        elif energy_db > self.thresholds.get('high_energy_db', -5):
            problems.append('too_loud')

        # Check clarity
        if quality['clarity_score'] < 0.5:
            problems.append('unclear_articulation')

        # Check confidence
        if quality['confidence_score'] < 0.4:
            problems.append('low_confidence')

        # Check pauses
        pause_ratio = prosodic.get('pauses', {}).get('pause_ratio', 0.2)
        if pause_ratio < 0.1:
            problems.append('no_pauses')
        elif pause_ratio > 0.4:
            problems.append('excessive_pauses')

        return problems

    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract spectral features"""
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]

        return {
            'mfcc_stats': {
                'mean': np.mean(mfccs, axis=1).tolist(),
                'std': np.std(mfccs, axis=1).tolist()
            },
            'spectral': {
                'centroid_mean': float(np.mean(spectral_centroid)),
                'rolloff_mean': float(np.mean(spectral_rolloff))
            }
        }

    def _assess_audio_quality(self, audio: np.ndarray, sr: int) -> Dict:
        """Assess technical audio quality"""
        max_val = np.max(np.abs(audio))
        is_clipping = max_val > 0.99

        # Simple SNR estimation
        signal_power = np.mean(audio ** 2)
        noise_floor = np.percentile(np.abs(audio), 5) ** 2
        snr = 10 * np.log10(signal_power / (noise_floor + 1e-10))

        return {
            'is_clipping': bool(is_clipping),
            'snr_db': float(snr),
            'max_amplitude': float(max_val)
        }
