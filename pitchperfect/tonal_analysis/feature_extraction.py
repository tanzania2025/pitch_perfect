# pitchperfect/tonal_analysis/feature_extraction.py
import numpy as np
import librosa
from typing import Dict, Optional


class ProsodyExtractor:
    """Extract prosodic features from audio"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def extract(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract comprehensive prosodic features"""

        # Pitch extraction using YIN
        f0 = librosa.yin(audio, fmin=50, fmax=500)
        f0_valid = f0[f0 > 0]

        # Energy/RMS
        rms = librosa.feature.rms(y=audio)[0]

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]

        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)

        # Onset detection for syllable estimation
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        peaks = librosa.util.peak_pick(
            onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10
        )

        # Calculate speaking rate
        duration = len(audio) / sr
        syllables = len(peaks)
        syllables_per_second = syllables / duration if duration > 0 else 0
        words_per_minute = (syllables / 1.5) / duration * 60 if duration > 0 else 0

        # Pause detection
        pause_info = self._detect_pauses(audio, sr)

        return {
            "pitch": {
                "mean_hz": float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0,
                "std_hz": float(np.std(f0_valid)) if len(f0_valid) > 0 else 0,
                "median_hz": float(np.median(f0_valid)) if len(f0_valid) > 0 else 0,
                "range_hz": float(np.ptp(f0_valid)) if len(f0_valid) > 0 else 0,
                "voiced_ratio": len(f0_valid) / len(f0) if len(f0) > 0 else 0,
            },
            "energy": {
                "mean_db": float(20 * np.log10(np.mean(rms) + 1e-10)),
                "std_db": float(20 * np.log10(np.std(rms) + 1e-10)),
                "rms": float(np.mean(rms)),
                "dynamic_range_db": float(
                    20 * np.log10(np.max(rms) / (np.min(rms) + 1e-10))
                ),
            },
            "tempo": {
                "speaking_rate_wpm": float(words_per_minute),
                "syllables_per_second": float(syllables_per_second),
                "tempo_bpm": float(tempo),
            },
            "pauses": pause_info,
        }

    def _detect_pauses(self, audio: np.ndarray, sr: int) -> Dict:
        """Detect pauses in audio"""
        frame_length = int(sr * 0.1)  # 100ms frames
        hop_length = frame_length // 2

        energy = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]
        threshold = np.mean(energy) * 0.1

        is_silence = energy < threshold

        pause_frames = np.sum(is_silence)
        total_frames = len(is_silence)

        # Count pause segments
        pause_starts = np.where(np.diff(is_silence.astype(int)) == 1)[0]
        pause_count = len(pause_starts)

        return {
            "pause_ratio": (
                float(pause_frames / total_frames) if total_frames > 0 else 0
            ),
            "pause_count": pause_count,
            "average_pause_duration": float(
                pause_frames * hop_length / sr / max(1, pause_count)
            ),
        }
