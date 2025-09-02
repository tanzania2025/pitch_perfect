from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
# pitchperfect/utils/audio_processing.py
import numpy as np
import soundfile as sf


class AudioProcessor:
    """Audio processing utilities used across modules"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.sample_rate = self.config.get("sample_rate", 16000)

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio array and sample rate"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr

    def save_audio(self, audio: np.ndarray, path: str, sr: int = None):
        """Save audio array to file"""
        if sr is None:
            sr = self.sample_rate
        sf.write(path, audio, sr)

    def extract_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract basic audio features"""
        features = {
            "duration": len(audio) / sr,
            "rms_energy": float(np.sqrt(np.mean(audio**2))),
            "zero_crossing_rate": float(
                np.mean(librosa.feature.zero_crossing_rate(audio)[0])
            ),
        }

        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 50 and pitch < 500:
                pitch_values.append(pitch)

        if pitch_values:
            features["pitch_mean"] = float(np.mean(pitch_values))
            features["pitch_std"] = float(np.std(pitch_values))
        else:
            features["pitch_mean"] = 0
            features["pitch_std"] = 0

        return features

    def detect_pauses(
        self, audio: np.ndarray, sr: int, min_silence_ms: int = 100
    ) -> Dict:
        """Detect pauses in audio"""
        frame_length = int(sr * min_silence_ms / 1000)
        hop_length = frame_length // 2

        energy = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]
        threshold = np.mean(energy) * 0.1

        is_silence = energy < threshold

        pause_frames = np.sum(is_silence)
        total_frames = len(is_silence)
        pause_ratio = pause_frames / total_frames if total_frames > 0 else 0

        return {
            "pause_ratio": float(pause_ratio),
            "pause_count": int(np.sum(np.diff(is_silence.astype(int)) > 0)),
        }

    def normalize_audio_level(self, db_value: float, target_db: float = -15) -> float:
        """Calculate normalization factor for audio level"""
        if db_value >= target_db:
            return 1.0

        db_difference = target_db - db_value
        return 10 ** (db_difference / 20)


def _which_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def extract_wav(
    input_video_path: str | Path,
    output_wav_path: str | Path,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    """Extract audio as WAV from a video file using ffmpeg.

    - Forces PCM 16-bit, target sample rate, and mono if requested
    - Creates parent directory for the output if needed
    """
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required but not found in PATH")

    src = Path(input_video_path)
    dst = Path(output_wav_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    # Create parent directories before running ffmpeg
    dst.parent.mkdir(parents=True, exist_ok=True)

    channels = 1 if mono else 2
    cmd = [
        ffmpeg,
        "-y",  # overwrite
        "-i",
        str(src),
        "-vn",  # no video
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(dst),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return dst
