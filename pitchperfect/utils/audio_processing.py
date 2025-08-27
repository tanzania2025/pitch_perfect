from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional


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
