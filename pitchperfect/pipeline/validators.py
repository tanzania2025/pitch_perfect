from __future__ import annotations

from pathlib import Path


def ensure_audio_exists(audio_path: str | Path) -> Path:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Audio path must be a file: {path}")
    return path
