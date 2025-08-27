from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pitchperfect.speech_to_text import AudioTranscriber
from pitchperfect.pipeline.validators import ensure_audio_exists


@dataclass
class PipelineResult:
    transcript: str
    sentiment: Optional[str] = None
    improved_text: Optional[str] = None
    output_audio_path: Optional[str] = None


class PipelineOrchestrator:
    """High-level orchestrator that wires together core components.

    This minimal stub currently performs only transcription. Extend to add
    sentiment analysis, LLM improvements, tonal analysis, and TTS.
    """

    def __init__(
        self,
        transcriber: Optional[AudioTranscriber] = None,
    ) -> None:
        self.transcriber = transcriber or AudioTranscriber()

    def process_audio(
        self,
        audio_path: str | Path,
        target_voice: Optional[str] = None,
        improve_content: bool = False,
    ) -> PipelineResult:
        path = ensure_audio_exists(audio_path)
        transcription = self.transcriber.transcribe(path)
        return PipelineResult(
            transcript=transcription.get("text", ""),
        )


class MainPipeline(PipelineOrchestrator):
    """Compatibility alias used by Makefile tests."""

    pass
