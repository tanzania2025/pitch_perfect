import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import dotenv

try:
    import openai
except Exception:  # pragma: no cover
    openai = None  # type: ignore


def _ensure_env_loaded() -> None:
    dotenv.load_dotenv(override=False)


def _hash_file(file_path: Path, extra: str = "") -> str:
    sha = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    if extra:
        sha.update(extra.encode("utf-8"))
    return sha.hexdigest()


class AudioTranscriber:
    """Transcribe audio using OpenAI Whisper with optional disk caching."""

    def __init__(
        self,
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        output_dir: Optional[Path] = None,
        use_cache: bool = True,
    ) -> None:
        _ensure_env_loaded()
        self.model = model
        self.language = language
        self.prompt = prompt
        self.use_cache = use_cache

        default_output = Path(os.getenv("OUTPUT_DIR", "outputs")) / "transcripts"
        self.output_dir = Path(output_dir or default_output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        api_key = os.getenv("OPENAI_API_KEY")
        if openai is not None:
            openai.api_key = api_key

    def _cache_key(self, audio_path: Path) -> str:
        meta = {
            "model": self.model,
            "language": self.language,
            "prompt": self.prompt or "",
            "filename": audio_path.name,
        }
        return _hash_file(audio_path, extra=json.dumps(meta, sort_keys=True))

    def _cache_file(self, key: str) -> Path:
        return self.output_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[dict]:
        cache_path = self._cache_file(key)
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _write_cache(self, key: str, payload: dict) -> None:
        cache_path = self._cache_file(key)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def transcribe(self, audio_path: str | Path) -> dict:
        if openai is None:
            raise RuntimeError("openai package is not installed. Please install from requirements.txt")

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        key = self._cache_key(path)
        if self.use_cache:
            cached = self._read_cache(key)
            if cached is not None:
                return cached

        with path.open("rb") as f:
            response = openai.audio.transcriptions.create(
                model=self.model,
                file=f,
                language=self.language,
                prompt=self.prompt,
                response_format="verbose_json",
                temperature=0,
            )

        result = json.loads(response.model_dump_json()) if hasattr(response, "model_dump_json") else json.loads(str(response))

        payload = {
            "text": result.get("text", ""),
            "segments": result.get("segments"),
            "language": result.get("language", self.language),
            "model": self.model,
            "source": str(path),
        }

        if self.use_cache:
            self._write_cache(key, payload)

        return payload

    def transcribe_batch(self, audio_paths: Iterable[str | Path]) -> List[Tuple[str, dict]]:
        results: List[Tuple[str, dict]] = []
        for ap in audio_paths:
            ap_path = Path(ap)
            try:
                result = self.transcribe(ap_path)
            except Exception as exc:
                result = {"error": str(exc), "text": ""}
            results.append((str(ap_path), result))
        return results
