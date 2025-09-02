from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Literal, Tuple, Union

from pitchperfect.config.settings import MELD_INTERIM_DIR, MELD_WAV_DIR
from pitchperfect.utils.audio_processing import extract_wav

Split = Literal["train", "dev", "test"]
Mode = Literal["batch", "lazy"]

# Use config paths for consistent directory resolution
DEFAULT_MELD_INTERIM_ROOT = MELD_INTERIM_DIR
DEFAULT_MELD_WAV_OUT = MELD_WAV_DIR


def find_meld_split(
    root: str | Path = DEFAULT_MELD_INTERIM_ROOT, split: Split = "train"
) -> List[Path]:
    """Return a list of .mp4 files for the given MELD split.

    Defaults to the organized interim layout: data/interim/MELD/{split}/.../*.mp4
    """
    base = Path(root)
    split_dir = base / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(split_dir.rglob("*.mp4"))


def process_meld_split(
    split: Split = "train",
    mode: Mode = "lazy",
    root: str | Path = DEFAULT_MELD_INTERIM_ROOT,
    out_dir: str | Path = DEFAULT_MELD_WAV_OUT,
    sample_rate: int = 16000,
) -> Union[List[Tuple[Path, Path]], Iterator[Path]]:
    """Process MELD split with configurable mode.

    Args:
        split: Which split to process ('train', 'dev', 'test')
        mode: 'batch' returns list of (mp4_path, wav_path) tuples, 'lazy' returns iterator of wav_paths
        root: Source directory for MP4 files
        out_dir: Output directory for WAV files
        sample_rate: Audio sample rate for WAV conversion

    Returns:
        If mode='batch': List of tuples (mp4_path, wav_path)
        If mode='lazy': Iterator yielding wav_paths one at a time
    """
    mp4_files = find_meld_split(root, split)
    out = Path(out_dir)
    skipped = 0

    if mode == "batch":
        # Batch mode: convert all files and return list
        results: List[Tuple[Path, Path]] = []

        for mp4 in mp4_files:
            try:
                rel = mp4.relative_to(Path(root))
                wav_rel = rel.with_suffix(".wav")
                wav_path = out / wav_rel
                extract_wav(mp4, wav_path, sample_rate=sample_rate, mono=True)
                results.append((mp4, wav_path))
            except Exception as e:
                print(f"⚠️  Skipping corrupted file {mp4.name}: {e}")
                skipped += 1
                continue

        if skipped > 0:
            print(f"⚠️  Skipped {skipped} corrupted files")

        return results

    else:
        # Lazy mode: yield wav paths one at a time
        def lazy_generator():
            nonlocal skipped
            for mp4 in mp4_files:
                try:
                    rel = mp4.relative_to(Path(root))
                    wav_rel = rel.with_suffix(".wav")
                    wav_path = out / wav_rel
                    if not wav_path.exists():
                        extract_wav(mp4, wav_path, sample_rate=sample_rate, mono=True)
                    yield wav_path
                except Exception as e:
                    print(f"⚠️  Skipping corrupted file {mp4.name}: {e}")
                    skipped += 1
                    continue

            if skipped > 0:
                print(f"⚠️  Skipped {skipped} corrupted files")

        return lazy_generator()


# Backward compatibility aliases
def prepare_meld_split(
    root: str | Path = DEFAULT_MELD_INTERIM_ROOT,
    split: Split = "train",
    out_dir: str | Path = DEFAULT_MELD_WAV_OUT,
    sample_rate: int = 16000,
) -> List[Tuple[Path, Path]]:
    """Backward compatibility alias for process_meld_split(mode='batch')"""
    return process_meld_split(
        split=split, mode="batch", root=root, out_dir=out_dir, sample_rate=sample_rate
    )


def iter_wavs(
    root: str | Path = DEFAULT_MELD_INTERIM_ROOT,
    split: Split = "train",
    out_dir: str | Path = DEFAULT_MELD_WAV_OUT,
    sample_rate: int = 16000,
) -> Iterator[Path]:
    """Backward compatibility alias for process_meld_split(mode='lazy')"""
    return process_meld_split(
        split=split, mode="lazy", root=root, out_dir=out_dir, sample_rate=sample_rate
    )
