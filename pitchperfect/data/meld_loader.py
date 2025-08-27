from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Literal, Tuple

from pitchperfect.utils.audio_processing import extract_wav


Split = Literal["train", "dev", "test"]

# Default organized layout created by `make organize-meld`:
# data/interim/MELD/{train,dev,test}
DEFAULT_MELD_INTERIM_ROOT = Path("data/interim/MELD")
DEFAULT_MELD_WAV_OUT = Path("data/processed/meld_wav")


def find_meld_split(root: str | Path = DEFAULT_MELD_INTERIM_ROOT, split: Split = "train") -> List[Path]:
    """Return a list of .mp4 files for the given MELD split.

    Defaults to the organized interim layout: data/interim/MELD/{split}/.../*.mp4
    """
    base = Path(root)
    split_dir = base / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(split_dir.rglob("*.mp4"))


def prepare_meld_split(
    root: str | Path = DEFAULT_MELD_INTERIM_ROOT,
    split: Split = "train",
    out_dir: str | Path = DEFAULT_MELD_WAV_OUT,
    sample_rate: int = 16000,
) -> List[Tuple[Path, Path]]:
    """Convert all .mp4 in a split to .wav under out_dir (organized interim assumed).

    Returns a list of tuples (mp4_path, wav_path).
    Skips corrupted/invalid MP4 files and continues processing.
    """
    mp4_files = find_meld_split(root, split)
    out = Path(out_dir)
    results: List[Tuple[Path, Path]] = []
    skipped = 0

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


def iter_wavs(
    root: str | Path = DEFAULT_MELD_INTERIM_ROOT,
    split: Split = "train",
    out_dir: str | Path = DEFAULT_MELD_WAV_OUT,
    sample_rate: int = 16000,
) -> Iterator[Path]:
    """Yield converted wav paths for the given split, converting on the fly if needed
    (organized interim assumed). Skips corrupted files."""
    mp4_files = find_meld_split(root, split)
    out = Path(out_dir)
    skipped = 0

    for mp4 in mp4_files:
        try:
            rel = mp4.relative_to(Path(root))
            wav_rel = rel.with_suffix(".wav")
            wav_path = out / wav_rel
            if not wav_path.exists():
                print(f"wav_path doesnt exist: {wav_path}")
                extract_wav(mp4, wav_path, sample_rate=sample_rate, mono=True)
            yield wav_path
        except Exception as e:
            print(f"⚠️  Skipping corrupted file {mp4.name}: {e}")
            skipped += 1
            continue

    if skipped > 0:
        print(f"⚠️  Skipped {skipped} corrupted files")
