#!/usr/bin/env python3
"""
Speech-to-Text Analysis with MELD Dataset

This script demonstrates how to use the PitchPerfect speech-to-text capabilities
with the processed MELD dataset WAV files.
"""

from pitchperfect.speech_to_text import AudioTranscriber
from pitchperfect.data import meld_loader as ml
from pitchperfect.config.settings import MELD_WAV_DIR
from pathlib import Path
import os

def main():
    # Setup paths - use config paths for consistent directory resolution
    print(f"Using config paths for consistent directory resolution")

    DEFAULT_MELD_WAV_OUT = MELD_WAV_DIR
    print(f"WAV output directory: {DEFAULT_MELD_WAV_OUT}")
    print(f"Directory exists: {DEFAULT_MELD_WAV_OUT.exists()}")

    # List available WAV files
    base = Path(DEFAULT_MELD_WAV_OUT)
    split = 'dev'  # Can be 'train', 'dev', or 'test'
    split_dir = base / split
    print(f"Looking for WAV files in: {split_dir}")
    print(f"Split directory absolute path: {split_dir.absolute()}")

    if not split_dir.exists():
        print(f"❌ Split directory not found: {split_dir}")
        print(f"❌ Absolute path: {split_dir.absolute()}")
        print(f"❌ Parent directory exists: {split_dir.parent.exists()}")
        print(f"❌ Available in parent: {list(split_dir.parent.iterdir()) if split_dir.parent.exists() else 'N/A'}")
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    wav_files = sorted(split_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")
    print(f"First 5 files: {[f.name for f in wav_files[:5]]}")

    # Initialize transcriber and test single file
    t = AudioTranscriber(model="whisper-1", language="en", use_cache=True)

    if wav_files:
        # Test with first WAV file
        test_file = wav_files[0]
        print(f"Testing transcription with: {test_file.name}")

        try:
            res = t.transcribe(str(test_file))
            print(f"Transcript: {res['text']}")
            print(f"Language: {res.get('language', 'Unknown')}")
            print(f"Model: {res.get('model', 'Unknown')}")
        except Exception as e:
            print(f"Error transcribing {test_file.name}: {e}")
    else:
        print("No WAV files found to test")

    # Batch processing example
    if wav_files:
        # Test with first 3 files for batch processing
        test_files = [str(f) for f in wav_files[:3]]
        print(f"Testing batch transcription with {len(test_files)} files...")

        try:
            pairs = t.transcribe_batch(test_files)
            for path, r in pairs:
                filename = Path(path).name
                if 'error' in r:
                    print(f"❌ {filename}: {r['error']}")
                else:
                    print(f"✅ {filename}: {r['text'][:100]}...")
        except Exception as e:
            print(f"Error in batch processing: {e}")
    else:
        print("No WAV files found for batch processing")

    # Using meld_loader utilities
    print("\n" + "="*50)
    print("Using meld_loader utilities:")
    print("="*50)

    # List available splits
    available_splits = ['train', 'dev', 'test']
    for split_name in available_splits:
        try:
            wav_count = len(list((DEFAULT_MELD_WAV_OUT / split_name).rglob("*.wav")))
            print(f"{split_name}: {wav_count} WAV files")
        except Exception:
            print(f"{split_name}: No WAV files found")

    # Example of using iter_wavs (lazy loading)
    print(f"\nLazy loading WAV files from dev split:")
    try:
        wav_iterator = ml.iter_wavs(split='dev')
        count = 0
        for wav_path in wav_iterator:
            count += 1
            if count <= 3:  # Show first 3
                print(f"  {count}: {wav_path.name}")
            elif count == 4:
                print(f"  ... and {len(list(ml.find_meld_split(split='dev')))} more files")
                break
    except Exception as e:
        print(f"Error with iter_wavs: {e}")

if __name__ == "__main__":
    main()
