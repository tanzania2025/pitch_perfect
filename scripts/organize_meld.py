#!/usr/bin/env python3
"""Organize MELD.Raw data into interim folder with train/dev/test splits."""

import shutil
from pathlib import Path


def organize_meld_data():
    """Move and organize MELD data from external to interim."""
    external_dir = Path("data/external/MELD.Raw")
    interim_dir = Path("data/interim/MELD")

    if not external_dir.exists():
        print(f"❌ MELD.Raw directory not found at {external_dir}")
        return

    # Create interim structure
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Define source and destination mappings
    splits = {
        "train": external_dir / "train" / "train_splits",
        "dev": external_dir / "dev_splits_complete",
        "test": external_dir / "output_repeated_splits_test",
    }

    for split_name, source_dir in splits.items():
        if not source_dir.exists():
            print(f"⚠️  Source directory not found: {source_dir}")
            continue

        dest_dir = interim_dir / split_name
        dest_dir.mkdir(exist_ok=True)

        print(f"📁 Organizing {split_name} split...")

        # Copy MP4 files
        mp4_files = list(source_dir.glob("*.mp4"))
        for mp4_file in mp4_files:
            dest_file = dest_dir / mp4_file.name
            if not dest_file.exists():
                shutil.copy2(mp4_file, dest_file)
                print(f"  📹 Copied {mp4_file.name}")

        # Copy CSV files if they exist
        csv_files = list(source_dir.glob("*.csv"))
        for csv_file in csv_files:
            dest_file = dest_dir / csv_file.name
            if not dest_file.exists():
                shutil.copy2(csv_file, dest_file)
                print(f"  📊 Copied {csv_file.name}")

        print(
            f"  ✅ {split_name}: {len(mp4_files)} MP4 files, {len(csv_files)} CSV files"
        )

    print(f"\n🎯 MELD data organized in {interim_dir}")
    print("Next step: Convert MP4 to WAV using 'make meld-to-wav'")


if __name__ == "__main__":
    organize_meld_data()
