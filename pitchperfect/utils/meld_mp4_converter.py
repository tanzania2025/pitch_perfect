#!/usr/bin/env python3
"""
MELD MP4 to Audio Converter
Converts MP4 video files from MELD dataset to audio files for emotion recognition
"""

import os
import subprocess
import pandas as pd
from pathlib import Path
import shutil

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg found and ready")
            return True
        else:
            return False
    except FileNotFoundError:
        return False

def install_ffmpeg_instructions():
    """Provide instructions to install ffmpeg"""
    print("❌ FFmpeg not found. Please install it first:")
    print()
    print("🍎 macOS:")
    print("   brew install ffmpeg")
    print()
    print("🐧 Ubuntu/Debian:")
    print("   sudo apt update && sudo apt install ffmpeg")
    print()
    print("🪟 Windows:")
    print("   1. Download from https://ffmpeg.org/download.html")
    print("   2. Add to system PATH")
    print("   3. Or use: winget install ffmpeg")
    print()
    print("After installing, run this script again.")

def find_mp4_files():
    """Find all MP4 files in the directory structure"""
    print("🔍 Searching for MP4 files...")
    print("=" * 40)

    found_files = {
        'csv_files': [],
        'mp4_folders': [],
        'mp4_files': []
    }

    # Search for CSV files
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            if file.endswith('.csv') and any(keyword in file.lower() for keyword in ['train', 'dev', 'test', 'meld']):
                full_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(full_path)
                    found_files['csv_files'].append({
                        'path': full_path,
                        'name': file,
                        'rows': len(df),
                        'columns': list(df.columns)
                    })
                except:
                    pass

        # Look for MP4 files
        mp4_files = [f for f in files if f.endswith('.mp4')]
        if mp4_files:
            found_files['mp4_folders'].append({
                'path': root,
                'count': len(mp4_files),
                'sample_files': mp4_files[:5]
            })
            found_files['mp4_files'].extend([os.path.join(root, f) for f in mp4_files])

    return found_files

def display_found_files(found_files):
    """Display what MP4 files were found"""
    print("\n📄 CSV Files Found:")
    for csv_info in found_files['csv_files']:
        print(f"✅ {csv_info['name']}")
        print(f"   📁 Location: {csv_info['path']}")
        print(f"   📊 Rows: {csv_info['rows']}")
        if 'Emotion' in csv_info['columns']:
            try:
                df = pd.read_csv(csv_info['path'])
                emotions = df['Emotion'].unique()
                print(f"   😊 Emotions: {list(emotions)}")
            except:
                pass
        print()

    print("🎬 MP4 Video Files Found:")
    if not found_files['mp4_folders']:
        print("❌ No MP4 files found")
    else:
        total_mp4 = sum(folder['count'] for folder in found_files['mp4_folders'])
        print(f"Total MP4 files: {total_mp4}")
        print()

        for folder_info in found_files['mp4_folders']:
            print(f"📁 {folder_info['path']}")
            print(f"   🎬 MP4 files: {folder_info['count']}")
            print(f"   📝 Examples: {folder_info['sample_files']}")
            print()

def convert_mp4_to_wav(mp4_path, wav_path):
    """Convert a single MP4 file to WAV audio"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)

        # FFmpeg command to extract audio
        cmd = [
            'ffmpeg',
            '-i', mp4_path,           # Input MP4 file
            '-vn',                    # No video
            '-acodec', 'pcm_s16le',   # Audio codec: 16-bit PCM
            '-ar', '16000',           # Sample rate: 16kHz
            '-ac', '1',               # Mono audio
            '-y',                     # Overwrite output file
            wav_path                  # Output WAV file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True
        else:
            print(f"❌ Error converting {mp4_path}: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Exception converting {mp4_path}: {e}")
        return False

def batch_convert_mp4s(found_files, max_files_per_folder=None):
    """Convert MP4 files to WAV files"""
    print("🎵 Converting MP4 files to WAV audio...")
    print("=" * 50)

    # Mapping of source folders to target folders
    folder_mapping = {
        'train': 'train',
        'dev': 'dev_splits_complete',
        'test': 'output_repeated_splits_test'
    }

    total_converted = 0
    total_failed = 0

    for mp4_folder in found_files['mp4_folders']:
        source_path = mp4_folder['path']
        folder_name = os.path.basename(source_path).lower()

        # Determine target folder
        target_folder = None
        for key, value in folder_mapping.items():
            if key in folder_name:
                target_folder = value
                break

        if not target_folder:
            # Try to infer from parent directory names
            parent_parts = source_path.lower().split(os.sep)
            for key, value in folder_mapping.items():
                if any(key in part for part in parent_parts):
                    target_folder = value
                    break

        if not target_folder:
            print(f"⚠️  Could not determine target folder for {source_path}")
            target_folder = 'unknown'

        print(f"\n📁 Processing {source_path} → {target_folder}/")

        # Get MP4 files
        mp4_files = [f for f in os.listdir(source_path) if f.endswith('.mp4')]

        if max_files_per_folder:
            mp4_files = mp4_files[:max_files_per_folder]
            print(f"   📝 Converting first {len(mp4_files)} files (limited for testing)")
        else:
            print(f"   📝 Converting all {len(mp4_files)} files")

        # Convert each MP4 file
        for i, mp4_file in enumerate(mp4_files, 1):
            mp4_path = os.path.join(source_path, mp4_file)

            # Generate WAV filename
            wav_filename = os.path.splitext(mp4_file)[0] + '.wav'
            wav_path = os.path.join(target_folder, wav_filename)

            # Skip if already exists
            if os.path.exists(wav_path):
                continue

            print(f"   🔄 Converting {i}/{len(mp4_files)}: {mp4_file}", end=" ... ")

            if convert_mp4_to_wav(mp4_path, wav_path):
                print("✅")
                total_converted += 1
            else:
                print("❌")
                total_failed += 1

            # Progress update every 10 files
            if i % 10 == 0:
                print(f"   📊 Progress: {i}/{len(mp4_files)} files processed")

    print(f"\n🎉 Conversion Complete!")
    print(f"✅ Successfully converted: {total_converted} files")
    if total_failed > 0:
        print(f"❌ Failed to convert: {total_failed} files")

def organize_csv_files(found_files):
    """Organize CSV files to expected locations"""
    print("\n📄 Organizing CSV Files...")
    print("=" * 30)

    target_files = {
        'train_sent_emo.csv': ['train', 'training'],
        'dev_sent_emo.csv': ['dev', 'development', 'val', 'validation'],
        'test_sent_emo.csv': ['test', 'testing']
    }

    for target_name, keywords in target_files.items():
        found = False

        for csv_info in found_files['csv_files']:
            csv_name = csv_info['name'].lower()
            csv_path = csv_info['path'].lower()

            if any(keyword in csv_name or keyword in csv_path for keyword in keywords):
                source_path = csv_info['path']

                if source_path != target_name:
                    print(f"📄 {csv_info['name']} → {target_name}")
                    try:
                        shutil.copy2(source_path, target_name)
                        print(f"   ✅ Copied successfully")
                    except Exception as e:
                        print(f"   ❌ Copy failed: {e}")
                else:
                    print(f"📄 {target_name} already in place")

                found = True
                break

        if not found:
            print(f"❌ Could not find CSV file for {target_name}")

def verify_setup():
    """Verify the final setup"""
    print("\n✅ Verification:")
    print("=" * 30)

    # Check CSV files
    csv_files = ['train_sent_emo.csv', 'dev_sent_emo.csv', 'test_sent_emo.csv']
    csv_ok = True

    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            print(f"✅ {csv_file}: {len(df)} samples")
        else:
            print(f"❌ {csv_file}: Missing")
            csv_ok = False

    # Check audio folders
    audio_folders = ['train', 'dev_splits_complete', 'output_repeated_splits_test']
    audio_ok = True

    for folder in audio_folders:
        if os.path.exists(folder):
            wav_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
            if len(wav_files) > 0:
                print(f"✅ {folder}/: {len(wav_files)} WAV files")
            else:
                print(f"⚠️  {folder}/: No WAV files")
                audio_ok = False
        else:
            print(f"❌ {folder}/: Folder missing")
            audio_ok = False

    if csv_ok and audio_ok:
        print(f"\n🎉 Setup Complete! Ready to run:")
        print(f"python standalone_meld_test.py")
    else:
        print(f"\n⚠️  Setup incomplete. Check the issues above.")

def main():
    """Main function"""
    print("🎬 MELD MP4 to Audio Converter")
    print("=" * 40)
    print("This script converts MELD MP4 video files to WAV audio files")
    print()

    # Check if ffmpeg is available
    if not check_ffmpeg():
        install_ffmpeg_instructions()
        return

    # Find MP4 files
    found_files = find_mp4_files()
    display_found_files(found_files)

    if not found_files['mp4_files']:
        print("❌ No MP4 files found!")
        print("Make sure you're in the directory with MELD MP4 video files.")
        return

    total_mp4s = len(found_files['mp4_files'])
    print(f"📊 Found {total_mp4s} MP4 files total")

    # Ask what to do
    print(f"\nOptions:")
    print(f"1. Convert first 20 MP4s per folder (quick test)")
    print(f"2. Convert first 100 MP4s per folder (medium test)")
    print(f"3. Convert ALL MP4s (full dataset - may take 30+ minutes)")
    print(f"4. Just organize CSV files (no conversion)")

    choice = input(f"\nSelect option (1/2/3/4): ").strip()

    # Organize CSV files first
    organize_csv_files(found_files)

    if choice == '1':
        batch_convert_mp4s(found_files, max_files_per_folder=20)
        verify_setup()
    elif choice == '2':
        batch_convert_mp4s(found_files, max_files_per_folder=100)
        verify_setup()
    elif choice == '3':
        print(f"\n⚠️  This will convert {total_mp4s} files and may take 30+ minutes.")
        confirm = input(f"Continue? (y/n): ").lower().strip()
        if confirm == 'y':
            batch_convert_mp4s(found_files)
            verify_setup()
        else:
            print("👋 Conversion cancelled.")
    elif choice == '4':
        verify_setup()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
