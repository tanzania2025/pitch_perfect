#!/usr/bin/env python3
"""
Speech-to-Text Module Example

This example demonstrates how to use the speech-to-text module to transcribe audio files.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pitchperfect.speech_to_text.transcriber import Transcriber
from config import load_config


def demonstrate_speech_to_text():
    """Demonstrate speech-to-text functionality"""
    print("🎙️  Speech-to-Text Module Demo")
    print("=" * 50)

    # Load configuration
    config = load_config()

    # Initialize transcriber
    transcriber = Transcriber(config)

    # Example audio file path (you can replace with your own)
    audio_file = "data/sample_audio.wav"  # Replace with actual audio file

    if not os.path.exists(audio_file):
        print(f"⚠️  Sample audio file not found: {audio_file}")
        print("Creating a synthetic example instead...")

        # Mock result to show expected output format
        result = {
            "text": "Hello, this is a sample transcription of an audio file.",
            "language": "en",
            "duration": 3.5,
            "confidence": 0.95,
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.5,
                    "text": "Hello, this is a sample transcription of an audio file."
                }
            ]
        }
        print("📝 Transcription Result (Mock):")

    else:
        print(f"🎵 Processing audio file: {audio_file}")
        try:
            # Transcribe the audio
            result = transcriber.transcribe(audio_file)
            print("📝 Transcription Result:")
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return

    # Display results
    print(f"   Text: '{result['text']}'")
    print(f"   Language: {result.get('language', 'N/A')}")
    print(f"   Duration: {result.get('duration', 'N/A')} seconds")
    print(f"   Confidence: {result.get('confidence', 'N/A')}")

    if 'segments' in result:
        print(f"   Segments: {len(result['segments'])} segments")
        for i, segment in enumerate(result['segments'][:3]):  # Show first 3 segments
            print(f"     {i+1}. [{segment['start']:.1f}s - {segment['end']:.1f}s]: {segment['text']}")

    print("\n✅ Speech-to-Text demonstration completed!")


def show_transcriber_capabilities():
    """Show what the transcriber can do"""
    print("\n🔍 Transcriber Capabilities:")
    print("-" * 30)
    print("• Supports multiple audio formats (WAV, MP3, M4A, FLAC)")
    print("• Automatic language detection")
    print("• Timestamps for segments")
    print("• Confidence scores")
    print("• Noise reduction")
    print("• Multiple speaker detection (where applicable)")


if __name__ == "__main__":
    demonstrate_speech_to_text()
    show_transcriber_capabilities()

    print("\n💡 Usage Tips:")
    print("- Place audio files in the 'data/' directory")
    print("- Supported formats: .wav, .mp3, .m4a, .flac")
    print("- For best results, use clear audio with minimal background noise")
