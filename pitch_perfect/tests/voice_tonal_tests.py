#!/usr/bin/env python3
"""
Test script for Voice Tonal Analysis System
This script helps you test the analyzer with different methods
"""

import os
import sys
import time
import numpy as np
from voice_tonal_analyzer import VoiceTonalAnalyzer
import soundfile as sf

def create_test_audio():
    """
    Create a simple test audio file for testing
    """
    print("Creating test audio file...")

    # Generate a simple sine wave with some variation
    duration = 10  # seconds
    sample_rate = 16000

    # Create time array
    t = np.linspace(0, duration, duration * sample_rate, False)

    # Create a varying frequency sine wave (simulating speech-like patterns)
    frequency_base = 150  # Base frequency (Hz)
    frequency_variation = np.sin(2 * np.pi * 0.5 * t) * 50  # Variation
    frequency = frequency_base + frequency_variation

    # Generate sine wave with amplitude variation
    amplitude_variation = 0.5 + 0.3 * np.sin(2 * np.pi * 0.3 * t)
    audio_signal = amplitude_variation * np.sin(2 * np.pi * frequency * t)

    # Add some pauses (silence)
    pause_start = int(3 * sample_rate)
    pause_end = int(4 * sample_rate)
    audio_signal[pause_start:pause_end] = 0

    pause_start2 = int(7 * sample_rate)
    pause_end2 = int(7.5 * sample_rate)
    audio_signal[pause_start2:pause_end2] = 0

    # Save the test audio
    test_file = "test_audio.wav"
    sf.write(test_file, audio_signal, sample_rate)
    print(f"Test audio created: {test_file}")
    return test_file

def test_with_sample_audio():
    """
    Test with the generated sample audio
    """
    print("\n" + "="*50)
    print("TESTING WITH GENERATED SAMPLE AUDIO")
    print("="*50)

    # Create test audio
    test_file = create_test_audio()

    # Initialize analyzer
    analyzer = VoiceTonalAnalyzer(whisper_model_size="tiny")  # Use tiny for faster testing

    # Analyze the test audio
    results = analyzer.analyze_audio(test_file)

    if results:
        print("\n✅ SUCCESS: Analysis completed!")
        print(analyzer.generate_summary(results))

        # Save results
        analyzer.save_results(results, "test_results.json")
        print("Test results saved to test_results.json")
    else:
        print("❌ FAILED: Analysis failed")

    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"Cleaned up test file: {test_file}")

def test_with_your_audio(audio_file_path):
    """
    Test with your own audio file
    """
    print(f"\n" + "="*50)
    print(f"TESTING WITH YOUR AUDIO: {audio_file_path}")
    print("="*50)

    if not os.path.exists(audio_file_path):
        print(f"❌ ERROR: File not found: {audio_file_path}")
        return False

    # Initialize analyzer
    analyzer = VoiceTonalAnalyzer(whisper_model_size="base")

    # Analyze your audio
    start_time = time.time()
    results = analyzer.analyze_audio(audio_file_path)
    end_time = time.time()

    if results:
        print(f"\n✅ SUCCESS: Analysis completed in {end_time - start_time:.2f} seconds!")
        print(analyzer.generate_summary(results))

        # Save results with timestamp
        output_file = f"analysis_results_{int(time.time())}.json"
        analyzer.save_results(results, output_file)
        print(f"Results saved to {output_file}")
        return True
    else:
        print("❌ FAILED: Analysis failed")
        return False

def record_and_test():
    """
    Record audio from microphone and test (requires pyaudio)
    """
    try:
        import pyaudio
        import wave
    except ImportError:
        print("❌ PyAudio not installed. Install with: pip install pyaudio")
        return

    print("\n" + "="*50)
    print("RECORDING FROM MICROPHONE")
    print("="*50)

    # Recording parameters
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 16000
    record_seconds = 10
    filename = "recorded_test.wav"

    p = pyaudio.PyAudio()

    print(f"🎙️  Recording for {record_seconds} seconds... Speak now!")

    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    frames = []
    for i in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

        # Progress indicator
        if i % (rate // chunk) == 0:
            print(f"Recording... {i // (rate // chunk) + 1}s")

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording finished!")

    # Save the recording
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio saved as {filename}")

    # Test the recorded audio
    success = test_with_your_audio(filename)

    # Ask if user wants to keep the file
    keep = input("Do you want to keep the recorded file? (y/n): ").lower().strip()
    if keep != 'y':
        os.remove(filename)
        print(f"Deleted {filename}")

def check_dependencies():
    """
    Check if all required dependencies are installed
    """
    print("Checking dependencies...")

    required_modules = [
        'librosa',
        'whisper',
        'textblob',
        'numpy',
        'scipy',
        'soundfile'
    ]

    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - MISSING")
            missing_modules.append(module)

    if missing_modules:
        print(f"\n⚠️  Missing modules: {', '.join(missing_modules)}")
        print("Install with: pip install " + " ".join(missing_modules))
        return False
    else:
        print("\n🎉 All required dependencies are installed!")
        return True

def main():
    """
    Main testing function with menu
    """
    print("🎤 VOICE TONAL ANALYZER - TEST SUITE")
    print("="*40)

    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before testing.")
        return

    while True:
        print("\nSelect a test option:")
        print("1. Test with generated sample audio (recommended first test)")
        print("2. Test with your own audio file")
        print("3. Record from microphone and test")
        print("4. Check system dependencies")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == '1':
            test_with_sample_audio()

        elif choice == '2':
            audio_path = input("Enter path to your audio file: ").strip()
            if audio_path:
                test_with_your_audio(audio_path)
            else:
                print("No file path provided.")

        elif choice == '3':
            record_and_test()

        elif choice == '4':
            check_dependencies()

        elif choice == '5':
            print("👋 Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

# Example usage for different scenarios
def quick_test_examples():
    """
    Quick test examples you can uncomment and run
    """

    # Example 1: Test with a specific file
    # test_with_your_audio("path/to/your/audio.wav")

    # Example 2: Test just the sample audio
    # test_with_sample_audio()

    # Example 3: Just check if everything is working
    # check_dependencies()

    pass

if __name__ == "__main__":
    # Run the main menu
    main()

    # Or uncomment one of these for direct testing:
    # test_with_sample_audio()  # Quick test
    # check_dependencies()      # Just check setup
