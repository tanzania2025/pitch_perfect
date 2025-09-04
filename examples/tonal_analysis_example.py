#!/usr/bin/env python3
"""
Tonal Analysis Module Example

This example demonstrates how to use the tonal analysis module to extract acoustic features and analyze speech patterns.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pitchperfect.tonal_analysis.analyzer import TonalAnalyzer
from config import load_config


def demonstrate_tonal_analysis():
    """Demonstrate tonal analysis functionality"""
    print("🎵 Tonal Analysis Module Demo")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Initialize tonal analyzer
    analyzer = TonalAnalyzer(config)
    
    # Example audio file path
    audio_file = "data/sample_audio.wav"
    
    if not os.path.exists(audio_file):
        print(f"⚠️  Sample audio file not found: {audio_file}")
        print("Creating a synthetic example instead...")
        
        # Mock result to show expected output format
        result = {
            "speaking_rate": 165.5,  # words per minute
            "pitch_mean": 145.2,     # Hz
            "pitch_std": 28.7,       # Hz
            "pitch_range": 85.3,     # Hz
            "volume_mean": 62.1,     # dB
            "volume_std": 8.4,       # dB
            "pause_count": 12,
            "pause_duration_mean": 0.65,  # seconds
            "pause_duration_total": 7.8,  # seconds
            "f0_variability": 0.19,
            "spectral_features": {
                "mfcc_mean": [1.2, -0.8, 0.3, 0.1, -0.2],
                "spectral_centroid": 2150.4,
                "spectral_rolloff": 4250.8,
                "zero_crossing_rate": 0.045
            },
            "prosody_features": {
                "speech_rhythm": "moderate",
                "pitch_contour": "varied",
                "energy_pattern": "consistent"
            },
            "acoustic_problems": ["speaking_too_fast", "insufficient_pauses"]
        }
        print("🎼 Tonal Analysis Result (Mock):")
        
    else:
        print(f"🎵 Processing audio file: {audio_file}")
        try:
            # Analyze the audio
            result = analyzer.analyze(audio_file)
            print("🎼 Tonal Analysis Result:")
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return
    
    # Display prosody results
    print("\n📊 Prosody Features:")
    print(f"   Speaking Rate: {result.get('speaking_rate', 'N/A')} WPM")
    print(f"   Pitch Mean: {result.get('pitch_mean', 'N/A')} Hz")
    print(f"   Pitch Variation: {result.get('pitch_std', 'N/A')} Hz")
    print(f"   Volume Mean: {result.get('volume_mean', 'N/A')} dB")
    print(f"   Pause Count: {result.get('pause_count', 'N/A')}")
    print(f"   Average Pause: {result.get('pause_duration_mean', 'N/A')} sec")
    print(f"   F0 Variability: {result.get('f0_variability', 'N/A')}")
    
    # Display spectral features
    if 'spectral_features' in result:
        spec = result['spectral_features']
        print("\n🎛️  Spectral Features:")
        print(f"   Spectral Centroid: {spec.get('spectral_centroid', 'N/A')} Hz")
        print(f"   Spectral Rolloff: {spec.get('spectral_rolloff', 'N/A')} Hz")
        print(f"   Zero Crossing Rate: {spec.get('zero_crossing_rate', 'N/A')}")
    
    # Display prosody assessment
    if 'prosody_features' in result:
        prosody = result['prosody_features']
        print("\n🎭 Prosody Assessment:")
        print(f"   Speech Rhythm: {prosody.get('speech_rhythm', 'N/A')}")
        print(f"   Pitch Contour: {prosody.get('pitch_contour', 'N/A')}")
        print(f"   Energy Pattern: {prosody.get('energy_pattern', 'N/A')}")
    
    # Display identified problems
    if 'acoustic_problems' in result and result['acoustic_problems']:
        print("\n⚠️  Identified Issues:")
        for problem in result['acoustic_problems']:
            print(f"   • {problem.replace('_', ' ').title()}")
    else:
        print("\n✅ No major acoustic issues detected!")
    
    print("\n✅ Tonal Analysis demonstration completed!")


def show_acoustic_features():
    """Explain what acoustic features are analyzed"""
    print("\n🔬 Analyzed Acoustic Features:")
    print("-" * 35)
    
    features = {
        "Speaking Rate": "Words per minute (WPM) - pace of speech",
        "Pitch (F0)": "Fundamental frequency - perceived as voice height",
        "Pitch Variation": "Standard deviation of pitch - voice expressiveness", 
        "Volume/Intensity": "Loudness level in decibels",
        "Pause Patterns": "Frequency and duration of silences",
        "Spectral Centroid": "Brightness/clarity of voice",
        "MFCC": "Mel-frequency coefficients - voice timbre",
        "Zero Crossing": "Rate of signal sign changes - voice roughness"
    }
    
    for feature, description in features.items():
        print(f"• {feature}: {description}")


def show_prosody_analysis():
    """Explain prosody analysis capabilities"""
    print("\n🎼 Prosody Analysis:")
    print("-" * 20)
    print("• Rhythm patterns in speech")
    print("• Pitch contour variations") 
    print("• Stress and emphasis patterns")
    print("• Energy distribution across speech")
    print("• Timing and phrasing analysis")
    print("• Emotional expression through voice")


def show_common_problems():
    """Show common acoustic problems that can be detected"""
    print("\n🚨 Detectable Speech Issues:")
    print("-" * 30)
    
    problems = [
        "Speaking too fast or too slow",
        "Monotone delivery (lack of pitch variation)",
        "Insufficient pauses between ideas",
        "Volume inconsistencies",
        "Poor voice quality/roughness",
        "Lack of emphasis on key points",
        "Irregular rhythm patterns"
    ]
    
    for problem in problems:
        print(f"• {problem}")


if __name__ == "__main__":
    demonstrate_tonal_analysis()
    show_acoustic_features()
    show_prosody_analysis()
    show_common_problems()
    
    print("\n💡 Usage Tips:")
    print("- Use high-quality audio recordings for best results")
    print("- Longer audio samples (30+ seconds) provide more reliable analysis")
    print("- Consider speaker's natural voice characteristics when interpreting results")
    print("- Combine with sentiment analysis for comprehensive speech assessment")