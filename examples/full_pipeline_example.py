#!/usr/bin/env python3
"""
Full Pipeline Example

This example demonstrates how to use the complete Pitch Perfect pipeline to process audio from start to finish.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pitchperfect.pipeline.orchestrator import PipelineOrchestrator
from config import load_config


def demonstrate_full_pipeline():
    """Demonstrate the complete pipeline functionality"""
    print("🚀 Full Pipeline Demo")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Check for required API keys
    missing_keys = []
    if not config.get('llm_processing', {}).get('openai', {}).get('api_key'):
        missing_keys.append("OpenAI")
    if not config.get('text_to_speech', {}).get('elevenlabs', {}).get('api_key'):
        missing_keys.append("ElevenLabs")
    
    if missing_keys:
        print(f"⚠️  Missing API keys for: {', '.join(missing_keys)}")
        print("Using mock pipeline example instead...")
        demonstrate_with_mock_data()
        return
    
    # Initialize pipeline orchestrator
    try:
        orchestrator = PipelineOrchestrator(config)
        
        # Example inputs
        audio_file = "data/sample_speech.wav"
        voice_sample = "data/voice_sample.wav"  # Optional for voice cloning
        
        user_preferences = {
            "target_style": "professional",
            "improvement_focus": "clarity"
        }
        
        if not os.path.exists(audio_file):
            print(f"⚠️  Sample audio file not found: {audio_file}")
            print("Using mock pipeline example instead...")
            demonstrate_with_mock_data()
            return
        
        print(f"🎵 Processing audio file: {audio_file}")
        if os.path.exists(voice_sample):
            print(f"🎭 Using voice sample for cloning: {voice_sample}")
        
        print("\n🔄 Running complete pipeline...")
        
        # Process through complete pipeline
        results = orchestrator.process(
            audio_path=audio_file,
            voice_sample_path=voice_sample if os.path.exists(voice_sample) else None,
            user_preferences=user_preferences
        )
        
        display_pipeline_results(results)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Using mock pipeline example instead...")
        demonstrate_with_mock_data()


def demonstrate_with_mock_data():
    """Show example output with mock data"""
    print("\n🎵 Input: sample_speech.wav")
    print("🎭 Voice Sample: voice_sample.wav")
    print("⚙️  Preferences: Professional style, focus on clarity")
    
    # Mock complete pipeline results
    mock_results = {
        "timestamp": "2024-03-15T10:30:00",
        "input_audio": "data/sample_speech.wav",
        
        "transcription": {
            "text": "Um, so like, today I want to talk about, uh, artificial intelligence and how it's, you know, changing our world. It's like, really important stuff.",
            "language": "en",
            "duration": 8.5,
            "confidence": 0.92,
            "segments": [
                {"start": 0.0, "end": 8.5, "text": "Um, so like, today I want to talk about, uh, artificial intelligence and how it's, you know, changing our world. It's like, really important stuff."}
            ]
        },
        
        "sentiment": {
            "emotion": "neutral",
            "confidence": 0.73,
            "overall_sentiment": "neutral",
            "emotions": {
                "neutral": 0.6,
                "excitement": 0.25,
                "uncertainty": 0.15
            }
        },
        
        "tonal": {
            "speaking_rate": 185,
            "pitch_mean": 148,
            "pitch_std": 22,
            "volume_mean": 63,
            "pause_count": 6,
            "pause_duration_mean": 0.4,
            "f0_variability": 0.12,
            "acoustic_problems": ["speaking_too_fast", "insufficient_pauses", "filler_words"]
        },
        
        "improvements": {
            "improved_text": "Today I want to discuss artificial intelligence and how it is transforming our world. This is truly important subject matter.",
            "issues": {
                "text_issues": ["filler_words", "informal_language"],
                "delivery_issues": ["speaking_too_fast", "insufficient_pauses"],
                "severity": "medium",
                "confidence": 0.85
            },
            "feedback": {
                "summary": "Your speech is good but could benefit from some refinements.",
                "key_improvements": [
                    "Removed filler words for clearer communication",
                    "Improved sentence structure and flow",
                    "Adjusted pacing for better comprehension"
                ],
                "speaking_tips": [
                    "Slow down slightly for better clarity",
                    "Add strategic pauses between key points",
                    "Use more formal language for professional presentations"
                ],
                "severity": "medium",
                "issues_found": 4
            },
            "prosody_guide": {
                "rate_multiplier": 0.85,
                "pitch_multiplier": 1.1,
                "pitch_variation_multiplier": 1.4,
                "volume_multiplier": 1.0,
                "pause_locations": [2, 5, 9],
                "emphasis_count": 3,
                "target_wpm": 155
            },
            "ssml_markup": '<speak>Today I want to <emphasis level="moderate">discuss artificial intelligence</emphasis> <break time="500ms"/> and how it is <emphasis level="strong">transforming</emphasis> our <emphasis level="moderate">world</emphasis>. <break time="300ms"/> This is truly <emphasis level="moderate">important subject matter</emphasis>.</speak>',
            "summary_feedback": "Your speech is good but could benefit from some refinements. Found 4 areas for improvement. Improvements: Removed filler words for clearer communication; Adjusted pacing for better comprehension. Key tip: Slow down slightly for better clarity"
        },
        
        "synthesis": {
            "status": "success",
            "audio_length": 7.2,
            "voice_id": "cloned_voice_xyz789",
            "clone_similarity": 0.89,
            "output_path": "outputs/generated_audio/improved_speech.mp3",
            "file_size": 115680,
            "sample_rate": 22050
        },
        
        "metrics": {
            "processing_time_seconds": 12.7,
            "original_word_count": 24,
            "improved_word_count": 18,
            "issues_found": 4,
            "severity": "medium"
        }
    }
    
    display_pipeline_results(mock_results)


def display_pipeline_results(results):
    """Display complete pipeline results"""
    print("\n🎯 Complete Pipeline Results:")
    print("=" * 35)
    
    # Transcription results
    if 'transcription' in results:
        trans = results['transcription']
        print(f"\n🎤 Speech-to-Text:")
        print(f"   Original Text: \"{trans['text'][:80]}{'...' if len(trans['text']) > 80 else ''}\"")
        print(f"   Duration: {trans.get('duration', 'N/A')} seconds")
        print(f"   Confidence: {trans.get('confidence', 'N/A'):.2%}")
    
    # Sentiment results
    if 'sentiment' in results:
        sent = results['sentiment']
        print(f"\n🎭 Sentiment Analysis:")
        print(f"   Primary Emotion: {sent['emotion']} ({sent['confidence']:.2%})")
        print(f"   Overall Sentiment: {sent.get('overall_sentiment', 'N/A')}")
    
    # Tonal analysis results
    if 'tonal' in results:
        tonal = results['tonal']
        print(f"\n🎵 Tonal Analysis:")
        print(f"   Speaking Rate: {tonal.get('speaking_rate', 'N/A')} WPM")
        print(f"   Pitch Mean: {tonal.get('pitch_mean', 'N/A')} Hz")
        print(f"   Issues Detected: {len(tonal.get('acoustic_problems', []))}")
    
    # LLM improvements
    if 'improvements' in results:
        imp = results['improvements']
        print(f"\n🤖 LLM Improvements:")
        print(f"   Improved Text: \"{imp['improved_text'][:80]}{'...' if len(imp['improved_text']) > 80 else ''}\"")
        print(f"   Issues Found: {imp['feedback'].get('issues_found', 0)}")
        print(f"   Severity: {imp['feedback'].get('severity', 'N/A')}")
        
        if 'summary_feedback' in imp:
            print(f"\n📋 Summary Feedback:")
            print(f"   {imp['summary_feedback']}")
    
    # TTS results
    if 'synthesis' in results:
        synth = results['synthesis']
        if synth.get('status') == 'success':
            print(f"\n🔊 Text-to-Speech:")
            print(f"   Status: ✅ {synth['status']}")
            print(f"   Audio Length: {synth.get('audio_length', 'N/A')} seconds")
            print(f"   Output File: {synth.get('output_path', 'N/A')}")
            if 'clone_similarity' in synth:
                print(f"   Voice Clone Similarity: {synth['clone_similarity']:.2%}")
        else:
            print(f"\n🔊 Text-to-Speech: ❌ {synth.get('status', 'failed')}")
    
    # Performance metrics
    if 'metrics' in results:
        metrics = results['metrics']
        print(f"\n📊 Performance Metrics:")
        print(f"   Processing Time: {metrics.get('processing_time_seconds', 'N/A')} seconds")
        print(f"   Word Reduction: {metrics.get('original_word_count', 0)} → {metrics.get('improved_word_count', 0)}")
        word_reduction = 0
        if metrics.get('original_word_count', 0) > 0:
            word_reduction = ((metrics['original_word_count'] - metrics['improved_word_count']) / metrics['original_word_count']) * 100
        print(f"   Efficiency Gain: {word_reduction:.1f}% more concise")


def show_pipeline_stages():
    """Explain the pipeline stages"""
    print("\n🔄 Pipeline Stages:")
    print("-" * 20)
    
    stages = [
        ("1. Speech-to-Text", "Transcribe audio using OpenAI Whisper"),
        ("2. Sentiment Analysis", "Analyze emotional content using transformer models"),
        ("3. Tonal Analysis", "Extract acoustic features and prosody patterns"),
        ("4. LLM Processing", "Improve text content and generate speaking guidance"),
        ("5. Text-to-Speech", "Synthesize improved speech with voice cloning")
    ]
    
    for stage, description in stages:
        print(f"{stage}: {description}")


def show_integration_benefits():
    """Show benefits of integrated pipeline"""
    print("\n🌟 Integration Benefits:")
    print("-" * 25)
    
    benefits = [
        "End-to-end speech improvement automation",
        "Context-aware processing across all stages", 
        "Consistent voice characteristics preserved",
        "Comprehensive feedback combining all analyses",
        "Optimized processing with shared data",
        "Professional-quality output generation"
    ]
    
    for benefit in benefits:
        print(f"   • {benefit}")


if __name__ == "__main__":
    demonstrate_full_pipeline()
    show_pipeline_stages()
    show_integration_benefits()
    
    print("\n💡 Usage Tips:")
    print("- Requires both OpenAI and ElevenLabs API keys")
    print("- Use high-quality input audio for best results")
    print("- Voice samples enable personalized output")
    print("- Processing time varies with audio length and complexity")
    print("- Results improve with longer, clearer input audio")