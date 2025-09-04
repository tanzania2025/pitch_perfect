#!/usr/bin/env python3
"""
Text-to-Speech Module Example

This example demonstrates how to use the text-to-speech module to synthesize improved speech with voice cloning.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pitchperfect.text_to_speech.synthesis import Synthesizer
from config import load_config


def demonstrate_text_to_speech():
    """Demonstrate text-to-speech functionality"""
    print("🔊 Text-to-Speech Module Demo")
    print("=" * 50)
    
    # Load configuration (Note: requires ElevenLabs API key)
    config = load_config()
    
    if not config.get('text_to_speech', {}).get('elevenlabs', {}).get('api_key'):
        print("⚠️  ElevenLabs API key not configured. Using mock examples...")
        demonstrate_with_mock_data()
        return
    
    try:
        # Initialize synthesizer
        synthesizer = Synthesizer(config)
        
        # Sample improved text and SSML
        improved_text = "Today I want to discuss artificial intelligence and how it is transforming our world."
        
        ssml_markup = '''<speak>
        Today I want to <emphasis level="moderate">discuss artificial intelligence</emphasis> 
        <break time="500ms"/> and how it is <emphasis level="strong">transforming</emphasis> 
        our <emphasis level="moderate">world</emphasis>.
        </speak>'''
        
        output_path = "outputs/generated_audio/example_synthesis.mp3"
        
        print(f"📝 Text to synthesize: \"{improved_text}\"")
        print(f"🎵 SSML markup: {ssml_markup}")
        print("\n🔄 Synthesizing speech...")
        
        # Basic synthesis
        result = synthesizer.synthesize(
            ssml=ssml_markup,
            output_path=output_path
        )
        
        display_synthesis_results(result, "Basic Synthesis")
        
        # Voice cloning example (if voice sample is available)
        voice_sample = "data/voice_sample.wav"
        if os.path.exists(voice_sample):
            print("\n🎭 Attempting voice cloning synthesis...")
            clone_result = synthesizer.synthesize_with_clone(
                text=improved_text,
                clone_audio_path=voice_sample,
                ssml=ssml_markup
            )
            display_synthesis_results(clone_result, "Voice Cloning Synthesis")
        else:
            print(f"\n⚠️  Voice sample not found: {voice_sample}")
            print("Voice cloning requires a sample audio file of the target voice.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Using mock data instead...")
        demonstrate_with_mock_data()


def demonstrate_with_mock_data():
    """Show example output with mock data"""
    print("\n📝 Text to synthesize:")
    print("\"Today I want to discuss artificial intelligence and how it is transforming our world.\"")
    
    print("\n🎵 SSML markup:")
    ssml = '''<speak>
    Today I want to <emphasis level="moderate">discuss artificial intelligence</emphasis> 
    <break time="500ms"/> and how it is <emphasis level="strong">transforming</emphasis> 
    our <emphasis level="moderate">world</emphasis>.
    </speak>'''
    print(ssml)
    
    # Mock synthesis results
    mock_results = {
        "basic": {
            "status": "success",
            "audio_length": 5.2,
            "voice_id": "default_voice",
            "output_path": "outputs/generated_audio/example_synthesis.mp3",
            "file_size": 83456,
            "sample_rate": 22050,
            "audio": b"mock_audio_data"  # Would be actual audio bytes
        },
        "cloned": {
            "status": "success", 
            "audio_length": 5.2,
            "voice_id": "cloned_voice_abc123",
            "clone_similarity": 0.92,
            "output_path": "outputs/generated_audio/cloned_synthesis.mp3",
            "file_size": 85234,
            "sample_rate": 22050,
            "audio": b"mock_cloned_audio_data"
        }
    }
    
    display_synthesis_results(mock_results["basic"], "Basic Synthesis (Mock)")
    display_synthesis_results(mock_results["cloned"], "Voice Cloning Synthesis (Mock)")


def display_synthesis_results(result, title):
    """Display synthesis results"""
    print(f"\n✨ {title} Results:")
    print("-" * (len(title) + 10))
    
    if result.get("status") == "success":
        print(f"✅ Status: {result['status']}")
        print(f"⏱️  Audio Length: {result.get('audio_length', 'N/A')} seconds")
        print(f"🗣️  Voice ID: {result.get('voice_id', 'N/A')}")
        
        if 'clone_similarity' in result:
            print(f"🎭 Clone Similarity: {result['clone_similarity']:.2%}")
        
        print(f"💾 Output Path: {result.get('output_path', 'N/A')}")
        print(f"📁 File Size: {result.get('file_size', 0):,} bytes")
        print(f"🎵 Sample Rate: {result.get('sample_rate', 'N/A')} Hz")
        
        if result.get('audio'):
            print(f"🎧 Audio Data: {len(result['audio']):,} bytes available")
    else:
        print(f"❌ Status: {result.get('status', 'failed')}")
        if 'error' in result:
            print(f"Error: {result['error']}")


def show_voice_options():
    """Show available voice options and features"""
    print("\n🎭 Voice Options & Features:")
    print("-" * 30)
    
    print("📢 ElevenLabs Preset Voices:")
    voices = [
        "Rachel - Calm, young adult female",
        "Drew - Well-rounded, middle-aged male", 
        "Clyde - Middle-aged, warm male",
        "Paul - Authoritative, middle-aged male",
        "Domi - Strong, confident female",
        "Dave - Smooth, conversational male",
        "Fin - Pleasant, approachable male"
    ]
    
    for voice in voices:
        print(f"   • {voice}")
    
    print("\n🔊 Synthesis Features:")
    features = [
        "SSML support for emphasis and pauses",
        "Voice cloning from audio samples",
        "Adjustable speech rate and pitch",
        "Multiple audio formats (MP3, WAV)",
        "High-quality neural voice synthesis",
        "Real-time streaming capabilities"
    ]
    
    for feature in features:
        print(f"   • {feature}")


def show_ssml_examples():
    """Show SSML markup examples"""
    print("\n📝 SSML Markup Examples:")
    print("-" * 25)
    
    examples = {
        "Basic emphasis": '<speak>This is <emphasis level="strong">very important</emphasis>.</speak>',
        "Pauses": '<speak>First point. <break time="1s"/> Second point.</speak>',
        "Speed control": '<speak><prosody rate="slow">Speak slowly</prosody> and <prosody rate="fast">speak quickly</prosody>.</speak>',
        "Pitch control": '<speak><prosody pitch="high">High pitch</prosody> and <prosody pitch="low">low pitch</prosody>.</speak>',
        "Combined": '<speak>Welcome to <emphasis level="moderate">Pitch Perfect</emphasis>. <break time="500ms"/> Let\'s <prosody rate="110%">improve your speech</prosody>!</speak>'
    }
    
    for name, ssml in examples.items():
        print(f"\n{name}:")
        print(f"   {ssml}")


def show_voice_cloning_tips():
    """Show tips for voice cloning"""
    print("\n🎭 Voice Cloning Tips:")
    print("-" * 22)
    
    tips = [
        "Use high-quality audio samples (22kHz or higher)",
        "Sample should be 30-60 seconds long",
        "Clear speech without background noise",
        "Single speaker only in the sample",
        "Varied emotional expressions improve cloning",
        "Multiple samples can be combined for better results"
    ]
    
    for tip in tips:
        print(f"   • {tip}")


if __name__ == "__main__":
    demonstrate_text_to_speech()
    show_voice_options()
    show_ssml_examples()
    show_voice_cloning_tips()
    
    print("\n💡 Usage Tips:")
    print("- Requires ElevenLabs API key for full functionality")
    print("- SSML markup provides fine control over synthesis")
    print("- Voice cloning works best with high-quality samples")
    print("- Output audio can be used directly or processed further")
    print("- Consider audio format and quality requirements for your use case")