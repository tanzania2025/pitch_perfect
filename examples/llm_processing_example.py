#!/usr/bin/env python3
"""
LLM Processing Module Example

This example demonstrates how to use the LLM processing module to improve text content and generate speech markup.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pitchperfect.llm_processing.improvement_generator import ImprovementGenerator
from config import load_config


def demonstrate_llm_processing():
    """Demonstrate LLM processing functionality"""
    print("🤖 LLM Processing Module Demo")
    print("=" * 50)
    
    # Load configuration (Note: requires OpenAI API key)
    config = load_config()
    
    if not config.get('llm_processing', {}).get('openai', {}).get('api_key'):
        print("⚠️  OpenAI API key not configured. Using mock examples...")
        demonstrate_with_mock_data()
        return
    
    try:
        # Initialize improvement generator
        generator = ImprovementGenerator(config)
        
        # Sample inputs
        original_text = "Um, so like, today I want to talk about, uh, artificial intelligence and how it's, you know, changing our world. It's like, really important stuff."
        
        text_sentiment = {
            "emotion": "neutral",
            "confidence": 0.75,
            "overall_sentiment": "neutral",
            "emotions": {
                "neutral": 0.6,
                "excitement": 0.2,
                "uncertainty": 0.2
            }
        }
        
        acoustic_features = {
            "speaking_rate": 180,  # words per minute
            "pitch_mean": 150,
            "pitch_std": 25,
            "volume_mean": 65,
            "pause_count": 8,
            "pause_duration_mean": 0.8,
            "f0_variability": 0.15,
            "acoustic_problems": ["speaking_too_fast", "insufficient_pauses"]
        }
        
        user_preferences = {
            "target_style": "professional",
            "improvement_focus": "clarity"
        }
        
        print(f"📝 Original Text: \"{original_text}\"")
        print("\n🔄 Processing improvements...")
        
        # Generate improvements
        result = generator.generate_improvements(
            text=original_text,
            text_sentiment=text_sentiment,
            acoustic_features=acoustic_features,
            user_preferences=user_preferences
        )
        
        display_improvement_results(result)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Using mock data instead...")
        demonstrate_with_mock_data()


def demonstrate_with_mock_data():
    """Show example output with mock data"""
    print("\n📝 Original Text:")
    print("\"Um, so like, today I want to talk about, uh, artificial intelligence and how it's, you know, changing our world.\"")
    
    # Mock improvement result
    from dataclasses import dataclass
    from typing import Dict, List
    
    @dataclass
    class MockResult:
        improved_text: str
        original_text: str
        prosody_guide: Dict
        ssml_markup: str
        issues: Dict
        feedback: Dict
        emphasis_words: List
        summary_feedback: str
    
    mock_result = MockResult(
        improved_text="Today I want to discuss artificial intelligence and how it is transforming our world.",
        original_text="Um, so like, today I want to talk about, uh, artificial intelligence and how it's, you know, changing our world.",
        prosody_guide={
            "rate_multiplier": 0.9,
            "pitch_multiplier": 1.1,
            "pitch_variation_multiplier": 1.3,
            "volume_multiplier": 1.0,
            "pause_locations": [2, 5, 8],
            "emphasis_count": 3,
            "target_wpm": 160
        },
        ssml_markup='<speak>Today I want to <emphasis level="moderate">discuss artificial intelligence</emphasis> <break time="500ms"/> and how it is <emphasis level="strong">transforming</emphasis> our <emphasis level="moderate">world</emphasis>.</speak>',
        issues={
            "text_issues": ["filler_words", "informal_language"],
            "delivery_issues": ["speaking_too_fast", "insufficient_pauses"],
            "severity": "medium",
            "confidence": 0.85
        },
        feedback={
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
        emphasis_words=["artificial intelligence", "transforming", "world"],
        summary_feedback="Your speech is good but could benefit from some refinements. Found 4 areas for improvement. Improvements: Removed filler words for clearer communication; Adjusted pacing for better comprehension. Key tip: Slow down slightly for better clarity"
    )
    
    display_improvement_results(mock_result)


def display_improvement_results(result):
    """Display the improvement results"""
    print("\n✨ Improvement Results:")
    print("-" * 25)
    
    print(f"📝 Improved Text: \"{result.improved_text}\"")
    
    print(f"\n🎯 Issues Identified:")
    issues = result.issues
    if isinstance(issues, dict):
        if 'text_issues' in issues:
            print(f"   Text Issues: {', '.join(issues['text_issues'])}")
        if 'delivery_issues' in issues:
            print(f"   Delivery Issues: {', '.join(issues['delivery_issues'])}")
        print(f"   Severity: {issues.get('severity', 'N/A')}")
    
    print(f"\n💡 Key Improvements:")
    for improvement in result.feedback.get('key_improvements', []):
        print(f"   • {improvement}")
    
    print(f"\n🗣️  Speaking Tips:")
    for tip in result.feedback.get('speaking_tips', []):
        print(f"   • {tip}")
    
    print(f"\n🎵 Prosody Adjustments:")
    prosody = result.prosody_guide
    print(f"   • Rate: {prosody.get('rate_multiplier', 1.0):.1f}x (Target: {prosody.get('target_wpm', 'N/A')} WPM)")
    print(f"   • Pitch Variation: {prosody.get('pitch_variation_multiplier', 1.0):.1f}x")
    print(f"   • Emphasis Points: {prosody.get('emphasis_count', 0)}")
    print(f"   • Strategic Pauses: {len(prosody.get('pause_locations', []))}")
    
    print(f"\n🔤 Emphasis Words: {', '.join(result.emphasis_words)}")
    
    print(f"\n📢 SSML Markup:")
    print(f"   {result.ssml_markup}")
    
    print(f"\n📋 Summary Feedback:")
    print(f"   {result.summary_feedback}")


def show_improvement_capabilities():
    """Show what the LLM processing can improve"""
    print("\n🛠️  LLM Processing Capabilities:")
    print("-" * 35)
    
    capabilities = {
        "Text Improvements": [
            "Remove filler words (um, uh, like, you know)",
            "Improve sentence structure and grammar",
            "Enhance vocabulary and word choice",
            "Adjust formality level",
            "Fix repetitive phrases"
        ],
        "Speech Coaching": [
            "Identify speaking pace issues",
            "Suggest strategic pause locations",
            "Recommend emphasis points",
            "Provide personalized speaking tips",
            "Generate SSML for text-to-speech"
        ],
        "Style Adaptation": [
            "Professional presentation style",
            "Casual conversational tone",
            "Academic/formal language",
            "Motivational/inspirational tone",
            "Storytelling enhancement"
        ]
    }
    
    for category, items in capabilities.items():
        print(f"\n{category}:")
        for item in items:
            print(f"   • {item}")


def show_user_preferences():
    """Show available user preference options"""
    print("\n⚙️  User Preference Options:")
    print("-" * 30)
    
    preferences = {
        "target_style": ["professional", "casual", "academic", "motivational"],
        "improvement_focus": ["clarity", "confidence", "engagement", "all"],
        "formality_level": ["formal", "neutral", "informal"],
        "pace_preference": ["slower", "maintain", "faster"]
    }
    
    for pref, options in preferences.items():
        print(f"{pref}: {', '.join(options)}")


if __name__ == "__main__":
    demonstrate_llm_processing()
    show_improvement_capabilities()
    show_user_preferences()
    
    print("\n💡 Usage Tips:")
    print("- Requires OpenAI API key for full functionality")
    print("- Longer text samples provide better improvement suggestions")
    print("- Combine with acoustic analysis for comprehensive feedback")
    print("- Experiment with different target styles for various scenarios")
    print("- SSML output can be used directly with text-to-speech systems")