#!/usr/bin/env python3
"""
Text Sentiment Analysis Module Example

This example demonstrates how to use the sentiment analysis module to analyze emotions and sentiment in text.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pitchperfect.text_sentiment_analysis.analyzer import TextSentimentAnalyzer
from config import load_config


def demonstrate_sentiment_analysis():
    """Demonstrate sentiment analysis functionality"""
    print("🎭 Text Sentiment Analysis Module Demo")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Initialize sentiment analyzer
    analyzer = TextSentimentAnalyzer(config)
    
    # Sample texts with different sentiments
    sample_texts = [
        "I am absolutely thrilled about this amazing opportunity! This is fantastic!",
        "I'm feeling quite nervous about the presentation tomorrow. I hope it goes well.",
        "The weather is okay today. Nothing particularly exciting happening.",
        "I am disappointed with the results. This is really frustrating and annoying.",
        "Thank you so much for your help. I really appreciate your kindness and support.",
        "I'm excited about the new project, but also a bit worried about the deadline."
    ]
    
    print("📊 Analyzing sample texts...")
    print()
    
    for i, text in enumerate(sample_texts, 1):
        print(f"Text {i}: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        
        try:
            # Analyze sentiment
            result = analyzer.analyze(text)
            
            print(f"   Primary Emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
            print(f"   Overall Sentiment: {result.get('overall_sentiment', 'N/A')}")
            
            # Show emotion breakdown if available
            if 'emotions' in result:
                print("   Emotion Breakdown:")
                sorted_emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)
                for emotion, score in sorted_emotions[:3]:  # Top 3 emotions
                    print(f"     • {emotion}: {score:.3f}")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error analyzing text: {e}")
            print()
    
    print("✅ Sentiment Analysis demonstration completed!")


def demonstrate_emotion_detection():
    """Demonstrate emotion detection capabilities"""
    print("\n🧠 Emotion Detection Capabilities:")
    print("-" * 40)
    
    emotions_info = {
        "joy": "Happiness, excitement, satisfaction",
        "sadness": "Disappointment, melancholy, grief",
        "anger": "Frustration, irritation, rage",
        "fear": "Anxiety, worry, nervousness",
        "surprise": "Amazement, astonishment, wonder",
        "disgust": "Dislike, revulsion, distaste",
        "neutral": "Calm, balanced, objective"
    }
    
    for emotion, description in emotions_info.items():
        print(f"• {emotion.capitalize()}: {description}")


def show_use_cases():
    """Show practical use cases for sentiment analysis"""
    print("\n🎯 Practical Use Cases:")
    print("-" * 25)
    print("• Speech coaching: Identify emotional tone in presentations")
    print("• Content analysis: Analyze written content for emotional impact")
    print("• Customer feedback: Understand customer sentiment in reviews")
    print("• Communication training: Help improve emotional expression")
    print("• Mental health: Monitor emotional patterns in speech/text")


if __name__ == "__main__":
    demonstrate_sentiment_analysis()
    demonstrate_emotion_detection()
    show_use_cases()
    
    print("\n💡 Usage Tips:")
    print("- Longer texts generally provide more accurate analysis")
    print("- Consider context when interpreting results")
    print("- Multiple emotions can be present simultaneously")
    print("- Confidence scores help gauge reliability of predictions")