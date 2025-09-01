# tests/test_pipeline.py
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from pitchperfect.pipeline.orchestrator import PipelineOrchestrator
from pitchperfect.text_sentiment_analysis.analyzer import TextSentimentAnalyzer
from pitchperfect.tonal_analysis.analyzer import TonalAnalyzer

def test_sentiment_analyzer():
    """Test sentiment analyzer"""
    config = load_config()
    analyzer = TextSentimentAnalyzer(config)

    result = analyzer.analyze("I am very happy today!")

    assert 'emotion' in result
    assert 'confidence' in result
    assert 'valence' in result
    assert result['sentiment'] in ['positive', 'negative', 'neutral']

def test_tonal_analyzer():
    """Test tonal analyzer with dummy audio"""
    import numpy as np

    config = load_config()
    analyzer = TonalAnalyzer(config)

    # Create dummy audio
    sr = 16000
    duration = 2
    audio = np.random.randn(sr * duration) * 0.1

    result = analyzer.analyze(audio)

    assert 'prosodic_features' in result
    assert 'voice_quality' in result
    assert 'acoustic_problems' in result

def test_pipeline_integration():
    """Test basic pipeline integration"""
    config = load_config()
    pipeline = PipelineOrchestrator(config)

    assert pipeline.transcriber is not None
    assert pipeline.sentiment_analyzer is not None
    assert pipeline.tonal_analyzer is not None
    assert pipeline.improvement_generator is not None
    assert pipeline.synthesizer is not None

if __name__ == "__main__":
    test_sentiment_analyzer()
    test_tonal_analyzer()
    test_pipeline_integration()
    print("All tests passed!")
