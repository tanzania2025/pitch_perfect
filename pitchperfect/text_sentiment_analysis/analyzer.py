# pitchperfect/text_sentiment_analysis/analyzer.py
from typing import Dict, List, Optional
from transformers import pipeline
import torch
from .preprocessing import TextPreprocessor
import logging

logger = logging.getLogger(__name__)


class TextSentimentAnalyzer:
    """Sentiment and emotion analyzer for text"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.preprocessor = TextPreprocessor()

        # Get model from config
        model_name = self.config.get("text_sentiment_analysis", {}).get(
            "model", "j-hartmann/emotion-english-distilroberta-base"
        )

        # Initialize transformer pipeline
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "text-classification", model=model_name, device=device, top_k=None
        )
        logger.info(f"Sentiment analyzer initialized with {model_name}")

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment and emotion from text

        Returns schema for llm_processing:
        {
            'emotion': str,
            'confidence': float,
            'emotion_scores': dict,
            'valence': float,
            'arousal': float,
            'sentiment': str
        }
        """
        # Preprocess text
        clean_text = self.preprocessor.clean(text)

        # Get predictions
        results = self.classifier(clean_text)

        # Process results
        emotion_scores = {}
        for result_list in results:
            for item in result_list:
                emotion_scores[item["label"].lower()] = item["score"]

        # Get primary emotion
        primary_emotion = max(emotion_scores, key=emotion_scores.get)

        # Calculate metrics
        valence = self._calculate_valence(emotion_scores)
        arousal = self._calculate_arousal(emotion_scores)

        return {
            "emotion": primary_emotion,
            "confidence": emotion_scores[primary_emotion],
            "emotion_scores": emotion_scores,
            "valence": valence,
            "arousal": arousal,
            "sentiment": self._determine_sentiment(valence),
            "text_length": len(text.split()),
        }

    def _calculate_valence(self, scores: Dict) -> float:
        """Calculate emotional valence (-1 to 1)"""
        positive = ["joy", "happy", "surprise", "love"]
        negative = ["sadness", "sad", "anger", "angry", "fear", "disgust"]

        pos_score = sum(scores.get(e, 0) for e in positive)
        neg_score = sum(scores.get(e, 0) for e in negative)

        if pos_score + neg_score == 0:
            return 0

        return (pos_score - neg_score) / (pos_score + neg_score)

    def _calculate_arousal(self, scores: Dict) -> float:
        """Calculate emotional arousal (0 to 1)"""
        high_arousal = ["anger", "angry", "fear", "surprise", "joy", "happy"]
        high_score = sum(scores.get(e, 0) for e in high_arousal)
        total = sum(scores.values())

        return high_score / total if total > 0 else 0.5

    def _determine_sentiment(self, valence: float) -> str:
        """Determine sentiment from valence"""
        if valence > 0.2:
            return "positive"
        elif valence < -0.2:
            return "negative"
        return "neutral"
