# pitchperfect/text_sentiment_analysis/preprocessing.py
import re
from typing import List, Optional


class TextPreprocessor:
    """Text preprocessing for sentiment analysis"""

    def __init__(self):
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
        }

    def clean(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)

        # Remove extra whitespace
        text = " ".join(text.split())

        # Keep punctuation for emotion detection
        # But remove special characters
        text = re.sub(r"[^\w\s.,!?;:\'-]", "", text)

        return text.strip()
