# pitchperfect/llm_processing/emphasis_identifier.py
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pitchperfect.utils.text_processing import TextProcessor

from .helper_functions import HelperFunctions


@dataclass
class EmphasisWord:
    """Data class for words to emphasize"""

    word: str
    position: int
    level: str  # 'light', 'medium', 'strong'
    reason: str


class EmphasisIdentifier:
    """Identifies words that should be emphasized in speech"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.text_processor = TextProcessor(config)
        self.helpers = HelperFunctions()

        # Load word categories from config
        emphasis_config = self.config.get("llm_processing", {}).get("emphasis", {})
        self.power_words = emphasis_config.get(
            "power_words",
            {
                "strong": ["must", "never", "always", "definitely", "absolutely"],
                "medium": ["should", "important", "significant", "key", "main"],
                "light": ["might", "could", "perhaps", "consider"],
            },
        )
        self.emotion_words = emphasis_config.get(
            "emotion_words",
            {
                "positive": ["amazing", "excellent", "wonderful", "fantastic"],
                "negative": ["terrible", "awful", "concerning", "problematic"],
                "urgent": ["immediately", "now", "urgent", "quickly"],
            },
        )

    def identify(
        self, text: str, sentiment: Optional[Dict] = None
    ) -> List[EmphasisWord]:
        """Identify words to emphasize"""
        words = self.text_processor.tokenize(text)
        emphasis_points = []

        for i, word in enumerate(words):
            word_clean = word.lower().strip(".,!?;:")

            # Check power words
            emphasis = self._check_power_words(word_clean, i, word)
            if emphasis:
                emphasis_points.append(emphasis)
                continue

            # Check emotion words
            if sentiment:
                emphasis = self._check_emotion_words(word_clean, i, word, sentiment)
                if emphasis:
                    emphasis_points.append(emphasis)
                    continue

            # Check numbers
            if re.match(r"\d+\.?\d*%?", word):
                emphasis_points.append(EmphasisWord(word, i, "medium", "number"))

        # Add transitions
        transitions = self.helpers.identify_transition_words(text)
        for pos, trans_word in transitions:
            emphasis_points.append(EmphasisWord(trans_word, pos, "light", "transition"))

        # Remove duplicates
        seen = set()
        unique = []
        for emp in emphasis_points:
            if emp.position not in seen:
                unique.append(emp)
                seen.add(emp.position)

        return sorted(unique, key=lambda x: x.position)

    def _check_power_words(
        self, word_clean: str, position: int, original: str
    ) -> Optional[EmphasisWord]:
        """Check if word is a power word"""
        for level, words in self.power_words.items():
            if word_clean in words:
                return EmphasisWord(original, position, level, "power_word")
        return None

    def _check_emotion_words(
        self, word_clean: str, position: int, original: str, sentiment: Dict
    ) -> Optional[EmphasisWord]:
        """Check if word is an emotion word"""
        emotion = sentiment.get("emotion", "neutral")

        if emotion in ["happy", "joy"]:
            if word_clean in self.emotion_words.get("positive", []):
                return EmphasisWord(original, position, "medium", "positive_emotion")
        elif emotion in ["sad", "angry", "fear"]:
            if word_clean in self.emotion_words.get("negative", []):
                return EmphasisWord(original, position, "medium", "negative_emotion")

        if word_clean in self.emotion_words.get("urgent", []):
            return EmphasisWord(original, position, "strong", "urgency")

        return None

    def to_simple_format(
        self, emphasis_words: List[EmphasisWord]
    ) -> List[Tuple[int, str, str]]:
        """Convert to simple format for compatibility"""
        return [(e.position, e.word, e.level) for e in emphasis_words]
