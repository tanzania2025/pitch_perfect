# pitchperfect/utils/text_processing.py
import re
from typing import List, Dict, Tuple, Optional

class TextProcessor:
    """Utilities for text processing used across modules"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'I mean', 'basically', 'actually']

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Fix basic punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        # Ensure proper sentence capitalization
        sentences = text.split('. ')
        sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
        text = '. '.join(sentences)
        return text

    def detect_filler_words(self, text: str) -> List[Dict]:
        """Detect filler words in text"""
        text_lower = text.lower()
        found_fillers = []
        for filler in self.filler_words:
            if filler in text_lower:
                count = text_lower.count(filler)
                found_fillers.append({'word': filler, 'count': count})
        return found_fillers

    def tokenize(self, text: str) -> List[str]:
        """Simple word tokenization"""
        return text.split()

    def calculate_average_sentence_length(self, text: str) -> float:
        """Calculate average sentence length"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return 0

        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (0-1)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def extract_numbers(self, text: str) -> List[str]:
        """Extract numbers and percentages from text"""
        pattern = r'\b\d+\.?\d*%?\b'
        return re.findall(pattern, text)

    def identify_transition_words(self, text: str) -> List[Tuple[int, str]]:
        """Identify transition words in text"""
        transitions = [
            'however', 'therefore', 'furthermore', 'additionally',
            'moreover', 'nevertheless', 'consequently', 'meanwhile'
        ]

        words = text.split()
        found = []

        for i, word in enumerate(words):
            if word.lower().strip('.,;:') in transitions:
                found.append((i, word))

        return found
