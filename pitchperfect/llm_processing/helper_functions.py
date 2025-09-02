# pitchperfect/llm_processing/helper_functions.py
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class HelperFunctions:
    """Helper functions for LLM processing"""

    @staticmethod
    def has_filler_words(text: str, filler_list: List[str] = None) -> bool:
        """Check for filler words"""
        if filler_list is None:
            filler_list = ["um", "uh", "er", "ah", "like", "you know", "I mean"]

        text_lower = text.lower()
        return any(filler in text_lower for filler in filler_list)

    @staticmethod
    def lacks_structure(text: str) -> bool:
        """Check if text lacks proper structure"""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return True

        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        return avg_length < 5 or avg_length > 30

    @staticmethod
    def post_process_text(text: str) -> str:
        """Clean up improved text"""
        # Remove any GPT artifacts
        text = text.strip()

        # Remove text in brackets that might be instructions
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?Note:.*?\)", "", text)

        # Ensure proper capitalization
        sentences = text.split(". ")
        sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
        text = ". ".join(sentences)

        # Remove double spaces
        text = " ".join(text.split())

        # Ensure ends with punctuation
        if text and text[-1] not in ".!?":
            text += "."

        return text

    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        """Extract numbers and percentages from text"""
        pattern = r"\b\d+\.?\d*%?\b"
        return re.findall(pattern, text)

    @staticmethod
    def identify_transition_words(text: str) -> List[Tuple[int, str]]:
        """Identify transition words in text"""
        transitions = [
            "however",
            "therefore",
            "furthermore",
            "additionally",
            "moreover",
            "nevertheless",
            "consequently",
            "meanwhile",
        ]

        words = text.split()
        found = []

        for i, word in enumerate(words):
            if word.lower().strip(".,;:") in transitions:
                found.append((i, word))

        return found
