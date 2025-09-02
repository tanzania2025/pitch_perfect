# pitchperfect/llm_processing/issue_identifier.py
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pitchperfect.utils.text_processing import TextProcessor
from .helper_functions import HelperFunctions


@dataclass
class Issues:
    """Data class for identified issues"""

    text_issues: List[str] = field(default_factory=list)
    delivery_issues: List[str] = field(default_factory=list)
    severity: str = "low"
    details: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "text_issues": self.text_issues,
            "delivery_issues": self.delivery_issues,
            "severity": self.severity,
            "details": self.details,
        }


class IssueIdentifier:
    """Identifies issues in speech text and delivery"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.text_processor = TextProcessor(config)
        self.helpers = HelperFunctions()

        # Load thresholds
        self.thresholds = self.config.get("llm_processing", {}).get("thresholds", {})

    def identify(self, text: str, sentiment: Dict, acoustic_features: Dict) -> Issues:
        """
        Identify all issues in speech

        Args:
            text: Original text from speech_to_text
            sentiment: Results from text_sentiment_analysis
            acoustic_features: Results from tonal_analysis
        """
        issues = Issues()

        # Text issues
        issues.text_issues = self._identify_text_issues(text, sentiment)

        # Delivery issues from acoustic analysis
        if "acoustic_problems" in acoustic_features:
            issues.delivery_issues = acoustic_features["acoustic_problems"]

        # Calculate severity
        issues.severity = self._calculate_severity(issues)

        # Add details
        issues.details = self._get_issue_details(text, sentiment, acoustic_features)

        return issues

    def _identify_text_issues(self, text: str, sentiment: Dict) -> List[str]:
        """Identify text-based issues"""
        issues = []

        # Check for filler words
        fillers = self.text_processor.detect_filler_words(text)
        if fillers:
            issues.append("filler_words")

        # Check structure
        if self.helpers.lacks_structure(text):
            issues.append("poor_structure")

        # Check sentence length
        avg_length = self.text_processor.calculate_average_sentence_length(text)
        if avg_length > 25:
            issues.append("sentences_too_long")
        elif avg_length < 5:
            issues.append("sentences_too_short")

        # Check emotion
        if (
            sentiment.get("emotion") == "neutral"
            and sentiment.get("confidence", 0) > 0.8
        ):
            issues.append("lacks_emotion")

        # Check vocabulary repetition
        words = text.split()
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                issues.append("repetitive_vocabulary")

        return issues

    def _calculate_severity(self, issues: Issues) -> str:
        """Calculate overall severity"""
        total = len(issues.text_issues) + len(issues.delivery_issues)

        critical_issues = ["too_quiet", "too_fast", "monotone", "poor_structure"]
        has_critical = any(
            issue in critical_issues
            for issue in issues.text_issues + issues.delivery_issues
        )

        if total >= 5 or has_critical:
            return "high"
        elif total >= 3:
            return "medium"
        return "low"

    def _get_issue_details(self, text: str, sentiment: Dict, acoustic: Dict) -> Dict:
        """Get detailed information about issues"""
        details = {}

        # Filler words
        fillers = self.text_processor.detect_filler_words(text)
        if fillers:
            details["filler_words"] = fillers

        # Speaking rate
        prosodic = acoustic.get("prosodic_features", {})
        tempo = prosodic.get("tempo", {})
        if tempo:
            details["speaking_rate_wpm"] = tempo.get("speaking_rate_wpm", 0)

        # Emotion
        details["current_emotion"] = sentiment.get("emotion", "unknown")
        details["emotion_confidence"] = sentiment.get("confidence", 0)

        # Voice quality
        quality = acoustic.get("voice_quality", {})
        details["monotone_score"] = quality.get("monotone_score", 0)
        details["confidence_score"] = quality.get("confidence_score", 0)

        return details
