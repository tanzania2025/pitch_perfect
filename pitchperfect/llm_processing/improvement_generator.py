# pitchperfect/llm_processing/improvement_generator.py
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .emphasis_identifier import EmphasisIdentifier
from .issue_identifier import IssueIdentifier
from .prosody_calculator import ProsodyCalculator
from .ssml_generator import SSMLGenerator
from .text_improver import TextImprover

logger = logging.getLogger(__name__)


@dataclass
class ImprovementResult:
    """Complete improvement result"""

    improved_text: str
    original_text: str
    prosody_guide: Dict
    ssml_markup: str
    issues: Dict
    feedback: Dict
    emphasis_words: List


class ImprovementGenerator:
    """
    Main orchestrator for LLM processing module
    Takes sentiment and tonal analysis outputs and generates improvements
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Initialize all components
        self.issue_identifier = IssueIdentifier(config)
        self.emphasis_identifier = EmphasisIdentifier(config)
        self.prosody_calculator = ProsodyCalculator(config)
        self.text_improver = TextImprover(config)
        self.ssml_generator = SSMLGenerator(config)

        logger.info("ImprovementGenerator initialized")

    def generate_improvements(
        self,
        text: str,
        text_sentiment: Dict,
        acoustic_features: Dict,
        user_preferences: Optional[Dict] = None,
    ) -> ImprovementResult:
        """
        Generate improvements from analysis results

        Args:
            text: Original text from speech_to_text
            text_sentiment: From text_sentiment_analysis module
            acoustic_features: From tonal_analysis module
            user_preferences: Optional preferences

        Returns:
            ImprovementResult with all improvements
        """
        preferences = user_preferences or {"target_style": "professional"}

        # Step 1: Identify issues
        logger.info("Identifying issues...")
        issues = self.issue_identifier.identify(text, text_sentiment, acoustic_features)

        # Step 2: Improve text
        logger.info("Improving text...")
        improved_text = self.text_improver.improve(
            text,
            issues,
            text_sentiment,
            preferences.get("target_style", "professional"),
        )

        # Step 3: Identify emphasis
        logger.info("Identifying emphasis words...")
        emphasis_words = self.emphasis_identifier.identify(
            improved_text, text_sentiment
        )

        # Step 4: Calculate prosody
        logger.info("Calculating prosody adjustments...")
        prosody_adjustment = self.prosody_calculator.calculate(
            acoustic_features, issues.delivery_issues, text_sentiment
        )

        # Step 5: Create prosody guide
        prosody_guide = self._create_prosody_guide(
            improved_text, prosody_adjustment, emphasis_words
        )

        # Step 6: Generate SSML
        logger.info("Generating SSML...")
        ssml_markup = self.ssml_generator.generate(
            improved_text, prosody_guide, emphasis_words
        )

        # Step 7: Generate feedback
        feedback = self._generate_feedback(issues, prosody_adjustment)

        logger.info("Improvement generation complete")

        return ImprovementResult(
            improved_text=improved_text,
            original_text=text,
            prosody_guide=prosody_guide,
            ssml_markup=ssml_markup,
            issues=issues.to_dict(),
            feedback=feedback,
            emphasis_words=self.emphasis_identifier.to_simple_format(emphasis_words),
        )

    def _create_prosody_guide(self, text: str, adjustment, emphasis_words) -> Dict:
        """Create prosody guide for TTS"""
        return {
            "rate_multiplier": adjustment.rate_multiplier,
            "pitch_multiplier": adjustment.pitch_multiplier,
            "pitch_variation_multiplier": adjustment.pitch_variation_multiplier,
            "volume_multiplier": adjustment.volume_multiplier,
            "pause_locations": adjustment.pause_locations,
            "emphasis_count": len(emphasis_words),
            "target_wpm": adjustment.target_wpm,
        }

    def _generate_feedback(self, issues, adjustment) -> Dict:
        """Generate user feedback"""
        # Summary
        if issues.severity == "high":
            summary = "Your speech needs significant improvements in both content and delivery."
        elif issues.severity == "medium":
            summary = "Your speech is good but could benefit from some refinements."
        else:
            summary = "Your speech is well-structured with minor areas for enhancement."

        # Key improvements
        improvements = []
        if "filler_words" in issues.text_issues:
            improvements.append("Removed filler words for clearer communication")
        if "poor_structure" in issues.text_issues:
            improvements.append("Improved sentence structure and flow")
        if "monotone" in issues.delivery_issues:
            improvements.append("Added variety to combat monotone delivery")
        if "too_fast" in issues.delivery_issues:
            improvements.append("Adjusted pacing for better comprehension")

        # Tips
        tips = []
        for reason in adjustment.adjustment_reasons:
            tips.append(reason)

        return {
            "summary": summary,
            "key_improvements": improvements,
            "speaking_tips": tips,
            "severity": issues.severity,
            "issues_found": len(issues.text_issues) + len(issues.delivery_issues),
        }
