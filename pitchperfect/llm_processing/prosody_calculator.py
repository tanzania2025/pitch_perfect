# pitchperfect/llm_processing/prosody_calculator.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ProsodyAdjustment:
    """Data class for prosody adjustments"""

    rate_multiplier: float = 1.0
    pitch_multiplier: float = 1.0
    pitch_variation_multiplier: float = 1.0
    volume_multiplier: float = 1.0
    pause_locations: List[Dict] = field(default_factory=list)
    target_wpm: Optional[float] = None
    adjustment_reasons: List[str] = field(default_factory=list)


class ProsodyCalculator:
    """Calculates prosody adjustments based on acoustic features"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Load thresholds
        self.thresholds = self.config.get("llm_processing", {}).get("thresholds", {})

        # Target ranges
        self.target_ranges = {
            "speaking_rate_wpm": (120, 180),
            "pitch_variation_hz": (30, 80),
            "pause_ratio": (0.15, 0.25),
            "energy_db": (-20, -10),
        }

    def calculate(
        self,
        acoustic_features: Dict,
        issues: List[str],
        sentiment: Optional[Dict] = None,
    ) -> ProsodyAdjustment:
        """Calculate prosody adjustments needed"""
        adjustment = ProsodyAdjustment()

        # Rate adjustment
        rate_data = self._calculate_rate(acoustic_features, issues)
        adjustment.rate_multiplier = rate_data["multiplier"]
        adjustment.target_wpm = rate_data["target_wpm"]

        # Pitch adjustment
        pitch_data = self._calculate_pitch(acoustic_features, issues, sentiment)
        adjustment.pitch_multiplier = pitch_data["base_multiplier"]
        adjustment.pitch_variation_multiplier = pitch_data["variation_multiplier"]

        # Volume adjustment
        volume_data = self._calculate_volume(acoustic_features, issues)
        adjustment.volume_multiplier = volume_data["multiplier"]

        # Pause adjustments
        pause_data = self._calculate_pauses(acoustic_features, issues)
        adjustment.pause_locations = pause_data["locations"]

        # Add reasons
        if rate_data["adjustment_needed"]:
            adjustment.adjustment_reasons.append(
                f"Adjust rate to {rate_data['target_wpm']:.0f} wpm"
            )
        if pitch_data["needs_variation"]:
            adjustment.adjustment_reasons.append("Increase pitch variation")
        if volume_data["adjustment_needed"]:
            adjustment.adjustment_reasons.append(
                f"Adjust volume by {volume_data['db_change']:.1f}dB"
            )

        return adjustment

    def _calculate_rate(self, features: Dict, issues: List[str]) -> Dict:
        """Calculate rate adjustment"""
        prosodic = features.get("prosodic_features", {})
        current = prosodic.get("tempo", {}).get("speaking_rate_wpm", 160)
        target = current

        if "too_fast" in issues:
            target = current * 0.85
        elif "too_slow" in issues:
            target = current * 1.15

        target = np.clip(target, *self.target_ranges["speaking_rate_wpm"])

        return {
            "current_wpm": current,
            "target_wpm": target,
            "multiplier": target / current if current > 0 else 1.0,
            "adjustment_needed": abs(target - current) > 10,
        }

    def _calculate_pitch(
        self, features: Dict, issues: List[str], sentiment: Dict
    ) -> Dict:
        """Calculate pitch adjustment"""
        prosodic = features.get("prosodic_features", {})
        pitch_std = prosodic.get("pitch", {}).get("std_hz", 25)

        result = {
            "base_multiplier": 1.0,
            "variation_multiplier": 1.0,
            "needs_variation": False,
        }

        # Check monotone
        if "monotone" in issues or pitch_std < 30:
            result["variation_multiplier"] = 1.5
            result["needs_variation"] = True

        # Emotion-based adjustment
        if sentiment:
            emotion = sentiment.get("emotion", "neutral")
            if emotion in ["happy", "excited"]:
                result["base_multiplier"] = 1.1
            elif emotion in ["sad", "angry"]:
                result["base_multiplier"] = 0.95

        return result

    def _calculate_volume(self, features: Dict, issues: List[str]) -> Dict:
        """Calculate volume adjustment"""
        prosodic = features.get("prosodic_features", {})
        current_db = prosodic.get("energy", {}).get("mean_db", -15)

        result = {"multiplier": 1.0, "db_change": 0, "adjustment_needed": False}

        if "too_quiet" in issues:
            target_db = -15
            result["db_change"] = target_db - current_db
            result["multiplier"] = 10 ** (result["db_change"] / 20)
            result["adjustment_needed"] = True
        elif "too_loud" in issues:
            target_db = -15
            result["db_change"] = target_db - current_db
            result["multiplier"] = 10 ** (result["db_change"] / 20)
            result["adjustment_needed"] = True

        return result

    def _calculate_pauses(self, features: Dict, issues: List[str]) -> Dict:
        """Calculate pause adjustments"""
        prosodic = features.get("prosodic_features", {})
        pause_ratio = prosodic.get("pauses", {}).get("pause_ratio", 0.2)

        locations = []

        if "no_pauses" in issues or pause_ratio < 0.1:
            locations = [
                {"type": "after_sentence", "duration_ms": 400},
                {"type": "after_comma", "duration_ms": 200},
                {"type": "before_emphasis", "duration_ms": 300},
            ]

        return {"locations": locations}
