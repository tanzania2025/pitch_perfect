import openai
from openai import OpenAI
from typing import Dict, Tuple, List, Any
import re

class EmphasisIdentifier:
    """Identifies words that should be emphasized in speech"""

    def __init__(self):
        self.power_words = {
            'must', 'never', 'always', 'definitely', 'absolutely',
            'crucial', 'vital', 'essential', 'significant', 'important'
        }

    def identify_emphasis_words(self, text: str, sentiment: Dict[str, Any]) -> List[Tuple[int, str]]:
        """Identify words that should be emphasized"""
        emphasis_words = []
        words = text.split()

        for i, word in enumerate(words):
            word_clean = word.lower().strip('.,!?;:')

            # Check power words
            if word_clean in self.power_words:
                emphasis_words.append((i, word))

            # Check numbers and statistics
            if re.match(r'\b\d+%?\b', word):
                emphasis_words.append((i, word))

        return emphasis_words

class ProsodyCalculator:
    """Calculates prosody adjustments for speech synthesis"""

    def calculate_pitch_adjustments(self, acoustic_features: Dict[str, Any], sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate pitch variation needs"""
        adjustments = {
            'base_pitch': acoustic_features.get('pitch_mean', 150),
            'variation_needed': 'none',
            'adjustment_factor': 1.0
        }

        if acoustic_features.get('monotone_score', 0) > 0.6:
            adjustments['variation_needed'] = 'increase'
            adjustments['adjustment_factor'] = 1.3

        return adjustments

    def calculate_speaking_rate(self, acoustic_features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal speaking rate"""
        current_rate = acoustic_features.get('speaking_rate', 3.0)

        if current_rate > 4.0:
            target_rate = current_rate * 0.85
        elif current_rate < 2.5:
            target_rate = current_rate * 1.15
        else:
            target_rate = current_rate

        return {
            'current_rate': current_rate,
            'target_rate': target_rate,
            'adjustment': 'slower' if target_rate < current_rate else 'faster' if target_rate > current_rate else 'maintain'
        }

class ImprovementGenerator:
    def __init__(self, api_key: str = None):
        """Initialize the improvement generator"""
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            # Use environment variable or default
            import os
            api_key = os.getenv('OPENAI_API_KEY', 'your-key')
            self.openai_client = OpenAI(api_key=api_key)

        self.emphasis_identifier = EmphasisIdentifier()
        self.prosody_calculator = ProsodyCalculator()

    def generate_improvements(self, text: str, text_sentiment: Dict[str, Any], acoustic_features: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text improvements and prosody guidance

        Args:
            text: Original transcribed text
            text_sentiment: {'emotion': 'neutral', 'score': 0.85}
            acoustic_features: {
                'pitch_mean': 150, 'pitch_std': 25,
                'speaking_rate': 3.2, 'energy': 0.05,
                'pause_ratio': 0.15, 'monotone_score': 0.7
            }

        Returns:
            improved_text: Enhanced version of the text
            prosody_guide: Instructions for speech synthesis
        """

        # Step 1: Analyze current issues
        issues = self.identify_issues(text, text_sentiment, acoustic_features)

        # Step 2: Generate improved text
        improved_text = self.improve_text(text, issues, text_sentiment)

        # Step 3: Create prosody guidance
        prosody_guide = self.create_prosody_guide(
            improved_text,
            acoustic_features,
            issues,
            text_sentiment
        )

        return improved_text, prosody_guide

    def identify_issues(self, text: str, sentiment: Dict[str, Any], acoustic_features: Dict[str, Any]) -> Dict[str, Any]:
        """Identify speaking and text issues"""
        issues = {
            'text_issues': [],
            'delivery_issues': [],
            'severity': 'low'  # low, medium, high
        }

        # Text-based issues
        if self.has_filler_words(text):
            issues['text_issues'].append('filler_words')

        if self.lacks_structure(text):
            issues['text_issues'].append('poor_structure')

        if sentiment.get('emotion') == 'neutral' and sentiment.get('score', 0) > 0.8:
            issues['text_issues'].append('lacks_emotion')

        # Acoustic issues
        if acoustic_features.get('monotone_score', 0) > 0.6:
            issues['delivery_issues'].append('monotone')

        speaking_rate = acoustic_features.get('speaking_rate', 3.0)
        if speaking_rate > 4.5:
            issues['delivery_issues'].append('too_fast')
        elif speaking_rate < 2.5:
            issues['delivery_issues'].append('too_slow')

        if acoustic_features.get('energy', 0.05) < 0.02:
            issues['delivery_issues'].append('low_energy')

        if acoustic_features.get('pause_ratio', 0.15) < 0.1:
            issues['delivery_issues'].append('needs_pauses')

        # Set severity
        total_issues = len(issues['text_issues']) + len(issues['delivery_issues'])
        if total_issues >= 4:
            issues['severity'] = 'high'
        elif total_issues >= 2:
            issues['severity'] = 'medium'

        return issues

    def has_filler_words(self, text: str) -> bool:
        """Check for filler words"""
        fillers = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'I mean', 'basically', 'actually']
        text_lower = text.lower()
        return any(filler in text_lower for filler in fillers)

    def lacks_structure(self, text: str) -> bool:
        """Check if text lacks proper structure"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return True

        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        # Too many short sentences or one very long sentence
        return avg_length < 5 or avg_length > 30

    def improve_text(self, text: str, issues: Dict[str, Any], sentiment: Dict[str, Any]) -> str:
        """Generate improved version of the text using OpenAI"""
        try:
            # Create improvement prompt based on identified issues
            prompt = self.create_improvement_prompt(text, issues, sentiment)

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a speech improvement expert. Improve the given text based on the identified issues while maintaining the original meaning."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            improved_text = response.choices[0].message.content.strip()
            return self.post_process_text(improved_text)

        except Exception as e:
            print(f"Error improving text: {e}")
            return text  # Return original text if improvement fails

    def create_improvement_prompt(self, text: str, issues: Dict[str, Any], sentiment: Dict[str, Any]) -> str:
        """Create a prompt for text improvement"""
        prompt_parts = [
            f"Original text: {text}",
            f"Identified issues: {', '.join(issues['text_issues'] + issues['delivery_issues'])}",
            f"Current sentiment: {sentiment.get('emotion', 'neutral')} (confidence: {sentiment.get('score', 0)})"
        ]

        if 'filler_words' in issues['text_issues']:
            prompt_parts.append("Remove filler words and unnecessary phrases")
        if 'poor_structure' in issues['text_issues']:
            prompt_parts.append("Improve sentence structure and flow")
        if 'lacks_emotion' in issues['text_issues']:
            prompt_parts.append("Add emotional engagement while maintaining clarity")

        prompt_parts.append("Provide only the improved text without explanations.")

        return "\n".join(prompt_parts)

    def create_prosody_guide(self, improved_text: str, acoustic_features: Dict[str, Any], issues: Dict[str, Any], sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed prosody instructions"""
        prosody_guide = {
            'rate': self.prosody_calculator.calculate_speaking_rate(acoustic_features),
            'pitch': self.prosody_calculator.calculate_pitch_adjustments(acoustic_features, sentiment),
            'emphasis_words': self.emphasis_identifier.identify_emphasis_words(improved_text, sentiment),
            'pause_locations': self.identify_pause_points(improved_text),
            'volume': self.calculate_volume_adjustment(acoustic_features),
            'emotion_target': self.determine_target_emotion(sentiment, issues),
            'ssml_markup': None  # Will be generated if needed
        }

        return prosody_guide

    def identify_pause_points(self, text: str) -> List[int]:
        """Identify natural pause points in the text"""
        pause_points = []
        sentences = text.split('.')

        current_pos = 0
        for sentence in sentences:
            if sentence.strip():
                current_pos += len(sentence) + 1  # +1 for the period
                pause_points.append(current_pos)

        return pause_points

    def calculate_volume_adjustment(self, acoustic_features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate volume adjustments needed"""
        energy = acoustic_features.get('energy', 0.05)

        if energy < 0.02:
            return {'adjustment': 'increase', 'factor': 1.5}
        elif energy > 0.1:
            return {'adjustment': 'decrease', 'factor': 0.8}
        else:
            return {'adjustment': 'maintain', 'factor': 1.0}

    def determine_target_emotion(self, sentiment: Dict[str, Any], issues: Dict[str, Any]) -> str:
        """Determine target emotional tone"""
        if 'lacks_emotion' in issues.get('text_issues', []):
            # Suggest more engaging emotion
            if sentiment.get('emotion') == 'neutral':
                return 'confident_friendly'

        # Enhance existing emotion
        emotion_map = {
            'happy': 'enthusiastic',
            'sad': 'empathetic',
            'angry': 'assertive',
            'neutral': 'confident',
            'fear': 'calm_reassuring'
        }

        return emotion_map.get(sentiment.get('emotion', 'neutral'), 'confident')

    def post_process_text(self, text: str) -> str:
        """Clean up improved text"""
        # Remove any GPT artifacts
        text = text.strip()

        # Ensure proper capitalization
        sentences = text.split('. ')
        sentences = [s[0].upper() + s[1:] if s and len(s) > 0 else s for s in sentences]
        text = '. '.join(sentences)

        # Remove double spaces
        text = ' '.join(text.split())

        return text
