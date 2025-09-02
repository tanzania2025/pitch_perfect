# pitchperfect/llm_processing/text_improver.py
import re
from typing import Dict, List, Optional

from openai import OpenAI

from pitchperfect.utils.text_processing import TextProcessor

from .helper_functions import HelperFunctions
from .issue_identifier import Issues


class TextImprover:
    """Improves text using OpenAI API"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.text_processor = TextProcessor(config)
        self.helpers = HelperFunctions()

        # OpenAI setup
        openai_config = self.config.get("llm_processing", {}).get("openai", {})
        api_key = openai_config.get("api_key")

        if not api_key:
            raise ValueError("OpenAI API key required in config")

        self.client = OpenAI(api_key=api_key)
        self.model = openai_config.get("model", "gpt-4")
        self.temperature = openai_config.get("temperature", 0.7)

    def improve(
        self,
        text: str,
        issues: Issues,
        sentiment: Dict,
        target_style: str = "professional",
    ) -> str:
        """
        Improve text based on identified issues

        Returns improved text string
        """
        # Build prompt
        prompt = self._build_prompt(text, issues, sentiment, target_style)

        # Get improvement
        improved = self._call_openai(prompt)

        # Post-process
        improved = self.helpers.post_process_text(improved)

        # Verify improvement maintains meaning
        similarity = self.text_processor.calculate_similarity(text, improved)
        if similarity < 0.3:
            # Too different, try again with stricter prompt
            prompt = self._build_strict_prompt(text, issues)
            improved = self._call_openai(prompt)
            improved = self.helpers.post_process_text(improved)

        return improved

    def _build_prompt(
        self, text: str, issues: Issues, sentiment: Dict, style: str
    ) -> str:
        """Build improvement prompt"""
        prompt = f"""Improve this speech for {style} delivery.

Original: "{text}"

Current emotion: {sentiment.get('emotion', 'unknown')}

Issues to fix:"""

        # Add text issues
        for issue in issues.text_issues:
            prompt += f"\n- {self._describe_issue(issue)}"

        # Add delivery support
        if "monotone" in issues.delivery_issues:
            prompt += "\n- Add variety to combat monotone delivery"
        if "low_confidence" in issues.delivery_issues:
            prompt += "\n- Use more confident language"

        prompt += """

Requirements:
1. Keep the same core message
2. Sound natural when spoken
3. Remove all filler words
4. Improve flow and structure
5. Keep length within ±20% of original

Return ONLY the improved text."""

        return prompt

    def _build_strict_prompt(self, text: str, issues: Issues) -> str:
        """Build stricter prompt when first attempt diverges too much"""
        return f"""Make minimal improvements to this text:

"{text}"

ONLY fix these specific issues:
- Remove filler words (um, uh, like, you know)
- Fix any grammar errors
- Keep everything else as similar as possible

Return the improved text."""

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a speech improvement expert.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.config.get("llm_processing", {})
                .get("openai", {})
                .get("max_tokens", 500),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""

    def _describe_issue(self, issue: str) -> str:
        """Get human-readable description"""
        descriptions = {
            "filler_words": "Remove filler words",
            "poor_structure": "Improve sentence structure",
            "lacks_emotion": "Add emotional engagement",
            "sentences_too_long": "Shorten long sentences",
            "sentences_too_short": "Combine short sentences",
            "repetitive_vocabulary": "Vary vocabulary",
        }
        return descriptions.get(issue, issue)
