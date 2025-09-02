# pitchperfect/llm_processing/ssml_generator.py
import re
from typing import Dict, List, Optional
from xml.sax.saxutils import escape
from .emphasis_identifier import EmphasisWord


class SSMLGenerator:
    """Generates SSML markup for speech synthesis"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def generate(
        self, text: str, prosody_guide: Dict, emphasis_words: List[EmphasisWord] = None
    ) -> str:
        """Generate SSML markup"""
        # Escape text
        text = escape(text)

        # Apply prosody
        content = self._apply_prosody(text, prosody_guide)

        # Apply emphasis
        if emphasis_words:
            content = self._apply_emphasis(content, emphasis_words)

        # Apply pauses
        if prosody_guide.get("pause_locations"):
            content = self._apply_pauses(content, prosody_guide["pause_locations"])

        # Format numbers
        content = self._format_special(content)

        # Wrap in speak
        ssml = f"<speak>{content}</speak>"

        return ssml if self._validate(ssml) else f"<speak>{escape(text)}</speak>"

    def _apply_prosody(self, text: str, guide: Dict) -> str:
        """Apply prosody settings"""
        rate = self._format_rate(guide.get("rate_multiplier", 1.0))
        pitch = self._format_pitch(guide.get("pitch_multiplier", 1.0))
        volume = self._format_volume(guide.get("volume_multiplier", 1.0))

        if rate != "100%" or pitch != "+0%" or volume != "+0dB":
            return f'<prosody rate="{rate}" pitch="{pitch}" volume="{volume}">{text}</prosody>'

        return text

    def _apply_emphasis(self, content: str, emphasis_words: List[EmphasisWord]) -> str:
        """Apply emphasis tags"""
        words = content.split()

        for emp in sorted(emphasis_words, key=lambda x: x.position, reverse=True):
            if 0 <= emp.position < len(words):
                level = {"light": "reduced", "medium": "moderate", "strong": "strong"}[
                    emp.level
                ]
                words[emp.position] = (
                    f'<emphasis level="{level}">{words[emp.position]}</emphasis>'
                )

        return " ".join(words)

    def _apply_pauses(self, content: str, pauses: List[Dict]) -> str:
        """Insert pause breaks"""
        for pause in pauses:
            duration = pause.get("duration_ms", 200)
            pause_type = pause.get("type", "")

            if pause_type == "after_sentence":
                content = re.sub(r"\.", f'.<break time="{duration}ms"/>', content)
            elif pause_type == "after_comma":
                content = content.replace(",", f',<break time="{duration}ms"/>')

        return content

    def _format_special(self, content: str) -> str:
        """Format numbers and special content"""
        # Percentages
        content = re.sub(
            r"(\d+)%", r'<say-as interpret-as="percentage">\1%</say-as>', content
        )

        # Large numbers
        def format_number(match):
            num = match.group(0)
            if len(num) > 4:
                return f'<say-as interpret-as="number">{num}</say-as>'
            return num

        content = re.sub(r"\b\d{5,}\b", format_number, content)

        return content

    def _format_rate(self, multiplier: float) -> str:
        return f"{int(multiplier * 100)}%"

    def _format_pitch(self, multiplier: float) -> str:
        if multiplier == 1.0:
            return "+0%"
        percent = int((multiplier - 1.0) * 100)
        return f"+{percent}%" if percent > 0 else f"{percent}%"

    def _format_volume(self, multiplier: float) -> str:
        if multiplier == 1.0:
            return "+0dB"
        import math

        db = 20 * math.log10(multiplier)
        return f"+{db:.1f}dB" if db > 0 else f"{db:.1f}dB"

    def _validate(self, ssml: str) -> bool:
        """Basic SSML validation"""
        try:
            import xml.etree.ElementTree as ET

            ET.fromstring(ssml)
            return True
        except:
            return False
