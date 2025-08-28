def create_prosody_guide(self, improved_text, acoustic_features, issues, sentiment):
    """Generate detailed prosody instructions"""

    prosody_guide = {
        'rate': self.calculate_target_rate(acoustic_features, issues),
        'pitch': self.calculate_pitch_adjustments(acoustic_features, sentiment),
        'emphasis_words': self.identify_emphasis_words(improved_text, sentiment),
        'pause_locations': self.identify_pause_points(improved_text),
        'volume': self.calculate_volume_adjustment(acoustic_features),
        'emotion_target': self.determine_target_emotion(sentiment, issues),
        'ssml_markup': None  # Will be generated
    }

    # Generate SSML for speech synthesis
    prosody_guide['ssml_markup'] = self.generate_ssml(improved_text, prosody_guide)

    return prosody_guide

def calculate_target_rate(self, acoustic_features, issues):
    """Calculate optimal speaking rate"""

    current_rate = acoustic_features['speaking_rate']

    if 'too_fast' in issues['delivery_issues']:
        target_rate = current_rate * 0.85  # Slow down by 15%
    elif 'too_slow' in issues['delivery_issues']:
        target_rate = current_rate * 1.15  # Speed up by 15%
    else:
        target_rate = current_rate

    # Ensure within reasonable bounds (2.5 - 4.0 words/sec)
    target_rate = max(2.5, min(4.0, target_rate))

    return {
        'multiplier': target_rate / current_rate,
        'target_wpm': int(target_rate * 60),
        'adjustment': 'slower' if target_rate < current_rate else 'faster' if target_rate > current_rate else 'maintain'
    }

def calculate_pitch_adjustments(self, acoustic_features, sentiment):
    """Calculate pitch variation needs"""

    adjustments = {
        'base_pitch': acoustic_features['pitch_mean'],
        'variation_needed': 'none',
        'adjustment_factor': 1.0
    }

    if acoustic_features['monotone_score'] > 0.6:
        adjustments['variation_needed'] = 'increase'
        adjustments['adjustment_factor'] = 1.3  # 30% more variation

    # Emotion-based adjustments
    if sentiment['emotion'] in ['happy', 'excited']:
        adjustments['base_pitch'] *= 1.1  # Slightly higher
    elif sentiment['emotion'] in ['sad', 'angry']:
        adjustments['base_pitch'] *= 0.95  # Slightly lower

    return adjustments

def identify_emphasis_words(self, text, sentiment):
    """Identify words that should be emphasized"""

    emphasis_words = []

    # Key words based on sentiment
    emotion_keywords = {
        'positive': ['great', 'excellent', 'amazing', 'wonderful', 'fantastic'],
        'negative': ['concerned', 'worried', 'important', 'critical', 'urgent'],
        'neutral': ['key', 'main', 'focus', 'essential', 'primary']
    }

    # Power words that should always be emphasized
    power_words = ['must', 'never', 'always', 'definitely', 'absolutely',
                   'crucial', 'vital', 'essential', 'significant']

    # Numbers and statistics
    import re
    numbers = re.findall(r'\b\d+%?\b', text)

    # Find emphasis words in text
    text_lower = text.lower()
    words = text.split()

    for i, word in enumerate(words):
        word_clean = word.lower().strip('.,!?;:')

        # Check emotion keywords
        if sentiment['emotion'] in ['happy', 'joy', 'excited']:
            if word_clean in emotion_keywords['positive']:
                emphasis_words.append((i, word))
        elif sentiment['emotion'] in ['sad', 'angry', 'fear']:
            if word_clean in emotion_keywords['negative']:
                emphasis_words.append((i, word))

        # Check power words
        if word_clean in power_words:
            emphasis_words.append((i, word))

        # Check numbers
        if any(num in word for num in numbers):
            emphasis_words.append((i, word))

    return emphasis_words

def identify_pause_points(self, text):
    """Identify where to insert pauses"""

    pause_points = []
    sentences = text.split('.')

    for i, sentence in enumerate(sentences):
        if len(sentence.split()) > 15:  # Long sentence
            # Add pause after commas
            if ',' in sentence:
                pause_points.append({'type': 'short', 'after': ','})

        # Add pause after important transitions
        transitions = ['however', 'therefore', 'furthermore', 'additionally']
        for transition in transitions:
            if transition in sentence.lower():
                pause_points.append({'type': 'medium', 'after': transition})

    # Add pause before key points
    if 'important' in text.lower() or 'key' in text.lower():
        pause_points.append({'type': 'medium', 'before': 'important|key'})

    return pause_points
